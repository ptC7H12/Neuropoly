"""
Polymarket API access, shared by live_bid.py, collect_trades.py and
paper_trades.py.

Which endpoint for what
-----------------------
  gamma-api.polymarket.com/markets    market metadata, public
  data-api.polymarket.com/trades      public trade history  ← use this
  clob.polymarket.com/trades          NOT usable: it is the authenticated
                                      L2 endpoint and returns *your own*
                                      trades.  Unauthenticated calls answer
                                      401 {"error":"Unauthorized/Invalid api key"}

Trades are fetched per MARKET (conditionId), not per token.  The training
pipeline aggregates both tokens of a market into one bucket, so yes_ratio
means "share of fills that were on the YES side".  Fetching a single token
would make that constant and the feature meaningless at inference time.

Gamma quirk: `outcomes`, `outcomePrices` and `clobTokenIds` come back as
JSON-encoded *strings*, not arrays.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import polars as pl

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Max rows the data-api returns per request
_PAGE_LIMIT = 500

# Max market IDs Gamma returns per request.  Measured: asking for more just
# returns 100, and omitting `limit` altogether silently caps the response at
# 20 — which looks like missing markets rather than a paging limit.
_GAMMA_ID_BATCH = 100


class PolymarketAPIError(RuntimeError):
    """Raised when the API cannot be reached or answers with an error."""


def _get(url: str, params: dict | list | None = None, timeout: int = 15,
         retries: int = 3) -> Any:
    """
    GET with a small exponential backoff. Requires `requests`.

    `params` may be a dict, or a list of (key, value) pairs when the same
    key has to repeat — the Gamma batch lookup needs `?id=1&id=2&…`.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment issue
        raise PolymarketAPIError(
            "'requests' is not installed. Run: pip install requests"
        ) from exc

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # network hiccup, 429, 5xx …
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise PolymarketAPIError(f"GET {url} failed after {retries} tries: {last_exc}")


def _json_list(value: Any) -> list:
    """Gamma returns list fields as JSON strings — decode them defensively."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class MarketInfo:
    """Normalised market metadata from the Gamma API."""

    condition_id: str
    question: str = ""
    # clobTokenIds[i] belongs to outcomes[i]; index 0 is the YES/first outcome
    # and corresponds to `token1` in markets.csv.
    token_yes: str = ""
    token_no: str = ""
    outcomes: list[str] = field(default_factory=list)
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    end_date: Optional[datetime] = None       # tz-naive UTC
    raw: dict = field(default_factory=dict)

    def side_of(self, asset_id: str) -> Optional[int]:
        """1 if `asset_id` is the YES token, 0 if NO, None if unknown."""
        if asset_id and asset_id == self.token_yes:
            return 1
        if asset_id and asset_id == self.token_no:
            return 0
        return None


def _parse_end_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).astimezone(timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def fetch_market(token_id: str) -> Optional[MarketInfo]:
    """
    Look up a market by one of its CLOB token IDs.

    Returns None when the market is unknown; raises PolymarketAPIError when
    the API itself is unreachable.
    """
    data = _get(f"{GAMMA_API}/markets", params={"clob_token_ids": token_id})

    if isinstance(data, dict):
        data = data.get("markets") or data.get("data") or []
    if not isinstance(data, list) or not data:
        return None

    m = data[0]
    tokens = _json_list(m.get("clobTokenIds"))
    prices = _json_list(m.get("outcomePrices"))

    return MarketInfo(
        condition_id=str(m.get("conditionId") or ""),
        question=str(m.get("question") or m.get("title") or ""),
        token_yes=str(tokens[0]) if len(tokens) > 0 else "",
        token_no=str(tokens[1]) if len(tokens) > 1 else "",
        outcomes=[str(o) for o in _json_list(m.get("outcomes"))],
        yes_price=_safe_float(prices[0]) if len(prices) > 0 else None,
        no_price=_safe_float(prices[1]) if len(prices) > 1 else None,
        volume=_safe_float(m.get("volumeNum") or m.get("volume")),
        liquidity=_safe_float(m.get("liquidityNum") or m.get("liquidity")),
        end_date=_parse_end_date(
            m.get("endDate") or m.get("end_date_iso") or m.get("close_time")
        ),
        raw=m,
    )


def fetch_markets_by_id(
    market_ids: list[int] | list[str],
    include_tags: bool = True,
    batch_size: int = _GAMMA_ID_BATCH,
) -> dict[int, dict]:
    """
    Look up markets by their Gamma id, in batches.

    Returns {market_id: market dict}.  IDs the API no longer serves are
    simply absent — the caller decides what that means.

    Three quirks are handled here, all measured against the live API:

    * `limit` must be sent explicitly.  Without it the response is capped at
      20 rows regardless of how many ids were asked for, which looks like
      missing markets rather than a paging limit.
    * The `closed` filter partitions the result rather than widening it.
      Asking 10 active + 10 closed ids returns 10 active without `closed`,
      and 10 closed with `closed=true` — never all 20.  So each batch is
      requested twice and merged; omitting the second pass silently drops
      every historical market.
    * Closed markets carry no USABLE tags.  They do come back with a tag
      list, but it holds only the placeholder "All" — 99 of 100 closed
      markets in a sample, and 0 of 100 with a tag that identifies a
      subject.  Active markets are tagged properly 100 % of the time.
      So this lookup grades a classifier on the still-active subset; it
      cannot label a historical dataset.
    """
    out: dict[int, dict] = {}
    ids = [str(i) for i in market_ids]

    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        base: list[tuple[str, str]] = [("id", i) for i in chunk]
        base.append(("limit", str(len(chunk))))
        if include_tags:
            base.append(("include_tag", "true"))

        # Pass 1 picks up open markets, pass 2 the closed ones
        for extra in ([], [("closed", "true")]):
            page = _get(f"{GAMMA_API}/markets", params=base + extra)
            if isinstance(page, dict):
                page = page.get("data", [])
            for m in page or []:
                try:
                    out[int(m["id"])] = m
                except (KeyError, TypeError, ValueError):
                    continue

    return out


def market_tags(market: dict) -> list[str]:
    """Tag labels of a Gamma market dict, empty if it carries none."""
    return [
        str(t.get("label", ""))
        for t in (market.get("tags") or [])
        if t and t.get("label")
    ]


def fetch_trades(
    condition_id: str,
    since_unix: int | None = None,
    max_trades: int = 5000,
) -> list[dict]:
    """
    Fetch public trades for a market, newest first.

    Pages through the data-api until `since_unix` is reached or `max_trades`
    rows have been collected.  Both tokens of the market are returned, which
    is what the training pipeline aggregates over.
    """
    if not condition_id:
        raise ValueError("condition_id is required — fetch it via fetch_market()")

    out: list[dict] = []
    offset = 0

    while len(out) < max_trades:
        page = _get(
            f"{DATA_API}/trades",
            params={
                "market": condition_id,
                "limit": _PAGE_LIMIT,
                "offset": offset,
            },
        )
        if isinstance(page, dict):
            page = page.get("data", [])
        if not page:
            break

        out.extend(page)
        offset += len(page)

        # Results are newest-first: stop once we page past the cutoff
        if since_unix is not None:
            oldest = page[-1].get("timestamp")
            if isinstance(oldest, (int, float)) and oldest < since_unix:
                break
        if len(page) < _PAGE_LIMIT:
            break

    if since_unix is not None:
        out = [
            t for t in out
            if isinstance(t.get("timestamp"), (int, float))
            and t["timestamp"] >= since_unix
        ]
    return out[:max_trades]


def normalize_trades(raw_trades: list[dict], market: MarketInfo) -> pl.DataFrame:
    """
    Convert data-api trade dicts into the schema the pipeline expects.

    Mirrors convert_to_parquet + data_loader:
      - `price` is normalised to P(YES): NO-side trades become 1 - price
      - `usd_amount` = price * size, in the token's own quote
      - `is_yes` comes from outcomeIndex (0 = YES) or the asset token ID,
        never from a fallback that would make it constant
    """
    schema = {
        "timestamp": pl.Datetime("us"),
        "price": pl.Float64,
        "usd_amount": pl.Float64,
        "token_amount": pl.Float64,
        "is_yes": pl.Int8,
    }

    rows = []
    for t in raw_trades:
        ts = _parse_timestamp(t.get("timestamp"))
        price = _safe_float(t.get("price"))
        size = _safe_float(t.get("size"))
        if ts is None or price is None or size is None:
            continue

        is_yes = _resolve_side(t, market)
        if is_yes is None:
            continue

        rows.append(
            {
                "timestamp": ts,
                # YES-normalised price, matching pipeline/data_loader
                "price": price if is_yes == 1 else 1.0 - price,
                "usd_amount": price * size,
                "token_amount": size,
                "is_yes": is_yes,
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows, schema=schema).sort("timestamp")


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Unix seconds or ISO-8601 → tz-naive UTC datetime."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
    try:
        return (
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .replace(tzinfo=None)
        )
    except (ValueError, TypeError):
        return None


def _resolve_side(trade: dict, market: MarketInfo) -> Optional[int]:
    """
    YES (1) or NO (0) for one trade.

    Order of preference:
      1. outcomeIndex   — authoritative, 0 = first outcome = YES
      2. asset token ID — matched against the market's clobTokenIds
      3. outcome string — only for the plain Yes/No wording

    Returns None if the side cannot be determined.  Guessing here would be
    worse than dropping the row: a constant is_yes silently destroys
    yes_ratio, which is the feature the whole label is built on.
    """
    idx = trade.get("outcomeIndex")
    if isinstance(idx, int) and idx in (0, 1):
        return 1 if idx == 0 else 0
    if isinstance(idx, str) and idx.isdigit() and int(idx) in (0, 1):
        return 1 if int(idx) == 0 else 0

    side = market.side_of(str(trade.get("asset", "")))
    if side is not None:
        return side

    outcome = str(trade.get("outcome", "")).strip().upper()
    if outcome == "YES":
        return 1
    if outcome == "NO":
        return 0
    return None
