"""
The trainable universe — which markets this pipeline will learn from.

Four exclusions, for two different kinds of reason.

Structural: high-frequency candles
----------------------------------
Every lag and rolling feature is computed `.over("market_id")` (see
`rowgroups.py`) with windows up to `max(rolling_windows) = 48` buckets, and
a label needs `forward_window_buckets = 6` more.  At the default 5-minute
aggregation a market must trade for well over four hours before it produces
a single complete feature row.

A `btc-updown-5m` market lives 300 seconds — exactly one bucket.  Measured
on the 3,195,250-market export, 604,581 slugs (18.9 %) carry a scheduled
unix timestamp and 667,319 (20.9 %) are candle instruments.  They contribute
no trainable rows, so excluding them costs nothing the pipeline could have
used.  Pooling them by family does not rescue them either: the windows
restart per `market_id` regardless of how the markets are grouped.

Thematic: sports results and asset prices
-----------------------------------------
Excluded by choice, not by structure.  Both are dominated by mechanical,
externally-driven repricing rather than the belief-revision behaviour the
model is meant to learn.

Crypto prices went first; equities, indices and other externally-priced
assets followed for the identical reason, once a registry built without
that rule showed 7,016 of them (0.74 % of the surviving universe) sitting
in the same `<symbol>-above-<strike>-on-<date>` template as the crypto
candles — they had only survived because they are not crypto.

Why this cannot just call `segments.classify()`
-----------------------------------------------
It mostly does — but the direction of use is inverted, and that matters.

`segments.py` is tuned for **precision**: its own docstring says "anything
unmatched becomes `other`, never a subject segment", because a too-eager
rule would drop foreign markets into a segment someone then trains on.
Measured against Gamma tags: precision 100 %, but recall 97.4 % (sports),
72.5 % (crypto), 84.0 % (politics).

Selecting a segment cares about precision.  *Excluding* one cares about
**recall** — every miss is a market that stays in the training set.  At
72.5 % crypto recall, roughly a quarter of crypto markets would survive.

So the crypto rule here is deliberately wider than `segments.classify()`:
unambiguous tickers the segment list omits, plus a price-bet shape
(`-above-`, `-reach-`, `-close-above-` next to a number) that only fires in
combination with a crypto hint.  Widening it trades precision for recall on
purpose; `build_registry.py --validate` reports how many markets each extra
rule removes so the trade stays visible instead of implicit.

The crypto *hint* rule is still intentionally not applied on shape alone:
`_PRICE_SHAPE` needs a crypto signal beside it, so a prose market like
`will-supply-chain-costs-exceed-5-billion` stays.  Templated asset prices
are caught separately by `_ASSET_PRICE`, which anchors on the symbol at the
start of the slug.

A known gap follows from that anchoring: prose phrasings such as
`will-sp500-close-above-6000` are kept, because only the templated
`spx-above-6000-on-<date>` form matches.  Polymarket mints the templated
form in bulk and the prose form rarely, so this trades a little recall for
a rule that cannot misfire on ordinary sentences.
"""

from __future__ import annotations

import re

import polars as pl

from pipeline import segments
from pipeline.families import is_periodic

# Order matters: the first rule that fires names the reason, so a candle
# market is reported as "periodic" rather than "crypto" even though it is
# both.  Structural exclusions come first because they are not negotiable.
EXCLUDE_REASONS = ("periodic", "sports", "crypto", "asset_price", "short_lived")

# Templated strike bets on an externally-priced asset:
# `<symbol>-above|below-<strike>-on-<date>`.  Structurally identical to the
# crypto candles — a mechanical repricing driven by an outside feed rather
# than by belief revision — and excluded for the same reason.  The crypto
# rule runs first, so `bitcoin-above-...` is reported as crypto; what lands
# here is equities (aapl, nvda, tsla, msft, googl, amzn, meta, pltr, nflx,
# open, mu, abnb, rklb), indices (spx, ndx, djia, rut, nya), private
# valuations (spcx) and CS:GO skin prices (case, knife, glove).
_ASSET_PRICE = re.compile(r"^[a-z]{1,6}-(?:above|below)-\$?\d", re.I)


def is_asset_price(slug: str) -> bool:
    """True for `<symbol>-above|below-<strike>-...` templated price bets."""
    return bool(_ASSET_PRICE.match((slug or "").strip()))

# Tickers and words that are unambiguously crypto and are missing from
# `segments._CRYPTO_TICKER`.  Short English words (`sei`, `tao`, `dai`,
# `sand`, `mana`) are deliberately left out of this list — they are handled
# by the price-shape rule below, which needs a second signal before firing.
_EXTRA_TICKER = (
    r"zec|xlm|algo|hbar|mkr|rune|inj|tia|jup|pyth|ena|ondo|ftm|egld|imx|"
    r"snx|kas|usdt|usdc|busd|steth|wbtc|matic|ldo|crv|comp|yfi|cake|floki|"
    r"fartcoin|popcat|brett|mog|turbo|goat|virtual"
)
_EXTRA_WORD = (
    r"coinbase|binance|kraken|bitfinex|tether|satoshi|hashrate|mining-difficulty|"
    r"market-cap|marketcap|spot-etf|token-price|coin-price|dominance"
)
_CRYPTO_EXTRA = re.compile(
    rf"(?:^|[-\s])(?:{_EXTRA_TICKER})(?:[-\s]|$)|(?:{_EXTRA_WORD})", re.I
)

# The shape of a price bet: a comparison word sitting next to a number.
# Only meaningful here in combination with a crypto hint.
_PRICE_SHAPE = re.compile(
    r"(?:above|below|reach|reaches|hit|hits|exceed|exceeds|dip-to|drop-to|"
    r"close-above|close-below|greater-than|less-than)[-\s]*\$?\d|"
    r"\d+k?[-\s]*(?:or-higher|or-lower|by-)",
    re.I,
)

# A weak crypto hint: enough to qualify a price-shaped market, not enough
# on its own.  Includes the ambiguous short tickers left out above.
_CRYPTO_HINT = re.compile(
    r"(?:^|[-\s])(?:sei|tao|dai|sand|mana|axs|grt|ada|vet|qnt|stx|"
    r"rndr|fet|arkm|strk)(?:[-\s]|$)|"
    r"coin|token|crypto|wallet|defi|blockchain|onchain",
    re.I,
)


# Compact fixture slugs: league code, two team codes, ISO date.
#
# `segments._SPORT_FIXTURE` has the same intent but spells the team codes
# `[a-z]{2,4}`, so it misses every abbreviation containing a digit — which
# is most esports rosters (`9z`, `t1`, `nv2`, `rrq1`, `fox1`) and a good
# deal of football (`har2`, `hay1`).  Measured against a registry built
# without this rule, 86,014 markets (9.1 % of the surviving universe) were
# sports fixtures: efa 25,663 · lol 17,715 · itf 12,345 · val 8,133 ·
# crint, cwbb, bun, dfb, ahl, codmw and more.  A 25-slug random sample was
# sports without exception.
#
# The shape is specific enough to stand alone: three short tokens followed
# by a full ISO date.  It is kept here rather than widened in `segments.py`
# because that classifier is deliberately precision-tuned for *selecting* a
# segment, and this module is the one that needs recall for *excluding* one.
_FIXTURE_SHAPE = re.compile(
    r"^[a-z]{2,6}-[a-z0-9]{1,5}-[a-z0-9]{1,5}-\d{4}-\d{2}-\d{2}", re.I
)


def is_sports_fixture(slug: str) -> bool:
    """True for compact `league-team-team-date` fixture slugs."""
    return bool(_FIXTURE_SHAPE.match((slug or "").strip()))


def is_crypto_price(
    slug: str, question: str = "", segment: str | None = None
) -> bool:
    """
    True for crypto-price markets, wider than `segments.classify() == "crypto"`.

    Three ways to qualify:

    1. the segment classifier already says crypto;
    2. an unambiguous ticker or crypto venue the classifier's list omits;
    3. a price-bet shape *and* a weak crypto hint — this is the pair that
       catches templates like `bitcoin-above-70400-on-june-1-2026-12pm-et`
       when the asset name is spelled in a way rule 2 misses.

    Pass `segment` when the caller already classified the market.  The
    classifier runs several regexes over slug + question, and over 3.2 M
    markets recomputing it here (and again in `exclusion_reason`) tripled
    the work for an identical answer.
    """
    text = f"{slug or ''} {question or ''}".strip()
    if not text:
        return False
    if segment is None:
        segment = segments.classify(slug or "", question or "")
    if segment == "crypto":
        return True
    if _CRYPTO_EXTRA.search(text):
        return True
    return bool(_PRICE_SHAPE.search(text) and _CRYPTO_HINT.search(text))


def exclusion_reason(
    slug: str,
    question: str = "",
    duration_minutes: float | None = None,
    min_duration_minutes: float = 720.0,
    segment: str | None = None,
) -> str | None:
    """
    Why this market is excluded, or None when it stays in the universe.

    `min_duration_minutes` defaults to 720 (12 h) — three times the longest
    feature window, so a surviving market has room for the window to fill
    *and* for labelled rows to follow it.  A market with unknown duration
    (`None`) is kept: `end_date_iso` is missing for ~0.4 % of the export,
    and dropping on missing metadata would silently bias the universe.

    `segment` short-circuits the classifier when the caller already has it.
    """
    slug = slug or ""
    question = question or ""

    if is_periodic(slug):
        return "periodic"
    if segment is None:
        segment = segments.classify(slug, question)
    if segment == "sports" or is_sports_fixture(slug):
        return "sports"
    if is_crypto_price(slug, question, segment=segment):
        return "crypto"
    if is_asset_price(slug):
        return "asset_price"
    if duration_minutes is not None and duration_minutes < min_duration_minutes:
        return "short_lived"
    return None


def add_universe(
    markets: pl.DataFrame,
    slug_col: str = "market_slug",
    question_col: str = "question",
    duration_col: str | None = "duration_minutes",
    min_duration_minutes: float = 720.0,
) -> pl.DataFrame:
    """
    Add `exclude_reason` (nullable) and `keep` (Boolean) to a markets frame.

    Every market keeps a row.  The universe is a *view* over the registry,
    not a subset of it — so `market_id` stays stable when the filter is
    retuned, and any Parquet built against an earlier filter stays valid.
    """
    slugs = (
        markets[slug_col].to_list()
        if slug_col in markets.columns
        else [""] * markets.height
    )
    questions = (
        markets[question_col].to_list()
        if question_col in markets.columns
        else [""] * markets.height
    )
    if duration_col and duration_col in markets.columns:
        durations = markets[duration_col].to_list()
    else:
        durations = [None] * markets.height

    # Reuse the segment when the frame already carries one (classify_frame
    # ran first), so the classifier's regexes run once per market instead of
    # three times.
    if "segment" in markets.columns:
        seg_values = markets["segment"].to_list()
    else:
        seg_values = [None] * markets.height

    reasons = [
        exclusion_reason(s or "", q or "", d, min_duration_minutes, segment=seg)
        for s, q, d, seg in zip(slugs, questions, durations, seg_values)
    ]
    return markets.with_columns(
        pl.Series("exclude_reason", reasons, dtype=pl.Utf8),
    ).with_columns(
        pl.col("exclude_reason").is_null().alias("keep"),
    )
