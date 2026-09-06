"""
Market segmentation — sports / crypto / politics / other.

Why this is regex work and not a lookup
---------------------------------------
markets.csv carries no category (only id, question, answer1/2, volume,
closedTime, market_slug, ticker, token1/2), and no script in this repo
produces that file — it is an external export.

Polymarket's Gamma API does know categories, but only for markets that are
still open: in a 60-market sample every CLOSED market came back with an
empty tag list, while 500/500 active ones were tagged.  A historical trade
dataset is almost entirely closed markets, so the API cannot label it.

What the API can do is grade this classifier: for active markets we have
both the slug and the authoritative tag.  classify_markets.py --validate
does exactly that and reports precision/recall per segment, so the accuracy
is a measured number rather than an assumption.

Design rules
------------
- `market_slug` is the primary signal, `question` the fallback.
- Rule order is part of the definition: sports, then crypto, then politics.
  A market mentioning both stays deterministic.
- Anything unmatched becomes "other", never a subject segment.  A filter
  built on this must not silently drop data into the wrong bucket, and a
  too-eager sports rule would do exactly that.
"""

from __future__ import annotations

import re

import polars as pl

SEGMENTS = ("sports", "crypto", "politics", "other")

# ── Sports ────────────────────────────────────────────────────────────────
# Leagues and competitions, as they appear in real slugs
_SPORT_LEAGUE = (
    r"nba|nfl|nhl|mlb|ncaa|wnba|mls|cfb|cbb|"
    r"epl|ucl|uel|efl|laliga|bundesliga|seriea|ligue1|eredivisie|"
    r"bl1|bl2|es1|es2|it1|it2|fr1|en1|en2|pt1|nl1|tr1|"
    r"uefa|fifa|copa|concacaf|afcon"
)
# Individual sports and series
_SPORT_KIND = (
    r"f1|formula1|ufc|mma|boxing|atp|wta|tennis|golf|pga|liv-golf|masters|"
    r"cricket|ipl|nascar|motogp|cycling|tour-de-france|olympics?|"
    r"soccer|football|basketball|baseball|hockey|rugby|darts|snooker"
)
# Match / competition structure
_SPORT_STRUCT = (
    r"-vs-|halftime|-ou-|over-under|first-goalscorer|clean-sheet|"
    r"ballon-dor|drivers-champion|constructors-champion|"
    r"super-bowl|world-series|stanley-cup|world-cup|champions-league|"
    r"playoffs?|finals?|grand-slam|wimbledon|us-open|french-open"
)
# Compact fixture slugs, e.g. "bl2-elv-svd-2025-11-30-elv"
# league code - team - team - date
_SPORT_FIXTURE = re.compile(
    r"^[a-z]{2,6}-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d{2}-\d{2}", re.I
)

_SPORT = re.compile(
    rf"(?:^|[-\s])(?:{_SPORT_LEAGUE}|{_SPORT_KIND})(?:[-\s]|$)|{_SPORT_STRUCT}",
    re.I,
)

# ── Crypto ────────────────────────────────────────────────────────────────
_CRYPTO_TICKER = (
    r"btc|eth|sol|xrp|doge|ada|bnb|ltc|avax|link|dot|matic|uni|aave|sui|"
    r"apt|arb|op|ton|trx|shib|pepe|wif|bonk|hype|near|fil|icp|atom"
)
_CRYPTO_WORD = (
    r"bitcoin|ethereum|solana|dogecoin|ripple|cardano|litecoin|uniswap|"
    r"crypto|altcoin|stablecoin|defi|nft|memecoin|airdrop|pumpfun|"
    r"onchain|blockchain|halving|staking"
)
# Crypto-native market shapes: fully-diluted-valuation bets and token launches
_CRYPTO_STRUCT = r"-fdv-|fdv-above|token-launch|launch-a-token|-airdrop"

_CRYPTO = re.compile(
    rf"(?:^|[-\s])(?:{_CRYPTO_TICKER}|{_CRYPTO_WORD})(?:[-\s]|$)|{_CRYPTO_STRUCT}",
    re.I,
)

# ── Politics ──────────────────────────────────────────────────────────────
# Structural terms only.  Person names are unbounded, so recall here is
# deliberately limited — unmatched political markets land in "other", which
# is the safe direction.
_POLITICS_OFFICE = (
    r"election|elections|elected|president|presidential|"
    r"senate|senator|governor|midterms?|primary|primaries|congress|"
    r"parliament|chancellor|prime-minister|mayor|nominee|nomination|"
    r"impeach|impeachment|cabinet|referendum|coalition|"
    r"republican|democrat|democrats|gop|scotus"
)
# Geopolitics and policy — the Gamma tag "Geopolitics" maps to politics too,
# so these have to be covered or recall collapses.
_POLITICS_GEO = (
    r"nato|g7|g20|opec|sanctions|ceasefire|treaty|annex|"
    r"tariffs?|shutdown|veto|pardon|pardons|indict|indicted|"
    r"nobel-peace|peace-deal|peace-plan|military-clash|invasion|"
    r"statehood|recognize|recognise|coup|referendum"
)
# Named political figures.  Deliberately a short, high-confidence list —
# person names are unbounded, so this is not a route to full recall.
_POLITICS_FIGURE = r"trump|biden|harris|putin|zelensky|xi-jinping|netanyahu|maduro"

_POLITICS = re.compile(
    rf"(?:^|[-\s])(?:{_POLITICS_OFFICE}|{_POLITICS_GEO}|{_POLITICS_FIGURE})"
    rf"(?:[-\s]|$)",
    re.I,
)


def classify(slug: str, question: str = "") -> str:
    """
    Segment for one market.

    Checked in a fixed order (sports, crypto, politics) so a market matching
    several patterns always lands in the same place.  Returns "other" when
    nothing matches — never a subject segment on a guess.
    """
    slug = (slug or "").strip()
    question = (question or "").strip()
    text = f"{slug} {question}"

    if _SPORT_FIXTURE.match(slug) or _SPORT.search(text):
        return "sports"
    if _CRYPTO.search(text):
        return "crypto"
    if _POLITICS.search(text):
        return "politics"
    return "other"


def classify_frame(
    markets: pl.DataFrame,
    slug_col: str = "market_slug",
    question_col: str = "question",
) -> pl.DataFrame:
    """Add a `segment` column to a markets frame."""
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
    segments = [classify(s or "", q or "") for s, q in zip(slugs, questions)]
    return markets.with_columns(pl.Series("segment", segments, dtype=pl.Utf8))


def load_segment_map(path: str) -> dict[int, str]:
    """
    Read a segment map written by classify_markets.py into
    {market_id: segment}.
    """
    df = pl.read_parquet(path) if str(path).endswith(".parquet") else pl.read_csv(path)
    missing = {"market_id", "segment"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing column(s): {', '.join(sorted(missing))}. "
            f"Build it with classify_markets.py."
        )
    return dict(
        zip(
            df["market_id"].cast(pl.Int64).to_list(),
            df["segment"].to_list(),
        )
    )


def segment_of(market_id, segment_map: dict[int, str]) -> str:
    """
    Segment of one market, with the documented default.

    A market missing from the map counts as "other" — not as "unknown" and
    not as dropped.  Both filter paths must agree on this, or `--segment
    other` selects different market sets in sweep_horizon.py and
    benchmark_strategies.py.
    """
    if market_id is None:
        return "other"
    try:
        return segment_map.get(int(market_id), "other")
    except (TypeError, ValueError):
        return "other"


def segment_series(market_ids, segment_map: dict[int, str]) -> pl.Series:
    """Vectorised `segment_of` for a whole column."""
    return pl.Series(
        "segment",
        [segment_of(m, segment_map) for m in market_ids],
        dtype=pl.Utf8,
    )


# Gamma tag labels that identify a segment, for validation and enrichment.
# Lower-cased on both sides before comparing.
_TAG_TO_SEGMENT = {
    "sports": "sports", "nfl": "sports", "nba": "sports", "mlb": "sports",
    "nhl": "sports", "soccer": "sports", "football": "sports",
    "basketball": "sports", "baseball": "sports", "tennis": "sports",
    "golf": "sports", "ufc": "sports", "mma": "sports", "boxing": "sports",
    "cricket": "sports", "f1": "sports", "formula 1": "sports",
    "olympics": "sports", "epl": "sports", "ncaa": "sports",
    "crypto": "crypto", "crypto prices": "crypto", "bitcoin": "crypto",
    "ethereum": "crypto", "altcoins": "crypto", "defi": "crypto",
    "politics": "politics", "elections": "politics", "us election": "politics",
    "midterms": "politics", "geopolitics": "politics",
    "global elections": "politics", "world elections": "politics",
    "primaries": "politics", "president": "politics",
}


def segment_from_tags(tags: list[str]) -> str | None:
    """
    Authoritative segment from Gamma tags, or None if the tags say nothing.

    Sports wins over crypto wins over politics, mirroring classify()'s order
    so the two can be compared row for row.
    """
    mapped = {
        _TAG_TO_SEGMENT[t.strip().lower()]
        for t in (tags or [])
        if t and t.strip().lower() in _TAG_TO_SEGMENT
    }
    for seg in ("sports", "crypto", "politics"):
        if seg in mapped:
            return seg
    return None
