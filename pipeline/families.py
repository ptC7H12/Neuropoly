"""
Market families — grouping templated markets that repeat under a new name.

Why this exists
---------------
Polymarket does not mint one market per question.  It mints the *same*
question over and over on a schedule, giving each instance a fresh
condition_id and a slug that differs only in a timestamp or a date:

    btc-updown-5m-1770859800
    btc-updown-5m-1770861900          <- 300 seconds later
    elon-musk-of-tweets-september-9-september-16-180-199
    elon-musk-of-tweets-october-1-october-8-200-219

Measured on the full 3,195,250-market export: 604,581 slugs (18.9 %) end in
a unix timestamp, and the 28 largest families account for 608,701 markets.
Treated as 3.2 M unrelated instruments the data is hopeless; treated as a
few thousand recurring instruments it is stationary.

`market_id` is therefore the wrong grouping key for anything that wants to
learn "how does this kind of market behave".  This module supplies the
right one.

Two keys, deliberately
----------------------
`family()` masks every digit run and month name but keeps the token
structure, so `btc-updown-5m-*` and `btc-updown-15m-*` stay apart — the
window length is part of the instrument, not noise.

`family_root()` truncates at the first mask instead, collapsing everything
that shares a subject up to its first varying slot.  It merges
`elon-musk-of-tweets-<m>-#-<m>-#-#-#` with its year-carrying variant, which
`family()` splits.  Note it does *not* merge `btc-updown-5m` into
`btc-updown`: the duration is not a mask, so it survives truncation.  That
is deliberate — see the digit-run rule below — and moot in practice, since
the candle instruments leave the universe entirely.

Neither is "correct".  Which granularity supports separate models is a
question for the diagnostic, not an assumption to bake in here, so both are
cheap to compute and both are written to the registry.

Periodicity is a separate question from family
----------------------------------------------
`is_periodic()` asks whether a market is one of the high-frequency candle
instruments.  Those are excluded from the trainable universe for a
structural reason, not a thematic one: every lag and rolling feature is
computed `.over("market_id")` (see `rowgroups.py`) with windows up to 48
buckets, so a market that lives 300 seconds — one bucket at the default
5-minute aggregation — yields no complete feature row at all.  Pooling them
by family does not help, because the windows still restart per market.
"""

from __future__ import annotations

import re

import polars as pl

# Digit runs and month names are the two things that vary between instances
# of the same template.  Weekday names are left alone: they appear in
# subjects ("super-bowl-sunday") more often than as template slots.
# A digit run followed by a duration unit is part of the instrument's
# identity, not a template slot: `btc-updown-5m-*` and `btc-updown-15m-*`
# are different instruments that happen to share a prefix, and masking both
# to `#m` would merge them into one family.  Everything else — timestamps,
# strikes, dates, counts — is a slot.
#
# The unit is matched as part of the token rather than tested with a
# lookahead.  A lookahead lets `\d+` backtrack until it fits: `15m` would
# match only `1`, mask it, and yield `#5m` — which collides with `115m`.
# Consuming the unit keeps the digit run atomic.
_DIGIT_RUN = re.compile(r"\d+(?:[mhd](?=-|$))?")

# A month name only counts as a template slot when a number follows it —
# that is what a date looks like (`september-9`).  Without that guard the
# rule fires on ordinary words: "may" is a modal verb far more often than a
# month ("will-x-may-happen"), and `mar`/`sep`/`oct` are ambiguous too.
# Requiring the digit costs nothing, because every real date has one.
_MONTH = re.compile(
    r"(?:^|-)(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)(?=-\d)"
)

DIGIT_MASK = "#"
MONTH_MASK = "<m>"

# A trailing unix timestamp, optionally followed by a strike or index
# suffix: `...-1770859800` and `...-1756483200-4300` both match.  Nine to
# eleven digits keeps it from firing on years or vote counts.
_UNIX_SUFFIX = re.compile(r"-\d{9,11}(?:-\d+)?$")

# Slug fragments that name a high-frequency price-candle instrument.
_CANDLE = re.compile(r"updown|up-or-down|multistrike")


def _mask_run(match: re.Match) -> str:
    """Mask one digit run, unless it carries a duration unit (`5m`, `4h`)."""
    token = match.group(0)
    return token if token[-1] in "mhd" else DIGIT_MASK


def family(slug: str) -> str:
    """
    Template key for one market: the slug with every varying part masked.

    Keeps token structure, so different window lengths stay different
    families:

        >>> family("btc-updown-5m-1770859800")
        'btc-updown-5m-#'
        >>> family("btc-updown-15m-1760045400")
        'btc-updown-15m-#'

    An empty or missing slug yields "" rather than raising — the caller
    decides whether that is an error.
    """
    s = (slug or "").strip().lower()
    if not s:
        return ""
    s = _MONTH.sub(lambda m: ("-" if m.group(0).startswith("-") else "") + MONTH_MASK, s)
    return _DIGIT_RUN.sub(_mask_run, s)


def _is_slot(token: str) -> bool:
    """True when a masked token carries a varying value (`#`, `<m>`, `#pm`)."""
    return DIGIT_MASK in token or MONTH_MASK in token


def family_root(slug: str) -> str:
    """
    Coarse family key: the masked slug with its trailing slots removed.

        >>> family_root("btc-updown-5m-1770859800")
        'btc-updown-5m'
        >>> family_root("elon-musk-of-tweets-september-9-september-16-180-199")
        'elon-musk-of-tweets'

    Trailing rather than leading, because a template's varying part is
    almost always its suffix — a date, a timestamp, a strike, a bucket
    range.  Cutting at the *first* mask instead would collapse every slug
    that merely happens to contain an early number: `will-2026-be-...`
    would become `will`, a bucket that then swallows thousands of unrelated
    markets.  Trailing removal leaves such a slug intact.

    Falls back to the full masked key when every token is a slot, so a
    market never ends up with an empty group.
    """
    fam = family(slug)
    if not fam:
        return ""
    tokens = fam.split("-")
    while tokens and _is_slot(tokens[-1]):
        tokens.pop()
    return "-".join(tokens) or fam


def is_periodic(slug: str) -> bool:
    """
    True when the slug names a scheduled, high-frequency instrument.

    Two independent signals, either is enough:

    * a trailing unix timestamp — the scheduler stamps the resolution
      instant into the name (`btc-updown-5m-1770859800`);
    * a candle keyword (`updown`, `up-or-down`, `multistrike`).

    The keyword test catches the older `btc-up-or-down-15m-*` naming, whose
    timestamp suffix is identical but whose prefix is not, and the timestamp
    test catches scheduled instruments that never say "updown".
    """
    s = (slug or "").strip().lower()
    if not s:
        return False
    return bool(_UNIX_SUFFIX.search(s) or _CANDLE.search(s))


def family_series(slugs, root: bool = False) -> pl.Series:
    """Vectorised `family` (or `family_root`) over a slug column."""
    fn = family_root if root else family
    name = "family_root" if root else "family"
    return pl.Series(name, [fn(s or "") for s in slugs], dtype=pl.Utf8)


def periodic_series(slugs) -> pl.Series:
    """Vectorised `is_periodic` over a slug column."""
    return pl.Series(
        "is_periodic", [is_periodic(s or "") for s in slugs], dtype=pl.Boolean
    )


def add_families(
    markets: pl.DataFrame, slug_col: str = "market_slug"
) -> pl.DataFrame:
    """
    Add `family`, `family_root` and `is_periodic` to a markets frame.

    Mirrors `segments.classify_frame` so the two can be applied in either
    order to the same frame.
    """
    slugs = (
        markets[slug_col].to_list()
        if slug_col in markets.columns
        else [""] * markets.height
    )
    return markets.with_columns(
        family_series(slugs),
        family_series(slugs, root=True),
        periodic_series(slugs),
    )
