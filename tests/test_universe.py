"""
The trainable universe — exclusion rules.

Every slug is a REAL one from poly_data's export.  The crypto cases in
particular were collected from a build_registry.py --validate run, so the
"widened rule" tests pin behaviour against markets the segment classifier
actually misses rather than against invented examples.

Run with:  python tests/test_universe.py   (or: pytest tests/)
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.universe import (
    EXCLUDE_REASONS,
    add_universe,
    exclusion_reason,
    is_crypto_price,
)

# Real crypto markets the precision-tuned segment classifier does NOT catch.
# Collected from --validate output; these are the recall gap the wider rule
# exists to close.
_CRYPTO_MISSED_BY_CLASSIFIER = [
    "will-usdt-depeg-by-june-30",
    "will-usdc-repeg-by-march-31",
    "will-binancecom-become-insolvent-by-jan-31-2023",
    "will-binanceus-shut-down-by-april-30",
    "will-coinbase-resume-usdc-redemptions-by-monday-night",
]
# Price bets that are NOT crypto.  The instruction was to remove crypto
# prices, not every price bet, so these must survive.
_PRICE_BUT_NOT_CRYPTO = [
    "will-sp500-close-above-6000-on-june-1",
    "will-supply-chain-costs-exceed-5-billion",
    "will-the-highest-temperature-in-london-be-between-63f-and-64f",
    "will-gas-prices-reach-5-dollars-by-september",
]
# Markets that must stay in the universe.
_KEPT = [
    "us-presidential-election-winner-2028",
    "elon-musk-of-tweets-september-9-september-16-180-199",
    "israel-hamas-ceasefire-by-june-30",
    "will-donald-trump-sign-an-executive-order-on-immigration",
]


def test_widened_crypto_rule_catches_what_the_classifier_misses():
    from pipeline import segments

    for slug in _CRYPTO_MISSED_BY_CLASSIFIER:
        assert segments.classify(slug) != "crypto", (
            f"{slug} — test premise broken: the classifier now catches this, "
            f"so it no longer demonstrates the recall gap"
        )
        assert is_crypto_price(slug), f"{slug} should be excluded as crypto"
    print(f"  {len(_CRYPTO_MISSED_BY_CLASSIFIER)} recall-gap markets excluded")


def test_price_shape_alone_does_not_exclude():
    """The shape rule needs a crypto hint as a second signal."""
    for slug in _PRICE_BUT_NOT_CRYPTO:
        assert not is_crypto_price(slug), f"{slug} is not a crypto market"
        assert exclusion_reason(slug) != "crypto", slug
    print(f"  {len(_PRICE_BUT_NOT_CRYPTO)} non-crypto price bets kept")


def test_esports_and_digit_team_fixtures_are_excluded():
    """`segments._SPORT_FIXTURE` spells team codes `[a-z]{2,4}` and so misses
    every abbreviation with a digit in it.  A registry built without the wider
    rule kept 86,014 such fixtures — 9.1% of the surviving universe."""
    from pipeline import segments

    leaked = [
        "val-9z-nv2-2025-10-17",                       # Valorant
        "lol-t1-dk-2026-01-31-game4-kill-over-31pt5",  # League of Legends prop
        "efa-har2-hay1-2026-08-08-spread-away-1pt5",   # football, digit codes
        "itf-figl-river-2026-07-01-set-handicap-home-1pt5",
        "cwbb-cark-bella-2026-02-14",
        "codmw-tor1-nyc-2026-03-04",
    ]
    for slug in leaked:
        assert segments.classify(slug) != "sports", (
            f"{slug} — test premise broken: segments.py now catches this"
        )
        assert exclusion_reason(slug) == "sports", (
            f"{slug} -> {exclusion_reason(slug)}"
        )
    print(f"  {len(leaked)} digit-coded fixtures excluded as sports")


def test_fixture_shape_does_not_swallow_ordinary_markets():
    """The shape needs three short tokens *and* a full ISO date."""
    from pipeline.universe import is_sports_fixture

    for slug in _KEPT + _PRICE_BUT_NOT_CRYPTO:
        assert not is_sports_fixture(slug), slug
    assert not is_sports_fixture("us-presidential-election-winner-2028")
    assert not is_sports_fixture("elon-musk-of-tweets-september-9-september-16")
    print("  fixture shape stays off non-sports slugs")


# Templated strike bets on a non-crypto asset.  Real slugs from the registry.
_ASSET_PRICE = [
    "aapl-above-260-on-september-12-2026",
    "nvda-above-180-on-august-8-2026",
    "spx-above-6500-on-july-3-2026",
    "spcx-above-145-on-july-3-2026",     # SpaceX valuation
    "knife-above-195-on-january-20",     # CS:GO skin price
    "open-above-7-on-november-14-2025",  # Opendoor
]


def test_asset_price_markets_are_excluded():
    """Equities, indices and other externally-priced assets use the same
    `<symbol>-above-<strike>-on-<date>` template as the crypto candles and are
    excluded for the same reason."""
    for slug in _ASSET_PRICE:
        assert exclusion_reason(slug) == "asset_price", (
            f"{slug} -> {exclusion_reason(slug)}"
        )
    print(f"  {len(_ASSET_PRICE)} templated asset-price markets excluded")


def test_asset_price_rule_needs_the_symbol_anchor():
    """The rule anchors on a short symbol at the start of the slug, so it
    cannot misfire on ordinary sentences that merely contain "above"."""
    from pipeline.universe import is_asset_price

    for slug in _KEPT + [
        "will-supply-chain-costs-exceed-5-billion",
        "will-the-highest-temperature-in-london-be-between-63f-and-64f",
    ]:
        assert not is_asset_price(slug), slug
    print("  asset-price rule stays off prose slugs")


def test_every_declared_reason_can_actually_fire():
    """Guards against a reason being declared in EXCLUDE_REASONS but never
    wired into exclusion_reason — which is exactly what happened to
    `asset_price` when it was added: the helper existed, the constant listed
    it, and nothing ever called it, so the report showed a permanent zero."""
    examples = {
        "periodic": "btc-updown-5m-1770859800",
        "sports": "val-9z-nv2-2025-10-17",
        "crypto": "will-usdt-depeg-by-june-30",
        "asset_price": "aapl-above-260-on-september-12-2026",
    }
    for reason in EXCLUDE_REASONS:
        if reason == "short_lived":
            got = exclusion_reason("some-plain-market", duration_minutes=1.0)
        else:
            assert reason in examples, f"no example slug for reason {reason!r}"
            got = exclusion_reason(examples[reason])
        assert got == reason, f"reason {reason!r} never fires (got {got!r})"
    print(f"  all {len(EXCLUDE_REASONS)} declared reasons fire")


def test_markets_in_the_universe_are_kept():
    for slug in _KEPT:
        assert exclusion_reason(slug) is None, (
            f"{slug} was excluded as {exclusion_reason(slug)}"
        )
    print(f"  {len(_KEPT)} long-lived markets kept")


def test_rule_order_is_deterministic():
    """A candle is both periodic and crypto; it must report the same reason
    every time, or the exclusion report is not reproducible."""
    assert exclusion_reason("btc-updown-5m-1770859800") == "periodic"
    assert exclusion_reason("zec-updown-5m-1785888600") == "periodic"
    # sports outranks crypto for a market that reads as both
    assert exclusion_reason("will-btc-sponsor-the-super-bowl-2027") == "sports"
    print("  structural rules outrank thematic ones")


def test_duration_rule_and_its_default():
    slug = "will-something-political-happen"
    assert exclusion_reason(slug, duration_minutes=360.0) == "short_lived"
    assert exclusion_reason(slug, duration_minutes=1440.0) is None
    assert exclusion_reason(slug, duration_minutes=None) is None, (
        "unknown duration must not drop a market — end_date_iso is missing "
        "for ~0.4% of the export and dropping on it would bias the universe"
    )
    # boundary is inclusive-below
    assert exclusion_reason(slug, duration_minutes=720.0) is None
    assert exclusion_reason(slug, duration_minutes=719.9) == "short_lived"
    print("  duration threshold and its null-safe default hold")


def test_every_reason_is_a_declared_one():
    slugs = _KEPT + _CRYPTO_MISSED_BY_CLASSIFIER + [
        "btc-updown-5m-1770859800",
        "bl2-elv-svd-2025-11-30-elv",
    ]
    for slug in slugs:
        r = exclusion_reason(slug)
        assert r is None or r in EXCLUDE_REASONS, (slug, r)
    print("  no reason outside EXCLUDE_REASONS")


def test_add_universe_partitions_the_frame():
    """`keep` and `exclude_reason` must be exact complements — the report in
    build_registry.py asserts kept + dropped == total and would fail loudly
    otherwise."""
    slugs = _KEPT + _PRICE_BUT_NOT_CRYPTO + _CRYPTO_MISSED_BY_CLASSIFIER + [
        "btc-updown-5m-1770859800",
        "bl2-elv-svd-2025-11-30-elv",
    ]
    df = pl.DataFrame({"market_slug": slugs, "question": [""] * len(slugs)})
    out = add_universe(df)

    assert out.height == df.height, "add_universe must not drop rows"
    kept = out["keep"].sum()
    dropped = out["exclude_reason"].is_not_null().sum()
    assert kept + dropped == out.height, (kept, dropped, out.height)
    assert kept == len(_KEPT) + len(_PRICE_BUT_NOT_CRYPTO), kept
    print(f"  frame partitions cleanly: {kept} kept, {dropped} excluded")


def test_universe_is_a_view_not_a_subset():
    """Every market keeps a row so that market_id stays stable when the
    filter is retuned."""
    df = pl.DataFrame(
        {"market_slug": ["btc-updown-5m-1770859800"], "question": [""]}
    )
    strict = add_universe(df, min_duration_minutes=1e9)
    loose = add_universe(df, min_duration_minutes=0.0)
    assert strict.height == loose.height == 1
    print("  row count is independent of the filter setting")


def test_precomputed_segment_matches_recomputed():
    """`add_universe` reuses a `segment` column when the frame carries one,
    to avoid running the classifier three times per market.  That shortcut is
    only safe while both paths agree — pin it, because a change to
    classify_frame's inputs would otherwise diverge silently."""
    from pipeline import segments

    slugs = _KEPT + _PRICE_BUT_NOT_CRYPTO + _CRYPTO_MISSED_BY_CLASSIFIER + [
        "btc-updown-5m-1770859800",
        "bl2-elv-svd-2025-11-30-elv",
    ]
    df = pl.DataFrame({"market_slug": slugs, "question": [""] * len(slugs)})

    without = add_universe(df)["exclude_reason"].to_list()
    with_seg = add_universe(segments.classify_frame(df))["exclude_reason"].to_list()

    assert without == with_seg, [
        (s, a, b) for s, a, b in zip(slugs, without, with_seg) if a != b
    ]
    print(f"  precomputed and recomputed segments agree on {len(slugs)} markets")


def test_missing_columns_do_not_raise():
    out = add_universe(pl.DataFrame({"other": [1, 2]}))
    assert out.height == 2
    assert out["keep"].all(), "no slug means no evidence to exclude on"
    print("  a frame without slug/question keeps its rows")


if __name__ == "__main__":
    print("=" * 60)
    print("  trainable universe")
    print("=" * 60)
    test_widened_crypto_rule_catches_what_the_classifier_misses()
    test_price_shape_alone_does_not_exclude()
    test_esports_and_digit_team_fixtures_are_excluded()
    test_fixture_shape_does_not_swallow_ordinary_markets()
    test_asset_price_markets_are_excluded()
    test_asset_price_rule_needs_the_symbol_anchor()
    test_every_declared_reason_can_actually_fire()
    test_markets_in_the_universe_are_kept()
    test_rule_order_is_deterministic()
    test_duration_rule_and_its_default()
    test_every_reason_is_a_declared_one()
    test_add_universe_partitions_the_frame()
    test_universe_is_a_view_not_a_subset()
    test_precomputed_segment_matches_recomputed()
    test_missing_columns_do_not_raise()
    print("  ALL PASSED")
