"""
Market segmentation.

The classifier is regex over slugs, so the tests use REAL slugs collected
from the Gamma API rather than invented ones — an invented slug proves
nothing about a pattern meant to match production data.

Run with:  python tests/test_segments.py   (or: pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.segments import (
    SEGMENTS,
    classify,
    classify_frame,
    load_segment_map,
    segment_from_tags,
)

# Real slugs, taken from live Gamma responses
_SPORTS = [
    "will-lewis-hamilton-be-the-2026-f1-drivers-champion",
    "will-mohamed-salah-win-the-2026-ballon-dor",
    "bl2-elv-svd-2025-11-30-elv",            # compact fixture slug
    "es2-cas-lpm-2025-11-30-cas",
    "will-sung-jae-im-win-the-2026-masters",
    "will-mclaren-be-the-2026-f1-constructors-champion",
]
_CRYPTO = [
    "will-btc-reach-150k-by-december-31-2026",
    "will-uni-reach-15pt50-by-december-31-2026",
    "reya-fdv-above-300m-one-day-after-launch-389",
    "will-extended-launch-a-token-by-december-31-2026",
]
_POLITICS = [
    "xi-jinping-out-before-2027",
    "will-gavin-newsom-win-the-2028-democratic-presidential-nomination-568",
    "will-trump-pardon-ghislaine-maxwell-before-2027",
    "china-x-taiwan-military-clash-before-2027",
]
_OTHER = [
    "will-kim-kardashian-and-kanye-west-divorce-before-2027",
    "will-airbnb-begin-publicly-trading-before-jan-2027",
]


def test_real_slugs_land_in_the_right_segment():
    for slug in _SPORTS:
        assert classify(slug) == "sports", slug
    for slug in _CRYPTO:
        assert classify(slug) == "crypto", slug
    for slug in _POLITICS:
        assert classify(slug) == "politics", slug
    print(f"  {len(_SPORTS)} sports / {len(_CRYPTO)} crypto / "
          f"{len(_POLITICS)} politics slugs classified correctly")


def test_unknown_slugs_fall_through_to_other():
    """
    Unmatched markets must never be guessed into a subject segment — a
    filter built on this would silently train on the wrong data.
    """
    for slug in _OTHER + ["", "some-completely-unrelated-thing-2027"]:
        assert classify(slug) == "other", slug
    print("  unknown slugs fall through to `other`")


def test_rule_order_is_deterministic():
    """
    A slug matching several segments must always land in the same one.
    The documented order is sports, then crypto, then politics.
    """
    both = "nba-finals-btc-prop-2026-presidential-election"
    assert classify(both) == "sports", classify(both)
    assert classify("btc-price-after-the-election") == "crypto"
    # And it must not depend on where the match sits in the string
    assert classify("election-nba-game") == "sports"
    print("  rule order sports > crypto > politics holds")


def test_question_is_used_when_the_slug_says_nothing():
    assert classify("market-12345", "Will the NBA finals go to game 7?") == "sports"
    assert classify("market-12345", "") == "other"
    print("  question is used as a fallback signal")


def test_segment_from_tags_matches_classify_order():
    """Validation only works if both sides resolve conflicts the same way."""
    assert segment_from_tags(["Sports", "Crypto"]) == "sports"
    assert segment_from_tags(["Crypto", "Politics"]) == "crypto"
    assert segment_from_tags(["Politics"]) == "politics"
    # The placeholder tag that closed markets carry says nothing
    assert segment_from_tags(["All"]) is None
    assert segment_from_tags([]) is None
    print("  tag resolution mirrors classify() and ignores the `All` placeholder")


def test_classify_frame_and_round_trip():
    tmp = Path("_test_segments.parquet")
    markets = pl.DataFrame({
        "market_id": [1, 2, 3, 4],
        "market_slug": [_SPORTS[0], _CRYPTO[0], _POLITICS[0], _OTHER[0]],
        "question": ["", "", "", ""],
    })
    out = classify_frame(markets)
    assert out["segment"].to_list() == ["sports", "crypto", "politics", "other"]

    try:
        out.select(["market_id", "segment"]).write_parquet(tmp)
        loaded = load_segment_map(str(tmp))
        assert loaded == {1: "sports", 2: "crypto", 3: "politics", 4: "other"}
        print("  classify_frame + load_segment_map round-trip")
    finally:
        tmp.unlink(missing_ok=True)


def test_load_segment_map_rejects_a_bad_file():
    tmp = Path("_test_bad_segments.parquet")
    try:
        pl.DataFrame({"market_id": [1]}).write_parquet(tmp)
        try:
            load_segment_map(str(tmp))
            raise AssertionError("missing `segment` column was accepted")
        except ValueError as exc:
            assert "segment" in str(exc)
        print("  a map without `segment` is rejected with a useful message")
    finally:
        tmp.unlink(missing_ok=True)


def test_favourite_and_longshot_are_complements():
    """
    The two price-based strategies must pick opposite sides of the same
    bucket, and their chosen prices must add up to 1 — otherwise one of
    them is being charged the other one's cost.
    """
    from benchmark_strategies import strat_favourite, strat_longshot

    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.02, 0.98, n)
    df = pl.DataFrame({
        "win": rng.integers(0, 2, n).astype(np.int8),
        "entry_token_price": p,
        "entry_token_price_opp": 1.0 - p,
    })

    _, _, _, _, take_fav = strat_favourite(df)
    _, _, _, _, take_long = strat_longshot(df)

    assert np.array_equal(take_fav, ~take_long), "sides are not complementary"

    fav_price = np.where(take_fav, p, 1.0 - p)
    long_price = np.where(take_long, p, 1.0 - p)
    assert np.allclose(fav_price + long_price, 1.0)
    # The favourite is by definition the dearer side
    assert np.all(fav_price >= long_price)
    print("  favourite/longshot are exact complements, prices sum to 1")


def test_y_true_flips_with_the_chosen_side():
    """`win` scores the dominant side, so taking the other side must flip it."""
    from benchmark_strategies import strat_favourite

    # Dominant side is the cheap one (0.2), so `favourite` takes the other
    df = pl.DataFrame({
        "win": np.array([1, 0], dtype=np.int8),
        "entry_token_price": np.array([0.2, 0.2]),
        "entry_token_price_opp": np.array([0.8, 0.8]),
    })
    y_true, _, _, _, take_dom = strat_favourite(df)
    assert not take_dom.any(), "favourite should take the 0.8 side"
    assert y_true.tolist() == [0.0, 1.0], y_true.tolist()
    print("  y_true flips when the non-dominant side is taken")


if __name__ == "__main__":
    print("=" * 60)
    print("  market segmentation")
    print("=" * 60)
    test_real_slugs_land_in_the_right_segment()
    test_unknown_slugs_fall_through_to_other()
    test_rule_order_is_deterministic()
    test_question_is_used_when_the_slug_says_nothing()
    test_segment_from_tags_matches_classify_order()
    test_classify_frame_and_round_trip()
    test_load_segment_map_rejects_a_bad_file()
    test_favourite_and_longshot_are_complements()
    test_y_true_flips_with_the_chosen_side()
    print("  ALL PASSED")
