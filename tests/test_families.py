"""
Market families and periodicity.

Like test_segments.py, every slug here is a REAL one taken from poly_data's
3,195,250-market export.  Invented slugs would prove nothing about regexes
written to match production naming.

Run with:  python tests/test_families.py   (or: pytest tests/)
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.families import (
    add_families,
    family,
    family_root,
    is_periodic,
)

# Scheduled candle instruments — the structural exclusion.
_PERIODIC = [
    "btc-updown-5m-1770859800",
    "eth-updown-15m-1760040900",
    "sol-multistrike-4h-1756483200-212",
    "btc-up-or-down-15m-1757726100",   # older naming, same instrument
    "zec-updown-5m-1785888600",
]
# Long-lived markets that must NOT be mistaken for scheduled ones.
_NOT_PERIODIC = [
    "elon-musk-of-tweets-september-9-september-16-180-199",
    "us-presidential-election-winner-2028",
    "will-trump-pardon-ghislaine-maxwell-before-2027",
    "bl2-elv-svd-2025-11-30-elv",       # a date, not a unix timestamp
]


def test_instances_of_one_template_share_a_family():
    a = family("btc-updown-5m-1770859800")
    b = family("btc-updown-5m-1770861900")   # 300 s later, same instrument
    assert a == b == "btc-updown-5m-#", (a, b)
    print("  consecutive instances collapse to one family")


def test_window_length_is_identity_not_noise():
    """5m, 15m and 4h are different instruments sharing a prefix.

    Guards the atomic digit-run rule: a naive lookahead lets `\\d+`
    backtrack, so `15m` masks to `#5m` and collides with `115m`.
    """
    fams = {
        w: family(f"btc-updown-{w}-1770859800")
        for w in ("5m", "15m", "115m", "4h", "1d")
    }
    assert len(set(fams.values())) == len(fams), fams
    assert fams["5m"] == "btc-updown-5m-#", fams["5m"]
    assert fams["15m"] == "btc-updown-15m-#", fams["15m"]
    print("  window lengths stay distinct families")


def test_month_names_only_mask_in_front_of_a_date():
    """"may" is a modal verb far more often than a month."""
    assert family("will-trump-may-win-the-nobel-peace-prize") == (
        "will-trump-may-win-the-nobel-peace-prize"
    )
    assert family("elon-musk-of-tweets-may-1-may-8-100-119") == (
        "elon-musk-of-tweets-<m>-#-<m>-#-#-#"
    )
    print("  month masking requires a following number")


def test_family_root_strips_trailing_slots_not_leading_ones():
    """Cutting at the first mask would collapse unrelated markets together."""
    assert family_root("will-2026-be-the-hottest-year") == (
        "will-#-be-the-hottest-year"
    ), "an early number must not collapse the slug to 'will'"
    assert (
        family_root("elon-musk-of-tweets-september-9-september-16-180-199")
        == "elon-musk-of-tweets"
    )
    assert family_root("us-presidential-election-winner-2028") == (
        "us-presidential-election-winner"
    )
    print("  family_root removes trailing slots only")


def test_family_root_never_returns_empty():
    """A slug made only of slots keeps its masked key rather than vanishing."""
    assert family_root("2026-01-01") != ""
    assert family_root("") == ""
    print("  family_root has no empty bucket")


def test_periodicity_detection():
    for slug in _PERIODIC:
        assert is_periodic(slug), f"{slug} should be periodic"
    for slug in _NOT_PERIODIC:
        assert not is_periodic(slug), f"{slug} should not be periodic"
    print(f"  {len(_PERIODIC)} scheduled / {len(_NOT_PERIODIC)} long-lived classified")


def test_a_calendar_date_is_not_a_unix_timestamp():
    """`bl2-...-2025-11-30-elv` must not read as a scheduled instrument.

    The timestamp rule requires 9-11 digits precisely so that years and
    counts do not trigger it.
    """
    assert not is_periodic("bl2-elv-svd-2025-11-30-elv")
    assert is_periodic("x-1770859800")          # 10 digits
    assert not is_periodic("x-2028")            # a year
    assert not is_periodic("x-12345678")        # 8 digits
    print("  only 9-11 digit suffixes count as timestamps")


def test_add_families_on_a_frame():
    df = pl.DataFrame({"market_slug": _PERIODIC[:2] + _NOT_PERIODIC[:2]})
    out = add_families(df)
    for col in ("family", "family_root", "is_periodic"):
        assert col in out.columns, col
    assert out["is_periodic"].to_list() == [True, True, False, False]
    assert out.height == df.height
    print("  add_families adds three columns and keeps every row")


def test_missing_slug_column_does_not_raise():
    """Mirrors segments.classify_frame's tolerance of a missing column."""
    out = add_families(pl.DataFrame({"other": [1, 2]}))
    assert out["family"].to_list() == ["", ""]
    print("  a frame without market_slug yields empty families")


if __name__ == "__main__":
    print("=" * 60)
    print("  market families")
    print("=" * 60)
    test_instances_of_one_template_share_a_family()
    test_window_length_is_identity_not_noise()
    test_month_names_only_mask_in_front_of_a_date()
    test_family_root_strips_trailing_slots_not_leading_ones()
    test_family_root_never_returns_empty()
    test_periodicity_detection()
    test_a_calendar_date_is_not_a_unix_timestamp()
    test_add_families_on_a_frame()
    test_missing_slug_column_does_not_raise()
    print("  ALL PASSED")
