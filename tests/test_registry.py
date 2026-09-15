"""
Market registry id assignment.

The registry's whole job is to give every market one stable integer id.  Both
properties have been broken in practice, and both break *quietly*: a
duplicated id makes the trades join wrong without raising anything.

Run with:  python tests/test_registry.py   (or: pytest tests/)
"""

import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_registry import merge_existing_ids


def _markets(condition_ids):
    return pl.DataFrame({"condition_id": list(condition_ids)})


def _write_prior(tmpdir, pairs):
    """Write a prior registry from (condition_id, market_id) pairs."""
    path = Path(tmpdir) / "prior.parquet"
    pl.DataFrame(
        {
            "condition_id": [c for c, _ in pairs],
            "market_id": pl.Series([m for _, m in pairs], dtype=pl.Int32),
        }
    ).write_parquet(path)
    return str(path)


def test_fresh_registry_numbers_densely_from_zero():
    with tempfile.TemporaryDirectory() as d:
        out = merge_existing_ids(_markets(["0xa", "0xb", "0xc"]), str(Path(d) / "none.parquet"))
    assert out["market_id"].to_list() == [0, 1, 2]
    assert out["market_id"].dtype == pl.Int32
    print("  fresh registry numbers 0..n-1 as Int32")


def test_existing_ids_are_reused_and_new_ones_appended():
    with tempfile.TemporaryDirectory() as d:
        prior = _write_prior(d, [("0xa", 0), ("0xb", 1)])
        out = merge_existing_ids(_markets(["0xa", "0xb", "0xc", "0xd"]), prior)
    got = dict(zip(out["condition_id"].to_list(), out["market_id"].to_list()))
    assert got["0xa"] == 0 and got["0xb"] == 1, got
    assert sorted([got["0xc"], got["0xd"]]) == [2, 3], got
    print("  known markets keep their id, new ones continue the sequence")


def test_a_duplicated_prior_registry_does_not_multiply_rows():
    """A left join against a prior registry holding the same condition_id twice
    emits two rows per input market.  Observed for real: 3,386,625 deduped
    markets came back as 3,411,290 — exactly the 24,665 duplicates the previous
    registry still carried, each one silently sharing a market_id."""
    with tempfile.TemporaryDirectory() as d:
        prior = _write_prior(d, [("0xa", 0), ("0xa", 0), ("0xb", 1)])
        markets = _markets(["0xa", "0xb", "0xc"])
        out = merge_existing_ids(markets, prior)

    assert out.height == markets.height, (
        f"row multiplication: {markets.height} in, {out.height} out"
    )
    assert out["market_id"].n_unique() == out.height, "market_id not unique"
    print("  a duplicated prior registry no longer multiplies rows")


def test_ids_are_stable_across_a_rebuild():
    """market_id must not move when the export grows, or every Parquet built
    earlier refers to the wrong markets."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "reg.parquet")
        first = merge_existing_ids(_markets(["0xa", "0xb"]), path)
        first.write_parquet(path)
        second = merge_existing_ids(_markets(["0xa", "0xb", "0xc"]), path)

    before = dict(zip(first["condition_id"].to_list(), first["market_id"].to_list()))
    after = dict(zip(second["condition_id"].to_list(), second["market_id"].to_list()))
    for cid, mid in before.items():
        assert after[cid] == mid, f"{cid} moved from {mid} to {after[cid]}"
    print("  ids survive a rebuild against a grown export")


if __name__ == "__main__":
    print("=" * 60)
    print("  market registry")
    print("=" * 60)
    test_fresh_registry_numbers_densely_from_zero()
    test_existing_ids_are_reused_and_new_ones_appended()
    test_a_duplicated_prior_registry_does_not_multiply_rows()
    test_ids_are_stable_across_a_rebuild()
    print("  ALL PASSED")
