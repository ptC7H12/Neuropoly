"""
Batched streaming must produce EXACTLY the same values as one-market-at-a-time.

build_features_streaming and add_labels_streaming process several markets per
call to amortise Polars' per-call overhead.  That is only legitimate because
every expression is market-aware (.over("market_id") or row-wise).  These
tests pin that down — if someone later adds a feature that reads across
markets, they fail.

Run with:  python tests/test_streaming_equivalence.py
"""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig
from pipeline.aggregation import aggregate_trades
from pipeline.features import build_features_streaming
from pipeline.gap_handler import (
    apply_gap_exclusions,
    detect_consecutive_gaps,
    fill_buckets,
)
from pipeline.labeling import add_labels_streaming
from tests.test_pipeline_e2e import generate_synthetic_markets, generate_synthetic_trades

_TMP = Path("_test_stream")


def _prepare(cfg, n_markets=12):
    _TMP.mkdir(exist_ok=True)
    trades = generate_synthetic_trades(n_markets=n_markets, trades_per_market=200)
    markets = generate_synthetic_markets(n_markets=n_markets)
    path = aggregate_trades(trades, cfg.bucket, output_path=str(_TMP / "b.parquet"))
    bucketed = pl.scan_parquet(path).collect()
    p = fill_buckets(bucketed, cfg.bucket, cfg.gap, output_path=str(_TMP / "f.parquet"))
    p = detect_consecutive_gaps(p, cfg.gap, output_path=str(_TMP / "g.parquet"))
    p = apply_gap_exclusions(p, cfg.gap, output_path=str(_TMP / "h.parquet"))
    return p, markets


def test_batched_streaming_matches_single_market():
    cfg = PipelineConfig()
    cfg.gap.gap_start = datetime(2099, 1, 1)
    cfg.gap.gap_end = datetime(2099, 2, 1)

    try:
        filled, markets = _prepare(cfg)

        f1 = build_features_streaming(
            filled, markets, cfg.features,
            output_path=str(_TMP / "feat1.parquet"), batch_markets=1,
        )
        fN = build_features_streaming(
            filled, markets, cfg.features,
            output_path=str(_TMP / "featN.parquet"), batch_markets=5,
        )
        d1 = pl.read_parquet(f1)
        dN = pl.read_parquet(fN)
        assert d1.equals(dN), "batched features differ from per-market features"
        print(f"  features   batch=1 vs batch=5 identical  ({d1.height} rows, {d1.width} cols)")

        # The row-group-per-market invariant must survive batching, because
        # add_labels_streaming reads the file back on that assumption.
        n_markets = d1["market_id"].n_unique()
        assert pq.ParquetFile(fN).metadata.num_row_groups == n_markets, (
            "batched output must still be one row group per market"
        )
        print(f"  row-group-per-market invariant held ({n_markets} markets)")

        l1 = add_labels_streaming(
            f1, cfg.label, output_path=str(_TMP / "lab1.parquet"), batch_markets=1
        )
        lN = add_labels_streaming(
            fN, cfg.label, output_path=str(_TMP / "labN.parquet"), batch_markets=5
        )
        a1 = pl.read_parquet(l1)
        aN = pl.read_parquet(lN)
        assert a1.equals(aN), "batched labels differ from per-market labels"
        assert "trade_return" in a1.columns
        print(f"  labels     batch=1 vs batch=5 identical  ({a1.height} rows)")
    finally:
        import shutil
        shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  streaming batch-equivalence tests")
    print("=" * 60)
    test_batched_streaming_matches_single_market()
    print("  ALL PASSED")
