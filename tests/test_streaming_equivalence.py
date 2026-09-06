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

import numpy as np
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


def test_features_are_chunk_invariant():
    """
    train_chunked.py rebuilds features per time chunk, run_pipeline.py does
    it once over the full history.  Any feature whose value depends on how
    far back the frame starts silently differs between the two.

    volume_concentration used to: it divided by a cumulative sum since the
    market's first bucket, which restarted every chunk (measured: 100 % of
    buckets differing, cum_volume ~11x apart).  It now divides by the
    longest rolling window, which the chunk's context rows cover.
    """
    from datetime import timedelta

    from pipeline.features import build_features

    cfg = PipelineConfig()
    cfg.gap.gap_start = datetime(2099, 1, 1)
    cfg.gap.gap_end = datetime(2099, 2, 1)
    context = max(cfg.features.rolling_windows) * cfg.bucket.bucket_minutes

    try:
        _TMP.mkdir(exist_ok=True)
        trades = generate_synthetic_trades(n_markets=2, trades_per_market=3000)
        markets = generate_synthetic_markets(n_markets=2)
        bucketed = pl.read_parquet(
            aggregate_trades(trades, cfg.bucket, output_path=str(_TMP / "cb.parquet"))
        )
        t_min, t_max = bucketed["bucket_time"].min(), bucketed["bucket_time"].max()
        mid = t_min + (t_max - t_min) / 2

        def _features(frame, tag):
            p = fill_buckets(frame, cfg.bucket, cfg.gap,
                             output_path=str(_TMP / f"cf_{tag}.parquet"))
            p = detect_consecutive_gaps(p, cfg.gap,
                                        output_path=str(_TMP / f"cg_{tag}.parquet"))
            p = apply_gap_exclusions(p, cfg.gap,
                                     output_path=str(_TMP / f"ce_{tag}.parquet"))
            return build_features(pl.read_parquet(p), markets, cfg.features)

        full = _features(bucketed, "full")
        chunked = _features(
            bucketed.filter(
                pl.col("bucket_time") >= mid - timedelta(minutes=context)
            ),
            "chunk",
        )

        joined = (
            full.filter(pl.col("bucket_time") >= mid)
            .select(["market_id", "bucket_time", "volume_concentration"])
            .join(
                chunked.filter(pl.col("bucket_time") >= mid)
                .select(["market_id", "bucket_time", "volume_concentration"]),
                on=["market_id", "bucket_time"],
                suffix="_chunked",
            )
            .drop_nulls()
        )
        assert joined.height > 100, "not enough overlapping buckets to compare"

        a = joined["volume_concentration"].to_numpy()
        b = joined["volume_concentration_chunked"].to_numpy()
        # Rolling sums accumulate in a different order, so allow float noise
        # but nothing beyond it.
        assert np.allclose(a, b, rtol=1e-9), (
            f"volume_concentration differs between full and chunked runs: "
            f"max rel diff {np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)):.3e}"
        )
        print(f"  volume_concentration chunk-invariant ({joined.height} buckets)")
    finally:
        import shutil
        shutil.rmtree(_TMP, ignore_errors=True)


def test_live_bucket_truncation_matches_polars():
    """
    The live path drops the still-open bucket, so its idea of "current
    bucket" must be the same one aggregation.py builds.  Flooring the
    minute field by hand only agrees when bucket_minutes divides 60.
    """
    from pipeline.live_features import _truncate

    t = datetime(2024, 1, 1, 13, 47, 30)
    for bucket_minutes in (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 60, 90, 120, 240):
        expected = pl.Series([t]).dt.truncate(f"{bucket_minutes}m")[0]
        assert _truncate(t, bucket_minutes) == expected, (
            f"bucket_minutes={bucket_minutes}: "
            f"{_truncate(t, bucket_minutes)} != {expected}"
        )
    print("  live bucket truncation matches aggregation for 1-240 min")


if __name__ == "__main__":
    print("=" * 60)
    print("  streaming / cross-path consistency tests")
    print("=" * 60)
    test_batched_streaming_matches_single_market()
    test_features_are_chunk_invariant()
    test_live_bucket_truncation_matches_polars()
    print("  ALL PASSED")
