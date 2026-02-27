#!/usr/bin/env python3
"""
train_chunked.py — RAM-Efficient Incremental Training
======================================================
Processes the full dataset in time-ordered chunks to stay within a 25 GB RAM budget.

Strategy
--------
1. Stream the bucketed Parquet file in fixed-size time windows (e.g. 90 days).
2. For each chunk, carry over a context window of rows from the previous chunk
   so that lag / rolling features are computed correctly at chunk boundaries.
3. Run gap-handling, feature engineering, and labeling on the chunk.
4. Train (or continue training) LightGBM with warm-start via `init_model`.
5. Evaluate on a final hold-out test set (last 15 % of chronological data).

Memory profile (per chunk, 90-day window, 1 500 markets × 5-min buckets):
    ~1 500 * 90 * 288 = ~38 M rows bucketed → ~3–4 GB as float32 feature matrix.
    Peak ≈ 2× during feature construction → ~6–8 GB per chunk.
    Well within a 25 GB budget.

Usage
-----
    python train_chunked.py --bucketed bucketed.parquet --markets markets.parquet

    # Smaller chunks for tighter RAM (30 days):
    python train_chunked.py --bucketed bucketed.parquet --markets markets.parquet \\
        --chunk-days 30 --chunk-size 1000000

    # Resume / continue from a saved model:
    python train_chunked.py ... --init-model model_partial.txt
"""

import argparse
import gc
import sys
import time
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from pipeline.data_loader import load_markets
from pipeline.gap_handler import (
    fill_buckets,
    detect_consecutive_gaps,
    apply_gap_exclusions,
)
from pipeline.features import build_features, get_feature_columns
from pipeline.labeling import add_labels
from pipeline.model import predict, feature_importance
from pipeline.evaluation import evaluate, backtest, print_evaluation


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAM-efficient incremental LightGBM training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bucketed",     required=True, help="Pre-bucketed trades Parquet (from run_pipeline.py step 2)")
    p.add_argument("--markets",      required=True, help="Markets Parquet or CSV")
    p.add_argument("--markets-format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--chunk-days",   type=int,   default=90,   help="Days per training chunk (default: 90)")
    p.add_argument("--context-buckets", type=int, default=60,  help="Extra context rows for lag/rolling at chunk edges")
    p.add_argument("--model-path",   default="model.txt",       help="Output model path")
    p.add_argument("--init-model",   default=None,              help="Resume training from this model file")
    p.add_argument("--test-days",    type=int,   default=60,    help="Hold-out test period in days (from end)")
    p.add_argument("--n-estimators", type=int,   default=200,  help="LightGBM estimators per chunk")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves",   type=int,   default=31)
    p.add_argument("--max-depth",    type=int,   default=7)
    p.add_argument("--n-jobs",       type=int,   default=8)
    p.add_argument("--low-memory",   action="store_true",
                   help="Ultra-low-memory: smaller model, fewer features")
    p.add_argument("--no-eval",      action="store_true",
                   help="Skip final test-set evaluation")
    return p.parse_args()


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_markets(args: argparse.Namespace, cfg: PipelineConfig) -> pl.DataFrame:
    cfg.data.markets_path = args.markets
    cfg.data.markets_format = args.markets_format
    markets_df = load_markets(cfg.data).collect()
    print(f"  Markets: {markets_df.height} rows, {markets_df.width} columns")
    return markets_df


def _time_bounds(bucketed_path: str) -> tuple[pl.Datetime, pl.Datetime]:
    """Return (min_time, max_time) from the bucketed Parquet without loading it all."""
    lf = pl.scan_parquet(bucketed_path)
    result = lf.select([
        pl.col("bucket_time").min().alias("tmin"),
        pl.col("bucket_time").max().alias("tmax"),
    ]).collect()
    return result["tmin"][0], result["tmax"][0]


def _read_chunk(
    bucketed_path: str,
    t_start,        # inclusive (with context)
    t_end,          # inclusive
) -> pl.DataFrame:
    """Read a time-filtered slice from the bucketed Parquet (lazy → collected)."""
    lf = (
        pl.scan_parquet(bucketed_path)
        .filter(
            (pl.col("bucket_time") >= t_start) &
            (pl.col("bucket_time") <= t_end)
        )
    )
    return lf.collect()


# ── Per-chunk processing ───────────────────────────────────────────────────────

def _process_chunk(
    chunk_df: pl.DataFrame,
    markets_df: pl.DataFrame,
    cfg: PipelineConfig,
    chunk_start_actual,   # exclude context rows before this timestamp
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    Run gap handling, features, and labeling on one chunk.
    Returns (X, y, feature_names) for trainable rows in [chunk_start_actual, end].
    Returns None if no valid rows.

    Memory strategy: gap handling writes streaming Parquet (one row-group
    per market).  Then a **single** pass reads each row-group, computes
    features + labels in-memory for that single market, extracts the
    trainable numpy arrays, and discards the DataFrame — no intermediate
    Parquet files for features/labels, no repeated Arrow↔Polars churn.
    """
    if chunk_df.height == 0:
        return None

    # Gap handling — streaming writes, O(1) peak RAM per market
    filled_path = fill_buckets(chunk_df, cfg.bucket, cfg.gap,
                               output_path="_chunk_filled.parquet")
    del chunk_df
    gc.collect()

    filled_path = detect_consecutive_gaps(filled_path, cfg.gap,
                                          output_path="_chunk_filled_gaps.parquet")
    filled_path = apply_gap_exclusions(filled_path, cfg.gap,
                                       output_path="_chunk_filled_final.parquet")

    # ── Single streaming pass: features → labels → extract per market ──
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(filled_path)
    n_rg = pf.metadata.num_row_groups

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    feature_cols: list[str] | None = None
    total_rows = 0
    total_labeled = 0
    total_wins = 0

    for rg_idx in range(n_rg):
        market_df = pl.from_arrow(pf.read_row_group(rg_idx))

        # Features (single market — fast, small)
        featured = build_features(market_df, markets_df, cfg.features)
        del market_df

        if feature_cols is None:
            feature_cols = get_feature_columns(featured)

        # Labels (single market)
        labeled = add_labels(featured, cfg.label)
        del featured

        # Accumulate stats
        total_rows += labeled.height
        win_not_null = labeled["win"].is_not_null()
        n_labeled = win_not_null.sum()
        total_labeled += n_labeled
        if n_labeled > 0:
            total_wins += (labeled["win"] == 1).sum()

        # Extract trainable rows for this market
        trainable = labeled.filter(
            (pl.col("bucket_time") >= chunk_start_actual)
            & pl.col("win").is_not_null()
            & (~pl.col("exclude_from_training"))
        )
        del labeled

        if trainable.height > 0:
            X_parts.append(
                trainable.select(feature_cols).to_numpy().astype(np.float32)
            )
            y_parts.append(trainable["win"].to_numpy().astype(np.float32))
        del trainable

        if (rg_idx + 1) % 500 == 0 or (rg_idx + 1) == n_rg:
            gc.collect()
            print(
                f"    process: {rg_idx + 1}/{n_rg} markets", flush=True
            )

    # Release file handle
    del pf

    win_rate = total_wins / total_labeled if total_labeled > 0 else 0.0
    print(
        f"    Rows: {total_rows:,} | "
        f"Labeled: {total_labeled:,} | "
        f"Win rate: {win_rate:.3f}"
    )

    if not X_parts:
        return None

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()

    return X, y, feature_cols


# ── LightGBM helpers ───────────────────────────────────────────────────────────

def _lgbm_params(args: argparse.Namespace) -> dict:
    params = {
        "objective":        "binary",
        "boosting_type":    "gbdt",
        "learning_rate":    args.learning_rate,
        "num_leaves":       args.num_leaves,
        "max_depth":        args.max_depth,
        "min_child_samples":50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "n_jobs":           args.n_jobs,
        "verbose":         -1,
    }
    if args.low_memory:
        params.update({"num_leaves": 15, "max_depth": 5})
    return params


def _train_chunk(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_estimators: int,
    init_model=None,
    feature_names: list[str] | None = None,   # ← NEU
) -> lgb.Booster:
    """Train (or continue training) a LightGBM model on a chunk."""
    dataset = lgb.Dataset(
        X,
        label=y,
        feature_name=feature_names if feature_names else "auto",  # ← NEU
        free_raw_data=True,
    )

    callbacks = [lgb.log_evaluation(period=50)]

    booster = lgb.train(
        params,
        dataset,
        num_boost_round=n_estimators,
        init_model=init_model,
        callbacks=callbacks,
    )

    del dataset
    gc.collect()

    return booster


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = PipelineConfig()

    if args.low_memory:
        cfg.features.lag_buckets = [1, 3, 6]
        cfg.features.rolling_windows = [6, 12]
        cfg.features.cross_market_features = False
        print("  Low-memory mode: reduced features, smaller model")

    print("=" * 65)
    print("  CHUNKED INCREMENTAL TRAINING  (RAM-efficient)")
    print("=" * 65)

    # ── Load markets (small, keep in RAM) ────────────────────────
    print("\n[1] Loading markets …")
    markets_df = _load_markets(args, cfg)

    # ── Time bounds of the full bucketed dataset ──────────────────
    print("\n[2] Scanning time bounds …")
    t_min, t_max = _time_bounds(args.bucketed)
    print(f"  Data range: {t_min}  →  {t_max}")

    # Reserve last `test_days` as hold-out test set
    chunk_delta  = timedelta(days=args.chunk_days)
    context_us   = args.context_buckets * cfg.bucket.bucket_minutes * 60 * 1_000_000  # microseconds

    # Compute test split boundary
    t_test_start = t_max - timedelta(days=args.test_days)
    t_train_end  = t_test_start - timedelta(hours=1)   # 1-hour gap before test

    print(f"  Training:   {t_min}  →  {t_train_end}")
    print(f"  Test:       {t_test_start}  →  {t_max}")
    print(f"  Chunk size: {args.chunk_days} days")
    print(f"  Context:    {args.context_buckets} buckets at each boundary")

    # ── Chunk loop ────────────────────────────────────────────────
    print("\n[3] Incremental training …\n")
    params = _lgbm_params(args)

    booster = None
    if args.init_model:
        print(f"  Resuming from: {args.init_model}")
        booster = lgb.Booster(model_file=args.init_model)

    feature_names: list[str] = []
    chunk_num = 0
    t_chunk_start = t_min

    while t_chunk_start <= t_train_end:
        t_chunk_end = min(t_chunk_start + chunk_delta, t_train_end)
        chunk_num += 1

        # Start of the read window: include context before the chunk
        # so lag/rolling features at the chunk boundary are correctly computed.
        t_read_start_us = max(
            int(t_chunk_start.timestamp() * 1_000_000) - context_us,
            int(t_min.timestamp() * 1_000_000),
        )
        # Convert back to a compatible type for Polars filter
        t_read_start = pl.Series([t_read_start_us]).cast(
            pl.Datetime("us", t_chunk_start.time_zone if hasattr(t_chunk_start, "time_zone") else None)
        )[0]

        print(
            f"  Chunk {chunk_num:3d}: {t_chunk_start}  →  {t_chunk_end}",
            flush=True,
        )

        t0 = time.time()

        chunk_df = _read_chunk(args.bucketed, t_read_start, t_chunk_end)

        if chunk_df.height == 0:
            print(f"    (empty — skipping)")
            t_chunk_start = t_chunk_end + timedelta(minutes=cfg.bucket.bucket_minutes)
            continue

        print(f"    Read {chunk_df.height:,} rows (incl. context)")

        result = _process_chunk(chunk_df, markets_df, cfg, t_chunk_start)

        del chunk_df
        gc.collect()

        if result is None:
            print(f"    No trainable rows — skipping")
            t_chunk_start = t_chunk_end + timedelta(minutes=cfg.bucket.bucket_minutes)
            continue

        X, y, feature_names = result
        print(f"    X shape: {X.shape} | positives: {int(y.sum()):,}")

        booster = _train_chunk(
            X, y,
            params=params,
            n_estimators=args.n_estimators,
            init_model=booster,
            feature_names=feature_names,   # ← NEU
        )

        del X, y
        gc.collect()

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s | estimators so far: {booster.num_trees()}")

        # Checkpoint after each chunk
        booster.save_model(args.model_path)
        print(f"    Model saved → {args.model_path}")

        t_chunk_start = t_chunk_end + timedelta(minutes=cfg.bucket.bucket_minutes)

    if booster is None:
        print("\nNo training occurred. Check your data paths.")
        return 1

    # ── Final evaluation on hold-out test set ────────────────────
    if not args.no_eval:
        print(f"\n[4] Evaluating on test set ({t_test_start} → {t_max}) …")

        test_df = _read_chunk(args.bucketed, t_test_start, t_max)
        print(f"  Test chunk: {test_df.height:,} rows")

        result = _process_chunk(test_df, markets_df, cfg, t_test_start)
        del test_df
        gc.collect()

        if result is None:
            print("  No testable rows.")
        else:
            X_test, y_test, _ = result
            y_pred = predict(booster, X_test)
            del X_test
            gc.collect()

            metrics = evaluate(
                y_test, y_pred,
                threshold=cfg.backtest.entry_threshold,
            )
            bt = backtest(
                y_test, y_pred,
                entry_threshold=cfg.backtest.entry_threshold,
                fee_rate=cfg.backtest.fee_rate,
                max_position_usd=cfg.backtest.max_position_usd,
                kelly_sizing=cfg.backtest.kelly_sizing,
                kelly_cap=cfg.backtest.kelly_cap,
                initial_bankroll=cfg.backtest.initial_bankroll,
            )
            print_evaluation(metrics, bt)

            # Feature importance
            if feature_names:
                fi = feature_importance(booster, feature_names, top_n=20)
                print("\n  Top 20 Features (gain):")
                for i, (name, imp) in enumerate(fi, 1):
                    bar = "█" * int(imp / fi[0][1] * 25) if fi[0][1] > 0 else ""
                    print(f"    {i:>2}. {name:<35} {imp:>10.0f}  {bar}")

    print(f"\n[Done] Model saved to: {args.model_path}")
    print(f"       Total estimators: {booster.num_trees()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
