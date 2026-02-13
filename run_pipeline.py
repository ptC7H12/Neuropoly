#!/usr/bin/env python3
"""
Polymarket Trading Pipeline — Main Entry Point

Usage:
    python run_pipeline.py                          # Run with defaults
    python run_pipeline.py --trades data/trades.csv # Custom paths
    python run_pipeline.py --bucket-minutes 15      # Different bucket size
    python run_pipeline.py --dry-run                # Only show stats, no training

Steps:
    1. Load trades + markets data
    2. Aggregate into time buckets
    3. Detect and handle gaps
    4. Engineer features
    5. Generate labels
    6. Walk-forward split
    7. Train LightGBM with live monitoring
    8. Evaluate + backtest
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from pipeline.data_loader import load_trades, load_markets
from pipeline.aggregation import aggregate_trades
from pipeline.gap_handler import (
    detect_gaps,
    fill_buckets,
    detect_consecutive_gaps,
    apply_gap_exclusions,
)
from pipeline.features import build_features, get_feature_columns
from pipeline.labeling import add_labels, label_stats
from pipeline.splitter import walk_forward_split, print_split_info
from pipeline.model import train_model, predict, feature_importance
from pipeline.evaluation import evaluate, backtest, print_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Trading Pipeline")
    parser.add_argument("--trades", type=str, help="Path to trades data")
    parser.add_argument("--markets", type=str, help="Path to markets data")
    parser.add_argument("--trades-format", type=str, choices=["csv", "parquet", "sqlite"])
    parser.add_argument("--markets-format", type=str, choices=["csv", "parquet", "sqlite"])
    parser.add_argument("--sqlite-path", type=str, help="SQLite DB path (if both tables in same DB)")
    parser.add_argument("--trades-table", type=str, help="SQLite table name for trades")
    parser.add_argument("--markets-table", type=str, help="SQLite table name for markets")
    parser.add_argument("--bucket-minutes", type=int, help="Bucket size in minutes")
    parser.add_argument("--forward-window", type=int, help="Label forward window (buckets)")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-leaves", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--entry-threshold", type=float, help="Backtest entry threshold")
    parser.add_argument("--model-path", type=str, default="model.txt", help="Save path")
    parser.add_argument("--log-interval", type=int, help="Monitor update interval")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich dashboard")
    parser.add_argument("--dry-run", action="store_true", help="Stats only, no training")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Ultra-low memory mode: disable cross-market features, use streaming",
    )
    return parser.parse_args()


def apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Override config with CLI arguments."""

    if args.trades:
        cfg.data.trades_path = args.trades
    if args.markets:
        cfg.data.markets_path = args.markets
    if args.trades_format:
        cfg.data.trades_format = args.trades_format
    if args.markets_format:
        cfg.data.markets_format = args.markets_format
    if args.sqlite_path:
        cfg.data.sqlite_path = args.sqlite_path
    if args.trades_table:
        cfg.data.trades_table = args.trades_table
    if args.markets_table:
        cfg.data.markets_table = args.markets_table
    if args.bucket_minutes:
        cfg.bucket.bucket_minutes = args.bucket_minutes
    if args.forward_window:
        cfg.label.forward_window_buckets = args.forward_window
    if args.learning_rate:
        cfg.model.learning_rate = args.learning_rate
    if args.num_leaves:
        cfg.model.num_leaves = args.num_leaves
    if args.max_depth:
        cfg.model.max_depth = args.max_depth
    if args.n_jobs:
        cfg.model.n_jobs = args.n_jobs
    if args.entry_threshold:
        cfg.backtest.entry_threshold = args.entry_threshold
    if args.log_interval:
        cfg.monitor.log_interval = args.log_interval
    if args.no_rich:
        cfg.monitor.rich_dashboard = False

    # Low-memory mode optimizations
    if hasattr(args, "low_memory") and args.low_memory:
        cfg.features.cross_market_features = False
        cfg.features.lag_buckets = [1, 3, 6]
        cfg.features.rolling_windows = [6, 12]
        cfg.model.num_leaves = 15
        cfg.model.max_depth = 5
        print("  ⚠️  Low-memory mode enabled:")
        print("     - Cross-market features disabled")
        print("     - Reduced feature sets")
        print("     - Smaller model (num_leaves=15, max_depth=5)")

    return cfg


def main():
    args = parse_args()
    cfg = PipelineConfig()
    cfg = apply_args(cfg, args)

    print("=" * 60)
    print("  POLYMARKET TRADING PIPELINE")
    print("=" * 60)
    t0 = time.time()

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/8] Loading data...")
    trades_lf = load_trades(cfg.data)
    markets_lf = load_markets(cfg.data)

    # Collect markets eagerly (small dataset)
    markets_df = markets_lf.collect()
    print(f"  Markets loaded: {markets_df.height} rows")

    # ── Step 2: Aggregate into buckets ─────────────────────────
    print(f"\n[2/8] Aggregating trades into {cfg.bucket.bucket_minutes}-min buckets...")
    bucketed = aggregate_trades(trades_lf, cfg.bucket)
    n_markets = bucketed["market_id"].n_unique()
    print(f"  Bucketed: {bucketed.height} rows across {n_markets} markets")

    # ── Step 3: Gap handling ───────────────────────────────────
    print("\n[3/8] Detecting and handling gaps...")
    gap_summary = detect_gaps(bucketed, cfg.bucket, cfg.gap)
    print(f"  Gap summary for {gap_summary.height} markets:")
    print(f"    Mean gap ratio: {gap_summary['gap_ratio'].mean():.2%}")
    high_gap = gap_summary.filter(gap_summary["gap_ratio"] > 0.5)
    print(f"    Markets with >50% gaps: {high_gap.height}")

    filled = fill_buckets(bucketed, cfg.bucket, cfg.gap)
    filled = detect_consecutive_gaps(filled, cfg.gap)
    filled = apply_gap_exclusions(filled, cfg.gap)

    n_excluded = filled.filter(filled["exclude_from_training"]).height
    print(f"  After filling: {filled.height} rows ({n_excluded} excluded from training)")

    # ── Step 4: Feature engineering ────────────────────────────
    print("\n[4/8] Engineering features...")
    featured = build_features(filled, markets_df, cfg.features)
    feature_cols = get_feature_columns(featured)
    print(f"  Features: {len(feature_cols)} columns")

    # ── Step 5: Labeling ───────────────────────────────────────
    print(f"\n[5/8] Generating labels (forward window: {cfg.label.forward_window_buckets} buckets)...")
    labeled = add_labels(featured, cfg.label)
    stats = label_stats(labeled)
    print(f"  Total rows:  {stats['total_rows']}")
    print(f"  Labeled:     {stats['labeled']}")
    print(f"  Nullified:   {stats['nullified']}")
    print(f"  Win rate:    {stats['win_rate']:.3f}")
    if stats.get("mean_future_return") is not None:
        print(f"  Mean return: {stats['mean_future_return']:.5f}")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before training.")
        print(f"  Time: {time.time() - t0:.1f}s")
        return

    # ── Step 6: Train/Val/Test split ───────────────────────────
    print("\n[6/8] Walk-forward split...")
    split = walk_forward_split(labeled, feature_cols, cfg.split, cfg.label)
    print_split_info(split)

    # ── Step 7: Train LightGBM ─────────────────────────────────
    print("\n[7/8] Training LightGBM...")
    booster, monitor = train_model(
        split, cfg.model, cfg.monitor, save_path=args.model_path
    )

    # Feature importance
    fi = feature_importance(booster, split.feature_names, top_n=20)
    print("\n  Top 20 Features (by gain):")
    for i, (name, imp) in enumerate(fi, 1):
        bar = "█" * int(imp / fi[0][1] * 25) if fi[0][1] > 0 else ""
        print(f"    {i:>2}. {name:<35} {imp:>10.0f}  {bar}")

    # ── Step 8: Evaluation & Backtest ──────────────────────────
    print("\n[8/8] Evaluating on test set...")
    y_pred = predict(booster, split.test_X)

    metrics = evaluate(split.test_y, y_pred, threshold=cfg.backtest.entry_threshold)

    bt = backtest(
        split.test_y,
        y_pred,
        entry_threshold=cfg.backtest.entry_threshold,
        fee_rate=cfg.backtest.fee_rate,
        max_position_usd=cfg.backtest.max_position_usd,
        kelly_sizing=cfg.backtest.kelly_sizing,
        kelly_cap=cfg.backtest.kelly_cap,
        initial_bankroll=cfg.backtest.initial_bankroll,
    )

    print_evaluation(metrics, bt)

    total_time = time.time() - t0
    print(f"\nPipeline completed in {total_time:.1f}s")
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
