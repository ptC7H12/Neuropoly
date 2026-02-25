#!/usr/bin/env python3
"""
evaluate_model.py — Evaluate an existing model.txt against historical CSV/Parquet data.

Runs the full preprocessing pipeline (bucketing → gap handling → features → labels →
walk-forward split) but SKIPS training.  Instead it loads an existing model.txt and
evaluates it on each split (train / val / test).

Usage:
    # With already-converted Parquet (recommended, faster):
    python evaluate_model.py \
        --trades data/trades.parquet --trades-format parquet \
        --markets data/markets.parquet --markets-format parquet \
        --model model.txt

    # With raw CSV (note: orderFilled.csv must be converted first):
    python convert_to_parquet.py trades orderFilled.csv data/trades.parquet --markets markets.csv
    python convert_to_parquet.py markets markets.csv data/markets.parquet
    python evaluate_model.py \
        --trades data/trades.parquet --trades-format parquet \
        --markets data/markets.parquet --markets-format parquet

    # Quick test with fewer features (low RAM):
    python evaluate_model.py ... --low-memory

    # Custom threshold for backtest:
    python evaluate_model.py ... --entry-threshold 0.65

Output:
    - ROC AUC, Log Loss, Brier Score on train / val / test
    - Backtest (simulated trades, win rate, ROI, Sharpe, max drawdown)
    - Top 20 most important features
    - Equity curve in terminal
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

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
from pipeline.features import build_features_streaming, get_feature_columns
from pipeline.labeling import add_labels_streaming, label_stats_lazy
from pipeline.model import load_model, predict, feature_importance
from pipeline.splitter import walk_forward_split, print_split_info
from pipeline.evaluation import evaluate, backtest, print_evaluation, EvalMetrics, BacktestResult


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an existing model.txt on historical data (no retraining).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--trades",          default="data/trades.parquet")
    p.add_argument("--markets",         default="data/markets.parquet")
    p.add_argument("--trades-format",   default="parquet", choices=["csv", "parquet", "sqlite"])
    p.add_argument("--markets-format",  default="parquet", choices=["csv", "parquet", "sqlite"])
    p.add_argument("--model",           default="model.txt",
                   help="Path to trained LightGBM model (default: model.txt)")
    p.add_argument("--entry-threshold", type=float, default=0.6,
                   help="P(win) threshold for backtest (default: 0.6)")
    p.add_argument("--bucket-minutes",  type=int,   default=5)
    p.add_argument("--forward-window",  type=int,   default=6,
                   help="Label forward window in buckets (default: 6 = 30 min)")
    p.add_argument("--low-memory",      action="store_true",
                   help="Smaller feature set, less RAM")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep bucketed/filled/features/labeled Parquet files after evaluation")
    return p.parse_args()


def apply_args(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    cfg.data.trades_path     = args.trades
    cfg.data.markets_path    = args.markets
    cfg.data.trades_format   = args.trades_format
    cfg.data.markets_format  = args.markets_format
    cfg.bucket.bucket_minutes = args.bucket_minutes
    cfg.label.forward_window_buckets = args.forward_window
    cfg.backtest.entry_threshold = args.entry_threshold

    if args.low_memory:
        cfg.features.lag_buckets    = [1, 3, 6]
        cfg.features.rolling_windows = [6, 12]
        print("  Low-memory mode: reduced feature set")

    return cfg


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    booster,
    cfg: PipelineConfig,
) -> tuple[EvalMetrics, BacktestResult]:
    """Run evaluate + backtest for one split and return results."""
    y_pred = predict(booster, X)
    metrics = evaluate(y, y_pred, threshold=cfg.backtest.entry_threshold)
    bt = backtest(
        y, y_pred,
        entry_threshold=cfg.backtest.entry_threshold,
        fee_rate=cfg.backtest.fee_rate,
        max_position_usd=cfg.backtest.max_position_usd,
        kelly_sizing=cfg.backtest.kelly_sizing,
        kelly_cap=cfg.backtest.kelly_cap,
        initial_bankroll=cfg.backtest.initial_bankroll,
    )
    return metrics, bt


def print_split_eval(name: str, metrics: EvalMetrics, bt: BacktestResult) -> None:
    w = 55
    print(f"\n{'='*w}")
    print(f"  {name.upper()}")
    print(f"{'='*w}")
    print(f"  ROC AUC  : {metrics.roc_auc:.4f}")
    print(f"  Log Loss : {metrics.log_loss:.4f}")
    print(f"  Brier    : {metrics.brier_score:.4f}")
    print(f"  Accuracy : {metrics.accuracy:.4f}  (threshold {bt.win_rate:.0%})")
    print(f"  Precision: {metrics.precision:.4f}   Recall: {metrics.recall:.4f}   F1: {metrics.f1:.4f}")

    print(f"\n  --- Backtest (entry >= {bt.win_rate:.0%} threshold) ---")
    if bt.total_trades == 0:
        print("  No trades triggered (threshold too high or no qualifying predictions).")
    else:
        print(f"  Trades   : {bt.total_trades}")
        print(f"  Win rate : {bt.win_rate:.2%}")
        print(f"  Total PnL: ${bt.total_pnl:,.2f}")
        print(f"  ROI      : {bt.roi:.2%}")
        print(f"  Sharpe   : {bt.sharpe_ratio:.2f}")
        print(f"  Max DD   : ${bt.max_drawdown:,.2f}  ({bt.max_drawdown_pct:.2%})")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = PipelineConfig()
    cfg = apply_args(cfg, args)

    print("=" * 60)
    print("  MODEL EVALUATION  (no retraining)")
    print("=" * 60)
    print(f"  Model    : {args.model}")
    print(f"  Trades   : {args.trades}  [{args.trades_format}]")
    print(f"  Markets  : {args.markets}  [{args.markets_format}]")
    print(f"  Threshold: {args.entry_threshold}")
    print("=" * 60)
    t0 = time.time()

    # Temp file paths (cleaned up unless --keep-intermediates)
    _tmp = Path("_eval_tmp")
    _tmp.mkdir(exist_ok=True)
    bucketed_path  = str(_tmp / "bucketed.parquet")
    filled_path    = str(_tmp / "filled.parquet")
    features_path  = str(_tmp / "features.parquet")
    labeled_path   = str(_tmp / "labeled.parquet")

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/6] Loading data...")
    trades_lf  = load_trades(cfg.data)
    markets_lf = load_markets(cfg.data)
    markets_df = markets_lf.collect()
    print(f"  Markets: {markets_df.height} rows")

    # ── Step 2: Aggregate into buckets ─────────────────────────
    print(f"\n[2/6] Aggregating into {cfg.bucket.bucket_minutes}-min buckets...")
    aggregate_trades(trades_lf, cfg.bucket, output_path=bucketed_path)
    bucketed = pl.scan_parquet(bucketed_path).collect(engine="streaming")
    n_markets = bucketed["market_id"].n_unique()
    print(f"  {bucketed.height:,} rows across {n_markets} markets")

    # ── Step 3: Gap handling ───────────────────────────────────
    print("\n[3/6] Gap handling...")
    gap_summary = detect_gaps(bucketed, cfg.bucket, cfg.gap)
    print(f"  Mean gap ratio: {gap_summary['gap_ratio'].mean():.2%}  |  "
          f"Markets >50% gaps: {gap_summary.filter(gap_summary['gap_ratio'] > 0.5).height}")

    filled_path = fill_buckets(bucketed, cfg.bucket, cfg.gap,
                               output_path=str(_tmp / "filled.parquet"))
    del bucketed
    gc.collect()

    filled_path = detect_consecutive_gaps(filled_path, cfg.gap,
                                          output_path=str(_tmp / "filled_gaps.parquet"))
    filled_path = apply_gap_exclusions(filled_path, cfg.gap,
                                       output_path=str(_tmp / "filled_final.parquet"))

    stats_lf   = pl.scan_parquet(filled_path)
    n_total    = stats_lf.select(pl.len()).collect()[0, 0]
    n_excluded = stats_lf.filter(pl.col("exclude_from_training")).select(pl.len()).collect()[0, 0]
    print(f"  Rows: {n_total:,}  |  Excluded: {n_excluded:,}")

    # ── Step 4: Features ───────────────────────────────────────
    print("\n[4/6] Engineering features (streaming)...")
    features_path = build_features_streaming(filled_path, markets_df, cfg.features,
                                             output_path=str(_tmp / "features.parquet"))
    _schema = pl.scan_parquet(features_path).limit(0).collect()
    feature_cols = get_feature_columns(_schema)
    del _schema
    print(f"  {len(feature_cols)} feature columns")

    # ── Step 5: Labels ─────────────────────────────────────────
    print(f"\n[5/6] Generating labels (forward {cfg.label.forward_window_buckets} buckets = "
          f"{cfg.label.forward_window_buckets * cfg.bucket.bucket_minutes} min)...")
    labeled_path = add_labels_streaming(features_path, cfg.label,
                                        output_path=str(_tmp / "labeled.parquet"))
    lstats = label_stats_lazy(labeled_path)
    print(f"  Labeled: {lstats['labeled']:,}  |  Win rate: {lstats['win_rate']:.3f}")

    # ── Step 6: Split ──────────────────────────────────────────
    print("\n[6/6] Walk-forward split...")
    labeled = (
        pl.scan_parquet(labeled_path)
        .filter(pl.col("win").is_not_null())
        .collect()
    )
    split = walk_forward_split(labeled, feature_cols, cfg.split, cfg.label)
    del labeled
    gc.collect()
    print_split_info(split)

    # ── Load model (no training) ───────────────────────────────
    print(f"\nLoading model from {args.model} ...")
    if not Path(args.model).exists():
        print(f"ERROR: {args.model} not found. Train a model first with run_pipeline.py or train_chunked.py.")
        sys.exit(1)
    booster = load_model(args.model)
    model_features = booster.feature_name()
    print(f"  Model features: {len(model_features)}")
    print(f"  Split features: {len(split.feature_names)}")

    # Warn if there is a mismatch
    missing_in_data = set(model_features) - set(split.feature_names)
    extra_in_data   = set(split.feature_names) - set(model_features)
    if missing_in_data:
        print(f"  WARNING: {len(missing_in_data)} model features not in data "
              f"(will be NaN): {sorted(missing_in_data)[:5]}{'...' if len(missing_in_data) > 5 else ''}")
    if extra_in_data:
        print(f"  INFO: {len(extra_in_data)} data features not used by model "
              f"(ignored): {sorted(extra_in_data)[:5]}{'...' if len(extra_in_data) > 5 else ''}")

    # Align feature arrays to model's expected order (fill missing with NaN)
    def _align(X: np.ndarray, data_feats: list[str], model_feats: list[str]) -> np.ndarray:
        feat_idx = {f: i for i, f in enumerate(data_feats)}
        out = np.full((X.shape[0], len(model_feats)), np.nan, dtype=np.float32)
        for j, feat in enumerate(model_feats):
            if feat in feat_idx:
                out[:, j] = X[:, feat_idx[feat]]
        return out

    train_X = _align(split.train_X, split.feature_names, model_features)
    val_X   = _align(split.val_X,   split.feature_names, model_features)
    test_X  = _align(split.test_X,  split.feature_names, model_features)

    # ── Evaluate on all three splits ──────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    for split_name, X, y in [
        ("TRAIN (in-sample — expect high)",   train_X, split.train_y),
        ("VALIDATION (out-of-sample)",         val_X,   split.val_y),
        ("TEST (final holdout — trust this)", test_X,  split.test_y),
    ]:
        m, bt = _evaluate_split(split_name, X, y, booster, cfg)
        print_split_eval(split_name, m, bt)

    # Full evaluation printout for test set (includes equity curve + calibration)
    print("\n\n  === FULL TEST SET REPORT ===")
    test_pred = predict(booster, test_X)
    test_metrics = evaluate(split.test_y, test_pred, threshold=cfg.backtest.entry_threshold)
    test_bt = backtest(
        split.test_y, test_pred,
        entry_threshold=cfg.backtest.entry_threshold,
        fee_rate=cfg.backtest.fee_rate,
        max_position_usd=cfg.backtest.max_position_usd,
        kelly_sizing=cfg.backtest.kelly_sizing,
        kelly_cap=cfg.backtest.kelly_cap,
        initial_bankroll=cfg.backtest.initial_bankroll,
    )
    print_evaluation(test_metrics, test_bt)

    # ── Feature importance ─────────────────────────────────────
    fi = feature_importance(booster, model_features, top_n=20)
    print("\n  Top 20 Features (by gain):")
    max_imp = fi[0][1] if fi and fi[0][1] > 0 else 1
    for i, (name, imp) in enumerate(fi, 1):
        bar = "█" * int(imp / max_imp * 30)
        print(f"    {i:>2}. {name:<35}  {imp:>10.0f}  {bar}")

    # ── Cleanup ────────────────────────────────────────────────
    if not args.keep_intermediates:
        import shutil
        shutil.rmtree(_tmp, ignore_errors=True)
    else:
        print(f"\n  Intermediate files kept in: {_tmp}/")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
