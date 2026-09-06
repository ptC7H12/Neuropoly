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
import json
import sys
import time
from datetime import datetime, timezone
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
from pipeline.features import build_features, get_feature_columns
from pipeline.labeling import add_labels
from pipeline.model import load_model, predict, feature_importance
from pipeline.splitter import (
    SplitResult,
    print_split_info,
    purge_minutes,
    time_split_indices,
)
from pipeline.evaluation import evaluate, backtest, print_evaluation, EvalMetrics, BacktestResult
from pipeline.results_logger import append_to_log, print_log_history


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
                   help="Keep the bucketed/filled Parquet files after evaluation. "
                        "Features and labels are NOT written here — the streaming "
                        "path keeps them in RAM only. Use benchmark_strategies.py "
                        "--keep-intermediates if you need labeled.parquet.")
    p.add_argument("--log-file", default="results_log.jsonl", metavar="PATH",
                   help="JSONL file to append results to for historical tracking "
                        "(default: results_log.jsonl). Pass '' to disable.")
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
    X: np.ndarray,
    y: np.ndarray,
    ret: np.ndarray | None,
    price: np.ndarray | None,
    booster,
    cfg: PipelineConfig,
) -> tuple[EvalMetrics, BacktestResult]:
    """Run evaluate + backtest for one split and return results."""
    y_pred = predict(booster, X)
    metrics = evaluate(y, y_pred, threshold=cfg.backtest.entry_threshold)
    bt = backtest(
        y, y_pred, trade_returns=ret,
        entry_prices=price, cost=cfg.backtest.cost,
        entry_threshold=cfg.backtest.entry_threshold,
        fee_rate=cfg.backtest.fee_rate,
        max_position_usd=cfg.backtest.max_position_usd,
        kelly_sizing=cfg.backtest.kelly_sizing,
        kelly_cap=cfg.backtest.kelly_cap,
        initial_bankroll=cfg.backtest.initial_bankroll,
    )
    return metrics, bt


def print_split_eval(
    name: str, metrics: EvalMetrics, bt: BacktestResult, threshold: float
) -> None:
    w = 55
    print(f"\n{'='*w}")
    print(f"  {name.upper()}")
    print(f"{'='*w}")
    if metrics.roc_auc_defined:
        print(f"  ROC AUC  : {metrics.roc_auc:.4f}")
    else:
        print(f"  ROC AUC  : n/a  (only one class in this split)")
    print(f"  Log Loss : {metrics.log_loss:.4f}")
    print(f"  Brier    : {metrics.brier_score:.4f}")
    print(f"  Accuracy : {metrics.accuracy:.4f}  (threshold {threshold:.0%})")
    print(f"  Precision: {metrics.precision:.4f}   Recall: {metrics.recall:.4f}   F1: {metrics.f1:.4f}")

    print(f"\n  --- Backtest (entry >= {threshold:.0%} threshold) ---")
    if bt.total_trades == 0:
        print("  No trades triggered (threshold too high or no qualifying predictions).")
    else:
        print(f"  Trades     : {bt.total_trades}")
        print(f"  Win rate   : {bt.win_rate:.2%}   (price moved the right way)")
        print(f"  Profitable : {bt.profit_rate:.2%}   (after fee)")
        print(f"  Mean return: {bt.mean_trade_return:+.4%} per trade")
        print(f"  Mean cost  : {bt.mean_cost:.4%} per trade")
        print(f"  Total PnL  : ${bt.total_pnl:,.2f}  on ${bt.total_staked:,.2f} staked")
        print(f"  ROI        : {bt.roi:.2%}   (on deployed capital)")
        print(f"  Sharpe     : {bt.sharpe_ratio:.3f}  (per trade)")
        print(f"  Max DD     : ${bt.max_drawdown:,.2f}  ({bt.max_drawdown_pct:.2%})")
        if bt.payoff_model == "binary":
            print("  !! binary payoff model — ROI not meaningful")
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

    # ── Steps 4–6: Features → Labels → Split (single streaming pass) ──
    # Process one market at a time to avoid loading the full dataset.
    print("\n[4/6] Features + labels (streaming, one market at a time)...")
    import pyarrow.parquet as pq

    from pipeline.rowgroups import iter_market_row_groups

    pf = pq.ParquetFile(filled_path)
    n_rg = pf.metadata.num_row_groups

    feature_cols: list[str] | None = None
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ret_parts: list[np.ndarray] = []
    price_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    total_rows = 0
    total_labeled = 0
    total_wins = 0

    for rg_idx, market_df in iter_market_row_groups(pf):
        featured = build_features(market_df, markets_df, cfg.features)
        del market_df

        if feature_cols is None:
            feature_cols = get_feature_columns(featured)

        labeled = add_labels(featured, cfg.label)
        del featured

        total_rows += labeled.height
        n_lab = labeled["win"].is_not_null().sum()
        total_labeled += n_lab
        if n_lab > 0:
            total_wins += (labeled["win"] == 1).sum()

        trainable = labeled.filter(
            pl.col("win").is_not_null()
            & ~pl.col("exclude_from_training").fill_null(False)
        )
        del labeled

        if trainable.height > 0:
            X_parts.append(
                trainable.select(feature_cols).to_numpy().astype(np.float32)
            )
            y_parts.append(trainable["win"].to_numpy().astype(np.float32))
            ret_parts.append(
                trainable["trade_return"].to_numpy().astype(np.float64)
            )
            price_parts.append(
                trainable["entry_token_price"].to_numpy().astype(np.float64)
            )
            time_parts.append(
                trainable["bucket_time"].to_physical().to_numpy()
            )
        del trainable

        if (rg_idx + 1) % 500 == 0 or (rg_idx + 1) == n_rg:
            gc.collect()
            print(f"    process: {rg_idx + 1}/{n_rg} markets", flush=True)

    del pf
    gc.collect()

    win_rate = total_wins / total_labeled if total_labeled > 0 else 0.0
    print(f"  Rows: {total_rows:,}  |  Labeled: {total_labeled:,}  |  Win rate: {win_rate:.3f}")
    print(f"  {len(feature_cols)} feature columns")

    if not X_parts:
        print("ERROR: No trainable rows. Check data / config.")
        sys.exit(1)

    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    ret_all = np.concatenate(ret_parts)
    price_all = np.concatenate(price_parts)
    times_us = np.concatenate(time_parts)
    del X_parts, y_parts, ret_parts, price_parts, time_parts
    gc.collect()
    print(f"  Trainable rows: {len(y_all):,}")

    # ── Step 5: Sort by time ──────────────────────────────────
    print("\n[5/6] Sorting by time for walk-forward split...")
    sort_idx = np.argsort(times_us, kind="mergesort")
    X_all = X_all[sort_idx]
    y_all = y_all[sort_idx]
    ret_all = ret_all[sort_idx]
    price_all = price_all[sort_idx]
    times_us = times_us[sort_idx]
    del sort_idx
    gc.collect()

    # ── Step 6: Walk-forward split (on numpy arrays) ──────────
    # Uses the same time-based purge as pipeline/splitter.walk_forward_split,
    # so evaluation boundaries match training boundaries exactly.
    print("\n[6/6] Walk-forward split...")
    print(f"  Purge between splits: "
          f"{purge_minutes(cfg.split, cfg.label, cfg.bucket.bucket_minutes)} min")
    try:
        tr_sl, va_sl, te_sl = time_split_indices(
            times_us, cfg.split, cfg.label, cfg.bucket.bucket_minutes
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Extract time boundaries before discarding the time array
    def _us_to_dt(us_val: int):
        return datetime.fromtimestamp(us_val / 1_000_000, tz=timezone.utc)

    train_end_time = _us_to_dt(times_us[tr_sl.stop - 1])
    val_start_time = _us_to_dt(times_us[va_sl.start])
    val_end_time   = _us_to_dt(times_us[va_sl.stop - 1])
    test_start_time = _us_to_dt(times_us[te_sl.start])
    del times_us
    gc.collect()

    split = SplitResult(
        train_X=X_all[tr_sl],
        train_y=y_all[tr_sl],
        val_X=X_all[va_sl],
        val_y=y_all[va_sl],
        test_X=X_all[te_sl],
        test_y=y_all[te_sl],
        feature_names=feature_cols,
        train_ret=ret_all[tr_sl],
        val_ret=ret_all[va_sl],
        test_ret=ret_all[te_sl],
        train_price=price_all[tr_sl],
        val_price=price_all[va_sl],
        test_price=price_all[te_sl],
        train_end=train_end_time,
        val_start=val_start_time,
        val_end=val_end_time,
        test_start=test_start_time,
    )
    del X_all, y_all, ret_all, price_all
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

    # Detect Column_N pattern: model trained without feature names (old code)
    import re
    _col_n_pattern = re.compile(r"^Column_\d+$")
    all_generic = all(_col_n_pattern.match(f) for f in model_features)

    if all_generic and len(model_features) == len(split.feature_names):
        print(f"  FIX: Model has generic Column_N names — mapping by position to data features.")
        # Column_N maps to split.feature_names[N] positionally
        train_X = split.train_X
        val_X   = split.val_X
        test_X  = split.test_X
        # Use data feature names for reporting
        model_features = list(split.feature_names)
    elif all_generic:
        print(f"  WARNING: Model has generic Column_N names and feature count differs "
              f"({len(model_features)} vs {len(split.feature_names)}). Cannot map by position — results will be wrong.")
        train_X = split.train_X[:, :len(model_features)] if split.train_X.shape[1] >= len(model_features) else split.train_X
        val_X   = split.val_X[:, :len(model_features)] if split.val_X.shape[1] >= len(model_features) else split.val_X
        test_X  = split.test_X[:, :len(model_features)] if split.test_X.shape[1] >= len(model_features) else split.test_X
    else:
        # Normal case: match by name
        missing_in_data = set(model_features) - set(split.feature_names)
        extra_in_data   = set(split.feature_names) - set(model_features)
        if missing_in_data:
            print(f"  WARNING: {len(missing_in_data)} model features not in data "
                  f"(will be NaN): {sorted(missing_in_data)[:5]}{'...' if len(missing_in_data) > 5 else ''}")
        if extra_in_data:
            print(f"  INFO: {len(extra_in_data)} data features not used by model "
                  f"(ignored): {sorted(extra_in_data)[:5]}{'...' if len(extra_in_data) > 5 else ''}")

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

    _split_log: dict[str, dict] = {}
    _results: dict[str, tuple] = {}
    for split_name, X, y, ret, price, _key in [
        ("TRAIN (in-sample — expect high)",   train_X, split.train_y,
         split.train_ret, split.train_price, "train"),
        ("VALIDATION (out-of-sample)",         val_X,   split.val_y,
         split.val_ret,   split.val_price,   "val"),
        ("TEST (final holdout — trust this)", test_X,  split.test_y,
         split.test_ret,  split.test_price,  "test"),
    ]:
        m, bt = _evaluate_split(X, y, ret, price, booster, cfg)
        print_split_eval(split_name, m, bt, cfg.backtest.entry_threshold)
        _results[_key] = (m, bt)
        _split_log[_key] = {
            "auc":       round(m.roc_auc, 4),
            "log_loss":  round(m.log_loss, 4),
            "brier":     round(m.brier_score, 4),
            "roi":       round(bt.roi, 6),
            "roi_bankroll": round(bt.roi_bankroll, 4),
            "staked":    round(bt.total_staked, 2),
            "sharpe":    round(bt.sharpe_ratio, 3),
            "trades":    bt.total_trades,
            "win_rate":  round(bt.win_rate, 4),
            "profit_rate": round(bt.profit_rate, 4),
            "mean_ret":  round(bt.mean_trade_return, 6),
            "mean_cost": round(bt.mean_cost, 6),
            "skipped":   bt.skipped_untradeable,
        }

    # Full evaluation printout for test set (includes equity curve + calibration).
    # Reuses the results computed above instead of predicting a second time.
    print("\n\n  === FULL TEST SET REPORT ===")
    test_metrics, test_bt = _results["test"]
    print_evaluation(test_metrics, test_bt)

    # ── Feature importance ─────────────────────────────────────
    fi = feature_importance(booster, model_features, top_n=20)
    print("\n  Top 20 Features (by gain):")
    max_imp = fi[0][1] if fi and fi[0][1] > 0 else 1
    for i, (name, imp) in enumerate(fi, 1):
        bar = "█" * int(imp / max_imp * 30)
        print(f"    {i:>2}. {name:<35}  {imp:>10.0f}  {bar}")

    # ── Results log ────────────────────────────────────────────
    if args.log_file:
        _log_entry: dict = {
            "type":        "model",
            "ts":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model":       args.model,
            "trades":      args.trades,
            "threshold":   args.entry_threshold,
            "fwd_buckets": args.forward_window,
        }
        for _k, _d in _split_log.items():
            for _m, _v in _d.items():
                _log_entry[f"{_k}_{_m}"] = _v
        append_to_log(args.log_file, _log_entry)
        print(f"\n  Results appended to: {args.log_file}")
        print_log_history(args.log_file)

    # ── Cleanup ────────────────────────────────────────────────
    if not args.keep_intermediates:
        import shutil
        shutil.rmtree(_tmp, ignore_errors=True)
    else:
        print(f"\n  Intermediate files kept in: {_tmp}/")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
