#!/usr/bin/env python3
"""
benchmark_strategies.py — Compare classic Polymarket trading strategies on
historical data.  All strategies are evaluated on the same TEST split so
results are directly comparable to evaluate_model.py.

Strategies
----------
  baseline    Always bet the dominant bucket direction (no selectivity)
  random      Random bet (reference: expected AUC ≈ 0.50, ROI ≈ -fee)
  momentum    Follow the recent price trend (continuation / herding)
  reversion   Bet against overextended price moves (mean reversion)
  volume      Follow smart money — unusually large volume spikes
  closing     Near market close, prices converge to resolution (closing bias)
  contrarian  Bet against the crowd when BOTH yes_ratio AND price are extreme

All strategies that bet AGAINST the dominant bucket direction (reversion,
contrarian) have their ground-truth labels flipped so that their ROC AUC and
backtest PnL always measure "how well does this strategy win?", not whether
the dominant bucket side wins.  This makes all rows of the summary table
directly comparable.

Usage
-----
    python benchmark_strategies.py \\
        --trades data/trades.parquet \\
        --markets data/markets.parquet

    # Explicit formats (default: parquet):
    python benchmark_strategies.py \\
        --trades data/trades.parquet --trades-format parquet \\
        --markets data/markets.parquet --markets-format parquet

    # Lower entry threshold (useful when strategies produce weaker signals):
    python benchmark_strategies.py ... --entry-threshold 0.55

    # Keep intermediate Parquet files for debugging:
    python benchmark_strategies.py ... --keep-intermediates
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
from pipeline.splitter import walk_forward_split, print_split_info
from pipeline.evaluation import evaluate, backtest, EvalMetrics, BacktestResult
from pipeline.results_logger import append_to_log, print_log_history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark classic trading strategies vs ML model on Polymarket data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--trades",          default="data/trades.parquet")
    p.add_argument("--markets",         default="data/markets.parquet")
    p.add_argument("--trades-format",   default="parquet",
                   choices=["csv", "parquet", "sqlite"])
    p.add_argument("--markets-format",  default="parquet",
                   choices=["csv", "parquet", "sqlite"])
    p.add_argument("--entry-threshold", type=float, default=0.6,
                   help="P(win) threshold for backtest entry (default: 0.6)")
    p.add_argument("--bucket-minutes",  type=int,   default=5)
    p.add_argument("--forward-window",  type=int,   default=6,
                   help="Label forward window in buckets (default: 6 = 30 min)")
    p.add_argument("--seed",            type=int,   default=42,
                   help="Random seed for 'random' baseline strategy")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Keep temp Parquet files after run")
    p.add_argument("--log-file", default="results_log.jsonl", metavar="PATH",
                   help="JSONL file to append results to for historical tracking "
                        "(default: results_log.jsonl). Pass '' to disable.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Strategy result container
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    description: str
    bet_direction: str       # "follow" = bets WITH dominant bucket; "against" = opposite
    coverage: float          # fraction of test rows where strategy fires
    n_rows: int              # total test rows evaluated
    metrics: Optional[EvalMetrics] = None
    bt: Optional[BacktestResult] = None
    error: str = ""


# ---------------------------------------------------------------------------
# Helper: run evaluate + backtest, handle edge cases
# ---------------------------------------------------------------------------

def _run_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[EvalMetrics, BacktestResult]:
    """Evaluate and run backtest on a (y_true, y_pred) pair."""

    # Remove NaN / inf
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 1e-7, 1.0 - 1e-7)

    if len(y_true) < 10:
        raise ValueError("Too few valid rows for evaluation")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Only one class present in y_true (all wins or all losses)")

    metrics = evaluate(y_true, y_pred, threshold=cfg.backtest.entry_threshold)
    bt = backtest(
        y_true,
        y_pred,
        entry_threshold=cfg.backtest.entry_threshold,
        fee_rate=cfg.backtest.fee_rate,
        max_position_usd=cfg.backtest.max_position_usd,
        kelly_sizing=cfg.backtest.kelly_sizing,
        kelly_cap=cfg.backtest.kelly_cap,
        initial_bankroll=cfg.backtest.initial_bankroll,
    )
    return metrics, bt


# ---------------------------------------------------------------------------
# Strategies
# Each returns (y_true, y_pred, coverage, description)
#   y_true   : 0/1 numpy array — the ground truth FOR THIS STRATEGY'S BET
#              (inverted for "against" strategies so that 1 = strategy wins)
#   y_pred   : float [0, 1] — the strategy's confidence (higher = more confident)
#   coverage : fraction of rows where the strategy fires a signal
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _col(df: pl.DataFrame, name: str) -> Optional[np.ndarray]:
    """Return column as numpy array or None if column not present."""
    if name not in df.columns:
        return None
    return df[name].to_numpy()


# -- 1. Baseline -------------------------------------------------------------

def strat_baseline(df: pl.DataFrame) -> tuple:
    """
    Always bet the dominant bucket direction with full confidence.
    No selectivity at all — establishes the raw win rate.
    """
    win = df["win"].to_numpy().astype(np.float32)
    score = np.full(len(df), 0.95, dtype=np.float32)
    return win, score, 1.0, "Always bet dominant direction (no filter)"


# -- 2. Random ---------------------------------------------------------------

def strat_random(df: pl.DataFrame, seed: int = 42) -> tuple:
    """
    Random predictions — lower bound reference.
    Expected AUC ≈ 0.50, ROI ≈ -2 × fee_rate.
    """
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.0, 1.0, len(df)).astype(np.float32)
    win = df["win"].to_numpy().astype(np.float32)
    return win, score, 1.0, "Uniform random predictions (reference)"


# -- 3. Momentum -------------------------------------------------------------

def strat_momentum(df: pl.DataFrame) -> tuple:
    """
    Follow the recent price trend over the last 3 buckets (15 min).
    Hypothesis: short-term herding → price continues in the same direction.

    Fires when |directional momentum| > 0.005 (0.5 cent move in 15 min).
    Bets WITH the dominant bucket direction.
    """
    pc = _col(df, "price_change_lag3")
    if pc is None:
        pc = _col(df, "price_change_lag2")
    if pc is None:
        pc = _col(df, "price_change_lag1")
    if pc is None:
        return np.array([]), np.array([]), 0.0, "price_change_lag* not found"

    yes_bucket = df["yes_ratio"].to_numpy() > 0.5
    win = df["win"].to_numpy().astype(np.float32)

    # Directional momentum: positive = trend going in the winning direction
    # YES bucket wins when price rises  → positive change is good
    # NO  bucket wins when price falls  → negative change is good (flip sign)
    directional = np.where(yes_bucket, pc, -pc)
    directional = np.where(np.isfinite(directional), directional, 0.0)

    # Score: sigmoid of scaled momentum.  ±0.03 price move ≈ ±sigmoid(0.6) ≈ 0.645
    score = _sigmoid(directional * 20).astype(np.float32)

    fires = np.abs(directional) > 0.005
    score[~fires] = 0.5

    coverage = fires.mean()
    return win, score, float(coverage), "Follow 15-min price trend (continuation)"


# -- 4. Mean Reversion -------------------------------------------------------

def strat_reversion(df: pl.DataFrame, threshold: float = 0.04) -> tuple:
    """
    Bet AGAINST overextended price moves (revert to 24-bucket rolling mean).
    Hypothesis: extreme short-term price moves are noise and will revert.

    Fires when price_vs_ma24 is extreme in the direction opposite to winning.
    Bets AGAINST the dominant bucket direction → y_true is flipped.

    threshold: relative deviation from MA required to fire (default 4 %)
    """
    dev = _col(df, "price_vs_ma24")
    if dev is None:
        dev = _col(df, "price_vs_ma12")
    if dev is None:
        return np.array([]), np.array([]), 0.0, "price_vs_ma* not found"

    yes_bucket = df["yes_ratio"].to_numpy() > 0.5
    win = df["win"].to_numpy().astype(np.float32)
    dev = np.where(np.isfinite(dev), dev, 0.0)

    # Reversion fires when price has moved TOO FAR in the winning direction:
    #   YES bucket + price ABOVE MA → price will fall → YES bet LOSES
    #   NO  bucket + price BELOW MA → price will rise → NO  bet LOSES
    fires_yes = yes_bucket  & (dev >  threshold)
    fires_no  = ~yes_bucket & (dev < -threshold)
    fires = fires_yes | fires_no

    # Score: how extreme is the deviation? Large = strong reversion expected.
    # Normalise: 20 % deviation → score 1.0
    magnitude = np.abs(dev)
    score = np.clip(magnitude / 0.20, 0.0, 1.0)
    score = (0.5 + score * 0.5).astype(np.float32)  # map to [0.5, 1.0]
    score[~fires] = 0.5

    # Reversion WINS when dominant side LOSES → flip labels
    y_true_mr = 1.0 - win

    coverage = fires.mean()
    return y_true_mr, score, float(coverage), "Bet against 15-min overextension (4 % threshold)"


# -- 5. Volume Spike ---------------------------------------------------------

def strat_volume(df: pl.DataFrame, spike_x: float = 3.0) -> tuple:
    """
    Follow smart money when current bucket volume is unusually high.
    Hypothesis: informed traders are acting → follow their direction.

    Fires when current USD volume is ≥ spike_x × rolling 24-bucket average.
    Bets WITH the dominant bucket direction.

    spike_x: multiplier required to classify a bucket as a spike (default 3×)
    """
    total_usd = _col(df, "total_usd")
    vol_sum24 = _col(df, "volume_sum24")
    if total_usd is None or vol_sum24 is None:
        return np.array([]), np.array([]), 0.0, "total_usd / volume_sum24 not found"

    win = df["win"].to_numpy().astype(np.float32)

    avg24 = vol_sum24 / 24.0
    avg24 = np.where(avg24 > 0.0, avg24, np.nan)
    ratio = np.where(np.isfinite(avg24), total_usd / avg24, np.nan)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)

    fires = ratio >= spike_x

    # Score: higher ratio → more confidence.  Capped at 10× spike.
    score = np.clip((ratio - 1.0) / (spike_x * 3.0 - 1.0), 0.0, 1.0)
    score = (0.5 + score * 0.5).astype(np.float32)
    score[~fires] = 0.5

    coverage = fires.mean()
    return win, score, float(coverage), f"Follow volume spikes ≥ {spike_x}× 2-hour average"


# -- 6. Closing Bias ---------------------------------------------------------

def strat_closing(
    df: pl.DataFrame,
    max_days: float = 14.0,
    min_directional: float = 0.15,
) -> tuple:
    """
    Near market close, prices converge toward the final resolution value.
    Hypothesis: as the close approaches, the dominant price direction
    becomes more predictive of short-term price moves.

    Fires when days_to_close < max_days AND price is sufficiently directional.
    Bets WITH the dominant bucket direction.

    max_days        : maximum days-to-close to activate (default 14)
    min_directional : minimum |price - 0.5| to fire (default 0.15)
    """
    days = _col(df, "days_to_close")
    if days is None:
        return np.array([]), np.array([]), 0.0, "days_to_close not found (no close_time in markets)"

    price = df["mean_price"].to_numpy()
    yes_bucket = df["yes_ratio"].to_numpy() > 0.5
    win = df["win"].to_numpy().astype(np.float32)
    days = np.where(np.isfinite(days), days, 999.0)

    # Urgency: 0 when far from close, 1 when close tomorrow
    urgency = np.clip(1.0 - days / max_days, 0.0, 1.0)

    # Directional strength: how firmly does the current price point toward a side?
    #   YES bucket: price > 0.5 means leaning YES  → directional ∈ [0, 1]
    #   NO  bucket: price < 0.5 means leaning NO   → directional ∈ [0, 1]
    dir_yes = np.clip(price - 0.5, 0.0, 0.5) / 0.5
    dir_no  = np.clip(0.5 - price, 0.0, 0.5) / 0.5
    directional = np.where(yes_bucket, dir_yes, dir_no)

    fires = (
        (days >= 0.0) & (days < max_days)
        & (directional > min_directional)
        & (urgency > 0.05)
    )

    # Score: combination of urgency and directional confidence
    score = 0.5 + urgency * directional * 0.45
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    score[~fires] = 0.5

    coverage = fires.mean()
    return win, score, float(coverage), f"Near-close convergence (< {max_days} days, p > 0.5+{min_directional})"


# -- 7. Contrarian -----------------------------------------------------------

def strat_contrarian(
    df: pl.DataFrame,
    yr_thr: float = 0.65,
    price_thr: float = 0.65,
) -> tuple:
    """
    Mathematically bet against the crowd when BOTH signals agree:
      - yes_ratio is extreme (crowd consensus on direction)
      - mean_price is extreme in the same direction (market pricing reflects it)

    Hypothesis: when the crowd has already pushed price AND volume strongly
    to one side, the true probability is unlikely to be as extreme — expect
    a correction toward 0.5 in the next 30 minutes.

    Fires when BOTH:
      yes_ratio > yr_thr  AND  mean_price > price_thr   (crowd overbets YES)
      yes_ratio < 1-yr_thr AND mean_price < 1-price_thr (crowd overbets NO)

    Bets AGAINST the dominant bucket direction → y_true is flipped.

    yr_thr    : yes_ratio threshold (default 0.65)
    price_thr : price threshold (default 0.65)
    """
    yr = df["yes_ratio"].to_numpy()
    price = df["mean_price"].to_numpy()
    win = df["win"].to_numpy().astype(np.float32)

    crowd_yes = (yr > yr_thr)       & (price > price_thr)
    crowd_no  = (yr < 1 - yr_thr)   & (price < 1 - price_thr)
    fires = crowd_yes | crowd_no

    # Signal strength: how extreme are both indicators?
    yr_strength    = np.abs(yr - 0.5) * 2.0     # [0, 1]
    price_strength = np.abs(price - 0.5) * 2.0  # [0, 1]
    # Both must be extreme → use the minimum (weakest link)
    combined = np.minimum(yr_strength, price_strength)

    score = (0.5 + combined * 0.5).astype(np.float32)
    score[~fires] = 0.5

    # Contrarian wins when dominant side LOSES → flip ground truth
    y_true_c = 1.0 - win

    coverage = fires.mean()
    return (
        y_true_c,
        score,
        float(coverage),
        f"Against crowd when yes_ratio >{yr_thr:.0%} AND price >{price_thr:.0%} (or symmetric NO)",
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_W = 60  # column width for section headers


def _banner(title: str) -> None:
    print(f"\n{'=' * _W}")
    print(f"  {title}")
    print(f"{'=' * _W}")


def _print_comparison_table(results: list[StrategyResult], threshold: float) -> None:
    """Print a compact comparison table of all strategies."""

    print(f"\n  Entry threshold : {threshold:.0%}")
    print(f"  (ROC AUC & backtest are on the TEST split only)\n")

    hdr = (
        f"  {'Strategy':<16}"
        f"{'Dir':<8}"
        f"{'Cover':>7}"
        f"{'AUC':>8}"
        f"{'Brier':>7}"
        f"{'WinRate':>8}"
        f"{'ROI':>8}"
        f"{'Sharpe':>7}"
        f"{'Trades':>7}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for r in results:
        if r.error:
            print(f"  {r.name:<16}  ERROR: {r.error}")
            continue
        m  = r.metrics
        bt = r.bt
        dir_tag = "follow" if r.bet_direction == "follow" else "AGAINST"
        print(
            f"  {r.name:<16}"
            f"{dir_tag:<8}"
            f"{r.coverage:>6.1%} "
            f"{m.roc_auc:>7.4f}"
            f"{m.brier_score:>7.4f}"
            f"{bt.win_rate:>7.1%} "
            f"{bt.roi:>+8.1%}"
            f"{bt.sharpe_ratio:>7.2f}"
            f"{bt.total_trades:>7}"
        )

    print(sep)
    print(
        f"\n  Dir=follow  → strategy bets WITH the dominant bucket side   (win = original label)\n"
        f"  Dir=AGAINST → strategy bets AGAINST the dominant side        "
        f"(win = flipped label = crowd is wrong)\n"
        f"  Cover       → % of test rows where strategy fires a signal\n"
        f"  AUC/Brier   → ML metrics on the strategy's own win criterion\n"
        f"  Trades      → trades executed in backtest (strategy-entry ≥ {threshold:.0%})"
    )


def _print_strategy_detail(r: StrategyResult, threshold: float) -> None:
    """Print detailed block for one strategy (mirrors evaluate_model.py format)."""

    if r.error:
        return

    m  = r.metrics
    bt = r.bt
    w  = 55

    print(f"\n{'='*w}")
    print(f"  {r.name.upper()}  —  {r.description}")
    print(f"  Direction : {r.bet_direction}")
    print(f"  Coverage  : {r.coverage:.1%}  ({int(r.coverage * r.n_rows):,} / {r.n_rows:,} rows)")
    print(f"{'='*w}")

    print(f"  ROC AUC  : {m.roc_auc:.4f}")
    print(f"  Log Loss : {m.log_loss:.4f}")
    print(f"  Brier    : {m.brier_score:.4f}")
    print(f"  Accuracy : {m.accuracy:.4f}   Precision: {m.precision:.4f}")
    print(f"  Recall   : {m.recall:.4f}    F1:        {m.f1:.4f}")

    print(f"\n  --- Backtest (entry P >= {threshold:.0%}) ---")
    if bt.total_trades == 0:
        print(f"  No trades triggered — try lowering --entry-threshold")
    else:
        print(f"  Trades   : {bt.total_trades}")
        print(f"  Win rate : {bt.win_rate:.2%}")
        print(f"  Total PnL: ${bt.total_pnl:,.2f}")
        print(f"  ROI      : {bt.roi:+.2%}")
        print(f"  Sharpe   : {bt.sharpe_ratio:.2f}")
        print(f"  Max DD   : ${bt.max_drawdown:,.2f}  ({bt.max_drawdown_pct:.2%})")

    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = PipelineConfig()
    cfg.data.trades_path    = args.trades
    cfg.data.markets_path   = args.markets
    cfg.data.trades_format  = args.trades_format
    cfg.data.markets_format = args.markets_format
    cfg.bucket.bucket_minutes            = args.bucket_minutes
    cfg.label.forward_window_buckets     = args.forward_window
    cfg.backtest.entry_threshold         = args.entry_threshold

    _banner("STRATEGY BENCHMARK  (no model required)")
    print(f"  Trades   : {args.trades}  [{args.trades_format}]")
    print(f"  Markets  : {args.markets}  [{args.markets_format}]")
    print(f"  Threshold: {args.entry_threshold}")
    print(f"  Window   : {args.forward_window} buckets × {args.bucket_minutes} min = "
          f"{args.forward_window * args.bucket_minutes} min label horizon")
    t0 = time.time()

    # -- Temp files ----------------------------------------------------------
    _tmp = Path("_bench_tmp")
    _tmp.mkdir(exist_ok=True)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    trades_lf  = load_trades(cfg.data)
    markets_lf = load_markets(cfg.data)
    markets_df = markets_lf.collect()
    print(f"  Markets: {markets_df.height} rows")

    # ── Step 2: Bucketing ────────────────────────────────────────────────────
    print(f"\n[2/6] Aggregating into {cfg.bucket.bucket_minutes}-min buckets...")
    bucketed_path = aggregate_trades(
        trades_lf, cfg.bucket, output_path=str(_tmp / "bucketed.parquet")
    )
    bucketed = pl.scan_parquet(bucketed_path).collect(engine="streaming")
    n_markets = bucketed["market_id"].n_unique()
    print(f"  {bucketed.height:,} rows across {n_markets} markets")

    # ── Step 3: Gap handling ─────────────────────────────────────────────────
    print("\n[3/6] Gap handling...")
    gap_summary = detect_gaps(bucketed, cfg.bucket, cfg.gap)
    print(
        f"  Mean gap ratio: {gap_summary['gap_ratio'].mean():.2%}  |  "
        f"Markets >50% gaps: {gap_summary.filter(gap_summary['gap_ratio'] > 0.5).height}"
    )
    filled_path = fill_buckets(
        bucketed, cfg.bucket, cfg.gap, output_path=str(_tmp / "filled.parquet")
    )
    del bucketed
    gc.collect()
    filled_path = detect_consecutive_gaps(
        filled_path, cfg.gap, output_path=str(_tmp / "filled_gaps.parquet")
    )
    filled_path = apply_gap_exclusions(
        filled_path, cfg.gap, output_path=str(_tmp / "filled_final.parquet")
    )
    stats_lf   = pl.scan_parquet(filled_path)
    n_total    = stats_lf.select(pl.len()).collect()[0, 0]
    n_excluded = stats_lf.filter(pl.col("exclude_from_training")).select(pl.len()).collect()[0, 0]
    print(f"  Rows: {n_total:,}  |  Excluded: {n_excluded:,}")

    # ── Step 4: Features ─────────────────────────────────────────────────────
    print("\n[4/6] Engineering features (streaming)...")
    features_path = build_features_streaming(
        filled_path, markets_df, cfg.features,
        output_path=str(_tmp / "features.parquet")
    )
    _schema       = pl.scan_parquet(features_path).limit(0).collect()
    feature_cols  = get_feature_columns(_schema)
    del _schema
    print(f"  {len(feature_cols)} feature columns")

    # ── Step 5: Labels ───────────────────────────────────────────────────────
    print(f"\n[5/6] Generating labels (forward {cfg.label.forward_window_buckets} buckets = "
          f"{cfg.label.forward_window_buckets * cfg.bucket.bucket_minutes} min)...")
    labeled_path = add_labels_streaming(
        features_path, cfg.label, output_path=str(_tmp / "labeled.parquet")
    )
    lstats = label_stats_lazy(labeled_path)
    print(f"  Labeled: {lstats['labeled']:,}  |  Win rate: {lstats['win_rate']:.3f}")

    # ── Step 6: Walk-forward split ───────────────────────────────────────────
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

    # -- Extract test DataFrame (all columns, same rows as split.test_y) -----
    # Re-read labeled parquet and filter to test period by timestamp.
    labeled_all = (
        pl.scan_parquet(labeled_path)
        .filter(
            pl.col("win").is_not_null()
            & ~pl.col("exclude_from_training").fill_null(False)
        )
        .collect()
        .sort("bucket_time")
    )
    test_df = labeled_all.filter(pl.col("bucket_time") >= split.test_start)
    del labeled_all
    gc.collect()

    # Sanity-check: row count should match split.test_y
    if abs(test_df.height - len(split.test_y)) > max(5, int(0.01 * len(split.test_y))):
        print(
            f"  WARNING: test_df rows ({test_df.height:,}) deviate from "
            f"split.test_y ({len(split.test_y):,}) — using test_df"
        )
    n_test = test_df.height
    print(f"\n  Test rows used for strategy evaluation: {n_test:,}")

    # ── Run all strategies ───────────────────────────────────────────────────
    STRATEGIES = [
        ("baseline",    "follow",  lambda df: strat_baseline(df)),
        ("random",      "follow",  lambda df: strat_random(df, seed=args.seed)),
        ("momentum",    "follow",  lambda df: strat_momentum(df)),
        ("reversion",   "against", lambda df: strat_reversion(df, threshold=0.04)),
        ("volume",      "follow",  lambda df: strat_volume(df, spike_x=3.0)),
        ("closing",     "follow",  lambda df: strat_closing(df, max_days=14.0)),
        ("contrarian",  "against", lambda df: strat_contrarian(df, yr_thr=0.65, price_thr=0.65)),
    ]

    results: list[StrategyResult] = []

    print(f"\n  Running {len(STRATEGIES)} strategies on {n_test:,} test rows...")

    for name, direction, fn in STRATEGIES:
        try:
            y_true, y_pred, coverage, description = fn(test_df)

            if len(y_true) == 0:
                results.append(StrategyResult(
                    name=name, description=description,
                    bet_direction=direction, coverage=0.0, n_rows=n_test,
                    error="no signal / feature not available",
                ))
                continue

            metrics, bt = _run_eval(y_true, y_pred, cfg)
            results.append(StrategyResult(
                name=name, description=description,
                bet_direction=direction,
                coverage=coverage, n_rows=n_test,
                metrics=metrics, bt=bt,
            ))
            print(f"  [{name:<12}]  coverage={coverage:.1%}  "
                  f"AUC={metrics.roc_auc:.4f}  "
                  f"ROI={bt.roi:+.1%}  trades={bt.total_trades}")

        except Exception as exc:
            results.append(StrategyResult(
                name=name, description="",
                bet_direction=direction, coverage=0.0, n_rows=n_test,
                error=str(exc),
            ))
            print(f"  [{name:<12}]  ERROR: {exc}")

    # ── Summary table ────────────────────────────────────────────────────────
    _banner("COMPARISON TABLE  (TEST split)")
    _print_comparison_table(results, args.entry_threshold)

    # ── Detailed per-strategy blocks ─────────────────────────────────────────
    _banner("DETAILED RESULTS PER STRATEGY")
    for r in results:
        _print_strategy_detail(r, args.entry_threshold)

    # -- Interpretation guide ------------------------------------------------
    _banner("HOW TO READ THESE RESULTS")
    print(
        "\n  IMPORTANT — strategy direction matters:\n"
        "\n  follow   strategies: P(win) = P(dominant bucket side wins)"
        "\n                       ROC AUC > 0.5 = better than random AT selecting which\n"
        "                       dominant-side bets actually win"
        "\n  AGAINST  strategies: y_true is flipped → P(win) = P(the CROWD IS WRONG)"
        "\n                       ROC AUC > 0.5 = better than random AT detecting\n"
        "                       when the crowd has overbought/oversold"
        "\n\n  Direct comparison with evaluate_model.py:"
        "\n    The 'baseline' strategy shows the unconditional win rate (what you get"
        "\n    with zero selectivity).  Any strategy above 'baseline' in Win Rate"
        "\n    is adding real value through its filtering signal."
        "\n    The ML model's ROI/Sharpe from evaluate_model.py can be placed mentally"
        "\n    above the 'random' row for comparison."
        "\n\n  AGAINST strategies are profitable if ROI > 0 — meaning the market"
        "\n    systematically overreacts and contrarian bets recover money."
    )

    # -- Results log ---------------------------------------------------------
    if args.log_file:
        _log_entry: dict = {
            "type":        "bench",
            "ts":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trades":      args.trades,
            "threshold":   args.entry_threshold,
            "fwd_buckets": args.forward_window,
            "n_test":      n_test,
        }
        for r in results:
            if not r.error and r.bt is not None:
                _log_entry[f"{r.name}_auc"]    = round(r.metrics.roc_auc, 4)
                _log_entry[f"{r.name}_roi"]    = round(r.bt.roi, 4)
                _log_entry[f"{r.name}_sharpe"] = round(r.bt.sharpe_ratio, 3)
                _log_entry[f"{r.name}_trades"] = r.bt.total_trades
            else:
                _log_entry[f"{r.name}_roi"] = None
        append_to_log(args.log_file, _log_entry)
        print(f"\n  Results appended to: {args.log_file}")
        print_log_history(args.log_file)

    # -- Cleanup -------------------------------------------------------------
    if not args.keep_intermediates:
        import shutil
        shutil.rmtree(_tmp, ignore_errors=True)
    else:
        print(f"\n  Intermediate files kept in: {_tmp}/")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
