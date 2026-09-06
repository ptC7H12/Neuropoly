"""
End-to-end test with synthetic data.
Validates the entire pipeline works from raw trades to predictions.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import polars as pl

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig
from pipeline.aggregation import aggregate_trades
from pipeline.gap_handler import fill_buckets, detect_gaps, detect_consecutive_gaps, apply_gap_exclusions
from pipeline.features import build_features, get_feature_columns
from pipeline.labeling import add_labels, label_stats
from pipeline.splitter import walk_forward_split, print_split_info


def generate_synthetic_trades(
    n_markets: int = 5,
    trades_per_market: int = 5000,
    start_date: datetime = datetime(2024, 1, 1),
    seed: int = 42,
) -> pl.LazyFrame:
    """Generate synthetic trade data."""

    rng = np.random.default_rng(seed)
    rows = []

    for market_id in range(1, n_markets + 1):
        price = 0.5  # Start at 50%
        ts = start_date

        for _ in range(trades_per_market):
            # Random walk price
            price += rng.normal(0, 0.005)
            price = max(0.01, min(0.99, price))

            usd = rng.exponential(50) + 10
            tokens = usd / price

            rows.append({
                "timestamp": ts,
                "market_id": market_id,
                "is_yes": int(rng.random() > 0.4),
                "price": round(price, 4),
                "usd_amount": round(usd, 2),
                "token_amount": round(tokens, 2),
            })

            # Advance time randomly (1-30 minutes)
            ts += timedelta(minutes=int(rng.integers(1, 30)))

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    df = df.with_columns(pl.col("is_yes").cast(pl.Int8))

    # Add is_buy column
    df = df.with_columns(
        pl.Series("is_buy", rng.integers(0, 2, len(rows)).astype(np.int8))
    )

    return df.sort("timestamp").lazy()


def generate_synthetic_markets(n_markets: int = 5) -> pl.DataFrame:
    """Generate synthetic market snapshot data."""

    rows = []
    for market_id in range(1, n_markets + 1):
        yes_p = round(np.random.uniform(0.1, 0.9), 4)
        rows.append({
            "market_id": market_id,
            "question": f"Test market {market_id}?",
            "volume": round(np.random.uniform(10000, 1000000), 2),
            "liquidity": round(np.random.uniform(1000, 50000), 2),
            "yes_price": yes_p,
            "no_price": round(1 - yes_p, 4),
            "close_time": datetime(2025, 12, 31),
        })

    return pl.DataFrame(rows).with_columns(
        pl.col("close_time").cast(pl.Datetime("us"))
    )


def test_full_pipeline():
    """Run the entire pipeline on synthetic data."""
    try:
        _run_full_pipeline()
    finally:
        _cleanup()


def _run_full_pipeline():
    print("=" * 60)
    print("  E2E TEST: Synthetic Data Pipeline")
    print("=" * 60)

    cfg = PipelineConfig()
    # Use smaller gap period for test (no data gap in synthetic data)
    cfg.gap.gap_start = datetime(2099, 1, 1)
    cfg.gap.gap_end = datetime(2099, 2, 1)
    # Smaller model for speed
    cfg.model.n_estimators = 100
    cfg.model.early_stopping_rounds = 10
    cfg.monitor.rich_dashboard = False
    cfg.monitor.log_file = None
    cfg.monitor.log_interval = 20

    # Generate data
    print("\n[1] Generating synthetic data...")
    trades_lf = generate_synthetic_trades(n_markets=5, trades_per_market=5000)
    markets_df = generate_synthetic_markets(n_markets=5)
    print(f"  Trades: {trades_lf.collect().height} rows")
    print(f"  Markets: {markets_df.height} rows")

    # Aggregate
    print("\n[2] Aggregating into buckets...")
    bucketed_path = aggregate_trades(trades_lf, cfg.bucket)
    bucketed = pl.scan_parquet(bucketed_path).collect(engine="streaming")
    print(f"  Bucketed: {bucketed.height} rows")
    assert bucketed.height > 0, "Bucketed should have rows"

    # Gap handling
    print("\n[3] Handling gaps...")
    gap_summary = detect_gaps(bucketed, cfg.bucket, cfg.gap)
    print(f"  Gap summary: {gap_summary.height} markets")

    filled_path = fill_buckets(bucketed, cfg.bucket, cfg.gap,
                               output_path="test_filled.parquet")
    filled_path = detect_consecutive_gaps(filled_path, cfg.gap,
                                          output_path="test_filled_gaps.parquet")
    filled_path = apply_gap_exclusions(filled_path, cfg.gap,
                                       output_path="test_filled_final.parquet")
    filled = pl.scan_parquet(filled_path).collect()
    print(f"  Filled: {filled.height} rows")
    assert filled.height >= bucketed.height, "Filled should have at least as many rows"

    # Features
    print("\n[4] Building features...")
    featured = build_features(filled, markets_df, cfg.features)
    feature_cols = get_feature_columns(featured)
    print(f"  Features: {len(feature_cols)} columns")
    assert len(feature_cols) > 20, f"Expected >20 features, got {len(feature_cols)}"

    # Labels
    print("\n[5] Generating labels...")
    labeled = add_labels(featured, cfg.label)
    stats = label_stats(labeled)
    print(f"  Stats: {stats}")
    assert stats["labeled"] > 0, "Should have labeled rows"
    assert 0.2 < stats["win_rate"] < 0.8, f"Win rate {stats['win_rate']} seems extreme"

    # Split
    print("\n[6] Walk-forward split...")
    split = walk_forward_split(
        labeled, feature_cols, cfg.split, cfg.label, cfg.bucket.bucket_minutes
    )
    print_split_info(split)
    assert split.train_X.shape[0] > 0, "Train set empty"
    assert split.val_X.shape[0] > 0, "Val set empty"
    assert split.test_X.shape[0] > 0, "Test set empty"

    # Train
    print("\n[7] Training LightGBM...")
    from pipeline.model import train_model, predict

    booster, monitor = train_model(split, cfg.model, cfg.monitor, save_path=None)

    # Predict
    print("\n[8] Evaluating...")
    y_pred = predict(booster, split.test_X)
    assert len(y_pred) == len(split.test_y), "Prediction length mismatch"
    assert all(0 <= p <= 1 for p in y_pred), "Predictions should be probabilities"

    from pipeline.evaluation import evaluate, backtest, print_evaluation

    metrics = evaluate(split.test_y, y_pred)
    bt = backtest(
        split.test_y, y_pred,
        trade_returns=split.test_ret,
        entry_threshold=cfg.backtest.entry_threshold,
        fee_rate=cfg.backtest.fee_rate,
        max_position_usd=cfg.backtest.max_position_usd,
        kelly_sizing=cfg.backtest.kelly_sizing,
        kelly_cap=cfg.backtest.kelly_cap,
        initial_bankroll=cfg.backtest.initial_bankroll,
    )
    print_evaluation(metrics, bt)

    # The backtest must use realised price moves, never the even-money
    # placeholder — otherwise ROI is off by orders of magnitude.
    assert bt.payoff_model == "return", "Backtest fell back to the binary payoff"
    assert split.test_ret is not None, "Split did not carry trade returns"

    # No label may leak across a split boundary: the gap between the last
    # training row and the first validation row must cover the label horizon.
    from pipeline.splitter import purge_minutes
    from datetime import timedelta
    need = timedelta(minutes=purge_minutes(cfg.split, cfg.label, cfg.bucket.bucket_minutes))
    assert split.val_start - split.train_end >= need, (
        f"train→val gap {split.val_start - split.train_end} < required {need}"
    )
    assert split.test_start - split.val_end >= need, (
        f"val→test gap {split.test_start - split.val_end} < required {need}"
    )

    # Realised returns must be economically sane: a 30-minute move on a
    # prediction market is a few percent, not a few hundred percent.
    finite = split.test_ret[np.isfinite(split.test_ret)]
    assert np.abs(finite).mean() < 0.5, (
        f"mean |trade_return| = {np.abs(finite).mean():.3f} — implausible"
    )

    print("\n  ALL ASSERTIONS PASSED")
    print("=" * 60)


def _cleanup() -> None:
    """Remove the intermediate Parquet files this test writes."""
    for name in (
        "bucketed.parquet",
        "test_filled.parquet",
        "test_filled_gaps.parquet",
        "test_filled_final.parquet",
    ):
        Path(name).unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        test_full_pipeline()
    finally:
        _cleanup()
