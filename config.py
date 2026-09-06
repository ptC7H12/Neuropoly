"""
Pipeline configuration — all tuneable parameters in one place.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class DataConfig:
    """Paths and data source settings."""

    trades_path: str = "data/trades.csv"
    markets_path: str = "data/polymarket_active.csv"

    # Supported formats: "csv", "parquet", "sqlite"
    trades_format: str = "csv"
    markets_format: str = "csv"

    # If sqlite, specify the table names
    sqlite_path: Optional[str] = None
    trades_table: str = "trades"
    markets_table: str = "markets"

    # Column mapping — trades
    trades_timestamp_col: str = "timestamp"
    trades_market_id_col: str = "market_id"
    trades_side_col: str = "side"
    trades_price_col: str = "price"
    trades_usd_col: str = "usd_amount"
    trades_token_col: str = "token_amount"
    trades_direction_col: str = "direction"

    # Side mapping: token1 → YES, token2 → NO
    side_yes: str = "token1"
    side_no: str = "token2"


@dataclass
class BucketConfig:
    """Aggregation settings."""

    bucket_minutes: int = 5
    whale_threshold_usd: float = 1000.0  # Trades > this are "whale" trades


@dataclass
class GapConfig:
    """Gap detection and handling."""

    # Explicit gap period to exclude from training (Oct 2025 – Feb 2026)
    gap_start: Optional[datetime] = None
    gap_end: Optional[datetime] = None

    # Max consecutive empty buckets before flagging as gap
    max_empty_buckets: int = 48  # 48 × 5min = 4 hours

    # If True, fill gaps with NaN rows; if False, drop gap periods entirely
    fill_gaps: bool = True

    def __post_init__(self):
        if self.gap_start is None:
            self.gap_start = datetime(2025, 10, 1)
        if self.gap_end is None:
            self.gap_end = datetime(2026, 2, 1)


@dataclass
class LabelConfig:
    """Labeling strategy."""

    # Forward-looking window for label generation (in buckets)
    # 6 buckets × 5 min = 30 minutes forward
    forward_window_buckets: int = 6

    # Minimum price move to count as win (avoids labeling noise)
    min_price_move: float = 0.001

    # Also generate regression target (continuous return)
    include_regression_target: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering settings."""

    # Lag features: how many past buckets to look at
    lag_buckets: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])

    # Rolling window sizes (in buckets)
    rolling_windows: list[int] = field(default_factory=lambda: [6, 12, 24, 48])

    # Whether to include time-of-day features
    time_features: bool = True

    # Whether to include cross-market features
    cross_market_features: bool = True


@dataclass
class SplitConfig:
    """Walk-forward train/val/test split."""

    # Fraction of data for training
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Safety gap between two splits, in WALL-CLOCK MINUTES.
    # This is added on top of the label's own forward window, which is purged
    # automatically (see pipeline/splitter.purge_minutes).  Measuring the gap
    # in minutes rather than in rows is essential: rows are interleaved across
    # thousands of markets, so a row-based gap collapses to seconds of wall
    # time and lets labels leak across the boundary.
    split_gap_minutes: int = 60


@dataclass
class ModelConfig:
    """LightGBM parameters."""

    objective: str = "binary"
    boosting_type: str = "gbdt"
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 7
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_estimators: int = 5000
    n_jobs: int = 10
    early_stopping_rounds: int = 50
    verbose: int = -1  # Suppressed; monitor.py handles output

    def to_lgbm_params(self) -> dict:
        return {
            "objective": self.objective,
            "boosting_type": self.boosting_type,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_estimators": self.n_estimators,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }


@dataclass
class MonitorConfig:
    """Live training monitor settings."""

    # How often to refresh the dashboard (in LightGBM iterations)
    log_interval: int = 10

    # Show feature importance in dashboard
    show_feature_importance: bool = True
    top_n_features: int = 15

    # Save training log to file
    log_file: Optional[str] = "training_log.jsonl"

    # Enable rich terminal dashboard
    rich_dashboard: bool = True


@dataclass
class CostConfig:
    """
    Round-trip trading cost, as a fraction of the position's value.

        cost(tp) = ( spread_abs + fee_legs * fee_rate * min(tp, 1-tp) ) / tp

    where `tp` is the price of the token actually held (P(YES) for a YES
    position, 1 - P(YES) for a NO one).

    Why it cannot be a flat percentage
    ----------------------------------
    Costs are quoted in absolute price units — a spread is so many ticks —
    but a position's value is `tp` per share.  The relative cost therefore
    scales with 1/tp, and at the price extremes it explodes.  Measured on
    120 live order books:

        min(p, 1-p)      median spread   spread / price
        0.35 - 0.50           0.0270            63 %
        0.20 - 0.35           0.0200             8 %
        0.10 - 0.20           0.0370            28 %
        0.05 - 0.10           0.0160            23 %
        0.02 - 0.05           0.0020             6 %
        0.00 - 0.02           0.0010            40 %

    A flat rate hides that completely, and the direction of the error is the
    dangerous one: it makes cheap-looking trades out of the markets where
    trading is in fact most expensive.

    Calibration
    -----------
    `spread_abs` is the one empirical input and it varies a lot between
    markets — across those 120 books the quartiles were 0.001 / 0.010 /
    0.039.  The default is the median.  Set it from the markets you actually
    trade; the 1/tp shape holds regardless of the constant.

    `fee_rate` is Polymarket's own schedule (Gamma `feeSchedule.rate`,
    taker-only): fee per share = rate * min(p, 1-p).  `fee_legs` is how many
    legs you cross as taker — 2 for taker in and out, 1 if you expect to
    exit as maker, 0 for a pure maker strategy.
    """

    # Absolute bid/ask spread in price units, paid once per round trip
    spread_abs: float = 0.010

    # Polymarket taker fee rate (Gamma feeSchedule.rate)
    fee_rate: float = 0.04

    # Number of legs crossed as taker (0, 1 or 2)
    fee_legs: int = 2

    # Trades whose round-trip cost exceeds this fraction of the position are
    # treated as untradeable and skipped, instead of being booked as a
    # near-total loss.  1.0 = "the costs eat the whole position".
    max_cost: float = 1.0

    def round_trip_cost(self, token_price):
        """
        Cost of entering and exiting, as a fraction of the position's value.

        Accepts a float or a numpy array.
        """
        import numpy as np

        tp = np.clip(np.asarray(token_price, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        fee = self.fee_legs * self.fee_rate * np.minimum(tp, 1.0 - tp)
        cost = (self.spread_abs + fee) / tp
        return cost if cost.ndim else float(cost)


@dataclass
class BacktestConfig:
    """Backtesting / evaluation settings."""

    # Probability threshold to enter a trade
    entry_threshold: float = 0.6

    # Price-aware round-trip cost model (see CostConfig).
    cost: "CostConfig" = field(default_factory=lambda: CostConfig())

    # Flat fallback cost, as a fraction of the stake.  Only used when the
    # backtest is given no entry prices to compute a per-trade cost from.
    fee_rate: float = 0.02

    # Max position size (USD)
    max_position_usd: float = 10.0

    # Use Kelly criterion for sizing
    kelly_sizing: bool = False

    # Kelly fraction cap (max fraction of bankroll per trade)
    kelly_cap: float = 0.25

    # Initial bankroll
    initial_bankroll: float = 100.0


@dataclass
class PipelineConfig:
    """Master config combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    bucket: BucketConfig = field(default_factory=BucketConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Random seed for reproducibility
    seed: int = 42
