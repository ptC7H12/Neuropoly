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

    # Gap between splits (in buckets) to avoid leakage
    split_gap_buckets: int = 12  # 12 × 5min = 1 hour gap

    # Purge window: remove labels whose forward window overlaps with next split
    purge_forward_buckets: int = 6  # Must match LabelConfig.forward_window_buckets


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
class BacktestConfig:
    """Backtesting / evaluation settings."""

    # Probability threshold to enter a trade
    entry_threshold: float = 0.6

    # Transaction fee (Polymarket fee)
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
