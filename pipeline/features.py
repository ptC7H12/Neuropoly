"""
Feature engineering: bucket-level, market-level, lag, rolling, cross, and time features.
"""

import gc
import math
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from datetime import datetime

from config import FeatureConfig


def build_features(
    bucketed: pl.DataFrame,
    markets: pl.DataFrame,
    cfg: FeatureConfig,
) -> pl.DataFrame:
    """
    Build the full feature matrix from bucketed trades + market snapshots.

    Feature groups:
    1. Bucket-level (already in bucketed data)
    2. Lag features (past bucket values)
    3. Rolling features (moving averages, std)
    4. Market-level (from markets table)
    5. Cross / relative features
    6. Time features (hour, day-of-week, etc.)
    """

    df = bucketed.clone()

    # Ensure markets is a DataFrame (not LazyFrame)
    if hasattr(markets, "collect"):
        markets = markets.collect()

    # 1. Lag features — per market
    df = _add_lag_features(df, cfg)

    # 2. Rolling features — per market
    df = _add_rolling_features(df, cfg)

    # 3. Market-level features
    df = _add_market_features(df, markets)

    # 4. Cross / relative features
    df = _add_cross_features(df)

    # 5. Time features
    if cfg.time_features:
        df = _add_time_features(df)

    return df


def build_features_streaming(
    filled_path: str,
    markets: pl.DataFrame,
    cfg: FeatureConfig,
    output_path: str = "features.parquet",
) -> str:
    """
    Build features one market at a time from a gap-filled Parquet file.

    *filled_path* must contain exactly one row group per market — this is
    guaranteed by the gap_handler streaming pipeline (fill_buckets →
    detect_consecutive_gaps → apply_gap_exclusions all write with PyArrow,
    one market per write_table call).

    Writes the feature matrix to *output_path* via a streaming PyArrow
    writer — peak RAM = one market's rows.  Returns output_path.
    """

    output_path = str(Path(output_path))

    if hasattr(markets, "collect"):
        markets = markets.collect()

    pf = pq.ParquetFile(filled_path)
    n_rg = pf.metadata.num_row_groups
    writer = None

    for rg_idx in range(n_rg):
        market_df = pl.from_arrow(pf.read_row_group(rg_idx))

        featured = build_features(market_df, markets, cfg)
        del market_df

        arrow_tbl = featured.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=arrow_tbl.schema,
                compression="SNAPPY",
                version="2.6",
            )
        writer.write_table(arrow_tbl)

        del featured, arrow_tbl
        gc.collect()

        if (rg_idx + 1) % 100 == 0 or (rg_idx + 1) == n_rg:
            print(f"  build_features: {rg_idx + 1}/{n_rg} markets", flush=True)

    if writer:
        writer.close()
    else:
        # Edge case: empty filled file — write empty Parquet
        pl.scan_parquet(filled_path).collect().write_parquet(output_path)

    return output_path


def _add_lag_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Add lagged values of key columns per market."""

    lag_cols = ["mean_price", "total_usd", "trade_count", "momentum", "yes_ratio"]

    exprs = []
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in cfg.lag_buckets:
            exprs.append(
                pl.col(col)
                .shift(lag)
                .over("market_id")
                .alias(f"{col}_lag{lag}")
            )

    if exprs:
        df = df.with_columns(exprs)

    # Price change from lag
    for lag in cfg.lag_buckets:
        lag_col = f"mean_price_lag{lag}"
        if lag_col in df.columns:
            df = df.with_columns(
                (pl.col("mean_price") - pl.col(lag_col)).alias(
                    f"price_change_lag{lag}"
                ),
                (
                    (pl.col("mean_price") - pl.col(lag_col)) / pl.col(lag_col)
                )
                .fill_nan(None)
                .alias(f"price_return_lag{lag}"),
            )

    return df


def _add_rolling_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Add rolling statistics per market."""

    for window in cfg.rolling_windows:
        # Rolling mean price
        df = df.with_columns(
            pl.col("mean_price")
            .rolling_mean(window_size=window)
            .over("market_id")
            .alias(f"price_ma{window}"),
        )

        # Rolling std (volatility)
        df = df.with_columns(
            pl.col("mean_price")
            .rolling_std(window_size=window)
            .over("market_id")
            .alias(f"price_std{window}"),
        )

        # Rolling sum of USD volume
        df = df.with_columns(
            pl.col("total_usd")
            .rolling_sum(window_size=window)
            .over("market_id")
            .alias(f"volume_sum{window}"),
        )

        # Rolling mean trade count
        df = df.with_columns(
            pl.col("trade_count")
            .rolling_mean(window_size=window)
            .over("market_id")
            .alias(f"trade_count_ma{window}"),
        )

        # Rolling momentum mean
        df = df.with_columns(
            pl.col("momentum")
            .rolling_mean(window_size=window)
            .over("market_id")
            .alias(f"momentum_ma{window}"),
        )

    # Price relative to rolling average (mean reversion signal)
    for window in cfg.rolling_windows:
        ma_col = f"price_ma{window}"
        df = df.with_columns(
            ((pl.col("mean_price") - pl.col(ma_col)) / pl.col(ma_col))
            .fill_nan(None)
            .alias(f"price_vs_ma{window}"),
        )

    return df


def _add_market_features(df: pl.DataFrame, markets: pl.DataFrame) -> pl.DataFrame:
    """Join market snapshot features to bucketed trades."""

    # Select relevant market columns
    market_cols = ["market_id"]
    optional_cols = [
        "yes_price",
        "no_price",
        "volume",
        "liquidity",
        "close_time",
        "question",
    ]
    for col in optional_cols:
        if col in markets.columns:
            market_cols.append(col)

    market_features = markets.select(market_cols)

    # Join
    df = df.join(market_features, on="market_id", how="left")

    # Derived market features
    if "yes_price" in df.columns and "no_price" in df.columns:
        # Spread
        df = df.with_columns(
            (pl.col("yes_price") - pl.col("no_price"))
            .abs()
            .alias("market_spread"),
        )

        # Entropy: -p*log(p) - (1-p)*log(1-p) — vectorized
        p = pl.col("yes_price").clip(1e-9, 1.0 - 1e-9)
        df = df.with_columns(
            (-(p * p.log(base=2) + (1 - p) * (1 - p).log(base=2)))
            .alias("market_entropy"),
        )

    # Days to close
    if "close_time" in df.columns:
        # Normalize to tz-naive Datetime so subtraction works regardless of
        # whether close_time was loaded from a tz-aware Parquet or a raw CSV.
        close_dtype = df.schema["close_time"]
        if getattr(close_dtype, "time_zone", None):
            close_col = pl.col("close_time").dt.replace_time_zone(None)
        else:
            close_col = pl.col("close_time").cast(pl.Datetime("us"))

        df = df.with_columns(close_col.alias("close_time_dt"))
        df = df.with_columns(
            (
                (pl.col("close_time_dt") - pl.col("bucket_time")).dt.total_seconds()
                / 86400
            ).alias("days_to_close"),
        )
        df = df.drop("close_time_dt")

    return df


def _add_cross_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cross features: relationships between trade and market data."""

    # Entry price vs market price
    if "yes_price" in df.columns:
        df = df.with_columns(
            (pl.col("mean_price") - pl.col("yes_price")).alias("entry_vs_market"),
        )

    # Trade size relative to market liquidity
    if "liquidity" in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col("liquidity"))
            .fill_nan(None)
            .alias("trade_size_vs_liquidity"),
        )

    # Volume concentration: bucket volume / market total volume
    if "volume" in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col("volume"))
            .fill_nan(None)
            .alias("volume_concentration"),
        )

    # Whale ratio weighted by momentum
    if "whale_count" in df.columns:
        df = df.with_columns(
            (
                pl.col("whale_count").cast(pl.Float64)
                / pl.col("trade_count").cast(pl.Float64)
            )
            .fill_nan(0.0)
            .alias("whale_ratio"),
        )

        df = df.with_columns(
            (pl.col("whale_ratio") * pl.col("momentum")).alias(
                "whale_momentum"
            ),
        )

    # Distance to close weighted by momentum
    if "days_to_close" in df.columns:
        df = df.with_columns(
            (pl.col("momentum") / (pl.col("days_to_close") + 1))
            .fill_nan(None)
            .alias("momentum_per_day"),
        )

    return df


def _add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Time-based features from bucket_time."""

    df = df.with_columns(
        pl.col("bucket_time").dt.hour().alias("hour"),
        pl.col("bucket_time").dt.weekday().alias("day_of_week"),
    )

    # Cyclical encoding for hour (sin/cos) — native Polars ops
    df = df.with_columns(
        (pl.col("hour").cast(pl.Float64) * 2 * math.pi / 24)
        .sin()
        .alias("hour_sin"),
        (pl.col("hour").cast(pl.Float64) * 2 * math.pi / 24)
        .cos()
        .alias("hour_cos"),
        (pl.col("day_of_week").cast(pl.Float64) * 2 * math.pi / 7)
        .sin()
        .alias("dow_sin"),
        (pl.col("day_of_week").cast(pl.Float64) * 2 * math.pi / 7)
        .cos()
        .alias("dow_cos"),
    )

    # Is weekend
    df = df.with_columns(
        (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend"),
    )

    return df


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes metadata and target)."""

    exclude = {
        "bucket_time",
        "market_id",
        "is_empty_bucket",
        "in_gap",
        "in_long_gap",
        "exclude_from_training",
        "win",
        "future_return",
        "future_price",
        "question",
        "close_time",
    }

    return [col for col in df.columns if col not in exclude]


def _entropy(p: float) -> float:
    """Binary entropy for probability p."""
    if p is None or p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
