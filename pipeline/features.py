"""
Feature engineering: bucket-level, market-level, lag, rolling, cross, and time features.
"""

import gc
import math
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from pipeline.rowgroups import (
    iter_market_row_groups,
    write_market_table,
)
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
    df = _add_cross_features(df, cfg)

    # 5. Time features
    if cfg.time_features:
        df = _add_time_features(df)

    return df


def build_features_streaming(
    filled_path: str,
    markets: pl.DataFrame,
    cfg: FeatureConfig,
    output_path: str = "features.parquet",
    batch_markets: int = 100,
) -> str:
    """
    Build features from a gap-filled Parquet file, `batch_markets` markets at
    a time.

    *filled_path* must contain exactly one row group per market — this is
    guaranteed by the gap_handler streaming pipeline (fill_buckets →
    detect_consecutive_gaps → apply_gap_exclusions all write with PyArrow,
    one market per write_table call).  The same invariant is preserved on
    output, because add_labels_streaming relies on it.

    Why batch: every feature here is already market-aware via .over(
    "market_id"), so a batch produces exactly the same numbers as one market
    at a time — but a single-market call spends ~17 ms of Polars per-call
    overhead on a frame of a few dozen rows.  At 3 200 markets that was 76 s
    of the 130 s preprocessing chain.  Batching amortises it.

    RAM cost: peak is `batch_markets` markets instead of one.  Lower it if a
    chunk is memory-tight; batch_markets=1 restores the old behaviour.

    Returns output_path.
    """

    output_path = str(Path(output_path))

    if hasattr(markets, "collect"):
        markets = markets.collect()

    pf = pq.ParquetFile(filled_path)
    n_rg = pf.metadata.num_row_groups
    writer = None
    batch_markets = max(1, batch_markets)

    for batch_start in range(0, n_rg, batch_markets):
        group_ids = list(range(batch_start, min(batch_start + batch_markets, n_rg)))
        batch_df = pl.from_arrow(pf.read_row_groups(group_ids))

        featured = build_features(batch_df, markets, cfg)
        del batch_df

        # One row group per market on the way out — add_labels_streaming
        # reads this file back on that assumption.
        for part in featured.partition_by("market_id", maintain_order=True):
            arrow_tbl = part.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema=arrow_tbl.schema,
                    compression="SNAPPY",
                    version="2.6",
                )
            write_market_table(writer, arrow_tbl)
            del arrow_tbl, part

        del featured
        gc.collect()

        done = min(batch_start + batch_markets, n_rg)
        print(f"  build_features: {done}/{n_rg} markets", flush=True)

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


def _add_cross_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
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

    # Volume concentration: this bucket's volume as a share of the volume
    # traded in the trailing window.
    #
    # Two denominators were wrong before:
    #   * `volume` from the markets table — the market's total LIFETIME
    #     volume as of the CSV export.  For a bucket in 2020 that is a value
    #     from the future, and being constant per market it also works as a
    #     market-identity feature.
    #   * a cumulative sum since the market's first bucket — causal, but not
    #     chunk-invariant: train_chunked.py rebuilds features per 90-day
    #     chunk, so the sum restarted at zero every chunk while run_pipeline
    #     accumulated over the full history.  Same feature name, values
    #     differing by ~11x on identical buckets.
    #
    # A trailing window is causal AND identical in both paths, because
    # train_chunked reads --context-buckets of history before each chunk.
    _win = max(cfg.rolling_windows) if cfg.rolling_windows else 1
    _vol_sum = f"volume_sum{_win}"
    if _vol_sum in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col(_vol_sum))
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


def get_feature_columns(df) -> list[str]:
    """Return the list of feature column names (excludes metadata and target).

    Accepts a pl.DataFrame, pl.LazyFrame, or a Polars Schema object so that
    callers can avoid a full collect() just to obtain column names.
    """

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
        "trade_return",
        "trade_return_opp",
        "entry_token_price",
        "entry_token_price_opp",
        "question",
        "close_time",
        # Snapshot columns from the markets table: their value is the state
        # at export time, not at bucket time, so using them directly is
        # look-ahead.  They stay in the frame because derived features are
        # built from them, but they are not fed to the model.
        "volume",
        "liquidity",
    }

    # Schema objects (returned by collect_schema()) expose .names()
    if hasattr(df, "names"):
        columns = df.names()
    else:
        columns = df.columns

    return [col for col in columns if col not in exclude]


def _entropy(p: float) -> float:
    """Binary entropy for probability p."""
    if p is None or p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def report_degenerate_features(
    labeled_path: str,
    feature_cols: list[str],
    max_listed: int = 10,
) -> dict[str, list[str]]:
    """
    Find features that carry no information: all-null, or a single value.

    A model cannot split on these, so they are dead weight — and an all-null
    column usually means the source data never had that field (markets.csv
    has no yes_price / no_price / liquidity, for instance, so
    convert_to_parquet writes them as null).  Worth knowing before reading a
    feature-importance table.
    """
    import polars as pl

    lf = pl.scan_parquet(labeled_path)
    present = [c for c in feature_cols if c in lf.collect_schema().names()]
    if not present:
        return {"all_null": [], "constant": []}

    stats = lf.select(
        [pl.col(c).null_count().alias(f"{c}__nulls") for c in present]
        + [pl.col(c).n_unique().alias(f"{c}__uniq") for c in present]
        + [pl.len().alias("__n")]
    ).collect()

    n = int(stats["__n"][0])
    all_null, constant = [], []
    for c in present:
        if int(stats[f"{c}__nulls"][0]) == n:
            all_null.append(c)
        elif int(stats[f"{c}__uniq"][0]) <= 1:
            constant.append(c)

    if all_null or constant:
        print("\n  Degenerate features (no information for the model):")
        if all_null:
            shown = ", ".join(all_null[:max_listed])
            more = f" (+{len(all_null) - max_listed} more)" if len(all_null) > max_listed else ""
            print(f"    all null ({len(all_null)}): {shown}{more}")
        if constant:
            shown = ", ".join(constant[:max_listed])
            more = f" (+{len(constant) - max_listed} more)" if len(constant) > max_listed else ""
            print(f"    constant ({len(constant)}): {shown}{more}")

    return {"all_null": all_null, "constant": constant}
