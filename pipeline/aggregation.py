"""
Trade aggregation into fixed-size time buckets per market.
Produces bucketed trade statistics (OHLC-style + volume metrics).

All price columns are P(YES): pipeline/data_loader normalises NO-side
trade prices to their YES equivalent before aggregation, so open/close/
high/low/mean/vwap describe the market, not the YES/NO trade mix.

MEMORY OPTIMIZED:
- No .collect()
- Writes directly to Parquet via sink_parquet
"""

import polars as pl
from pathlib import Path

from config import BucketConfig


def bucket_agg_exprs(cfg: BucketConfig, has_buy: bool = False) -> list:
    """
    The bucket aggregation expressions, in one place.

    Shared by the batch pipeline and the live path so a bucket built at
    inference time is byte-for-byte the same shape as one built during
    training.
    """
    exprs = [
        pl.len().alias("trade_count"),
        pl.col("usd_amount").sum().alias("total_usd"),
        pl.col("token_amount").sum().alias("total_tokens"),
        pl.col("price").first().alias("open_price"),
        pl.col("price").last().alias("close_price"),
        pl.col("price").max().alias("high_price"),
        pl.col("price").min().alias("low_price"),
        pl.col("price").mean().alias("mean_price"),
        (pl.col("price") * pl.col("usd_amount")).sum().alias("_price_x_usd"),
        pl.col("is_yes").mean().alias("yes_ratio"),
        (pl.col("usd_amount") > cfg.whale_threshold_usd).sum().alias("whale_count"),
        pl.col("usd_amount")
        .filter(pl.col("usd_amount") > cfg.whale_threshold_usd)
        .sum()
        .alias("whale_usd"),
    ]
    if has_buy:
        exprs.append(pl.col("is_buy").mean().alias("buy_ratio"))
    return exprs


def _finalize_buckets(frame):
    """Derived columns computed after the group_by (lazy or eager)."""
    return (
        frame.with_columns(
            (pl.col("_price_x_usd") / pl.col("total_usd"))
            .fill_nan(None)
            .alias("vwap"),
        )
        .drop("_price_x_usd")
        .with_columns(
            (pl.col("close_price") - pl.col("open_price")).alias("momentum"),
        )
        .with_columns(
            pl.col("whale_usd").fill_null(0.0),
            pl.col("whale_count").fill_null(0),
        )
    )


def aggregate_trades(
    trades: pl.LazyFrame,
    cfg: BucketConfig,
    output_path: str = "bucketed.parquet",
) -> str:
    """
    Aggregate raw trades into time buckets per market.

    Instead of returning a DataFrame (RAM heavy),
    this function writes directly to a Parquet file and returns the file path.
    """

    bucket_dur = f"{cfg.bucket_minutes}m"
    output_path = str(Path(output_path))

    # Truncate timestamp to bucket boundary
    trades_bucketed = trades.with_columns(
        pl.col("timestamp")
        .dt.truncate(bucket_dur)
        .alias("bucket_time"),
    )

    has_buy = "is_buy" in trades_bucketed.collect_schema().names()

    bucketed_lazy = _finalize_buckets(
        trades_bucketed
        .group_by(["market_id", "bucket_time"])
        .agg(bucket_agg_exprs(cfg, has_buy))
        .sort(["market_id", "bucket_time"])
    )

    # IMPORTANT: write directly to disk instead of collect
    bucketed_lazy.sink_parquet(output_path)

    print(f"  Bucketed data written to: {output_path}")

    return output_path


def aggregate_trades_eager(
    trades: pl.DataFrame,
    cfg: BucketConfig,
    market_id: int = 0,
) -> pl.DataFrame:
    """
    Bucket a single market's trades in memory, for the live path.

    Same expressions as aggregate_trades, so a live bucket matches a
    training bucket exactly.  `trades` needs timestamp / price / usd_amount /
    token_amount / is_yes, with prices already normalised to P(YES).
    """
    bucket_dur = f"{cfg.bucket_minutes}m"
    has_buy = "is_buy" in trades.columns

    bucketed = (
        trades.with_columns(
            pl.col("timestamp").dt.truncate(bucket_dur).alias("bucket_time"),
            pl.lit(market_id, dtype=pl.Int64).alias("market_id"),
        )
        .group_by(["market_id", "bucket_time"])
        .agg(bucket_agg_exprs(cfg, has_buy))
        .sort(["market_id", "bucket_time"])
    )

    return _finalize_buckets(bucketed)
