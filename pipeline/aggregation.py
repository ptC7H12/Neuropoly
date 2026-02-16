"""
Trade aggregation into fixed-size time buckets per market.
Produces bucketed trade statistics (OHLC-style + volume metrics).

MEMORY OPTIMIZED:
- No .collect()
- Writes directly to Parquet via sink_parquet
"""

import polars as pl
from pathlib import Path

from config import BucketConfig


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

    # Aggregation expressions
    agg_exprs = [
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

    # Optional buy ratio
    if "is_buy" in trades_bucketed.collect_schema().names():
        agg_exprs.append(pl.col("is_buy").mean().alias("buy_ratio"))

    # Build lazy aggregation
    bucketed_lazy = (
        trades_bucketed
        .group_by(["market_id", "bucket_time"])
        .agg(agg_exprs)
        .sort(["market_id", "bucket_time"])
    )

    # VWAP
    bucketed_lazy = bucketed_lazy.with_columns(
        (pl.col("_price_x_usd") / pl.col("total_usd"))
        .fill_nan(None)
        .alias("vwap"),
    ).drop("_price_x_usd")

    # Momentum
    bucketed_lazy = bucketed_lazy.with_columns(
        (pl.col("close_price") - pl.col("open_price")).alias("momentum"),
    )

    # Fill nulls
    bucketed_lazy = bucketed_lazy.with_columns(
        pl.col("whale_usd").fill_null(0.0),
        pl.col("whale_count").fill_null(0),
    )

    # IMPORTANT: write directly to disk instead of collect
    bucketed_lazy.sink_parquet(output_path)

    print(f"  Bucketed data written to: {output_path}")

    return output_path
