"""
Trade aggregation into fixed-size time buckets per market.
Produces bucketed trade statistics (OHLC-style + volume metrics).
"""

import polars as pl

from config import BucketConfig


def aggregate_trades(
    trades: pl.LazyFrame,
    cfg: BucketConfig,
) -> pl.DataFrame:
    """
    Aggregate raw trades into 5-minute buckets per market.

    Returns a DataFrame with one row per (market_id, bucket_time) containing:
    - trade_count, total_usd, total_tokens
    - open_price, close_price, high_price, low_price
    - mean_price, vwap (volume-weighted average price)
    - buy_ratio (fraction of YES trades)
    - whale_count, whale_usd (trades > whale threshold)
    - momentum (close - open price change)
    """

    bucket_dur = f"{cfg.bucket_minutes}m"

    # Truncate timestamp to bucket boundary
    trades_bucketed = trades.with_columns(
        pl.col("timestamp")
        .dt.truncate(bucket_dur)
        .alias("bucket_time"),
    )

    # Aggregate per (market_id, bucket_time)
    agg_exprs = [
        # Counts
        pl.len().alias("trade_count"),
        pl.col("usd_amount").sum().alias("total_usd"),
        pl.col("token_amount").sum().alias("total_tokens"),
        # Price stats
        pl.col("price").first().alias("open_price"),
        pl.col("price").last().alias("close_price"),
        pl.col("price").max().alias("high_price"),
        pl.col("price").min().alias("low_price"),
        pl.col("price").mean().alias("mean_price"),
        # VWAP: sum(price * usd) / sum(usd)
        (pl.col("price") * pl.col("usd_amount")).sum().alias("_price_x_usd"),
        # YES/NO ratio
        pl.col("is_yes").mean().alias("yes_ratio"),
        # Whale trades
        (pl.col("usd_amount") > cfg.whale_threshold_usd)
        .sum()
        .alias("whale_count"),
        pl.col("usd_amount")
        .filter(pl.col("usd_amount") > cfg.whale_threshold_usd)
        .sum()
        .alias("whale_usd"),
    ]

    # Add buy ratio if is_buy column exists
    if "is_buy" in trades_bucketed.collect_schema().names():
        agg_exprs.append(pl.col("is_buy").mean().alias("buy_ratio"))

    bucketed = (
        trades_bucketed.group_by(["market_id", "bucket_time"])
        .agg(agg_exprs)
        .sort(["market_id", "bucket_time"])
    ).collect()

    # Compute VWAP
    bucketed = bucketed.with_columns(
        (pl.col("_price_x_usd") / pl.col("total_usd"))
        .fill_nan(None)
        .alias("vwap"),
    ).drop("_price_x_usd")

    # Momentum = close - open
    bucketed = bucketed.with_columns(
        (pl.col("close_price") - pl.col("open_price")).alias("momentum"),
    )

    # Fill null whale_usd with 0
    bucketed = bucketed.with_columns(
        pl.col("whale_usd").fill_null(0.0),
        pl.col("whale_count").fill_null(0),
    )

    return bucketed
