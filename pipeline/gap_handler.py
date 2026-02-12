"""
Gap detection, filling, and exclusion for bucketed trade data.

Handles two types of gaps:
1. Explicit gap period (e.g., Oct 2025 – Feb 2026 data outage)
2. Implicit gaps (markets with no trades for extended periods)
"""

import polars as pl
from datetime import timedelta

from config import GapConfig, BucketConfig


def detect_gaps(
    bucketed: pl.DataFrame,
    bucket_cfg: BucketConfig,
    gap_cfg: GapConfig,
) -> pl.DataFrame:
    """
    Analyse gaps per market. Returns a summary DataFrame:
    - market_id, first_bucket, last_bucket, bucket_count,
      expected_count, missing_count, gap_ratio,
      in_explicit_gap (bool)
    """

    bucket_minutes = bucket_cfg.bucket_minutes

    summary = bucketed.group_by("market_id").agg(
        pl.col("bucket_time").min().alias("first_bucket"),
        pl.col("bucket_time").max().alias("last_bucket"),
        pl.col("bucket_time").count().alias("bucket_count"),
    )

    # Calculate expected bucket count
    summary = summary.with_columns(
        (
            (pl.col("last_bucket") - pl.col("first_bucket"))
            .dt.total_seconds()
            / (bucket_minutes * 60)
            + 1
        )
        .cast(pl.Int64)
        .alias("expected_count"),
    )

    summary = summary.with_columns(
        (pl.col("expected_count") - pl.col("bucket_count")).alias("missing_count"),
    )

    summary = summary.with_columns(
        (pl.col("missing_count") / pl.col("expected_count")).alias("gap_ratio"),
    )

    # Flag markets that overlap with the explicit gap period
    if gap_cfg.gap_start and gap_cfg.gap_end:
        gap_start_dt = gap_cfg.gap_start
        gap_end_dt = gap_cfg.gap_end
        summary = summary.with_columns(
            (
                (pl.col("first_bucket") < gap_end_dt)
                & (pl.col("last_bucket") > gap_start_dt)
            ).alias("overlaps_explicit_gap"),
        )

    return summary


def fill_buckets(
    bucketed: pl.DataFrame,
    bucket_cfg: BucketConfig,
    gap_cfg: GapConfig,
) -> pl.DataFrame:
    """
    For each market, create a complete time series of buckets.
    Missing buckets are filled with NaN values and flagged.

    Buckets inside the explicit gap period are marked as
    `in_gap=True` so they can be excluded from training.
    """

    bucket_minutes = bucket_cfg.bucket_minutes
    all_filled = []

    market_ids = bucketed["market_id"].unique().sort().to_list()

    for market_id in market_ids:
        market_df = bucketed.filter(pl.col("market_id") == market_id)

        start = market_df["bucket_time"].min()
        end = market_df["bucket_time"].max()

        if start is None or end is None:
            continue

        # Generate complete bucket range
        n_buckets = int((end - start).total_seconds() / (bucket_minutes * 60)) + 1
        full_range = pl.DataFrame(
            {
                "bucket_time": pl.datetime_range(
                    start, end, timedelta(minutes=bucket_minutes), eager=True
                ),
                "market_id": [market_id] * n_buckets,
            }
        )

        # Ensure types match
        full_range = full_range.with_columns(
            pl.col("market_id").cast(pl.Int64),
        )

        # Left join: keep all buckets, fill missing with null
        merged = full_range.join(
            market_df, on=["market_id", "bucket_time"], how="left"
        )

        # Flag: is this a filled (empty) bucket?
        merged = merged.with_columns(
            pl.col("trade_count").is_null().alias("is_empty_bucket"),
        )

        # Flag: is this bucket in the explicit gap period?
        if gap_cfg.gap_start and gap_cfg.gap_end:
            merged = merged.with_columns(
                (
                    (pl.col("bucket_time") >= gap_cfg.gap_start)
                    & (pl.col("bucket_time") < gap_cfg.gap_end)
                ).alias("in_gap"),
            )
        else:
            merged = merged.with_columns(
                pl.lit(False).alias("in_gap"),
            )

        # Fill numeric columns with 0 for trade counts, NaN for prices
        count_cols = ["trade_count", "whale_count"]
        usd_cols = ["total_usd", "total_tokens", "whale_usd"]
        price_cols = [
            "open_price",
            "close_price",
            "high_price",
            "low_price",
            "mean_price",
            "vwap",
            "momentum",
        ]

        for col in count_cols:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0))

        for col in usd_cols:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0.0))

        # For price columns in empty buckets: forward-fill from last known price
        for col in price_cols:
            if col in merged.columns:
                merged = merged.with_columns(
                    pl.col(col).forward_fill().alias(col),
                )

        # yes_ratio, buy_ratio: fill with 0.5 (neutral) for empty buckets
        for col in ["yes_ratio", "buy_ratio"]:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0.5))

        all_filled.append(merged)

    if not all_filled:
        return bucketed.with_columns(
            pl.lit(False).alias("is_empty_bucket"),
            pl.lit(False).alias("in_gap"),
        )

    return pl.concat(all_filled).sort(["market_id", "bucket_time"])


def detect_consecutive_gaps(
    filled: pl.DataFrame,
    gap_cfg: GapConfig,
) -> pl.DataFrame:
    """
    Detect runs of consecutive empty buckets per market.
    Returns the filled DataFrame with an additional column
    `in_long_gap` for runs exceeding max_empty_buckets.
    """

    results = []

    for market_id in filled["market_id"].unique().sort().to_list():
        market_df = filled.filter(pl.col("market_id") == market_id)

        # Compute run-length encoding of is_empty_bucket
        empty = market_df["is_empty_bucket"].to_list()
        run_lengths = []
        current_run = 0

        for is_empty in empty:
            if is_empty:
                current_run += 1
            else:
                current_run = 0
            run_lengths.append(current_run)

        # Also compute backward run length (how long the gap continues)
        backward_runs = []
        current_run = 0
        for is_empty in reversed(empty):
            if is_empty:
                current_run += 1
            else:
                current_run = 0
            backward_runs.append(current_run)
        backward_runs.reverse()

        # A bucket is in a long gap if it's part of a consecutive run
        # that exceeds the threshold
        max_run = [max(f, b) for f, b in zip(run_lengths, backward_runs)]
        in_long_gap = [r > gap_cfg.max_empty_buckets for r in max_run]

        market_df = market_df.with_columns(
            pl.Series("in_long_gap", in_long_gap),
        )

        results.append(market_df)

    if not results:
        return filled.with_columns(pl.lit(False).alias("in_long_gap"))

    return pl.concat(results).sort(["market_id", "bucket_time"])


def apply_gap_exclusions(
    filled: pl.DataFrame,
    gap_cfg: GapConfig,
) -> pl.DataFrame:
    """
    Combine all gap flags into a single `exclude_from_training` column.
    A bucket is excluded if:
    - It falls in the explicit gap period, OR
    - It's part of a long consecutive gap
    """

    filled = filled.with_columns(
        (pl.col("in_gap") | pl.col("in_long_gap")).alias("exclude_from_training"),
    )

    return filled
