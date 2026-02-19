"""
Gap detection, filling, and exclusion for bucketed trade data.

Handles two types of gaps:
1. Explicit gap period (e.g., Oct 2025 – Feb 2026 data outage)
2. Implicit gaps (markets with no trades for extended periods)

Memory strategy
---------------
fill_buckets() and detect_consecutive_gaps() write results directly to
Parquet via a streaming PyArrow writer — only one market's data lives in
RAM at any time.  apply_gap_exclusions() uses Polars' sink_parquet() so
the final flag column is also computed without a full collect().
"""

import gc
from pathlib import Path
from datetime import timedelta

import polars as pl
import pyarrow.parquet as pq

from config import GapConfig, BucketConfig


# ── Gap detection (summary only — stays small, kept in RAM) ───────────────────

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


# ── Streaming gap fill ────────────────────────────────────────────────────────

def fill_buckets(
    bucketed: pl.DataFrame,
    bucket_cfg: BucketConfig,
    gap_cfg: GapConfig,
    output_path: str = "filled.parquet",
) -> str:
    """
    For each market, create a complete time series of buckets.
    Missing buckets are filled with NaN / 0 and flagged.

    Writes one market at a time directly to *output_path* via a
    PyArrow streaming writer — peak RAM = one market's rows.

    Returns the output file path.
    """

    bucket_minutes = bucket_cfg.bucket_minutes
    output_path = str(Path(output_path))

    market_ids = bucketed["market_id"].unique().sort().to_list()
    n_markets = len(market_ids)
    writer = None

    for idx, market_id in enumerate(market_ids, 1):
        market_df = bucketed.filter(pl.col("market_id") == market_id)

        start = market_df["bucket_time"].min()
        end = market_df["bucket_time"].max()

        if start is None or end is None:
            del market_df
            continue

        # Generate complete bucket range for this market
        n_buckets = int((end - start).total_seconds() / (bucket_minutes * 60)) + 1
        full_range = pl.DataFrame(
            {
                "bucket_time": pl.datetime_range(
                    start, end, timedelta(minutes=bucket_minutes), eager=True
                ),
                "market_id": [market_id] * n_buckets,
            }
        ).with_columns(pl.col("market_id").cast(pl.Int64))

        # Left join: keep all buckets, fill missing with null
        merged = full_range.join(
            market_df, on=["market_id", "bucket_time"], how="left"
        )
        del full_range, market_df

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
            merged = merged.with_columns(pl.lit(False).alias("in_gap"))

        # Fill numeric columns
        count_cols = ["trade_count", "whale_count"]
        usd_cols   = ["total_usd", "total_tokens", "whale_usd"]
        price_cols = [
            "open_price", "close_price", "high_price",
            "low_price", "mean_price", "vwap", "momentum",
        ]

        for col in count_cols:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0))

        for col in usd_cols:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0.0))

        # Forward-fill price columns from last known value
        for col in price_cols:
            if col in merged.columns:
                merged = merged.with_columns(
                    pl.col(col).forward_fill().alias(col)
                )

        # Neutral fill for ratio columns
        for col in ["yes_ratio", "buy_ratio"]:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0.5))

        # ── stream-write this market's rows to Parquet ──
        arrow_tbl = merged.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=arrow_tbl.schema,
                compression="SNAPPY",
                version="2.6",
            )
        writer.write_table(arrow_tbl)

        del merged, arrow_tbl
        gc.collect()

        if idx % 100 == 0 or idx == n_markets:
            print(f"  fill_buckets: {idx}/{n_markets} markets", flush=True)

    if writer:
        writer.close()
    else:
        # Edge case: nothing was written — create empty Parquet with base schema
        _write_empty_filled(output_path, bucketed)

    return output_path


def _write_empty_filled(output_path: str, bucketed: pl.DataFrame) -> None:
    """Write an empty Parquet with the expected schema."""
    empty = bucketed.clear().with_columns(
        pl.lit(False).alias("is_empty_bucket"),
        pl.lit(False).alias("in_gap"),
    )
    empty.write_parquet(output_path)


# ── Streaming consecutive-gap detection ───────────────────────────────────────

def detect_consecutive_gaps(
    filled_path: str,
    gap_cfg: GapConfig,
    output_path: str = "filled_gaps.parquet",
) -> str:
    """
    Detect runs of consecutive empty buckets per market.
    Adds `in_long_gap` (bool) column for runs exceeding max_empty_buckets.

    Reads *filled_path* market-by-market via Polars lazy scan (predicate
    pushdown keeps RAM minimal).  Writes results to *output_path*.

    Returns the output file path.
    """

    output_path = str(Path(output_path))

    # Fetch only the market IDs — tiny query
    market_ids = (
        pl.scan_parquet(filled_path)
        .select("market_id")
        .unique()
        .sort("market_id")
        .collect()["market_id"]
        .to_list()
    )

    n_markets = len(market_ids)
    writer = None

    for idx, market_id in enumerate(market_ids, 1):
        market_df = (
            pl.scan_parquet(filled_path)
            .filter(pl.col("market_id") == market_id)
            .collect()
        )

        # Run-length encoding of is_empty_bucket
        empty = market_df["is_empty_bucket"].to_list()

        forward_run, backward_run = [], []
        cur = 0
        for v in empty:
            cur = cur + 1 if v else 0
            forward_run.append(cur)

        cur = 0
        for v in reversed(empty):
            cur = cur + 1 if v else 0
            backward_run.append(cur)
        backward_run.reverse()

        max_run = [max(f, b) for f, b in zip(forward_run, backward_run)]
        in_long_gap = [r > gap_cfg.max_empty_buckets for r in max_run]

        market_df = market_df.with_columns(
            pl.Series("in_long_gap", in_long_gap),
        )

        arrow_tbl = market_df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=arrow_tbl.schema,
                compression="SNAPPY",
                version="2.6",
            )
        writer.write_table(arrow_tbl)

        del market_df, arrow_tbl, empty, forward_run, backward_run, max_run, in_long_gap
        gc.collect()

        if idx % 100 == 0 or idx == n_markets:
            print(f"  detect_consecutive_gaps: {idx}/{n_markets} markets", flush=True)

    if writer:
        writer.close()
    else:
        _copy_parquet_with_col(filled_path, output_path, "in_long_gap", False)

    return output_path


# ── Streaming gap exclusion flag ──────────────────────────────────────────────

def apply_gap_exclusions(
    filled_path: str,
    gap_cfg: GapConfig,
    output_path: str = "filled_final.parquet",
) -> str:
    """
    Combine gap flags into `exclude_from_training`.
    Uses Polars sink_parquet (streaming engine) — O(1) RAM.

    Returns the output file path.
    """

    output_path = str(Path(output_path))

    (
        pl.scan_parquet(filled_path)
        .with_columns(
            (pl.col("in_gap") | pl.col("in_long_gap")).alias("exclude_from_training")
        )
        .sink_parquet(output_path)
    )

    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy_parquet_with_col(src: str, dst: str, col_name: str, default_val) -> None:
    """Copy a Parquet file and add a constant column."""
    (
        pl.scan_parquet(src)
        .with_columns(pl.lit(default_val).alias(col_name))
        .sink_parquet(dst)
    )
