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

from pipeline.rowgroups import (
    iter_market_row_groups,
    write_market_table,
)

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

def fill_market_buckets(
    market_df: pl.DataFrame,
    bucket_cfg: BucketConfig,
    gap_cfg: GapConfig,
) -> pl.DataFrame:
    """
    Complete one market's bucket series: every bucket between its first and
    last observation exists, missing ones are filled and flagged.

    Shared with the live path (pipeline/live_features) so that rolling
    windows cover the same wall-clock span at inference as during training.
    Without this, `price_ma48` means "last 4 hours" in training but "last 48
    buckets that happened to have trades" live — which can span days.
    """
    bucket_minutes = bucket_cfg.bucket_minutes

    start = market_df["bucket_time"].min()
    end = market_df["bucket_time"].max()
    if start is None or end is None:
        return market_df

    market_id = market_df["market_id"][0]

    # Generate complete bucket range for this market
    n_buckets = int((end - start).total_seconds() / (bucket_minutes * 60)) + 1
    full_range = pl.DataFrame(
        {
            "bucket_time": pl.datetime_range(
                start, end, timedelta(minutes=bucket_minutes), eager=True
            ),
            "market_id": [market_id] * n_buckets,
        }
    ).with_columns(pl.col("market_id").cast(market_df.schema["market_id"]))

    # Left join: keep all buckets, fill missing with null
    merged = full_range.join(market_df, on=["market_id", "bucket_time"], how="left")

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
    usd_cols = ["total_usd", "total_tokens", "whale_usd"]
    price_cols = [
        "open_price", "close_price", "high_price",
        "low_price", "mean_price", "vwap", "momentum",
    ]

    fills = []
    for col in count_cols:
        if col in merged.columns:
            fills.append(pl.col(col).fill_null(0))
    for col in usd_cols:
        if col in merged.columns:
            fills.append(pl.col(col).fill_null(0.0))
    # Forward-fill price columns from last known value
    for col in price_cols:
        if col in merged.columns:
            fills.append(pl.col(col).forward_fill().alias(col))
    # Neutral fill for ratio columns
    for col in ["yes_ratio", "buy_ratio"]:
        if col in merged.columns:
            fills.append(pl.col(col).fill_null(0.5))

    return merged.with_columns(fills) if fills else merged


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

    output_path = str(Path(output_path))

    # partition_by splits the frame in ONE pass.  The previous version ran
    # `bucketed.filter(market_id == m)` inside the loop, i.e. a full scan of
    # the whole frame per market — O(markets x rows).  Measured at constant
    # row count: 50 markets 1.0 s, 200 markets 3.2 s, 800 markets 12.1 s.
    partitions = bucketed.partition_by("market_id", maintain_order=True)
    n_markets = len(partitions)
    writer = None

    for idx, market_df in enumerate(partitions, 1):
        merged = fill_market_buckets(market_df, bucket_cfg, gap_cfg)

        # ── stream-write this market's rows to Parquet ──
        arrow_tbl = merged.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=arrow_tbl.schema,
                compression="SNAPPY",
                version="2.6",
            )
        write_market_table(writer, arrow_tbl)

        # Release this partition's memory as we go
        partitions[idx - 1] = None
        del merged, arrow_tbl, market_df
        if idx % 500 == 0:
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

    Reads *filled_path* row-group by row-group (fill_buckets writes exactly
    one row group per market), so the entire file is scanned **once** in
    O(1) RAM — no repeated full-file scans.

    Returns the output file path.
    """

    output_path = str(Path(output_path))

    pf = pq.ParquetFile(filled_path)
    n_rg = pf.metadata.num_row_groups

    writer = None

    # One row group = one market; the generator enforces it rather than
    # assuming it (see pipeline/rowgroups).
    for rg_idx, market_df in iter_market_row_groups(pf):

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

        max_run    = [max(f, b) for f, b in zip(forward_run, backward_run)]
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
        write_market_table(writer, arrow_tbl)

        del market_df, arrow_tbl, empty, forward_run, backward_run, max_run, in_long_gap
        if (rg_idx + 1) % 500 == 0:
            gc.collect()

        if (rg_idx + 1) % 100 == 0 or (rg_idx + 1) == n_rg:
            print(f"  detect_consecutive_gaps: {rg_idx + 1}/{n_rg} markets", flush=True)

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
    Reads *filled_path* row-group by row-group (one market per row group,
    guaranteed by detect_consecutive_gaps) — O(1) RAM.

    Returns the output file path.
    """

    output_path = str(Path(output_path))

    pf = pq.ParquetFile(filled_path)
    n_rg = pf.metadata.num_row_groups
    writer = None

    for rg_idx, market_df in iter_market_row_groups(pf):
        market_df = market_df.with_columns(
            (pl.col("in_gap") | pl.col("in_long_gap")).alias("exclude_from_training")
        )

        arrow_tbl = market_df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=arrow_tbl.schema,
                compression="SNAPPY",
                version="2.6",
            )
        write_market_table(writer, arrow_tbl)

        del market_df, arrow_tbl
        if (rg_idx + 1) % 500 == 0:
            gc.collect()

    if writer:
        writer.close()
    else:
        _copy_parquet_with_col(filled_path, output_path, "exclude_from_training", False)

    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy_parquet_with_col(src: str, dst: str, col_name: str, default_val) -> None:
    """Copy a Parquet file and add a constant column."""
    (
        pl.scan_parquet(src)
        .with_columns(pl.lit(default_val).alias(col_name))
        .sink_parquet(dst)
    )
