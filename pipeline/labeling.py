"""
Label generation for the trading pipeline.

Binary label:  win ∈ {0, 1}
- YES side: win=1 if future_price > entry_price + min_move
- NO side:  win=1 if future_price < entry_price - min_move

Regression target: future_return = (future_price - entry_price) / entry_price
"""

import gc
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from config import LabelConfig


def add_labels(
    df: pl.DataFrame,
    cfg: LabelConfig,
) -> pl.DataFrame:
    """
    Add win label and optional regression target.

    Uses mean_price of the current bucket as entry price.
    Uses mean_price N buckets forward as future price.

    Buckets marked `exclude_from_training` or `is_empty_bucket`
    will NOT receive labels (set to null).
    """

    # Compute future price: mean_price shifted backward by forward_window
    # (shift(-N) looks N rows into the future)
    df = df.with_columns(
        pl.col("mean_price")
        .shift(-cfg.forward_window_buckets)
        .over("market_id")
        .alias("future_price"),
    )

    # Compute future return
    df = df.with_columns(
        (
            (pl.col("future_price") - pl.col("mean_price"))
            / pl.col("mean_price")
        )
        .fill_nan(None)
        .alias("future_return"),
    )

    # Binary label based on yes_ratio (majority side in bucket)
    # If yes_ratio > 0.5 → bucket is predominantly YES → win if price goes up
    # If yes_ratio <= 0.5 → bucket is predominantly NO → win if price goes down
    df = df.with_columns(
        pl.when(pl.col("yes_ratio") > 0.5)
        # YES-dominated bucket: win if future price rises
        .then(
            pl.when(
                pl.col("future_price") > pl.col("mean_price") + cfg.min_price_move
            )
            .then(pl.lit(1))
            .when(
                pl.col("future_price") < pl.col("mean_price") - cfg.min_price_move
            )
            .then(pl.lit(0))
            .otherwise(pl.lit(None))  # Price didn't move enough → ambiguous
        )
        # NO-dominated bucket: win if future price drops
        .otherwise(
            pl.when(
                pl.col("future_price") < pl.col("mean_price") - cfg.min_price_move
            )
            .then(pl.lit(1))
            .when(
                pl.col("future_price") > pl.col("mean_price") + cfg.min_price_move
            )
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
        )
        .alias("win"),
    )

    # Nullify labels for excluded or empty buckets
    if "exclude_from_training" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("exclude_from_training"))
            .then(pl.lit(None))
            .otherwise(pl.col("win"))
            .alias("win"),
            pl.when(pl.col("exclude_from_training"))
            .then(pl.lit(None))
            .otherwise(pl.col("future_return"))
            .alias("future_return"),
        )

    # Nullify labels for empty buckets (no real trades → no real entry)
    if "is_empty_bucket" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("is_empty_bucket"))
            .then(pl.lit(None))
            .otherwise(pl.col("win"))
            .alias("win"),
            pl.when(pl.col("is_empty_bucket"))
            .then(pl.lit(None))
            .otherwise(pl.col("future_return"))
            .alias("future_return"),
        )

    # Cast win to Int8 (nullable)
    df = df.with_columns(pl.col("win").cast(pl.Int8))

    return df


def label_stats(df: pl.DataFrame) -> dict:
    """Return summary statistics about labels."""

    labeled = df.filter(pl.col("win").is_not_null())
    total = len(labeled)

    if total == 0:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

    wins = labeled.filter(pl.col("win") == 1).height
    losses = labeled.filter(pl.col("win") == 0).height
    nullified = df.filter(pl.col("win").is_null()).height

    return {
        "total_rows": len(df),
        "labeled": total,
        "nullified": nullified,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0.0,
        "mean_future_return": (
            labeled["future_return"].mean() if "future_return" in labeled.columns else None
        ),
        "std_future_return": (
            labeled["future_return"].std() if "future_return" in labeled.columns else None
        ),
    }


def add_labels_streaming(
    features_path: str,
    cfg: LabelConfig,
    output_path: str = "labeled.parquet",
) -> str:
    """
    Add win/future_return labels one market at a time from a features Parquet
    file.  Peak RAM = one market's rows (same row-group-per-market invariant
    guaranteed by build_features_streaming).

    Returns the output file path.
    """

    output_path = str(Path(output_path))

    pf = pq.ParquetFile(features_path)
    n_rg = pf.metadata.num_row_groups
    writer = None

    for rg_idx in range(n_rg):
        market_df = pl.from_arrow(pf.read_row_group(rg_idx))

        labeled_df = add_labels(market_df, cfg)
        del market_df

        arrow_tbl = labeled_df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=arrow_tbl.schema,
                compression="SNAPPY",
                version="2.6",
            )
        writer.write_table(arrow_tbl)

        del labeled_df, arrow_tbl
        gc.collect()

    if writer:
        writer.close()
    else:
        pl.scan_parquet(features_path).collect().write_parquet(output_path)

    return output_path


def label_stats_lazy(labeled_path: str) -> dict:
    """
    Compute label statistics from a labeled Parquet file without loading
    it fully into RAM.  Uses a single lazy aggregation pass.
    """

    lf = pl.scan_parquet(labeled_path)
    schema_names = lf.collect_schema().names()
    has_future_return = "future_return" in schema_names

    agg_exprs = [
        pl.len().alias("n_total"),
        pl.col("win").is_not_null().sum().alias("n_labeled"),
        (pl.col("win") == 1).sum().alias("wins"),
        (pl.col("win") == 0).sum().alias("losses"),
    ]
    if has_future_return:
        agg_exprs += [
            pl.col("future_return").mean().alias("mean_future_return"),
            pl.col("future_return").std().alias("std_future_return"),
        ]

    row = lf.select(agg_exprs).collect()

    n_total   = int(row["n_total"][0])
    n_labeled = int(row["n_labeled"][0])
    wins      = int(row["wins"][0])
    losses    = int(row["losses"][0])

    result = {
        "total_rows": n_total,
        "labeled":    n_labeled,
        "nullified":  n_total - n_labeled,
        "wins":       wins,
        "losses":     losses,
        "win_rate":   wins / n_labeled if n_labeled > 0 else 0.0,
    }
    if has_future_return:
        result["mean_future_return"] = row["mean_future_return"][0]
        result["std_future_return"]  = row["std_future_return"][0]

    return result
