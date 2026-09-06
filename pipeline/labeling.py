"""
Label generation for the trading pipeline.

Binary label:  win ∈ {0, 1}
- YES side: win=1 if future_price > entry_price + min_move
- NO side:  win=1 if future_price < entry_price - min_move

Regression target: future_return = (future_price - entry_price) / entry_price

Realised trade returns (used by the backtest — NOT features):
- trade_return      = return of holding the DOMINANT side of the bucket
                      for `forward_window_buckets` buckets
- trade_return_opp  = return of holding the OPPOSITE side instead

`win` only says whether the price moved the right way; it says nothing about
*how far*.  A share bought at 0.50 that moves to 0.501 is a win, but it pays
0.2 %, not 100 %.  The backtest therefore has to use trade_return, not `win`.
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

    # ── Realised trade returns ───────────────────────────────────────────
    #
    # Entry at mean_price of the current bucket, exit at mean_price of the
    # bucket `forward_window_buckets` ahead.  A YES share costs `p`, a NO
    # share costs `1 - p`, so the two sides have different denominators:
    #
    #   YES: buy at p,      sell at p'      → (p' - p) / p
    #   NO : buy at (1-p),  sell at (1-p')  → (p - p') / (1 - p)
    #
    # trade_return     → betting WITH the dominant bucket side (what `win` scores)
    # trade_return_opp → betting AGAINST it (the complementary token)
    _entry = pl.col("mean_price").clip(1e-6, 1.0 - 1e-6)
    _exit = pl.col("future_price")
    _ret_yes = (_exit - _entry) / _entry
    _ret_no = (_entry - _exit) / (1.0 - _entry)
    _yes_dominant = pl.col("yes_ratio") > 0.5

    df = df.with_columns(
        pl.when(_yes_dominant)
        .then(_ret_yes)
        .otherwise(_ret_no)
        .fill_nan(None)
        .alias("trade_return"),
        pl.when(_yes_dominant)
        .then(_ret_no)
        .otherwise(_ret_yes)
        .fill_nan(None)
        .alias("trade_return_opp"),
        # Price of the token actually held.  The backtest needs it because
        # trading costs are quoted in absolute price units, so their share of
        # a position scales with 1/price (see config.CostConfig).
        pl.when(_yes_dominant)
        .then(_entry)
        .otherwise(1.0 - _entry)
        .alias("entry_token_price"),
        pl.when(_yes_dominant)
        .then(1.0 - _entry)
        .otherwise(_entry)
        .alias("entry_token_price_opp"),
    )

    # Nullify targets for excluded or empty buckets.
    # Excluded: inside a known data gap.  Empty: no real trades in the bucket,
    # so mean_price is only a forward-filled carry-over — no real entry price.
    _targets = [
        "win", "future_return", "trade_return", "trade_return_opp",
        "entry_token_price", "entry_token_price_opp",
    ]
    for flag in ("exclude_from_training", "is_empty_bucket"):
        if flag not in df.columns:
            continue
        df = df.with_columns(
            [
                pl.when(pl.col(flag))
                .then(pl.lit(None))
                .otherwise(pl.col(col))
                .alias(col)
                for col in _targets
            ]
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
        "mean_trade_return": (
            labeled["trade_return"].mean() if "trade_return" in labeled.columns else None
        ),
    }


def add_labels_streaming(
    features_path: str,
    cfg: LabelConfig,
    output_path: str = "labeled.parquet",
    batch_markets: int = 100,
) -> str:
    """
    Add win / future_return / trade_return labels from a features Parquet
    file, `batch_markets` markets at a time.

    Every label expression is market-aware (shift(-N).over("market_id") or
    row-wise), so a batch yields exactly the same values as one market at a
    time while amortising Polars' per-call overhead.  Peak RAM is
    `batch_markets` markets' rows; pass 1 to process strictly one at a time.

    Returns the output file path.
    """

    output_path = str(Path(output_path))

    pf = pq.ParquetFile(features_path)
    n_rg = pf.metadata.num_row_groups
    writer = None
    batch_markets = max(1, batch_markets)

    for batch_start in range(0, n_rg, batch_markets):
        group_ids = list(range(batch_start, min(batch_start + batch_markets, n_rg)))
        batch_df = pl.from_arrow(pf.read_row_groups(group_ids))

        labeled_df = add_labels(batch_df, cfg)
        del batch_df

        # Preserve one row group per market
        for part in labeled_df.partition_by("market_id", maintain_order=True):
            arrow_tbl = part.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema=arrow_tbl.schema,
                    compression="SNAPPY",
                    version="2.6",
                )
            writer.write_table(arrow_tbl)
            del arrow_tbl, part

        del labeled_df
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
    has_trade_return = "trade_return" in schema_names
    if has_trade_return:
        agg_exprs.append(pl.col("trade_return").mean().alias("mean_trade_return"))

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
    if has_trade_return:
        result["mean_trade_return"] = row["mean_trade_return"][0]

    return result
