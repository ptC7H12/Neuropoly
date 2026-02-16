"""
Data loading for trades and markets.
Supports CSV, Parquet, and SQLite sources.
"""

import polars as pl

from config import DataConfig


def load_trades(cfg: DataConfig) -> pl.LazyFrame:
    """Load trades data as a LazyFrame for memory efficiency."""

    if cfg.trades_format == "csv":
        lf = pl.scan_csv(
            cfg.trades_path,
            try_parse_dates=True,
            infer_schema_length=10000,
        )
    elif cfg.trades_format == "parquet":
        lf = pl.scan_parquet(cfg.trades_path)
        
    elif cfg.trades_format == "sqlite":
        db_path = cfg.sqlite_path or cfg.trades_path
        lf = pl.read_database_uri(
            f"SELECT * FROM {cfg.trades_table}",
            uri=f"sqlite:///{db_path}"
        ).lazy()

    else:
        raise ValueError(f"Unsupported trades format: {cfg.trades_format}")

    # Normalize column names and types
    lf = _normalize_trades(lf, cfg)
    return lf


def load_markets(cfg: DataConfig) -> pl.LazyFrame:
    """Load markets data as a LazyFrame."""

    if cfg.markets_format == "csv":
        lf = pl.scan_csv(
            cfg.markets_path,
            try_parse_dates=True,
            infer_schema_length=5000,
        )
    elif cfg.markets_format == "parquet":
        lf = pl.scan_parquet(cfg.markets_path)
    elif cfg.markets_format == "sqlite":
        db_path = cfg.sqlite_path or cfg.markets_path
        lf = pl.read_database_uri(
            f"SELECT * FROM {cfg.markets_table}",
            uri=f"sqlite:///{db_path}"
        ).lazy()

    else:
        raise ValueError(f"Unsupported markets format: {cfg.markets_format}")

    lf = _normalize_markets(lf)
    return lf


def _normalize_trades(lf: pl.LazyFrame, cfg: DataConfig) -> pl.LazyFrame:
    """Ensure consistent column names and types for trades."""

    lf = lf.rename(
        {
            cfg.trades_timestamp_col: "timestamp",
            cfg.trades_market_id_col: "market_id",
            cfg.trades_side_col: "side",
            cfg.trades_price_col: "price",
            cfg.trades_usd_col: "usd_amount",
            cfg.trades_token_col: "token_amount",
        }
    )

    # Parse timestamp if string
    lf = lf.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp"),
        pl.col("market_id").cast(pl.Int64).alias("market_id"),
        pl.col("price").cast(pl.Float64).alias("price"),
        pl.col("usd_amount").cast(pl.Float64).alias("usd_amount"),
        pl.col("token_amount").cast(pl.Float64).alias("token_amount"),
    )

    # Map side: token1 → 1 (YES), token2 → 0 (NO)
    lf = lf.with_columns(
        pl.when(pl.col("side") == cfg.side_yes)
        .then(pl.lit(1))
        .when(pl.col("side") == cfg.side_no)
        .then(pl.lit(0))
        .otherwise(pl.lit(-1))
        .cast(pl.Int8)
        .alias("is_yes"),
    )

    # Parse direction for buy/sell signal
    if cfg.trades_direction_col in lf.collect_schema().names():
        lf = lf.with_columns(
            pl.col(cfg.trades_direction_col)
            .str.contains("BUY")
            .cast(pl.Int8)
            .alias("is_buy"),
        )

    # Select only needed columns
    cols = ["timestamp", "market_id", "is_yes", "price", "usd_amount", "token_amount"]
    if "is_buy" in lf.collect_schema().names():
        cols.append("is_buy")

    lf = lf.select(cols).sort("timestamp")
    return lf


def _normalize_markets(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ensure consistent column names and types for markets."""

    # Handle potential column name variations
    rename_map = {}
    known_cols = lf.collect_schema().names()
    if "id" in known_cols:
        rename_map["id"] = "market_id"

    if rename_map:
        lf = lf.rename(rename_map)

    lf = lf.with_columns(
        pl.col("market_id").cast(pl.Int32),
        pl.col("yes_price").cast(pl.Float32),
        pl.col("no_price").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
        pl.col("liquidity").cast(pl.Float32),
    )

    return lf
