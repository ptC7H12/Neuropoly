"""
Data loading for trades and markets.
Supports CSV, Parquet, and SQLite sources.
"""

import polars as pl
from pathlib import Path

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
        # MEMORY-OPTIMIZED: Convert to Parquet if not already done
        import sqlite3
        
        db_path = cfg.sqlite_path or cfg.trades_path
        db_path = Path(db_path)
        
        # Create parquet file path in same directory as database
        parquet_path = db_path.parent / f"{db_path.stem}_trades.parquet"
        
        if not parquet_path.exists():
            print(f"  ⚠️  Parquet file not found: {parquet_path}")
            print(f"  Converting SQLite to Parquet (one-time operation)...")
            print(f"  This will improve memory efficiency for future runs.")
            print()
            
            conn = sqlite3.connect(str(db_path))
            
            # Get total row count for progress tracking
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {cfg.trades_table}")
                total_rows = cursor.fetchone()[0]
                print(f"  Total rows to convert: {total_rows:,}")
            except Exception:
                total_rows = None
            
            # Read in chunks, collect them, then write once
            chunk_size = 5_000_000  # 5M rows per chunk
            offset = 0
            chunk_num = 0
            chunks = []
            
            while True:
                query = f"SELECT * FROM {cfg.trades_table} LIMIT {chunk_size} OFFSET {offset}"
                chunk = pl.read_database(query, connection=conn)
                
                if chunk.height == 0:
                    break
                
                chunks.append(chunk)
                
                chunk_num += 1
                offset += chunk.height
                
                if total_rows:
                    progress = (offset / total_rows) * 100
                    print(f"    Chunk {chunk_num}: {offset:,} / {total_rows:,} rows ({progress:.1f}%)")
                else:
                    print(f"    Chunk {chunk_num}: {offset:,} rows processed")
                
                if chunk.height < chunk_size:
                    break
            
            conn.close()
            
            # Concatenate and write to Parquet
            print(f"  Writing {len(chunks)} chunks to Parquet...")
            full_df = pl.concat(chunks, how="vertical_relaxed")
            full_df.write_parquet(
                parquet_path,
                compression="zstd",
                compression_level=3,
                statistics=True,
                row_group_size=100_000,
                use_pyarrow=False,
            )
            
            file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Conversion complete! Output: {parquet_path} ({file_size_mb:.1f} MB)")
            print()
        
        # Load from Parquet (memory-efficient lazy loading)
        lf = pl.scan_parquet(parquet_path)

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
        # MEMORY-OPTIMIZED: Convert to Parquet if not already done
        import sqlite3
        
        db_path = cfg.sqlite_path or cfg.markets_path
        db_path = Path(db_path)
        
        # Create parquet file path in same directory as database
        parquet_path = db_path.parent / f"{db_path.stem}_markets.parquet"
        
        if not parquet_path.exists():
            print(f"  Converting markets table to Parquet...")
            
            conn = sqlite3.connect(str(db_path))
            
            # Markets table is typically smaller, can load in larger chunks
            chunk_size = 1_000_000
            offset = 0
            chunks = []
            
            while True:
                query = f"SELECT * FROM {cfg.markets_table} LIMIT {chunk_size} OFFSET {offset}"
                chunk = pl.read_database(query, connection=conn)
                
                if chunk.height == 0:
                    break
                
                chunks.append(chunk)
                offset += chunk.height
                
                if chunk.height < chunk_size:
                    break
            
            conn.close()
            
            # Concatenate and write
            if chunks:
                full_df = pl.concat(chunks, how="vertical_relaxed")
                full_df.write_parquet(
                    parquet_path,
                    compression="zstd",
                    compression_level=3,
                    use_pyarrow=False,
                )
            
            print(f"  ✓ Markets converted: {parquet_path}")
        
        lf = pl.scan_parquet(parquet_path)

    else:
        raise ValueError(f"Unsupported markets format: {cfg.markets_format}")

    lf = _normalize_markets(lf)
    return lf


def _normalize_trades(lf: pl.LazyFrame, cfg: DataConfig) -> pl.LazyFrame:
    """Ensure consistent column names and types for trades."""

    # Rename to standard names
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

    # Optional direction column
    if hasattr(cfg, 'trades_direction_col') and cfg.trades_direction_col:
        lf = lf.rename({cfg.trades_direction_col: "direction"})

    # Cast to appropriate types (use Float32/Int32 for memory efficiency)
    lf = lf.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col("market_id").cast(pl.Int32),
        pl.col("price").cast(pl.Float32),
        pl.col("usd_amount").cast(pl.Float32),
        pl.col("token_amount").cast(pl.Float32),
    )

    return lf


def _normalize_markets(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Ensure consistent types for markets data."""
    
    # Cast to appropriate types (use Float32/Int32 for memory efficiency)
    lf = lf.with_columns(
        pl.col("market_id").cast(pl.Int32),
        pl.col("yes_price").cast(pl.Float32),
        pl.col("no_price").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
        pl.col("liquidity").cast(pl.Float32),
    )

    return lf
