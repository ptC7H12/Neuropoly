"""
Data loading for trades and markets.
Supports CSV, Parquet, and SQLite sources.
MEMORY OPTIMIZED: Uses streaming conversion for large SQLite databases.
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
        # MEMORY-OPTIMIZED: Convert to Parquet with streaming if not already done
        import sqlite3
        
        db_path = cfg.sqlite_path or cfg.trades_path
        db_path = Path(db_path)
        
        # Create parquet file path in same directory as database
        parquet_path = db_path.parent / f"{db_path.stem}_trades.parquet"
        
        if not parquet_path.exists():
            print(f"  ⚠️  Parquet file not found: {parquet_path}")
            print(f"  Converting SQLite to Parquet (one-time operation)...")
            print(f"  Using streaming mode for minimal RAM usage.")
            print()
            
            _convert_table_streaming(
                db_path=db_path,
                table_name=cfg.trades_table,
                output_path=parquet_path,
                chunk_size=2_000_000,
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
        # MEMORY-OPTIMIZED: Convert to Parquet with streaming
        import sqlite3
        
        db_path = cfg.sqlite_path or cfg.markets_path
        db_path = Path(db_path)
        
        # Create parquet file path in same directory as database
        parquet_path = db_path.parent / f"{db_path.stem}_markets.parquet"
        
        if not parquet_path.exists():
            print(f"  Converting markets table to Parquet...")
            
            _convert_table_streaming(
                db_path=db_path,
                table_name=cfg.markets_table,
                output_path=parquet_path,
                chunk_size=500_000,  # Smaller chunks for markets
            )
            
            print(f"  ✓ Markets converted: {parquet_path}")
        
        lf = pl.scan_parquet(parquet_path)

    else:
        raise ValueError(f"Unsupported markets format: {cfg.markets_format}")

    lf = _normalize_markets(lf)
    return lf


def _convert_table_streaming(db_path: Path, table_name: str, output_path: Path, chunk_size: int):
    """Convert SQLite table to Parquet using streaming (minimal RAM)."""
    import sqlite3
    
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("  ⚠️  PyArrow not found, falling back to slower method...")
        print("  Install PyArrow for better performance: pip install pyarrow")
        _convert_table_fallback(db_path, table_name, output_path, chunk_size)
        return
    
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Get total row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"  Total rows to convert: {total_rows:,}")
        
        offset = 0
        chunk_num = 0
        writer = None
        
        while offset < total_rows:
            query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
            chunk_df = pl.read_database(query, connection=conn)
            
            if chunk_df.height == 0:
                break
            
            # Convert to PyArrow and write incrementally
            arrow_table = chunk_df.to_arrow()
            
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema=arrow_table.schema,
                    compression='SNAPPY',
                    version='2.6',
                )
            
            writer.write_table(arrow_table)
            
            chunk_num += 1
            offset += chunk_df.height
            
            progress = (offset / total_rows) * 100
            print(f"    Chunk {chunk_num}: {offset:,} / {total_rows:,} rows ({progress:.1f}%)")
            
            # Clean up
            del chunk_df
            del arrow_table
            
            if offset >= total_rows:
                break
        
        if writer:
            writer.close()
            
    finally:
        conn.close()


def _convert_table_fallback(db_path: Path, table_name: str, output_path: Path, chunk_size: int):
    """Fallback conversion without PyArrow (uses more RAM but works without dependencies)."""
    import sqlite3
    
    conn = sqlite3.connect(str(db_path))
    
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"  Total rows to convert: {total_rows:,}")
        
        offset = 0
        chunk_num = 0
        chunks = []
        
        # Process in smaller batches to avoid OOM
        batch_size = min(chunk_size, 1_000_000)
        
        while offset < total_rows:
            query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
            chunk = pl.read_database(query, connection=conn)
            
            if chunk.height == 0:
                break
            
            chunks.append(chunk)
            offset += chunk.height
            chunk_num += 1
            
            # Write batch every 5 chunks to limit RAM
            if len(chunks) >= 5:
                batch_df = pl.concat(chunks, how="vertical_relaxed")
                batch_df.write_parquet(
                    output_path,
                    compression="snappy",
                    use_pyarrow=False,
                )
                chunks = []
                print(f"    Progress: {offset:,} / {total_rows:,} rows")
        
        # Write remaining chunks
        if chunks:
            batch_df = pl.concat(chunks, how="vertical_relaxed")
            batch_df.write_parquet(
                output_path,
                compression="snappy",
                use_pyarrow=False,
            )
            
    finally:
        conn.close()


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
