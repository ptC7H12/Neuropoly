#!/usr/bin/env python3
"""
SQLite to Parquet Converter - TRUE STREAMING VERSION
Converts large SQLite tables to Parquet with minimal memory usage.
Uses PyArrow for incremental Parquet writing.
"""

import sqlite3
import polars as pl
from pathlib import Path
import argparse
import time


def convert_table_to_parquet_streaming(
    db_path: str,
    table_name: str,
    output_path: str,
    chunk_size: int = 2_000_000,
    compression: str = "snappy",
):
    """
    Convert SQLite table to Parquet with TRUE streaming (no memory accumulation).
    
    Args:
        db_path: Path to SQLite database
        table_name: Name of table to export
        output_path: Path for output Parquet file
        chunk_size: Number of rows per chunk (default: 2 million for low RAM)
        compression: Compression algorithm ('snappy', 'gzip', 'zstd')
    """
    
    db_path = Path(db_path)
    output_path = Path(output_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    print(f"\n{'='*60}")
    print(f"SQLite to Parquet Conversion (STREAMING MODE)")
    print(f"{'='*60}")
    print(f"Database:    {db_path}")
    print(f"Table:       {table_name}")
    print(f"Output:      {output_path}")
    print(f"Chunk size:  {chunk_size:,} rows")
    print(f"Compression: {compression}")
    print(f"{'='*60}\n")
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Get total row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"Total rows in table: {total_rows:,}\n")
        
        if total_rows == 0:
            print("⚠️  Table is empty!")
            return
        
        # Use PyArrow for streaming write
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        offset = 0
        chunk_num = 0
        start_time = time.time()
        writer = None
        schema = None
        
        while offset < total_rows:
            chunk_start = time.time()
            
            # Read chunk from SQLite
            query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
            chunk_df = pl.read_database(query, connection=conn)
            
            if chunk_df.height == 0:
                break
            
            # Convert to PyArrow Table
            arrow_table = chunk_df.to_arrow()
            
            # Initialize writer on first chunk
            if writer is None:
                schema = arrow_table.schema
                writer = pq.ParquetWriter(
                    output_path,
                    schema=schema,
                    compression=compression.upper(),
                    version='2.6',
                )
            
            # Write chunk to Parquet file
            writer.write_table(arrow_table)
            
            chunk_num += 1
            offset += chunk_df.height
            
            # Progress report
            chunk_time = time.time() - chunk_start
            elapsed = time.time() - start_time
            progress_pct = (offset / total_rows) * 100
            rows_per_sec = offset / elapsed if elapsed > 0 else 0
            
            print(f"  Chunk {chunk_num:3d}: {offset:12,} / {total_rows:,} rows "
                  f"({progress_pct:5.1f}%) | "
                  f"{chunk_time:5.1f}s | "
                  f"{rows_per_sec:8,.0f} rows/s")
            
            # Clear chunk from memory
            del chunk_df
            del arrow_table
            
            if offset >= total_rows:
                break
        
        # Close writer
        if writer:
            writer.close()
        
        # Final statistics
        elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"✓ Conversion complete!")
        print(f"{'='*60}")
        print(f"Total time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Rows processed: {offset:,}")
        print(f"Output size:    {file_size_mb:,.1f} MB")
        print(f"Avg speed:      {offset/elapsed:,.0f} rows/s")
        print(f"Compression:    {compression}")
        print(f"{'='*60}\n")
        
    except ImportError:
        print("\n❌ PyArrow not installed!")
        print("Install with: pip install pyarrow")
        raise
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert SQLite table to Parquet format (streaming mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (2M chunk size, snappy compression)
  python convert_to_parquet.py trades.db trades trades.parquet
  
  # Smaller chunks for very low RAM systems
  python convert_to_parquet.py trades.db trades trades.parquet --chunk-size 1000000
  
  # Different compression (snappy is fastest, zstd is smallest)
  python convert_to_parquet.py trades.db trades trades.parquet --compression zstd
  
  # Convert markets table
  python convert_to_parquet.py trades.db markets markets.parquet
  
Compression comparison:
  - snappy: Fastest, larger files, good for speed
  - gzip:   Medium speed, medium compression
  - zstd:   Slower, best compression, good for storage
        """
    )
    
    parser.add_argument("database", help="Path to SQLite database file")
    parser.add_argument("table", help="Name of table to convert")
    parser.add_argument("output", help="Output Parquet file path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Number of rows per chunk (default: 2,000,000 for low RAM)"
    )
    parser.add_argument(
        "--compression",
        choices=["snappy", "gzip", "zstd"],
        default="snappy",
        help="Compression algorithm (default: snappy - fastest)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_table_to_parquet_streaming(
            db_path=args.database,
            table_name=args.table,
            output_path=args.output,
            chunk_size=args.chunk_size,
            compression=args.compression,
        )
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
