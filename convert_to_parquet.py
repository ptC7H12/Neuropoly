#!/usr/bin/env python3
"""
SQLite to Parquet Converter
Converts large SQLite tables to Parquet format with chunked processing.
"""

import sqlite3
import polars as pl
from pathlib import Path
import argparse
import time


def convert_table_to_parquet(
    db_path: str,
    table_name: str,
    output_path: str,
    chunk_size: int = 5_000_000,
    compression: str = "zstd",
    compression_level: int = 3,
):
    """
    Convert SQLite table to Parquet with memory-efficient chunked processing.
    
    Args:
        db_path: Path to SQLite database
        table_name: Name of table to export
        output_path: Path for output Parquet file
        chunk_size: Number of rows per chunk (default: 5 million)
        compression: Compression algorithm ('zstd', 'snappy', 'gzip', 'lz4')
        compression_level: Compression level (1-22 for zstd)
    """
    
    db_path = Path(db_path)
    output_path = Path(output_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    print(f"\n{'='*60}")
    print(f"SQLite to Parquet Conversion")
    print(f"{'='*60}")
    print(f"Database:    {db_path}")
    print(f"Table:       {table_name}")
    print(f"Output:      {output_path}")
    print(f"Chunk size:  {chunk_size:,} rows")
    print(f"Compression: {compression} (level {compression_level})")
    print(f"{'='*60}\n")
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Get total row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"Total rows in table: {total_rows:,}\n")
        
        # Check if table has data
        if total_rows == 0:
            print("⚠️  Table is empty!")
            return
        
        # Collect all chunks first, then concatenate and write
        chunks = []
        offset = 0
        chunk_num = 0
        start_time = time.time()
        
        while offset < total_rows:
            chunk_start = time.time()
            
            # Read chunk
            query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
            chunk_df = pl.read_database(query, connection=conn)
            
            if chunk_df.height == 0:
                break
            
            chunks.append(chunk_df)
            
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
            
            if chunk_df.height < chunk_size:
                break
        
        # Concatenate all chunks and write to single Parquet file
        print(f"\n  Concatenating {len(chunks)} chunks and writing to Parquet...")
        write_start = time.time()
        
        full_df = pl.concat(chunks, how="vertical_relaxed")
        full_df.write_parquet(
            output_path,
            compression=compression,
            compression_level=compression_level,
            statistics=True,
            row_group_size=100_000,
            use_pyarrow=False,
        )
        
        write_time = time.time() - write_start
        
        # Final statistics
        elapsed = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"✓ Conversion complete!")
        print(f"{'='*60}")
        print(f"Total time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Write time:     {write_time:.1f}s")
        print(f"Rows processed: {offset:,}")
        print(f"Output size:    {file_size_mb:,.1f} MB")
        print(f"Avg speed:      {offset/elapsed:,.0f} rows/s")
        print(f"{'='*60}\n")
        
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert SQLite table to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert trades table with defaults
  python convert_to_parquet.py trades.db trades trades.parquet
  
  # Custom chunk size and compression
  python convert_to_parquet.py trades.db trades trades.parquet --chunk-size 10000000 --compression snappy
  
  # Convert markets table
  python convert_to_parquet.py trades.db markets markets.parquet --chunk-size 1000000
        """
    )
    
    parser.add_argument("database", help="Path to SQLite database file")
    parser.add_argument("table", help="Name of table to convert")
    parser.add_argument("output", help="Output Parquet file path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Number of rows per chunk (default: 5,000,000)"
    )
    parser.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip", "lz4", "uncompressed"],
        default="zstd",
        help="Compression algorithm (default: zstd)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Compression level, 1-22 for zstd (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_table_to_parquet(
            db_path=args.database,
            table_name=args.table,
            output_path=args.output,
            chunk_size=args.chunk_size,
            compression=args.compression,
            compression_level=args.compression_level,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
