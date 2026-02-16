# Einmaliges Konvertierungsskript
import sqlite3
import polars as pl
from pathlib import Path

db_path = "your_database.db"
table_name = "trades"
output_parquet = "trades.parquet"

conn = sqlite3.connect(db_path)

# Chunked export
chunk_size = 5_000_000
offset = 0
first = True

while True:
    query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
    chunk = pl.read_database(query, connection=conn)
    
    if chunk.height == 0:
        break
    
    chunk.write_parquet(
        output_parquet,
        compression="zstd",  # Bessere Kompression als snappy
        compression_level=3,
        statistics=True,
        row_group_size=100_000,
        mode="append" if not first else "overwrite"
    )
    
    print(f"Exported {offset:,} - {offset + chunk.height:,} rows")
    offset += chunk.size
    first = False
    
    if chunk.height < chunk_size:
        break

conn.close()
