"""
The "one row group = one market" invariant, made explicit.

Seven places in this pipeline read a Parquet file row group by row group and
treat each row group as exactly one market.  That is what makes the streaming
design correct: every lag, rolling window and label uses `.over("market_id")`,
so a row group holding two markets would silently blend them at the seam, and
a market split across two row groups would restart its windows mid-series.

PyArrow does not guarantee it.  `ParquetWriter.write_table()` splits a table
that exceeds `row_group_size` (default ~1.05 M rows) into several row groups
— a 3 M row table becomes 3 row groups.  At 5-minute buckets that is about
ten years of a single market, so it is unlikely rather than impossible, and
the failure would be silent.

Two halves, and both are needed:

* `write_market_table()` pins `row_group_size` to the table length, so one
  write is always one row group.
* `iter_market_row_groups()` checks on the way back in that each row group
  really holds a single market, and raises with a useful message otherwise.

The check costs nothing worth measuring: the frame is already in memory, and
`n_unique()` on one market's `market_id` column is trivial next to the
feature work that follows.
"""

from __future__ import annotations

from typing import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def write_market_table(
    writer: pq.ParquetWriter,
    table: pa.Table,
) -> None:
    """
    Write one market's rows as exactly one row group.

    Without the explicit `row_group_size` PyArrow would split a large market
    across several row groups and quietly break the invariant every reader
    here relies on.
    """
    writer.write_table(table, row_group_size=max(1, table.num_rows))


def iter_market_row_groups(
    parquet_file: pq.ParquetFile,
    market_col: str = "market_id",
    validate: bool = True,
) -> Iterator[tuple[int, pl.DataFrame]]:
    """
    Yield (row_group_index, DataFrame) for each row group, one market each.

    Raises ValueError if a row group holds more than one market — that means
    the file was not written through `write_market_table`, and everything
    computed from it downstream would be wrong at the seam rather than
    obviously broken.
    """
    n_rg = parquet_file.metadata.num_row_groups

    for rg_idx in range(n_rg):
        df = pl.from_arrow(parquet_file.read_row_group(rg_idx))

        if validate and market_col in df.columns and df.height:
            n_markets = df[market_col].n_unique()
            if n_markets != 1:
                raise ValueError(
                    f"Row group {rg_idx} holds {n_markets} markets, expected 1. "
                    f"Every lag, rolling window and label in this pipeline is "
                    f"computed per market from a single row group, so this file "
                    f"would produce wrong values at the seam. Regenerate it with "
                    f"the current pipeline (writes go through "
                    f"pipeline.rowgroups.write_market_table)."
                )

        yield rg_idx, df


def assert_one_market_per_row_group(path: str, market_col: str = "market_id") -> int:
    """
    Validate a whole file and return its row-group count.

    Reads only the market id column, so it is cheap enough to run as a
    standalone check.
    """
    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    for rg_idx in range(n_rg):
        tbl = pf.read_row_group(rg_idx, columns=[market_col])
        if tbl.num_rows and pl.from_arrow(tbl)[market_col].n_unique() != 1:
            raise ValueError(
                f"{path}: row group {rg_idx} holds more than one market."
            )
    return n_rg
