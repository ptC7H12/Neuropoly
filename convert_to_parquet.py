#!/usr/bin/env python3
"""
CSV to Parquet Converter — RAM-Efficient Streaming
===================================================
Converts markets.csv and orderFilled.csv to Parquet format in chunks.
No full-file loads — only chunk_size rows live in RAM at any time.

Usage:
  # Convert orderFilled trades CSV (joins with markets for market_id/side)
  python convert_to_parquet.py trades orderFilled.csv trades.parquet --markets markets.csv

  # Convert markets CSV
  python convert_to_parquet.py markets markets.csv markets.parquet

  # Smaller chunks for very low RAM (default: 500_000 rows)
  python convert_to_parquet.py trades orderFilled.csv trades.parquet \\
      --markets markets.csv --chunk-size 100000

  # Different compression
  python convert_to_parquet.py trades orderFilled.csv trades.parquet \\
      --markets markets.csv --compression zstd
"""

import csv
import time
from pathlib import Path
import argparse
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_lines(filepath: Path) -> int:
    """Count data rows (excluding header) without loading file into RAM."""
    count = 0
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for _ in reader:
            count += 1
    return count


def _iter_csv_chunks(filepath: Path, chunk_size: int):
    """
    Yield chunks of rows as list[dict] from a CSV file.
    Uses csv.DictReader for true streaming — only chunk_size rows in RAM.
    """
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        chunk: list[dict] = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _write_chunk(writer_ref: list, output_path: Path, arrow_table: pa.Table,
                 compression: str):
    """Initialize ParquetWriter on first call, then write table chunk."""
    if writer_ref[0] is None:
        writer_ref[0] = pq.ParquetWriter(
            output_path,
            schema=arrow_table.schema,
            compression=compression.upper(),
            version="2.6",
        )
    writer_ref[0].write_table(arrow_table)


def _progress(chunk_num: int, total_rows: int, estimated_total: int,
               chunk_time: float, start_time: float):
    elapsed = time.time() - start_time
    speed = total_rows / elapsed if elapsed > 0 else 0
    pct = f"{(total_rows / estimated_total * 100):5.1f}%" if estimated_total else "  ?%"
    print(
        f"  Chunk {chunk_num:4d}: {total_rows:12,} / "
        f"{estimated_total:,} ({pct}) | {chunk_time:5.1f}s | {speed:9,.0f} rows/s"
    )


# ── Markets conversion ─────────────────────────────────────────────────────────

# PyArrow schema for the normalized markets Parquet
MARKETS_SCHEMA = pa.schema([
    pa.field("market_id",   pa.int32()),
    pa.field("question",    pa.string()),
    pa.field("answer1",     pa.string()),
    pa.field("answer2",     pa.string()),
    pa.field("volume",      pa.float32()),
    pa.field("close_time",  pa.timestamp("us", tz="UTC")),
    pa.field("market_slug", pa.string()),
    pa.field("ticker",      pa.string()),
    pa.field("token1",      pa.string()),
    pa.field("token2",      pa.string()),
    # These are not in markets.csv — will be filled as null; LightGBM handles nulls
    pa.field("yes_price",   pa.float32()),
    pa.field("no_price",    pa.float32()),
    pa.field("liquidity",   pa.float32()),
])


def _parse_datetime_utc(value: str) -> int | None:
    """
    Parse ISO-8601 or 'YYYY-MM-DD HH:MM:SS+00' strings to microseconds since epoch.
    Returns None for empty/invalid values (PyArrow will store as null).
    """
    if not value or value.strip() == "":
        return None
    value = value.strip()
    # Try common formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S+00",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000)
        except ValueError:
            continue
    return None


def _safe_float32(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int32(value: str) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def convert_markets_csv(
    input_path: str,
    output_path: str,
    chunk_size: int = 500_000,
    compression: str = "snappy",
):
    """
    Convert markets.csv → markets.parquet in RAM-efficient chunks.

    Column mapping:
        id          → market_id (Int32)
        question    → question
        answer1     → answer1
        answer2     → answer2
        volume      → volume (Float32)
        closedTime  → close_time (Timestamp UTC)
        market_slug → market_slug
        ticker      → ticker
        token1      → token1  (used for trades join)
        token2      → token2  (used for trades join)
        [new]       → yes_price, no_price, liquidity (null — not in source)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"\n{'='*65}")
    print("  markets.csv → Parquet (STREAMING MODE)")
    print(f"{'='*65}")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Chunk size:  {chunk_size:,}")
    print(f"  Compression: {compression}")
    print(f"{'='*65}\n")

    print("  Counting rows …", end="", flush=True)
    estimated_total = _count_lines(input_path)
    print(f" {estimated_total:,}")

    writer_ref = [None]
    total_rows = 0
    chunk_num = 0
    start_time = time.time()

    for chunk in _iter_csv_chunks(input_path, chunk_size):
        chunk_start = time.time()

        market_ids  = []
        questions   = []
        answer1s    = []
        answer2s    = []
        volumes     = []
        close_times = []
        slugs       = []
        tickers     = []
        token1s     = []
        token2s     = []

        for row in chunk:
            market_ids.append(_safe_int32(row.get("id", "")))
            questions.append(row.get("question", "") or "")
            answer1s.append(row.get("answer1", "") or "")
            answer2s.append(row.get("answer2", "") or "")
            volumes.append(_safe_float32(row.get("volume", "")))
            close_times.append(_parse_datetime_utc(row.get("closedTime", "")))
            slugs.append(row.get("market_slug", "") or "")
            tickers.append(row.get("ticker", "") or "")
            token1s.append(row.get("token1", "") or "")
            token2s.append(row.get("token2", "") or "")

        n = len(chunk)
        arrow_table = pa.table(
            {
                "market_id":  pa.array(market_ids,  type=pa.int32()),
                "question":   pa.array(questions,   type=pa.string()),
                "answer1":    pa.array(answer1s,    type=pa.string()),
                "answer2":    pa.array(answer2s,    type=pa.string()),
                "volume":     pa.array(volumes,     type=pa.float32()),
                "close_time": pa.array(close_times, type=pa.timestamp("us", tz="UTC")),
                "market_slug":pa.array(slugs,       type=pa.string()),
                "ticker":     pa.array(tickers,     type=pa.string()),
                "token1":     pa.array(token1s,     type=pa.string()),
                "token2":     pa.array(token2s,     type=pa.string()),
                "yes_price":  pa.array([None] * n,  type=pa.float32()),
                "no_price":   pa.array([None] * n,  type=pa.float32()),
                "liquidity":  pa.array([None] * n,  type=pa.float32()),
            }
        )

        _write_chunk(writer_ref, output_path, arrow_table, compression)

        chunk_num += 1
        total_rows += n
        _progress(chunk_num, total_rows, estimated_total,
                  time.time() - chunk_start, start_time)

    if writer_ref[0]:
        writer_ref[0].close()

    elapsed = time.time() - start_time
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\n{'='*65}")
    print(f"  Conversion complete!")
    print(f"  Rows:  {total_rows:,} | Size: {size_mb:.1f} MB | Time: {elapsed:.1f}s")
    print(f"{'='*65}\n")


# ── Trades (orderFilled) conversion ───────────────────────────────────────────

# PyArrow schema for normalized trades Parquet
TRADES_SCHEMA = pa.schema([
    pa.field("timestamp",    pa.timestamp("us", tz="UTC")),
    pa.field("market_id",    pa.int64()),
    pa.field("side",         pa.string()),   # "token1" (YES) or "token2" (NO)
    pa.field("price",        pa.float64()),  # USDC per conditional token [0, 1]
    pa.field("usd_amount",   pa.float64()),  # USDC value (6-decimal adjusted)
    pa.field("token_amount", pa.float64()),  # Token quantity (6-decimal adjusted)
    pa.field("direction",    pa.string()),   # "BUY" or "SELL" from maker's view
    pa.field("tx_hash",      pa.string()),
])

# USDC / CTF tokens both use 6 decimals on Polygon
_USDC_DECIMALS = 1_000_000


def _build_token_lookup(markets_path: str) -> dict[str, tuple[int, str]]:
    """
    Read markets.csv and build a dict:
        token_id (str) → (market_id: int, side: str)  # side ∈ {"token1", "token2"}

    Loads entire markets.csv into a dict — this is fine since markets are small
    (typically < 20k rows) compared to hundreds of millions of trade rows.
    """
    lookup: dict[str, tuple[int, str]] = {}
    with open(markets_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = _safe_int32(row.get("id", ""))
            if market_id is None:
                continue
            t1 = (row.get("token1") or "").strip()
            t2 = (row.get("token2") or "").strip()
            if t1:
                lookup[t1] = (market_id, "token1")
            if t2:
                lookup[t2] = (market_id, "token2")
    return lookup


def convert_trades_csv(
    input_path: str,
    output_path: str,
    markets_path: str,
    chunk_size: int = 500_000,
    compression: str = "snappy",
):
    """
    Convert orderFilled.csv → trades.parquet in RAM-efficient chunks.

    orderFilled.csv schema:
        timestamp          — Unix timestamp (integer seconds)
        maker              — Ethereum address (maker)
        makerAssetId       — Token ID or "0" for USDC
        makerAmountFilled  — Amount (6 decimal units)
        taker              — Ethereum address (taker)
        takerAssetId       — Token ID or "0" for USDC
        takerAmountFilled  — Amount (6 decimal units)
        transactionHash    — On-chain tx hash

    Each trade appears TWICE (maker-side and taker-side).
    We keep only rows where makerAssetId != "0"
    (= the token-seller leg, which gives price = taker_USDC / maker_tokens).

    Output schema matches pipeline/data_loader.py expectations:
        timestamp, market_id, side, price, usd_amount, token_amount, direction, tx_hash
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"\n{'='*65}")
    print("  orderFilled.csv → Parquet (STREAMING MODE)")
    print(f"{'='*65}")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Markets:     {markets_path}")
    print(f"  Chunk size:  {chunk_size:,}")
    print(f"  Compression: {compression}")
    print(f"{'='*65}\n")

    # Step 1: Build token→market lookup (small, fits in RAM easily)
    print("  Building token lookup from markets.csv …", end="", flush=True)
    token_lookup = _build_token_lookup(markets_path)
    print(f" {len(token_lookup):,} token entries\n")

    # Step 2: Count rows for progress display
    print("  Counting rows in orderFilled.csv …", end="", flush=True)
    raw_total = _count_lines(input_path)
    print(f" {raw_total:,} (includes duplicate legs)\n")

    writer_ref = [None]
    total_rows = 0
    skipped_no_market = 0
    skipped_usdc_leg = 0
    chunk_num = 0
    start_time = time.time()

    for chunk in _iter_csv_chunks(input_path, chunk_size):
        chunk_start = time.time()

        timestamps    = []
        market_ids    = []
        sides         = []
        prices        = []
        usd_amounts   = []
        token_amounts = []
        directions    = []
        tx_hashes     = []

        for row in chunk:
            maker_asset = (row.get("makerAssetId") or "").strip()
            taker_asset = (row.get("takerAssetId") or "").strip()

            # Skip USDC-leg rows (makerAssetId == "0")
            # These are the duplicate of the token-leg with roles swapped.
            if maker_asset == "0":
                skipped_usdc_leg += 1
                continue

            # Resolve market_id and side from the token ID
            entry = token_lookup.get(maker_asset)
            if entry is None:
                skipped_no_market += 1
                continue

            market_id, side = entry

            # Parse amounts (6-decimal fixed point)
            maker_raw = row.get("makerAmountFilled") or "0"
            taker_raw = row.get("takerAmountFilled") or "0"
            try:
                maker_units = int(maker_raw)
                taker_units = int(taker_raw)
            except ValueError:
                skipped_no_market += 1
                continue

            token_qty = maker_units / _USDC_DECIMALS   # conditional tokens
            usd_qty   = taker_units / _USDC_DECIMALS   # USDC

            # Price = USDC per token; guard against zero division
            price = usd_qty / token_qty if token_qty > 0 else None

            # Unix timestamp (seconds) → microseconds
            ts_str = (row.get("timestamp") or "").strip()
            try:
                ts_us = int(ts_str) * 1_000_000
            except ValueError:
                ts_us = None

            # Direction: maker is selling tokens → SELL from maker's perspective.
            # takerAssetId == "0" means taker pays USDC = taker is BUYING tokens.
            direction = "BUY" if taker_asset == "0" else "SELL"

            timestamps.append(ts_us)
            market_ids.append(market_id)
            sides.append(side)
            prices.append(price)
            usd_amounts.append(usd_qty)
            token_amounts.append(token_qty)
            directions.append(direction)
            tx_hashes.append(row.get("transactionHash") or "")

        if not timestamps:
            continue

        arrow_table = pa.table(
            {
                "timestamp":    pa.array(timestamps,    type=pa.timestamp("us", tz="UTC")),
                "market_id":    pa.array(market_ids,    type=pa.int64()),
                "side":         pa.array(sides,         type=pa.string()),
                "price":        pa.array(prices,        type=pa.float64()),
                "usd_amount":   pa.array(usd_amounts,   type=pa.float64()),
                "token_amount": pa.array(token_amounts, type=pa.float64()),
                "direction":    pa.array(directions,    type=pa.string()),
                "tx_hash":      pa.array(tx_hashes,     type=pa.string()),
            }
        )

        _write_chunk(writer_ref, output_path, arrow_table, compression)

        chunk_num += 1
        total_rows += len(timestamps)
        _progress(chunk_num, total_rows, raw_total // 2,  # approx: 50% are token-leg
                  time.time() - chunk_start, start_time)

    if writer_ref[0]:
        writer_ref[0].close()

    elapsed = time.time() - start_time
    size_mb = output_path.stat().st_size / 1_048_576 if output_path.exists() else 0
    print(f"\n{'='*65}")
    print(f"  Conversion complete!")
    print(f"  Rows written:         {total_rows:,}")
    print(f"  Skipped (USDC leg):   {skipped_usdc_leg:,}")
    print(f"  Skipped (no market):  {skipped_no_market:,}")
    print(f"  Output size:          {size_mb:.1f} MB")
    print(f"  Time:                 {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"{'='*65}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet (RAM-efficient streaming).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert orderFilled trades CSV → trades.parquet
  python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \\
      --markets markets.csv

  # Convert markets CSV → markets.parquet
  python convert_to_parquet.py markets markets.csv data/markets.parquet

  # Low-RAM: use smaller chunks (default 500 000)
  python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \\
      --markets markets.csv --chunk-size 100000

  # Better compression (smaller files, slower write)
  python convert_to_parquet.py markets markets.csv data/markets.parquet \\
      --compression zstd

Compression:
  snappy — fastest, medium size  (recommended)
  gzip   — medium speed, smaller
  zstd   — slowest write, smallest files
        """,
    )
    parser.add_argument(
        "source_type",
        choices=["trades", "markets"],
        help='Type of CSV: "trades" (orderFilled.csv) or "markets" (markets.csv)',
    )
    parser.add_argument("input",  help="Path to input CSV file")
    parser.add_argument("output", help="Path to output Parquet file")
    parser.add_argument(
        "--markets",
        metavar="MARKETS_CSV",
        help='Path to markets.csv (required when source_type="trades")',
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per processing chunk (default: 500 000). "
             "Lower for less RAM usage, higher for speed.",
    )
    parser.add_argument(
        "--compression",
        choices=["snappy", "gzip", "zstd"],
        default="snappy",
        help="Parquet compression codec (default: snappy)",
    )

    args = parser.parse_args()

    # Validate
    if args.source_type == "trades" and not args.markets:
        parser.error('--markets is required when source_type="trades"')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.source_type == "markets":
            convert_markets_csv(
                input_path=args.input,
                output_path=args.output,
                chunk_size=args.chunk_size,
                compression=args.compression,
            )
        else:
            convert_trades_csv(
                input_path=args.input,
                output_path=args.output,
                markets_path=args.markets,
                chunk_size=args.chunk_size,
                compression=args.compression,
            )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1
    except ImportError:
        print("\nERROR: pyarrow not installed. Run: pip install pyarrow")
        return 1
    except Exception as e:
        import traceback
        print(f"\nERROR during conversion: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
