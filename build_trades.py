#!/usr/bin/env python3
"""
orderFilled.csv -> trades.parquet, joined and filtered in one streaming pass.

Why this reads the raw events instead of poly_data's processed/trades.csv
------------------------------------------------------------------------
poly_data is the collection layer; the join, the universe filter and the ML
live here. Two measurements put stage 3 out of reach on this host anyway:

* Size. orderFilled.csv is ~163 GB (~640 M rows). processed/trades.csv adds a
  66-char hex market_id per row and an ISO timestamp, landing near ~195 GB
  against 270 GB free — and ~73 % of it would be discarded here, because the
  trainable universe is ~27 % of markets.
* Memory. `process_live._discover_missing_tokens()` runs before the chunked
  loop and ignores PROCESS_CHUNK_SIZE: two `.unique().collect()` passes over
  the whole file, then Python sets of 77-char strings.

Joining against the registry with `how="inner"` applies the universe filter
*during* the scan, so the expensive work happens once and on 27 % of the data.

What it costs
-------------
The Gamma backfill is lost. poly_data's stage 3 fetches markets for token ids
absent from the CLOB list; here such a trade simply finds no registry row and
drops out. The CLOB list covers all ~3.4 M markets, so this should be an edge
case — `--count-unmatched` measures it instead of assuming it.

Output schema matches what `pipeline/data_loader.py` expects with the stock
`config.py` column names, so no config change is needed beyond the path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

DEFAULT_ORDERS = "/root/poly_data/data/orderFilled.csv"
DEFAULT_REGISTRY = "data/market_registry.parquet"
DEFAULT_OUT = "data/trades.parquet"

# Raw integers in the export carry 6 decimals — USDC and CTF tokens both do.
DECIMALS = 1_000_000

ORDER_SCHEMA = {
    "timestamp": pl.Int64,
    "maker": pl.Utf8,
    "makerAssetId": pl.Utf8,
    "makerAmountFilled": pl.Int64,
    "taker": pl.Utf8,
    "takerAssetId": pl.Utf8,
    "takerAmountFilled": pl.Int64,
    "transactionHash": pl.Utf8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Join orderFilled.csv against the market registry."
    )
    p.add_argument("--orders", default=DEFAULT_ORDERS)
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--all-markets",
        action="store_true",
        help="keep every market, not only the trainable universe",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="only read the first N events"
    )
    p.add_argument(
        "--count-unmatched",
        action="store_true",
        help="extra pass counting events whose token is in no registry row "
        "(the cost of dropping poly_data's Gamma backfill)",
    )
    return p.parse_args()


def token_lookup(registry_path: str, all_markets: bool) -> pl.DataFrame:
    """
    (asset_id, market_id, side) — one row per token, two per market.

    Small by construction: ~1.8 M rows for the universe, so the join's build
    side stays in memory no matter how large the event stream is.
    """
    cols = ["market_id", "token1", "token2", "keep"]
    reg = pl.read_parquet(registry_path, columns=cols)
    if not all_markets:
        reg = reg.filter(pl.col("keep"))
    return (
        reg.drop("keep")
        .unpivot(
            index="market_id",
            on=["token1", "token2"],
            variable_name="side",
            value_name="asset_id",
        )
        .drop_nulls("asset_id")
        .filter(pl.col("asset_id").str.len_chars() > 0)
    )


def build(orders: str, lookup: pl.DataFrame, limit: int | None) -> pl.LazyFrame:
    """
    The whole transform as one lazy plan, mirroring
    `update_utils/process_live.py::_processed_df`.

    Directions are the **taker's** view: `takerAssetId == "0"` means the taker
    paid USDC, so the taker bought. `convert_to_parquet.py` calls its column
    "from maker's view" but computes exactly this — the docstring is wrong,
    and picking the maker's side would invert `is_buy` for the whole dataset.
    """
    lf = pl.scan_csv(orders, schema_overrides=ORDER_SCHEMA, ignore_errors=True)
    if limit:
        lf = lf.head(limit)

    maker_is_usdc = pl.col("makerAssetId") == "0"
    taker_is_usdc = pl.col("takerAssetId") == "0"

    lf = lf.with_columns(
        pl.when(maker_is_usdc)
        .then(pl.col("takerAssetId"))
        .otherwise(pl.col("makerAssetId"))
        .alias("asset_id")
    )

    # inner join == the universe filter, applied during the scan
    lf = lf.join(lookup.lazy(), on="asset_id", how="inner")

    maker_amt = pl.col("makerAmountFilled") / DECIMALS
    taker_amt = pl.col("takerAmountFilled") / DECIMALS

    lf = lf.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp"),
        pl.when(taker_is_usdc)
        .then(taker_amt)
        .otherwise(maker_amt)
        .alias("usd_amount"),
        pl.when(taker_is_usdc)
        .then(maker_amt)
        .otherwise(taker_amt)
        .alias("token_amount"),
        pl.when(taker_is_usdc)
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("SELL"))
        .alias("direction"),
        pl.col("transactionHash").alias("tx_hash"),
    )

    # price = USDC per outcome token, from whichever leg paid USDC.
    # process_live divides without a guard and can emit inf/NaN; a price
    # outside (0,1) is not a conditional-token price, so drop the row.
    lf = lf.with_columns(
        (pl.col("usd_amount") / pl.col("token_amount")).alias("price")
    ).filter(
        pl.col("price").is_finite()
        & (pl.col("price") > 0.0)
        & (pl.col("price") < 1.0)
        & (pl.col("token_amount") > 0.0)
    )

    return lf.select(
        "timestamp",
        pl.col("market_id").cast(pl.Int32),
        "side",
        "price",
        "usd_amount",
        "token_amount",
        "direction",
        "tx_hash",
    )


def count_unmatched(orders: str, registry_path: str, limit: int | None) -> None:
    """
    How many events reference a token that is in **no** registry row at all.

    Deliberately measured against the *full* registry, not the universe
    lookup. Against the filtered one this would count every excluded market as
    a miss — and since the candle instruments are hyperactive, that number
    reads like catastrophic data loss when it is just the filter doing its job.
    What matters here is only what poly_data's Gamma backfill would have
    recovered and this script cannot.
    """
    lookup = token_lookup(registry_path, all_markets=True)
    lf = pl.scan_csv(orders, schema_overrides=ORDER_SCHEMA, ignore_errors=True)
    if limit:
        lf = lf.head(limit)
    lf = lf.with_columns(
        pl.when(pl.col("makerAssetId") == "0")
        .then(pl.col("takerAssetId"))
        .otherwise(pl.col("makerAssetId"))
        .alias("asset_id")
    ).select("asset_id")
    known = lookup.select("asset_id").lazy()
    total = lf.select(pl.len()).collect(engine="streaming").item()
    missed = (
        lf.join(known, on="asset_id", how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if total:
        print(f"  Events whose token is in no market at all: "
              f"{missed:,} / {total:,} ({missed / total:.2%})")
        print("  (this is the Gamma-backfill loss; the universe filter is separate)")
    else:
        print("  no events")


def main() -> int:
    args = parse_args()
    print("=" * 62)
    print("  TRADES")
    print("=" * 62)
    print(f"  Orders   : {args.orders}")
    print(f"  Registry : {args.registry}")
    print(f"  Output   : {args.out}")
    print(f"  Universe : {'all markets' if args.all_markets else 'trainable only'}")

    if not Path(args.orders).exists():
        raise SystemExit(f"not found: {args.orders}")
    if not Path(args.registry).exists():
        raise SystemExit(
            f"not found: {args.registry} — run build_registry.py first"
        )

    lookup = token_lookup(args.registry, args.all_markets)
    print(f"\n  Token lookup: {lookup.height:,} tokens "
          f"({lookup['market_id'].n_unique():,} markets)")

    if args.count_unmatched:
        print("\n  Counting unmatched events ...", flush=True)
        count_unmatched(args.orders, args.registry, args.limit)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print("\n  Streaming join ...", flush=True)
    t0 = time.time()
    build(args.orders, lookup, args.limit).sink_parquet(args.out)
    dt = time.time() - t0

    size = Path(args.out).stat().st_size
    n = pl.scan_parquet(args.out).select(pl.len()).collect().item()
    print(f"\n  Rows written : {n:,}")
    print(f"  Size         : {size / 2**30:.2f} GiB")
    print(f"  Time         : {dt / 60:.1f} min")
    print(f"\n  Next: python census_markets.py --trades {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
