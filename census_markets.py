#!/usr/bin/env python3
"""
D0 — the trainability census.

Answers the question that has to come before "should we train per group?":
**how much of this data can the pipeline learn from at all?**

Why it is not obvious
---------------------
Every lag and rolling feature is computed `.over("market_id")` (see
`pipeline/rowgroups.py`) over windows up to `max(FeatureConfig.rolling_windows)`
buckets, and a label needs `LabelConfig.forward_window_buckets` more. At the
default 5-minute aggregation that is 48 + 6 = 54 buckets — **4.5 hours of
market life**, not 54 trades.

Two counts per market, because they answer different halves of that:

* `active` — buckets that actually contain trades. Labels need both ends real
  (`README.md`: a label whose exit bucket never traded is a phantom), so this
  bounds how many labels a market can produce.
* `span` — first to last bucket inclusive. `pipeline/gap_handler.py` fills the
  holes, so this is what the rolling windows see.

A market can have a long span and almost no active buckets; those are the ones
that look trainable by span and produce nothing usable.

The decision this feeds
-----------------------
From `docs/universe-and-grouping.md`: if less than ~20 % of *volume* sits in
trainable markets, the feature architecture is the problem and no amount of
per-group modelling will help. Volume share is therefore the headline number,
not the market count — one deep market is worth thousands of shallow ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import BucketConfig, FeatureConfig, LabelConfig

DEFAULT_TRADES = "data/trades.parquet"
DEFAULT_REGISTRY = "data/market_registry.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D0 — trainability census.")
    p.add_argument("--trades", default=DEFAULT_TRADES)
    p.add_argument("--registry", default=DEFAULT_REGISTRY)
    p.add_argument("--bucket-minutes", type=int, default=None)
    p.add_argument(
        "--top", type=int, default=15, help="how many groups to list (default 15)"
    )
    return p.parse_args()


def thresholds(bucket_minutes: int) -> tuple[int, int, int]:
    """(feature window, label window, required buckets) — all from config."""
    feat = max(FeatureConfig().rolling_windows)
    label = LabelConfig().forward_window_buckets
    return feat, label, feat + label


def census(trades_path: str, bucket_minutes: int) -> pl.DataFrame:
    """One row per market: active buckets, span, trades, volume, first/last."""
    step = f"{bucket_minutes}m"
    per_bucket = (
        pl.scan_parquet(trades_path)
        .with_columns(pl.col("timestamp").dt.truncate(step).alias("bucket_time"))
        .group_by(["market_id", "bucket_time"])
        .agg(
            pl.len().alias("trades"),
            pl.col("usd_amount").sum().alias("usd"),
        )
    )
    us = bucket_minutes * 60 * 1_000_000
    return (
        per_bucket.group_by("market_id")
        .agg(
            pl.len().alias("active"),
            pl.col("trades").sum().alias("trades"),
            pl.col("usd").sum().alias("usd"),
            pl.col("bucket_time").min().alias("first"),
            pl.col("bucket_time").max().alias("last"),
        )
        .with_columns(
            (
                (pl.col("last") - pl.col("first")).dt.total_microseconds() // us + 1
            ).alias("span")
        )
        .collect(engine="streaming")
    )


def _share(df: pl.DataFrame, mask: pl.Expr, total_usd: float) -> tuple[int, float]:
    sub = df.filter(mask)
    return sub.height, (sub["usd"].sum() / total_usd if total_usd else 0.0)


def report(df: pl.DataFrame, need: int, feat: int, label: int, bm: int) -> None:
    n = df.height
    total_usd = float(df["usd"].sum())
    print(f"\n{'=' * 66}\n  D0 — TRAINABILITY CENSUS\n{'=' * 66}")
    print(f"  Bucket size          {bm} min")
    print(f"  Feature window       {feat} buckets ({feat * bm / 60:.1f} h)")
    print(f"  Label window         {label} buckets ({label * bm} min)")
    print(f"  Required per market  {need} buckets ({need * bm / 60:.1f} h)")
    print(f"\n  Markets that traded  {n:,}")
    print(f"  USD volume           {total_usd / 1e9:.2f} bn")

    print(f"\n  Active buckets per market (buckets that really traded):")
    q = df["active"]
    for label_, v in (("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p99", 0.99)):
        print(f"    {label_}  {q.quantile(v):>10,.0f}")
    print(f"    max  {q.max():>10,}")

    print(f"\n  {'':<34}{'markets':>12}{'share':>9}{'volume':>10}")
    print(f"  {'-' * 64}")
    for name, mask in (
        (f"span >= {need} (gap-filled window fits)", pl.col("span") >= need),
        (f"active >= {need} (enough real buckets)", pl.col("active") >= need),
        (
            f"both  >= {need}  <- trainable",
            (pl.col("span") >= need) & (pl.col("active") >= need),
        ),
    ):
        c, s = _share(df, mask, total_usd)
        print(f"  {name:<34}{c:>12,}{c / n:>8.1%}{s:>10.1%}")

    trainable = df.filter((pl.col("span") >= need) & (pl.col("active") >= need))
    rows = int((trainable["active"] - need).clip(lower_bound=0).sum())
    print(f"\n  Upper bound on feature rows: {rows:,}")
    print("  (active buckets past the warm-up; gap handling and the "
          "both-ends-real\n   label rule will cut into this)")

    verdict = _share(df, (pl.col("span") >= need) & (pl.col("active") >= need), total_usd)[1]
    print(f"\n{'=' * 66}")
    if verdict < 0.20:
        print(f"  VERDICT: {verdict:.1%} of volume is trainable — below the 20 % line.")
        print("  The feature architecture is the constraint, not the model.")
        print("  Per-group training cannot fix this; shorter windows or a")
        print("  family-level time series can.")
    else:
        print(f"  VERDICT: {verdict:.1%} of volume is trainable — above the 20 % line.")
        print("  Proceed to D1 (cost floor per group).")
    print("=" * 66)


def by_group(df: pl.DataFrame, registry: str, need: int, top: int) -> None:
    reg = pl.read_parquet(registry, columns=["market_id", "segment", "family_root"])
    j = df.join(reg, on="market_id", how="left")
    ok = (pl.col("span") >= need) & (pl.col("active") >= need)

    for col, title in (("segment", "SEGMENT"), ("family_root", "FAMILY")):
        agg = (
            j.group_by(col)
            .agg(
                pl.len().alias("markets"),
                ok.sum().alias("trainable"),
                pl.col("usd").sum().alias("usd"),
                pl.when(ok).then(pl.col("usd")).otherwise(0.0).sum().alias("usd_ok"),
            )
            .with_columns(
                (pl.col("usd_ok") / pl.col("usd").replace(0.0, None)).alias("vol_ok")
            )
            .sort("usd", descending=True)
            .head(top)
        )
        print(f"\n{'=' * 66}\n  BY {title}\n{'=' * 66}")
        print(f"  {'':<34}{'markets':>9}{'trainable':>11}{'vol':>8}{'vol ok':>8}")
        print(f"  {'-' * 64}")
        for row in agg.iter_rows(named=True):
            name = str(row[col])[:32]
            vol = row["usd"] / 1e9
            print(f"  {name:<34}{row['markets']:>9,}{row['trainable']:>11,}"
                  f"{vol:>7.2f}B{(row['vol_ok'] or 0):>8.1%}")


def main() -> int:
    args = parse_args()
    bm = args.bucket_minutes or BucketConfig().bucket_minutes
    feat, label, need = thresholds(bm)

    for p in (args.trades, args.registry):
        if not Path(p).exists():
            raise SystemExit(f"not found: {p}")

    print("  Counting buckets ...", flush=True)
    df = census(args.trades, bm)
    report(df, need, feat, label, bm)
    by_group(df, args.registry, need, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
