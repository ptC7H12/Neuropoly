#!/usr/bin/env python3
"""
Market registry — the bridge from poly_data's export to this pipeline.

What it solves
--------------
poly_data v2 identifies a market by its on-chain `condition_id`, a 66-char
hex string.  This pipeline casts `market_id` to Int64 in eleven places
(`data_loader.py:80,147`, `segments.py:173`, `classify_markets.py:91,101,164`,
`aggregation.py:126`, `live_features.py:49`, `convert_to_parquet.py:92,284`).
Feeding it hex fails in two ways, and the quiet one is worse:

* `convert_to_parquet._build_token_lookup` calls `_safe_int32(row["id"])`,
  gets None, and `continue`s — every market row is skipped, the token lookup
  ends up empty and `trades.parquet` is written with zero rows.  No error.
* `segments.segment_of` catches the ValueError and returns "other", so a hex
  id would label all 3.19 M markets `other` rather than raising.

The registry assigns a dense Int32 `market_id` and carries the hex
`condition_id` alongside it, so every downstream cast stays valid and the
join key survives.  Four bytes per row instead of sixty-six also matters at
100 M+ trade rows on a 62 GB box.

Stable ids
----------
Ids are assigned to **every** market, not only the kept ones, and a prior
registry is reloaded so existing markets keep their id.  Both properties
exist for the same reason: `market_id` must not change when the universe
filter is retuned or when poly_data appends new markets, or every Parquet
built earlier silently refers to the wrong markets.

The universe is therefore a *view* (`keep` / `exclude_reason` columns), not
a subset.

Usage
-----
    python build_registry.py --markets /root/poly_data/data/markets.csv
    python build_registry.py --markets ... --limit 50000     # quick trial
    python build_registry.py --markets ... --validate        # recall check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from pipeline import segments
from pipeline.families import add_families
from pipeline.universe import EXCLUDE_REASONS, add_universe, is_crypto_price

# Only these are read from the 7 GB export.  `description` in particular is
# large and unused; projecting it away is most of the memory saving.
SOURCE_COLUMNS = [
    "id",
    "clobTokenIds",
    "question",
    "market_slug",
    "end_date_iso",
    "accepting_order_timestamp",
]

DEFAULT_MARKETS = "/root/poly_data/data/markets.csv"
DEFAULT_OUT = "data/market_registry.parquet"
DEFAULT_SEGMENT_OUT = "data/market_segments.parquet"

# Rows per processing slice.  The per-row work (regex classification) is a
# Python loop, so the frame is walked in slices to keep peak memory bounded
# rather than materialising 3.2 M question strings as Python objects at once.
CHUNK = 250_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the market registry from poly_data's markets.csv."
    )
    p.add_argument("--markets", default=DEFAULT_MARKETS, help="poly_data markets.csv")
    p.add_argument("--out", default=DEFAULT_OUT, help="registry Parquet output")
    p.add_argument(
        "--segments-out",
        default=DEFAULT_SEGMENT_OUT,
        help="projection consumed by --segments flags of the existing scripts",
    )
    p.add_argument(
        "--min-duration-minutes",
        type=float,
        default=720.0,
        help="markets shorter than this are excluded (default 720 = 12 h, "
        "three times the longest feature window)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="only read the first N markets"
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="report how much each widened crypto rule removes beyond the "
        "segment classifier",
    )
    return p.parse_args()


def token_exprs() -> list[pl.Expr]:
    """
    token1 / token2 from the `clobTokenIds` JSON-array string.

    Same vectorised approach as poly_data's `poly_utils.utils._token_exprs`:
    strip brackets/quotes/whitespace and split on comma.  Roughly 100x faster
    than a per-row json.loads over 3.2 M markets, and malformed values yield
    nulls instead of raising.
    """
    parts = pl.col("clobTokenIds").str.replace_all(r'[\[\]"\s]', "").str.split(",")

    def nth(i: int) -> pl.Expr:
        v = parts.list.get(i, null_on_oob=True)
        return pl.when(v.str.len_chars() > 0).then(v).otherwise(None)

    return [nth(0).alias("token1"), nth(1).alias("token2")]


def load_markets(path: str, limit: int | None) -> pl.DataFrame:
    """Read the projected columns from the export, as strings throughout."""
    lf = pl.scan_csv(
        path,
        schema_overrides={c: pl.Utf8 for c in SOURCE_COLUMNS},
        infer_schema_length=10_000,
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    have = lf.collect_schema().names()
    missing = [c for c in SOURCE_COLUMNS if c not in have]
    if missing:
        raise SystemExit(
            f"{path} is missing column(s): {', '.join(missing)}.\n"
            f"Expected poly_data v2's markets.csv schema."
        )
    lf = lf.select(SOURCE_COLUMNS)
    if limit:
        lf = lf.head(limit)
    df = lf.collect(engine="streaming")

    # poly_data's markets.csv can contain the same market twice.  Its
    # update_markets resume path dedups only against the last ~1500 rows
    # (`_read_tail_ids`), so a market re-fetched after a restart is appended
    # again if its first copy sits further up the file.  Measured after one
    # restart: 24,665 duplicate condition_ids in 3,221,098 rows.
    #
    # poly_data reads around this the same way — `get_markets()` does
    # `.unique(subset=["id"], keep="first")` — and the registry must too, or
    # a duplicated condition_id becomes a duplicated market_id and the whole
    # trades join is silently wrong.
    before = df.height
    df = df.unique(subset=["id"], keep="first", maintain_order=True)
    if df.height < before:
        print(f"  Dropped {before - df.height:,} duplicate condition_ids")
    return df


# poly_data writes both timestamps as UTC with a literal Z suffix
# (`2024-07-01T21:34:04Z`).  The format is given explicitly because polars
# refuses to infer one when the data carries a zone, and because inference
# over 3.2 M rows is slower than parsing with a known pattern.
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_ts(col: str) -> pl.Expr:
    """Parse one of poly_data's ISO timestamps; unparseable values → null."""
    return pl.col(col).str.to_datetime(
        format=TS_FORMAT, strict=False, time_unit="us"
    )


def add_duration(df: pl.DataFrame) -> pl.DataFrame:
    """
    Minutes from accepting orders to scheduled close.

    `end_date_iso` is a *scheduled* close, and for the high-frequency
    families it is a midnight stamp shared by every market resolving that
    day — so this is only meaningful for the long-lived markets that survive
    the other rules anyway.  Unparseable values become null, which
    `universe.exclusion_reason` treats as "keep".
    """
    return df.with_columns(
        (
            (parse_ts("end_date_iso") - parse_ts("accepting_order_timestamp"))
            .dt.total_minutes()
            .cast(pl.Float64)
        ).alias("duration_minutes")
    )


def classify_chunked(df: pl.DataFrame, min_duration: float) -> pl.DataFrame:
    """Apply segment, family and universe rules in memory-bounded slices."""
    out: list[pl.DataFrame] = []
    for start in range(0, df.height, CHUNK):
        part = df.slice(start, CHUNK)
        part = segments.classify_frame(part)
        part = add_families(part)
        part = add_universe(part, min_duration_minutes=min_duration)
        out.append(part)
        done = min(start + CHUNK, df.height)
        print(f"    classified {done:>10,} / {df.height:,}", flush=True)
    return pl.concat(out) if out else df


def merge_existing_ids(df: pl.DataFrame, out_path: str) -> pl.DataFrame:
    """
    Assign `market_id`, reusing ids from a prior registry where present.

    New markets get ids after the current maximum, so a rebuild after
    poly_data appended markets leaves every existing id untouched.
    """
    prior: pl.DataFrame | None = None
    if Path(out_path).exists():
        try:
            prior = pl.read_parquet(out_path, columns=["condition_id", "market_id"])
        except Exception as e:  # a corrupt or older registry must not block a rebuild
            print(f"  ! ignoring unreadable existing registry ({e})")

    if prior is None or prior.is_empty():
        return df.with_row_index("market_id").with_columns(
            pl.col("market_id").cast(pl.Int32)
        )

    # The join partner needs the same dedup as the input.  A registry written
    # before load_markets() deduped carries duplicate condition_ids, and a left
    # join against those *multiplies* rows: one duplicated prior row turns one
    # input market into two identical output rows sharing a market_id.  Seen in
    # practice — 3,386,625 deduped markets came back out as 3,411,290, exactly
    # the 24,665 duplicates the previous registry still held.
    before = prior.height
    prior = prior.unique(subset=["condition_id"], keep="first")
    if prior.height < before:
        print(
            f"  Prior registry had {before - prior.height:,} duplicate "
            f"condition_ids — deduped before the join"
        )

    df = df.join(prior, on="condition_id", how="left")
    next_id = int(prior["market_id"].max()) + 1
    fresh = df.filter(pl.col("market_id").is_null()).height
    print(f"  Reused {df.height - fresh:,} existing ids, {fresh:,} new")

    # Number the new markets consecutively from next_id.  A running count of
    # the nulls gives each one a distinct offset without leaving gaps, which
    # a row-position offset would.
    return df.with_columns(
        pl.when(pl.col("market_id").is_not_null())
        .then(pl.col("market_id"))
        .otherwise(
            pl.col("market_id").is_null().cum_sum() - 1 + next_id
        )
        .cast(pl.Int32)
        .alias("market_id")
    )


def report(df: pl.DataFrame) -> None:
    n = df.height
    print(f"\n{'=' * 62}\n  UNIVERSE\n{'=' * 62}")
    print(f"  Markets total              {n:>12,}")
    counts = dict(
        df.group_by("exclude_reason").len().iter_rows()  # (reason|None, count)
    )
    dropped = 0
    for reason in EXCLUDE_REASONS:
        c = counts.get(reason, 0)
        dropped += c
        print(f"    excluded: {reason:<16} {c:>12,}  {c / n:6.2%}")
    kept = counts.get(None, 0)
    print(f"  {'-' * 58}")
    print(f"  KEPT                       {kept:>12,}  {kept / n:6.2%}")
    assert dropped + kept == n, f"reasons do not partition: {dropped}+{kept} != {n}"

    keep_df = df.filter(pl.col("keep"))
    if keep_df.is_empty():
        return
    print(f"\n  Segments among kept markets:")
    for seg, c in keep_df.group_by("segment").len().sort("len", descending=True).iter_rows():
        print(f"    {seg:<18} {c:>12,}  {c / keep_df.height:6.2%}")

    print(f"\n  Largest families among kept markets:")
    top = (
        keep_df.group_by("family_root")
        .len()
        .sort("len", descending=True)
        .head(12)
    )
    for fam, c in top.iter_rows():
        print(f"    {str(fam)[:68]:<70} {c:>9,}")


def report_validation(df: pl.DataFrame) -> None:
    """
    How much wider is the crypto rule than the segment classifier?

    `universe.is_crypto_price` deliberately trades precision for recall,
    because the classifier is used here to *exclude* rather than select.
    This prints what that decision actually costs, so it stays a visible
    trade rather than an implicit one.
    """
    print(f"\n{'=' * 62}\n  CRYPTO RULE — WIDENING vs. THE CLASSIFIER\n{'=' * 62}")
    sub = df.filter(pl.col("exclude_reason") == "crypto")
    if sub.is_empty():
        print("  No markets excluded as crypto.")
        return
    base = sum(
        1
        for s, q in zip(sub["market_slug"].to_list(), sub["question"].to_list())
        if segments.classify(s or "", q or "") == "crypto"
    )
    extra = sub.height - base
    print(f"  Excluded as crypto           {sub.height:>12,}")
    print(f"    by segments.classify()     {base:>12,}")
    print(f"    only by the wider rules    {extra:>12,}  {extra / sub.height:6.2%}")
    if extra:
        print("\n  Sample caught only by the wider rules:")
        shown = 0
        for s, q in zip(sub["market_slug"].to_list(), sub["question"].to_list()):
            if segments.classify(s or "", q or "") != "crypto":
                print(f"    {str(s)[:70]}")
                shown += 1
                if shown >= 10:
                    break


def main() -> int:
    args = parse_args()
    print("=" * 62)
    print("  MARKET REGISTRY")
    print("=" * 62)
    print(f"  Source : {args.markets}")
    print(f"  Output : {args.out}")
    print(f"  Min duration: {args.min_duration_minutes:.0f} min")

    print("\n  Reading markets ...", flush=True)
    df = load_markets(args.markets, args.limit)
    print(f"  Read {df.height:,} markets")

    df = df.rename({"id": "condition_id"}).with_columns(token_exprs())
    df = add_duration(df)

    print("\n  Classifying ...", flush=True)
    df = classify_chunked(df, args.min_duration_minutes)

    df = merge_existing_ids(df, args.out)

    registry = df.select(
        "market_id",
        "condition_id",
        "token1",
        "token2",
        "market_slug",
        "question",
        parse_ts("end_date_iso").alias("close_time"),
        "duration_minutes",
        "segment",
        "family",
        "family_root",
        "is_periodic",
        "exclude_reason",
        "keep",
    ).sort("market_id")

    # One row per market, one id per market.  Both have been violated by a
    # silent row multiplication before (see merge_existing_ids), and a
    # duplicated market_id makes every downstream trades join quietly wrong —
    # so fail loudly here rather than write a corrupt registry.
    dupes = registry.height - registry["condition_id"].n_unique()
    if dupes:
        raise SystemExit(
            f"ABORT: {dupes:,} duplicate condition_ids in the registry "
            f"({registry.height:,} rows). Refusing to write a registry whose "
            f"join key is not unique."
        )
    if registry["market_id"].n_unique() != registry.height:
        raise SystemExit("ABORT: market_id is not unique.")

    report(registry)
    if args.validate:
        report_validation(registry)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    registry.write_parquet(args.out)
    print(f"\n  Written: {args.out}  ({registry.height:,} rows)")

    # Projection in exactly the shape segments.load_segment_map expects, so
    # `sweep_horizon.py --segments` and `benchmark_strategies.py --segments`
    # keep working unchanged.
    Path(args.segments_out).parent.mkdir(parents=True, exist_ok=True)
    registry.select(
        pl.col("market_id").cast(pl.Int64),
        "segment",
        pl.lit("slug").alias("source"),
    ).write_parquet(args.segments_out)
    print(f"  Written: {args.segments_out}")

    kept = int(registry["keep"].sum())
    print("\n  Next:")
    print(f"    python census_markets.py --registry {args.out}")
    print(f"    ({kept:,} markets in the universe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
