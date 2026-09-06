#!/usr/bin/env python3
"""
classify_markets.py — Segment-Map bauen (sports / crypto / politics / other)

Erzeugt aus markets.parquet eine Zuordnung `market_id -> segment`, die
sweep_horizon.py und benchmark_strategies.py als Filter benutzen.

Warum nicht einfach die API fragen
----------------------------------
markets.csv hat kein Kategorie-Feld, und Polymarkets Gamma-API taggt nur
noch offene Maerkte: von 100 geschlossenen Maerkten hatte kein einziger
einen brauchbaren Segment-Tag (nur den Platzhalter "All"), waehrend aktive
zu 100 % sauber getaggt sind. Ein historischer Datensatz besteht fast
ausschliesslich aus geschlossenen Maerkten.

Deshalb: aus dem Slug klassifizieren — und mit `--validate` gegen die
API-Tags der noch aktiven Maerkte benoten. Das liefert eine gemessene
Trefferquote statt einer Behauptung.

Usage:
    python classify_markets.py --markets data/markets.parquet \
        --out data/market_segments.parquet

    # zusaetzlich gegen die API-Tags benoten
    python classify_markets.py --markets data/markets.parquet \
        --out data/market_segments.parquet --validate
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.segments import SEGMENTS, classify_frame, segment_from_tags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a market_id -> segment map from market slugs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--markets", default="data/markets.parquet")
    p.add_argument("--markets-format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--out", default="data/market_segments.parquet")
    p.add_argument("--validate", action="store_true",
                   help="Grade the classifier against Gamma tags. Only the "
                        "still-active markets can be graded — closed ones "
                        "carry no usable tag.")
    p.add_argument("--validate-limit", type=int, default=2000,
                   help="Max markets to look up for validation (default 2000)")
    return p.parse_args()


def _load_markets(path: str, fmt: str) -> pl.DataFrame:
    df = pl.read_parquet(path) if fmt == "parquet" else pl.read_csv(path)
    if "market_id" not in df.columns and "id" in df.columns:
        df = df.rename({"id": "market_id"})
    if "market_id" not in df.columns:
        raise SystemExit(
            f"ERROR: {path} has no `market_id` (or `id`) column. "
            f"Columns: {', '.join(df.columns)}"
        )
    return df


def _print_distribution(df: pl.DataFrame) -> None:
    total = df.height
    counts = (
        df.group_by("segment").len().sort("len", descending=True)
    )
    print(f"\n  {'Segment':<10} {'Maerkte':>9} {'Anteil':>8}")
    print("  " + "-" * 29)
    for row in counts.iter_rows(named=True):
        print(f"  {row['segment']:<10} {row['len']:>9,} {row['len']/total:>7.1%}")
    print("  " + "-" * 29)
    print(f"  {'gesamt':<10} {total:>9,}")


def _validate(df: pl.DataFrame, limit: int) -> None:
    """Grade the slug classifier against Gamma's own tags."""
    from pipeline.polymarket_api import (
        PolymarketAPIError,
        fetch_markets_by_id,
        market_tags,
    )

    ids = df["market_id"].cast(pl.Int64).to_list()[:limit]
    print(f"\n  Frage {len(ids):,} Markt-IDs bei der Gamma-API ab "
          f"(100 pro Request, zwei Durchgaenge je Batch) …")
    try:
        fetched = fetch_markets_by_id(ids, include_tags=True)
    except PolymarketAPIError as exc:
        print(f"  API nicht erreichbar: {exc}")
        return

    predicted = dict(
        zip(df["market_id"].cast(pl.Int64).to_list(), df["segment"].to_list())
    )

    pairs = []
    for mid, market in fetched.items():
        truth = segment_from_tags(market_tags(market))
        if truth is None:
            continue          # closed market, or tags say nothing
        pairs.append((truth, predicted.get(mid, "other")))

    print(f"  API kannte {len(fetched):,} davon, "
          f"{len(pairs):,} hatten einen brauchbaren Segment-Tag.")
    if not pairs:
        print("\n  Keine benotbaren Maerkte. Das ist bei einem rein "
              "historischen\n  Datensatz der Normalfall — geschlossene "
              "Maerkte tragen keine Tags.")
        return

    segs = list(SEGMENTS)
    cm = {(t, p): 0 for t in segs for p in segs}
    for t, p in pairs:
        cm[(t, p)] += 1

    print(f"\n  Konfusionsmatrix (Zeile = API-Tag, Spalte = Klassifikator)")
    print("           " + "".join(f"{s:>10}" for s in segs))
    for t in segs[:-1]:
        print(f"  {t:<9}" + "".join(f"{cm[(t, p)]:>10}" for p in segs))

    print(f"\n  {'Segment':<10} {'Precision':>10} {'Recall':>9} {'n':>8}")
    print("  " + "-" * 40)
    for s in segs[:-1]:
        tp = cm[(s, s)]
        fp = sum(cm[(t, s)] for t in segs if t != s)
        fn = sum(cm[(s, p)] for p in segs if p != s)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {s:<10} {prec:>10.1%} {rec:>8.1%} {tp + fn:>8,}")
    print("  " + "-" * 40)
    print("\n  Precision zaehlt hier mehr als Recall: was als `sports` "
          "markiert ist,\n  sollte auch Sport sein. Unklares landet "
          "absichtlich in `other`,\n  nicht in einem Sachsegment.")
    print(f"\n  ACHTUNG: benotet wurden nur {len(pairs):,} noch aktive "
          f"Maerkte.\n  Diese Quote laesst sich nicht ohne Weiteres auf "
          f"den ganzen Datensatz\n  hochrechnen — alte Maerkte koennen "
          f"anders benannt sein.")


def main() -> int:
    args = parse_args()

    print("=" * 62)
    print("  MARKET SEGMENTATION")
    print("=" * 62)
    print(f"  Markets: {args.markets}")
    print(f"  Output : {args.out}")

    markets = _load_markets(args.markets, args.markets_format)
    print(f"  Geladen: {markets.height:,} Maerkte")

    classified = classify_frame(markets)
    _print_distribution(classified)

    out = classified.select(
        pl.col("market_id").cast(pl.Int64),
        pl.col("segment"),
        pl.lit("slug").alias("source"),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.out.endswith(".csv"):
        out.write_csv(args.out)
    else:
        out.write_parquet(args.out)
    print(f"\n  Geschrieben: {args.out}")

    if args.validate:
        print("\n" + "=" * 62)
        print("  VALIDIERUNG GEGEN GAMMA-TAGS")
        print("=" * 62)
        _validate(classified, args.validate_limit)

    print("\n  Weiter mit:")
    print(f"    python sweep_horizon.py --trades data/trades.parquet \\")
    print(f"        --segments {args.out} --by-segment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
