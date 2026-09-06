#!/usr/bin/env python3
"""
sweep_horizon.py — Wie lange muss man halten, damit sich das ueberhaupt traegt?

Beantwortet EINE Frage, ohne ein Modell zu trainieren:

    Bewegt sich der Preis im Label-Fenster weit genug, um die Handelskosten
    zu bezahlen — und bei wie vielen Buckets?

Warum das vor jedem Modelltraining kommt
----------------------------------------
Die Kosten sind bekannt (pipeline-Kostenmodell, siehe config.CostConfig): bei
einem Tokenpreis um 0.50 rund 10 % der Position, bei 0.05 rund 28 %. Wenn die
Preisbewegung im gewaehlten Fenster diese Schwelle so gut wie nie erreicht,
kann KEIN Modell profitabel sein — auch ein perfektes nicht. Dann ist nicht
das Modell das Problem, sondern die Haltedauer.

Die Spalte `Ret>Cost` ist deshalb eine OBERGRENZE: sie unterstellt, dass man
fuer jeden Bucket die bessere der beiden Seiten kennt. Ein reales Modell liegt
darunter.

Beide Seiten werden einzeln gerechnet, nicht ueber |Rendite| genaehert. Eine
YES- und eine NO-Share desselben Marktes sind keine Spiegelbilder: bei
P(YES)=0.20 kostet die YES-Share 0.20 und die NO-Share 0.80, also hat dieselbe
Preisbewegung dort -5.0 % bzw. +1.25 % Rendite — und 13 % bzw. 3.2 % Kosten.

Aufwand
-------
Bucketing, Luecken-Behandlung und Features haengen nicht vom Horizont ab —
nur das Labeln. Der Sweep laeuft die teuren Schritte deshalb genau einmal und
danach je Horizont nur noch das Labeling. Features werden gar nicht gebaut,
sie spielen fuer diese Frage keine Rolle.

Usage:
    python sweep_horizon.py \
        --trades data/trades.parquet --markets data/markets.parquet \
        --windows 6 12 24 48 96 288
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from pipeline.rowgroups import iter_market_row_groups

sys.path.insert(0, str(Path(__file__).parent))

from config import CostConfig, PipelineConfig
from pipeline.aggregation import aggregate_trades
from pipeline.data_loader import load_trades
from pipeline.segments import SEGMENTS, load_segment_map, segment_of
from pipeline.gap_handler import (
    apply_gap_exclusions,
    detect_consecutive_gaps,
    fill_buckets,
)
from pipeline.labeling import add_labels

# Reservoir size per horizon — bounds RAM while keeping quantiles accurate
_RESERVOIR = 200_000

# Token-price bands for the breakdown.  Costs scale with 1/price, so the
# answer can differ completely between them.
_BANDS = [
    ("< 0.05", 0.0, 0.05),
    ("0.05-0.15", 0.05, 0.15),
    ("0.15-0.35", 0.15, 0.35),
    ("0.35-0.65", 0.35, 0.65),
    ("> 0.65", 0.65, 1.01),
]


def best_side(
    ret_dom: np.ndarray,
    ret_opp: np.ndarray,
    px_dom: np.ndarray,
    px_opp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick the better of the two sides per bucket and return (return, price)
    for that side.

    The pairing matters.  A YES share costs `p`, a NO share `1 - p`, so the
    two sides are not mirror images: at P(YES)=0.20 a drop to 0.19 is -5.0 %
    on the YES side but only +1.25 % on the NO side, and the costs are 13 %
    against 3.2 %.  Approximating the opposite side as |return of the
    dominant side| and charging the dominant side's price would overstate
    the ceiling on both counts.
    """
    take_dom = ret_dom >= ret_opp
    return np.where(take_dom, ret_dom, ret_opp), np.where(take_dom, px_dom, px_opp)


class Accumulator:
    """Exact counters plus a bounded reservoir sample for quantiles."""

    def __init__(self, seed: int = 42):
        self.n = 0
        self.n_over_cost = 0
        self.sum_ret = 0.0
        self.sum_cost = 0.0
        self.sum_excess = 0.0          # sum of (|ret| - cost) where positive
        self._res_ret: list[np.ndarray] = []
        self._res_cost: list[np.ndarray] = []
        self._res_n = 0
        self._rng = np.random.default_rng(seed)

    def add(self, ret: np.ndarray, cost: np.ndarray) -> None:
        if len(ret) == 0:
            return
        over = ret > cost
        self.n += len(ret)
        self.n_over_cost += int(over.sum())
        self.sum_ret += float(ret.sum())
        self.sum_cost += float(cost.sum())
        self.sum_excess += float((ret[over] - cost[over]).sum())

        # Keep at most _RESERVOIR samples, sampled uniformly
        if self._res_n < _RESERVOIR:
            self._res_ret.append(ret)
            self._res_cost.append(cost)
            self._res_n += len(ret)
        elif self._rng.random() < 0.05:
            k = min(len(ret), 2000)
            idx = self._rng.choice(len(ret), size=k, replace=False)
            self._res_ret.append(ret[idx])
            self._res_cost.append(cost[idx])

    def quantiles(self) -> tuple[float, float, float]:
        if not self._res_ret:
            return 0.0, 0.0, 0.0
        r = np.concatenate(self._res_ret)
        c = np.concatenate(self._res_cost)
        return (
            float(np.median(r)),
            float(np.percentile(r, 90)),
            float(np.median(c)),
        )

    @property
    def share_over_cost(self) -> float:
        return self.n_over_cost / self.n if self.n else 0.0

    @property
    def max_roi(self) -> float:
        """
        ROI with perfect foresight: trade only where |return| beats the cost,
        and always pick the winning side.  No model can do better.
        """
        return self.sum_excess / self.n_over_cost if self.n_over_cost else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep the label horizon against the trading-cost floor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--trades", default="data/trades.parquet")
    p.add_argument("--markets", default="data/markets.parquet")
    p.add_argument("--trades-format", default="parquet",
                   choices=["csv", "parquet", "sqlite"])
    p.add_argument("--markets-format", default="parquet",
                   choices=["csv", "parquet", "sqlite"])
    p.add_argument("--bucket-minutes", type=int, default=5)
    p.add_argument("--windows", type=int, nargs="+",
                   default=[3, 6, 12, 24, 48, 96, 288],
                   help="Label horizons in buckets (default: 3 6 12 24 48 96 288)")
    p.add_argument("--spread-abs", type=float, default=None,
                   help="Override CostConfig.spread_abs (default: 0.010, the "
                        "median measured across 120 live order books)")
    p.add_argument("--fee-legs", type=int, default=None,
                   help="Taker legs crossed per round trip (0, 1 or 2)")
    p.add_argument("--by-band", action="store_true",
                   help="Also break the best horizon down by token price")
    p.add_argument("--segments", default=None, metavar="PATH",
                   help="Segment map from classify_markets.py "
                        "(market_id -> sports/crypto/politics/other)")
    p.add_argument("--by-segment", action="store_true",
                   help="Break every horizon down by market segment "
                        "(requires --segments)")
    p.add_argument("--segment", default=None, metavar="NAME",
                   help="Restrict the whole run to one segment "
                        "(requires --segments)")
    p.add_argument("--keep-intermediates", action="store_true")
    return p.parse_args()


def _fmt_horizon(buckets: int, bucket_minutes: int) -> str:
    mins = buckets * bucket_minutes
    if mins < 60:
        return f"{mins} min"
    if mins < 1440:
        return f"{mins/60:.1f} h".replace(".0 h", " h")
    return f"{mins/1440:.1f} d".replace(".0 d", " d")


def main() -> int:
    args = parse_args()

    cfg = PipelineConfig()
    cfg.data.trades_path = args.trades
    cfg.data.markets_path = args.markets
    cfg.data.trades_format = args.trades_format
    cfg.data.markets_format = args.markets_format
    cfg.bucket.bucket_minutes = args.bucket_minutes

    cost = CostConfig()
    if args.spread_abs is not None:
        cost.spread_abs = args.spread_abs
    if args.fee_legs is not None:
        cost.fee_legs = args.fee_legs

    print("=" * 78)
    print("  HORIZON SWEEP  —  does the price move far enough to pay the costs?")
    print("=" * 78)
    print(f"  Trades  : {args.trades}")
    print(f"  Buckets : {args.bucket_minutes} min")
    print(f"  Cost    : spread_abs={cost.spread_abs}  fee_rate={cost.fee_rate}  "
          f"fee_legs={cost.fee_legs}")
    print(f"  Horizons: {', '.join(_fmt_horizon(w, args.bucket_minutes) for w in args.windows)}")
    print("=" * 78)

    # ── Segment map ──────────────────────────────────────────────────────
    segment_map: dict[int, str] = {}
    if args.segments:
        segment_map = load_segment_map(args.segments)
        print(f"  Segments: {args.segments}  ({len(segment_map):,} markets)")
        if args.segment and args.segment not in SEGMENTS:
            print(f"ERROR: unknown segment '{args.segment}'. "
                  f"Known: {', '.join(SEGMENTS)}")
            return 1
        if args.segment:
            print(f"  Filter  : only `{args.segment}`")
    elif args.by_segment or args.segment:
        print("ERROR: --by-segment / --segment need --segments PATH "
              "(build it with classify_markets.py).")
        return 1

    tmp = Path("_sweep_tmp")
    tmp.mkdir(exist_ok=True)

    # ── Preprocessing, once ──────────────────────────────────────────────
    print("\n[1/3] Bucketing …")
    trades_lf = load_trades(cfg.data)
    bucketed_path = aggregate_trades(
        trades_lf, cfg.bucket, output_path=str(tmp / "bucketed.parquet")
    )
    bucketed = pl.scan_parquet(bucketed_path).collect(engine="streaming")
    print(f"  {bucketed.height:,} buckets across "
          f"{bucketed['market_id'].n_unique():,} markets")

    print("\n[2/3] Gap handling …")
    path = fill_buckets(bucketed, cfg.bucket, cfg.gap,
                        output_path=str(tmp / "filled.parquet"))
    del bucketed
    gc.collect()
    path = detect_consecutive_gaps(path, cfg.gap,
                                   output_path=str(tmp / "gaps.parquet"))
    path = apply_gap_exclusions(path, cfg.gap,
                                output_path=str(tmp / "final.parquet"))

    # ── One pass over the data, all horizons at once ─────────────────────
    print(f"\n[3/3] Labeling for {len(args.windows)} horizons (single pass) …")
    accs = {w: Accumulator() for w in args.windows}
    band_accs = {w: {b[0]: Accumulator() for b in _BANDS} for w in args.windows}
    seg_accs = {w: {sg: Accumulator() for sg in SEGMENTS} for w in args.windows}
    label_cfgs = {
        w: type(cfg.label)(**{**cfg.label.__dict__, "forward_window_buckets": w})
        for w in args.windows
    }

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    skipped_markets = 0
    for rg, market_df in iter_market_row_groups(pf):

        # One row group is exactly one market, so the segment is a single
        # dict lookup — no join needed.
        market_segment = "other"
        if segment_map:
            mid = market_df["market_id"][0] if market_df.height else None
            market_segment = segment_of(mid, segment_map)
            if args.segment and market_segment != args.segment:
                skipped_markets += 1
                del market_df
                continue

        for w in args.windows:
            labeled = add_labels(market_df, label_cfgs[w])
            valid = labeled.filter(
                pl.col("trade_return").is_not_null()
                & pl.col("entry_token_price").is_not_null()
            )
            if valid.height == 0:
                continue

            # Perfect foresight means picking the better SIDE, and the two
            # sides are not mirror images: a YES share costs p, a NO share
            # 1-p, so both the return and the cost differ.  Using |return|
            # with one price would overstate the ceiling — at P(YES)=0.20 by
            # a factor of 4, while charging 13 % cost instead of 3.2 %.
            ret_dom = valid["trade_return"].to_numpy().astype(np.float64)
            ret_opp = valid["trade_return_opp"].to_numpy().astype(np.float64)
            px_dom = valid["entry_token_price"].to_numpy().astype(np.float64)
            px_opp = valid["entry_token_price_opp"].to_numpy().astype(np.float64)

            ret, price = best_side(ret_dom, ret_opp, px_dom, px_opp)

            finite = np.isfinite(ret) & np.isfinite(price)
            ret, price = ret[finite], price[finite]
            if len(ret) == 0:
                continue
            c = np.asarray(cost.round_trip_cost(price), dtype=np.float64)

            accs[w].add(ret, c)
            if args.by_segment:
                seg_accs[w][market_segment].add(ret, c)
            if args.by_band:
                for name, lo, hi in _BANDS:
                    m = (price >= lo) & (price < hi)
                    if m.any():
                        band_accs[w][name].add(ret[m], c[m])
            del labeled, valid
        del market_df
        if (rg + 1) % 500 == 0:
            gc.collect()
        if (rg + 1) % 200 == 0 or (rg + 1) == n_rg:
            print(f"    {rg + 1}/{n_rg} markets", flush=True)
    del pf
    if args.segment and skipped_markets:
        print(f"    {skipped_markets:,} markets skipped (other segments)")

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULT")
    print("=" * 78)
    print(f"\n  {'Horizon':>9} {'Labeled':>12} {'MedianRet':>12} {'p90Ret':>10}"
          f" {'MedianCost':>11} {'Ret>Cost':>10} {'MaxROI*':>9}")
    print("  " + "-" * 76)
    for w in args.windows:
        a = accs[w]
        if a.n == 0:
            print(f"  {_fmt_horizon(w, args.bucket_minutes):>9} {'no data':>12}")
            continue
        med, p90, medcost = a.quantiles()
        print(f"  {_fmt_horizon(w, args.bucket_minutes):>9} {a.n:>12,} "
              f"{med:>11.3%} {p90:>9.3%} {medcost:>10.2%} "
              f"{a.share_over_cost:>9.2%} {a.max_roi:>+8.2%}")
    print("  " + "-" * 76)
    print("\n  MedianRet = return of the BETTER of the two sides (YES or NO).")
    print("  Ret>Cost  = share of buckets where that return beats that side's cost.")
    print("  MaxROI*  = ROI with PERFECT foresight (trade only those buckets and")
    print("             always pick the right side). A ceiling no model can beat.")

    if not any(a.n for a in accs.values()):
        print("\n  No horizon produced any labeled bucket. Every window is longer")
        print("  than the markets' own history — try shorter --windows.")
        if not args.keep_intermediates:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        return 1

    print(f"\n  The verdict moves with `spread_abs` ({cost.spread_abs}). It is the one")
    print( "  empirical input and it varies a lot between markets (quartiles across")
    print( "  120 live order books: 0.001 / 0.010 / 0.039). Re-run with --spread-abs")
    print( "  set to what you actually see in the markets you trade.")

    best = max(args.windows, key=lambda w: accs[w].share_over_cost)
    if accs[best].n:
        print(f"\n  Best horizon by Ret>Cost: "
              f"{_fmt_horizon(best, args.bucket_minutes)} "
              f"({accs[best].share_over_cost:.2%})")
        if accs[best].share_over_cost < 0.05:
            print("\n  READ THIS: under 5 % of buckets clear their own costs even")
            print("  with perfect foresight. At this horizon the problem is not the")
            print("  model — it is the holding period. Try longer windows before")
            print("  touching features or the label definition.")

    if args.by_segment:
        print(f"\n{'=' * 78}")
        print("  BY MARKET SEGMENT")
        print("=" * 78)
        for sg in SEGMENTS:
            if not any(seg_accs[w][sg].n for w in args.windows):
                continue
            print(f"\n  --- {sg} ---")
            print(f"  {'Horizon':>9} {'Labeled':>12} {'MedianRet':>12}"
                  f" {'MedianCost':>11} {'Ret>Cost':>10} {'MaxROI*':>9}")
            print("  " + "-" * 66)
            for w in args.windows:
                a = seg_accs[w][sg]
                if a.n == 0:
                    continue
                med, _p90, medcost = a.quantiles()
                print(f"  {_fmt_horizon(w, args.bucket_minutes):>9} {a.n:>12,} "
                      f"{med:>11.3%} {medcost:>10.2%} "
                      f"{a.share_over_cost:>9.2%} {a.max_roi:>+8.2%}")
        print("\n  Unterscheiden sich `Ret>Cost` und `MedianCost` zwischen den")
        print("  Segmenten deutlich, lohnt eine Trennung. Sind sie aehnlich, ist")
        print("  `segment` als kategoriales Feature in EINEM Modell der")
        print("  guenstigere Weg — getrennte Modelle bekommen weniger Daten.")

    if args.by_band:
        print(f"\n{'=' * 78}")
        print(f"  BY TOKEN PRICE  —  horizon {_fmt_horizon(best, args.bucket_minutes)}")
        print("=" * 78)
        print(f"\n  {'Token price':>12} {'Labeled':>12} {'MedianRet':>12}"
              f" {'MedianCost':>11} {'Ret>Cost':>10} {'MaxROI*':>9}")
        print("  " + "-" * 70)
        for name, _lo, _hi in _BANDS:
            a = band_accs[best][name]
            if a.n == 0:
                continue
            med, _p90, medcost = a.quantiles()
            print(f"  {name:>12} {a.n:>12,} {med:>11.3%} {medcost:>10.2%} "
                  f"{a.share_over_cost:>9.2%} {a.max_roi:>+8.2%}")
        print("  " + "-" * 70)
        print("\n  Costs scale with 1/price, so a horizon can work in the liquid")
        print("  middle and be hopeless at the extremes — or the other way round.")

    if not args.keep_intermediates:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"\n  Intermediates kept in: {tmp}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
