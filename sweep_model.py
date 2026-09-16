#!/usr/bin/env python3
"""
Parametersuche: welche Einstellungen holen am meisten aus dem Modell?

Warum das so und nicht anders laeuft
------------------------------------
Das Modell ist auskonvergiert — zwischen Iteration 150 und 390 steigt die
Validierungs-AUC um 0,003, die Trainings-AUC um 0,036.  Mehr Baeume bringen
nichts.  Die Frage ist, an welchen Stellschrauben ueberhaupt etwas haengt.

Drei Dinge machen diese Suche bezahlbar und ehrlich:

* **Features haengen nicht vom Horizont ab.**  `forward_window_buckets` kommt in
  `features.py` nirgends vor.  Eine `features.parquet` reicht fuer alle
  Horizonte; nur das Labeln muss wiederholt werden (~12 min statt ~26 min
  Vorverarbeitung).  Fuer reine Hyperparameter kostet ein Lauf ~30 s.
* **Ausgewaehlt wird auf VALIDIERUNG.**  Der Test-Split wird hier nirgends
  angefasst.  Genau das ist bisher schiefgegangen: die Schwelle 0,65, mit der
  der Paper-Trader live laeuft, wurde am Test-Split abgelesen.
* **Entschieden wird nach ROI, berichtet wird beides.**  D2 hat gezeigt, dass
  ein AUC-Unterschied von 0,009 wirtschaftlich wirkungslos war.  AUC und ROI
  koennen auseinanderlaufen, und wenn sie das tun, ist das eine Aussage ueber
  das Problem und keine Panne.

Robustheit ist keine Kuer
-------------------------
Der ROI ist schwanzgetragen: bei den Margenhebeln drehten 50 von 7.585 Trades
das Vorzeichen, und bei der Gewichtung sah eine Variante nach +49 % aus und war
ohne ihre besten fuenfzig Trades bei -15 %.  Jeder Lauf protokolliert deshalb
den ROI zusaetzlich ohne seine besten 10 und 50 Trades sowie ein
Bootstrap-Intervall.  Ohne diese Huerde waehlt ein Gitter zuverlaessig den
gluecklichsten statt den besten Lauf.

Jeder Lauf wird protokolliert, auch die schlechten — ein Gitter ohne seine
Fehlschlaege ist nicht auswertbar.

Benutzung
---------
    cd /root/Neuropoly
    ./runguard.sh --max-rss 10 --min-avail 14 -- \
        .venv/bin/python -u sweep_model.py --horizons 24 48 96 288
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from pipeline.evaluation import evaluate
from pipeline.features import get_feature_columns
from pipeline.labeling import add_labels_streaming
from pipeline.model import predict, train_model
from pipeline.splitter import walk_forward_split

# Gemessener Spread aus den Live-Orderbuechern (volumengewichtet), nicht geraten.
SPREAD = 0.0083
FEE = 0.04
FEE_LEGS = 2
MAX_COST = 0.5          # schliesst die Lottoscheine aus, wie bei den Margenhebeln
MIN_TRADES = 300        # unter dieser Trade-Zahl ist ein ROI auf Validierung Rauschen
THRESHOLDS = (0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parametersuche auf Validierung.")
    p.add_argument("--features", default="features.parquet")
    p.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 96, 288])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    p.add_argument("--out-dir", default="sweep")
    p.add_argument("--log", default="sweep/sweep_log.jsonl")
    p.add_argument("--keep-models", action="store_true",
                   help="jedes Modell speichern (~1,2 MB je Lauf)")
    p.add_argument("--min-avail-gb", type=float, default=14.0,
                   help="vor jedem Training warten, bis so viel Speicher frei ist")
    return p.parse_args()


GRID = {
    "num_leaves": (15, 31, 63),
    "max_depth": (5, 7, 9),
    "min_child_samples": (50, 200, 500),
    "learning_rate": (0.03, 0.05),
}


def grid_points():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, values))


def avail_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return float("inf")


def wait_for_memory(min_gb: float, label: str) -> None:
    """
    Warten statt blind starten.

    Diese Box ist ein LXC-Gast ohne cgroup-Limit und ohne Swap: ein Prozess, der
    uebercommittet, reisst den ganzen Host mit.  runguard.sh faengt den eigenen
    Lauf ab, aber hier wird gar nicht erst gestartet, wenn es eng ist.
    """
    waited = 0
    while avail_gb() < min_gb:
        if waited == 0:
            print(f"    warte auf Speicher ({avail_gb():.1f} < {min_gb} GiB) vor {label} …",
                  flush=True)
        time.sleep(15)
        waited += 15
        if waited > 900:
            raise SystemExit(f"nach 15 min immer noch unter {min_gb} GiB — abgebrochen")


def labeled_path_for(horizon: int, out_dir: Path) -> Path:
    return out_dir / f"labeled_h{horizon}.parquet"


def ensure_labels(features: str, horizon: int, out_dir: Path, cfg) -> Path:
    """Labeln je Horizont, einmal.  Vorhandene Datei wird wiederverwendet."""
    target = labeled_path_for(horizon, out_dir)
    if target.exists():
        print(f"  [{horizon}] Label-Datei vorhanden, wird wiederverwendet")
        return target
    lab_cfg = type(cfg.label)(**{**cfg.label.__dict__, "forward_window_buckets": horizon})
    print(f"  [{horizon}] labeln … (dauert ~12 min)", flush=True)
    t = time.time()
    wait_for_memory(10.0, f"Labeling h={horizon}")
    add_labels_streaming(features, lab_cfg, output_path=str(target))
    print(f"  [{horizon}] fertig in {(time.time()-t)/60:.1f} min", flush=True)
    return target


def round_trip_cost(price: np.ndarray) -> np.ndarray:
    p = np.clip(np.nan_to_num(price, nan=0.5), 1e-6, 1 - 1e-6)
    return (SPREAD + FEE_LEGS * FEE * np.minimum(p, 1 - p)) / p


def score_validation(y_pred: np.ndarray, split, rng) -> dict:
    """
    Wirtschaftliche Bewertung auf dem VALIDIERUNGS-Split.

    Die Schwelle wird hier mitgewaehlt — sie ist kostenlos, weil nur neu
    bewertet und nicht neu trainiert wird, und sie gehoert genauso auf die
    Validierung wie alles andere.
    """
    price, ret = split.val_price, split.val_ret
    cost = round_trip_cost(price)
    tradeable = cost <= MAX_COST

    best = None
    for thr in THRESHOLDS:
        sel = (y_pred >= thr) & tradeable
        n = int(sel.sum())
        if n < MIN_TRADES:
            continue
        pnl = ret[sel] - cost[sel]
        roi = float(pnl.mean())
        if best is None or roi > best["val_roi"]:
            srt = np.sort(pnl)[::-1]
            boot = [rng.choice(pnl, len(pnl), replace=True).mean() for _ in range(500)]
            best = {
                "threshold": thr,
                "val_trades": n,
                "val_roi": roi,
                "val_roi_drop10": float(np.delete(srt, range(min(10, n))).mean()),
                "val_roi_drop50": float(np.delete(srt, range(min(50, n))).mean()),
                "val_roi_ci_lo": float(np.percentile(boot, 2.5)),
                "val_roi_ci_hi": float(np.percentile(boot, 97.5)),
                "val_roi_median": float(np.median(pnl)),
                "val_share_positive": float((pnl > 0).mean()),
                "val_mean_cost": float(cost[sel].mean()),
                "val_median_price": float(np.median(price[sel])),
            }
    if best is None:
        return {"threshold": None, "val_trades": 0, "val_roi": None}
    return best


def done_run_ids(log_path: Path) -> set[str]:
    """Wiederaufnahme: was schon protokolliert ist, wird nicht neu gerechnet."""
    if not log_path.exists():
        return set()
    ids = set()
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["run_id"])
            except Exception:
                continue
    return ids


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"; model_dir.mkdir(exist_ok=True)
    log_path = Path(args.log); log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    cfg.monitor.rich_dashboard = False
    cfg.monitor.log_file = str(out_dir / "training_log.jsonl")

    combos = list(grid_points())
    total = len(args.horizons) * len(combos) * len(args.seeds)
    already = done_run_ids(log_path)
    print("=" * 70)
    print("  PARAMETERSUCHE")
    print("=" * 70)
    print(f"  Horizonte   : {args.horizons}")
    print(f"  Gitterpunkte: {len(combos)}   Seeds: {args.seeds}")
    print(f"  Laeufe      : {total}   davon schon erledigt: {len(already)}")
    print(f"  Bewertung   : VALIDIERUNG (Test wird nicht angefasst)")
    print(f"  Kostenmodell: spread={SPREAD} fee={FEE} legs={FEE_LEGS}")

    rng = np.random.default_rng(0)
    t_start = time.time()
    done = 0

    for horizon in args.horizons:
        labeled = ensure_labels(args.features, horizon, out_dir, cfg)

        wait_for_memory(args.min_avail_gb, f"Split h={horizon}")
        lf = pl.scan_parquet(labeled)
        feature_cols = get_feature_columns(lf.collect_schema())
        frame = lf.filter(pl.col("win").is_not_null()).collect()
        lab_cfg = type(cfg.label)(**{**cfg.label.__dict__,
                                     "forward_window_buckets": horizon})
        try:
            split = walk_forward_split(frame, feature_cols, cfg.split, lab_cfg,
                                       cfg.bucket.bucket_minutes)
        except ValueError as e:
            print(f"  [{horizon}] Split nicht moeglich: {e}")
            del frame
            continue
        del frame
        print(f"  [{horizon}] Train {len(split.train_y):,} | Val {len(split.val_y):,} "
              f"| Test {len(split.test_y):,}", flush=True)

        for combo in combos:
            for seed in args.seeds:
                run_id = (f"h{horizon}_nl{combo['num_leaves']}_md{combo['max_depth']}"
                          f"_mc{combo['min_child_samples']}_lr{combo['learning_rate']}"
                          f"_s{seed}")
                done += 1
                if run_id in already:
                    continue

                wait_for_memory(args.min_avail_gb, run_id)
                mc = type(cfg.model)(**{**cfg.model.__dict__, **combo})
                save_to = str(model_dir / f"{run_id}.txt") if args.keep_models else None

                t0 = time.time()
                booster, _ = train_model(split, mc, cfg.monitor,
                                         save_path=save_to, seed=seed)
                secs = time.time() - t0
                y_val = predict(booster, split.val_X)
                stat = evaluate(split.val_y, y_val)
                econ = score_validation(y_val, split, rng)

                entry = {
                    "run_id": run_id,
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "horizon": horizon, "seed": seed, "weighting": "none",
                    **combo,
                    "n_trees": booster.num_trees(),
                    "train_secs": round(secs, 1),
                    "val_auc": round(float(stat.roc_auc), 5) if stat.roc_auc_defined else None,
                    "val_brier": round(float(stat.brier_score), 5),
                    "val_logloss": round(float(stat.log_loss), 5),
                    **{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in econ.items()},
                    "model_path": save_to,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

                el = time.time() - t_start
                eta = el / max(done, 1) * (total - done) / 3600
                roi = entry.get("val_roi")
                print(f"  [{done:>4}/{total}] {run_id:<44} "
                      f"AUC {entry['val_auc'] or 0:.4f}  "
                      f"ROI {roi if roi is None else f'{roi:>7.2%}'}  "
                      f"({secs:.0f}s, ETA {eta:.1f} h)", flush=True)

        del split

    print(f"\n  fertig in {(time.time()-t_start)/3600:.1f} h  ->  {log_path}")
    print(f"  Auswertung: .venv/bin/python sweep_report.py --log {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
