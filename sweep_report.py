#!/usr/bin/env python3
"""
Auswertung der Parametersuche.

Die eigentliche Frage ist nicht „welcher Lauf war der beste?" — bei 432 Laeufen
ist der beste mit hoher Wahrscheinlichkeit der gluecklichste.  Die Frage ist,
**welche Stellschraube ueberhaupt wirkt**, und nur das laesst sich uebertragen.

Deshalb zwei Ansichten:

* eine Rangliste, streng gefiltert nach Robustheit;
* eine Aufschluesselung je Achse — was bringt der Horizont im Mittel ueber alle
  anderen Einstellungen, was `num_leaves`, was die Lernrate.

Dazu die Seed-Streuung: dieselbe Konfiguration mit zwei Zufallsstartwerten
lieferte im Test -1,55 % und -0,76 % ROI.  Ist der Unterschied zwischen zwei
Konfigurationen kleiner als der zwischen zwei Seeds derselben Konfiguration,
misst man Rauschen.

    cd /root/Neuropoly && .venv/bin/python sweep_report.py --log sweep/sweep_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

AXES = ("horizon", "num_leaves", "max_depth", "min_child_samples", "learning_rate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parametersuche auswerten.")
    p.add_argument("--log", default="sweep/sweep_log.jsonl")
    p.add_argument("--top", type=int, default=15)
    return p.parse_args()


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("val_roi") is not None:
                rows.append(e)
    return rows


def robust(e: dict) -> bool:
    """
    Ueberlebt diese Konfiguration die Robustheitspruefung?

    Der ROI ist schwanzgetragen — bei den Margenhebeln drehten 50 von 7.585
    Trades das Vorzeichen.  Eine Konfiguration zaehlt nur, wenn sie auch ohne
    ihre besten 50 Trades positiv bleibt und ihr Bootstrap-Intervall die Null
    ausschliesst.
    """
    return (
        (e.get("val_roi") or 0) > 0
        and (e.get("val_roi_drop50") or -1) > 0
        and (e.get("val_roi_ci_lo") or -1) > 0
    )


def fmt(v, pct=False, dash="   n/a"):
    if v is None:
        return dash
    return f"{v:>7.2%}" if pct else f"{v:>7.4f}"


def main() -> int:
    args = parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"nicht gefunden: {path}")
    rows = load(path)
    if not rows:
        raise SystemExit("keine auswertbaren Laeufe im Protokoll")

    n_robust = sum(robust(e) for e in rows)
    print("=" * 96)
    print("  PARAMETERSUCHE — AUSWERTUNG")
    print("=" * 96)
    print(f"  Laeufe protokolliert : {len(rows):,}")
    print(f"  davon robust positiv : {n_robust:,}  "
          f"(ROI > 0, auch ohne die 50 besten Trades, Bootstrap ohne Null)")

    # ── Seed-Streuung: die Messlatte fuer jeden Unterschied ──────────────
    by_cfg = defaultdict(list)
    for e in rows:
        key = tuple(e.get(a) for a in AXES)
        by_cfg[key].append(e["val_roi"])
    spreads = [max(v) - min(v) for v in by_cfg.values() if len(v) > 1]
    if spreads:
        print(f"\n  Seed-Streuung derselben Konfiguration: Median "
              f"{st.median(spreads):.2%}, p90 "
              f"{sorted(spreads)[int(len(spreads)*0.9)]:.2%}")
        print("  Unterschiede unterhalb dieser Groesse sind Rauschen, kein Signal.")

    # ── Rangliste ────────────────────────────────────────────────────────
    print(f"\n{'=' * 96}\n  RANGLISTE nach ROI auf Validierung\n{'=' * 96}")
    print(f"  {'run_id':<42}{'AUC':>8}{'ROI':>9}{'-50':>9}{'CI-lo':>9}"
          f"{'Trades':>8}{'Preis':>8}  robust")
    print("  " + "-" * 92)
    for e in sorted(rows, key=lambda x: -x["val_roi"])[: args.top]:
        print(f"  {e['run_id']:<42}{fmt(e.get('val_auc'))}{fmt(e['val_roi'], True)}"
              f"{fmt(e.get('val_roi_drop50'), True)}{fmt(e.get('val_roi_ci_lo'), True)}"
              f"{e.get('val_trades', 0):>8,}{e.get('val_median_price') or 0:>8.3f}"
              f"   {'ja' if robust(e) else 'nein'}")

    # ── Je Achse: wirkt die Stellschraube? ───────────────────────────────
    print(f"\n{'=' * 96}\n  JE STELLSCHRAUBE  (Mittel ueber alle anderen Einstellungen)\n{'=' * 96}")
    for axis in AXES:
        groups = defaultdict(list)
        for e in rows:
            groups[e.get(axis)].append(e)
        if len(groups) < 2:
            continue
        print(f"\n  {axis}")
        print(f"    {'Wert':>8}{'n':>6}{'O ROI':>10}{'bester ROI':>12}"
              f"{'O AUC':>9}{'robust':>8}")
        print("    " + "-" * 53)
        for val in sorted(groups, key=lambda v: (v is None, v)):
            g = groups[val]
            rois = [x["val_roi"] for x in g]
            aucs = [x["val_auc"] for x in g if x.get("val_auc") is not None]
            print(f"    {str(val):>8}{len(g):>6}{st.mean(rois):>10.2%}"
                  f"{max(rois):>12.2%}{(st.mean(aucs) if aucs else 0):>9.4f}"
                  f"{sum(robust(x) for x in g):>8}")

    print(f"\n{'=' * 96}")
    if n_robust == 0:
        print("  Keine Konfiguration besteht die Robustheitspruefung.")
        print("  Das ist ein Ergebnis, kein Fehler: es heisst, dass an diesen")
        print("  Stellschrauben nichts haengt, was einen Gewinn traegt — und dass")
        print("  der Hebel in der Ausfuehrung liegt, nicht im Modell.")
    else:
        print("  Kandidaten fuers Paper-Trading: die robusten Laeufe oben, dazu")
        print("  das heutige Modell als Vergleichsanker. Der Test-Split wird erst")
        print("  fuer den einen Gewinner angefasst.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
