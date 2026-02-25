"""
pipeline/results_logger.py — Append evaluation results to a JSONL log file
and display historical comparison tables.

Each run appends one JSON object (one line) to the log file.
The log file can be shared between evaluate_model.py and
benchmark_strategies.py so you get a unified history in one place.

Entry types
-----------
  type == "model"  →  written by evaluate_model.py
  type == "bench"  →  written by benchmark_strategies.py

Log format (one JSON object per line):
  model entry:
    {"type": "model", "ts": "2024-01-15T14:23:00Z", "model": "model.txt",
     "trades": "data/trades.parquet", "threshold": 0.6, "fwd_buckets": 6,
     "train_auc": 0.72, "train_roi": 0.085, "train_sharpe": 1.1,
     "val_auc": 0.68,   "val_roi":   0.041,
     "test_auc": 0.651, "test_roi":  0.032, "test_sharpe": 1.21,
     "test_trades": 432, "test_win_rate": 0.63}

  bench entry:
    {"type": "bench", "ts": "2024-01-15T14:30:00Z",
     "trades": "data/trades.parquet", "threshold": 0.6, "fwd_buckets": 6,
     "n_test": 4500,
     "baseline_roi": -0.012, "baseline_auc": 0.50, "baseline_trades": 4500,
     "momentum_roi":  0.021, ...}
"""

from __future__ import annotations

import json
from pathlib import Path

# Strategy names in display order (must match benchmark_strategies.py STRATEGIES list)
_BENCH_STRATS = [
    "baseline", "random", "momentum", "reversion",
    "volume", "closing", "contrarian",
]

_W = 78  # table width


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def append_to_log(log_path: str, entry: dict) -> None:
    """Append *entry* as a single JSON line to *log_path*."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _load_entries(log_path: str) -> list[dict]:
    try:
        raw = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
    except FileNotFoundError:
        return []
    entries: list[dict] = []
    for line in raw:
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_log_history(log_path: str, max_rows: int = 10) -> None:
    """
    Print a compact comparison table of past runs from *log_path*.

    Shows the last *max_rows* model evaluations and the last *max_rows*
    strategy benchmarks (if any exist).
    """
    entries = _load_entries(log_path)
    if not entries:
        return

    model_runs = [e for e in entries if e.get("type") == "model"]
    bench_runs = [e for e in entries if e.get("type") == "bench"]

    if model_runs:
        tail = model_runs[-max_rows:]
        print(f"\n{'─' * _W}")
        print(f"  Model evaluation history  "
              f"({len(tail)} of {len(model_runs)} run(s) shown)")
        print(f"{'─' * _W}")
        print(
            f"  {'Timestamp':<22}"
            f" {'Model':<16}"
            f" {'Thr':>5}"
            f" {'TestAUC':>8}"
            f" {'TestROI':>9}"
            f" {'Sharpe':>7}"
            f" {'Trades':>7}"
        )
        print(
            f"  {'─'*22} {'─'*16} {'─'*5} {'─'*8} {'─'*9} {'─'*7} {'─'*7}"
        )
        for e in tail:
            auc = e.get("test_auc", float("nan"))
            roi = e.get("test_roi", float("nan"))
            shp = e.get("test_sharpe", float("nan"))
            trd = e.get("test_trades", 0)
            try:
                roi_str = f"{roi:>+9.1%}"
            except (ValueError, TypeError):
                roi_str = f"{'N/A':>9}"
            try:
                shp_str = f"{shp:>7.2f}"
            except (ValueError, TypeError):
                shp_str = f"{'N/A':>7}"
            try:
                auc_str = f"{auc:>8.4f}"
            except (ValueError, TypeError):
                auc_str = f"{'N/A':>8}"
            print(
                f"  {e.get('ts', '?'):<22}"
                f" {Path(e.get('model', '')).name:<16}"
                f" {e.get('threshold', 0):>5.0%}"
                f" {auc_str}"
                f" {roi_str}"
                f" {shp_str}"
                f" {trd:>7}"
            )

    if bench_runs:
        tail = bench_runs[-max_rows:]
        print(f"\n{'─' * _W}")
        print(f"  Strategy benchmark history  "
              f"({len(tail)} of {len(bench_runs)} run(s) shown)  — ROI per strategy")
        print(f"{'─' * _W}")
        hdr = f"  {'Timestamp':<22} {'Thr':>5}"
        for s in _BENCH_STRATS:
            hdr += f" {s[:8]:>9}"
        print(hdr)
        print(f"  {'─'*22} {'─'*5}" + "".join(f" {'─'*9}" for _ in _BENCH_STRATS))
        for e in tail:
            row = f"  {e.get('ts', '?'):<22} {e.get('threshold', 0):>5.0%}"
            for s in _BENCH_STRATS:
                v = e.get(f"{s}_roi")
                try:
                    row += f" {v:>+9.1%}" if v is not None else f" {'N/A':>9}"
                except (ValueError, TypeError):
                    row += f" {'ERR':>9}"
            print(row)

    if model_runs or bench_runs:
        print(f"{'─' * _W}")
