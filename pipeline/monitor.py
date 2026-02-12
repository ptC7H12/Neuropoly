"""
Live training monitor with rich terminal dashboard.

Displays in real-time during LightGBM training:
- Iteration progress bar
- Train/Val loss curves (text-based)
- Current best iteration & score
- Top feature importances
- Elapsed time & ETA

Also logs metrics to a JSONL file for later analysis.
"""

import json
import time
from pathlib import Path
from typing import Optional

from config import MonitorConfig


class TrainingMonitor:
    """
    LightGBM callback that provides live training feedback.

    Usage:
        monitor = TrainingMonitor(cfg)
        model.fit(X, y, callbacks=[monitor.callback()])
        monitor.finish()
    """

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.metrics_history: list[dict] = []
        self.start_time: float = 0.0
        self.best_score: float = float("inf")
        self.best_iteration: int = 0
        self.total_iterations: int = 0
        self.feature_names: list[str] = []
        self._log_file = None

        if cfg.log_file:
            self._log_file = open(cfg.log_file, "w")

        # Try to import rich for pretty output
        self._rich_available = False
        self._live = None
        self._table = None
        if cfg.rich_dashboard:
            try:
                from rich.live import Live
                from rich.table import Table
                from rich.panel import Panel
                from rich.console import Console
                from rich.layout import Layout
                from rich.text import Text
                self._rich_available = True
                self._console = Console()
            except ImportError:
                pass

    def set_feature_names(self, names: list[str]) -> None:
        self.feature_names = names

    def set_total_iterations(self, n: int) -> None:
        self.total_iterations = n

    def callback(self):
        """Return a LightGBM-compatible callback function."""
        self.start_time = time.time()

        def _callback(env):
            iteration = env.iteration + 1
            results = {}

            # Extract evaluation results
            for data_name, eval_name, value, is_higher_better in env.evaluation_result_list:
                key = f"{data_name}_{eval_name}"
                results[key] = value

                # Track best score (using validation metric)
                if data_name == "valid_0":
                    if is_higher_better:
                        if value > self.best_score or self.best_score == float("inf"):
                            self.best_score = value
                            self.best_iteration = iteration
                    else:
                        if value < self.best_score:
                            self.best_score = value
                            self.best_iteration = iteration

            results["iteration"] = iteration
            results["elapsed_s"] = time.time() - self.start_time
            self.metrics_history.append(results)

            # Log to file
            if self._log_file:
                self._log_file.write(json.dumps(results) + "\n")
                self._log_file.flush()

            # Display update
            if iteration % self.cfg.log_interval == 0 or iteration == 1:
                self._display(env, iteration, results)

        return _callback

    def _display(self, env, iteration: int, results: dict) -> None:
        """Render the current training state."""

        elapsed = time.time() - self.start_time
        iters_per_sec = iteration / elapsed if elapsed > 0 else 0
        total = self.total_iterations or "?"

        if self._rich_available:
            self._display_rich(env, iteration, results, elapsed, iters_per_sec, total)
        else:
            self._display_plain(iteration, results, elapsed, iters_per_sec, total)

    def _display_plain(
        self, iteration: int, results: dict, elapsed: float, ips: float, total
    ) -> None:
        """Simple text output."""

        parts = [f"[{iteration:>5}/{total}]"]
        parts.append(f"  {elapsed:>6.1f}s")
        parts.append(f"  ({ips:.1f} it/s)")

        for key, value in results.items():
            if key in ("iteration", "elapsed_s"):
                continue
            parts.append(f"  {key}: {value:.6f}")

        parts.append(f"  best: {self.best_score:.6f} @{self.best_iteration}")

        print(" | ".join(parts))

    def _display_rich(
        self, env, iteration: int, results: dict, elapsed: float, ips: float, total
    ) -> None:
        """Rich terminal dashboard output."""

        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.columns import Columns

        # Progress info
        if isinstance(total, int) and total > 0:
            pct = iteration / total * 100
            eta = (total - iteration) / ips if ips > 0 else 0
            progress_text = (
                f"Iteration {iteration}/{total}  ({pct:.1f}%)  "
                f"| {elapsed:.0f}s elapsed  | ETA {eta:.0f}s  "
                f"| {ips:.1f} it/s"
            )
        else:
            progress_text = (
                f"Iteration {iteration}  | {elapsed:.0f}s elapsed  | {ips:.1f} it/s"
            )

        # Metrics table
        metrics_table = Table(title="Metrics", show_header=True, expand=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green", justify="right")

        for key, value in sorted(results.items()):
            if key in ("iteration", "elapsed_s"):
                continue
            metrics_table.add_row(key, f"{value:.6f}")

        metrics_table.add_row(
            "best_score", f"{self.best_score:.6f}", style="bold yellow"
        )
        metrics_table.add_row(
            "best_iteration", str(self.best_iteration), style="bold yellow"
        )

        # Feature importance (if available)
        panels = [Panel(metrics_table, title="Training Progress")]

        if (
            self.cfg.show_feature_importance
            and hasattr(env, "model")
            and self.feature_names
        ):
            try:
                importance = env.model.feature_importance(importance_type="gain")
                if len(importance) == len(self.feature_names):
                    pairs = sorted(
                        zip(self.feature_names, importance),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    fi_table = Table(
                        title=f"Top {self.cfg.top_n_features} Features",
                        show_header=True,
                    )
                    fi_table.add_column("Feature", style="cyan")
                    fi_table.add_column("Importance", style="magenta", justify="right")
                    fi_table.add_column("Bar", style="blue")

                    max_imp = pairs[0][1] if pairs else 1
                    for name, imp in pairs[: self.cfg.top_n_features]:
                        bar_len = int(imp / max_imp * 20) if max_imp > 0 else 0
                        bar = "█" * bar_len
                        fi_table.add_row(name, f"{imp:.0f}", bar)

                    panels.append(Panel(fi_table, title="Feature Importance"))
            except Exception:
                pass

        # Print using rich
        self._console.clear()
        self._console.print(
            Panel(Text(progress_text, style="bold"), title="LightGBM Training")
        )
        for p in panels:
            self._console.print(p)

        # Mini loss curve (last 20 points)
        self._print_loss_curve()

    def _print_loss_curve(self) -> None:
        """Print a simple text-based loss curve from recent history."""

        if len(self.metrics_history) < 2:
            return

        from rich.panel import Panel

        # Get validation AUC history (or logloss)
        val_key = None
        for key in self.metrics_history[-1]:
            if key.startswith("valid_0"):
                val_key = key
                break

        if not val_key:
            return

        recent = self.metrics_history[-40:]
        values = [m.get(val_key, 0) for m in recent]

        if not values:
            return

        # Normalize to 0-1 range for display
        min_v = min(values)
        max_v = max(values)
        if max_v == min_v:
            return

        height = 8
        width = len(values)
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        for i, v in enumerate(values):
            row = int((v - min_v) / (max_v - min_v) * (height - 1))
            row = height - 1 - row  # Invert (top = high)
            canvas[row][i] = "●"

        lines = ["".join(row) for row in canvas]
        curve_text = f"  {max_v:.4f} ┤{lines[0]}\n"
        for line in lines[1:-1]:
            curve_text += f"          │{line}\n"
        curve_text += f"  {min_v:.4f} ┤{lines[-1]}"

        self._console.print(
            Panel(curve_text, title=f"{val_key} (last {len(values)} checkpoints)")
        )

    def finish(self) -> dict:
        """Finalize monitoring, close files, return summary."""

        if self._log_file:
            self._log_file.close()

        elapsed = time.time() - self.start_time if self.start_time else 0

        summary = {
            "total_iterations": len(self.metrics_history),
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "elapsed_seconds": elapsed,
            "log_file": self.cfg.log_file,
        }

        if self._rich_available:
            from rich.panel import Panel
            self._console.print(
                Panel(
                    f"Training complete!\n"
                    f"  Best score: {self.best_score:.6f} at iteration {self.best_iteration}\n"
                    f"  Total time: {elapsed:.1f}s\n"
                    f"  Log file: {self.cfg.log_file}",
                    title="Training Summary",
                    style="bold green",
                )
            )
        else:
            print(f"\n{'='*50}")
            print(f"Training complete!")
            print(f"  Best score: {self.best_score:.6f} @ iteration {self.best_iteration}")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"  Log file: {self.cfg.log_file}")
            print(f"{'='*50}")

        return summary

    def get_history_df(self):
        """Return metrics history as a polars DataFrame."""
        import polars as pl
        if not self.metrics_history:
            return pl.DataFrame()
        return pl.DataFrame(self.metrics_history)
