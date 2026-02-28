"""
Evaluation metrics and backtesting simulation.

Metrics:
- ROC AUC, Log Loss, Brier Score
- Calibration curve
- Accuracy at threshold

Backtesting:
- Simulated trading based on P(win) threshold
- Kelly criterion position sizing
- ROI, Sharpe ratio, max drawdown tracking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""

    accuracy: float = 0.0
    roc_auc: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Calibration
    calibration_bins: list[float] = field(default_factory=list)
    calibration_predicted: list[float] = field(default_factory=list)
    calibration_actual: list[float] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Container for backtesting results."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Time series
    equity_curve: list[float] = field(default_factory=list)
    trade_pnls: list[float] = field(default_factory=list)


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> EvalMetrics:
    """
    Compute evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred_proba: Predicted probabilities P(win)
        threshold: Classification threshold
    """
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        log_loss,
        brier_score_loss,
        precision_score,
        recall_score,
        f1_score,
    )

    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = EvalMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_pred_proba),
        log_loss=log_loss(y_true, y_pred_proba),
        brier_score=brier_score_loss(y_true, y_pred_proba),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )

    # Calibration curve
    metrics = _add_calibration(metrics, y_true, y_pred_proba, n_bins=10)

    return metrics


def _add_calibration(
    metrics: EvalMetrics,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> EvalMetrics:
    """Compute calibration curve (predicted vs actual win rate per bin)."""

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_predicted = []
    bin_actual = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_predicted.append(y_pred_proba[mask].mean())
        bin_actual.append(y_true[mask].mean())

    metrics.calibration_bins = bin_centers
    metrics.calibration_predicted = bin_predicted
    metrics.calibration_actual = bin_actual

    return metrics


def backtest(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    entry_threshold: float = 0.6,
    fee_rate: float = 0.02,
    max_position_usd: float = 100.0,
    kelly_sizing: bool = True,
    kelly_cap: float = 0.25,
    initial_bankroll: float = 10000.0,
) -> BacktestResult:
    """
    Simulate trading based on model predictions.

    Trading rules:
    - Enter trade if P(win) > entry_threshold
    - Position size: Kelly criterion or fixed
    - Win: +payout - fee
    - Loss: -stake - fee
    """

    bankroll = initial_bankroll
    equity_curve = [bankroll]
    trade_pnls = []
    peak = bankroll
    max_dd = 0.0

    winning = 0
    losing = 0
    total = 0

    for i in range(len(y_true)):
        p_win = y_pred_proba[i]

        if p_win < entry_threshold:
            equity_curve.append(bankroll)
            continue

        # Position sizing
        if kelly_sizing:
            # Kelly fraction: f* = (p*b - q) / b
            # For binary outcome with even odds: f* = 2p - 1
            # With fee adjustment
            b = 1.0 - fee_rate  # Net odds
            q = 1.0 - p_win
            kelly_f = (p_win * b - q) / b
            kelly_f = max(0, min(kelly_f, kelly_cap))
            stake = bankroll * kelly_f
        else:
            stake = max_position_usd

        stake = min(stake, max_position_usd, bankroll * 0.5)

        if stake < 1.0 or bankroll < 10.0:
            equity_curve.append(bankroll)
            continue

        total += 1
        actual_win = y_true[i]
        fee = stake * fee_rate

        if actual_win == 1:
            # Win: Polymarket zahlt 1:1, fee wird auf den Gewinn berechnet
            pnl = stake * (1.0 - fee_rate)
            winning += 1
        else:
            # Loss: Einsatz verloren, keine zusätzliche Fee
            pnl = -stake
            losing += 1

        bankroll += pnl
        trade_pnls.append(pnl)
        equity_curve.append(bankroll)

        # Track drawdown
        if bankroll > peak:
            peak = bankroll
        dd = peak - bankroll
        if dd > max_dd:
            max_dd = dd

    # Compute summary statistics
    result = BacktestResult(
        total_trades=total,
        winning_trades=winning,
        losing_trades=losing,
        win_rate=winning / total if total > 0 else 0.0,
        total_pnl=bankroll - initial_bankroll,
        roi=(bankroll - initial_bankroll) / initial_bankroll if initial_bankroll > 0 else 0.0,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd / peak if peak > 0 else 0.0,
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
    )

    # Sharpe ratio (annualized, assuming ~288 buckets per day for 5-min)
    if trade_pnls:
        pnl_arr = np.array(trade_pnls)
        if pnl_arr.std() > 0:
            daily_factor = np.sqrt(288)  # 5-min to daily
            result.sharpe_ratio = (pnl_arr.mean() / pnl_arr.std()) * daily_factor

    return result


def print_evaluation(metrics: EvalMetrics, bt: BacktestResult) -> None:
    """Print formatted evaluation results."""

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    print("\n  Classification Metrics:")
    print(f"    ROC AUC:      {metrics.roc_auc:.4f}")
    print(f"    Log Loss:     {metrics.log_loss:.4f}")
    print(f"    Brier Score:  {metrics.brier_score:.4f}")
    print(f"    Accuracy:     {metrics.accuracy:.4f}")
    print(f"    Precision:    {metrics.precision:.4f}")
    print(f"    Recall:       {metrics.recall:.4f}")
    print(f"    F1:           {metrics.f1:.4f}")

    print("\n  Calibration (predicted → actual):")
    for pred, act in zip(metrics.calibration_predicted, metrics.calibration_actual):
        bar_pred = "█" * int(pred * 30)
        bar_act = "▓" * int(act * 30)
        print(f"    P={pred:.2f} → A={act:.2f}  {bar_pred}|{bar_act}")

    print("\n" + "-" * 60)
    print("  BACKTEST RESULTS")
    print("-" * 60)
    print(f"    Total trades:     {bt.total_trades}")
    print(f"    Win rate:         {bt.win_rate:.2%}")
    print(f"    Total PnL:        ${bt.total_pnl:,.2f}")
    print(f"    ROI:              {bt.roi:.2%}")
    print(f"    Sharpe Ratio:     {bt.sharpe_ratio:.2f}")
    print(f"    Max Drawdown:     ${bt.max_drawdown:,.2f} ({bt.max_drawdown_pct:.2%})")

    # Mini equity curve
    if bt.equity_curve:
        _print_equity_curve(bt.equity_curve)

    print("=" * 60)


def _print_equity_curve(equity: list[float], width: int = 50, height: int = 10) -> None:
    """Print a simple text-based equity curve."""

    if len(equity) < 2:
        return

    # Sample to fit width
    step = max(1, len(equity) // width)
    sampled = equity[::step]

    min_v = min(sampled)
    max_v = max(sampled)
    if max_v == min_v:
        return

    print("\n  Equity Curve:")

    canvas = [[" " for _ in range(len(sampled))] for _ in range(height)]

    for i, v in enumerate(sampled):
        row = int((v - min_v) / (max_v - min_v) * (height - 1))
        row = height - 1 - row
        canvas[row][i] = "●"

    for i, line in enumerate(canvas):
        if i == 0:
            label = f"${max_v:>10,.0f}"
        elif i == height - 1:
            label = f"${min_v:>10,.0f}"
        else:
            label = " " * 11
        print(f"    {label} │{''.join(line)}")

    print(f"    {'':>11} └{'─' * len(sampled)}")
