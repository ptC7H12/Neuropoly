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
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import CostConfig


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

    # Trades that actually made money after costs.  Diverges from win_rate
    # because `win` only asks whether the price moved the right way, not
    # whether the move was big enough to cover the fee.
    profitable_trades: int = 0
    profit_rate: float = 0.0
    mean_trade_return: float = 0.0
    # Mean round-trip cost actually charged, as a fraction of the position
    mean_cost: float = 0.0
    # Candidate trades dropped because their cost exceeded CostConfig.max_cost
    skipped_untradeable: int = 0

    total_pnl: float = 0.0
    total_staked: float = 0.0
    # Return on the capital actually deployed (sum of PnL / sum of stakes).
    # Independent of bankroll size and row order, so it describes the
    # strategy rather than the funding assumption.
    roi: float = 0.0
    # Growth of the simulated bankroll.  Depends on initial_bankroll,
    # max_position_usd and the order the rows happen to be in — read it as
    # context for the equity curve, not as a measure of the strategy.
    roi_bankroll: float = 0.0
    sharpe_ratio: float = 0.0          # per trade, NOT annualised
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # "return" = realised price move (correct);  "binary" = legacy even-money
    # placeholder, only used when no trade returns were supplied.
    payoff_model: str = "return"
    # True if the run stopped early because the bankroll was exhausted
    # True if the simulated bankroll ran out.  Strategy statistics are still
    # computed over every qualifying trade; only the equity curve stops.
    ruined: bool = False

    # Time series (one entry per EXECUTED trade, not per candidate row)
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
    trade_returns: Optional[np.ndarray] = None,
    entry_prices: Optional[np.ndarray] = None,
    cost: Optional["CostConfig"] = None,
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
    - Enter a trade if P(win) >= entry_threshold
    - Position size: fixed, or Kelly (binary payoff model only)
    - PnL = stake * (realised return - fee_rate)

    `trade_returns` is the realised return of the position over the label's
    forward window (see pipeline/labeling.py:trade_return).  It is what makes
    the result economically meaningful: a bucket where the price moves from
    0.500 to 0.501 is a `win`, but it pays 0.2 % — not 100 %.

    If `trade_returns` is None the function falls back to the old even-money
    binary payoff (+stake on a win, -stake on a loss).  That model does NOT
    describe a prediction market and massively overstates ROI; the result is
    tagged with payoff_model="binary" so callers can flag it.

    Costs
    -----
    With `entry_prices` (the price of the token held) and a `CostConfig`, the
    round-trip cost is computed per trade and scales with 1/price — a spread
    is quoted in absolute price units, so it eats a far larger share of a
    position at 0.02 than at 0.50.  Trades whose cost exceeds
    CostConfig.max_cost are skipped as untradeable and counted in
    `skipped_untradeable`.

    Without entry prices the flat `fee_rate` is used, which understates the
    cost of exactly the trades that look most attractive.
    """

    use_returns = trade_returns is not None

    # Per-trade round-trip cost, as a fraction of the position
    if entry_prices is not None and cost is not None:
        costs = np.asarray(cost.round_trip_cost(np.asarray(entry_prices)),
                           dtype=np.float64)
        if len(costs) != len(y_true):
            raise ValueError(
                f"entry_prices has {len(costs)} rows but y_true has "
                f"{len(y_true)} — they must be aligned row for row."
            )
        max_cost = cost.max_cost
    else:
        costs = np.full(len(y_true), float(fee_rate), dtype=np.float64)
        max_cost = np.inf

    if use_returns:
        trade_returns = np.asarray(trade_returns, dtype=np.float64)
        if len(trade_returns) != len(y_true):
            raise ValueError(
                f"trade_returns has {len(trade_returns)} rows but y_true has "
                f"{len(y_true)} — they must be aligned row for row."
            )
        if kelly_sizing:
            # Kelly's f* = 2p - 1 assumes an even-money binary bet.  With real
            # price returns the payoff is asymmetric and tiny, so that formula
            # is meaningless here.  Fall back to fixed sizing rather than
            # invent a number.
            print(
                "  NOTE: kelly_sizing is not supported with realised returns "
                "— using fixed position sizing instead."
            )
            kelly_sizing = False

    bankroll = initial_bankroll
    equity_curve = [bankroll]
    trade_pnls = []
    trade_rets = []
    peak = bankroll
    max_dd = 0.0

    winning = 0
    losing = 0
    profitable = 0
    total = 0
    total_staked = 0.0
    skipped_untradeable = 0
    trade_costs = []
    ruined = False

    for i in range(len(y_true)):
        p_win = y_pred_proba[i]

        if p_win < entry_threshold:
            continue

        # A row without a realised return cannot be traded in the simulation
        if use_returns and not np.isfinite(trade_returns[i]):
            continue

        trade_cost = costs[i]
        if not np.isfinite(trade_cost) or trade_cost > max_cost:
            # The spread alone would swallow the position — no such trade
            # exists in practice, so booking it as a near-total loss would be
            # as wrong as pretending it was cheap.
            skipped_untradeable += 1
            continue

        # Position sizing
        if kelly_sizing:
            # Kelly sizes off the running bankroll, so this path cannot
            # continue once the bankroll is gone.
            if ruined:
                break
            # Kelly fraction: f* = (p*b - q) / b
            # For binary outcome with even odds: f* = 2p - 1
            # With fee adjustment
            b = 1.0 - trade_cost  # Net odds
            q = 1.0 - p_win
            kelly_f = (p_win * b - q) / b
            kelly_f = max(0, min(kelly_f, kelly_cap))
            stake = min(bankroll * kelly_f, max_position_usd, bankroll * 0.5)
            if stake < 1.0 or bankroll < 10.0:
                ruined = True
                break
        else:
            # Fixed sizing: the trade does not depend on the bankroll, so the
            # strategy statistics must not either.  The old code broke out of
            # the loop on ruin, which truncated the sample — with the default
            # bankroll of 100 and a stake of 10 that discarded roughly half
            # the trades and made win rate, profit rate and Sharpe describe a
            # prefix of the data instead of the strategy.
            stake = max_position_usd

        total += 1
        total_staked += stake
        actual_win = y_true[i]

        if use_returns:
            # Realised price move minus round-trip cost (fee + spread proxy),
            # both as a fraction of the notional stake.
            ret = float(trade_returns[i])
            pnl = stake * (ret - trade_cost)
            trade_rets.append(ret)
        else:
            # Legacy even-money model — kept only for backwards compatibility
            if actual_win == 1:
                pnl = stake * (1.0 - trade_cost)
            else:
                pnl = -stake
            trade_rets.append(pnl / stake)
        trade_costs.append(trade_cost)

        if actual_win == 1:
            winning += 1
        else:
            losing += 1
        if pnl > 0:
            profitable += 1

        trade_pnls.append(pnl)

        # Bankroll simulation runs alongside the strategy statistics and
        # freezes once it is exhausted — it drives the equity curve and the
        # drawdown, nothing else.
        if not ruined:
            bankroll += pnl
            equity_curve.append(bankroll)

            if bankroll > peak:
                peak = bankroll
            dd = peak - bankroll
            if dd > max_dd:
                max_dd = dd
            if bankroll < 10.0:
                ruined = True

    # Compute summary statistics
    result = BacktestResult(
        total_trades=total,
        winning_trades=winning,
        losing_trades=losing,
        win_rate=winning / total if total > 0 else 0.0,
        profitable_trades=profitable,
        profit_rate=profitable / total if total > 0 else 0.0,
        mean_trade_return=float(np.mean(trade_rets)) if trade_rets else 0.0,
        mean_cost=float(np.mean(trade_costs)) if trade_costs else 0.0,
        skipped_untradeable=skipped_untradeable,
        total_pnl=float(np.sum(trade_pnls)) if trade_pnls else 0.0,
        total_staked=total_staked,
        roi=(float(np.sum(trade_pnls)) / total_staked) if total_staked > 0 else 0.0,
        roi_bankroll=(
            (bankroll - initial_bankroll) / initial_bankroll
            if initial_bankroll > 0 else 0.0
        ),
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd / peak if peak > 0 else 0.0,
        payoff_model="return" if use_returns else "binary",
        ruined=ruined,
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
    )

    # Sharpe ratio PER TRADE (mean / std of trade returns).
    # Deliberately NOT annualised: the backtest has no trade timestamps, so
    # any scaling factor would be made up.
    if len(trade_rets) > 1:
        ret_arr = np.array(trade_rets)
        if ret_arr.std() > 0:
            result.sharpe_ratio = float(ret_arr.mean() / ret_arr.std())

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
    if bt.payoff_model == "binary":
        print("    !! PAYOFF MODEL: binary (even money) — NOT a prediction market.")
        print("       No realised returns were supplied; ROI is meaningless here.")
    print(f"    Total trades:     {bt.total_trades}")
    print(f"    Win rate:         {bt.win_rate:.2%}   (price moved the right way)")
    print(f"    Profitable:       {bt.profit_rate:.2%}   (after {'' if bt.payoff_model == 'binary' else 'the '}fee)")
    print(f"    Mean return/trade:{bt.mean_trade_return:+.4%}")
    print(f"    Mean cost/trade:  {bt.mean_cost:.4%}   (spread + fee, scales with 1/price)")
    if bt.skipped_untradeable:
        print(f"    Skipped:          {bt.skipped_untradeable} candidate(s) whose cost "
              f"exceeded the position value")
    print(f"    Total PnL:        ${bt.total_pnl:,.2f}  on ${bt.total_staked:,.2f} staked")
    print(f"    ROI on capital:   {bt.roi:.2%}   <- the strategy")
    print(f"    Sharpe (per trade):{bt.sharpe_ratio:.3f}")
    print(f"    Max Drawdown:     ${bt.max_drawdown:,.2f} ({bt.max_drawdown_pct:.2%})")
    print(f"    Bankroll growth:  {bt.roi_bankroll:.2%}   (depends on "
          f"initial_bankroll / max_position_usd)")
    if bt.ruined:
        print("    !! Bankroll exhausted — the equity curve stops there. Trade")
        print("       statistics above still cover every qualifying trade.")

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
