"""
Backtest invariants.

The headline numbers have to describe the STRATEGY, not the funding
assumption.  These tests pin that down.

Run with:  python tests/test_backtest.py   (or: pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CostConfig
from pipeline.evaluation import backtest


def _signal(n: int = 3000, seed: int = 0):
    """A weak but real edge, with realistic sub-percent price moves."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    ret = np.where(y == 1, rng.normal(0.008, 0.004, n), rng.normal(-0.008, 0.004, n))
    pred = np.clip(0.5 + (y - 0.5) * 0.3 + rng.normal(0, 0.08, n), 0, 1)
    return y, pred, ret


_KW = dict(entry_threshold=0.55, fee_rate=0.02, max_position_usd=10.0,
           kelly_sizing=False)


def test_strategy_stats_do_not_depend_on_bankroll():
    """
    A bankroll too small to fund every trade used to break out of the loop,
    truncating the sample: with the default bankroll of 100 and a stake of 10
    that discarded about half the trades, and ROI swung by a factor of 500
    between bankroll settings while the strategy was identical.
    """
    y, pred, ret = _signal()

    results = [
        backtest(y, pred, trade_returns=ret, initial_bankroll=b, **_KW)
        for b in (100.0, 100_000.0, 10_000_000.0)
    ]

    trades = {r.total_trades for r in results}
    assert len(trades) == 1, f"trade count varies with bankroll: {trades}"

    rois = {round(r.roi, 12) for r in results}
    assert len(rois) == 1, f"ROI varies with bankroll: {rois}"

    assert len({round(r.win_rate, 12) for r in results}) == 1
    assert len({round(r.sharpe_ratio, 12) for r in results}) == 1

    # The bankroll simulation itself is still allowed to differ — that is
    # what roi_bankroll is for.
    assert results[0].ruined and not results[-1].ruined
    print(f"  bankroll-independent: {results[0].total_trades} trades, "
          f"ROI {results[0].roi:+.3%}")


def test_roi_is_order_independent():
    y, pred, ret = _signal()
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))

    a = backtest(y, pred, trade_returns=ret, initial_bankroll=100.0, **_KW)
    b = backtest(y[perm], pred[perm], trade_returns=ret[perm],
                 initial_bankroll=100.0, **_KW)

    assert a.total_trades == b.total_trades
    assert abs(a.roi - b.roi) < 1e-12, f"{a.roi} != {b.roi}"
    print(f"  order-independent:    ROI {a.roi:+.3%}")


def test_roi_equals_mean_return_minus_fee():
    """
    With fixed sizing, ROI on deployed capital must reduce exactly to
    mean realised return minus the round-trip cost.  A sanity anchor: if
    this drifts, the PnL formula changed.
    """
    y, pred, ret = _signal()
    bt = backtest(y, pred, trade_returns=ret, initial_bankroll=100.0, **_KW)

    expected = bt.mean_trade_return - _KW["fee_rate"]
    assert abs(bt.roi - expected) < 1e-9, f"{bt.roi} != {expected}"
    assert abs(bt.total_staked - bt.total_trades * _KW["max_position_usd"]) < 1e-6
    print(f"  ROI == mean return - fee: {bt.roi:+.3%} "
          f"= {bt.mean_trade_return:+.3%} - {_KW['fee_rate']:.1%}")


def test_cost_scales_with_one_over_price():
    """
    Trading costs are quoted in absolute price units, so their share of a
    position grows as the token gets cheaper.  A flat rate hides that, and
    it hides it in the dangerous direction: the extreme-priced markets that
    show the biggest percentage moves are also the most expensive to trade.

    Measured on 120 live order books, the median spread was ~0.010 in
    absolute terms; at a token price of 0.50 that is 2 % of the position,
    at 0.02 it is 50 %.
    """
    cost = CostConfig(spread_abs=0.010, fee_rate=0.04, fee_legs=2)

    mid = cost.round_trip_cost(0.50)
    edge = cost.round_trip_cost(0.02)
    assert mid < 0.15, f"cost at 0.50 should stay modest, got {mid:.1%}"
    assert edge > 5 * mid, (
        f"cost at 0.02 ({edge:.1%}) should dwarf cost at 0.50 ({mid:.1%})"
    )

    # Monotonically decreasing in price across the whole range
    prices = np.array([0.005, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5])
    costs = cost.round_trip_cost(prices)
    assert np.all(np.diff(costs) < 0), f"cost must fall as price rises: {costs}"
    print(f"  cost 0.50 -> {mid:.1%},  0.02 -> {edge:.1%}")


def test_untradeable_trades_are_skipped_not_booked():
    """
    Where the round-trip cost exceeds the position's value there is no trade
    to make.  Booking it as a near-total loss would be as wrong as pretending
    it was cheap, so it must be skipped and counted.
    """
    y, pred, ret = _signal(n=400)
    # Half the candidates sit at a price where the spread alone eats the
    # position; the other half in the liquid middle.
    prices = np.where(np.arange(len(y)) % 2 == 0, 0.004, 0.50)
    cost = CostConfig(spread_abs=0.010, fee_rate=0.04, fee_legs=2, max_cost=1.0)

    bt = backtest(y, pred, trade_returns=ret, entry_prices=prices, cost=cost,
                  initial_bankroll=100.0, **_KW)

    assert bt.skipped_untradeable > 0, "cheap-token trades were not skipped"
    assert bt.total_trades > 0, "liquid trades were skipped too"
    # Only the 0.50 trades survive, so the mean cost must match that price
    assert abs(bt.mean_cost - cost.round_trip_cost(0.50)) < 1e-9
    print(f"  skipped {bt.skipped_untradeable} untradeable, "
          f"kept {bt.total_trades} at mean cost {bt.mean_cost:.1%}")


def test_price_aware_cost_beats_flat_rate_where_it_matters():
    """A flat 2 % makes extreme-priced trades look profitable when they are not."""
    y, pred, ret = _signal(n=1000)
    prices = np.full(len(y), 0.02)          # cheap token: costly to trade
    cost = CostConfig(spread_abs=0.010, fee_rate=0.04, fee_legs=2, max_cost=10.0)

    flat = backtest(y, pred, trade_returns=ret, initial_bankroll=1e9, **_KW)
    aware = backtest(y, pred, trade_returns=ret, entry_prices=prices, cost=cost,
                     initial_bankroll=1e9, **_KW)

    assert flat.mean_cost < aware.mean_cost, (
        f"flat {flat.mean_cost:.1%} should understate {aware.mean_cost:.1%}"
    )
    assert aware.roi < flat.roi
    print(f"  flat cost {flat.mean_cost:.1%} -> ROI {flat.roi:+.1%};  "
          f"price-aware {aware.mean_cost:.1%} -> ROI {aware.roi:+.1%}")


def test_binary_payoff_is_tagged():
    """Calling without realised returns must be visible in the result."""
    y, pred, _ = _signal()
    bt = backtest(y, pred, initial_bankroll=100.0, **_KW)
    assert bt.payoff_model == "binary"
    print("  binary payoff tagged")


def test_sweep_picks_the_matching_side_and_price():
    """
    sweep_horizon's ceiling must pair each bucket's return with the price of
    the side that produced it.

    The tempting shortcut — |return of the dominant side|, priced at the
    dominant side — is wrong twice over, because a YES share costs p and a
    NO share 1-p.  At P(YES)=0.20 falling to 0.19 that shortcut claims a
    5.0 % move where the NO side really pays 1.25 %, and charges 13 % of
    cost where the NO side really costs 3.2 %.
    """
    from sweep_horizon import best_side

    # One bucket: dominant YES at 0.20 loses, the NO side is the better bet
    ret_dom = np.array([-0.05])          # (0.19-0.20)/0.20
    ret_opp = np.array([+0.0125])        # (0.20-0.19)/(1-0.20)
    px_dom = np.array([0.20])
    px_opp = np.array([0.80])

    ret, price = best_side(ret_dom, ret_opp, px_dom, px_opp)
    assert ret[0] == ret_opp[0], "must take the better side's return"
    assert price[0] == px_opp[0], "must take that same side's price"

    cost = CostConfig()
    assert cost.round_trip_cost(price[0]) < cost.round_trip_cost(px_dom[0])

    # The naive shortcut would have claimed a 4x larger move
    assert abs(ret_dom[0]) / ret_opp[0] == 4.0

    # And where the dominant side wins, it is kept
    ret2, price2 = best_side(np.array([0.03]), np.array([-0.01]),
                             np.array([0.40]), np.array([0.60]))
    assert ret2[0] == 0.03 and price2[0] == 0.40
    print("  sweep side/price pairing correct")


if __name__ == "__main__":
    print("=" * 60)
    print("  backtest invariants")
    print("=" * 60)
    test_strategy_stats_do_not_depend_on_bankroll()
    test_roi_is_order_independent()
    test_roi_equals_mean_return_minus_fee()
    test_cost_scales_with_one_over_price()
    test_untradeable_trades_are_skipped_not_booked()
    test_price_aware_cost_beats_flat_rate_where_it_matters()
    test_binary_payoff_is_tagged()
    test_sweep_picks_the_matching_side_and_price()
    print("  ALL PASSED")
