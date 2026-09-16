"""
Sample weighting for the training rows.

The weighting itself turned out not to help — see
docs/universe-and-grouping.md — but the code stays, because the negative
result is worth being able to reproduce, and because `weight_mode="none"`
has to keep reproducing the unweighted model exactly.

Run with:  python tests/test_weights.py   (or: pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.model import WEIGHT_MODES, sample_weights

_RETURNS = np.array([0.0, 0.001, 0.01, 0.05, 0.5, np.nan, -0.02, np.inf])


def test_none_means_no_weights():
    """The default must leave training byte-identical to the unweighted path."""
    assert sample_weights(_RETURNS, "none") is None
    assert sample_weights(_RETURNS, None) is None
    print("  none yields no weight vector at all")


def test_mean_is_normalised_to_one():
    """LightGBM scales gradients by the weight, so an un-normalised vector
    would silently act as a different learning rate — and a grid would then be
    comparing weighting against learning rate rather than against itself."""
    for mode in ("abs", "sqrt"):
        w = sample_weights(_RETURNS, mode)
        assert abs(w.mean() - 1.0) < 0.05, (mode, w.mean())
    print("  abs and sqrt both normalise to mean 1")


def test_no_row_is_weighted_to_zero():
    """A weight of exactly 0 removes the row from training entirely. Deadband
    rows have |return| near 0 and would vanish — weighting should emphasise,
    not discard."""
    for mode in ("abs", "sqrt"):
        w = sample_weights(_RETURNS, mode)
        assert w.min() > 0.0, mode
    print("  every row keeps a positive weight")


def test_non_finite_returns_survive():
    """trade_return carries NaN for unlabelled rows and can carry inf from a
    division near zero; neither may poison the whole weight vector."""
    for mode in ("abs", "sqrt"):
        w = sample_weights(_RETURNS, mode)
        assert np.isfinite(w).all(), mode
        assert len(w) == len(_RETURNS)
    print("  NaN and inf become finite weights")


def test_direction_does_not_matter():
    """Weighting is about the size of the move, not its sign — a loss of the
    same magnitude teaches as much as a gain."""
    w = sample_weights(np.array([0.02, -0.02]), "abs")
    assert np.isclose(w[0], w[1])
    print("  a gain and a loss of equal size weigh the same")


def test_bad_input_raises_instead_of_silently_passing():
    for bad in ("quatsch", "ABS", ""):
        try:
            sample_weights(_RETURNS, bad)
        except ValueError:
            continue
        raise AssertionError(f"unknown mode {bad!r} was accepted")
    try:
        sample_weights(None, "abs")
    except ValueError:
        pass
    else:
        raise AssertionError("missing trade_return was accepted")
    print("  unknown modes and missing returns raise")


def test_every_declared_mode_works():
    for mode in WEIGHT_MODES:
        sample_weights(_RETURNS, mode)   # must not raise
    print(f"  all {len(WEIGHT_MODES)} declared modes are callable")


if __name__ == "__main__":
    print("=" * 60)
    print("  sample weighting")
    print("=" * 60)
    test_none_means_no_weights()
    test_mean_is_normalised_to_one()
    test_no_row_is_weighted_to_zero()
    test_non_finite_returns_survive()
    test_direction_does_not_matter()
    test_bad_input_raises_instead_of_silently_passing()
    test_every_declared_mode_works()
    print("  ALL PASSED")
