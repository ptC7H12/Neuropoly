"""
Evaluation edge cases.

Run with:  python tests/test_evaluation.py   (or: pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.evaluation import evaluate


def test_single_class_split_does_not_raise():
    """
    A split can legitimately hold one class only — a short window, or heavy
    gap exclusion.  roc_auc_score raises there, and evaluate_model.py had no
    guard, so the run died AFTER the whole preprocessing pipeline had
    finished.  Report NaN and carry on instead.
    """
    rng = np.random.default_rng(0)
    for y in (np.ones(50), np.zeros(50)):
        m = evaluate(y, rng.random(50))
        assert not m.roc_auc_defined
        assert np.isnan(m.roc_auc)
        # Everything that does not need two classes must still be usable
        assert 0.0 <= m.brier_score <= 1.0
        assert 0.0 <= m.accuracy <= 1.0
        assert np.isfinite(m.log_loss)
        assert m.calibration_predicted, "calibration curve should still exist"
    print("  single-class split returns NaN AUC instead of raising")


def test_two_class_split_is_unaffected():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 25, dtype=float)
    m = evaluate(y, rng.random(50))
    assert m.roc_auc_defined
    assert np.isfinite(m.roc_auc)
    print(f"  two-class split unchanged (AUC {m.roc_auc:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("  evaluation edge cases")
    print("=" * 60)
    test_single_class_split_does_not_raise()
    test_two_class_split_is_unaffected()
    print("  ALL PASSED")
