"""Conformal risk control of the per-chip false-negative rate."""

from __future__ import annotations

import numpy as np
import pytest

from floodfusion.calibration import CONFORMAL_MIN_POSITIVES, conformal_fnr_threshold

Z = np.linspace(-6.0, 6.0, 121)


def _chips(rng: np.random.Generator, n: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """n chips; each has flood pixels whose scores ~ N(mu_i, 1.5); FN(z) = flood pixels scoring <= z."""
    positives = rng.integers(CONFORMAL_MIN_POSITIVES, 2000, n)
    fn = np.empty((n, len(Z)))
    for i, p in enumerate(positives):
        scores = rng.normal(rng.normal(2.0 + shift, 1.0), 1.5, p)
        fn[i] = np.searchsorted(np.sort(scores), Z, side="right")
    return fn, positives.astype(float)


def test_largest_threshold_meeting_the_bound() -> None:
    fn = np.array([[0, 1, 5, 10], [0, 0, 2, 10]], float)
    pos = np.array([100, 100], float) * 1.0 + CONFORMAL_MIN_POSITIVES - 100
    out = conformal_fnr_threshold(fn, pos, np.array([-1.0, 0.0, 1.0, 2.0]), alpha=0.4)
    # risk = [0, .005, .035, .1]; bound = (2 r + 1)/3 = [.333, .337, .357, .4] -> the largest <= 0.4 is z = 2.
    assert out["score_threshold"] == 2.0 and out["n"] == 2 and out["feasible"]
    assert conformal_fnr_threshold(fn, pos, np.array([-1.0, 0.0, 1.0, 2.0]), alpha=0.35)["score_threshold"] == 0.0


def test_infeasible_and_empty_calibration_sets() -> None:
    fn = np.full((3, 4), 50.0)
    pos = np.full(3, 100.0)
    out = conformal_fnr_threshold(fn, pos, np.arange(4.0), alpha=0.1)
    assert not out["feasible"] and out["score_threshold"] == 0.0
    tiny = conformal_fnr_threshold(fn, np.full(3, 5.0), np.arange(4.0), alpha=0.1)
    assert tiny["n"] == 0 and not tiny["feasible"]


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="ascending"):
        conformal_fnr_threshold(np.zeros((2, 3)), np.full(2, 200.0), np.array([0.0, 2.0, 1.0]), alpha=0.1)
    with pytest.raises(ValueError, match="alpha"):
        conformal_fnr_threshold(np.zeros((2, 3)), np.full(2, 200.0), np.arange(3.0), alpha=1.5)


def test_the_guarantee_holds_on_exchangeable_chips_and_breaks_under_shift() -> None:
    rng = np.random.default_rng(0)
    alpha, risks, shifted = 0.1, [], []
    for _ in range(150):
        fn_cal, pos_cal = _chips(rng, 40)
        z = conformal_fnr_threshold(fn_cal, pos_cal, Z, alpha=alpha)["score_threshold"]
        k = int(np.searchsorted(Z, z))
        fn_new, pos_new = _chips(rng, 1)
        risks.append(fn_new[0, k] / pos_new[0])
        fn_far, pos_far = _chips(rng, 1, shift=-1.5)          # a new event: flood scores lower
        shifted.append(fn_far[0, k] / pos_far[0])
    assert np.mean(risks) <= alpha + 0.02                     # marginal guarantee (Monte Carlo slack)
    assert np.mean(shifted) > alpha + 0.05                    # it does not transfer to a shifted event
