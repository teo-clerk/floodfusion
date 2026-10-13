"""Scene score histograms: threshold-free metrics and the bounded EM self-calibration (v1.2 scorer)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion import histograms as h


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _on_centres(rng, n):
    """Probabilities at bin centres, so binning loses nothing and brute force is exact."""
    centres = (h.EDGES[:-1] + h.EDGES[1:]) / 2.0
    return _sigmoid(rng.choice(centres, size=n))


def test_histogram_bins_in_logit_with_under_and_overflow() -> None:
    prob = np.array([0.0, 1e-9, 0.5, 0.5, 1.0 - 1e-9, 1.0, np.nan])
    counts = h.histogram(prob)
    assert counts.shape == (h.N_BINS + 2,) and counts.sum() == 6            # NaN dropped
    assert counts[0] == 2 and counts[-1] == 2                               # beyond ±10 logit
    assert counts[h.N_BINS // 2 + 1] == 2                                   # p = 0.5 sits on the middle edge
    assert h.histogram(prob, np.array([1, 0, 0, 0, 0, 0, 1], bool)).sum() == 1


def test_confusion_and_ap_match_brute_force_on_exact_scores() -> None:
    rng = np.random.default_rng(0)
    prob = _on_centres(rng, 5000)
    label = rng.random(5000) < prob                                         # calibrated by construction
    pos, neg = h.histogram(prob, label), h.histogram(prob, ~label)
    for tau in (0.05, 0.25, 0.5, 0.9):
        edge = 1.0 / (1.0 + math.exp(-h.EDGES[np.searchsorted(h.EDGES, math.log(tau / (1 - tau)))]))
        flag = prob >= edge
        assert h.confusion_above(pos, neg, tau) == (int((flag & label).sum()), int((flag & ~label).sum()),
                                                    int((~flag & label).sum()), int((~flag & ~label).sum()))
    order = np.unique(prob)[::-1]
    tp = np.array([(label & (prob >= t)).sum() for t in order], dtype=float)
    fp = np.array([(~label & (prob >= t)).sum() for t in order], dtype=float)
    ap = np.sum(np.diff(np.concatenate([[0.0], tp / label.sum()])) * tp / (tp + fp))
    assert h.average_precision(pos, neg) == pytest.approx(ap, rel=1e-12)


def test_ap_is_one_when_separable_and_nan_without_flood() -> None:
    pos, neg = h.histogram(np.full(10, 0.9)), h.histogram(np.full(90, 0.1))
    assert h.average_precision(pos, neg) == pytest.approx(1.0)
    assert math.isnan(h.average_precision(np.zeros_like(pos), neg))


def test_oracle_iou_is_the_best_cut_and_reports_its_threshold() -> None:
    rng = np.random.default_rng(1)
    prob = _on_centres(rng, 4000)
    label = rng.random(4000) < prob
    pos, neg = h.histogram(prob, label), h.histogram(prob, ~label)
    best, tau = h.oracle_iou(pos, neg)
    brute = max(((label & (prob >= t)).sum() / ((prob >= t) | label).sum(), t) for t in np.unique(prob))
    assert best == pytest.approx(brute[0])
    tp, fp, fn, _ = h.confusion_above(pos, neg, tau)
    assert tp / (tp + fp + fn) == pytest.approx(best)
    assert h.oracle_iou(np.r_[5, np.zeros(h.N_BINS + 1)], np.zeros(h.N_BINS + 2)) == (1.0, 0.0)  # all in underflow


@pytest.mark.parametrize("pi_true", [0.02, 0.1, 0.3])
def test_em_recovers_a_scene_prior_from_unlabelled_scores(pi_true: float) -> None:
    rng = np.random.default_rng(2)
    pi0, n = 0.1, 200_000
    y = rng.random(n) < pi_true
    llr = np.where(y, rng.normal(2.0, 1.5, n), rng.normal(-2.0, 1.5, n))   # log-likelihood ratio, equal variances
    llr_true = llr * 4.0 / 1.5**2                                           # exact LLR of N(2, 1.5) vs N(-2, 1.5)
    prob = _sigmoid(llr_true + math.log(pi0 / (1 - pi0)))                   # calibrated under prior pi0
    delta = h.em_prior_shift(h.histogram(prob), pi0)
    est = 1.0 / (1.0 + math.exp(-(math.log(pi0 / (1 - pi0)) + delta)))
    assert est == pytest.approx(pi_true, rel=0.08)


def test_em_shift_is_bounded_and_empty_scenes_do_not_move() -> None:
    all_flood = h.histogram(np.full(1000, 0.999))
    assert h.em_prior_shift(all_flood, 0.05) == pytest.approx(h.MAX_PRIOR_SHIFT)
    assert h.em_prior_shift(np.zeros(h.N_BINS + 2), 0.05) == 0.0
    with pytest.raises(ValueError):
        h.em_prior_shift(all_flood, 1.0)


def test_shifted_threshold_moves_against_the_prior() -> None:
    assert h.shifted_threshold(0.25, 0.0) == pytest.approx(0.25)
    assert h.shifted_threshold(0.25, 1.0) < 0.25 < h.shifted_threshold(0.25, -1.0)   # more flood: lower bar
    with pytest.raises(ValueError):
        h.shifted_threshold(1.0, 0.0)
