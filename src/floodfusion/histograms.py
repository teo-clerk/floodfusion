r"""Score histograms of a scene: threshold-free metrics and per-scene self-calibration (v1.2 scorer).

v1.1 kept confusion counts at τ* and on a coarse grid of 19 thresholds. That was enough for its tests, but not to
ask whether a win survives a change of threshold: in v1.1, H4 flipped at the oracle threshold. v1.2 therefore keeps,
per scored pair, three histograms of each model's **calibrated** flood probability:

- ``pos``: scored pixels the reference maps as flood;
- ``neg``: scored pixels it maps as dry;
- ``all``: every valid pixel of the scene, labelled or not, for self-calibration.

Any threshold rule, the precision–recall curve and the oracle IoU then follow offline from the scores alone.

**Bins.** :data:`N_BINS` equal steps in **logit** over :data:`LOGIT_RANGE`, plus an underflow and an overflow bin.
Probability-uniform bins would put nearly every pixel of a scene in the first bin. One bin is 0.078 logit wide, the
resolution of every threshold computed here. Exact counts at τ* still come from the pixels themselves.

**Self-calibration.** The model's calibration assumed a deployment prior :math:`\pi_0`, and a scene has its own. The
EM of Saerens, Latinne and Decaestecker (2002, *Neural Computation* 14) estimates that prior from the scene's
unlabelled scores:

.. math:: \pi_{k+1} = \frac{1}{N}\sum_i \frac{w_k p_i}{w_k p_i + v_k (1 - p_i)}, \qquad
          w_k = \frac{\pi_k}{\pi_0},\quad v_k = \frac{1 - \pi_k}{1 - \pi_0}.

The shift is bounded, :math:`|\operatorname{logit}\hat\pi - \operatorname{logit}\pi_0| \le` :data:`MAX_PRIOR_SHIFT`,
because a miscalibrated model can drive EM to an extreme. Thresholding the re-weighted posterior at τ is the same as
thresholding the original one at :math:`\sigma(\operatorname{logit}\tau - \Delta)`.

Torch-free.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["EDGES", "LOGIT_RANGE", "MAX_PRIOR_SHIFT", "N_BINS", "average_precision", "confusion_above",
           "em_prior_shift", "histogram", "oracle_iou", "shifted_threshold"]

N_BINS = 256
LOGIT_RANGE = (-10.0, 10.0)
#: Inner bin edges in logit: bin 0 is everything below the first, bin N_BINS + 1 everything at or above the last.
EDGES = np.linspace(LOGIT_RANGE[0], LOGIT_RANGE[1], N_BINS + 1)
#: Bound on the self-calibration's logit shift of the scene prior.
MAX_PRIOR_SHIFT = 2.0
_EPS = 1e-12
_EDGE_TOL = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def histogram(prob: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``N_BINS + 2`` counts of the probabilities in ``prob`` (where ``mask`` is true and the value is finite)."""
    values = np.asarray(prob, dtype=np.float64)
    keep = np.isfinite(values) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(values))
    index = np.searchsorted(EDGES, _logit(values[keep]), side="right")
    return np.bincount(index, minlength=N_BINS + 2).astype(np.int64)


def _first_bin_at_or_above(threshold: float) -> int:
    """The bin whose lower edge is the first edge at or above ``logit(threshold)``: counts from there up are positive."""
    # A threshold that round-trips through a probability lands within float error of its edge: snap to it.
    return int(np.searchsorted(EDGES, float(_logit(np.array(threshold))) - _EDGE_TOL, side="left")) + 1


def confusion_above(pos: np.ndarray, neg: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """``(tp, fp, fn, tn)`` when every bin from the first edge at or above ``logit(threshold)`` counts as flood."""
    k = _first_bin_at_or_above(threshold)
    tp, fp = int(pos[k:].sum()), int(neg[k:].sum())
    return tp, fp, int(pos[:k].sum()), int(neg[:k].sum())


def _sweep(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True and false positives when the cut moves down from the top bin, one bin at a time."""
    return np.cumsum(np.asarray(pos)[::-1]), np.cumsum(np.asarray(neg)[::-1])


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    """AP, the step-wise area under the precision–recall curve with bins as tied scores; NaN without positives."""
    total = float(np.sum(pos))
    if total == 0:
        return float("nan")
    tp, fp = _sweep(pos, neg)
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros_like(tp, dtype=np.float64), where=flagged > 0)
    recall = tp / total
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def oracle_iou(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """The best IoU over bin edges, and that edge as a probability (chosen in-sample: optimistic by design)."""
    tp, fp = _sweep(pos, neg)
    fn = float(np.sum(pos)) - tp
    union = tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=np.float64), where=union > 0)
    best = int(np.argmax(iou))
    # Cut after reversed bin `best`: original bins N_BINS + 1 - best and up are flood; that bin's lower edge is
    # EDGES[N_BINS - best] (the underflow bin has no lower edge: everything is flood).
    edge = N_BINS - best
    tau = 0.0 if edge < 0 else 1.0 / (1.0 + math.exp(-float(EDGES[edge])))
    return float(iou[best]), tau


def _centres() -> np.ndarray:
    inner = (EDGES[:-1] + EDGES[1:]) / 2.0
    step = EDGES[1] - EDGES[0]
    return np.concatenate([[EDGES[0] - step / 2.0], inner, [EDGES[-1] + step / 2.0]])


def em_prior_shift(counts: np.ndarray, pi0: float, *, max_shift: float = MAX_PRIOR_SHIFT, tol: float = 1e-7,
                   max_iter: int = 500) -> float:
    """``Δ = logit π̂ − logit π0``: the scene's flood prior, estimated by EM from unlabelled scores, bounded."""
    if not 0.0 < pi0 < 1.0:
        raise ValueError(f"pi0 must lie in (0, 1), got {pi0}")
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = 1.0 / (1.0 + np.exp(-_centres()))
    lo, hi = (1.0 / (1.0 + math.exp(-(math.log(pi0 / (1 - pi0)) + s))) for s in (-max_shift, max_shift))
    pi = pi0
    for _ in range(max_iter):
        w, v = pi / pi0, (1.0 - pi) / (1.0 - pi0)
        posterior = w * p / (w * p + v * (1.0 - p))
        new = min(max(float(np.sum(counts * posterior) / n), lo), hi)
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return math.log(pi / (1.0 - pi)) - math.log(pi0 / (1.0 - pi0))


def shifted_threshold(tau: float, delta: float) -> float:
    """The threshold on the original posterior equivalent to ``tau`` on the posterior re-weighted by ``delta``."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must lie in (0, 1), got {tau}")
    return 1.0 / (1.0 + math.exp(-(math.log(tau / (1.0 - tau)) - delta)))
