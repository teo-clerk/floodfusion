r"""Metrics for severe class imbalance, and basin-level uncertainty.

At a ~2 % positive rate the metrics you are used to behave differently, in ways that are
derived in ``docs/06`` section 2 and summarised here:

- **Accuracy is disqualified.** The constant "dry" predictor scores 0.98 with IoU 0.
- **IoU and F1 are monotonically related**, :math:`\mathrm{IoU} = F_1/(2-F_1)`, so they
  can *never* rank two models differently. Reporting both is harmless but adds no
  discriminating information.
- **Precision is fragile.** With :math:`\rho = 49` dry pixels per flooded pixel and
  recall 0.90, a false-positive rate of just 1 % costs 35 % of precision.
- **ROC misleads, PR does not.** ROC's :math:`FPR = FP/(FP+TN)` is diluted by the
  enormous :math:`TN`; precision's denominator contains no :math:`TN` term. Report
  average precision; do **not** report ROC-AUC.

Two rules that the implementation must enforce:

1. **Metrics are computed over valid pixels only** — layover, shadow, permanent water,
   nodata and out-of-AOI are excluded (``docs/03`` section 5.4).
2. **Accumulate confusion matrices per basin, then compute the metric once per basin.**
   Never average per-chip IoU: chips with no flood have undefined IoU, and their
   proportion then drives the mean (``docs/06`` section 2.6).

References
----------
``docs/06_baselines_and_metrics.md`` section 2, section 7.
``docs/09_evaluation_ablation_and_report.md`` section 3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# ``torch`` is needed only for the type annotations below, and this module carries
# ``from __future__ import annotations``, so those are lazy strings at runtime. Importing it
# eagerly would make every pure-arithmetic test in this module — confusion-matrix pooling, the
# IoU/F1 identity, micro-F1 against accuracy — unrunnable on a machine without a 2 GB deep-learning
# stack, which is precisely the set of tests that should run anywhere.
# ``utils/shapes.py`` uses the same deferral for the same reason.
if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch


__all__ = [
    "ConfusionMatrix",
    "accumulate_confusion",
    "average_precision",
    "basin_bootstrap_ci",
    "expected_calibration_error",
    "holm_adjust",
    "macro_iou",
    "paired_bootstrap_delta",
    "precision_recall_curve",
    "sign_flip_p_value",
    "sum_confusions",
]


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts over valid pixels. Immutable; combine with :meth:`__add__`.

    Attributes
    ----------
    tp, fp, fn, tn
        Counts. Use ``int`` (Python arbitrary precision) or ``int64``: a full IW frame
        has :math:`4 \\times 10^{8}` pixels and ``int32`` overflows when you aggregate
        several.
    """

    tp: int
    fp: int
    fn: int
    tn: int

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        return ConfusionMatrix(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tn + other.tn,
        )

    @property
    def iou(self) -> float:
        r""":math:`TP / (TP + FP + FN)`. Returns ``float('nan')`` if the union is empty.

        A ``nan`` here is meaningful and must not be silently coerced to 0: an empty
        union means neither the label nor the prediction contains any flood, for which
        IoU is genuinely undefined. Coercing to 0 penalises a correct prediction;
        coercing to 1 rewards a vacuous one. Propagate the ``nan`` and exclude such
        basins explicitly.
        """
        denom = self.tp + self.fp + self.fn
        return float("nan") if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        r""":math:`2TP / (2TP + FP + FN)`.

        Equals :math:`2\,\mathrm{IoU}/(1+\mathrm{IoU})`; verify this identity in
        ``tests/test_metrics.py`` rather than trusting it.
        """
        denom = 2 * self.tp + self.fp + self.fn
        return float("nan") if denom == 0 else 2 * self.tp / denom

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float("nan") if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float("nan") if denom == 0 else self.tp / denom


def accumulate_confusion(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    thresholds: tuple[float, ...],
) -> tuple[ConfusionMatrix, ...]:
    """Accumulate one confusion matrix per threshold, in a single pass.

    Parameters
    ----------
    probabilities, target, valid
        ``(B, 1, H, W)``. ``valid`` is ``bool``.
    thresholds
        Thresholds at which to evaluate (any order; output follows the input order).

    Returns
    -------
    tuple[ConfusionMatrix, ...]
        One per threshold, same order. A pixel is predicted flooded iff ``p > t``
        (strict, as in the Bayes rule of ``docs/00`` section 6.2).

    Notes
    -----
    **This function is why evaluation fits in memory.** The naive approach stores every
    probability map across the split to build a PR curve later; at
    :math:`2\\times10^{7}` pixels in fp32 that is 84 MB per copy and grows without
    bound, and it is the direct cause of the epoch-41 OOM in debugging challenge C2
    (``docs/10`` section 3). Accumulating counts instead is :math:`O(1)` in memory.

    Implementation: the valid pixels are split by label, each group is **sorted once**,
    and ``searchsorted`` counts the predictions above every threshold — :math:`O(N \\log
    N + T)` instead of :math:`O(T \\cdot N)`. The comparison happens in the probability's
    own dtype (a float32 ``0.05`` is *not* greater than the threshold ``0.05``). Counts
    are returned as Python ``int`` so aggregation across basins cannot overflow.
    """
    import torch

    if probabilities.shape != target.shape or probabilities.shape != valid.shape:
        raise ValueError(
            f"shape mismatch: probabilities {tuple(probabilities.shape)}, target "
            f"{tuple(target.shape)}, valid {tuple(valid.shape)}"
        )
    if valid.dtype != torch.bool:
        raise TypeError(f"valid must be bool, got {valid.dtype}")
    with torch.inference_mode():
        mask = valid.reshape(-1)
        probs = probabilities.reshape(-1)[mask].detach()
        labels = target.reshape(-1)[mask] > 0.5
        pos_sorted = torch.sort(probs[labels]).values
        neg_sorted = torch.sort(probs[~labels]).values
        n_pos, n_neg = int(pos_sorted.numel()), int(neg_sorted.numel())
        t = torch.as_tensor(thresholds, dtype=probs.dtype, device=probs.device)
        # number of elements <= t  ->  elements > t is the complement
        pos_le = torch.searchsorted(pos_sorted, t, right=True)
        neg_le = torch.searchsorted(neg_sorted, t, right=True)
        tps = (n_pos - pos_le).tolist()
        fps = (n_neg - neg_le).tolist()
    return tuple(
        ConfusionMatrix(tp=int(tp), fp=int(fp), fn=n_pos - int(tp), tn=n_neg - int(fp))
        for tp, fp in zip(tps, fps)
    )


def sum_confusions(matrices: Sequence[ConfusionMatrix]) -> ConfusionMatrix:
    """Pool confusion matrices (exact integer sum)."""
    total = ConfusionMatrix(0, 0, 0, 0)
    for m in matrices:
        total = total + m
    return total


def precision_recall_curve(
    matrices: tuple[ConfusionMatrix, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert per-threshold confusion matrices into a PR curve.

    Returns
    -------
    precisions, recalls
        Parallel tuples, ordered by increasing recall.

    Notes
    -----
    Points are ordered by the **number of predicted positives** (``tp + fp``), which is
    the same as ordering by decreasing threshold whatever order the thresholds were
    given in, and breaks recall ties correctly: at equal recall the point with fewer
    predictions (higher precision) is reached first. Points whose precision is
    undefined (no positive predictions at all) are dropped explicitly rather than
    letting a ``nan`` into the curve; they carry zero recall and contribute nothing to
    :func:`average_precision`.
    """
    defined = [m for m in matrices if (m.tp + m.fp) > 0]
    if any(m.tp + m.fn == 0 for m in defined):
        raise ValueError("recall is undefined: the evaluated set contains no positive pixels")
    ordered = sorted(defined, key=lambda m: (m.tp + m.fp, -m.tp))
    return tuple(m.precision for m in ordered), tuple(m.recall for m in ordered)


def average_precision(
    precisions: tuple[float, ...],
    recalls: tuple[float, ...],
) -> float:
    r"""Average precision :math:`\mathrm{AP} = \sum_k (R_k - R_{k-1}) P_k`.

    Notes
    -----
    This is the **step-wise** definition (with :math:`R_0 = 0`), not the trapezoidal
    one. The trapezoidal rule interpolates between operating points that are not
    jointly achievable and systematically over-estimates.
    """
    if len(precisions) != len(recalls):
        raise ValueError("precisions and recalls must have the same length")
    ap = 0.0
    previous_recall = 0.0
    for p, r in zip(precisions, recalls):
        if r < previous_recall - 1e-12:
            raise ValueError("recalls must be non-decreasing; use precision_recall_curve")
        ap += max(r - previous_recall, 0.0) * p
        previous_recall = max(previous_recall, r)
    if not -1e-9 <= ap <= 1.0 + 1e-9:
        raise AssertionError(f"average precision {ap} outside [0, 1]")
    return float(min(max(ap, 0.0), 1.0))


def macro_iou(per_basin: Mapping[str, ConfusionMatrix]) -> float:
    """Mean of per-basin IoU (``docs/06`` section 2.6), ignoring undefined basins.

    A basin with an empty union (no flood in label *or* prediction) has undefined IoU and
    is excluded explicitly rather than counted as 0 or 1. Returns ``nan`` if every basin
    is undefined.
    """
    values = [m.iou for m in per_basin.values()]
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _bootstrap_statistic(
    matrices: np.ndarray, statistic: str
) -> float:
    """IoU of a (k, 4) array of [tp, fp, fn, tn] rows, pooled or macro."""
    if statistic == "pooled":
        tp, fp, fn = matrices[:, 0].sum(), matrices[:, 1].sum(), matrices[:, 2].sum()
        denom = tp + fp + fn
        return float("nan") if denom == 0 else float(tp / denom)
    denom = matrices[:, 0] + matrices[:, 1] + matrices[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(denom > 0, matrices[:, 0] / np.where(denom > 0, denom, 1), np.nan)
    finite = per[np.isfinite(per)]
    return float(finite.mean()) if finite.size else float("nan")


def _as_array(per_basin: Mapping[str, ConfusionMatrix], keys: Sequence[str]) -> np.ndarray:
    return np.array(
        [[per_basin[k].tp, per_basin[k].fp, per_basin[k].fn, per_basin[k].tn] for k in keys],
        dtype=np.float64,
    )


def basin_bootstrap_ci(
    per_basin: dict[str, ConfusionMatrix],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: str = "pooled",
) -> tuple[float, float, float]:
    r"""Bootstrap a confidence interval by resampling **basins**.

    Parameters
    ----------
    per_basin
        One confusion matrix per held-out basin.
    n_boot
        Bootstrap replicates.
    alpha
        Two-sided significance level.
    seed
        For reproducibility. Record it.
    statistic
        ``"pooled"`` — sum the resampled basins' counts, then compute IoU (the procedure
        of ``docs/09`` section 3.2); ``"macro"`` — mean of the resampled basins' IoU (the
        headline of ``docs/06`` section 2.6).

    Returns
    -------
    point, lower, upper
        Point estimate on the original basin set and the percentile interval.

    Notes
    -----
    **Resample the unit of independence.** Pixels are not independent: neighbouring
    pixels share a flood, a land cover, an acquisition and often a label polygon
    (``docs/05`` section 6). Bootstrapping over :math:`N \approx 10^{7}` pixels gives an
    interval of half-width :math:`\sim 3\times10^{-4}`, so a 0.02 difference appears
    overwhelmingly significant. It is not. With :math:`K \approx 12` basins the interval
    widens by roughly :math:`\sqrt{10^7/12} \approx 900`, to something of order
    :math:`\pm 0.05` — and a 0.02 difference is then plainly inside the noise, which is
    the correct conclusion.

    The basin bootstrap implicitly assumes perfect within-basin correlation, so the
    interval is **conservative** (``docs/09`` section 3.1 bridge): an honest lower bound on
    precision, not a tight interval.
    """
    if statistic not in {"pooled", "macro"}:
        raise ValueError(f"statistic must be 'pooled' or 'macro', got {statistic!r}")
    if not per_basin:
        raise ValueError("basin_bootstrap_ci needs at least one basin")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    keys = sorted(per_basin)
    data = _as_array(per_basin, keys)
    point = _bootstrap_statistic(data, statistic)
    rng = np.random.default_rng(seed)
    k = len(keys)
    idx = rng.integers(0, k, size=(n_boot, k))
    replicates = np.array([_bootstrap_statistic(data[row], statistic) for row in idx])
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return point, float("nan"), float("nan")
    lower, upper = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lower), float(upper)


def paired_bootstrap_delta(
    per_basin_a: dict[str, ConfusionMatrix],
    per_basin_b: dict[str, ConfusionMatrix],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: str = "pooled",
) -> tuple[float, float, float]:
    r"""Bootstrap the **difference** :math:`\mathrm{IoU}_B - \mathrm{IoU}_A`.

    Returns
    -------
    delta, lower, upper
        Point difference and its percentile interval. If the interval contains 0, the
        difference is not resolvable at this sample size.

    Notes
    -----
    **This is the correct test for comparing two ablation rows**, and it is strictly
    more powerful than comparing two independent intervals. The models are evaluated on
    the *same* basins, so their errors are strongly correlated — a hard basin is hard for
    both — and resampling one basin set for both models cancels that shared difficulty:
    :math:`\operatorname{Var}[B-A] = \operatorname{Var}[A] + \operatorname{Var}[B] -
    2\operatorname{Cov}[A,B]`.

    One basin sample is drawn per replicate and used for **both** models; drawing two
    would silently revert to the unpaired test.

    Raises
    ------
    ValueError
        If the two models were not evaluated on identical basin sets — comparing models
        evaluated on different basins is meaningless.
    """
    if set(per_basin_a) != set(per_basin_b):
        only_a = sorted(set(per_basin_a) - set(per_basin_b))
        only_b = sorted(set(per_basin_b) - set(per_basin_a))
        raise ValueError(
            f"paired comparison needs identical basin sets; only in A: {only_a}, only in B: {only_b}"
        )
    if statistic not in {"pooled", "macro"}:
        raise ValueError(f"statistic must be 'pooled' or 'macro', got {statistic!r}")
    keys = sorted(per_basin_a)
    if not keys:
        raise ValueError("paired_bootstrap_delta needs at least one basin")
    a, b = _as_array(per_basin_a, keys), _as_array(per_basin_b, keys)
    delta = _bootstrap_statistic(b, statistic) - _bootstrap_statistic(a, statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    deltas = np.array(
        [_bootstrap_statistic(b[row], statistic) - _bootstrap_statistic(a[row], statistic) for row in idx]
    )
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return float(delta), float("nan"), float("nan")
    lower, upper = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(delta), float(lower), float(upper)


def sign_flip_p_value(deltas: Sequence[float], *, n_mc: int = 200_000, seed: int = 0) -> float:
    r"""Two-sided paired permutation test of :math:`H_0`: the mean per-basin difference is 0.

    Under :math:`H_0` (the two rows are exchangeable within each basin) each per-basin
    difference is equally likely to carry either sign. The p-value is the fraction of the
    :math:`2^K` sign patterns whose mean is at least as extreme as the observed one — exact
    for :math:`K \le 20` (enumerated), Monte Carlo above. Unlike a percentile bootstrap it
    is valid at any K; with few basins it simply cannot reach significance
    (:math:`p \ge 2^{1-K}`: K = 3 gives at least 0.25).
    """
    d = np.asarray([x for x in deltas if math.isfinite(x)], dtype=np.float64)
    k = len(d)
    if k == 0:
        return float("nan")
    observed = abs(d.mean())
    if k <= 20:
        signs = 1 - 2 * ((np.arange(2**k)[:, None] >> np.arange(k)) & 1)   # every pattern of +-1
    else:
        signs = np.random.default_rng(seed).choice((-1, 1), size=(n_mc, k))
    null = np.abs((signs * d).mean(axis=1))
    return float((null >= observed - 1e-12).mean())


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control across the gaps of a table)."""
    finite = sorted(((p, k) for k, p in p_values.items() if math.isfinite(p)))
    m = len(finite)
    adjusted: dict[str, float] = {k: float("nan") for k, p in p_values.items() if not math.isfinite(p)}
    running = 0.0
    for rank, (p, key) in enumerate(finite):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> tuple[float, dict[str, np.ndarray]]:
    r"""Expected calibration error and the reliability-diagram data (``docs/09`` 4.2).

    .. math:: \mathrm{ECE} = \sum_m \frac{n_m}{N}\,\bigl|\mathrm{freq}(m) - \mathrm{conf}(m)\bigr|

    **Bin by quantile, not by equal width**: at a 2 % positive rate nearly every
    prediction sits near zero, so equal-width bins put almost all mass in one bin and the
    diagram is uninformative.

    Returns
    -------
    ece, diagram
        ``diagram`` holds per-bin ``confidence``, ``frequency`` and ``count``.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have the same number of elements")
    if p.size == 0:
        raise ValueError("expected_calibration_error needs at least one sample")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
    if edges.size < 2:
        edges = np.array([p.min(), p.max() + 1e-12])
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    count = np.bincount(bins, minlength=edges.size - 1).astype(np.float64)
    conf = np.bincount(bins, weights=p, minlength=edges.size - 1)
    freq = np.bincount(bins, weights=y, minlength=edges.size - 1)
    nonempty = count > 0
    conf_mean = np.where(nonempty, conf / np.where(nonempty, count, 1), np.nan)
    freq_mean = np.where(nonempty, freq / np.where(nonempty, count, 1), np.nan)
    ece = float(np.nansum((count / count.sum()) * np.abs(freq_mean - conf_mean)))
    return ece, {"confidence": conf_mean, "frequency": freq_mean, "count": count, "edges": edges}
