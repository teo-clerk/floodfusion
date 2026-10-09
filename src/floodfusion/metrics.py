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

from typing import TYPE_CHECKING
from dataclasses import dataclass

# ``torch`` is needed only for the type annotations below, and this module carries
# ``from __future__ import annotations``, so those are lazy strings at runtime. Importing it
# eagerly would make every pure-arithmetic test in this module — confusion-matrix pooling, the
# IoU/F1 identity, micro-F1 against accuracy — unrunnable on a machine without a 2 GB deep-learning
# stack, which is precisely the set of tests a student should be able to run on day one.
# ``utils/shapes.py`` uses the same deferral for the same reason.
if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch


__all__ = [
    "ConfusionMatrix",
    "accumulate_confusion",
    "average_precision",
    "basin_bootstrap_ci",
    "paired_bootstrap_delta",
    "precision_recall_curve",
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

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
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
        Ascending thresholds at which to evaluate.

    Returns
    -------
    tuple[ConfusionMatrix, ...]
        One per threshold, same order.

    Notes
    -----
    **This function is why evaluation fits in memory.** The naive approach stores every
    probability map across the split to build a PR curve later; at
    :math:`2\\times10^{7}` pixels in fp32 that is 84 MB per copy and grows without
    bound, and it is the direct cause of the epoch-41 OOM in debugging challenge C2
    (``docs/10`` section 3). Accumulating counts instead is :math:`O(1)` in memory.

    Do this under ``torch.inference_mode()``.

    TODO(student)
    -------------
    1. Mask to valid pixels once, outside the threshold loop.
    2. For each threshold, compute the four counts. Sorting the probabilities once and
       using ``searchsorted`` turns the loop from :math:`O(T \\cdot N)` into
       :math:`O(N \\log N + T)`, which matters at 199 thresholds.
    3. Return Python ``int`` counts so that aggregation across basins cannot overflow.
    """
    raise NotImplementedError("Implement confusion accumulation. See docs/08 section 7.3.")


def precision_recall_curve(
    matrices: tuple[ConfusionMatrix, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert per-threshold confusion matrices into a PR curve.

    Returns
    -------
    precisions, recalls
        Parallel tuples, ordered by increasing recall.

    TODO(student)
    -------------
    1. Map each matrix to (precision, recall).
    2. Sort by recall; handle the undefined-precision endpoint explicitly rather than
       letting a ``nan`` into the curve.
    """
    raise NotImplementedError("Implement the PR curve. See docs/06 section 7.")


def average_precision(
    precisions: tuple[float, ...],
    recalls: tuple[float, ...],
) -> float:
    r"""Average precision :math:`\mathrm{AP} = \sum_k (R_k - R_{k-1}) P_k`.

    Notes
    -----
    This is the **step-wise** definition, not the trapezoidal one. The trapezoidal rule
    interpolates between operating points that are not jointly achievable and
    systematically over-estimates. Use the step form and say which you used.

    TODO(student)
    -------------
    1. Accumulate the recall-weighted precision.
    2. Assert the result lies in ``[0, 1]``.
    """
    raise NotImplementedError("Implement average precision. See docs/06 section 7.3.")


def basin_bootstrap_ci(
    per_basin: dict[str, ConfusionMatrix],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
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

    Returns
    -------
    point, lower, upper
        Point estimate and percentile interval.

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

    **Aggregate counts, then compute the metric.** Do not average per-basin IoU values
    across the resample: that weights basins by nothing meaningful.

    TODO(student)
    -------------
    1. Sample :math:`K` basin keys with replacement, :math:`n_{\mathrm{boot}}` times.
    2. Sum their confusion matrices, then compute IoU.
    3. Return the point estimate and the ``[alpha/2, 1-alpha/2]`` percentiles.
    4. Verify against the pixel bootstrap and report the width ratio — ``docs/09``
       exercise 1.
    """
    raise NotImplementedError("Implement the basin bootstrap. See docs/09 section 3.2.")


def paired_bootstrap_delta(
    per_basin_a: dict[str, ConfusionMatrix],
    per_basin_b: dict[str, ConfusionMatrix],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
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
    both — and resampling one basin set for both models cancels that shared difficulty.

    Two models can have heavily overlapping individual CIs while their difference
    interval excludes zero.

    TODO(student)
    -------------
    1. Draw **one** basin sample per replicate and use it for **both** models. Drawing
       two independent samples silently reverts to the unpaired test.
    2. Assert both dictionaries have identical key sets, and raise if not — comparing
       models evaluated on different basins is meaningless.
    """
    raise NotImplementedError("Implement the paired bootstrap. See docs/09 section 3.3.")
