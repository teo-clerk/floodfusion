r"""Metrics under a ~2 % positive rate, and basin-level uncertainty.

The specification for :mod:`floodfusion.metrics`. See ``docs/06`` section 2 and
``docs/09`` section 3.

:class:`ConfusionMatrix` is given complete, so its tests pass on a fresh clone; the
accumulation, curve and bootstrap functions are skeletons.
"""

from __future__ import annotations

import math

import pytest

from floodfusion.metrics import (
    ConfusionMatrix,
    accumulate_confusion,
    average_precision,
    basin_bootstrap_ci,
    paired_bootstrap_delta,
    precision_recall_curve,
)

# --------------------------------------------------------------------------
# Optional dependency handling
# --------------------------------------------------------------------------
# ``floodfusion.metrics`` names ``torch`` in annotations only, so this module imports
# without PyTorch. A module-level ``pytest.importorskip`` would raise ``Skipped`` during
# collection and take the whole file with it, including the confusion-matrix algebra and
# the basin bootstrap — neither of which constructs a tensor. The guard therefore lives
# inside each test that builds one, following the ``_require_torch`` idiom of
# ``conftest.py``, so that only the tests needing tensors are skipped.


def _require_torch():
    """Return :mod:`torch`, skipping only the test that asked for it.

    Call it as the first statement of a test body, never at module level: a ``Skipped``
    raised while this module is being imported is attributed to the module as a whole,
    and would take the fourteen torch-free tests below with it.
    """
    return pytest.importorskip("torch", reason="PyTorch is required for this test")


class TestConfusionMatrix:
    def test_iou_definition(self) -> None:
        assert ConfusionMatrix(tp=10, fp=5, fn=5, tn=100).iou == pytest.approx(0.5)

    def test_f1_definition(self) -> None:
        assert ConfusionMatrix(tp=10, fp=5, fn=5, tn=100).f1 == pytest.approx(2 / 3)

    @pytest.mark.parametrize("tp,fp,fn", [(10, 5, 5), (1, 99, 1), (50, 1, 1), (7, 3, 11)])
    def test_iou_f1_identity(self, tp: int, fp: int, fn: int) -> None:
        r""":math:`\mathrm{IoU} = F_1/(2-F_1)`, so the two can NEVER rank models differently."""
        cm = ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=1000)
        assert cm.iou == pytest.approx(cm.f1 / (2 - cm.f1))

    @pytest.mark.parametrize("tp,fp,fn", [(10, 5, 5), (1, 99, 1), (7, 3, 11)])
    def test_f1_always_exceeds_iou(self, tp: int, fp: int, fn: int) -> None:
        cm = ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=1000)
        assert cm.f1 > cm.iou

    def test_perfect_prediction(self) -> None:
        cm = ConfusionMatrix(tp=100, fp=0, fn=0, tn=900)
        assert cm.iou == pytest.approx(1.0)
        assert cm.precision == pytest.approx(1.0)
        assert cm.recall == pytest.approx(1.0)

    def test_null_predictor_scores_zero_iou_despite_high_accuracy(self) -> None:
        """Why accuracy is disqualified: 0.98 accuracy, 0.00 IoU (``docs/06`` 2.2)."""
        cm = ConfusionMatrix(tp=0, fp=0, fn=20, tn=980)
        accuracy = (cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn)
        assert accuracy == pytest.approx(0.98)
        assert cm.iou == pytest.approx(0.0)
        assert cm.recall == pytest.approx(0.0)

    def test_empty_union_is_nan_not_zero(self) -> None:
        """Neither label nor prediction contains flood: IoU is genuinely undefined."""
        assert math.isnan(ConfusionMatrix(tp=0, fp=0, fn=0, tn=1000).iou)

    def test_matrices_add(self) -> None:
        a = ConfusionMatrix(1, 2, 3, 4)
        b = ConfusionMatrix(10, 20, 30, 40)
        assert (a + b) == ConfusionMatrix(11, 22, 33, 44)

    @pytest.mark.parametrize(
        ("fpr", "expected_precision"),
        [(0.001, 0.948), (0.005, 0.786), (0.01, 0.647), (0.05, 0.269)],
    )
    def test_precision_collapses_under_imbalance(
        self, fpr: float, expected_precision: float
    ) -> None:
        r"""A 1 % false-positive rate costs 35 % of precision at :math:`\rho = 49`.

        This is the arithmetic behind every warning in docs 01 and 02 about false
        positives: dry pixels outnumber wet ones 49:1, so a small rate on a large
        population overwhelms the positive class (``docs/06`` section 2.4).
        """
        n_pos, rho, recall = 10_000, 49, 0.90
        tp = int(recall * n_pos)
        fp = int(fpr * rho * n_pos)
        cm = ConfusionMatrix(tp=tp, fp=fp, fn=n_pos - tp, tn=rho * n_pos - fp)
        assert cm.precision == pytest.approx(expected_precision, abs=0.01)


class TestAccumulation:
    def test_returns_one_matrix_per_threshold(self) -> None:
        torch = _require_torch()

        probs = torch.rand(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) < 0.2).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        thresholds = (0.25, 0.5, 0.75)
        assert len(accumulate_confusion(probs, target, valid, thresholds)) == 3

    def test_counts_only_valid_pixels(self) -> None:
        """Invalid pixels are excluded from the METRIC as well as the loss."""
        torch = _require_torch()

        probs = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        valid[0, 0, :2, :2] = True
        (cm,) = accumulate_confusion(probs, target, valid, (0.5,))
        assert cm.tp + cm.fp + cm.fn + cm.tn == 4

    def test_higher_threshold_cannot_increase_recall(self) -> None:
        torch = _require_torch()

        probs = torch.rand(1, 1, 16, 16)
        target = (torch.rand(1, 1, 16, 16) < 0.3).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        low, high = accumulate_confusion(probs, target, valid, (0.2, 0.8))
        assert low.recall >= high.recall


class TestCurves:
    def test_average_precision_is_bounded(self) -> None:
        torch = _require_torch()

        probs = torch.rand(1, 1, 32, 32)
        target = (torch.rand(1, 1, 32, 32) < 0.1).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        thresholds = tuple(i / 20 for i in range(1, 20))
        p, r = precision_recall_curve(accumulate_confusion(probs, target, valid, thresholds))
        assert 0.0 <= average_precision(p, r) <= 1.0

    def test_perfect_ranking_gives_average_precision_one(self) -> None:
        torch = _require_torch()

        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, :2, :] = 1.0
        probs = target.clone() * 0.9 + 0.05
        valid = torch.ones_like(target, dtype=torch.bool)
        thresholds = tuple(i / 20 for i in range(1, 20))
        p, r = precision_recall_curve(accumulate_confusion(probs, target, valid, thresholds))
        assert average_precision(p, r) == pytest.approx(1.0, abs=1e-6)


class TestBootstrap:
    def _basins(self) -> dict[str, ConfusionMatrix]:
        return {
            f"b{i}": ConfusionMatrix(tp=100 + 10 * i, fp=40, fn=30, tn=5000)
            for i in range(12)
        }

    def test_interval_brackets_the_point_estimate(self) -> None:
        point, lo, hi = basin_bootstrap_ci(self._basins(), n_boot=500)
        assert lo <= point <= hi

    def test_basin_bootstrap_is_far_wider_than_a_pixel_bootstrap(self) -> None:
        """The central statistical point of ``docs/09`` section 3.1.

        Pixels are not independent samples. Resampling ~1e7 of them yields an interval
        of half-width ~3e-4, so a 0.02 difference looks overwhelmingly significant. It
        is not. The effective sample size is the number of BASINS.
        """
        _, lo, hi = basin_bootstrap_ci(self._basins(), n_boot=2000)
        assert (hi - lo) > 1e-3, (
            "an interval this narrow means you resampled pixels, not basins"
        )

    def test_paired_delta_of_identical_models_is_zero(self) -> None:
        basins = self._basins()
        delta, lo, hi = paired_bootstrap_delta(basins, basins, n_boot=500)
        assert delta == pytest.approx(0.0, abs=1e-9)
        assert lo <= 0.0 <= hi

    def test_paired_delta_requires_identical_basin_sets(self) -> None:
        """Comparing models evaluated on different basins is meaningless."""
        a = self._basins()
        b = {k: v for k, v in a.items() if k != "b0"}
        with pytest.raises(ValueError):
            paired_bootstrap_delta(a, b, n_boot=100)

    def test_is_reproducible_for_a_given_seed(self) -> None:
        basins = self._basins()
        assert basin_bootstrap_ci(basins, n_boot=200, seed=3) == basin_bootstrap_ci(
            basins, n_boot=200, seed=3
        )
