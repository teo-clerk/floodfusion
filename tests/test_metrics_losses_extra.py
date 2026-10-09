"""Cross-checks for the metrics and losses beyond the given specification."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from floodfusion.config import LossConfig
from floodfusion.metrics import (
    ConfusionMatrix,
    accumulate_confusion,
    average_precision,
    basin_bootstrap_ci,
    expected_calibration_error,
    macro_iou,
    paired_bootstrap_delta,
    precision_recall_curve,
)
from floodfusion.models.losses import CompoundFocalDiceLoss, FocalLoss, SoftDiceLoss


class TestFocalAgainstTorchvision:
    @pytest.mark.parametrize(("gamma", "alpha"), [(2.0, 0.75), (0.0, 0.5), (1.0, 0.25)])
    def test_matches_sigmoid_focal_loss(self, gamma: float, alpha: float) -> None:
        tv_ops = pytest.importorskip("torchvision.ops")
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(2, 1, 16, 16, generator=g) * 3
        target = (torch.rand(2, 1, 16, 16, generator=g) < 0.2).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        ours = FocalLoss(gamma=gamma, alpha=alpha)(logits, target, valid)
        reference = tv_ops.sigmoid_focal_loss(logits, target, alpha=alpha, gamma=gamma, reduction="mean")
        assert ours.item() == pytest.approx(reference.item(), rel=1e-5)

    def test_y0_gradient_matches_the_derivation(self) -> None:
        r"""docs/10 A4b: for y=0, dL/dz = (1-alpha) p^gamma [p - gamma (1-p) log(1-p)]."""
        z = torch.tensor([[[[0.7]]]], dtype=torch.float64, requires_grad=True)
        target = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        gamma, alpha = 2.0, 0.75
        FocalLoss(gamma=gamma, alpha=alpha)(z, target, valid).backward()
        p = torch.sigmoid(z.detach())
        analytic = (1 - alpha) * p**gamma * (p - gamma * (1 - p) * torch.log(1 - p))
        assert z.grad.item() == pytest.approx(analytic.item(), rel=1e-9)


class TestDiceDetails:
    def test_denominator_only_smoothing_would_give_zero_gradient(self) -> None:
        """The trap, demonstrated: 2A/(B+s) on an empty chip has exactly zero gradient."""
        logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        p = torch.sigmoid(logits)
        trap = 1 - (2 * (p * 0).sum()) / (p.sum() + 1.0)
        trap.backward()
        assert logits.grad.abs().sum().item() == 0.0
        ours_logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        SoftDiceLoss(1.0)(ours_logits, torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8, dtype=torch.bool)).backward()
        assert ours_logits.grad.abs().sum().item() > 0.0

    def test_fully_invalid_batch_is_finite(self) -> None:
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        total, _ = CompoundFocalDiceLoss(LossConfig())(logits, torch.zeros(1, 1, 4, 4), valid)
        total.backward()
        assert torch.isfinite(total) and torch.isfinite(logits.grad).all()

    def test_row12_weight_zero_still_reports_dice(self) -> None:
        cfg = LossConfig(dice_weight=0.0, focal_gamma=0.0, focal_alpha=0.5)
        logits = torch.randn(1, 1, 8, 8)
        target = (torch.rand(1, 1, 8, 8) < 0.3).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        total, parts = CompoundFocalDiceLoss(cfg)(logits, target, valid)
        assert total.item() == pytest.approx(parts["focal"].item())
        assert 0.0 <= parts["dice"].item() <= 1.0

    def test_rejects_float_mask(self) -> None:
        x = torch.zeros(1, 1, 2, 2)
        with pytest.raises(TypeError):
            CompoundFocalDiceLoss(LossConfig())(x, x, x)


class TestMetricExtras:
    def test_threshold_order_does_not_matter(self) -> None:
        g = torch.Generator().manual_seed(1)
        probs = torch.rand(1, 1, 16, 16, generator=g)
        target = (torch.rand(1, 1, 16, 16, generator=g) < 0.3).float()
        valid = torch.ones_like(target, dtype=torch.bool)
        up = accumulate_confusion(probs, target, valid, (0.2, 0.5, 0.8))
        down = accumulate_confusion(probs, target, valid, (0.8, 0.5, 0.2))
        assert up == tuple(reversed(down))

    def test_matches_a_direct_count(self) -> None:
        g = torch.Generator().manual_seed(2)
        probs = torch.rand(2, 1, 8, 8, generator=g)
        target = (torch.rand(2, 1, 8, 8, generator=g) < 0.4).float()
        valid = torch.rand(2, 1, 8, 8, generator=g) < 0.8
        (cm,) = accumulate_confusion(probs, target, valid, (0.37,))
        pred = probs > 0.37
        y = target > 0.5
        assert cm.tp == int((pred & y & valid).sum())
        assert cm.fp == int((pred & ~y & valid).sum())
        assert cm.fn == int((~pred & y & valid).sum())
        assert cm.tn == int((~pred & ~y & valid).sum())

    def test_macro_iou_ignores_undefined_basins(self) -> None:
        per_basin = {"a": ConfusionMatrix(10, 0, 10, 100), "b": ConfusionMatrix(0, 0, 0, 100)}
        assert macro_iou(per_basin) == pytest.approx(0.5)

    def test_macro_bootstrap_brackets_point(self) -> None:
        basins = {f"b{i}": ConfusionMatrix(50 + 10 * i, 20, 20, 1000) for i in range(8)}
        point, lo, hi = basin_bootstrap_ci(basins, n_boot=300, statistic="macro")
        assert lo <= point <= hi
        assert point == pytest.approx(macro_iou(basins))

    def test_paired_test_detects_a_consistent_improvement(self) -> None:
        """Overlapping individual CIs, yet a difference interval that excludes zero."""
        a = {f"b{i}": ConfusionMatrix(100 + 30 * i, 40, 40, 5000) for i in range(10)}
        b = {k: ConfusionMatrix(v.tp + 6, v.fp - 3, v.fn - 6, v.tn + 3) for k, v in a.items()}
        delta, lo, _hi = paired_bootstrap_delta(a, b, n_boot=1000)
        assert delta > 0 and lo > 0
        _, _a_lo, a_hi = basin_bootstrap_ci(a, n_boot=1000)
        _, b_lo, _b_hi = basin_bootstrap_ci(b, n_boot=1000)
        assert b_lo < a_hi, "individual intervals overlap — the unpaired view is inconclusive"

    def test_pr_curve_rejects_a_set_without_positives(self) -> None:
        with pytest.raises(ValueError):
            precision_recall_curve((ConfusionMatrix(0, 5, 0, 95),))

    def test_average_precision_requires_monotone_recall(self) -> None:
        with pytest.raises(ValueError):
            average_precision((1.0, 1.0), (0.8, 0.2))


class TestCalibrationError:
    def test_perfectly_calibrated_scores_have_small_ece(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.beta(0.3, 8, 200_000)
        y = rng.random(p.size) < p
        ece, diagram = expected_calibration_error(p, y, n_bins=15)
        assert ece < 0.005
        assert diagram["count"].sum() == p.size

    def test_overconfident_scores_have_large_ece(self) -> None:
        rng = np.random.default_rng(1)
        true_p = rng.beta(0.3, 8, 100_000)
        y = rng.random(true_p.size) < true_p
        overconfident = np.clip(true_p * 3, 0, 1)
        assert expected_calibration_error(overconfident, y)[0] > 0.02

    def test_quantile_bins_are_populated_at_low_prevalence(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.beta(0.2, 10, 50_000)
        y = rng.random(p.size) < p
        _, quantile = expected_calibration_error(p, y, strategy="quantile")
        _, uniform = expected_calibration_error(p, y, strategy="uniform")
        assert quantile["count"].min() > 0.5 * quantile["count"].mean()
        assert uniform["count"].max() > 0.8 * p.size
