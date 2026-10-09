r"""The compound focal + soft-Dice objective.

The specification for :mod:`floodfusion.models.losses`. Read it before implementing.

The most important test in this file is
:meth:`TestSoftDiceLoss.test_all_negative_chip_gives_nonzero_gradient`. It encodes the
smoothing-placement trap of ``docs/08`` section 2.4: with the smoothing constant in the
denominator only, an all-negative chip produces **exactly zero gradient** — and most
chips are all-negative, so the majority of your training signal silently vanishes.
"""

from __future__ import annotations

import math

import pytest

#: The subject of this module imports ``torch`` at module level, so the module
#: cannot be collected without it. The guard sits here rather than in
#: ``conftest.py``: a ``Skipped`` raised while the conftest is being imported is
#: attributed to the whole directory, and would take the torch-free modules with it.
torch = pytest.importorskip(
    "torch", reason="floodfusion.models.losses imports torch at module level"
)

from floodfusion.models.losses import (  # noqa: E402
    CompoundFocalDiceLoss,
    FocalLoss,
    SoftDiceLoss,
    masked_mean,
)

#: Opt in to the auto-xfail hook in ``conftest.py``. Delete this line once
#: ``floodfusion.models.losses`` is implemented, so regressions become failures.
EXPECT_UNIMPLEMENTED = True


# ---------------------------------------------------------------------------
# masked_mean
# ---------------------------------------------------------------------------
class TestMaskedMean:
    def test_matches_plain_mean_when_fully_valid(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.ones(4, dtype=torch.bool)
        assert masked_mean(x, mask).item() == pytest.approx(2.5)

    def test_ignores_masked_positions(self) -> None:
        x = torch.tensor([1.0, 2.0, 1e9, 1e9])
        mask = torch.tensor([True, True, False, False])
        assert masked_mean(x, mask).item() == pytest.approx(1.5)

    def test_empty_mask_returns_zero_not_nan(self) -> None:
        """A chip can be entirely layover or shadow. 0/0 would poison every parameter."""
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3, dtype=torch.bool)
        out = masked_mean(x, mask)
        assert torch.isfinite(out).all(), "empty mask must not produce NaN"
        assert out.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Focal
# ---------------------------------------------------------------------------
class TestFocalLoss:
    def test_returns_a_scalar(self, synthetic_batch) -> None:
        loss = FocalLoss()(
            torch.zeros_like(synthetic_batch["label"]),
            synthetic_batch["label"],
            synthetic_batch["valid"],
        )
        assert loss.ndim == 0

    def test_gamma_zero_recovers_weighted_bce(self) -> None:
        r"""At :math:`\gamma = 0` the modulating factor is 1 and focal *is* weighted BCE."""
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) < 0.3).float()
        valid = torch.ones_like(target, dtype=torch.bool)

        focal = FocalLoss(gamma=0.0, alpha=0.5)(logits, target, valid)

        p = torch.sigmoid(logits)
        bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        expected = (0.5 * bce).mean()
        assert focal.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_easy_examples_are_downweighted_by_gamma(self) -> None:
        r"""At :math:`\gamma=2`, :math:`p_t=0.9` is scaled by 0.01 versus 0.25 at 0.5."""
        easy = torch.full((1, 1, 1, 1), math.log(0.9 / 0.1))  # p = 0.9
        hard = torch.zeros(1, 1, 1, 1)  # p = 0.5
        target = torch.ones(1, 1, 1, 1)
        valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)

        loss_fn = FocalLoss(gamma=2.0, alpha=1.0)
        ratio = loss_fn(easy, target, valid) / loss_fn(hard, target, valid)
        # (1-0.9)^2 * -log(0.9)  /  (1-0.5)^2 * -log(0.5)
        expected = (0.01 * -math.log(0.9)) / (0.25 * -math.log(0.5))
        assert ratio.item() == pytest.approx(expected, rel=1e-3)

    def test_gradient_matches_the_analytic_form(self) -> None:
        r""":math:`\partial L/\partial z = \alpha(1-p)^\gamma[\gamma p \log p + p - 1]`."""
        z = torch.tensor([[[[0.3]]]], dtype=torch.float64, requires_grad=True)
        target = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)

        gamma, alpha = 2.0, 1.0
        FocalLoss(gamma=gamma, alpha=alpha)(z, target, valid).backward()

        p = torch.sigmoid(z.detach())
        analytic = alpha * (1 - p) ** gamma * (gamma * p * torch.log(p) + p - 1)
        assert z.grad.item() == pytest.approx(analytic.item(), rel=1e-6)

    def test_is_stable_for_extreme_logits(self) -> None:
        """Must be computed from logits; ``log(sigmoid(z))`` overflows past |z| ~ 40."""
        logits = torch.tensor([[[[-80.0, 80.0]]]])
        target = torch.tensor([[[[0.0, 1.0]]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        assert torch.isfinite(FocalLoss()(logits, target, valid)).all()


# ---------------------------------------------------------------------------
# Soft Dice
# ---------------------------------------------------------------------------
class TestSoftDiceLoss:
    def test_perfect_prediction_gives_near_zero_loss(self) -> None:
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 2:5, 2:5] = 1.0
        logits = torch.where(target > 0.5, 20.0, -20.0)
        valid = torch.ones_like(target, dtype=torch.bool)
        assert SoftDiceLoss(smooth=1.0)(logits, target, valid).item() < 1e-3

    def test_all_negative_chip_gives_nonzero_gradient(self) -> None:
        r"""**The smoothing-placement trap.** ``docs/08`` section 2.4.

        With :math:`s` in the denominator only, :math:`A = 0` gives
        :math:`\partial L/\partial p_j = 0` — the loss is maximal, constant, and supplies
        no signal at all. Most chips are all-negative, so that variant discards the
        majority of your training data. With :math:`s` in **both** numerator and
        denominator, :math:`\partial L/\partial p_j = s/(B+s)^2 > 0`.
        """
        logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        target = torch.zeros(1, 1, 8, 8)
        valid = torch.ones_like(target, dtype=torch.bool)

        SoftDiceLoss(smooth=1.0)(logits, target, valid).backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0, (
            "all-negative chip produced zero gradient: the smoothing constant is in the "
            "denominator only. See docs/08 section 2.4."
        )

    def test_all_negative_gradient_pushes_predictions_down(self) -> None:
        """The sign must be right, not merely non-zero."""
        logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
        target = torch.zeros(1, 1, 8, 8)
        valid = torch.ones_like(target, dtype=torch.bool)
        SoftDiceLoss(smooth=1.0)(logits, target, valid).backward()
        assert (logits.grad > 0).all(), "gradient descent must decrease the logits"

    def test_is_computed_over_the_batch_not_per_chip(self) -> None:
        """Accumulate A and B over the micro-batch (``docs/08`` section 2.4)."""
        target = torch.zeros(2, 1, 8, 8)
        target[0, 0, :4, :4] = 1.0  # chip 0 has flood, chip 1 has none
        logits = torch.where(target > 0.5, 4.0, -4.0)
        valid = torch.ones_like(target, dtype=torch.bool)

        both = SoftDiceLoss(smooth=1.0)(logits, target, valid)
        first = SoftDiceLoss(smooth=1.0)(logits[:1], target[:1], valid[:1])
        # A batch Dice is not the mean of per-chip Dice values.
        assert both.item() != pytest.approx(first.item(), abs=1e-6)


# ---------------------------------------------------------------------------
# Compound
# ---------------------------------------------------------------------------
class TestCompoundFocalDiceLoss:
    def test_returns_total_and_separate_terms(self, loss_config, synthetic_batch) -> None:
        """A total that plateaus while focal falls and Dice rises is a diagnosis."""
        total, parts = CompoundFocalDiceLoss(loss_config)(
            torch.zeros_like(synthetic_batch["label"]),
            synthetic_batch["label"],
            synthetic_batch["valid"],
        )
        assert total.ndim == 0
        assert {"focal", "dice"} <= set(parts)

    def test_perfect_prediction_is_below_1e_4(self, loss_config) -> None:
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:9, 4:9] = 1.0
        logits = torch.where(target > 0.5, 30.0, -30.0)
        valid = torch.ones_like(target, dtype=torch.bool)
        total, _ = CompoundFocalDiceLoss(loss_config)(logits, target, valid)
        assert total.item() < 1e-4

    def test_invalid_pixels_do_not_affect_the_loss(self, loss_config) -> None:
        """**The masking test.** Garbage at excluded pixels must change nothing.

        This catches the most common masking bug: an invalid pixel contributing to the
        Dice denominator silently penalises the model for predictions in a region it is
        not being asked about.
        """
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 1:3, 1:3] = 1.0
        logits = torch.zeros(1, 1, 8, 8)
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[0, 0, 5:, 5:] = False

        loss_fn = CompoundFocalDiceLoss(loss_config)
        before, _ = loss_fn(logits, target, valid)

        polluted = logits.clone()
        polluted[0, 0, 5:, 5:] = 50.0  # arbitrary garbage in the excluded region
        after, _ = loss_fn(polluted, target, valid)

        assert before.item() == pytest.approx(after.item(), rel=1e-6)

    def test_gradients_are_finite_under_bfloat16_autocast(self, loss_config) -> None:
        """bf16, not fp16: the Dice reduction sums ~5e5 terms (``docs/08`` section 5.1)."""
        logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
        target = (torch.rand(2, 1, 32, 32) < 0.02).float()
        valid = torch.ones_like(target, dtype=torch.bool)

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            total, _ = CompoundFocalDiceLoss(loss_config)(logits, target, valid)
        total.float().backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_weights_compose_the_two_terms(self) -> None:
        from floodfusion.config import LossConfig

        target = (torch.rand(1, 1, 16, 16) < 0.1).float()
        logits = torch.randn(1, 1, 16, 16)
        valid = torch.ones_like(target, dtype=torch.bool)

        total, parts = CompoundFocalDiceLoss(
            LossConfig(focal_weight=1.0, dice_weight=2.0)
        )(logits, target, valid)
        expected = parts["focal"] + 2.0 * parts["dice"]
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.slow
def test_focal_gradcheck_in_float64() -> None:
    """``P1-M4.2`` requires verification against ``gradcheck``."""
    logits = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    target = (torch.rand(1, 1, 4, 4) < 0.4).double()
    valid = torch.ones_like(target, dtype=torch.bool)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
    assert torch.autograd.gradcheck(
        lambda z: loss_fn(z, target, valid), (logits,), eps=1e-6, atol=1e-4
    )
