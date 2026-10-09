r"""Segmentation losses for a ~2 % positive rate.

Roughly 2 % of valid pixels are flooded. A loss that treats all pixels equally is 98 % a
statement about dry land, and its minimiser is very nearly the constant "dry" predictor —
which scores 98 % accuracy and 0.00 IoU (``docs/06`` section 2.2).

Throughout this module: :math:`z` is a **logit**, :math:`p = \sigma(z)`, and
:math:`y \in \{0,1\}` is the label. Losses receive logits, never probabilities
(``docs/07`` section 9.2), because the numerically stable formulation depends on the
cancellation derived in :class:`FocalLoss`.

All reductions are over **valid pixels only**. An invalid pixel that contributes to a
Dice denominator silently penalises the model for predictions in a region it is not
being asked about.

References
----------
``docs/08_training_recipe_and_vram.md`` section 2 — the full derivations.
``docs/10`` exercise A4 — derive these by hand.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from floodfusion.config import LossConfig

__all__ = ["FocalLoss", "SoftDiceLoss", "CompoundFocalDiceLoss", "masked_mean"]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is True.

    Parameters
    ----------
    values
        Any shape.
    mask
        Broadcastable to ``values``; ``bool``.

    Returns
    -------
    torch.Tensor
        Scalar. Zero when the mask is empty — **not** ``NaN``.

    Notes
    -----
    The empty-mask case is not hypothetical: a chip can be entirely layover, shadow or
    outside the AOI. ``(values * mask).sum() / mask.sum()`` then divides by zero and
    produces a ``NaN`` that propagates through the whole backward pass and poisons every
    parameter. Guard it.

    Accumulate the sum in ``float32`` even under ``autocast``: with 7 mantissa bits,
    summing :math:`5 \\times 10^{5}` terms in ``bfloat16`` accumulates real error
    (``docs/08`` section 5.2).

    TODO(student)
    -------------
    1. Broadcast the mask and cast it to the values' dtype.
    2. Sum in ``float32``.
    3. Return ``0`` (as a tensor on the right device, preserving the graph) when the
       mask is empty.
    """
    raise NotImplementedError("Implement masked_mean. See docs/08 section 5.2.")


class FocalLoss(nn.Module):
    r"""Focal loss: weight by difficulty rather than by class.

    Parameters
    ----------
    gamma
        Focusing parameter :math:`\gamma \ge 0`. ``0`` recovers weighted BCE.
    alpha
        Weight applied to the positive class.

    Notes
    -----
    **Definition.** With :math:`p_t = p` when :math:`y=1` and :math:`p_t = 1-p` when
    :math:`y=0` — the probability assigned to the *true* class:

    .. math:: L_{\mathrm{focal}} = -\alpha_t\,(1-p_t)^{\gamma}\log p_t

    **The modulating factor.** :math:`(1-p_t)^{\gamma}` is near 0 for well-classified
    examples and near 1 for badly classified ones. At :math:`\gamma = 2`, an example
    with :math:`p_t = 0.9` has its loss scaled by :math:`0.01`, one with
    :math:`p_t = 0.5` by :math:`0.25`. The vast population of easy dry pixels is
    suppressed by **100x**, and the gradient budget flows to flood boundaries,
    wind-roughened water and shadow.

    **Gradient**, for :math:`y = 1` (derive the :math:`y=0` case yourself —
    ``docs/10`` exercise A4b):

    .. math::

        \frac{\partial L_{\mathrm{focal}}}{\partial z}
        = \alpha\,(1-p)^{\gamma}\bigl[\gamma\,p\log p + p - 1\bigr]

    Two checks worth performing in the test:

    - :math:`\gamma = 0` gives :math:`\alpha(p-1) = \alpha(p - y)`, i.e. weighted BCE.
      For plain BCE the whole gradient collapses to the celebrated
      :math:`\partial L/\partial z = p - y`, in which the :math:`p(1-p)` factors from
      :math:`\partial L/\partial p` and :math:`\mathrm{d}p/\mathrm{d}z` cancel exactly.
      That cancellation is *why* you pass logits.
    - :math:`p \to 1` sends both :math:`(1-p)^{\gamma}` and the bracket to zero, so the
      gradient vanishes faster than BCE's — the intended suppression.

    **The cost.** Focal loss produces systematically **over-confident** outputs, because
    it deliberately stops penalising confident predictions. Its scores are not
    calibrated posteriors, so applying the cost-optimal threshold
    :math:`\tau^\star = 0.25` (``docs/00`` section 6.2) to them directly is numerology.
    Fit a temperature on validation first — ``docs/09`` section 4.

    Shape contract
    --------------
    ``forward(logits, target, valid)`` with all of ``(B, 1, H, W)``; ``valid`` is
    ``bool``. Returns a scalar.

    TODO(student)
    -------------
    1. Compute the per-pixel loss **from logits**, using
       ``F.binary_cross_entropy_with_logits(..., reduction="none")`` for the
       :math:`\log p_t` term rather than ``log(sigmoid(z))``, which overflows for
       :math:`|z| \gtrsim 40`.
    2. Form :math:`p_t` and :math:`\alpha_t` with ``torch.where``, not with Python
       branching — the input is a batch, not a scalar.
    3. Multiply by the modulating factor and reduce with :func:`masked_mean`.
    4. Verify the analytic gradient against ``torch.autograd.gradcheck`` in
       ``float64``. This is a required check in ``P1-M4.2``.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.gamma = gamma
        self.alpha = alpha
        raise NotImplementedError("Implement FocalLoss. See docs/08 section 2.3.")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


class SoftDiceLoss(nn.Module):
    r"""Soft-Dice loss: a differentiable set-overlap objective.

    Parameters
    ----------
    smooth
        Smoothing constant :math:`s > 0`, applied to **both numerator and denominator**.
        See the warning below; this is a correctness condition, not a tuning knob.

    Notes
    -----
    **Definition.** With :math:`A = \sum_i p_i y_i` and
    :math:`B = \sum_i p_i + \sum_i y_i` over valid pixels:

    .. math::

        \mathrm{Dice} = \frac{2A + s}{B + s}, \qquad L_{\mathrm{Dice}} = 1 - \mathrm{Dice}

    **Gradient.**

    .. math::

        \frac{\partial L_{\mathrm{Dice}}}{\partial p_j}
        = \frac{(2A + s) - 2 y_j (B + s)}{(B + s)^2}

    and :math:`\partial L/\partial z_j` follows by multiplying by :math:`p_j(1-p_j)`.

    **Property 1 — not decomposable.** The gradient at pixel :math:`j` depends on
    *every* pixel through :math:`A` and :math:`B`. That is precisely why it handles
    imbalance: it optimises a set-level overlap rather than a sum of independent
    per-pixel terms, so the enormous negative population does not enter as
    :math:`N^-` separate terms.

    It is also why its gradients are **noisy at small batch**, and why gradient
    accumulation over micro-batches is **not** equivalent to one large batch for this
    term (``docs/08`` section 5.3, trap 2). For focal loss accumulation is exact; for
    Dice it is an approximation that worsens as the micro-batch shrinks.

    .. warning::

       **The smoothing placement determines whether empty chips train the model.**

       On an all-negative chip (:math:`y_i = 0\ \forall i`, hence :math:`A = 0`):

       - With :math:`s` in the **denominator only**, :math:`\mathrm{Dice} = 2A/(B+s) = 0`
         and :math:`\partial L / \partial p_j = 0` — **exactly zero gradient**. The loss
         is maximal, constant, and completely uninformative.
       - With :math:`s` in **both**, :math:`\mathrm{Dice} = s/(B+s)`, so
         :math:`L = B/(B+s)` and :math:`\partial L/\partial p_j = s/(B+s)^2 > 0` — a
         genuine gradient pushing every prediction down, which is correct.

       Most chips contain no flood at all (``docs/05`` section 8.2), so the first
       variant discards the majority of your training signal. ``tests/test_losses.py``
       asserts the second.

    **Compute over the batch, not per chip.** Accumulate :math:`A` and :math:`B` over
    all valid pixels in the micro-batch. A per-chip Dice averaged over chips is
    dominated by near-empty chips whose Dice is degenerate.

    Shape contract
    --------------
    ``forward(logits, target, valid)`` with all of ``(B, 1, H, W)``. Returns a scalar.

    TODO(student)
    -------------
    1. Apply ``sigmoid`` to the logits **here** — Dice needs probabilities, unlike
       focal, which is formulated on logits for stability.
    2. Zero out invalid pixels **before** summing into :math:`A` and :math:`B`.
    3. Accumulate in ``float32`` (``docs/08`` section 5.2).
    4. Place ``smooth`` in both numerator and denominator. Then write the test that
       fails if you place it in the denominator only.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0.0:
            raise ValueError(
                "smooth must be strictly positive; with smooth == 0 an all-negative "
                "chip yields zero gradient. See docs/08 section 2.4."
            )
        self.smooth = smooth
        raise NotImplementedError("Implement SoftDiceLoss. See docs/08 section 2.4.")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


class CompoundFocalDiceLoss(nn.Module):
    r"""The project's objective: :math:`L = \alpha L_{\mathrm{focal}} + \beta L_{\mathrm{Dice}}`.

    Parameters
    ----------
    cfg
        Loss configuration supplying the weights, :math:`\gamma`,
        :math:`\alpha_{\mathrm{focal}}` and the Dice smoothing.

    Notes
    -----
    Justified by the complementary defects of the two terms:

    ================================== ========= ==========================
    property                           focal     soft-Dice
    ================================== ========= ==========================
    decomposable per pixel             yes       no
    gradient on empty chips            yes       only with correct smoothing
    directly optimises reported metric no        yes
    stable early in training           yes       no
    sensitive to micro-batch size      no        yes
    preserves calibration              poorly    poorly
    ================================== ========= ==========================

    Focal supplies dense, stable, per-pixel supervision everywhere; Dice aligns the
    objective with the evaluation metric and resists imbalance structurally. Neither
    alone is adequate.

    Ablation row 12 replaces this with plain BCE and measures what it was worth.

    **Return the terms separately as well as the total.** A total loss that plateaus
    while focal falls and Dice rises is a diagnosis; a single scalar hides it. Debugging
    challenge C3 in ``docs/10`` section 3 turns on exactly this observation.

    Shape contract
    --------------
    ``forward(logits, target, valid)`` with all of ``(B, 1, H, W)``; ``valid`` is
    ``bool``. Returns ``(total, {"focal": ..., "dice": ...})``.

    Validation the tests require
    ----------------------------
    1. A perfect prediction (logits :math:`\to \pm\infty` in the right places) gives a
       loss below :math:`10^{-4}`.
    2. An all-negative chip produces a **non-zero** gradient pushing predictions down.
    3. Loss is invariant to the values at invalid pixels — set them to arbitrary
       garbage and the loss must not change. This is the masking test, and it catches
       the most common masking bug.
    4. Gradients are finite under ``autocast(bfloat16)``.

    TODO(student)
    -------------
    1. Assert the three inputs share shape ``(B, 1, H, W)`` with
       ``utils.shapes.assert_shape``, and that ``valid`` is ``bool``.
    2. Compose the two terms with their configured weights.
    3. Consider the warm-up of ``docs/08`` section 3: :math:`\beta = 0` for the first
       few epochs, ramped in. Dice's gradient near random initialisation is nearly
       uninformative. If you use it, expose it as a scheduled parameter and record it.
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "Implement the compound loss. See docs/08 section 2.5."
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # noqa: D102
        raise NotImplementedError
