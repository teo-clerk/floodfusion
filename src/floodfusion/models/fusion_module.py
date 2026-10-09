r"""Cross-attention fusion with a learned availability gate.

This is the architectural centrepiece of Project 01 and the subject of milestone
``P1-M3``. The design is **derived** in ``docs/07_fusion_architectures.md`` section 5.4;
what follows restates the result and the shape contracts, not the argument. If you
cannot reproduce the derivation, return to that section before implementing — ``P1-M3``'s
comprehension question asks for it, and so does the design-review exercise in
``docs/10`` section 5, round 2.

The fusion equation
-------------------
.. math::

    F_{\mathrm{fused}} \;=\; F_s \;+\; g \odot
        \mathrm{CrossAttn}\bigl(Q = F_o,\; K = F_s,\; V = F_s\bigr)

with :math:`F_s` the SAR features, :math:`F_o` the optical features (or the learned
missing-modality embedding), and :math:`g \in [0,1]` a per-pixel gate.

Why this assignment
-------------------
Three facts, in order.

1. **Attention output is a convex combination of values.** Softmax weights are
   non-negative and sum to one, so the output lies in the convex hull of :math:`V`.
   With :math:`V` drawn from SAR, *every quantity the attention can ever retrieve is a
   real SAR feature* — even under a degenerate query.

2. **Both modalities share one grid.** They are co-registered by ``P1-M1``, so
   :math:`N_q = N_k` and position :math:`i` denotes the same patch of ground in both.
   The output is spatially aligned whichever stream is the query, so the choice is
   about semantics and availability, not geometry.

3. **Requirement R2** (``docs/00`` section 7) demands a usable map from SAR alone. With
   the residual and the gate on the SAR stream, :math:`g \to 0` gives
   :math:`F_{\mathrm{fused}} = F_s` **exactly**. R2 is satisfied structurally rather
   than by hoping the network learns it.

The honest caveat: the reverse assignment (:math:`Q` from SAR) is also defensible and is
used in parts of the literature. Implement this one, then run the reverse as an
ablation — ``docs/07`` section 5.5.

References
----------
``docs/07_fusion_architectures.md`` sections 4, 5, 6, 7, 8.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from floodfusion.config import ModelConfig

__all__ = ["CrossAttentionFusion", "MissingModalityEmbedding", "AvailabilityGate"]


class MissingModalityEmbedding(nn.Module):
    r"""A learned representation of "no optical observation here".

    Parameters
    ----------
    channels
        Feature width at the fusion stage, :math:`C_3`.

    Notes
    -----
    **Do not fill missing optical with zeros.** Zero is a legitimate value for a
    normalised reflectance channel, so the network could not distinguish "dark" from
    "absent" — and darkness is precisely the water signal (``docs/07`` section 7.2).

    Instead hold a learned parameter :math:`\mathbf{m} \in \mathbb{R}^{C}` and broadcast
    it over space:

    .. math:: F_o \leftarrow \mathbf{m} \otimes \mathbf{1}_{H \times W}

    Because :math:`\mathbf{m}` is spatially constant, the resulting queries are identical
    at every position and attention degenerates to a single global lookup over the SAR
    values. Combined with the gate driving :math:`g \to 0`, the residual then carries
    essentially pure :math:`F_s` — which is exactly R2.

    Shape contract
    --------------
    Parameter shape ``(1, channels, 1, 1)``; broadcast to ``(B, channels, H, W)``.

    TODO(student)
    -------------
    1. Register ``self.embedding`` as an ``nn.Parameter`` of shape ``(1, C, 1, 1)``.
    2. Initialise it small (e.g. ``normal_(0, 0.02)``), not zero: an exactly zero
       initialisation makes it indistinguishable from the fill value you are avoiding.
    3. In ``forward``, expand — do not ``repeat``. ``expand`` is a view and allocates
       nothing; ``repeat`` materialises a full ``(B, C, H, W)`` tensor for no reason.
    4. Confirm it receives gradient whenever it is used. ``tests/test_fusion_module.py``
       contract 5 asserts this.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        raise NotImplementedError(
            "Implement the missing-modality embedding. See docs/07 section 7.2."
        )

    def forward(self, batch: int, height: int, width: int, *, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


class AvailabilityGate(nn.Module):
    r"""Per-pixel gate :math:`g \in [0,1]` controlling the optical contribution.

    Parameters
    ----------
    sar_channels, optical_channels
        Feature widths at the fusion stage.
    bias_init
        Initial bias of the output layer. Negative by default so that
        :math:`g \approx 0.12` at initialisation.

    Notes
    -----
    Optical is not binary. A chip may be 30 % cloudy, thin-cirrus contaminated, or in
    cloud shadow — where ``docs/02`` section 5.2 shows the index actively lies in the
    direction of *false water*. A hard present/absent switch cannot express partial
    reliability, and a partially reliable optical input trusted fully is worse than none.

    .. math::

        g = \sigma\Bigl(\mathrm{MLP}\bigl([\,q\,;\,F_o\,;\,F_s\,]\bigr)\Bigr)
        \;\in\; [0,1]^{B \times 1 \times H \times W}

    with :math:`q` the ``optical_quality`` channel, downsampled to the fusion stage.

    **Per-pixel, not per-chip.** Cloud edges cut across chips; a scalar gate forces one
    decision for a chip that is half clear.

    **Applied multiplicatively to the attention output, never as an additive mask inside
    the softmax.** An additive attention mask forfeits the FlashAttention path
    (``docs/07`` section 4.3), and output gating is easier to reason about.

    **Negative bias initialisation is load-bearing.** The model starts near SAR-only and
    must *earn* its use of optical. Initialising near :math:`g = 1` makes early training
    depend on a modality that is usually absent.

    **Do not supervise the gate.** Forcing :math:`g = 1 - p_{\mathrm{cloud}}` imposes a
    relationship the data may not support; the model may reasonably learn to distrust
    optical even in clear conditions over turbid water (``docs/02`` section 5.1).

    Shape contract
    --------------
    ``(B, C_s, H, W)``, ``(B, C_o, H, W)``, ``(B, 1, H, W)`` in; ``(B, 1, H, W)`` out,
    with every value in ``[0, 1]``.

    TODO(student)
    -------------
    1. Concatenate the three inputs along the channel axis.
    2. Apply a small MLP — two 1x1 convolutions with an activation between — to one
       output channel.
    3. Initialise the final bias to ``bias_init`` and its weight small, so the gate is
       genuinely near ``sigmoid(bias_init)`` at step 0.
    4. Return ``sigmoid`` of the result.
    5. Expose the batch mean of ``g`` for logging: it is the single most informative
       diagnostic in this architecture (``docs/07`` section 8.3, and debugging challenge
       C4 in ``docs/10`` section 3).
    """

    def __init__(
        self,
        sar_channels: int,
        optical_channels: int,
        *,
        bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Implement the gate. See docs/07 section 8.")

    def forward(
        self,
        sar_features: torch.Tensor,
        optical_features: torch.Tensor,
        optical_quality: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


class CrossAttentionFusion(nn.Module):
    r"""Multi-head cross-attention fusion of optical queries against SAR keys/values.

    Parameters
    ----------
    sar_channels, optical_channels
        Feature widths at the fusion stage. Typically equal (:math:`C_3 = 256`).
    cfg
        Model configuration supplying ``attention_heads``, ``attention_head_dim`` and
        ``gate_bias_init``.

    Notes
    -----
    **The attention operation.** With :math:`Q \in \mathbb{R}^{N_q \times d_k}`,
    :math:`K \in \mathbb{R}^{N_k \times d_k}`, :math:`V \in \mathbb{R}^{N_k \times d_v}`:

    .. math::

        \mathrm{Attention}(Q,K,V) =
            \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
            \;\in\; \mathbb{R}^{N_q \times d_v}

    Shapes, step by step — these are what ``tests/test_fusion_module.py`` asserts:

    ==== ============================== ====================
    step expression                     shape
    ==== ============================== ====================
    1    :math:`QK^{\top}`              ``(N_q, N_k)``
    2    scale by :math:`1/\sqrt{d_k}`  ``(N_q, N_k)``
    3    softmax over the **last** axis ``(N_q, N_k)``, rows sum to 1
    4    :math:`\cdot V`                ``(N_q, d_v)``
    ==== ============================== ====================

    With heads and batch, tensors are ``(B, h, N, d)`` with
    :math:`d = d_{\mathrm{model}} / h`.

    **The scaling is not cosmetic.** Modelling :math:`q_i, k_i` as i.i.d. zero-mean
    unit-variance gives :math:`\operatorname{Var}[q \cdot k] = d_k`, so unscaled logits
    have standard deviation :math:`\sqrt{d_k} = 8` at :math:`d_k = 64` and spread over a
    range of roughly 48. Softmax of such logits is numerically one-hot, whereupon the
    Jacobian :math:`\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)` vanishes and
    **no gradient reaches** :math:`K` **or** :math:`V`. In ``bfloat16`` it is worse: at
    logit magnitude 24 the representable spacing is :math:`2^{4}\cdot2^{-7} = 0.125`, so
    logit differences below that are quantised away and the attention distribution is
    decided by rounding. Both failures are silent and neither produces a ``NaN``.
    See ``docs/07`` section 4.2.

    **Use SDPA.** ``torch.nn.functional.scaled_dot_product_attention`` dispatches to
    FlashAttention-2 on sm_89 and never materialises the :math:`(N_q, N_k)` matrix,
    making memory :math:`O(N)` rather than :math:`O(N^2)`. At the default fusion stage
    (:math:`32\times32`, :math:`N = 1024`) the naive matrix is 134 MB at ``B=8, h=8``;
    at stage 2 it is 2.15 GB and at stage 1 it is 34 GB. Reproduce that table for your
    own batch size — ``docs/07`` section 6.

    Conditions for the fast path: contiguous ``(B, h, N, d)`` tensors, head dimension
    :math:`\le 128`, and **no arbitrary additive mask**.

    Shape contract
    --------------
    ``forward(sar_features, optical_features, optical_quality)``

    ============================ ====================
    tensor                       shape
    ============================ ====================
    ``sar_features``             ``(B, C_s, H, W)``
    ``optical_features``         ``(B, C_o, H, W)``
    ``optical_quality``          ``(B, 1, H, W)``
    **returns** ``fused``        ``(B, C_s, H, W)``
    **returns** ``gate``         ``(B, 1, H, W)``
    ============================ ====================

    The output width equals the **SAR** width, because the residual adds to
    :math:`F_s`. Any projection must map the attention output back to :math:`C_s`.

    The contracts the tests enforce
    -------------------------------
    1. Attention weights sum to 1 along the key axis (verify with a naive reference
       implementation in ``float64``, not with SDPA, which never exposes them).
    2. With ``g`` forced to 0, output is **bit-identical** to ``sar_features``.
    3. Gradients reach both encoders when optical is present.
    4. Gradients reach the SAR encoder and the missing embedding — but **not** the
       optical encoder — when optical is absent.
    5. No ``NaN``/``Inf`` in forward or backward under ``autocast(bfloat16)``.

    Contract 2 is the mechanised statement of R2 and is the most valuable test in the
    file. Contract 4 catches the most common bug in this architecture: a fill value that
    still routes gradient into the optical encoder, quietly training it on fabricated
    inputs.

    TODO(student)
    -------------
    1. Build linear projections :math:`W_Q` (from ``optical_channels``), :math:`W_K`,
       :math:`W_V` (from ``sar_channels``), each to ``heads * head_dim``, and an output
       projection back to ``sar_channels``. 1x1 convolutions are the natural spelling
       for feature maps.
    2. Flatten spatial dims to tokens: ``(B, C, H, W) -> (B, h, N, d)`` with
       :math:`N = HW`. Use ``einops.rearrange`` or an explicit ``reshape`` +
       ``permute``, then ``.contiguous()`` — SDPA's fast path needs contiguity.
    3. Call ``F.scaled_dot_product_attention(q, k, v)``. Do **not** pass an additive
       mask. Do **not** re-scale: SDPA already applies :math:`1/\sqrt{d}` by default —
       applying it yourself as well scales by :math:`1/d` and is a real, silent bug.
    4. Reshape back to ``(B, C, H, W)`` and apply the output projection.
    5. Compute the gate and form ``fused = sar_features + gate * attn_out``.
    6. Assert every shape with ``utils.shapes.assert_shape``, binding ``B``, ``H``,
       ``W`` once across all three inputs.
    7. Return ``(fused, gate)`` — the gate is returned so the training loop can log its
       mean without a forward hook.
    """

    def __init__(
        self,
        sar_channels: int,
        optical_channels: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.heads = cfg.attention_heads
        self.head_dim = cfg.attention_head_dim
        self.inner_dim = self.heads * self.head_dim
        raise NotImplementedError(
            "Implement cross-attention fusion. The derivation is docs/07 section 5.4; "
            "the memory arithmetic is docs/07 section 6."
        )

    def forward(
        self,
        sar_features: torch.Tensor,
        optical_features: torch.Tensor,
        optical_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        raise NotImplementedError


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Naive attention that **returns the weights**, for testing only.

    Parameters
    ----------
    query, key, value
        ``(B, h, N_q, d)``, ``(B, h, N_k, d)``, ``(B, h, N_k, d_v)``.

    Returns
    -------
    output : torch.Tensor
        ``(B, h, N_q, d_v)``.
    weights : torch.Tensor
        ``(B, h, N_q, N_k)``, rows summing to 1.

    Notes
    -----
    SDPA never materialises the attention matrix, which is exactly why it is memory
    efficient — and exactly why you cannot assert "rows sum to one" against it. This
    function exists so ``tests/test_fusion_module.py`` can check contract 1 on small
    inputs in ``float64``, and so you can verify your SDPA call agrees with the
    definition.

    **Never call this in the model.** At the fusion stage it allocates the
    :math:`(N_q, N_k)` matrix the whole design avoids.

    TODO(student)
    -------------
    1. Compute the scaled scores, softmax over the last axis, and the weighted sum.
    2. Assert every intermediate shape.
    3. In the test, compare against ``F.scaled_dot_product_attention`` and confirm
       agreement to ``float64`` tolerance. If they disagree by a constant factor, you
       have double-scaled (TODO 3 in :class:`CrossAttentionFusion`).
    """
    raise NotImplementedError(
        "Implement the reference attention. See docs/07 section 4.1 and docs/10 "
        "exercise A2 for the backward pass you should be able to derive by hand."
    )
