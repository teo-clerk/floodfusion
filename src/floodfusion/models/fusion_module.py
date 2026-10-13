r"""Cross-attention fusion with a learned availability gate.

This is the architectural centrepiece of Project 01 and the subject of milestone
``P1-M3``. The design is **derived** in ``docs/07_fusion_architectures.md`` section 5.4;
what follows restates the result and the shape contracts, not the argument.

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from floodfusion.config import ModelConfig
from floodfusion.utils.shapes import assert_shape, named_dims

__all__ = ["AvailabilityGate", "CrossAttentionFusion", "MissingModalityEmbedding", "reference_attention"]


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

    Implementation notes
    --------------------
    1. ``self.embedding`` is an ``nn.Parameter`` of shape ``(1, C, 1, 1)``.
    2. It is initialised small (``normal_(0, 0.02)``), not zero: an exactly zero
       initialisation would make it indistinguishable from the fill value it replaces.
    3. ``forward`` expands — never ``repeat``\ s. ``expand`` is a view and allocates
       nothing; ``repeat`` would materialise a full ``(B, C, H, W)`` tensor for no reason.
    4. It receives gradient whenever it is used (``tests/test_fusion_module.py``,
       contract 5).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.embedding = nn.Parameter(torch.empty(1, channels, 1, 1))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, batch: int, height: int, width: int, *, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:  # noqa: D102
        # ``to`` is differentiable; ``expand`` is a view — nothing (B, C, H, W)-sized is
        # allocated, and the backward of expand sums the gradient into the parameter.
        return self.embedding.to(device=device, dtype=dtype).expand(batch, self.channels, height, width)


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

    Implementation notes
    --------------------
    1. The three inputs are concatenated along the channel axis.
    2. A small MLP — two 1x1 convolutions with an activation between — maps them to one
       output channel.
    3. The final bias starts at ``bias_init`` and its weight small, so the gate is
       genuinely near ``sigmoid(bias_init)`` at step 0.
    4. The result is ``sigmoid`` of that channel.
    5. The batch mean of ``g`` is logged: it is the single most informative diagnostic in
       this architecture (``docs/07`` section 8.3).
    """

    def __init__(
        self,
        sar_channels: int,
        optical_channels: int,
        *,
        bias_init: float = -2.0,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.hidden = nn.Conv2d(sar_channels + optical_channels + 1, hidden_channels, 1)
        self.act = nn.SiLU()
        self.out = nn.Conv2d(hidden_channels, 1, 1)
        # Small but NON-zero final weight: zero would make g spatially constant at step 0
        # (a scalar gate in disguise); small keeps g ~= sigmoid(bias_init) everywhere.
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out.bias, bias_init)

    def forward(
        self,
        sar_features: torch.Tensor,
        optical_features: torch.Tensor,
        optical_quality: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        dims = named_dims()
        assert_shape(sar_features, ("B", self.sar_channels, "H", "W"), dims, name="gate.sar")
        assert_shape(optical_features, ("B", self.optical_channels, "H", "W"), dims, name="gate.optical")
        assert_shape(optical_quality, ("B", 1, "H", "W"), dims, name="gate.quality")
        dtype = sar_features.dtype
        x = torch.cat(
            [optical_quality.to(dtype), optical_features.to(dtype), sar_features], dim=1
        )
        # Nothing follows the sigmoid: no floor, no clamp. g = 0 must be reachable EXACTLY
        # (requirement R2, tests/test_fusion_module.py contract 3).
        return torch.sigmoid(self.out(self.act(self.hidden(x))))


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

    Implementation notes
    --------------------
    1. Linear projections :math:`W_Q` (from ``optical_channels``), :math:`W_K`,
       :math:`W_V` (from ``sar_channels``), each to ``heads * head_dim``, and an output
       projection back to ``sar_channels``, all as 1x1 convolutions.
    2. Spatial dims are flattened to tokens, ``(B, C, H, W) -> (B, h, N, d)`` with
       :math:`N = HW`, and made ``.contiguous()`` — SDPA's fast path needs contiguity.
    3. ``F.scaled_dot_product_attention(q, k, v)`` gets no additive mask and no extra
       scaling: SDPA already applies :math:`1/\sqrt{d}`, and applying it again would scale by
       :math:`1/d` — a real, silent bug.
    4. The output is reshaped back to ``(B, C, H, W)`` and projected.
    5. ``fused = sar_features + gate * attn_out``.
    6. Every shape is asserted with ``utils.shapes.assert_shape``, binding ``B``, ``H``,
       ``W`` once across all three inputs.
    7. ``(fused, gate)`` is returned, so the training loop can log the gate mean without a
       forward hook.
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
        # Pre-norm on the token streams (per token, over channels) stabilises the logit
        # scale early in training (docs/07 section 4.2); it acts on the ATTENTION branch
        # only — the residual F_s is never normalised, projected or biased (R2).
        # ``fusion_query = "sar"`` is the reverse assignment (v1.1 row 9r): Q from SAR, K = V from
        # optical. The residual and the gate stay on the SAR stream, so g -> 0 is still exact (R2);
        # what is lost is "every retrievable value is a SAR feature".
        self.query_from_sar = cfg.fusion_query == "sar"
        q_in, kv_in = (sar_channels, optical_channels) if self.query_from_sar else (optical_channels, sar_channels)
        self.norm_q = nn.LayerNorm(q_in)
        self.norm_kv = nn.LayerNorm(kv_in)
        self.q_proj = nn.Linear(q_in, self.inner_dim)
        self.k_proj = nn.Linear(kv_in, self.inner_dim)
        self.v_proj = nn.Linear(kv_in, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, sar_channels)
        self.gate = AvailabilityGate(sar_channels, optical_channels, bias_init=cfg.gate_bias_init)

    def _heads(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, h*d) -> contiguous (B, h, N, d), the layout SDPA's fast path needs."""
        b, n, _ = tokens.shape
        return tokens.view(b, n, self.heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        sar_features: torch.Tensor,
        optical_features: torch.Tensor,
        optical_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        dims = named_dims()
        assert_shape(sar_features, ("B", self.sar_channels, "H", "W"), dims, name="sar_features")
        assert_shape(optical_features, ("B", self.optical_channels, "H", "W"), dims, name="optical_features")
        assert_shape(optical_quality, ("B", 1, "H", "W"), dims, name="optical_quality")
        b, h, w = dims["B"], dims["H"], dims["W"]

        sar_tokens = sar_features.flatten(2).transpose(1, 2)         # (B, N, C_s)
        opt_tokens = optical_features.flatten(2).transpose(1, 2)     # (B, N, C_o)
        q_tokens, kv_tokens = (sar_tokens, opt_tokens) if self.query_from_sar else (opt_tokens, sar_tokens)
        kv_in = self.norm_kv(kv_tokens)
        q = self._heads(self.q_proj(self.norm_q(q_tokens)))        # (B, h, N, d) from OPTICAL (reference)
        k = self._heads(self.k_proj(kv_in))                         # (B, h, N, d) from SAR (reference)
        v = self._heads(self.v_proj(kv_in))                         # (B, h, N, d) from SAR (reference)
        # SDPA applies 1/sqrt(d) itself. No additive mask (it would forfeit the Flash path)
        # and no dropout (it would make the SAR-only forward stochastic in train mode).
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, h * w, self.inner_dim)
        attn = self.out_proj(attn)                                  # (B, N, C_s)
        attn = attn.transpose(1, 2).reshape(b, self.sar_channels, h, w)

        gate = self.gate(sar_features, optical_features, optical_quality)
        assert_shape(gate, ("B", 1, "H", "W"), dims, name="gate")
        # The whole design in one line: a gated residual on the SAR stream.
        fused = sar_features + gate * attn
        return fused, gate


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

    Implementation notes
    --------------------
    1. Scaled scores, softmax over the key axis, then the weighted sum.
    2. Every intermediate shape is asserted.
    3. The tests compare it with ``F.scaled_dot_product_attention`` to ``float64``
       tolerance; a constant-factor disagreement would mean double scaling (note 3 of
       :class:`CrossAttentionFusion`).
    """
    dims = named_dims()
    assert_shape(query, ("B", "h", "Nq", "d"), dims, name="query")
    assert_shape(key, ("B", "h", "Nk", "d"), dims, name="key")
    assert_shape(value, ("B", "h", "Nk", "dv"), dims, name="value")
    scores = query @ key.transpose(-2, -1) / math.sqrt(dims["d"])
    assert_shape(scores, ("B", "h", "Nq", "Nk"), dims, name="scores")
    weights = torch.softmax(scores, dim=-1)  # over the KEY axis: rows sum to one
    output = weights @ value
    assert_shape(output, ("B", "h", "Nq", "dv"), dims, name="output")
    return output, weights
