r"""Decoder, head and the assembled dual-branch flood segmenter.

Architecture
------------
::

    sar (B,5,256,256) ─┐
    terrain (B,3,·,·) ─┴─► concat (B,8,256,256) ─► SAR encoder ─► s1,s2,s3,s4
                                                                   │  │  │
    optical (B,4,256,256) ─► Optical encoder ─────────► o3         │  │  │
                                    │                              │  │  │
              modality dropout / missing embedding                 │  │  │
                                    ▼                              ▼  │  │
                       CrossAttn(Q=o3, K=s3, V=s3) ──► × g ──────► + ──┘  │
                                                                   │      │
                                                             F_fused      │
                                                                   ▼      ▼
                                                        Decoder + skips (s1,s2)
                                                                   ▼
                                                 1x1 conv ─► logits (B,1,256,256)

Parameter budget (``docs/07`` section 9.4): ~19 M total — 8.5 M per encoder, 0.8 M for
fusion and gate, 1.5 M decoder. At 16 bytes per parameter under AdamW that is 304 MB of
static memory, which is small; **activations are the entire memory problem**
(``docs/08`` section 4.2).

References
----------
``docs/07_fusion_architectures.md`` sections 7, 9.
``docs/08_training_recipe_and_vram.md`` sections 4, 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    ModelConfig,
)

__all__ = ["UNetDecoder", "FloodSegmenter"]


class UNetDecoder(nn.Module):
    """U-Net decoder: successive upsampling with encoder skip connections.

    Parameters
    ----------
    encoder_channels
        Channel widths of the encoder stages, finest first.
    decoder_channels
        Channel widths of the decoder stages, coarsest first.
    norm_groups
        ``GroupNorm`` group count.

    Notes
    -----
    **Take skips from the SAR encoder, not the optical one** (``docs/07`` section 9.1).
    They must always exist — requirement R2 — and the SAR branch is the always-present
    one. Optical influence reaches the decoder through the fused stage-3 representation,
    which is the design's intent. Wiring an optical skip into the decoder silently
    breaks SAR-only inference and is caught by contract 2 in
    ``tests/test_fusion_module.py``.

    **Memory.** ``docs/08`` section 4.4 shows the final full-resolution decoder stage is
    one of the two largest activation consumers (:math:`256^2` at 32 channels), the other
    being encoder stage 1. If you enable activation checkpointing, those are the two
    places to apply it; checkpointing the coarse stages saves almost nothing.

    Shape contract
    --------------
    ``forward(fused, skips)`` where ``fused`` is ``(B, C_3, H/8, W/8)`` and ``skips`` is
    the SAR encoder's ``(s1, s2)`` at ``(B, C_1, H/2, W/2)`` and ``(B, C_2, H/4, W/4)``.
    Returns ``(B, decoder_channels[-1], H, W)``.

    TODO(student)
    -------------
    1. For each stage: upsample, concatenate the matching skip, then a ``ConvBlock``.
    2. Prefer ``Upsample(scale_factor=2, mode="bilinear") + Conv2d`` over
       ``ConvTranspose2d``: transposed convolution produces checkerboard artefacts at
       these scales, which in a flood mask look like periodic false detections.
    3. Assert at every stage that the upsampled tensor and its skip agree spatially
       before concatenating. An off-by-one from an odd input size fails here with a
       clear message rather than deep inside a later layer.
    """

    def __init__(
        self,
        encoder_channels: tuple[int, ...],
        decoder_channels: tuple[int, ...],
        norm_groups: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Implement the decoder. See docs/07 section 9.1.")

    def forward(
        self,
        fused: torch.Tensor,
        skips: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


class FloodSegmenter(nn.Module):
    r"""Dual-branch SAR–optical flood segmenter with gated cross-attention fusion.

    Parameters
    ----------
    cfg
        Model configuration.

    Notes
    -----
    **The head emits one logit channel, not two.** A 2-channel softmax over
    :math:`\{`dry, flood:math:`\}` is over-parameterised by exactly one degree of
    freedom and complicates the temperature calibration of ``docs/09`` section 4. Use a
    single logit.

    **Never apply a sigmoid inside the model.** The losses require logits
    (``docs/07`` section 9.2, and :mod:`floodfusion.models.losses`).

    **Modality dropout is applied here, per sample.** ``forward`` accepts
    ``optical_available`` as a ``(B,)`` boolean; the training loop flips entries of it
    according to ``cfg.modality_dropout_p`` before calling. Dropping per *batch* rather
    than per *sample* correlates the decision with everything else in the batch and
    halves the number of effective draws (``docs/07`` section 7.3).

    **Never drop SAR.** It is always available operationally; training for its absence
    spends capacity on a case that never occurs.

    Shape contract
    --------------
    =============================== ==================== ===========
    argument                        shape                dtype
    =============================== ==================== ===========
    ``sar``                         ``(B, 5, H, W)``     float
    ``terrain``                     ``(B, 3, H, W)``     float
    ``optical``                     ``(B, 4, H, W)``     float
    ``optical_quality``             ``(B, 1, H, W)``     float
    ``optical_available``           ``(B,)``             bool
    **returns** ``logits``          ``(B, 1, H, W)``     float
    **returns** ``gate``            ``(B, 1, H', W')``   float
    =============================== ==================== ===========

    ``optical`` may also be ``None``, which is equivalent to ``optical_available`` being
    all-False and is the spelling ablation row 11 uses.

    The contracts the tests enforce (``docs/07`` section 9.5)
    --------------------------------------------------------
    1. ``forward(sar, terrain, optical=None)`` returns ``(B, 1, H, W)`` without raising.
    2. Attention weights sum to 1 (checked via ``reference_attention``).
    3. With the gate forced to 0, output is **bit-identical** to the SAR-only path.
    4. Optical present: gradients reach both encoders.
    5. Optical absent: gradients reach the SAR encoder and the missing embedding, and
       are ``None`` or zero for the optical encoder.
    6. No ``NaN``/``Inf`` in forward or backward under ``autocast(bfloat16)``.
    7. The head outputs logits — values outside ``[0, 1]`` occur.

    TODO(student)
    -------------
    1. Build the SAR encoder over ``N_SAR_CHANNELS + N_TERRAIN_CHANNELS`` inputs and the
       optical encoder over ``N_OPTICAL_CHANNELS``.
    2. Build the fusion block, the missing-modality embedding, the decoder and a 1x1
       head.
    3. In ``forward``: concatenate SAR and terrain; run the SAR encoder; then either run
       the optical encoder or substitute the embedding, **per sample**, using
       ``torch.where`` on a broadcast ``optical_available`` rather than a Python loop.
    4. Short-circuit when ``optical_available`` is all-False: skipping the optical
       encoder entirely saves ~40 % of encoder activations at inference and is what
       makes ablation row 11 cheap to evaluate. Verify it does not change the result.
    5. Fuse, decode, project to logits.
    6. Return ``(logits, gate)``.
    7. Expose ``enable_gradient_checkpointing()`` toggling checkpointing on encoder
       stage 1 and the final decoder stage only (``docs/08`` section 5.4).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError(
            "Implement the segmenter. The assembly diagram is docs/07 section 9.3."
        )

    def forward(
        self,
        sar: torch.Tensor,
        terrain: torch.Tensor,
        optical: torch.Tensor | None = None,
        optical_quality: torch.Tensor | None = None,
        optical_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        raise NotImplementedError

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Toggle activation checkpointing on the two memory-dominant stages.

        Notes
        -----
        Use ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``. Trades
        roughly 30 % extra compute for a 2--3x activation reduction. At :math:`256^2`
        you do not need it; at :math:`512^2` it is the deciding lever
        (``docs/08`` section 5.4). Do not enable it reflexively — 30 % across a
        25 GPU-hour ablation grid is over an hour of pure waste if unnecessary.
        """
        raise NotImplementedError
