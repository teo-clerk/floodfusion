r"""Decoder, head and the assembled dual-branch flood segmenter.

Architecture
------------
::

    sar (B,5,256,256) ─┐
    terrain (B,3,·,·) ─┴─► concat (B,8,256,256) ─► SAR encoder ─► s1, s2, s3, s4
                                                                  │   │   │   │
                                                                  │   │   └─+─┘  s3' = s3 + up(s4)
    optical[available] (b,4,256,256) ─► Optical encoder ─► o3' ──┐  │   │   │      (TopDownMerge)
                                   (o3' = o3 + up(o4))           │  │   │   │
              learned missing-modality embedding (others) ──────┤  │   │   │
                                                                 ▼  │   │   ▼
                                  CrossAttn(Q=o3', K=s3', V=s3') ──► × g ──► + ─► F_fused
                                                                    │   │        │
                                                                    ▼   ▼        ▼
                                                       Decoder (bridge, up+s2, up+s1, up)
                                                                         ▼
                                                       1x1 conv ─► logits (B,1,256,256)

The one deviation from the diagram of ``docs/07`` section 9.3 is the top-down merge of
:math:`s_4` into :math:`s_3` (see :class:`~floodfusion.models.encoders.TopDownMerge`): the
literal diagram computes :math:`s_4` and never uses it.

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    ModelConfig,
)
from floodfusion.models.encoders import ConvBlock, ModalityEncoder, TopDownMerge
from floodfusion.models.fusion_module import CrossAttentionFusion, MissingModalityEmbedding
from floodfusion.utils.shapes import ShapeError, assert_shape, named_dims

__all__ = ["FloodSegmenter", "SegmentationHead", "UNetDecoder", "validate_backbone_config"]

#: Expected flooded fraction of valid pixels (docs/05 section 8.2), used to initialise the
#: head bias at the prior log-odds so training starts at the base rate, not at p = 0.5.
PRIOR_POSITIVE_RATE: float = 0.02


def validate_backbone_config(cfg: ModelConfig) -> None:
    """The decoder and the top-down merge fix the geometry: fuse at the penultimate stage.

    ``ModelConfig`` permits any ``fusion_stage`` in ``[1, len(encoder_channels)]``; this
    implementation supports exactly ``len(encoder_channels) - 1`` (3 by default) with
    ``len(decoder_channels) == fusion_stage + 1``, and says so loudly otherwise.
    """
    n_enc, n_dec = len(cfg.encoder_channels), len(cfg.decoder_channels)
    if cfg.fusion_stage != n_enc - 1 or n_dec != cfg.fusion_stage + 1:
        raise ValueError(
            f"fusion_stage={cfg.fusion_stage} with {n_enc} encoder and {n_dec} decoder "
            "widths is not supported: fusion runs at the penultimate encoder stage "
            f"(fusion_stage = {n_enc - 1}) and the decoder needs fusion_stage + 1 widths"
        )


class SegmentationHead(nn.Module):
    """1x1 convolution to ONE logit channel, bias initialised at the prior log-odds."""

    def __init__(self, in_channels: int, prior: float = PRIOR_POSITIVE_RATE) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.constant_(self.conv.bias, math.log(prior / (1.0 - prior)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.conv(x)  # logits — never a sigmoid inside the model (docs/07 9.2)


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

    Implementation notes
    --------------------
    1. Each stage upsamples, concatenates the matching skip, then applies a ``ConvBlock``.
    2. ``Upsample(scale_factor=2, mode="bilinear") + Conv2d`` rather than
       ``ConvTranspose2d``: transposed convolution produces checkerboard artefacts at these
       scales, which in a flood mask look like periodic false detections.
    3. Every stage asserts that the upsampled tensor and its skip agree spatially before
       concatenating, so an off-by-one from an odd input size fails here with a clear
       message rather than deep inside a later layer.
    """

    def __init__(
        self,
        encoder_channels: tuple[int, ...],
        decoder_channels: tuple[int, ...],
        norm_groups: int,
    ) -> None:
        super().__init__()
        encoder_channels = tuple(encoder_channels)
        decoder_channels = tuple(decoder_channels)
        self.n_up = len(decoder_channels) - 1  # == the encoder stage of the fused input
        if not 1 <= self.n_up <= len(encoder_channels):
            raise ValueError(
                f"{len(decoder_channels)} decoder widths need a fused input at stage "
                f"{self.n_up}, outside the {len(encoder_channels)}-stage encoder"
            )
        self.fused_channels = encoder_channels[self.n_up - 1]
        self.skip_channels = encoder_channels[: self.n_up - 1]   # s1 .. s_{k-1}, finest first
        self.out_channels = decoder_channels[-1]
        self.bridge = ConvBlock(self.fused_channels, decoder_channels[0], norm_groups)
        stages = []
        for i in range(1, self.n_up + 1):
            skip_stage = self.n_up - i                        # 1-indexed; 0 means "no skip"
            skip_c = encoder_channels[skip_stage - 1] if skip_stage >= 1 else 0
            stages.append(ConvBlock(decoder_channels[i - 1] + skip_c, decoder_channels[i], norm_groups))
        self.stages = nn.ModuleList(stages)
        #: Checkpointing the full-resolution stage — with encoder s1, the two largest
        #: activation consumers (docs/08 section 4.4). Read by utils.vram.
        self.final_stage_checkpointing = False

    def set_checkpointing(self, enabled: bool) -> None:
        """Toggle activation checkpointing on the final (full-resolution) stage."""
        self.final_stage_checkpointing = bool(enabled)
        self.stages[-1].checkpointing_enabled = bool(enabled)

    def forward(
        self,
        fused: torch.Tensor,
        skips: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:  # noqa: D102
        dims = named_dims()
        assert_shape(fused, ("B", self.fused_channels, "h", "w"), dims, name="decoder.fused")
        if len(skips) != len(self.skip_channels):
            raise ShapeError(
                f"decoder expects {len(self.skip_channels)} skips (s1..s{self.n_up - 1}), "
                f"got {len(skips)}"
            )
        x = self.bridge(fused)
        for i, stage in enumerate(self.stages, start=1):
            # Upsample + conv (inside the ConvBlock) rather than ConvTranspose2d: transposed
            # convolutions leave checkerboard artefacts that read as periodic false water.
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            skip_stage = self.n_up - i
            if skip_stage >= 1:
                skip = skips[skip_stage - 1]
                if skip.shape[0] != x.shape[0] or skip.shape[-2:] != x.shape[-2:]:
                    raise ShapeError(
                        f"decoder stage {i}: upsampled {tuple(x.shape)} does not align with "
                        f"skip s{skip_stage} {tuple(skip.shape)} — an odd input size?"
                    )
                assert_shape(skip, ("B", self.skip_channels[skip_stage - 1], None, None), dims,
                             name=f"skip s{skip_stage}")
                x = torch.cat([x, skip], dim=1)
            last = i == len(self.stages)
            if last and self.final_stage_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x


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

    Implementation notes
    --------------------
    1. The SAR encoder takes ``N_SAR_CHANNELS + N_TERRAIN_CHANNELS`` inputs and the optical
       encoder ``N_OPTICAL_CHANNELS``.
    2. Then the fusion block, the missing-modality embedding, the decoder and a 1x1 head.
    3. ``forward`` concatenates SAR and terrain and runs the SAR encoder, then either the
       optical encoder or the embedding, **per sample**, with ``torch.where`` on a broadcast
       ``optical_available`` rather than a Python loop.
    4. When ``optical_available`` is all-False the optical encoder is skipped entirely,
       which saves ~40 % of encoder activations at inference and makes ablation row 11
       cheap to evaluate.
    5. Fuse, decode, project to logits.
    6. Returns ``(logits, gate)``.
    7. ``enable_gradient_checkpointing()`` toggles activation checkpointing on the
       memory-dominant stages of both encoders and the decoder (``docs/08`` section 5.4).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        validate_backbone_config(cfg)
        enc = tuple(cfg.encoder_channels)
        c_fuse, c_deep = enc[-2], enc[-1]
        self.sar_in_channels = N_SAR_CHANNELS + (N_TERRAIN_CHANNELS if cfg.use_terrain else 0)
        self.sar_encoder = ModalityEncoder(self.sar_in_channels, cfg)
        self.sar_merge = TopDownMerge(c_deep, c_fuse, cfg.norm_groups)
        self.optical_encoder = ModalityEncoder(N_OPTICAL_CHANNELS, cfg)
        self.optical_merge = TopDownMerge(c_deep, c_fuse, cfg.norm_groups)
        # Feature-level dropout on REAL optical features only: the optical=None path must
        # stay deterministic in train mode (tests/test_segmenter.py, row-11 evaluation).
        self.optical_dropout = nn.Dropout2d(cfg.dropout)
        self.missing_optical = MissingModalityEmbedding(c_fuse)
        self.fusion = CrossAttentionFusion(c_fuse, c_fuse, cfg)
        self.decoder = UNetDecoder(enc, tuple(cfg.decoder_channels), cfg.norm_groups)
        self.head = SegmentationHead(cfg.decoder_channels[-1])

    def forward(
        self,
        sar: torch.Tensor,
        terrain: torch.Tensor,
        optical: torch.Tensor | None = None,
        optical_quality: torch.Tensor | None = None,
        optical_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        dims = named_dims()
        assert_shape(sar, ("B", N_SAR_CHANNELS, "H", "W"), dims, name="sar")
        assert_shape(terrain, ("B", N_TERRAIN_CHANNELS, "H", "W"), dims, name="terrain")
        batch = dims["B"]
        x = torch.cat([sar, terrain], dim=1) if self.cfg.use_terrain else sar
        s1, s2, s3, s4 = self.sar_encoder(x)
        f_sar = self.sar_merge(s3, s4)

        available = _resolve_availability(optical, optical_available, batch, sar.device)
        h3, w3 = f_sar.shape[-2:]
        quality = _quality_at(optical_quality, optical, available, (h3, w3), f_sar)

        # Start from the learned "no optical here" embedding for EVERY sample, then
        # overwrite the samples that genuinely have optical (after modality dropout,
        # which the training loop has already applied to `available`). The optical
        # encoder runs on those samples only: no compute — and no gradient — is spent on
        # fabricated inputs, and GroupNorm makes each sample independent of the others.
        f_opt = self.missing_optical(batch, h3, w3, device=f_sar.device, dtype=f_sar.dtype)
        if bool(available.any()):
            assert optical is not None  # guaranteed by _resolve_availability
            assert_shape(optical, ("B", N_OPTICAL_CHANNELS, "H", "W"), dims, name="optical")
            idx = available.nonzero(as_tuple=True)[0]
            _o1, _o2, o3, o4 = self.optical_encoder(optical.index_select(0, idx))
            o_feat = self.optical_dropout(self.optical_merge(o3, o4))
            f_opt = f_opt.index_copy(0, idx, o_feat.to(f_opt.dtype))

        fused, gate = self.fusion(f_sar, f_opt, quality)
        logits = self.head(self.decoder(fused, (s1, s2)))
        assert_shape(logits, ("B", 1, "H", "W"), dims, name="logits")
        return logits, gate

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
        self.sar_encoder.set_checkpointing(enabled)
        self.optical_encoder.set_checkpointing(enabled)
        self.decoder.set_checkpointing(enabled)


def _resolve_availability(
    optical: torch.Tensor | None,
    optical_available: torch.Tensor | None,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """(B,) bool availability; ``optical=None`` means "absent for every sample" (row 11)."""
    if optical is None:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if optical_available is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    available = optical_available.to(device=device).reshape(-1)
    if available.numel() != batch:
        raise ShapeError(f"optical_available has {available.numel()} entries for a batch of {batch}")
    return available.bool()


def _quality_at(
    optical_quality: torch.Tensor | None,
    optical: torch.Tensor | None,
    available: torch.Tensor,
    size: tuple[int, int],
    like: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel optical quality pooled to the fusion grid; zero where optical is absent."""
    batch = available.numel()
    if optical_quality is None:
        q = torch.ones(batch, 1, *size, device=like.device, dtype=like.dtype)
    else:
        q = F.adaptive_avg_pool2d(optical_quality.to(like.dtype), size)
    if optical is None:
        return torch.zeros_like(q)
    return q * available.view(-1, 1, 1, 1).to(q.dtype)
