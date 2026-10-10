"""Modality-specific encoders for the SAR and optical branches.

Two encoders with the same *structure* and **no shared weights**. The statistics of dB
backscatter and of surface reflectance are entirely different — one is a log-power
quantity spanning roughly :math:`[-30, 5]`, the other a bounded physical ratio in
:math:`[0, 1]` — so the first-layer filters that suit one are wrong for the other.
Weight sharing is a legitimate ablation and is expected to hurt; if it does not, that is
informative about how little the optical branch is contributing.

Every model in the ablation grid (rows 4--9) uses the same backbone built here —
:class:`ModalityEncoder` plus :class:`TopDownMerge` — so that row-to-row differences
measure *fusion*, not receptive field or capacity.

References
----------
``docs/07_fusion_architectures.md`` sections 3, 6.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from floodfusion.config import ModelConfig
from floodfusion.utils.shapes import assert_shape, named_dims

__all__ = ["ConvBlock", "EncoderFeatures", "ModalityEncoder", "TopDownMerge"]

#: Feature maps emitted by an encoder, ordered from finest to coarsest resolution.
EncoderFeatures = tuple[torch.Tensor, ...]


def _check_groups(channels: int, norm_groups: int) -> None:
    if norm_groups < 1 or channels % norm_groups != 0:
        raise ValueError(
            f"GroupNorm needs channels divisible by norm_groups: {channels} channels "
            f"cannot be split into {norm_groups} groups. Choose widths that are multiples "
            "of norm_groups (docs/07 section 3.3)."
        )


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with normalisation and activation.

    Parameters
    ----------
    in_channels, out_channels
        Channel counts.
    norm_groups
        Number of groups for ``GroupNorm``.

    Notes
    -----
    **Use ``GroupNorm``, not ``BatchNorm``** (``docs/07`` section 3.3). Two reasons
    specific to this project:

    1. The VRAM budget forces micro-batches of 4--8 with gradient accumulation, and
       ``BatchNorm`` statistics from a batch of 4 are noisy — with the noise correlated
       with chip content, because chips are spatially blocked.
    2. Under modality dropout the optical branch sees a mixture of real and
       missing-modality inputs within one batch, so its running statistics would be
       computed over a distribution that does not exist at inference.

    ``GroupNorm`` is per-sample and immune to both. It also makes a sample's output
    independent of which other samples share its batch — which is what lets the
    segmenter run the optical encoder on only the samples that have optical.

    Convolutions immediately followed by a normalisation carry no bias: the norm's shift
    subsumes it. Out-of-place SiLU throughout (an in-place activation would overwrite a
    tensor autograd needs and interacts badly with checkpointing).

    Shape contract
    --------------
    ``(B, in_channels, H, W) -> (B, out_channels, H, W)``. Spatial size is preserved;
    downsampling happens between blocks, not inside them.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        _check_groups(out_channels, norm_groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(x)))


class _DownStage(nn.Module):
    """Stride-2 3x3 conv + GN + SiLU (halves resolution, sets the width), then a ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        _check_groups(out_channels, norm_groups)
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.SiLU()
        self.block = ConvBlock(out_channels, out_channels, norm_groups)
        #: Read by ``utils.vram.collect_block_shapes``; toggled by the segmenter.
        self.checkpointing_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.checkpointing_enabled and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.act(self.norm(self.down(x))))


class ModalityEncoder(nn.Module):
    r"""Four-stage convolutional encoder for one modality.

    Parameters
    ----------
    in_channels
        8 for the SAR branch (5 SAR + 3 terrain), 5 without terrain (row 4), 4 for the
        optical branch, 12 for early fusion (row 7).
    cfg
        Model configuration supplying ``encoder_channels`` and ``norm_groups``.

    Notes
    -----
    **Terrain is concatenated to the SAR input**, not given a third encoder
    (``docs/07`` section 3.1). Terrain is always available and is physically tied to the
    radar geometry — :math:`\theta_{\mathrm{loc}}` is a *radar* quantity — so a third
    branch would need its own missing-modality handling for no benefit.

    Stage table for a 256-pixel chip, with the default
    ``encoder_channels = (64, 128, 256, 512)``:

    ===== ============ ========== =====================
    stage resolution   channels   tokens :math:`N = HW`
    ===== ============ ========== =====================
    s1    128 x 128    64         16 384
    s2    64 x 64      128        4 096
    s3    32 x 32      256        1 024
    s4    16 x 16      512        256
    ===== ============ ========== =====================

    Each stage is a stride-2 3x3 convolution (the stem for s1) followed by a
    :class:`ConvBlock`. Strided convolutions rather than max-pooling: the downsampling is
    learned, which matters for speckle, where a max picks the brightest speckle draw.

    Shape contract
    --------------
    ``(B, in_channels, H, W)`` in; a tuple of four tensors out, with shapes
    ``(B, C_i, H / 2**i, W / 2**i)`` for :math:`i = 1 \ldots 4`. ``H`` and ``W`` must be
    divisible by 16.

    The encoder returns **all** stages because the decoder needs s1 and s2 as skip
    connections, the fusion block needs s3, and :class:`TopDownMerge` folds s4 into s3.

    **Checkpointing**: ``docs/08`` section 4.4 shows s1 dominates encoder activation
    memory, so only s1 is checkpointable (``set_checkpointing``); checkpointing s4 saves
    almost nothing.
    """

    def __init__(self, in_channels: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        widths = tuple(cfg.encoder_channels)
        if len(widths) < 2:
            raise ValueError("encoder_channels needs at least two stages")
        stages = []
        previous = in_channels
        for width in widths:
            stages.append(_DownStage(previous, width, cfg.norm_groups))
            previous = width
        self.stages = nn.ModuleList(stages)
        self.widths = widths

    def set_checkpointing(self, enabled: bool) -> None:
        """Toggle activation checkpointing on stage 1 (the memory-dominant one)."""
        self.stages[0].checkpointing_enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> EncoderFeatures:  # noqa: D102
        dims = named_dims()
        assert_shape(x, ("B", self.in_channels, "H", "W"), dims, name="encoder_input")
        factor = 2 ** len(self.stages)
        if dims["H"] % factor or dims["W"] % factor:
            raise ValueError(
                f"input {dims['H']}x{dims['W']} must be divisible by {factor} for "
                f"{len(self.stages)} stride-2 stages"
            )
        features = []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            assert_shape(
                x, ("B", self.widths[i - 1], dims["H"] // 2**i, dims["W"] // 2**i), dims,
                name=f"s{i}",
            )
            features.append(x)
        return tuple(features)


class TopDownMerge(nn.Module):
    r"""Fold the coarsest stage into the fusion stage: :math:`s_3' = s_3 + \mathrm{up}(\mathrm{GN}(W s_4))`.

    An FPN-style lateral connection. The literal diagram of ``docs/07`` section 9.3
    computes :math:`s_4` and never consumes it, which would leave ~75 % of each encoder's
    parameters without gradient; the decoder contract (fused input at :math:`H/8` plus
    skips :math:`s_1, s_2`) leaves no place for it either. Merging it into :math:`s_3`
    gives the fusion stage — and therefore the SAR-only path that requirement R2 makes the
    operational case — a receptive field of ~150 px instead of ~70 px, and keeps the
    parameter budget at the documented ~19 M.

    The R2 contract is unaffected: :math:`F_{\mathrm{fused}} = F_s + g \odot
    \mathrm{CrossAttn}(\cdot)` with :math:`F_s := s_3'`, so :math:`g = 0` still returns
    the (merged) SAR features bit for bit.
    """

    def __init__(self, deep_channels: int, target_channels: int, norm_groups: int) -> None:
        super().__init__()
        _check_groups(target_channels, norm_groups)
        self.lateral = nn.Conv2d(deep_channels, target_channels, 1, bias=False)
        self.norm = nn.GroupNorm(norm_groups, target_channels)

    def forward(self, target: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:  # noqa: D102
        up = F.interpolate(self.norm(self.lateral(deep)), size=target.shape[-2:],
                           mode="bilinear", align_corners=False)
        return target + up
