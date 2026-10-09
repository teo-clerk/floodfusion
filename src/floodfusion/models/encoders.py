"""Modality-specific encoders for the SAR and optical branches.

Two encoders with the same *structure* and **no shared weights**. The statistics of dB
backscatter and of surface reflectance are entirely different — one is a log-power
quantity spanning roughly :math:`[-30, 5]`, the other a bounded physical ratio in
:math:`[0, 1]` — so the first-layer filters that suit one are wrong for the other.
Weight sharing is a legitimate ablation and is expected to hurt; if it does not, that is
informative about how little the optical branch is contributing.

References
----------
``docs/07_fusion_architectures.md`` sections 3, 6.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from floodfusion.config import ModelConfig

__all__ = ["ConvBlock", "ModalityEncoder", "EncoderFeatures"]

#: Feature maps emitted by an encoder, ordered from finest to coarsest resolution.
EncoderFeatures = tuple[torch.Tensor, ...]


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

    ``GroupNorm`` is per-sample and immune to both.

    Shape contract
    --------------
    ``(B, in_channels, H, W) -> (B, out_channels, H, W)``. Spatial size is preserved;
    downsampling happens between blocks, not inside them.

    TODO(student)
    -------------
    1. Build ``Conv2d -> GroupNorm -> activation`` twice.
    2. Use ``bias=False`` on convolutions immediately followed by a normalisation: the
       norm's shift parameter subsumes the bias, so the bias is redundant parameters and
       redundant memory.
    3. Ensure ``out_channels`` is divisible by ``norm_groups``, and raise a clear error
       if not — the ``GroupNorm`` failure message is not helpful.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        raise NotImplementedError("Implement ConvBlock. See docs/07 section 3.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


class ModalityEncoder(nn.Module):
    r"""Four-stage convolutional encoder for one modality.

    Parameters
    ----------
    in_channels
        8 for the SAR branch (5 SAR + 3 terrain), 4 for the optical branch.
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

    The token counts matter: they determine whether cross-attention is affordable at a
    given stage. Reproduce the memory arithmetic of ``docs/07`` section 6 for your own
    batch size before choosing ``fusion_stage``.

    Shape contract
    --------------
    ``(B, in_channels, H, W)`` in; a tuple of four tensors out, with shapes
    ``(B, C_i, H / 2**i, W / 2**i)`` for :math:`i = 1 \ldots 4`.

    The encoder returns **all** stages because the decoder needs s1 and s2 as skip
    connections and the fusion block needs s3. Returning only the last stage is a
    common simplification that makes the U-Net decoder impossible.

    TODO(student)
    -------------
    1. Build a stem that maps ``in_channels`` to ``encoder_channels[0]`` while halving
       resolution.
    2. Build one ``ConvBlock`` plus a downsample per subsequent stage.
    3. Collect and return every stage output as a tuple.
    4. Assert the returned shapes with ``utils.shapes.assert_shape``, binding ``B``,
       ``H`` and ``W`` once so a stride error anywhere is caught immediately.
    5. Support ``torch.utils.checkpoint`` on the **early** stages: ``docs/08`` section
       4.4 shows s1 dominates activation memory, and ``docs/08`` section 5.4 explains
       when to switch it on. Checkpointing s4 saves almost nothing.
    """

    def __init__(self, in_channels: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        raise NotImplementedError("Implement ModalityEncoder. See docs/07 section 3.2.")

    def forward(self, x: torch.Tensor) -> EncoderFeatures:  # noqa: D102
        raise NotImplementedError
