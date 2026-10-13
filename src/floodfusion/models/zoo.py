r"""The model zoo of the ablation grid (``MILESTONES.md`` section 6, rows 4--13).

Every architecture shares **one backbone** — :class:`~floodfusion.models.encoders.
ModalityEncoder` + :class:`~floodfusion.models.encoders.TopDownMerge` +
:class:`~floodfusion.models.segmenter.UNetDecoder` + a one-logit head — and differs only
in where and how optical evidence enters. Without that, the row-9 minus row-4 gap would
mix "fusion" with "receptive field" and "capacity", and measure neither.

======================= ====== ==========================================================
architecture            rows   what it is
======================= ====== ==========================================================
``unet_sar``            4, 5   one U-Net over SAR (+ terrain if ``use_terrain``)
``unet_optical``        6      one U-Net over the 4 optical channels (cloud-free chips)
``fusion_early``        7      one U-Net over all 12 channels; absent optical = constant 0
``fusion_late``         8      SAR U-Net and optical U-Net blended at the logit level
``fusion_crossattn``    9--13  :class:`~floodfusion.models.segmenter.FloodSegmenter`
======================= ====== ==========================================================

All share the forward signature ``(sar, terrain, optical=None, optical_quality=None,
optical_available=None) -> (logits, gate_or_None)``, so the training and evaluation code
paths are identical for every row (``docs/09`` section 2.3 rule 2).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from floodfusion.config import (
    N_OPTICAL_CHANNELS,
    N_SAR_CHANNELS,
    N_TERRAIN_CHANNELS,
    ModelConfig,
)
from floodfusion.models.encoders import ModalityEncoder, TopDownMerge
from floodfusion.models.segmenter import (
    FloodSegmenter,
    SegmentationHead,
    UNetDecoder,
    _resolve_availability,
    validate_backbone_config,
)
from floodfusion.utils.shapes import assert_shape, named_dims

__all__ = [
    "EarlyFusionSegmenter",
    "LateFusionSegmenter",
    "UNetSegmenter",
    "build_model",
    "count_parameters",
]


class _Backbone(nn.Module):
    """Encoder + top-down merge + U-Net decoder + one-logit head: ``(B,C,H,W) -> (B,1,H,W)``."""

    def __init__(self, in_channels: int, cfg: ModelConfig) -> None:
        super().__init__()
        validate_backbone_config(cfg)
        enc = tuple(cfg.encoder_channels)
        self.encoder = ModalityEncoder(in_channels, cfg)
        self.merge = TopDownMerge(enc[-1], enc[-2], cfg.norm_groups)
        self.decoder = UNetDecoder(enc, tuple(cfg.decoder_channels), cfg.norm_groups)
        self.head = SegmentationHead(cfg.decoder_channels[-1])

    def set_checkpointing(self, enabled: bool) -> None:
        self.encoder.set_checkpointing(enabled)
        self.decoder.set_checkpointing(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        s1, s2, s3, s4 = self.encoder(x)
        return self.head(self.decoder(self.merge(s3, s4), (s1, s2)))


class UNetSegmenter(nn.Module):
    """Single-modality U-Net: rows 4/5 (``unet_sar``) and 6 (``unet_optical``)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.architecture not in {"unet_sar", "unet_optical"}:
            raise ValueError(f"UNetSegmenter cannot build {cfg.architecture!r}")
        self.cfg = cfg
        self.optical_only = cfg.architecture == "unet_optical"
        if self.optical_only:
            in_channels = N_OPTICAL_CHANNELS
        else:
            in_channels = N_SAR_CHANNELS + (N_TERRAIN_CHANNELS if cfg.use_terrain else 0)
        self.in_channels = in_channels
        self.net = _Backbone(in_channels, cfg)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:  # noqa: D102
        self.net.set_checkpointing(enabled)

    def forward(  # noqa: D102
        self,
        sar: torch.Tensor,
        terrain: torch.Tensor,
        optical: torch.Tensor | None = None,
        optical_quality: torch.Tensor | None = None,
        optical_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        dims = named_dims()
        assert_shape(sar, ("B", N_SAR_CHANNELS, "H", "W"), dims, name="sar")
        if self.optical_only:
            if optical is None:
                raise ValueError(
                    "unet_optical (row 6) needs optical input; it is defined only on "
                    "cloud-free chips — filter the records, do not feed it absent optical"
                )
            assert_shape(optical, ("B", N_OPTICAL_CHANNELS, "H", "W"), dims, name="optical")
            x = optical
        else:
            assert_shape(terrain, ("B", N_TERRAIN_CHANNELS, "H", "W"), dims, name="terrain")
            x = torch.cat([sar, terrain], dim=1) if self.cfg.use_terrain else sar
        return self.net(x), None


class EarlyFusionSegmenter(nn.Module):
    """Row 7: channel concatenation of all 12 channels into one encoder.

    Missing optical **must be filled with a constant** — here 0, the normalised value the
    dataset also uses for invalid pixels — which the network then has to learn to read as
    "absent" although 0 also occurs legitimately (``docs/07`` section 2.1). Modality
    dropout applies the same fill per sample.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = _Backbone(N_SAR_CHANNELS + N_OPTICAL_CHANNELS + N_TERRAIN_CHANNELS, cfg)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:  # noqa: D102
        self.net.set_checkpointing(enabled)

    def forward(  # noqa: D102
        self,
        sar: torch.Tensor,
        terrain: torch.Tensor,
        optical: torch.Tensor | None = None,
        optical_quality: torch.Tensor | None = None,
        optical_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        dims = named_dims()
        assert_shape(sar, ("B", N_SAR_CHANNELS, "H", "W"), dims, name="sar")
        assert_shape(terrain, ("B", N_TERRAIN_CHANNELS, "H", "W"), dims, name="terrain")
        available = _resolve_availability(optical, optical_available, dims["B"], sar.device)
        if optical is None:
            optical = sar.new_zeros(dims["B"], N_OPTICAL_CHANNELS, dims["H"], dims["W"])
        else:
            assert_shape(optical, ("B", N_OPTICAL_CHANNELS, "H", "W"), dims, name="optical")
            optical = optical * available.view(-1, 1, 1, 1).to(optical.dtype)
        x = torch.cat([sar, optical.to(sar.dtype), terrain], dim=1)
        return self.net(x), None


class LateFusionSegmenter(nn.Module):
    r"""Row 8: two complete U-Nets combined at the decision (logit) level.

    .. math::  z = z_s + g\,(z_o - z_s), \qquad g = \sigma(a\,q + b)\cdot\mathbb{1}[\text{available}]

    with :math:`q` the per-pixel optical quality. Where optical is absent :math:`g = 0` and
    :math:`z = z_s` exactly — the baseline's override rule (``docs/06`` section 5.1 step 5)
    made soft and learnable. What it cannot express is cross-modal interaction *before* the
    decision (``docs/07`` section 2.2).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sar_in = N_SAR_CHANNELS + (N_TERRAIN_CHANNELS if cfg.use_terrain else 0)
        self.sar_net = _Backbone(sar_in, cfg)
        self.optical_net = _Backbone(N_OPTICAL_CHANNELS, cfg)
        self.blend_scale = nn.Parameter(torch.tensor(4.0))
        self.blend_bias = nn.Parameter(torch.tensor(cfg.gate_bias_init))

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:  # noqa: D102
        self.sar_net.set_checkpointing(enabled)
        self.optical_net.set_checkpointing(enabled)

    def forward(  # noqa: D102
        self,
        sar: torch.Tensor,
        terrain: torch.Tensor,
        optical: torch.Tensor | None = None,
        optical_quality: torch.Tensor | None = None,
        optical_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dims = named_dims()
        assert_shape(sar, ("B", N_SAR_CHANNELS, "H", "W"), dims, name="sar")
        assert_shape(terrain, ("B", N_TERRAIN_CHANNELS, "H", "W"), dims, name="terrain")
        x = torch.cat([sar, terrain], dim=1) if self.cfg.use_terrain else sar
        z_sar = self.sar_net(x)
        available = _resolve_availability(optical, optical_available, dims["B"], sar.device)
        gate = torch.zeros_like(z_sar)
        if not bool(available.any()):
            return z_sar, gate
        assert optical is not None
        assert_shape(optical, ("B", N_OPTICAL_CHANNELS, "H", "W"), dims, name="optical")
        idx = available.nonzero(as_tuple=True)[0]
        z_opt = self.optical_net(optical.index_select(0, idx))
        if optical_quality is None:
            q = torch.ones_like(z_opt)
        else:
            q = optical_quality.index_select(0, idx).to(z_opt.dtype)
        g_sub = torch.sigmoid(self.blend_scale * q + self.blend_bias)
        z_sub = z_sar.index_select(0, idx)
        blended = z_sub + g_sub * (z_opt.to(z_sub.dtype) - z_sub)
        logits = z_sar.index_copy(0, idx, blended)
        gate = gate.index_copy(0, idx, g_sub.to(gate.dtype))
        return logits, gate


def _terramind(cfg: ModelConfig, *, pretrained: bool = True) -> nn.Module:
    from floodfusion.models.terramind import TerraMindSegmenter

    return TerraMindSegmenter(cfg, pretrained=pretrained)


#: Architectures whose constructor fetches pre-trained weights (the rest train from scratch).
_PRETRAINED = {"terramind_tiny", "terramind_small", "terramind_base_fair", "terramind_base_fair_frozen",
               "terramind_small_fair"}


_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "terramind_tiny": _terramind,
    "terramind_small": _terramind,
    "terramind_base_fair": _terramind,
    "terramind_base_fair_frozen": _terramind,
    "terramind_small_fair": _terramind,
    "unet_sar": UNetSegmenter,
    "unet_optical": UNetSegmenter,
    "fusion_early": EarlyFusionSegmenter,
    "fusion_late": LateFusionSegmenter,
    "fusion_crossattn": FloodSegmenter,
}


def build_model(cfg: ModelConfig, *, pretrained: bool = True) -> nn.Module:
    """Instantiate the network for ``cfg.architecture`` (rows 4--13 of the grid).

    ``pretrained=False`` skips downloading foundation-model weights — for rebuilding a model
    whose trained state is about to be loaded from a checkpoint.
    """
    if cfg.architecture == "threshold_baseline":
        raise ValueError(
            "rows 1-3 are the non-learned physics baseline; run floodfusion.baseline_runner"
        )
    try:
        cls = _REGISTRY[cfg.architecture]
    except KeyError as exc:  # pragma: no cover - ModelConfig already validates
        raise ValueError(f"unknown architecture {cfg.architecture!r}") from exc
    return cls(cfg, pretrained=pretrained) if cfg.architecture in _PRETRAINED else cls(cfg)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
