r"""TerraMind (IBM / ESA, 2025) as a comparison row, behind the zoo's forward signature.

The question it answers: *why not just fine-tune the foundation model?* TerraMind-1.0 is a
multimodal any-to-any model pre-trained on 9 M aligned Sentinel-1/Sentinel-2/DEM samples;
TerraTorch provides it with a segmentation head. This wrapper uses TerraMind's own
Sen1Floods11 recipe (``IBM/terramind`` ``configs/terramind_v1_*_sen1floods11.yaml``):
frozen encoder, layers 3/6/9/12 reshaped into a pyramid, a UNet decoder (256-128-64-32).

Inputs are mapped from this project's batch onto TerraMind's modalities:

- ``S2L2A`` with the band subset ``GREEN, NIR_BROAD, SWIR_1`` (our B3, B8, B11): TerraTorch
  copies the pre-trained patch-embedding weights band by band. Our reflectance (0-1) is
  converted to digital numbers and standardised with TerraMind's pre-training statistics.
- ``S1RTC`` ``VV, VH``: TerraMind standardises the post-event gamma0 in dB per channel; our
  dataset already does exactly that with the training split's statistics, so the
  z-scored channels are passed through a learnable per-channel affine initialised to
  identity (the two standardisations differ only in their constants).
- The pre-event SAR, the change channel and terrain are **not** given to TerraMind: it is a
  single-date model, as published. That is part of the comparison, and stated with it.

A sample without usable optical is passed as ``S1RTC`` alone (TerraMind accepts any subset
of its modalities), so missing optical never reaches it as zeros; modality dropout and
the SAR-only evaluation (row 11's question) therefore apply unchanged. Only the encoder runs
per subset: its features (one token grid per layer, averaged over the modalities present)
are put back in batch order and the neck, decoder and head run **once** on the whole batch,
so their BatchNorm layers never see a sub-batch of one or two chips (v1.1 erratum,
``reports/v1_1_erratum.md``; in evaluation mode the two orders give the same logits). Within a chip that
has optical, **clouded pixels** (quality below the clear threshold) are set to TerraMind's
pre-training mean (0 after standardisation): it has no per-pixel gate, unlike row 9, and is
not handed cloud tops as if they were ground.

Caveats stated wherever its numbers appear: its encoder was pre-trained on global
Sentinel-1/2 data (so "never saw the event" holds for the decoder only); the band subset and
necks follow IBM's recipe, developed on Sen1Floods11's official splits; the training schedule
is row 9's, tuned for neither model; its SAR input is our z-score, not its pre-training
normalisation, adapted by four affine parameters.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from floodfusion.config import ModelConfig
from floodfusion.data.chipping import CLEAR_QUALITY

__all__ = ["TERRAMIND_VARIANTS", "TerraMindSegmenter"]

#: Architecture name -> TerraTorch backbone.
TERRAMIND_VARIANTS = {"terramind_tiny": "terramind_v1_tiny", "terramind_small": "terramind_v1_small"}
_S2_BANDS = ("GREEN", "NIR_BROAD", "SWIR_1")
#: TerraMind pre-training statistics of those bands (S2L2A digital numbers).
_S2_MEAN = (1718.197, 3083.234, 2424.884)
_S2_STD = (2038.973, 1871.918, 1434.261)
_REFLECTANCE_TO_DN = 10_000.0
_NECKS = [{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
          {"name": "ReshapeTokensToImage", "remove_cls_token": False},
          {"name": "LearnedInterpolateToPyramidal"}]


class TerraMindSegmenter(nn.Module):
    """TerraMind encoder (frozen) + UNet decoder; returns ``(logits, None)`` like every zoo model."""

    def __init__(self, cfg: ModelConfig, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from terratorch.models import EncoderDecoderFactory
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("the TerraMind row needs the optional extra: uv sync --extra fm") from exc
        self.model = EncoderDecoderFactory().build_model(
            task="segmentation", backbone=TERRAMIND_VARIANTS[cfg.architecture], backbone_pretrained=pretrained,
            backbone_modalities=["S2L2A", "S1RTC"],
            backbone_bands={"S2L2A": list(_S2_BANDS), "S1RTC": ["VV", "VH"]},
            necks=_NECKS, decoder="UNetDecoder", decoder_channels=list(cfg.decoder_channels),
            head_dropout=cfg.dropout, num_classes=1,
        )
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        self.sar_scale = nn.Parameter(torch.ones(1, 2, 1, 1))
        self.sar_shift = nn.Parameter(torch.zeros(1, 2, 1, 1))
        self.register_buffer("s2_mean", torch.tensor(_S2_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("s2_std", torch.tensor(_S2_STD).view(1, 3, 1, 1), persistent=False)

    def _encode(self, inputs: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], Any]:
        """Encoder features of one modality subset, and the (padded) input size."""
        from terratorch.models.utils import get_image_size, pad_images

        m = self.model
        if m.patch_size and m.padding is not None:
            inputs = pad_images(inputs, m.patch_size, m.padding)
        return list(m.encoder(inputs)), get_image_size(inputs)

    def _decode(self, features: list[torch.Tensor], input_size: Any, out_size: Any) -> torch.Tensor:
        """Neck, decoder and head on the whole batch (``PixelWiseModel.forward`` after its encoder)."""
        from terratorch.models.utils import center_crop

        m = self.model
        features = m.neck(features, image_size=input_size)
        mask = m.head(m.decoder([f.clone() for f in features]))
        if m.rescale and mask.shape[-2:] != input_size:
            mask = F.interpolate(mask, size=input_size, mode="bilinear")
        mask = center_crop(mask, out_size)
        return mask if mask.ndim == 4 else mask.unsqueeze(1)

    def forward(self, sar: torch.Tensor, terrain: torch.Tensor | None = None, optical: torch.Tensor | None = None,
                optical_quality: torch.Tensor | None = None, optical_available: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, Any]:
        s1 = sar[:, :2] * self.sar_scale + self.sar_shift           # post VV, VH (z-scored dB)
        out_size = self.model.image_size_out or tuple(sar.shape[-2:])
        if optical is None or optical_available is None or not bool(optical_available.any()):
            features, size = self._encode({"S1RTC": s1})
            return self._decode(features, size, out_size), None
        s2 = (optical[:, :3].clamp(0.0, 1.0) * _REFLECTANCE_TO_DN - self.s2_mean) / self.s2_std
        if optical_quality is not None:
            s2 = torch.where(optical_quality >= CLEAR_QUALITY, s2, torch.zeros_like(s2))
        available = optical_available.bool()
        if bool(available.all()):
            features, size = self._encode({"S2L2A": s2, "S1RTC": s1})
            return self._decode(features, size, out_size), None
        with_optical, size = self._encode({"S2L2A": s2[available], "S1RTC": s1[available]})
        sar_only, _ = self._encode({"S1RTC": s1[~available]})
        order = torch.cat([available.nonzero().flatten(), (~available).nonzero().flatten()])
        inverse = torch.argsort(order)
        features = [torch.cat([a, b.to(a.dtype)])[inverse] for a, b in zip(with_optical, sar_only, strict=True)]
        return self._decode(features, size, out_size), None
