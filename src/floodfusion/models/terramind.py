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

**The fair variants (v1.2,** :data:`FAIR_VARIANTS` **).** v1.1's TerraMind saw a single date, with a frozen
encoder. The v1.2 opponent is TerraMind as strong as an 8 GB GPU allows, trained on the same data as our model:

- **Siamese.** One encoder runs twice with shared weights: on the post-event inputs (S1, plus S2 where usable) and on
  the pre-event S1. Its tokens are concatenated per layer, so the decoder sees ``2 × D`` channels and learns the
  change itself. The pre-event pass has no S2: the stack has no pre-event optical, and a difference post − pre would
  confound "optical present" with "change".
- **Adaptation.** LoRA of rank :data:`LORA_RANK` on the encoder's attention projections
  (:mod:`floodfusion.models.lora`), or a frozen encoder. Size and adaptation follow the rule logged in
  ``reports/v1_2_deviations.md`` before any result: base with LoRA, else base frozen, else small with LoRA.

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

__all__ = ["FAIR_VARIANTS", "LORA_ALPHA", "LORA_RANK", "TERRAMIND_ARCHITECTURES", "TERRAMIND_VARIANTS",
           "TerraMindSegmenter"]

#: Architecture name -> TerraTorch backbone (v1.1: single date, frozen encoder).
TERRAMIND_VARIANTS = {"terramind_tiny": "terramind_v1_tiny", "terramind_small": "terramind_v1_small"}
LORA_RANK = 8
LORA_ALPHA = 16.0
#: v1.2 fair-fight variants: architecture name -> (TerraTorch backbone, LoRA rank; 0 = frozen encoder). Siamese.
FAIR_VARIANTS = {"terramind_base_fair": ("terramind_v1_base", LORA_RANK),
                 "terramind_base_fair_frozen": ("terramind_v1_base", 0),
                 "terramind_small_fair": ("terramind_v1_small", LORA_RANK)}
TERRAMIND_ARCHITECTURES = frozenset(TERRAMIND_VARIANTS) | frozenset(FAIR_VARIANTS)
_S2_BANDS = ("GREEN", "NIR_BROAD", "SWIR_1")
#: TerraMind pre-training statistics of those bands (S2L2A digital numbers).
_S2_MEAN = (1718.197, 3083.234, 2424.884)
_S2_STD = (2038.973, 1871.918, 1434.261)
_REFLECTANCE_TO_DN = 10_000.0
#: TerraMind pre-training statistics of its DEM modality (metres): the fair variants read a dem.tif sidecar.
_DEM_MEAN, _DEM_STD = 670.665, 951.272
_NECKS = [{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
          {"name": "ReshapeTokensToImage", "remove_cls_token": False},
          {"name": "LearnedInterpolateToPyramidal"}]


class _Siamese(nn.Module):
    """One TerraMind encoder on the post-event inputs and on the pre-event S1 (``S1RTC_pre``); tokens concatenated."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.out_channels = [2 * c for c in encoder.out_channels]

    def forward(self, x: dict[str, torch.Tensor], **kwargs: Any) -> list[torch.Tensor]:
        post = {k: v for k, v in x.items() if k != "S1RTC_pre"}   # DEM rides with the post-event pass
        after, before = self.encoder(post, **kwargs), self.encoder({"S1RTC": x["S1RTC_pre"]}, **kwargs)
        return [torch.cat([a, b.to(a.dtype)], dim=-1) for a, b in zip(after, before, strict=True)]


class TerraMindSegmenter(nn.Module):
    """TerraMind encoder (frozen) + UNet decoder; returns ``(logits, None)`` like every zoo model."""

    def __init__(self, cfg: ModelConfig, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from terratorch.models import EncoderDecoderFactory
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("the TerraMind row needs the optional extra: uv sync --extra fm") from exc
        modalities, bands = ["S2L2A", "S1RTC"], {"S2L2A": list(_S2_BANDS), "S1RTC": ["VV", "VH"]}
        self.siamese = cfg.architecture in FAIR_VARIANTS
        if self.siamese:   # TerraMind's native terrain input, as our model reads slope, HAND and incidence
            modalities, bands = [*modalities, "DEM"], {**bands, "DEM": ["DEM"]}
        if self.siamese:
            from terratorch.models.encoder_decoder_factory import _get_backbone

            name, rank = FAIR_VARIANTS[cfg.architecture]
            backbone: Any = {"backbone": _Siamese(_get_backbone(name, pretrained=pretrained, modalities=modalities,
                                                                bands=bands))}
        else:
            rank = 0
            backbone = {"backbone": TERRAMIND_VARIANTS[cfg.architecture], "backbone_pretrained": pretrained,
                        "backbone_modalities": modalities, "backbone_bands": bands}
        self.model = EncoderDecoderFactory().build_model(
            task="segmentation", **backbone, necks=_NECKS, decoder="UNetDecoder",
            decoder_channels=list(cfg.decoder_channels), head_dropout=cfg.dropout, num_classes=1,
        )
        if self.model.aux_heads:   # _decode reproduces PixelWiseModel.forward's main head only
            raise ValueError("the TerraMind wrapper does not support auxiliary heads")
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        if rank:
            from floodfusion.models.lora import apply_lora

            apply_lora(self.model.encoder, rank=rank, alpha=LORA_ALPHA)
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
                context: torch.Tensor | None = None, context_available: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, Any]:
        s1 = sar[:, :2] * self.sar_scale + self.sar_shift           # post VV, VH (z-scored dB)
        pre: dict[str, torch.Tensor] = {}
        if self.siamese:
            pre["S1RTC_pre"] = sar[:, 2:4] * self.sar_scale + self.sar_shift
            # The dem.tif sidecar in metres, standardised as in pre-training; a stack without one reads the mean (0).
            if context is not None and context_available is not None:
                dem = (context[:, :1] - _DEM_MEAN) / _DEM_STD
                pre["DEM"] = torch.where(context_available[:, :1].bool().view(-1, 1, 1, 1), dem, torch.zeros_like(dem))
            else:
                pre["DEM"] = sar.new_zeros(sar.shape[0], 1, *sar.shape[-2:])
        out_size = self.model.image_size_out or tuple(sar.shape[-2:])
        if optical is None or optical_available is None or not bool(optical_available.any()):
            features, size = self._encode({"S1RTC": s1, **pre})
            return self._decode(features, size, out_size), None
        s2 = (optical[:, :3].clamp(0.0, 1.0) * _REFLECTANCE_TO_DN - self.s2_mean) / self.s2_std
        if optical_quality is not None:
            s2 = torch.where(optical_quality >= CLEAR_QUALITY, s2, torch.zeros_like(s2))
        available = optical_available.bool()
        if bool(available.all()):
            features, size = self._encode({"S2L2A": s2, "S1RTC": s1, **pre})
            return self._decode(features, size, out_size), None
        with_optical, size = self._encode({"S2L2A": s2[available], "S1RTC": s1[available],
                                           **{k: v[available] for k, v in pre.items()}})
        sar_only, _ = self._encode({"S1RTC": s1[~available], **{k: v[~available] for k, v in pre.items()}})
        order = torch.cat([available.nonzero().flatten(), (~available).nonzero().flatten()])
        inverse = torch.argsort(order)
        features = [torch.cat([a, b.to(a.dtype)])[inverse] for a, b in zip(with_optical, sar_only, strict=True)]
        return self._decode(features, size, out_size), None
