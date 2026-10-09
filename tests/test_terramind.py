"""The TerraMind comparison row behind the zoo signature (needs the optional extra 'fm')."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("terratorch")
import torch  # noqa: E402

from floodfusion.config import load_config  # noqa: E402
from floodfusion.models.terramind import TerraMindSegmenter  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "comparisons" / "terramind_small.yaml"


@pytest.fixture(scope="module")
def model() -> TerraMindSegmenter:
    torch.manual_seed(0)
    return TerraMindSegmenter(load_config(CONFIG).model, pretrained=False).eval()  # no weight download


@pytest.mark.slow
def test_frozen_encoder_and_trainable_head(model) -> None:
    assert not any(p.requires_grad for p in model.model.encoder.parameters())
    assert any(p.requires_grad for p in model.model.decoder.parameters()) and model.sar_scale.requires_grad


@pytest.mark.slow
def test_mixed_optical_availability_routes_each_sample(model) -> None:
    sar, opt = torch.randn(3, 5, 64, 64), torch.rand(3, 4, 64, 64)
    available = torch.tensor([True, False, True])
    with torch.inference_mode():
        mixed, gate = model(sar, None, opt, None, available)
        alone, _ = model(sar[1:2], None, None, None, None)              # the same chip, SAR only
        both, _ = model(sar[0:1], None, opt[0:1], None, torch.tensor([True]))
    assert mixed.shape == (3, 1, 64, 64) and gate is None
    torch.testing.assert_close(mixed[1:2], alone)                        # never fed zeros as optical
    torch.testing.assert_close(mixed[0:1], both)
