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


@pytest.mark.slow
def test_clouded_pixels_reach_terramind_as_its_mean_not_as_cloud(model) -> None:
    sar = torch.randn(1, 5, 64, 64)
    clear, cloud = torch.rand(1, 4, 64, 64), torch.ones(1, 4, 64, 64)       # bright cloud tops
    quality = torch.zeros(1, 1, 64, 64)                                    # every pixel clouded
    with torch.inference_mode():
        a, _ = model(sar, None, clear, quality, torch.tensor([True]))
        b, _ = model(sar, None, cloud, quality, torch.tensor([True]))
    torch.testing.assert_close(a, b)                                      # the cloud values never reach it


def test_rebuilding_for_a_checkpoint_does_not_download_weights(monkeypatch) -> None:
    import floodfusion.models.terramind as tm
    from floodfusion.models.zoo import build_model

    seen = []
    monkeypatch.setattr(tm.TerraMindSegmenter, "__init__",
                        lambda self, cfg, *, pretrained=True: seen.append(pretrained) or torch.nn.Module.__init__(self))
    build_model(load_config(CONFIG).model, pretrained=False)
    assert seen == [False]


def test_terramind_gets_the_modality_dropout_its_config_sets() -> None:
    # v1.1 erratum: the set left TerraMind out, so it trained with p = 0 despite modality_dropout_p: 0.5.
    from floodfusion.models.terramind import TERRAMIND_VARIANTS
    from floodfusion.train import OPTICAL_FUSION_ARCHITECTURES

    assert set(TERRAMIND_VARIANTS) <= OPTICAL_FUSION_ARCHITECTURES
    assert load_config(CONFIG).model.modality_dropout_p == 0.5


@pytest.mark.slow
def test_neck_and_decoder_see_the_whole_batch_once(model) -> None:
    # v1.1 erratum: they ran once per optical subset, so BatchNorm trained on sub-batches of 1-3 chips.
    seen: list[int] = []
    handle = model.model.decoder.register_forward_pre_hook(lambda _m, args: seen.append(args[0][0].shape[0]))
    try:
        sar, opt = torch.randn(3, 5, 64, 64), torch.rand(3, 4, 64, 64)
        with torch.inference_mode():
            out, _ = model(sar, None, opt, None, torch.tensor([True, False, True]))
    finally:
        handle.remove()
    assert seen == [3] and out.shape == (3, 1, 64, 64)
