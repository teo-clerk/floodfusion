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


@pytest.mark.slow
@pytest.mark.parametrize("available", [[True, True], [False, False], [True, False]])
def test_wrapper_matches_terratorchs_own_forward_on_an_odd_size(model, available) -> None:
    # 70 x 50 is not a multiple of the 16 px patch: padding, rescale and centre crop all act.
    from floodfusion.models.terramind import _REFLECTANCE_TO_DN

    torch.manual_seed(1)
    sar, opt = torch.randn(2, 5, 70, 50), torch.rand(2, 4, 70, 50)
    mask = torch.tensor(available)
    with torch.inference_mode():
        ours, _ = model(sar, None, opt, None, mask)
        s1 = sar[:, :2] * model.sar_scale + model.sar_shift
        s2 = (opt[:, :3] * _REFLECTANCE_TO_DN - model.s2_mean) / model.s2_std
        for i, has_optical in enumerate(available):
            inputs = {"S2L2A": s2[i:i + 1], "S1RTC": s1[i:i + 1]} if has_optical else {"S1RTC": s1[i:i + 1]}
            upstream = model.model(inputs).output
            torch.testing.assert_close(ours[i:i + 1], upstream.reshape(1, 1, 70, 50), rtol=1e-4, atol=1e-5)


FAIR = Path(__file__).resolve().parents[1] / "configs" / "comparisons" / "terramind_small_fair.yaml"


@pytest.fixture(scope="module")
def fair() -> TerraMindSegmenter:
    torch.manual_seed(0)
    return TerraMindSegmenter(load_config(FAIR).model, pretrained=False).eval()


@pytest.mark.slow
def test_fair_variant_trains_only_lora_and_the_decoder(fair) -> None:
    from floodfusion.models.lora import LoRALinear

    enc = fair.model.encoder
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_a", "lora_b")) for n in trainable)
    assert sum(isinstance(m, LoRALinear) for m in enc.modules()) == 2 * 12          # qkv + proj in 12 blocks
    assert enc.out_channels == [2 * c for c in enc.encoder.out_channels]          # post and pre tokens side by side


@pytest.mark.slow
def test_fair_variant_reads_the_pre_event_radar(fair) -> None:
    sar = torch.randn(2, 5, 64, 64)
    changed = sar.clone()
    changed[:, 2:4] += 3.0                                                         # a different pre-event scene
    with torch.inference_mode():
        a, _ = fair(sar, None, None, None, None)
        b, _ = fair(changed, None, None, None, None)
        c, _ = fair(sar[:, [0, 1, 2, 3, 4]], None, None, None, None)
    assert not torch.allclose(a, b)
    torch.testing.assert_close(a, c)


@pytest.mark.slow
def test_fair_variant_routes_mixed_optical_and_keeps_one_decoder_pass(fair) -> None:
    seen: list[int] = []
    handle = fair.model.decoder.register_forward_pre_hook(lambda _m, args: seen.append(args[0][0].shape[0]))
    try:
        sar, opt = torch.randn(3, 5, 64, 64), torch.rand(3, 4, 64, 64)
        with torch.inference_mode():
            mixed, _ = fair(sar, None, opt, None, torch.tensor([True, False, True]))
            alone, _ = fair(sar[1:2], None, None, None, None)
    finally:
        handle.remove()
    assert seen[0] == 3
    torch.testing.assert_close(mixed[1:2], alone, rtol=1e-4, atol=1e-5)


def test_every_fair_variant_is_registered_for_dropout_and_its_sar_only_sibling() -> None:
    from floodfusion.case_study import SAR_ONLY_SIBLINGS
    from floodfusion.config import ModelConfig
    from floodfusion.crossval import EXTRA_ROWS, SAR_ONLY_EVALUATED
    from floodfusion.models.terramind import FAIR_VARIANTS
    from floodfusion.models.zoo import _PRETRAINED
    from floodfusion.train import OPTICAL_FUSION_ARCHITECTURES

    for row, arch in zip((24, 25, 26), ("terramind_base_fair", "terramind_base_fair_frozen", "terramind_small_fair"),
                         strict=True):
        assert arch in FAIR_VARIANTS and arch in ModelConfig.ARCHITECTURES and arch in _PRETRAINED
        assert arch in OPTICAL_FUSION_ARCHITECTURES and EXTRA_ROWS[row] == f"comparisons/{arch}"
        assert SAR_ONLY_EVALUATED[row](f"cmp_{arch}") == f"cmp_{arch}__eval_sar_only"
        assert SAR_ONLY_SIBLINGS[f"cmp_{arch}"] == f"cmp_{arch}__eval_sar_only"
        assert load_config(CONFIG.with_name(f"{arch}.yaml")).model.architecture == arch
