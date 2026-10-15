"""v1.2 context: sidecars reach the model through the dataset, and models without them ignore nothing they used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from floodfusion.config import DataConfig, GridConfig, NormalizationStats, load_config  # noqa: E402
from floodfusion.data.chipping import AnalysisGrid, write_event_stack  # noqa: E402
from floodfusion.data.dataset import ChipRecord, FloodChipDataset  # noqa: E402
from floodfusion.data.synthetic.chip import synthesize_chip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIZE = 32
GRID = AnalysisGrid(epsg=32634, transform=(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_650_000.0), width=SIZE, height=SIZE)


def _record(path: Path, chip_id: str = "c0") -> ChipRecord:
    return ChipRecord(chip_id=chip_id, basin_id="b0", event_id="e0", raster_path=path, window=(0, 0, SIZE, SIZE),
                      positive_fraction=0.1, valid_fraction=1.0, optical_available=False, relative_orbit=1,
                      orbit_state="ascending")


def _stack(tmp_path: Path, *, seasonal: bool) -> Path:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    d = tmp_path / "events" / "e0"
    d.mkdir(parents=True)
    raw = synthesize_chip(_record(d / "stack.tif"), GridConfig(), NormalizationStats(), height=SIZE, width=SIZE)
    path = write_event_stack(d / "stack.tif", raw, GRID)
    if seasonal:
        data = np.stack([np.full((SIZE, SIZE), -14.0, np.float32), np.full((SIZE, SIZE), -21.0, np.float32)])
        data[:, :4, :4] = np.nan
        with rasterio.open(d / "seasonal.tif", "w", driver="GTiff", dtype="float32", count=2, height=SIZE, width=SIZE,
                           crs=CRS.from_epsg(GRID.epsg), transform=Affine(*GRID.transform), nodata=float("nan")) as dst:
            dst.write(data)
    return path


def _ds(path: Path, sidecars=("seasonal",), training=False) -> FloodChipDataset:
    return FloodChipDataset([_record(path)], NormalizationStats(), DataConfig(chip_size=SIZE, chip_source="raster"),
                            GridConfig(), training=training, sidecars=sidecars)


def test_a_sidecar_on_disk_reaches_the_sample_normalised_and_available(tmp_path: Path) -> None:
    sample = _ds(_stack(tmp_path, seasonal=True))[0]
    norm = NormalizationStats()
    assert sample["context"].shape == (3, SIZE, SIZE) and sample["context_available"].tolist() == [True]
    assert sample["context"][0, 10, 10] == pytest.approx((-14.0 - norm.sar_mean_db[2]) / norm.sar_std_db[2], abs=1e-5)
    assert (sample["context"][:, :4, :4] == 0).all() and torch.isfinite(sample["context"]).all()   # NaN -> 0


def test_a_missing_sidecar_is_zeros_marked_unavailable_and_models_without_sidecars_get_no_context(tmp_path) -> None:
    path = _stack(tmp_path, seasonal=False)
    sample = _ds(path)[0]
    assert sample["context_available"].tolist() == [False] and (sample["context"] == 0).all()
    assert "context" not in _ds(path, sidecars=())[0]
    with pytest.raises(ValueError, match="unknown sidecar"):
        _ds(path, sidecars=("ndvi",))


def test_augmentation_moves_the_context_with_the_image(tmp_path: Path) -> None:
    torch.manual_seed(3)
    sample = _ds(_stack(tmp_path, seasonal=True), training=True)[0]
    unlabelled = sample["context"][0] == 0                               # the 4 x 4 NaN corner, wherever it went
    assert int(unlabelled.sum()) == 16
    rows, cols = unlabelled.nonzero(as_tuple=True)
    assert rows.max() - rows.min() == 3 and cols.max() - cols.min() == 3


def _model(arch_config: str):
    from floodfusion.models.zoo import build_model

    torch.manual_seed(0)
    return build_model(load_config(ROOT / "configs" / arch_config).model, pretrained=False)


def test_context_model_reads_context_only_where_available() -> None:
    model = _model("variants/row20_fusion_context.yaml").eval()
    assert model.sidecars == ("seasonal", "aef") and model.sar_in_channels == 5 + 3 + 3 + 8 + 2
    sar, terrain = torch.randn(2, 5, 64, 64), torch.randn(2, 3, 64, 64)
    ctx = torch.randn(2, 11, 64, 64)
    avail = torch.tensor([[True, False], [False, False]])
    with torch.inference_mode():
        a, _ = model(sar, terrain, context=ctx, context_available=avail)
        b, _ = model(sar, terrain, context=ctx * 5, context_available=avail)
        c, _ = model(sar, terrain)                                        # no context at all: all missing
    assert not torch.allclose(a[0], b[0])                                # sample 0 reads its seasonal layer
    torch.testing.assert_close(a[1], b[1])                               # sample 1 has none: values never read
    torch.testing.assert_close(a[1], c[1])
    with pytest.raises(Exception, match="do not match"):
        model(sar, terrain, context=ctx[:, :5], context_available=avail)


def test_context_dropout_trains_the_missing_embedding() -> None:
    from floodfusion.models.segmenter import CONTEXT_DROPOUT_P

    model = _model("variants/row21_fusion_context_seasonal.yaml").train()
    sar, terrain = torch.randn(8, 5, 32, 32), torch.randn(8, 3, 32, 32)
    torch.manual_seed(1)
    logits, _ = model(sar, terrain, context=torch.randn(8, 3, 32, 32), context_available=torch.ones(8, 1, dtype=bool))
    logits.mean().backward()
    assert 0 < CONTEXT_DROPOUT_P < 1 and model.missing_context[0].grad is not None


@pytest.mark.slow
def test_fair_terramind_reads_its_dem() -> None:
    pytest.importorskip("terratorch")
    model = _model("comparisons/terramind_small_fair.yaml").eval()
    sar = torch.randn(2, 5, 64, 64)
    dem = torch.full((2, 1, 64, 64), 300.0)
    on = torch.ones(2, 1, dtype=bool)
    with torch.inference_mode():
        a, _ = model(sar, None, context=dem, context_available=on)
        b, _ = model(sar, None, context=dem + 500.0, context_available=on)
        c, _ = model(sar, None, context=dem + 500.0, context_available=~on)
        d, _ = model(sar, None)
    assert not torch.allclose(a, b)
    torch.testing.assert_close(c, d)                                     # unavailable DEM = no DEM (its mean)
