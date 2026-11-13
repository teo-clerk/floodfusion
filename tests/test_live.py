"""floodfusion Live (``floodfusion.live``): a flood map of any place and date, offline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from floodfusion import live
from floodfusion.data.stac import NoMatchingPairError

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The frozen model, from its run folder alone
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_run(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from floodfusion.config import config_sha256, config_to_dict, load_config
    from floodfusion.models.zoo import build_model
    from floodfusion.runs import write_run_record

    overlay = tmp_path / "small.yaml"
    overlay.write_text("model:\n  encoder_channels: [16, 32, 64, 128]\n  decoder_channels: [64, 32, 16, 8]\n"
                       "  attention_heads: 2\n  attention_head_dim: 16\n  norm_groups: 4\n")
    cfg = load_config(ROOT / "configs" / "row09_fusion_crossattn.yaml", overlays=[overlay])
    run = tmp_path / "frozen" / "seed42"
    sha = config_sha256(config_to_dict(cfg))
    write_run_record(run, {"status": "finished", "config": config_to_dict(cfg), "config_sha256": sha})
    torch.manual_seed(0)
    model = build_model(cfg.model)
    torch.save({"model": model.state_dict(), "epoch": 3, "val_macro_iou": 0.5, "config_sha256": sha}, run / "best.pt")
    (run / "eval_test.json").write_text(json.dumps({"calibration": {"temperature": 1.5, "offset": 0.1},
                                                    "tau_star": 0.25}))
    return run, model.state_dict()


def test_the_predictor_loads_the_frozen_network_from_its_run_folder_alone(tiny_run) -> None:
    torch = pytest.importorskip("torch")
    run, state = tiny_run
    predictor = live.load_predictor(run, device=torch.device("cpu"))
    assert all(torch.equal(predictor.model.state_dict()[k], v) for k, v in state.items())
    assert predictor.calibration == {"temperature": 1.5, "offset": 0.1} and predictor.tau == 0.25
    assert predictor.sar_calibration is None and not predictor.model.training


def test_the_predictor_refuses_a_checkpoint_of_another_configuration(tiny_run) -> None:
    torch = pytest.importorskip("torch")
    run, _ = tiny_run
    state = torch.load(run / "best.pt", weights_only=True)
    torch.save({**state, "config_sha256": "another"}, run / "best.pt")
    with pytest.raises(ValueError, match="another configuration"):
        live.load_predictor(run, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# One map: tiles, sidecars, prediction, package
# ---------------------------------------------------------------------------
class _FakePredictor:
    tau = 0.25
    architecture = "fusion_crossattn_ctx_seasonal"

    def __init__(self) -> None:
        self.stacks: list[Path] = []

    def predict(self, stack: Path, device) -> tuple[np.ndarray, np.ndarray]:
        self.stacks.append(stack)
        return np.full((4, 4), 0.9, dtype=np.float32), np.zeros((4, 4), dtype=bool)


def test_map_area_maps_every_tile_with_a_pass_and_records_the_others(tmp_path: Path, monkeypatch) -> None:
    ingested, sidecars, packaged = [], [], []

    def fake_ingest(event, out_root, cfg, *, sources=None):
        if event.event_id.endswith("t00_01"):
            raise NoMatchingPairError("no same-orbit pre-event scene")
        assert event.label_path is None and event.flood_start == event.flood_end == date(2023, 5, 17)
        d = Path(out_root) / "events" / event.event_id
        d.mkdir(parents=True)
        (d / "stack.tif").write_bytes(b"")
        (d / "event.json").write_text(json.dumps({"post_scene_time": "2023-05-17T05:20:00Z"}))
        ingested.append(event.event_id)
        return {"n_chips": 1}

    monkeypatch.setattr(live, "ingest_event", fake_ingest)
    monkeypatch.setattr(live, "_build_sidecars", lambda d, names, client: sidecars.append((d.name, names)))
    monkeypatch.setattr(live, "_package_tile", lambda stack, prob, used, predictor, out: packaged.append(out.name)
                        or {"flood_km2": 0.16, "valid_km2": 0.16})
    predictor = _FakePredictor()
    summary = live.map_area((11.8, 44.15, 12.2, 44.3), date(2023, 5, 17), tmp_path, predictor=predictor,
                            name="forli", cfg=None, sources=live.Sources(client=None), device="cpu")
    assert ingested == ["forli_t00_00"]
    assert sidecars == [("forli_t00_00", ("seasonal",))]                 # row 21 reads the seasonal reference
    assert packaged == ["t00_00"] and predictor.stacks == [tmp_path / "events" / "forli_t00_00" / "stack.tif"]
    assert summary["tiles"]["t00_00"]["flood_km2"] == 0.16
    assert "no same-orbit" in summary["skipped"]["t00_01"]
    assert json.loads((tmp_path / "summary.json").read_text())["date"] == "2023-05-17"


def test_map_area_refuses_an_area_too_large_for_one_map(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tiles"):
        live.map_area((0.0, 0.0, 5.0, 5.0), date(2023, 5, 17), tmp_path, predictor=_FakePredictor(), name="x",
                      cfg=None, sources=live.Sources(client=None), device="cpu")


@pytest.mark.parametrize("bbox", [(12.0, 44.0, 11.0, 45.0), (11.0, 95.0, 12.0, 96.0), (190.0, 0.0, 191.0, 1.0)])
def test_a_box_must_be_west_south_east_north_in_degrees(bbox) -> None:
    with pytest.raises(ValueError, match="bbox"):
        live.check_bbox(bbox)


# ---------------------------------------------------------------------------
# Flood polygons for the live page
# ---------------------------------------------------------------------------
def test_flood_polygons_are_the_mapped_pixels_in_lon_lat() -> None:
    from rasterio.transform import from_origin

    prob = np.zeros((20, 20), dtype=np.float32)
    prob[5:15, 5:15] = 0.9                        # 10 x 10 pixels of 10 m: 0.01 km^2
    prob[0, 0] = np.nan                           # no data never maps
    collection = live.flood_polygons(prob, tau=0.25, transform=from_origin(500_000, 5_000_000, 10, 10), epsg=32632)
    [feature] = collection["features"]
    lon, lat = np.array(feature["geometry"]["coordinates"][0]).T
    assert 9.0005 < lon.min() < lon.max() < 9.0020 and 45.152 < lat.min() < lat.max() < 45.154
    assert feature["properties"]["area_km2"] == pytest.approx(0.01, rel=1e-6)


def test_map_area_can_pin_the_pass(tmp_path: Path, monkeypatch) -> None:
    """The watcher maps one named pass; a replay of a benchmark tile pins its pass and reference."""
    events = []

    def fake_ingest(event, out_root, cfg, *, sources=None):
        events.append(event)
        raise NoMatchingPairError("stop here")

    monkeypatch.setattr(live, "ingest_event", fake_ingest)
    live.map_area((11.8, 44.15, 12.0, 44.3), date(2023, 5, 17), tmp_path, predictor=_FakePredictor(), name="x",
                  cfg=None, sources=live.Sources(client=None), device="cpu", relative_orbit=117,
                  orbit_state="ascending", reference_before=date(2023, 5, 1))
    [event] = events
    assert (event.relative_orbit, event.orbit_state, event.reference_before) == (117, "ascending", date(2023, 5, 1))


def test_flood_polygons_are_simplified_within_a_pixel_and_keep_their_area() -> None:
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    prob = (rng.random((200, 200)) > 0.35).astype(np.float32)     # a ragged, pixel-edged flood
    prob = np.where(np.add.outer(np.arange(200), np.arange(200)) < 220, prob, 0.0)
    transform = from_origin(500_000, 5_000_000, 10, 10)
    collection = live.flood_polygons(prob, tau=0.25, transform=transform, epsg=32632)
    vertices = sum(len(ring) for f in collection["features"] for ring in f["geometry"]["coordinates"])
    pixels = int((prob > 0.25).sum())
    assert sum(f["properties"]["area_km2"] for f in collection["features"]) == pytest.approx(pixels * 1e-4, rel=1e-6)
    assert vertices < 2 * pixels                                     # a pixel-edged outline has about 4 per pixel


def test_every_command_line_entry_point_resolves() -> None:
    import importlib
    import tomllib

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert {"floodfusion-live", "floodfusion-weights"} <= set(scripts)
    for target in scripts.values():
        module, _, name = target.partition(":")
        assert callable(getattr(importlib.import_module(module), name)), target
