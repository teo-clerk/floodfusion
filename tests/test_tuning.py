"""v1.2's equal tuning budget: inner split, rate choice, overlay hand-over (offline; training is faked)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from floodfusion import tuning
from floodfusion.config import ExperimentConfig, load_config
from floodfusion.data.encoding import encode_stack
from floodfusion.data.records import ChipRecord, load_manifest_meta, write_manifest
from floodfusion.splits import SplitAssignment, freeze_splits, load_splits

CFG = ExperimentConfig()


def _fold0(windows: dict[str, str], val_group: str = "V") -> SplitAssignment:
    train = tuple(c for c, g in windows.items() if g != val_group)
    return SplitAssignment(train_chips=train, val_chips=tuple(c for c, g in windows.items() if g == val_group),
                           test_chips=("t0",), train_basins=tuple(sorted({windows[c] for c in train})),
                           val_basins=(val_group,), test_basins=("T",), buffer_m=0.0,
                           strategy="leave_one_basin_out", manifest_sha256="abc")


def test_the_inner_split_holds_out_whole_groups_of_fold_0s_training_data() -> None:
    windows = {f"g{g}_c{i}": f"G{g}" for g in range(10) for i in range(10)} | {"v0": "V", "v1": "V"}
    inner = tuning.inner_split(_fold0(windows), windows)
    assert set(inner.train_chips) | set(inner.val_chips) == {c for c, g in windows.items() if g != "V"}
    assert not set(inner.train_basins) & set(inner.val_basins)
    assert {windows[c] for c in inner.val_chips} == set(inner.val_basins)
    assert len(inner.val_chips) == 20                                     # two of ten equal groups: 20 %
    assert inner.test_chips == ("v0", "v1") and inner.test_basins == ("V",)   # fold 0's validation, never scored
    assert inner == tuning.inner_split(_fold0(windows), windows)          # seeded
    with pytest.raises(ValueError, match="two training groups"):
        tuning.inner_split(_fold0({"a": "G0", "v": "V"}), {"a": "G0", "v": "V"})


def test_the_best_rate_wins_and_an_exact_tie_goes_to_the_earlier_rate() -> None:
    assert tuning.best_rate({3e-5: 0.50, 1e-4: 0.52, 3e-4: 0.51}) == 1e-4
    assert tuning.best_rate({3e-5: 0.52, 1e-4: 0.50, 3e-4: 0.52}) == 3e-5


def test_the_overlay_sets_the_rate_as_a_number(tmp_path: Path) -> None:
    overlay = tmp_path / "lr.yaml"
    overlay.write_text(tuning.overlay_text(3e-5, "row09_fusion_crossattn"))
    cfg = load_config(Path("configs/row09_fusion_crossattn.yaml"), overlays=[overlay])
    assert isinstance(cfg.train.learning_rate, float) and cfg.train.learning_rate == pytest.approx(3e-5)


def _pool(tmp_path: Path) -> Path:
    """A pool root with one 16 px stack per group (G0..G5 train, V fold-0 validation, T test) and fold 0's split."""
    root, n = tmp_path / "pool", 16
    vv = np.full((n, n), -15.0)
    raw = encode_stack(vv_post_db=vv, vh_post_db=vv - 7, vv_pre_db=vv, vh_pre_db=vv - 7, reflectance=None,
                       slope_deg=np.ones((n, n)), hand_m=np.ones((n, n)), local_incidence_deg=np.full((n, n), 35.0),
                       label=np.zeros((n, n)), validity=np.zeros((n, n), np.uint8), optical_quality=None,
                       landcover=None, grid_cfg=CFG.grid, norm=CFG.norm)
    records = []
    for g in ["G0", "G1", "G2", "G3", "G4", "G5", "V", "T"]:
        d = root / "events" / g
        d.mkdir(parents=True)
        with rasterio.open(d / "stack.tif", "w", driver="GTiff", dtype="int16", count=16, height=n, width=n,
                           crs=CRS.from_epsg(32633), transform=from_origin(0, 160, 10, 10)) as dst:
            dst.write(raw)
        records.append(ChipRecord(chip_id=f"{g}_c0", basin_id=g, event_id=g, raster_path=d / "stack.tif",
                                  window=(0, 0, n, n), positive_fraction=0.0, valid_fraction=1.0,
                                  optical_available=False, relative_orbit=1, orbit_state="ascending", eval_grid=True))
    write_manifest(records, root / "chips" / "manifest.parquet")
    windows = {r.chip_id: r.basin_id for r in records}
    fold0 = _fold0({c: g for c, g in windows.items() if g != "T"})
    fold0 = SplitAssignment(**{**fold0.__dict__, "test_chips": ("T_c0",),
                               "manifest_sha256": load_manifest_meta(root / "chips" / "manifest.parquet")
                               ["manifest_sha256"]})
    freeze_splits(fold0, root / "chips" / "cv" / "kfold__fold0.json")
    return root


def test_each_family_trains_once_per_rate_and_hands_over_its_best(tmp_path: Path, monkeypatch) -> None:
    from floodfusion import train

    root = _pool(tmp_path)
    monkeypatch.chdir(Path(__file__).parents[1])
    calls = []
    score = {"row09_fusion_crossattn": {3e-5: 0.40, 1e-4: 0.45, 3e-4: 0.44},
             "terramind_small_fair": {3e-5: 0.30, 1e-4: 0.35, 3e-4: 0.41}}

    def fake_main(argv):
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        rate = float(next(a for a in argv if a.startswith("train.learning_rate=")).split("=")[1])
        calls.append((run_dir.parent.name, rate))
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"status": "finished",
                                                      "best_val_macro_iou": score[run_dir.parent.name][rate]}))
        return 0

    monkeypatch.setattr(train, "main", fake_main)
    reports, runs = tmp_path / "reports", tmp_path / "runs"
    reports.mkdir()
    argv = ["--data-root", str(root), "--rows", "9", "26", "--overlay", "configs/overlays/kurosiwo.yaml",
            "--runs-root", str(runs), "--reports", str(reports)]
    assert tuning.main(argv) == 0
    assert len(calls) == 6 and {r for _, r in calls} == set(tuning.GRID)
    inner = load_splits(root / "chips" / tuning.SPLIT_FILE)
    assert inner.test_chips == ("V_c0",) and len(inner.val_basins) >= 1
    assert (root / "chips" / tuning.NORM_FILE).is_file()
    assert load_config(Path("configs/row09_fusion_crossattn.yaml"),
                       overlays=[reports / "v12_lr_row09_fusion_crossattn.yaml"]).train.learning_rate == 1e-4
    assert "3.0e-04" in (reports / "v12_lr_terramind_small_fair.yaml").read_text()
    assert json.loads((reports / "v12_tuning.json").read_text())["rows"]["terramind_small_fair"]["chosen"] == 3e-4
    assert tuning.main(argv) == 0 and len(calls) == 6                   # resumable: nothing trained twice
