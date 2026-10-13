"""The v1.2 add-on adoption check on whole CV test stacks (offline: encoded stacks, fake predictors)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from floodfusion import cv_addons as ca
from floodfusion.config import ExperimentConfig, ValidityFlag
from floodfusion.data.encoding import encode_stack
from floodfusion.data.records import ChipRecord, write_manifest

CFG = ExperimentConfig()
N = 16


def _raw(label: np.ndarray, hand: np.ndarray) -> np.ndarray:
    vv = np.full((N, N), -15.0)
    vv[0, 0] = np.nan                                                    # SAR nodata: not scored
    validity = np.zeros((N, N), np.uint8)
    validity[0, 1] = ValidityFlag.LAYOVER                                # flagged: not scored
    return encode_stack(vv_post_db=vv, vh_post_db=vv - 7, vv_pre_db=vv, vh_pre_db=vv - 7, reflectance=None,
                        slope_deg=np.ones((N, N)), hand_m=hand, local_incidence_deg=np.full((N, N), 35.0),
                        label=label, validity=validity, optical_quality=None, landcover=None, grid_cfg=CFG.grid,
                        norm=CFG.norm)


def test_scored_pixels_are_the_chip_metrics_own() -> None:
    label = np.zeros((N, N))
    label[:, : N // 2] = 1.0
    label[0, 2] = np.nan                                                 # unlabelled: not scored
    hand = np.full((N, N), 3.0)
    hand[:, -1] = 40.0
    scored, flood, hand_m = ca.scene_layers(_raw(label, hand), CFG.grid, CFG.norm)
    assert not scored[0, 0] and not scored[0, 1] and not scored[0, 2]
    assert scored.sum() == N * N - 3
    assert flood[5, 0] and not flood[5, -2] and not flood[0, 2]
    assert hand_m[3, -1] == pytest.approx(40.0, rel=1e-3) and hand_m[3, 0] == pytest.approx(3.0, rel=1e-3)


def _entry(prob, label, hand, group):
    return {"group": group, **ca.stack_entry(prob, np.ones_like(label), label, hand)}


def _flood_left(prob_flood: float, prob_dry: float, hand_right: float):
    label = np.zeros((N, N), bool)
    label[:, : N // 2] = True
    prob = np.where(label, prob_flood, prob_dry)
    hand = np.where(label, 2.0, hand_right)
    return prob, label, hand


def test_an_add_on_needs_both_the_mean_gain_and_most_groups() -> None:
    clean = _flood_left(0.9, 0.1, hand_right=2.0)                       # perfect already, low terrain
    alarms_on_hills = _flood_left(0.9, 0.8, hand_right=40.0)            # false alarms 40 m above the drainage
    scores = {"runs": {"r": {"tau": 0.5, "pi0": 0.5, "stacks": {
        "a": _entry(*clean, "A"), "b": _entry(*alarms_on_hills, "B"), "c": _entry(*clean, "C")}}}}
    result = ca.adoption(scores)
    assert result["per_group"]["B"]["plain"] == pytest.approx(0.5) and result["per_group"]["B"]["hand"] == 1.0
    assert result["hand"]["gain"] == pytest.approx(0.5 / 3) and result["hand"]["groups_helped"] == 1
    assert not result["hand"]["adopted"]                                 # one group of three: not a majority
    scores["runs"]["r"]["stacks"]["c"] = _entry(*alarms_on_hills, "C")
    assert ca.adoption(scores)["hand"]["adopted"]                        # two of three


def test_self_calibration_moves_the_threshold_to_the_scenes_own_prior() -> None:
    timid = _flood_left(0.4, 0.02, hand_right=2.0)                      # half the scene floods; scores too low
    scores = {"runs": {"r": {"tau": 0.5, "pi0": 0.05, "stacks": {"a": _entry(*timid, "A")}}}}
    result = ca.adoption(scores)
    assert result["per_group"]["A"]["plain"] == 0.0
    assert result["per_group"]["A"]["selfcal"] == 1.0 and result["selfcal"]["adopted"]
    assert result["macro_iou"]["selfcal+hand"] == 1.0


def test_every_test_stack_of_every_fold_is_predicted_whole(tmp_path: Path) -> None:
    label = np.zeros((N, N))
    label[:, : N // 2] = 1.0
    records = []
    for name, group in (("S1", "G1"), ("S2", "G2"), ("S3", "G3")):
        d = tmp_path / "events" / name
        d.mkdir(parents=True)
        with rasterio.open(d / "stack.tif", "w", driver="GTiff", dtype="int16", count=16, height=N, width=N,
                           crs=CRS.from_epsg(32633), transform=from_origin(0, 160, 10, 10)) as dst:
            dst.write(_raw(label, np.full((N, N), 2.0)))
        records.append(ChipRecord(chip_id=f"{name}_c0", basin_id=group, event_id=name, raster_path=d / "stack.tif",
                                  window=(0, 0, N, N), positive_fraction=0.5, valid_fraction=1.0,
                                  optical_available=False, relative_orbit=1, orbit_state="ascending"))
    manifest = tmp_path / "chips" / "manifest.parquet"
    write_manifest(records, manifest)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_chips": ["S1_c0"], "val_chips": [], "test_chips": ["S2_c0", "S3_c0"]}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"split_file": str(split)}))
    (run / "eval_test.json").write_text(json.dumps({"tau_star": 0.5, "calibration": {"pi_deploy": 0.3}}))
    seen = []

    def predictor(run_dir):
        def predict(stack):
            seen.append(stack.parent.name)
            return np.where(label > 0.5, 0.9, 0.1).astype(np.float32)
        return predict

    scores = ca.score_runs([run], manifest, predictor=predictor)
    assert seen == ["S2", "S3"]                                          # test stacks only
    stacks = scores["runs"][str(run)]["stacks"]
    assert {k: v["group"] for k, v in stacks.items()} == {"S2": "G2", "S3": "G3"}
    assert sum(stacks["S2"]["plain"]["pos"]) == N * N // 2 - 2          # (0, 0) SAR nodata, (0, 1) layover
    out, report = tmp_path / "scores.json.gz", tmp_path / "addons.json"
    import gzip
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(scores, f)
    assert ca.main(["report", "--scores", str(out), "--out", str(report)]) == 0
    assert json.loads(report.read_text())["macro_iou"]["plain"] == pytest.approx(1.0)


def test_the_cli_scores_with_each_runs_own_model(tmp_path: Path, monkeypatch) -> None:
    from floodfusion import case_study, train

    loaded = []

    def fake_learned(run_dir, device):
        loaded.append(Path(run_dir).name)
        return case_study._Model("fake", lambda stack: np.zeros((N, N), np.float32), 0.5)

    monkeypatch.setattr(case_study, "_learned", fake_learned)
    monkeypatch.setattr(train, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(ca, "score_runs", lambda runs, manifest, *, predictor: (
        [predictor(r) for r in runs], {"hand_max_m": ca.HAND_MAX_M, "runs": {}})[1])
    out = tmp_path / "s.json.gz"
    runs = [tmp_path / "fold0", tmp_path / "fold1"]
    assert ca.main(["score", "--runs", *map(str, runs), "--manifest", "m.parquet", "--out", str(out)]) == 0
    assert loaded == ["fold0", "fold1"] and out.is_file()
