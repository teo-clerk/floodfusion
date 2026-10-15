"""The blind test's frozen spec, written from the pre-registered choices (offline: fake run folders)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("freeze_spec",
                                               Path(__file__).parents[1] / "scripts" / "v12_freeze_spec.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def _finished(root: Path, folder: Path) -> None:
    for s in fs.SEEDS:
        run = root / folder / f"seed{s}"
        run.mkdir(parents=True)
        (run / "best.pt").write_bytes(b"w")
        (run / "eval_test.json").write_text("{}")


@pytest.fixture
def runs(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    for folder in (fs.FINAL_ROOT / "row21_fusion_context_seasonal", fs.FINAL_ROOT / "cmp_terramind_base_fair",
                   fs.V11_BEST):
        _finished(tmp_path, folder)
    return tmp_path


def test_the_spec_names_the_selected_models_and_the_three_tests(runs: Path) -> None:
    from floodfusion.blind_analysis import load_blind_tests
    from floodfusion.cems_scores import load_spec

    out = runs / "prospective_v12.yaml"
    (runs / "sel.json").write_text(json.dumps({"chosen": "row21_fusion_context_seasonal", "add_ons": ["selfcal"]}))
    (runs / "probe.json").write_text(json.dumps({"chosen": "terramind_base_fair"}))
    assert fs.main(["--selection", str(runs / "sel.json"), "--probe", str(runs / "probe.json"), "--out", str(out)]) == 0
    spec, tests = load_spec(out), load_blind_tests(out)
    assert [m.key for m in spec["models"]] == ["ours", "tm_fair", "v11", "gfm"]
    assert spec["models"][1].runs[0].endswith("cmp_terramind_base_fair/seed42") and spec["models"][3].kind == "gfm"
    assert [t["id"] for t in tests] == ["P5", "P6", "P7"]
    assert all(t["metric_b"] == "iou_selfcal" for t in tests) and tests[0]["also_ap"] and tests[1]["gfm_covered"]


def test_an_unimplemented_add_on_missing_runs_or_no_opponent_stop_the_freeze(runs: Path) -> None:
    probe = {"chosen": "terramind_base_fair"}
    with pytest.raises(NotImplementedError, match="HAND gate"):
        fs.build_spec({"chosen": "row21_fusion_context_seasonal", "add_ons": ["hand"]}, probe)
    with pytest.raises(FileNotFoundError, match="row09_fusion_crossattn"):
        fs.build_spec({"chosen": "row09_fusion_crossattn", "add_ons": []}, probe)
    with pytest.raises(ValueError, match="no fair TerraMind"):
        fs.build_spec({"chosen": "row21_fusion_context_seasonal", "add_ons": []}, {"chosen": None})
    plain = fs.build_spec({"chosen": "row21_fusion_context_seasonal", "add_ons": []}, probe)
    assert {t["metric_b"] for t in plain["blind_tests"]} == {"iou_tau"}
