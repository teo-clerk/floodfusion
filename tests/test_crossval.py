"""Event-level cross-validation: fold construction, the published split, and the CV report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floodfusion.crossval import Fold, leave_one_group_out, official_splits
from floodfusion.data.records import ChipRecord
from floodfusion.splits import STRATEGY_LOBO, STRATEGY_PUBLISHED, freeze_splits, load_splits


def _record(chip_id: str, group: str, *, eval_grid: bool = True, source: str = "") -> ChipRecord:
    return ChipRecord(chip_id, group, f"S1F11_{source or chip_id}", Path("x.tif"), (0, 0, 256, 256), 0.1, 1.0,
                      True, 1, "ascending", eval_grid=eval_grid)


def _records(groups=("S1F11_A", "S1F11_B", "S1F11_C", "S1F11_D"), per_group: int = 4) -> list[ChipRecord]:
    out = []
    for g in groups:
        for i in range(per_group):
            out.append(_record(f"{g}_{i}", g, eval_grid=i % 2 == 0))
    return out


class TestLeaveOneGroupOut:
    def test_each_group_is_tested_once_and_validated_by_the_next(self) -> None:
        folds = leave_one_group_out(_records(), manifest_sha256="m")
        assert list(folds) == ["S1F11_A", "S1F11_B", "S1F11_C", "S1F11_D"]
        for i, (group, split) in enumerate(folds.items()):
            nxt = list(folds)[(i + 1) % 4]
            assert split.test_basins == (group,) and split.val_basins == (nxt,)
            assert set(split.train_basins) == set(folds) - {group, nxt}
            assert all(c.startswith(group) for c in split.test_chips) and len(split.test_chips) == 2  # eval grid only
            assert len(split.train_chips) == 8                                                        # every window
            assert split.strategy == STRATEGY_LOBO and split.manifest_sha256 == "m"

    def test_folds_round_trip_through_the_validating_loader(self, tmp_path) -> None:
        for group, split in leave_one_group_out(_records()).items():
            freeze_splits(split, tmp_path / f"{group}.json")
            assert load_splits(tmp_path / f"{group}.json") == split

    def test_too_few_groups_is_refused(self) -> None:
        with pytest.raises(ValueError, match=">= 3 groups"):
            leave_one_group_out(_records(groups=("S1F11_A", "S1F11_B")))


class TestPublishedSplit:
    def test_mapping_from_source_chips_and_the_published_strategy(self, tmp_path) -> None:
        records = [_record(f"w{i}", "S1F11_Ghana" if i < 6 else "S1F11_Bolivia", eval_grid=i % 2 == 0,
                           source=f"src{i // 2}") for i in range(8)]
        chip_split = {"src0": "train", "src1": "val", "src2": "test", "src3": "bolivia"}
        splits = official_splits(records, chip_split, source_of=lambda r: r.event_id.removeprefix("S1F11_"))
        official, bolivia = splits["official"], splits["official_bolivia"]
        assert official.train_chips == ("w0", "w1") and official.val_chips == ("w2",) and official.test_chips == ("w4",)
        assert bolivia.test_chips == ("w6",) and bolivia.train_chips == official.train_chips
        assert official.strategy == STRATEGY_PUBLISHED and official.test_basins == ("S1F11_Ghana",)
        freeze_splits(official, tmp_path / "o.json")
        with pytest.warns(UserWarning, match="published"):  # groups overlap by design: loaded, loudly
            assert load_splits(tmp_path / "o.json") == official

    def test_published_strategy_is_a_valid_config_choice(self) -> None:
        from floodfusion.config import DataConfig

        assert DataConfig(split_strategy="published").split_strategy == "published"
        with pytest.raises(ValueError, match="split_strategy"):
            DataConfig(split_strategy="whatever")

    def test_fold_overrides(self) -> None:
        fold = Fold("loeo", "S1F11_A", "cv/loeo__S1F11_A.json", "cv/loeo__S1F11_A.norm_stats.json", STRATEGY_LOBO)
        assert fold.overrides() == ["data.split_file_name=cv/loeo__S1F11_A.json",
                                    "data.norm_stats_file=cv/loeo__S1F11_A.norm_stats.json",
                                    "data.split_strategy=leave_one_basin_out"]


# ---------------------------------------------------------------------------
# The report on fake run directories
# ---------------------------------------------------------------------------
EVENTS = [f"S1F11_E{i}" for i in range(9)]


def _cm_json(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "tn": 1000, "iou": tp / (tp + fp + fn)}


def _write_run(d: Path, row: int, seed: int | None, per_basin: dict, per_chip: dict | None = None,
               chip_basin: dict | None = None, conformal: dict | None = None) -> None:
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "name": d.parent.name, "ablation_row": row, "status": "finished", "git_sha": "abc", "git_dirty": False,
        "seed": seed, "trained": seed is not None, "train_seconds": 60.0, "peak_vram_bytes": 2**30,
        "config_sha256_without_seed": "c"}))
    ious = [v["iou"] for v in per_basin.values()]
    (d / "eval_test.json").write_text(json.dumps({
        "name": d.parent.name, "ablation_row": row, "macro_iou": float(np.mean(ious)), "manifest_sha256": "m",
        "test_chips_sha256": "t", "baseline": seed is None, "per_basin": per_basin,
        "per_chip_tau_star": per_chip or {}, "per_chip_basin": chip_basin or {},
        **({"conformal": conformal} if conformal else {})}))


@pytest.fixture()
def cv_runs(tmp_path, monkeypatch):
    import floodfusion.utils.env as env

    monkeypatch.setattr(env, "git_state", lambda *a, **k: {"git_sha": "abc", "git_branch": "x", "git_dirty": False})
    root = tmp_path / "runs"
    for k, event in enumerate(EVENTS):
        fold = root / "loeo" / event
        chips = {f"{event}_c{j}": [40 + k, 10, 10, 500] for j in range(2)}   # LOEO model, per test chip
        basin_of = dict.fromkeys(chips, event)
        _write_run(fold / "row02_baseline_adaptive" / "baseline", 2, None, {event: _cm_json(50, 30, 30)})
        _write_run(fold / "row04_unet_sar_only" / "seed42", 4, 42, {event: _cm_json(60 + k, 20, 20)})
        _write_run(fold / "row09_fusion_crossattn" / "seed42", 9, 42, {event: _cm_json(80 + k, 20, 20)},
                   chips, basin_of, conformal={"alpha": 0.1, "test_mean_chip_fnr_by_basin": {event: 0.05 + 0.01 * k}})
    # The official (within-event) model, on the same chips: better in every event.
    chips = {f"{e}_c{j}": [70, 5, 5, 500] for e in EVENTS for j in range(2)}
    _write_run(root / "official" / "official" / "row09_fusion_crossattn" / "seed42", 9, 42,
               {e: _cm_json(140, 10, 10) for e in EVENTS}, chips, {c: c.rsplit("_", 1)[0] for c in chips})
    return root


class TestCvReport:
    def test_folds_merge_into_one_row_per_seed(self, cv_runs) -> None:
        from floodfusion.cv_report import _collect, merge_folds

        folds, problems = _collect(cv_runs / "loeo", allow_dirty=False)
        merged = merge_folds(folds)
        assert not problems and set(merged) == {"2", "4", "9"}
        (row9,) = merged["9"]
        assert sorted(row9["eval"]["per_basin"]) == EVENTS and len(row9["run"]["folds"]) == 9

    def test_an_event_tested_twice_is_refused(self, cv_runs) -> None:
        from floodfusion.cv_report import _collect, merge_folds

        folds, _ = _collect(cv_runs / "loeo", allow_dirty=False)
        folds["duplicate"] = folds[EVENTS[0]]
        with pytest.raises(ValueError, match="more than one fold"):
            merge_folds(folds)

    def test_table_has_verdicts_and_the_same_chip_leakage_gap(self, cv_runs, tmp_path) -> None:
        from floodfusion.cv_report import build_cv_table

        text = build_cv_table(cv_runs, out=tmp_path / "t.md", n_boot=500)
        result = json.loads((tmp_path / "t.json").read_text())
        gap = result["gaps"]["9-4"]
        assert gap["n_basins"] == 9 and gap["basins_b_better"] == 9 and gap["resolvable"] is True
        leak = result["leakage"]["9/seed42"]
        assert leak["events"] == EVENTS and leak["delta"] > 0 and leak["resolvable"] is True
        # LOEO re-scored on the official test chips, not the fold's own per-event matrix:
        assert leak["iou_loeo_same_chips"] == pytest.approx(np.mean([(40 + k) / (60 + k) for k in range(9)]), abs=1e-9)
        assert "same chips" in text and "| 9 |" in text
        # Conformal: events E0..E5 have mean chip FNR 0.05..0.10 <= 0.1, E6..E8 exceed it.
        assert result["conformal"]["9/seed42"]["events_held"] == 6 and "| 9 | 42 | 0.1 | 6/9 |" in text
