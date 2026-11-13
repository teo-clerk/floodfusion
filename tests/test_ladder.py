"""v1.3's adaptation ladder (``floodfusion.ladder``; protocol v1.3 §3): the rungs from histograms, and the rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floodfusion import histograms, ladder
from floodfusion.data.records import ChipRecord, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def _hist(**bins: int) -> np.ndarray:
    """Counts in the named bins (``b200=100``), zero elsewhere."""
    out = np.zeros(histograms.N_BINS + 2, dtype=np.int64)
    for name, n in bins.items():
        out[int(name[1:])] = n
    return out


def _unit(pos: dict[str, int], neg: dict[str, int]) -> dict[str, np.ndarray]:
    return {"pos": _hist(**pos), "neg": _hist(**neg)}


def test_rung_one_scores_each_unit_at_tau_star():
    units = {"EMSR700": _unit({"b200": 100}, {"b100": 1000, "b240": 25})}
    assert ladder.rung1(units, tau=0.25) == {"EMSR700": pytest.approx(100 / 125)}


def test_rung_two_fits_each_folds_threshold_on_the_other_folds_only():
    # Fold 0 (A, B): flood at bin 200, dry at bin 100. Fold 1 (C): flood at bin 150, dry at bin 120.
    units = {"A": _unit({"b200": 100}, {"b100": 1000}), "B": _unit({"b200": 50}, {"b100": 500}),
             "C": _unit({"b150": 100}, {"b120": 50})}
    fold_of = {"A": "fold0", "B": "fold0", "C": "fold1"}
    ious, cuts = ladder.rung2(units, fold_of, cost_ratio=3.0)
    # C is cut where fold 0 is cut best (lowest zero-cost cut, between bins 100 and 200): its dry bin 120 floods.
    assert ious["C"] == pytest.approx(100 / 150)
    # A and B are cut where C is cut best (between 120 and 150): bin 200 floods, bin 100 stays dry.
    assert ious["A"] == pytest.approx(1.0) and ious["B"] == pytest.approx(1.0)
    assert set(cuts) == {"fold0", "fold1"} and cuts["fold1"] < cuts["fold0"]


def test_a_cluster_pools_its_activations_each_cut_at_its_own_threshold():
    units = {"A": _unit({"b200": 100}, {"b120": 50}), "B": _unit({"b150": 100}, {"b120": 50}),
             "C": _unit({"b200": 10}, {})}
    cut_a = ladder._edge_tau(110)                      # bins from 111 flood: A's dry bin 120 counts as flood
    cut_b = ladder._edge_tau(130)                      # bins from 131 flood: B's dry bin 120 stays dry
    ious = ladder.cluster_ious(units, {"A": cut_a, "B": cut_b, "C": cut_b}, {"A": "AB", "B": "AB"})
    assert ious == {"AB": pytest.approx(200 / 250), "C": pytest.approx(1.0)}


def test_rung_two_needs_every_unit_in_a_fold():
    with pytest.raises(KeyError, match="EMSR701"):
        ladder.rung2({"EMSR701": _unit({"b200": 1}, {})}, {}, cost_ratio=3.0)


@pytest.mark.parametrize(("upper", "adopted"), [
    ({"A": 0.51, "B": 0.51, "C": 0.40}, True),      # +0.007, better in 2 of 3
    ({"A": 0.504, "B": 0.504, "C": 0.40}, False),   # +0.0027: below the minimum gain
    ({"A": 0.60, "B": 0.40, "C": 0.40}, False),     # +0.033 but better in 1 of 3 (ties do not count)
])
def test_the_adoption_rule_needs_the_gain_and_a_majority(upper, adopted):
    verdict = ladder.adopt({"A": 0.50, "B": 0.50, "C": 0.40}, upper)
    assert verdict["adopted"] is adopted and verdict["n"] == 3


def test_the_adoption_rule_refuses_different_unit_sets():
    with pytest.raises(ValueError, match="same activations"):
        ladder.adopt({"A": 0.5}, {"B": 0.5})


def test_fold_of_activation_reads_the_frozen_test_groups(tmp_path: Path):
    chips = tmp_path / "chips"
    stack = tmp_path / "stack.tif"
    stack.write_bytes(b"")
    records = [ChipRecord(chip_id=f"{a}_AOI01_t00_00_{i}", basin_id=f"POOL_cems_{g}", event_id=f"{a}_AOI01_t00_00",
                          raster_path=stack, window=(0, 0, 256, 256), positive_fraction=0.1, valid_fraction=1.0,
                          optical_available=False, relative_orbit=1, orbit_state="ascending")
               for a, g in (("EMSR700", "EMSR700"), ("EMSR701", "EMSR700"), ("EMSR702", "EMSR702")) for i in range(2)]
    write_manifest(records, chips / "manifest.parquet")
    (chips / "cv").mkdir()
    for name, test in (("fold0", ["POOL_cems_EMSR700"]), ("fold1", ["POOL_cems_EMSR702"])):
        (chips / "cv" / f"kfold__{name}.json").write_text(json.dumps({"test_basins": test}))
    (chips / "cv" / "folds.json").write_text(json.dumps(
        [{"protocol": "kfold", "name": n, "split_file": f"cv/kfold__{n}.json", "norm_file": "", "strategy": "x"}
         for n in ("fold0", "fold1")] + [{"protocol": "all", "name": "all", "split_file": "cv/all__all.json",
                                         "norm_file": "", "strategy": "x"}]))
    assert ladder.fold_of_activation(chips) == {"EMSR700": "fold0", "EMSR701": "fold0", "EMSR702": "fold1"}


def test_rung_one_on_the_committed_scores_reproduces_the_committed_binned_iou():
    """The code check of protocol v1.3 §3: three seeds of row 21 give 0.418 ("IoU at τ*, binned")."""
    scores = ladder.load_scores(ROOT / "reports" / "v12_cems_rebuilt_scores.json.gz")
    per_seed = [ladder.rung1(ladder.unit_histograms(scores, seed=s), tau=0.25) for s in ("seed42", "seed43", "seed44")]
    assert len(per_seed[0]) == 38
    per_unit = {a: np.mean([p[a] for p in per_seed]) for a in per_seed[0]}
    assert round(float(np.mean(list(per_unit.values()))), 3) == 0.418


# ---------------------------------------------------------------------------
# Rung 3: fold models scored on the pairs they held out, and the three-rung report
# ---------------------------------------------------------------------------
def _case(tmp_path: Path, pairs: dict[str, tuple[str, str]]) -> Path:
    """A benchmark folder: ``{pair key: (activation, stratum)}`` as pair.json files."""
    root = tmp_path / "bench"
    for key, (activation, stratum) in pairs.items():
        (root / key).mkdir(parents=True)
        (root / key / "pair.json").write_text(json.dumps({"key": key, "activation": activation, "stratum": stratum}))
    return root


def test_fold_pairs_lists_every_stratum_of_each_folds_activations(tmp_path: Path):
    case = _case(tmp_path, {"EMSR700_AOI01": ("EMSR700", "sar"), "EMSR700_AOI02": ("EMSR700", "optical"),
                            "EMSR701_AOI01": ("EMSR701", "sar"), "EMSR664_AOI01": ("EMSR664", "sar")})
    assert ladder.fold_pairs(case, {"EMSR700": "fold0", "EMSR701": "fold1"}) == {
        "fold0": ["EMSR700_AOI01", "EMSR700_AOI02"], "fold1": ["EMSR701_AOI01"]}


def test_score_r3_scores_each_fold_model_on_its_own_pairs_only(tmp_path: Path, monkeypatch):
    from floodfusion import cems_scores_v12

    case = _case(tmp_path, {"EMSR700_AOI01": ("EMSR700", "sar"), "EMSR701_AOI01": ("EMSR701", "sar")})
    calls = []

    def fake_score(spec, case_root, device, *, records=None, pairs_only=None):
        [model] = spec["models"]
        calls.append((model.key, model.runs, sorted(pairs_only)))
        return {"pairs": {k: {"pair": {"activation": k[:7]}} for k in pairs_only}, "activations": {k[:7]: {}
                for k in pairs_only}, "priors": {"r3": {"seed42": {"tau": 0.25, "pi0": 0.1}}},
                "models": {"r3": {"runs": list(model.runs)}}}

    monkeypatch.setattr(cems_scores_v12, "score", fake_score)
    runs = tmp_path / "runs"
    merged = ladder.score_r3(runs, case, {"EMSR700": "fold0", "EMSR701": "fold1"}, device="cpu")
    run = runs / "kfold" / "{}" / ladder.RUN_NAME / ladder.SEED
    assert calls == [("r3", (str(run).format("fold0"),), ["EMSR700_AOI01"]),
                     ("r3", (str(run).format("fold1"),), ["EMSR701_AOI01"])]
    assert sorted(merged["pairs"]) == ["EMSR700_AOI01", "EMSR701_AOI01"]
    assert merged["fold_priors"] == {"fold0": {"tau": 0.25, "pi0": 0.1}, "fold1": {"tau": 0.25, "pi0": 0.1}}


def _scores(model: str, units: dict[str, dict[str, np.ndarray]], *, flood: int = 100) -> dict:
    """A scores document: one radar-drawn pair per activation, activations 10° apart (no clusters)."""
    pairs, acts = {}, {}
    for i, (code, hist) in enumerate(sorted(units.items())):
        h = {k: v.tolist() for k, v in hist.items()} | {"all": (hist["pos"] + hist["neg"]).tolist()}
        pairs[f"{code}_AOI01"] = {"pair": {"activation": code, "stratum": "sar"}, "scored_px": 1000,
                                  "flood_px": flood, "clear_px": 0, "hist": {model: {"seed42": h}}}
        acts[code] = {"eventTime": "2023-05-01T00:00:00",
                      "extent": f"POLYGON (({10 * i} 0, {10 * i} 1, {10 * i + 1} 1, {10 * i + 1} 0, {10 * i} 0))"}
    return {"pairs": pairs, "activations": acts, "priors": {}, "models": {}}


def test_the_ladder_report_tests_fine_tuning_against_the_highest_adopted_rung():
    # tau = 0.25 floods every bin from 115 up. Row 21 puts dry pixels in the flood pixels' bin, so no threshold
    # separates them: R1 = R2 = 2/3 (R2 not adopted). R3 moves the dry pixels to bin 100: IoU 1.
    codes = ("EMSR700", "EMSR701", "EMSR702")
    v12 = _scores("ours", {c: _unit({"b200": 100}, {"b200": 50}) for c in codes})
    r3 = _scores("r3", {c: _unit({"b200": 100}, {"b100": 50}) for c in codes})
    r3["fold_priors"] = {f: {"tau": 0.25, "pi0": 0.1} for f in ("fold0", "fold1", "fold2")}
    fold_of = dict(zip(codes, ("fold0", "fold1", "fold2"), strict=True))
    report = ladder.ladder_report(v12, r3, fold_of)
    assert not report["verdicts"]["r2_over_r1"]["adopted"] and report["verdicts"]["r3_over_base"]["base"] == "r1"
    assert report["r1"] == {c: pytest.approx(2 / 3) for c in codes} and report["r3"] == {c: 1.0 for c in codes}
    assert report["verdicts"]["r3_over_base"]["adopted"] and report["v13_model"] == "R3"
    assert "R3, fine-tuned" in ladder.render_ladder(report)


def test_the_ladder_report_refuses_scores_from_different_stacks():
    v12 = _scores("ours", {"EMSR700": _unit({"b200": 100}, {})})
    r3 = _scores("r3", {"EMSR700": _unit({"b200": 100}, {})}, flood=99)
    r3["fold_priors"] = {"fold0": {"tau": 0.25, "pi0": 0.1}}
    with pytest.raises(ValueError, match="different stacks"):
        ladder.ladder_report(v12, r3, {"EMSR700": "fold0"})


def _chips(tmp_path: Path, folds: dict[str, str]) -> Path:
    """A pooled chip folder: one chip per activation, each activation its own group; ``{activation: fold}``."""
    chips, stack = tmp_path / "chips", tmp_path / "stack.tif"
    stack.write_bytes(b"")
    write_manifest([ChipRecord(chip_id=f"{a}_AOI01_t00_00_0", basin_id=f"POOL_cems_{a}", event_id=f"{a}_AOI01_t00_00",
                               raster_path=stack, window=(0, 0, 256, 256), positive_fraction=0.1, valid_fraction=1.0,
                               optical_available=False, relative_orbit=1, orbit_state="ascending") for a in folds],
                   chips / "manifest.parquet")
    (chips / "cv").mkdir()
    names = sorted(set(folds.values()))
    for name in names:
        (chips / "cv" / f"kfold__{name}.json").write_text(json.dumps(
            {"test_basins": [f"POOL_cems_{a}" for a, f in folds.items() if f == name]}))
    (chips / "cv" / "folds.json").write_text(json.dumps(
        [{"protocol": "kfold", "name": n, "split_file": f"cv/kfold__{n}.json", "norm_file": "", "strategy": "x"}
         for n in names]))
    return chips


def test_the_command_line_writes_both_ladder_reports(tmp_path: Path):
    from floodfusion.cems_scores_v12 import write_scores

    codes = ("EMSR700", "EMSR701", "EMSR702")
    v12 = _scores("ours", {c: _unit({"b200": 100}, {"b200": 50}) for c in codes})
    for entry in v12["pairs"].values():                     # rung 2's report also lists seeds 43 and 44
        entry["hist"]["ours"] |= {s: entry["hist"]["ours"]["seed42"] for s in ("seed43", "seed44")}
    r3 = _scores("r3", {c: _unit({"b200": 100}, {"b100": 50}) for c in codes})
    r3["fold_priors"] = {f: {"tau": 0.25, "pi0": 0.1} for f in ("fold0", "fold1", "fold2")}
    write_scores(v12, tmp_path / "v12.json.gz")
    write_scores(r3, tmp_path / "r3.json.gz")
    chips = _chips(tmp_path, dict(zip(codes, ("fold0", "fold1", "fold2"), strict=True)))
    assert ladder.main(["r2", "--scores", str(tmp_path / "v12.json.gz"), "--chips", str(chips),
                        "--out", str(tmp_path / "r2.json")]) == 0
    assert "not adopted" in (tmp_path / "r2.md").read_text()
    assert ladder.main(["report", "--scores", str(tmp_path / "v12.json.gz"), "--r3", str(tmp_path / "r3.json.gz"),
                        "--chips", str(chips), "--out", str(tmp_path / "ladder.json")]) == 0
    assert json.loads((tmp_path / "ladder.json").read_text())["v13_model"] == "R3"
    assert "**v1.3's model:** R3" in (tmp_path / "ladder.md").read_text()
