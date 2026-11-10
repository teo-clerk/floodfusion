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
