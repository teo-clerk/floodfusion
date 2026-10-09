"""Regression tests for the defects found by the code / ML-methodology / security reviews.

Each test names the failure it pins: it failed (or would have) before the fix.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from floodfusion.calibration import calibrate_logits
from floodfusion.data.records import ChipRecord
from floodfusion.data.stac import NoMatchingPairError, group_by_time
from floodfusion.splits import _partition_basins, random_chip_split


# ---------------------------------------------------------------------------
# Splits: optical spread over the splits; row 13 changes ONLY the training split
# ---------------------------------------------------------------------------
class TestSplits:
    def test_optical_bearing_basins_reach_val_and_test(self) -> None:
        flood = {f"B{i}": w for i, w in enumerate([4.0, 2.2, 1.5, 1.1, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3])}
        optical = {"B0": 32, "B4": 23, "B5": 32, "B2": 10, "B6": 9}  # five optical-bearing basins
        split = _partition_basins(flood, 0.6, 0.2, seed=0, secondary=optical)
        bearing_in = {s: sum(1 for b in optical if split[b] == s) for s in ("train", "val", "test")}
        assert bearing_in["train"] >= 2 and bearing_in["val"] >= 1 and bearing_in["test"] >= 2
        assert set(split.values()) == {"train", "val", "test"}

    def test_partition_is_deterministic(self) -> None:
        flood = {f"B{i}": float(i + 1) for i in range(9)}
        optical = {"B1": 5.0, "B4": 5.0, "B7": 5.0}
        assert _partition_basins(flood, 0.6, 0.2, seed=3, secondary=optical) == \
            _partition_basins(flood, 0.6, 0.2, seed=3, secondary=optical)

    def test_row13_keeps_row9_validation(self) -> None:
        ids = [f"c{i:03d}" for i in range(60)]
        basins = {c: f"B{i % 6}" for i, c in enumerate(ids)}
        test, val = ids[:10], ids[10:20]
        leaky = random_chip_split(ids, basins, seed=1, fixed_test_chips=test, fixed_val_chips=val,
                                  match_counts={"train": (25, 10), "val": (10, 5)}, positive_chips=ids[::2])
        assert leaky.test_chips == tuple(sorted(test)) and leaky.val_chips == tuple(sorted(val))
        assert len(leaky.train_chips) == 25 and not set(leaky.train_chips) & set(val + test)

    def test_row13_rejects_overlapping_fixed_sets(self) -> None:
        ids = [f"c{i}" for i in range(10)]
        with pytest.raises(ValueError, match="overlap"):
            random_chip_split(ids, {c: "B" for c in ids}, fixed_test_chips=ids[:3], fixed_val_chips=ids[2:5])


# ---------------------------------------------------------------------------
# Catalogue grouping and the drop exception
# ---------------------------------------------------------------------------
class TestCatalogue:
    def test_tiles_of_one_pass_straddling_a_minute_are_one_overpass(self) -> None:
        t0 = datetime(2023, 11, 7, 10, 21, 59, tzinfo=UTC)
        times = [t0.isoformat(), (t0 + timedelta(seconds=2)).isoformat(), (t0 + timedelta(days=3)).isoformat()]
        assert group_by_time(times) == [[0, 1], [2]]

    def test_drop_exception_is_a_lookup_error_but_not_a_key_error(self) -> None:
        assert issubclass(NoMatchingPairError, LookupError) and not issubclass(KeyError, NoMatchingPairError)


# ---------------------------------------------------------------------------
# Security / robustness of the readers
# ---------------------------------------------------------------------------
class TestReaders:
    def test_file_scheme_cannot_smuggle_a_gdal_virtual_path(self) -> None:
        from floodfusion.data.sources import check_href

        with pytest.raises(ValueError):
            check_href("file:///vsicurl/https://evil.example/x.tif")

    def test_db_layer_fails_the_linear_power_check(self) -> None:
        from floodfusion.data.sources import verify_rtc_linear_power

        db = np.random.default_rng(0).normal(-12, 3, 10_000)
        db[:500] = np.abs(db[:500])  # a few positive dB values must not rescue it
        with pytest.raises(ValueError, match="dB"):
            verify_rtc_linear_power(db)

    def test_shadow_simulation_survives_nan_incidence(self) -> None:
        from floodfusion.data.coregistration import simulate_backscatter

        theta = np.full((3, 3), 30.0)
        theta[0, 0] = np.nan
        shadow = np.zeros((3, 3), bool)
        shadow[1, 1] = True
        assert np.isfinite(simulate_backscatter(theta, shadow=shadow)[1, 1])


# ---------------------------------------------------------------------------
# Calibration on degenerate validation sets
# ---------------------------------------------------------------------------
def test_single_class_validation_gives_the_identity_not_a_crash() -> None:
    result = calibrate_logits(np.linspace(-3, 3, 100), np.zeros(100), pi_train=0.1, pi_deploy=0.0)
    assert result.method == "identity" and result.temperature == 1.0 and result.offset == 0.0


# ---------------------------------------------------------------------------
# Training data: an empty row-6 split is a skipped row, not a crashed grid
# ---------------------------------------------------------------------------
def test_empty_split_is_a_dedicated_error() -> None:
    from floodfusion.train import EmptySplitError, _filter_row6

    cloudy = ChipRecord("c0", "B0", "E", Path("x.tif"), (0, 0, 256, 256), 0.1, 1.0, True, 1, "ascending",
                        optical_clear_fraction=0.3)
    assert _filter_row6([cloudy]) == ()
    assert issubclass(EmptySplitError, ValueError)


# ---------------------------------------------------------------------------
# The table: matched seeds, and no verdict with too few basins
# ---------------------------------------------------------------------------
def _run(root: Path, name: str, row: int, seed: int | None, cms: dict[str, tuple[int, int, int]],
         test_sha: str = "t", config_sha: str = "c") -> None:
    d = root / name / (f"seed{seed}" if seed is not None else "baseline")
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "name": name, "ablation_row": row, "status": "finished", "git_sha": "abc", "git_dirty": False,
        "seed": seed, "trained": seed is not None, "train_seconds": 100.0, "peak_vram_bytes": 2**30,
        "config_sha256_without_seed": config_sha}))
    ious = [tp / (tp + fp + fn) for tp, fp, fn in cms.values()]
    (d / "eval_test.json").write_text(json.dumps({
        "name": name, "ablation_row": row, "macro_iou": float(np.mean(ious)), "manifest_sha256": "m",
        "test_chips_sha256": test_sha, "baseline": seed is None,
        "per_basin": {b: {"tp": tp, "fp": fp, "fn": fn, "tn": 1000} for b, (tp, fp, fn) in cms.items()}}))


@pytest.fixture()
def table_root(tmp_path, monkeypatch):
    import floodfusion.utils.env as env

    monkeypatch.setattr(env, "git_state", lambda *a, **k: {"git_sha": "abc", "git_branch": "x", "git_dirty": False})
    good = {"B1": (80, 10, 10), "B2": (70, 15, 15), "B3": (85, 5, 10)}
    bad = {"B1": (10, 80, 10), "B2": (10, 80, 10), "B3": (10, 80, 10)}
    _run(tmp_path, "row09_fusion_crossattn", 9, 42, good)
    _run(tmp_path, "row09_fusion_crossattn", 9, 43, bad)  # a terrible second seed
    _run(tmp_path, "row11_fusion_eval_sar_only", 11, 42, {"B1": (75, 12, 13), "B2": (68, 16, 16), "B3": (80, 8, 12)})
    return tmp_path


def test_gaps_compare_matched_seeds_only(table_root) -> None:
    from floodfusion.reporting import build_ablation_table

    build_ablation_table(table_root, out=table_root / "t.md", n_boot=300)
    gap = json.loads((table_root / "t.json").read_text())["gaps"]["9-11"]
    assert gap["seeds"] == [42]
    assert gap["delta"] > 0  # pooling row 9's bad seed 43 against row 11's seed 42 would flip the sign


def test_no_verdict_with_three_basins(table_root) -> None:
    from floodfusion.reporting import build_ablation_table

    text = build_ablation_table(table_root, out=table_root / "t.md", n_boot=300)
    gap = json.loads((table_root / "t.json").read_text())["gaps"]["9-11"]
    assert gap["resolvable"] is None and gap["basins_b_better"] == 3 and "K = 3 < 8" in text


def test_mixed_test_sets_are_refused(table_root) -> None:
    from floodfusion.reporting import build_ablation_table

    _run(table_root, "row05_unet_sar_dem", 5, 42, {"B1": (75, 12, 13)}, test_sha="other")
    with pytest.raises(ValueError, match="different test sets"):
        build_ablation_table(table_root, n_boot=100)


def test_seeds_of_one_row_must_share_a_configuration(table_root) -> None:
    from floodfusion.reporting import build_ablation_table

    _run(table_root, "row11_fusion_eval_sar_only", 11, 43, {"B1": (75, 12, 13), "B2": (68, 16, 16), "B3": (80, 8, 12)},
         config_sha="different")
    with pytest.raises(ValueError, match="different configurations"):
        build_ablation_table(table_root, n_boot=100)


# ---------------------------------------------------------------------------
# Statistics (ML-methodology review): an exact test with multiplicity control
# ---------------------------------------------------------------------------
def test_sign_flip_test_is_exact_and_cannot_reach_significance_with_three_basins() -> None:
    from floodfusion.metrics import sign_flip_p_value

    assert sign_flip_p_value([0.1, 0.2, 0.3]) == pytest.approx(0.25)           # 2 of 8 patterns as extreme
    assert sign_flip_p_value([0.05] * 10) == pytest.approx(2 / 1024)
    assert sign_flip_p_value([0.1, -0.1, 0.1, -0.1]) == pytest.approx(1.0)


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    from floodfusion.metrics import holm_adjust

    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert np.isnan(adj["d"])


@pytest.mark.slow
def test_percentile_basin_bootstrap_undercovers_with_few_basins() -> None:
    """The reason for MIN_BASINS_FOR_VERDICT: simulated coverage of a nominal 95 % interval."""
    rng = np.random.default_rng(0)

    def coverage(k: int, sims: int = 400, n_boot: int = 2000) -> float:
        hits = 0
        for _ in range(sims):
            x = rng.normal(0.0, 1.0, k)
            boots = x[rng.integers(0, k, (n_boot, k))].mean(axis=1)
            lo, hi = np.quantile(boots, [0.025, 0.975])
            hits += lo <= 0.0 <= hi
        return hits / sims

    assert coverage(4) < 0.85                      # ~0.80: far below the nominal 0.95
    assert coverage(10) > coverage(4)              # it improves with K, still below nominal
