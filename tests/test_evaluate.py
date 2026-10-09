"""Calibration, the threshold mapping, the report builder and the product layer (docs/09)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from floodfusion.calibration import (
    LogitReservoir,
    apply_calibration,
    calibrate_logits,
    fit_platt,
    fit_temperature,
    prior_correction_offset,
)
from floodfusion.metrics import ConfusionMatrix, average_precision, precision_recall_curve


def _labels_from(z: np.ndarray, a: float, b: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(z.size) < 1.0 / (1.0 + np.exp(-(a * z + b)))).astype(float)


class TestCalibration:
    def test_temperature_is_recovered(self) -> None:
        z = np.random.default_rng(1).normal(0, 4, 200_000)
        assert fit_temperature(z, _labels_from(z, 0.5, 0.0)) == pytest.approx(2.0, rel=0.03)

    def test_platt_is_recovered(self) -> None:
        z = np.random.default_rng(2).normal(-3, 2, 300_000)
        a, b = fit_platt(z, _labels_from(z, 1.7, 1.2))
        assert a == pytest.approx(1.7, rel=0.05) and b == pytest.approx(1.2, abs=0.08)

    def test_prior_offset_is_one_logit_constant(self) -> None:
        """docs/05 8.3: o_corrected = o_model * (pi_d / (1 - pi_d)) * ((1 - pi_t) / pi_t)."""
        c = prior_correction_offset(0.5, 0.02)
        p = 0.6
        odds = p / (1 - p) * (0.02 / 0.98) * (0.5 / 0.5)
        assert float(apply_calibration(np.array([math.log(p / (1 - p))]), 1.0, c)[0]) == pytest.approx(odds / (1 + odds))

    def test_parsimony_keeps_the_temperature_when_it_suffices(self) -> None:
        z = np.random.default_rng(3).normal(0, 3, 200_000)
        result = calibrate_logits(z, _labels_from(z, 0.5, 0.0), pi_train=0.5, pi_deploy=0.5)
        assert result.method == "temperature" and result.temperature == pytest.approx(2.0, rel=0.05)
        assert result.ece_calibrated < result.ece_raw

    def test_platt_is_selected_when_the_prior_shift_is_wrong(self) -> None:
        """An under-confident model: the fixed prior shift and one scale cannot both be right."""
        z = np.random.default_rng(4).normal(-3, 2, 300_000)
        result = calibrate_logits(z, _labels_from(z, 1.7, 1.2), pi_train=0.07, pi_deploy=0.025)
        assert result.method == "platt" and result.nll_platt < result.nll_temperature
        assert result.temperature == pytest.approx(1 / 1.7, rel=0.06)

    def test_calibration_leaves_ranking_and_ap_unchanged(self) -> None:
        """docs/09 4.4: a monotone map cannot change AP."""
        rng = np.random.default_rng(5)
        z = rng.normal(-2, 2, 50_000)
        y = _labels_from(z, 1.0, 0.0).astype(bool)

        def ap(scores: np.ndarray) -> float:
            cms = tuple(ConfusionMatrix(int((scores > t)[y].sum()), int((scores > t)[~y].sum()),
                                        int((scores <= t)[y].sum()), int((scores <= t)[~y].sum()))
                        for t in np.quantile(scores, np.linspace(0.001, 0.999, 300)))
            return average_precision(*precision_recall_curve(cms))

        assert ap(z) == pytest.approx(ap(apply_calibration(z, 2.7, -1.3)), abs=1e-9)

    def test_rejects_single_class(self) -> None:
        with pytest.raises(ValueError):
            fit_temperature(np.zeros(10), np.zeros(10))


class TestThresholdMapping:
    @pytest.mark.parametrize("t,c", [(1.0, 0.0), (2.5, -1.1), (0.6, 0.7)])
    def test_logit_space_comparison_is_exact(self, t: float, c: float) -> None:
        """p = sigma((z + c)/T) > tau  iff  z > T * logit(tau) - c (train.validate)."""
        z = np.linspace(-12, 12, 5001)
        for tau in (0.05, 0.25, 0.5, 0.9):
            direct = apply_calibration(z, t, c) > tau
            mapped = z > t * math.log(tau / (1 - tau)) - c
            assert (direct == mapped).mean() > 0.9995  # equality up to float rounding at the edge


class TestReservoir:
    def test_capacity_and_uniformity(self) -> None:
        r = LogitReservoir(1000, seed=1)
        for start in range(0, 100_000, 7000):
            chunk = np.arange(start, min(start + 7000, 100_000))
            r.add(chunk.astype(np.float32), (chunk % 2).astype(np.uint8))
        z, y = r.arrays()
        assert r.seen == 100_000 and z.size == 1000
        assert np.percentile(z, 50) == pytest.approx(50_000, rel=0.1)
        assert set(np.unique(y)) <= {0, 1}

    def test_small_streams_are_kept_whole(self) -> None:
        r = LogitReservoir(100)
        r.add(np.arange(10.0), np.ones(10))
        z, _ = r.arrays()
        assert sorted(z.tolist()) == list(range(10))


# ---------------------------------------------------------------------------
# Report builder on fabricated run directories
# ---------------------------------------------------------------------------
def _write_run(root, name, row, per_basin, *, sha="abc", dirty=False, trained=True, baseline=False):
    d = root / name / ("baseline" if baseline else "seed42")
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "name": name, "ablation_row": row, "status": "finished", "git_sha": sha, "git_dirty": dirty,
        "trained": trained, "train_seconds": 360.0, "peak_vram_bytes": 2 * 1024**3,
        "vram_prediction": {"relative_error": 0.05}}))
    cms = {b: {"tp": tp, "fp": fp, "fn": fn, "tn": 1000} for b, (tp, fp, fn) in per_basin.items()}
    ious = [tp / (tp + fp + fn) for tp, fp, fn in per_basin.values()]
    (d / "eval_test.json").write_text(json.dumps({
        "name": name, "ablation_row": row, "per_basin": cms, "macro_iou": float(np.mean(ious)),
        "pooled_iou": 0.5, "latency": {"latency_ms_median": 6.0}, "synthetic_data": True,
        "manifest_sha256": "f" * 64, "baseline": baseline, "calibration": {"method": "temperature"}}))


@pytest.fixture()
def fake_runs(tmp_path, monkeypatch):
    import floodfusion.utils.env as env

    monkeypatch.setattr(env, "git_state", lambda *a, **k: {"git_sha": "abc", "git_branch": "x", "git_dirty": False})
    basins = ("B1", "B2", "B3")
    _write_run(tmp_path, "row02_baseline_adaptive", 2, dict(zip(basins, [(50, 30, 30), (40, 40, 30), (60, 20, 20)])),
               trained=False, baseline=True)
    _write_run(tmp_path, "row09_fusion_crossattn", 9, dict(zip(basins, [(80, 10, 10), (70, 15, 15), (85, 5, 10)])))
    _write_run(tmp_path, "row11_fusion_eval_sar_only", 11, dict(zip(basins, [(75, 12, 13), (68, 16, 16), (80, 8, 12)])),
               trained=False)
    _write_run(tmp_path, "row13_fusion_random_split", 13, dict(zip(basins, [(90, 5, 5), (85, 8, 7), (92, 3, 5)])))
    return tmp_path


class TestAblationTable:
    def test_main_table_has_the_milestones_columns_and_all_rows(self, fake_runs) -> None:
        from floodfusion.reporting import build_ablation_table

        text = build_ablation_table(fake_runs, out=fake_runs / "table.md", n_boot=500)
        assert "| #   | Configuration | Purpose — the question the row answers | IoU | F1  | Peak VRAM | Train h | ms/chip |" in text
        for row in range(1, 14):
            assert f"\n| {row} | " in text
        assert "SYNTHETIC BENCHMARK" in text and "not run" in text
        assert (fake_runs / "table.json").is_file() and (fake_runs / "figures" / "ablation_basin_dots.png").is_file()

    def test_gaps_are_paired_and_signed(self, fake_runs) -> None:
        from floodfusion.reporting import build_ablation_table

        build_ablation_table(fake_runs, out=fake_runs / "t.md", n_boot=500)
        gaps = json.loads((fake_runs / "t.json").read_text())["gaps"]
        assert gaps["13-9"]["delta"] > 0 and gaps["9-11"]["delta"] > 0 and gaps["9-2"]["delta"] > 0

    def test_refuses_runs_from_another_sha(self, fake_runs, monkeypatch) -> None:
        import floodfusion.utils.env as env
        from floodfusion.reporting import build_ablation_table

        monkeypatch.setattr(env, "git_state", lambda *a, **k: {"git_sha": "def", "git_branch": "x", "git_dirty": False})
        with pytest.raises(ValueError, match=r"P1-M6\.3"):
            build_ablation_table(fake_runs, n_boot=200)
        assert "DIRTY" in build_ablation_table(fake_runs, n_boot=200, allow_dirty=True)


class TestProduct:
    def test_classes_and_flags(self) -> None:
        from floodfusion.config import ValidityFlag
        from floodfusion.predict import PRODUCT_FLAGS, build_flags, product_classes

        prob = np.array([[0.9, 0.1, 0.3, np.nan], [0.9, 0.9, 0.2, 0.5]], dtype=np.float32)
        validity = np.array([[0, 0, 0, ValidityFlag.NODATA], [ValidityFlag.SHADOW, ValidityFlag.PERMANENT_WATER, 0, 0]])
        slope = np.array([[1.0, 15.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        landcover = np.array([[10, 50, 40, 40], [40, 80, 40, 40]], dtype=np.uint8)
        flags = build_flags(validity, slope, landcover, prob, np.ones_like(prob, bool), slope_max_deg=10.0,
                            tau=0.25, cost_ratio=3.0)
        classes = product_classes(prob, flags, 0.25)
        assert classes.tolist() == [[1, 0, 1, 255], [255, 2, 0, 1]]
        assert flags[0, 0] >> PRODUCT_FLAGS["DENSE_FOREST"] & 1 and flags[0, 1] >> PRODUCT_FLAGS["DENSE_URBAN"] & 1
        assert flags[0, 1] >> PRODUCT_FLAGS["STEEP_SLOPE"] & 1
        assert flags[0, 2] >> PRODUCT_FLAGS["LOW_CONFIDENCE"] & 1  # 0.3 flips if r changes by 1.5x
        assert not flags[1, 3] >> PRODUCT_FLAGS["LOW_CONFIDENCE"] & 1
