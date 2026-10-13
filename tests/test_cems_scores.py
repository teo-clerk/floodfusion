"""CEMS benchmark scoring: units (activations, clusters), test subsets, verdict rules and the report."""

from __future__ import annotations

import json

import numpy as np
import pytest
from shapely.geometry import box

from floodfusion.cems_scores import (
    MIN_UNITS_FOR_VERDICT,
    GapTest,
    ModelSpec,
    activation_clusters,
    load_spec,
    one_sided_sign_flip_p,
    report,
    run_tests,
    units,
)


def _spec(**overrides) -> dict:
    spec = {
        "models": [ModelSpec("sar", "SAR U-Net", ("r4",)), ModelSpec("fus", "Fusion", ("r9",))],
        "tests": [GapTest("P1", "fusion helps", "sar", "fus", "optical"),
                  GapTest("P2", "R2 holds", "sar", "fus", "sar_only", margin=0.05),
                  GapTest("P4", "anything", "sar", "fus", "all")],
        "primary_stratum": "sar", "exclude": ("EMSR664",), "optical_usable": 0.5, "sar_only_below": 0.1,
        "overlapping_pairs": (),
    }
    return {**spec, **overrides}


def _cms(iou: float) -> list[list[int]]:
    """1000 flood px; IoU = tp / (tp + fp + fn) with fp = fn."""
    tp = 1000
    miss = round(tp * (1 / iou - 1) / 2)
    return [[tp, miss, miss, 100_000]] * 3


def _pair(activation: str, aoi: int, *, stratum: str = "sar", clear: float = 0.0, sar_iou: float = 0.4,
          fus_iou: float = 0.5, seeds: int = 1, flood_px: int = 1000) -> tuple[str, dict]:
    key = f"{activation}_AOI{aoi:02d}"
    per_seed = lambda iou: {f"seed{42 + s}": _cms(iou) for s in range(seeds)}  # noqa: E731
    return key, {"pair": {"activation": activation, "aoi": aoi, "product": key, "stratum": stratum, "sensors": ["PAZ"],
                          "offset_h": 3.0, "pass_time": "", "product_time": ""},
                 "scored_px": 10_000, "clear_px": int(clear * 10_000), "flood_px": flood_px,
                 "models": {"sar": per_seed(sar_iou), "fus": per_seed(fus_iou)}}


def _scores(pairs: list[tuple[str, dict]], events: dict | None = None) -> dict:
    acts = sorted({p["pair"]["activation"] for _, p in pairs})
    events = events or {a: {"eventTime": f"2024-0{1 + i % 9}-01T00:00:00",
                            "extent": box(i * 5.0, 0.0, i * 5.0 + 0.1, 0.1).wkt} for i, a in enumerate(acts)}
    return {"pairs": dict(pairs), "activations": events,
            "models": {"sar": {"name": "SAR U-Net"}, "fus": {"name": "Fusion"}}}


def test_aois_and_seeds_pool_per_activation_and_excluded_or_other_strata_drop_out() -> None:
    scores = _scores([_pair("EMSR700", 1, seeds=3), _pair("EMSR700", 2, seeds=3), _pair("EMSR664", 1),
                      _pair("EMSR701", 1, stratum="optical")])
    us = units(scores, _spec(), stratum="sar")
    assert list(us) == ["EMSR700"]
    assert us["EMSR700"]["scored_px"] == 20_000 and us["EMSR700"]["models"]["sar"][0][0] == 6000   # 2 AOIs x 3 seeds


def test_activations_close_in_space_and_time_form_one_cluster() -> None:
    events = {"EMSR701": {"eventTime": "2024-09-14T00:00:00", "extent": box(16.0, 50.0, 16.5, 50.5).wkt},
              "EMSR702": {"eventTime": "2024-09-20T00:00:00", "extent": box(17.5, 50.0, 18.0, 50.5).wkt},  # ~70 km
              "EMSR703": {"eventTime": "2024-12-30T00:00:00", "extent": box(16.0, 50.0, 16.5, 50.5).wkt},  # too late
              "EMSR704": {"eventTime": "2024-09-15T00:00:00", "extent": box(40.0, 10.0, 40.5, 10.5).wkt}}  # far
    clusters = activation_clusters(events)
    assert clusters == {"EMSR701": "EMSR701", "EMSR702": "EMSR701", "EMSR703": "EMSR703", "EMSR704": "EMSR704"}
    scores = _scores([_pair(a, 1) for a in events], events)
    assert sorted(units(scores, _spec(), stratum="sar", unit="cluster")) == ["EMSR701", "EMSR703", "EMSR704"]


def test_verdicts_need_eight_units_and_holm_covers_two_sided_tests_only() -> None:
    n = MIN_UNITS_FOR_VERDICT
    pairs = [_pair(f"EMSR{700 + i}", 1, clear=0.9) for i in range(n)] + \
            [_pair(f"EMSR{800 + i}", 1, clear=0.0, fus_iou=0.41) for i in range(3)]
    us = units(_scores(pairs), _spec(), stratum="sar")
    results = {r["id"]: r for r in run_tests(us, _spec())}
    assert results["P1"]["k"] == n and results["P1"]["wins"] == n and results["P1"]["holds"]
    assert results["P2"]["k"] == 3 and results["P2"]["holds"] is None and results["P2"]["verdict"].startswith("none")
    assert results["P4"]["k"] == n + 3 and "holm_p" in results["P4"] and "holm_p" not in results["P2"]
    assert results["P4"]["holm_p"] == pytest.approx(2 * results["P4"]["p"])
    assert results["P1"]["holm_p"] == pytest.approx(max(results["P1"]["p"], results["P4"]["holm_p"]))


def test_non_inferiority_is_an_exact_one_sided_test_of_the_margin_shifted_gap() -> None:
    spec = _spec(tests=[GapTest("P2", "R2 holds", "sar", "fus", "sar_only", margin=0.05)])
    slightly_worse = [_pair(f"EMSR{700 + i}", 1, fus_iou=0.40 - 0.002 * (i % 3)) for i in range(MIN_UNITS_FOR_VERDICT)]
    [r] = run_tests(units(_scores(slightly_worse), spec, stratum="sar"), spec)
    assert r["holds"] and r["p"] == pytest.approx(1 / 2 ** MIN_UNITS_FOR_VERDICT) and r["verdict"].startswith("non-inferior")
    much_worse = [_pair(f"EMSR{700 + i}", 1, fus_iou=0.30) for i in range(MIN_UNITS_FOR_VERDICT)]
    [r] = run_tests(units(_scores(much_worse), spec, stratum="sar"), spec)
    assert not r["holds"] and r["verdict"].startswith("not shown")


def test_one_sided_sign_flip_is_exact_for_few_units_and_never_zero_by_monte_carlo() -> None:
    assert one_sided_sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(1 / 8)
    assert one_sided_sign_flip_p([-1.0, -1.0, -1.0]) == pytest.approx(1.0)
    p = one_sided_sign_flip_p([1.0] * 30)
    assert 0 < p < 1e-4


def test_units_with_no_scorable_iou_do_not_count_towards_k() -> None:
    pairs = [_pair(f"EMSR{700 + i}", 1, clear=0.9) for i in range(4)]
    empty = _pair("EMSR799", 1, clear=0.9)
    empty[1]["models"] = {"sar": {"seed42": [[0, 0, 0, 100]] * 3}, "fus": {"seed42": [[0, 0, 0, 100]] * 3}}
    [r] = run_tests(units(_scores([*pairs, empty]), _spec(), stratum="sar"),
                    _spec(tests=[GapTest("P1", "q", "sar", "fus", "optical")]))
    assert r["k"] == 4


def test_report_shows_both_unit_definitions_and_the_sensitivity_analyses(tmp_path) -> None:
    pairs = [_pair(f"EMSR{700 + i}", 1, clear=0.9) for i in range(MIN_UNITS_FOR_VERDICT)]
    pairs.append(_pair("EMSR799", 1, clear=0.3, flood_px=50))           # partial cloud, under 1 km² of flood
    spec = _spec(overlapping_pairs=("EMSR700_AOI01",))
    result = report(_scores(pairs), spec, tmp_path / "cems.md")
    text = (tmp_path / "cems.md").read_text(encoding="utf-8")
    assert "Per activation" in text and "Per cluster" in text and "Supported" in text and "| P1 |" in text
    assert "Partial cloud" in text and "km² of flood" in text and "overlap Kuro Siwo" in text
    data = json.loads((tmp_path / "cems.json").read_text(encoding="utf-8"))
    assert data["units"] == result["units"] and "EMSR799" in data["units"]["activation"]
    small = next(v for k, v in data["descriptive"].items() if "km² of flood" in k)
    assert all("EMSR799" not in r["units"] for r in small)


def test_spec_refuses_tests_on_unknown_models(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("models: [{key: a, name: A, runs: [x]}]\ntests: [{id: P1, question: q, a: a, b: zzz}]\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="unknown models"):
        load_spec(path)


def test_the_no_optical_windows_of_a_fusion_model_are_read_with_its_sar_only_calibration() -> None:
    from floodfusion.case_study import recalibrate_without_optical

    cal, sar_cal = {"temperature": 0.8, "offset": 1.6}, {"temperature": 1.4, "offset": 0.4}
    z = np.array([[-3.0, 0.5], [2.0, -1.0]])
    prob = 1 / (1 + np.exp(-(z + cal["offset"]) / cal["temperature"]))
    used = np.array([[True, False], [True, False]])
    out = recalibrate_without_optical(prob.astype(np.float32), used, cal, sar_cal)
    expected = 1 / (1 + np.exp(-(z + sar_cal["offset"]) / sar_cal["temperature"]))
    np.testing.assert_allclose(out[used], prob[used], rtol=1e-6)
    np.testing.assert_allclose(out[~used], expected[~used], rtol=1e-4)


def test_the_committed_report_regenerates_from_the_committed_scores(tmp_path) -> None:
    from pathlib import Path

    from floodfusion.cems_scores import main

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "cems.md"
    assert main(["report", "--spec", str(root / "configs" / "datasets" / "cems_models.yaml"),
                 "--scores", str(root / "reports" / "cems_benchmark_scores.json"), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == (root / "reports" / "cems_benchmark.md").read_text(encoding="utf-8")
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8")) == \
        json.loads((root / "reports" / "cems_benchmark.json").read_text(encoding="utf-8"))
