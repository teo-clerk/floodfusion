"""CEMS benchmark scoring: activation pooling, test subsets, verdict rules and the report."""

from __future__ import annotations

import json

import pytest

from floodfusion.cems_scores import MIN_UNITS_FOR_VERDICT, ModelSpec, GapTest, activations, load_spec, report, run_tests


def _spec(**overrides) -> dict:
    spec = {
        "models": [ModelSpec("sar", "SAR U-Net", ("r4",)), ModelSpec("fus", "Fusion", ("r9",))],
        "tests": [GapTest("P1", "fusion helps", "sar", "fus", "optical"),
                  GapTest("P2", "R2 holds", "sar", "fus", "sar_only", margin=0.05),
                  GapTest("P4", "anything", "sar", "fus", "all")],
        "primary_stratum": "sar", "exclude": ("EMSR664",), "optical_usable": 0.5, "sar_only_below": 0.1,
    }
    return {**spec, **overrides}


def _pair(activation: str, aoi: int, *, stratum: str = "sar", clear: float = 0.0, sar_iou: float = 0.4,
          fus_iou: float = 0.5) -> tuple[str, dict]:
    """A pair with 1000 flood px; IoU = tp / (tp + fp + fn) with fp = fn."""
    def cms(iou: float) -> list[list[int]]:
        tp = 1000
        miss = round(tp * (1 / iou - 1) / 2)
        return [[tp, miss, miss, 100_000]] * 3

    key = f"{activation}_AOI{aoi:02d}"
    return key, {"pair": {"activation": activation, "aoi": aoi, "product": key, "stratum": stratum, "sensors": ["PAZ"],
                          "offset_h": 3.0, "pass_time": "", "product_time": ""},
                 "scored_px": 10_000, "clear_px": int(clear * 10_000), "flood_px": 1000,
                 "models": {"sar": cms(sar_iou), "fus": cms(fus_iou)}}


def _scores(pairs: list[tuple[str, dict]]) -> dict:
    return {"pairs": dict(pairs), "models": {"sar": {"name": "SAR U-Net"}, "fus": {"name": "Fusion"}}}


def test_aois_pool_per_activation_and_excluded_or_other_strata_drop_out() -> None:
    scores = _scores([_pair("EMSR700", 1), _pair("EMSR700", 2), _pair("EMSR664", 1), _pair("EMSR701", 1, stratum="optical")])
    units = activations(scores, _spec(), stratum="sar")
    assert list(units) == ["EMSR700"]
    assert units["EMSR700"]["scored_px"] == 20_000 and units["EMSR700"]["models"]["sar"][0][0] == 2000


def test_verdicts_need_eight_units_and_holm_covers_two_sided_tests_only() -> None:
    n = MIN_UNITS_FOR_VERDICT
    pairs = [_pair(f"EMSR{700 + i}", 1, clear=0.9) for i in range(n)] + \
            [_pair(f"EMSR{800 + i}", 1, clear=0.0, fus_iou=0.41) for i in range(3)]
    units = activations(_scores(pairs), _spec(), stratum="sar")
    results = {r["id"]: r for r in run_tests(units, _spec())}
    assert results["P1"]["k"] == n and results["P1"]["wins"] == n and results["P1"]["verdict"].startswith("significant")
    assert results["P2"]["k"] == 3 and results["P2"]["verdict"].startswith("none: K = 3")
    assert results["P4"]["k"] == n + 3 and "holm_p" in results["P4"] and "holm_p" not in results["P2"]
    # Holm over the two eligible two-sided tests: the smaller p (P4, K = 11) is doubled, the other kept.
    assert results["P4"]["holm_p"] == pytest.approx(2 * results["P4"]["p"])
    assert results["P1"]["holm_p"] == pytest.approx(max(results["P1"]["p"], results["P4"]["holm_p"]))


def test_non_inferiority_is_judged_by_the_lower_bound() -> None:
    worse = [_pair(f"EMSR{700 + i}", 1, fus_iou=0.40 - 0.002 * (i % 3)) for i in range(MIN_UNITS_FOR_VERDICT)]
    spec = _spec(tests=[GapTest("P2", "R2 holds", "sar", "fus", "sar_only", margin=0.05)])
    [r] = run_tests(activations(_scores(worse), spec, stratum="sar"), spec)
    assert r["verdict"].startswith("non-inferior") and r["ci"][0] > -0.05
    much_worse = [_pair(f"EMSR{700 + i}", 1, fus_iou=0.30) for i in range(MIN_UNITS_FOR_VERDICT)]
    [r] = run_tests(activations(_scores(much_worse), spec, stratum="sar"), spec)
    assert r["verdict"].startswith("not shown non-inferior")


def test_report_is_written_from_the_scores_alone(tmp_path) -> None:
    pairs = [_pair(f"EMSR{700 + i}", 1, clear=0.9) for i in range(MIN_UNITS_FOR_VERDICT)]
    result = report(_scores(pairs), _spec(), tmp_path / "cems.md")
    text = (tmp_path / "cems.md").read_text(encoding="utf-8")
    assert "| P1 |" in text and "Fusion − SAR U-Net" in text and "EMSR700" in text
    assert json.loads((tmp_path / "cems.json").read_text(encoding="utf-8"))["units"] == result["units"]


def test_spec_refuses_tests_on_unknown_models(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("models: [{key: a, name: A, runs: [x]}]\ntests: [{id: P1, question: q, a: a, b: zzz}]\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="unknown models"):
        load_spec(path)
