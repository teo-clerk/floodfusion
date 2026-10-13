"""v1.2's confirmatory analysis on synthetic blind-test scores (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floodfusion import blind_analysis as ba
from floodfusion import cems_scores_v12 as v12
from floodfusion import prospective as pr

SPEC = {"exclude": (), "primary_stratum": "sar"}
TESTS = [{"id": "P5", "a": "tm", "b": "ours", "also_ap": True},
         {"id": "P6", "a": "gfm", "b": "ours", "gfm_covered": True},
         {"id": "P7", "a": "v11", "b": "ours"}]
SEEDS = ("seed42", "seed43", "seed44")


def _prob(label: np.ndarray, noise: float, rng) -> np.ndarray:
    """Scores that separate flood from dry less well as ``noise`` grows."""
    return np.clip(np.where(label, 0.75, 0.15) + rng.normal(0, noise, label.shape), 0.0, 1.0)


def _scores(n: int = 12, noise: dict | None = None, gfm_units: int = 12, seed: int = 0) -> dict:
    noise = noise or {"ours": 0.10, "tm": 0.30, "v11": 0.30, "gfm": 0.30}
    rng = np.random.default_rng(seed)
    pairs, activations = {}, {}
    for i in range(n):
        code = f"EMSR{900 + i}"
        label = np.zeros((40, 40), bool)
        label[:, : 10 + i] = True
        scored = valid = np.ones_like(label)
        entry = {"pair": {"activation": code, "stratum": "sar"}, "scored_px": int(scored.sum()),
                 "flood_px": int(label.sum()), "gfm_missing_px": 0 if i < gfm_units else int(scored.sum())}
        for key in ("ours", "tm", "v11"):
            for s in SEEDS:
                prob = _prob(label, noise[key], rng)
                v12.add_tile(entry, key, s, v12.tile_counts(prob, 0.5, True, label, scored),
                             v12.tile_histograms(prob, label, scored, valid))
        gfm = _prob(label, noise["gfm"], rng)
        v12.add_tile(entry, "gfm", "gfm", v12.tile_counts((gfm > 0.5).astype(float), 0.5, False, label, scored),
                     v12.tile_histograms(gfm, label, scored, valid))
        pairs[f"{code}_AOI01"] = entry
        lon = 10 * i - 60                                                  # 10° apart: every activation its own cluster
        activations[code] = {"eventTime": f"2027-0{1 + i % 9}-10T00:00:00Z",
                             "extent": f"POLYGON (({lon} 0, {lon + 1} 0, {lon + 1} 1, {lon} 1, {lon} 0))"}
    priors = {k: {s: {"tau": 0.5, "pi0": 0.3} for s in SEEDS} for k in ("ours", "tm", "v11")}
    return {"pairs": pairs, "activations": activations, "priors": priors,
            "models": {k: {"name": k} for k in ("ours", "tm", "v11", "gfm")}}


def test_a_clearly_better_model_is_supported_and_too_few_gfm_units_give_no_verdict() -> None:
    result = ba.verdicts(_scores(gfm_units=7), SPEC, TESTS)
    assert result["P5"]["verdict"] == "supported" and result["P7"]["verdict"] == "supported"
    assert result["P6"]["verdict"] == "no verdict" and result["P6"]["activation"]["k"] == 7
    p5 = result["P5"]["activation"]
    assert p5["delta"] > 0 and p5["delta_ap"] > 0 and p5["p"] == max(p5["p_iou"], p5["p_ap"])
    assert p5["k"] == 12 and result["P5"]["cluster"]["k"] == 12           # 10° apart: one cluster each


def test_a_clearly_worse_model_is_refuted() -> None:
    worse = {"ours": 0.35, "tm": 0.08, "v11": 0.08, "gfm": 0.08}
    result = ba.verdicts(_scores(noise=worse), SPEC, TESTS)
    assert {r["verdict"] for r in result.values()} == {"refuted"}


def test_p5_needs_the_threshold_free_win_too() -> None:
    scores = _scores()
    for entry in scores["pairs"].values():                                # TerraMind ranks perfectly, mis-thresholded
        for s in SEEDS:
            entry["hist"]["tm"][s]["pos"] = [0] * 257 + [entry["flood_px"]]
            entry["hist"]["tm"][s]["neg"] = [entry["scored_px"] - entry["flood_px"]] + [0] * 257
    p5 = ba.verdicts(scores, SPEC, TESTS)["P5"]
    assert p5["activation"]["delta"] > 0 and p5["activation"]["delta_ap"] < 0
    assert p5["verdict"] == "not supported"


def test_the_reveal_reads_only_scores_the_ledger_vouches_for(tmp_path: Path) -> None:
    full = _scores()
    ledger, sealed = pr.Ledger(tmp_path / "ledger.jsonl"), tmp_path / "sealed"
    for key, entry in full["pairs"].items():
        code = entry["pair"]["activation"]
        part = {"pairs": {key: entry}, "activations": {code: full["activations"][code]}, "priors": full["priors"],
                "models": full["models"]}
        sha = pr.seal(json.loads(json.dumps(part)), sealed, key)
        ledger.append({"key": key, "activation": code, "event_date": "2027-01-10", "status": "scored",
                       "stratum": "sar", "sealed_sha256": sha})
    merged = ba.merge_sealed(ledger, sealed)
    assert set(merged["pairs"]) == set(full["pairs"]) and set(merged["activations"]) == set(full["activations"])

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("primary_stratum: sar\nexclude: []\ntests: []\nmodels:\n"
                         + "".join(f"  - {{key: {k}, name: {k}, runs: []}}\n" for k in ("ours", "tm", "v11", "gfm"))
                         + "blind_tests:\n" + "".join(f"  - {json.dumps(t)}\n" for t in TESTS))
    out = tmp_path / "verdicts.md"
    assert ba.main(["reveal", "--spec", str(spec_path), "--ledger", str(ledger.path), "--sealed", str(sealed),
                    "--out", str(out)]) == 0
    assert json.loads(out.with_suffix(".json").read_text())["P7"]["verdict"] == "supported"
    assert "| P5 | ours − tm | activation | 12 |" in out.read_text()

    first = sorted(sealed.glob("*.json"))[0]
    first.write_text(first.read_text().replace('"stratum": "sar"', '"stratum": "sar" '))
    with pytest.raises(ValueError, match="do not match the ledger"):
        ba.merge_sealed(ledger, sealed)


def test_a_spec_must_name_known_models_and_some_tests(tmp_path: Path) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text("models:\n  - {key: ours, name: o, runs: []}\nblind_tests:\n  - {id: P5, a: tm, b: ours}\n")
    with pytest.raises(ValueError, match="unknown models"):
        ba.load_blind_tests(spec)
    spec.write_text("models:\n  - {key: ours, name: o, runs: []}\n")
    with pytest.raises(ValueError, match="no blind_tests"):
        ba.load_blind_tests(spec)
