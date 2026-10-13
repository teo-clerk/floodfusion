"""v1.2 CEMS scoring pieces: v1.1's counts unchanged, histograms beside them, GFM as a model without fake misses."""

from __future__ import annotations

import numpy as np
import pytest

from floodfusion import cems_scores_v12 as v12
from floodfusion import histograms


def _tile(seed: int = 0, shape=(20, 30)):
    rng = np.random.default_rng(seed)
    prob = rng.random(shape)
    prob[0, :5] = np.nan                                                 # no prediction there: v1.1 reads "no flood"
    label = rng.random(shape) < 0.3
    scored = np.ones(shape, bool)
    scored[-3:] = False                                                  # outside the product's observed region
    valid = np.ones(shape, bool)
    valid[:, :2] = False
    return prob, label, scored, valid


def test_counts_are_v1_1s_at_tau_and_on_the_grid() -> None:
    prob, label, scored, _ = _tile()
    counts = v12.tile_counts(prob, 0.25, True, label, scored)
    assert len(counts) == 20                                             # tau* + 19 grid thresholds
    flag = np.nan_to_num(prob, nan=0.0) > 0.25
    assert counts[0] == [int((flag & label & scored).sum()), int((flag & ~label & scored).sum()),
                         int((~flag & label & scored).sum()), int((~flag & ~label & scored).sum())]
    assert len(v12.tile_counts(prob, 0.5, False, label, scored)) == 1    # a 0/1 map has no oracle grid


def test_histograms_split_the_scored_region_and_keep_every_valid_pixel_unlabelled() -> None:
    prob, label, scored, valid = _tile()
    h = v12.tile_histograms(prob, label, scored, valid)
    finite = ~np.isnan(prob)
    assert h["pos"].sum() == (finite & scored & label).sum() and h["neg"].sum() == (finite & scored & ~label).sum()
    assert h["all"].sum() == (finite & valid).sum()                     # beyond the scored region, without labels
    tp, fp, fn, tn = histograms.confusion_above(h["pos"], h["neg"], 0.5)
    assert tp + fn == h["pos"].sum() and fp + tn == h["neg"].sum()


def test_gfm_maps_keep_missing_pixels_missing() -> None:
    flood = np.array([[0, 1, 255], [1, 0, 0]], dtype=np.uint8)
    likelihood = np.array([[3, 90, 255], [255, 0, 40]], dtype=np.uint8)
    f, like = v12.gfm_maps({"ensemble_flood": flood, "ensemble_likelihood": likelihood}, flood.shape)
    np.testing.assert_array_equal(np.isnan(f), [[False, False, True], [True, False, False]])
    assert f[0, 1] == 1.0 and like[0, 1] == pytest.approx(0.9) and like[1, 2] == pytest.approx(0.4)
    none_f, none_like = v12.gfm_maps(None, (2, 3))
    assert np.isnan(none_f).all() and np.isnan(none_like).all()


def test_tiles_accumulate_per_model_and_seed() -> None:
    prob, label, scored, valid = _tile()
    entry: dict = {}
    for _ in range(2):
        v12.add_tile(entry, "ks_r9", "seed42", v12.tile_counts(prob, 0.25, True, label, scored),
                     v12.tile_histograms(prob, label, scored, valid))
    one = v12.tile_counts(prob, 0.25, True, label, scored)
    assert entry["models"]["ks_r9"]["seed42"] == [[2 * c for c in row] for row in one]
    assert sum(entry["hist"]["ks_r9"]["seed42"]["all"]) == 2 * int((~np.isnan(prob) & valid).sum())
    assert set(entry) == {"models", "hist"}                             # units() sums "models" only


def _case(tmp_path, n_tiles=2):
    import json

    pair = {"key": "EMSR700_AOI01", "activation": "EMSR700", "aoi": 1, "product": "P", "stratum": "sar",
            "sensors": ["ICEYE"], "offset_h": 1.0, "pass_time": "t", "product_time": "t",
            "region": "r.geojson", "label": "l.geojson"}
    root = tmp_path / "EMSR700_AOI01"
    for i in range(n_tiles):
        d = root / "events" / f"t{i}"
        d.mkdir(parents=True)
        (d / "stack.tif").write_bytes(b"")
        (d / "event.json").write_text("{}")
    (root / "pair.json").write_text(json.dumps(pair))
    return tmp_path


def test_score_loop_keeps_v1_1_counts_histograms_and_gfm_gaps(tmp_path, monkeypatch) -> None:
    from floodfusion.case_study import _Model
    from floodfusion.cems_scores import ModelSpec

    prob, label, scored, valid = _tile(3)
    monkeypatch.setattr(v12, "tile_masks", lambda stack, pair: (valid, scored, label, np.zeros_like(scored)))
    model = _Model("m", lambda stack: prob, 0.25, True)
    import floodfusion.cems_scores as v11
    import floodfusion.data.gfm as gfm_mod
    monkeypatch.setattr(v11, "_models", lambda spec, device: [("m", "seed42", model)])
    gfm_layers = {"ensemble_flood": np.where(label, 1, 0).astype(np.uint8),
                  "ensemble_likelihood": np.full(prob.shape, 60, np.uint8)}
    gfm_layers["ensemble_flood"][:, -4:] = 255                           # GFM has no value on the last columns
    monkeypatch.setattr(gfm_mod, "read_sidecar", lambda d: gfm_layers)
    spec = {"models": [ModelSpec("m", "Model", ("runs/x/seed42",)), ModelSpec("gfm", "GFM", (), "gfm")],
            "exclude": (), "primary_stratum": "sar"}
    monkeypatch.setattr(v12, "model_priors", lambda s: {"m": {"seed42": {"tau": 0.25, "pi0": 0.3}}})
    out = v12.score(spec, _case(tmp_path), "cpu")
    entry = out["pairs"]["EMSR700_AOI01"]
    assert entry["models"]["m"]["seed42"] == [[2 * c for c in r] for r in v12.tile_counts(prob, 0.25, True, label, scored)]
    covered = scored.copy()
    covered[:, -4:] = False
    assert entry["gfm_missing_px"] == 2 * int((scored & ~covered).sum())
    assert entry["models"]["gfm"]["gfm"][0][0] == 2 * int((label & covered).sum())   # GFM hits = its flood on covered
    assert entry["models"]["gfm"]["gfm"][0][2] == 0                     # gaps are not misses
    path = tmp_path / "s.json.gz"
    v12.write_scores(out, path)
    assert v12.load_scores(path)["pairs"]["EMSR700_AOI01"]["scored_px"] == entry["scored_px"]


def test_summary_applies_the_gfm_coverage_rule_and_self_calibrates() -> None:
    prob, label, scored, valid = _tile(4)
    entry_a: dict = {"pair": {"activation": "EMSR1", "stratum": "sar"}, "scored_px": int(scored.sum()),
                     "gfm_missing_px": 0}
    entry_b: dict = {"pair": {"activation": "EMSR2", "stratum": "sar"}, "scored_px": int(scored.sum()),
                     "gfm_missing_px": int(scored.sum()) // 2}            # GFM covers half: out of GFM's rows
    for entry in (entry_a, entry_b):
        for key, run in (("m", "seed42"), ("gfm", "gfm")):
            v12.add_tile(entry, key, run, v12.tile_counts(prob, 0.25, True, label, scored),
                         v12.tile_histograms(prob, label, scored, valid))
    scores = {"pairs": {"a": entry_a, "b": entry_b}, "activations": {},
              "priors": {"m": {"seed42": {"tau": 0.25, "pi0": 0.3}}},
              "models": {"m": {"name": "Model"}, "gfm": {"name": "GFM"}}}
    table = v12.summary(scores, {"exclude": (), "primary_stratum": "sar"})
    assert table["models"]["m"]["k"] == 2 and table["models"]["gfm"]["k"] == 1
    m = table["models"]["m"]
    assert 0 < m["iou_tau"] <= m["iou_oracle"] + 1e-9 and 0 < m["ap"] <= 1
    assert np.isfinite(m["iou_selfcal"]) and np.isnan(table["models"]["gfm"]["iou_selfcal"])   # GFM: no prior
    assert "| Model | 2 |" in v12._markdown([table])


def test_regression_against_v1_1_names_every_differing_count() -> None:
    v11 = {"pairs": {"a": {"models": {"m": {"seed42": [[1, 2, 3, 4]]}}}}}
    assert v12.check_against_v11({"pairs": {"a": {"models": {"m": {"seed42": [[1, 2, 3, 4]]}, "gfm": {}}}}}, v11) == []
    assert v12.check_against_v11({"pairs": {"a": {"models": {"m": {"seed42": [[1, 2, 3, 5]]}}}}}, v11) == ["a/m/seed42"]
    assert v12.check_against_v11({"pairs": {}}, v11) == ["a/m/seed42"]


def test_the_v12_spec_holds_every_v11_model_plus_the_corrected_terramind_and_gfm() -> None:
    from floodfusion.cems_scores import load_spec

    v11, spec = load_spec("configs/datasets/cems_models.yaml"), load_spec("configs/datasets/cems_models_v12.yaml")
    keys = [m.key for m in spec["models"]]
    assert keys[: len(v11["models"])] == [m.key for m in v11["models"]]
    assert keys[len(v11["models"]):] == ["s1f11_tm_fixed", "gfm"] and spec["tests"] == []
    assert {k: spec[k] for k in ("exclude", "primary_stratum", "overlapping_pairs")} == \
        {k: v11[k] for k in ("exclude", "primary_stratum", "overlapping_pairs")}
