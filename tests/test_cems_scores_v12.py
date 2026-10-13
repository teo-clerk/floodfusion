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
