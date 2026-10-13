r"""v1.2 scoring on the CEMS benchmark: v1.1's counts, plus per-pair score histograms and Copernicus GFM as a model.

The v1.1 scorer (:mod:`floodfusion.cems_scores`) and its committed reports stay untouched. This module scores
the same pairs, the same tiles and the same pixels, and keeps three things:

- **v1.1's confusion counts**, at τ* and on the oracle grid, under ``models`` with v1.1's exact layout. So
  :func:`floodfusion.cems_scores.units` and ``run_tests`` read v1.2 scores unchanged. Regression check: for the
  v1.1 models, these counts must equal the committed ``reports/cems_benchmark_scores.json``.
- **Score histograms** (:mod:`floodfusion.histograms`) under ``hist``, a separate key, because ``units()`` sums
  everything under ``models``. Per model and seed there are three: flood and dry pixels of the scored region, and
  every valid pixel of the scene, unlabelled, for the EM self-calibration.
- **GFM** (:mod:`floodfusion.data.gfm`), scored like any model:
  - its majority-vote ensemble flood extent is a 0/1 map with no oracle grid;
  - its ensemble likelihood (0–100, rescaled to [0, 1]) provides its histograms, and so its AP;
  - pixels without GFM (nodata 255, or a stack without a sidecar) are **not** counted as misses. They are counted
    in ``gfm_missing_px``, and a unit enters a GFM comparison only if GFM covers at least
    :data:`GFM_MIN_COVERAGE` of its scored pixels (``reports/v1_2_deviations.md``).

Products: © European Union, Copernicus Emergency Management Service (CEMS rapid mapping and GFM).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from floodfusion import histograms

__all__ = ["GFM_KEY", "GFM_MIN_COVERAGE", "add_tile", "gfm_maps", "tile_counts", "tile_histograms"]

#: The model key GFM is scored under.
GFM_KEY = "gfm"
#: A unit counts in a GFM comparison only if GFM covers this share of its scored pixels.
GFM_MIN_COVERAGE = 0.95
def tile_counts(prob: np.ndarray, tau: float, calibrated: bool, label: np.ndarray,
                scored: np.ndarray) -> list[list[int]]:
    """v1.1's ``[[tp, fp, fn, tn], ...]`` at τ* then on its oracle grid: v1.1's own function, so they cannot drift."""
    from floodfusion.cems_scores import _cms

    return _cms(prob, tau, calibrated, label, scored)


def tile_histograms(prob: np.ndarray, label: np.ndarray, scored: np.ndarray,
                    valid: np.ndarray) -> dict[str, np.ndarray]:
    """``pos`` / ``neg`` of the scored region and ``all`` valid pixels: histograms of ``prob`` (NaN dropped)."""
    return {"pos": histograms.histogram(prob, scored & label), "neg": histograms.histogram(prob, scored & ~label),
            "all": histograms.histogram(prob, valid)}


def gfm_maps(layers: Mapping[str, np.ndarray] | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``(flood, likelihood)`` as float maps from a GFM sidecar: 0/1 and [0, 1], NaN where GFM has no value."""
    if layers is None:
        nan = np.full(shape, np.nan)
        return nan, nan.copy()
    flood_raw, like_raw = layers["ensemble_flood"], layers["ensemble_likelihood"]
    flood = np.where(flood_raw == 255, np.nan, (flood_raw == 1).astype(np.float64))
    likelihood = np.where(like_raw == 255, np.nan, like_raw.astype(np.float64) / 100.0)
    # A pixel GFM classified but gave no likelihood for (or the reverse) has no comparable score: drop both.
    missing = np.isnan(flood) | np.isnan(likelihood)
    return np.where(missing, np.nan, flood), np.where(missing, np.nan, likelihood)


def add_tile(entry: dict[str, Any], key: str, run: str, counts: list[list[int]],
             hist: Mapping[str, np.ndarray]) -> None:
    """Accumulate one tile's counts and histograms into a pair ``entry`` (in place: the entry is being built)."""
    per_seed = entry.setdefault("models", {}).setdefault(key, {})
    previous = per_seed.get(run)
    per_seed[run] = counts if previous is None else [[a + b for a, b in zip(u, v)] for u, v in zip(previous, counts)]
    stored = entry.setdefault("hist", {}).setdefault(key, {}).get(run)
    entry["hist"][key][run] = {name: (np.asarray(h, dtype=np.int64) if stored is None
                                      else np.asarray(stored[name], dtype=np.int64) + h).tolist()
                               for name, h in hist.items()}
