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

import argparse
import gzip
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion import histograms

__all__ = [
    "GFM_KEY",
    "GFM_MIN_COVERAGE",
    "add_tile",
    "check_against_v11",
    "gfm_maps",
    "load_scores",
    "main",
    "model_priors",
    "pair_counts",
    "score",
    "summary",
    "tile_counts",
    "tile_histograms",
    "tile_masks",
    "write_scores",
]

LOGGER = logging.getLogger(__name__)

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


def tile_masks(stack: Path, pair: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(valid, scored, label, clear)`` of one tile, built exactly as v1.1's ``cems_scores._tile_masks`` builds its
    last three, plus ``valid``: the whole scene's valid pixels, for the unlabelled histogram."""
    from floodfusion.config import ValidityFlag
    from floodfusion.data.chipping import CLEAR_QUALITY, QUALITY_SCALE, read_event_stack
    from floodfusion.data.encoding import ChipBand
    from floodfusion.data.ingest import rasterise_vector

    raw, grid, _ = read_event_stack(stack)
    valid = (raw[ChipBand.VALIDITY].astype(np.int64) & ~int(ValidityFlag.OUTSIDE_AOI)) == 0
    scored = valid & rasterise_vector(Path(pair["region"]), grid)
    label = rasterise_vector(Path(pair["label"]), grid)
    has_optical = bool(json.loads(stack.with_name("event.json").read_text(encoding="utf-8")).get("has_optical"))
    clear = (raw[ChipBand.OPTICAL_QUALITY] >= int(round(CLEAR_QUALITY * QUALITY_SCALE))) if has_optical \
        else np.zeros(scored.shape, dtype=bool)
    return valid, scored, label, clear


def model_priors(spec: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """``{model: {seed: {"tau": τ*, "pi0": deployment prior}}}`` from each run's ``eval_test.json``."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for m in spec["models"]:
        if m.kind != "learned":
            continue
        for run in m.runs:
            ev = json.loads((Path(run) / "eval_test.json").read_text(encoding="utf-8"))
            out.setdefault(m.key, {})[Path(run).name] = {"tau": float(ev["tau_star"]),
                                                         "pi0": float(ev["calibration"]["pi_deploy"])}
    return out


def score(spec: Mapping[str, Any], case_root: Path, device: Any, *, records: Path | None = None) -> dict[str, Any]:
    """v1.1's per-pair counts for every model and seed, their histograms, and GFM (if the spec names it)."""
    from floodfusion.cems_scores import _models
    from floodfusion.data.gfm import read_sidecar
    from floodfusion.train import resolve_device

    device = resolve_device(device)
    learned = [m for m in spec["models"] if m.kind != "gfm"]
    models = _models({**spec, "models": learned}, device)
    with_gfm = len(learned) < len(spec["models"])
    pairs: dict[str, Any] = {}
    for pair_file in sorted(case_root.glob("EMSR*_AOI*/pair.json")):
        pair = json.loads(pair_file.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {"pair": {k: pair[k] for k in ("activation", "aoi", "product", "stratum", "sensors",
                                                               "offset_h", "pass_time", "product_time")},
                                 "scored_px": 0, "clear_px": 0, "flood_px": 0, "gfm_missing_px": 0,
                                 "sidecars": {"stacks": 0, "dem": 0, "seasonal": 0, "gfm": 0},
                                 "models": {}, "hist": {}}
        for stack in sorted((pair_file.parent / "events").glob("*/stack.tif")):
            if not stack.with_name("event.json").is_file():
                continue
            valid, scored, label, clear = tile_masks(stack, pair)
            if not scored.any():
                continue
            # What each model could read here: a missing sidecar is read as unavailable, never as an error.
            for name, present in (("stacks", True), ("dem", (stack.parent / "dem.tif").is_file()),
                                  ("seasonal", (stack.parent / "seasonal.tif").is_file()),
                                  ("gfm", (stack.parent / "gfm.tif").is_file())):
                entry["sidecars"][name] += int(present)
            entry["scored_px"] += int(scored.sum())
            entry["clear_px"] += int((clear & scored).sum())
            entry["flood_px"] += int((label & scored).sum())
            for key, run, model in models:
                prob = model.predict(stack)
                hist = tile_histograms(prob, label, scored, valid) if model.calibrated else {}
                add_tile(entry, key, run, tile_counts(prob, model.tau, model.calibrated, label, scored), hist)
            if with_gfm:
                flood, likelihood = gfm_maps(read_sidecar(stack.parent), scored.shape)
                covered = scored & ~np.isnan(flood)
                entry["gfm_missing_px"] += int((scored & ~covered).sum())
                add_tile(entry, GFM_KEY, "gfm", tile_counts(flood, 0.5, False, label, covered),
                         tile_histograms(likelihood, label, covered, valid & ~np.isnan(likelihood)))
        pairs[pair["key"]] = entry
        LOGGER.info("%s: %d scored px, GFM missing on %d", pair["key"], entry["scored_px"], entry["gfm_missing_px"])
    events = {}
    if records is not None:
        for code in sorted({e["pair"]["activation"] for e in pairs.values()}):
            rec = json.loads((records / f"{code}.json").read_text(encoding="utf-8"))
            events[code] = {"eventTime": rec["eventTime"], "extent": rec["extent"]}
    return {"pairs": pairs, "activations": events, "priors": model_priors(spec),
            "models": {m.key: {"name": m.name, "runs": list(m.runs), "kind": m.kind, "role": m.role}
                       for m in spec["models"]}}


def write_scores(scores: Mapping[str, Any], path: Path) -> None:
    """Gzipped JSON (the histograms make it ~10 MB raw)."""
    from floodfusion.data.records import atomic_write_bytes

    atomic_write_bytes(path, gzip.compress(json.dumps(scores, allow_nan=False).encode("utf-8"), mtime=0))


def load_scores(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def check_against_v11(scores: Mapping[str, Any], v11: Mapping[str, Any]) -> list[str]:
    """Pairs, models or seeds whose counts differ from v1.1's committed scores (empty: the regression holds)."""
    bad = []
    for key, entry in v11["pairs"].items():
        for model, per_seed in entry["models"].items():
            for seed, counts in per_seed.items():
                if scores["pairs"].get(key, {}).get("models", {}).get(model, {}).get(seed) != counts:
                    bad.append(f"{key}/{model}/{seed}")
    return bad


def pair_counts(scores: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, int]:
    """Pairs of the primary stratum with scored pixels, and how many of them GFM covers (>= the minimum share)."""
    pairs = [e for e in scores["pairs"].values() if e["scored_px"] and e["pair"]["stratum"] == spec["primary_stratum"]
             and e["pair"]["activation"] not in spec["exclude"]]
    covered = [e for e in pairs if 1.0 - e.get("gfm_missing_px", 0) / e["scored_px"] >= GFM_MIN_COVERAGE]
    return {"n_pairs": len(pairs), "gfm_pairs": len(covered)}


def _seed_hists(entries: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, np.ndarray]]:
    """``{seed: {"pos", "neg", "all"}}`` of one model, pooled over a unit's pairs (seeds kept apart)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for entry in entries:
        for seed, hist in entry.get("hist", {}).get(key, {}).items():
            if not hist:                     # an uncalibrated model (a baseline's 0/1 map) keeps no histogram
                continue
            acc = out.setdefault(seed, {})
            for name, counts in hist.items():
                acc[name] = acc.get(name, 0) + np.asarray(counts, dtype=np.int64)
    return out


def _units(scores: Mapping[str, Any], spec: Mapping[str, Any], unit: str) -> dict[str, list[dict[str, Any]]]:
    """The primary stratum's pairs grouped per activation or per cluster (EMSR664 and empty pairs dropped)."""
    from floodfusion.cems_scores import activation_clusters

    cluster_of = activation_clusters(scores["activations"]) if unit == "cluster" else {}
    out: dict[str, list[dict[str, Any]]] = {}
    for entry in scores["pairs"].values():
        pair = entry["pair"]
        if pair["activation"] in spec["exclude"] or not entry["scored_px"] or pair["stratum"] != spec["primary_stratum"]:
            continue
        out.setdefault(cluster_of.get(pair["activation"], pair["activation"]), []).append(entry)
    return out


def _iou(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp + fn) if tp + fp + fn else math.nan


def _seed_metrics(hist: Mapping[str, np.ndarray], prior: Mapping[str, float] | None) -> dict[str, float]:
    """One seed's threshold-free and threshold-moved metrics; self-calibration needs the seed's τ* and prior."""
    pos, neg = hist["pos"], hist["neg"]
    row = {"ap": histograms.average_precision(pos, neg), "iou_oracle": histograms.oracle_iou(pos, neg)[0]}
    if prior is not None:
        tau = prior["tau"]
        row["iou_tau_binned"] = _iou(*histograms.confusion_above(pos, neg, tau)[:3])
        shifted = histograms.shifted_threshold(tau, histograms.em_prior_shift(hist["all"], prior["pi0"]))
        row["iou_selfcal"] = _iou(*histograms.confusion_above(pos, neg, shifted)[:3])
    return row


def _unit_set(groups: Mapping[str, list[dict[str, Any]]], *, gfm_covered: bool) -> list[str]:
    """Units with reference flood in their scored region (and, if asked, covered by GFM on >= the minimum share)."""
    out = []
    for name, entries in groups.items():
        scored = sum(e["scored_px"] for e in entries)
        if sum(e["flood_px"] for e in entries) == 0:
            continue
        if gfm_covered and (scored == 0 or 1.0 - sum(e.get("gfm_missing_px", 0) for e in entries) / scored
                            < GFM_MIN_COVERAGE):
            continue
        out.append(name)
    return sorted(out)


METRICS = ("iou_tau", "iou_tau_binned", "iou_selfcal", "iou_oracle", "ap")
#: Resamples of the percentile bootstrap over units (seeded: the report regenerates byte for byte).
N_BOOTSTRAP = 2000


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0) -> list[float]:
    """95 % percentile interval of the macro mean over units, resampling units; NaNs with fewer than two units."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(N_BOOTSTRAP, len(v)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary(scores: Mapping[str, Any], spec: Mapping[str, Any], *, unit: str = "activation",
            gfm_common: bool = False) -> dict[str, Any]:
    """Per model, macro over **one** unit set of the primary stratum.

    - **IoU at τ***: v1.1's definition, on exact pixels with seeds pooled.
    - **The histogram metrics**, per seed with that seed's own τ* and deployment prior, then averaged over seeds:
      IoU at τ* on the binned scores (the baseline the self-calibration must be read against), self-calibrated IoU,
      oracle IoU and AP.

    The unit set is units with reference flood in their scored region. With ``gfm_common``, only units GFM covers on
    at least :data:`GFM_MIN_COVERAGE` of their scored pixels, so every model is compared with GFM on the same
    floods. GFM itself is shown only in that table. Descriptive: v1.2's confirmatory tests are the blind ones.
    """
    from floodfusion.cems_scores import _add, _cm

    groups = _units(scores, spec, unit)
    names = _unit_set(groups, gfm_covered=gfm_common)
    rows: dict[str, Any] = {}
    for key in scores["models"]:
        if key == GFM_KEY and not gfm_common:
            continue
        priors = scores.get("priors", {}).get(key, {})
        per: dict[str, dict[str, float]] = {}
        for name in names:
            pooled = None
            for e in groups[name]:
                for cms in e["models"].get(key, {}).values():
                    pooled = _add(pooled, cms)
            if pooled is None:
                continue
            seeds = [_seed_metrics(h, priors.get(seed)) for seed, h in _seed_hists(groups[name], key).items()]
            row = {"iou_tau": _cm(pooled[0]).iou}
            for m in METRICS[1:]:
                values = [r[m] for r in seeds if m in r and math.isfinite(r[m])]
                row[m] = float(np.mean(values)) if values else math.nan
            per[name] = row
        means = {m: [r[m] for r in per.values() if math.isfinite(r[m])] for m in METRICS}
        rows[key] = {"name": scores["models"][key]["name"], "k": len(per), "per_unit": per,
                     **{m: (float(np.mean(v)) if v else math.nan) for m, v in means.items()},
                     "ci": {m: _bootstrap_ci(v) for m, v in means.items()}}
    return {"unit": unit, "gfm_common": gfm_common, "units": names, "models": rows}


def _markdown(tables: Sequence[Mapping[str, Any]]) -> str:
    def fmt(x: float) -> str:
        return "—" if not math.isfinite(x) else f"{x:.3f}"

    lines = ["# CEMS benchmark, v1.2 scoring (descriptive)", "",
             "Primary stratum (products drawn on radar), EMSR664 excluded; units with reference flood in their scored "
             "region. IoU at τ* is v1.1's definition (exact pixels, seeds pooled; v1.1's own table also counted units "
             "without reference flood). The other columns are per seed, with the seed's τ* and prior, then averaged: "
             "IoU at τ* on the binned scores is the baseline to read the self-calibrated IoU against. "
             "Exploratory: this benchmark has been seen.", ""]
    for table in tables:
        scope = "units covered by GFM (≥ 95 % of scored pixels): every model on the same floods" \
            if table["gfm_common"] else "all units with reference flood"
        lines += [f"## Per {table['unit']}: {scope} (K = {len(table['units'])})", "",
                  "| Model | K | IoU at τ* | IoU at τ*, binned | IoU, self-calibrated | IoU, oracle | AP |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in table["models"].values():
            lines.append(f"| {row['name']} | {row['k']} | " + " | ".join(fmt(row[m]) for m in METRICS) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v1.2 scoring on the CEMS benchmark (descriptive).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--spec", type=Path, default=Path("configs/datasets/cems_models_v12.yaml"))
    sc.add_argument("--case-root", type=Path, default=Path("data/cems_benchmark"))
    sc.add_argument("--records", type=Path, default=Path("data/cems_records"))
    sc.add_argument("--out", type=Path, default=Path("reports/v12_cems_scores.json.gz"))
    sc.add_argument("--device", default=None)
    rp = sub.add_parser("report")
    rp.add_argument("--spec", type=Path, default=Path("configs/datasets/cems_models_v12.yaml"))
    rp.add_argument("--scores", type=Path, default=Path("reports/v12_cems_scores.json.gz"))
    rp.add_argument("--out", type=Path, default=Path("reports/v12_cems_benchmark.md"))
    rp.add_argument("--v11-scores", type=Path, default=Path("reports/cems_benchmark_scores.json"),
                    help="v1.1's committed scores: the v1.1 models' counts must equal them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from floodfusion.cems_scores import load_spec
    from floodfusion.runs import write_json

    spec = load_spec(args.spec)
    if args.cmd == "score":
        write_scores(score(spec, args.case_root, args.device, records=args.records), args.out)
        return 0
    scores = load_scores(args.scores)
    bad = check_against_v11(scores, json.loads(args.v11_scores.read_text(encoding="utf-8")))
    if bad:
        raise SystemExit(f"{len(bad)} counts differ from v1.1's committed scores, e.g. {bad[:3]}")
    tables = [summary(scores, spec, unit=u, gfm_common=g) for g in (False, True) for u in ("activation", "cluster")]
    args.out.write_text(_markdown(tables) + "\n", encoding="utf-8")
    write_json(args.out.with_suffix(".json"), {"tables": tables, **pair_counts(scores, spec)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
