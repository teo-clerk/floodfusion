r"""The deliverable product: a flood-extent GeoTIFF with its exclusion layer (R1, R4--R6).

Run as::

    python -m floodfusion.predict --run-dir runs/row09_fusion_crossattn/seed42 \
        --stack data/synthetic/events/E3_windy_plain/stack.tif --out reports/products/E3.tif

The whole event stack is predicted in 256 px windows with 64 px of context on every side
(overlap-and-crop: each pixel is predicted once, away from a window edge), with the
calibration fitted on validation by :mod:`floodfusion.evaluate` (prior correction +
temperature/Platt). Bands:

Two GeoTIFFs on the event grid (a GeoTIFF has one data type, so integer classes are never
stored as floats):

=================== ===== ======= =============================================================
file                band  dtype   content
=================== ===== ======= =============================================================
``<out>``           1     uint16  class at :math:`\tau^\star = 1/(1+r)`: 0 dry, 1 **flood**,
                                  2 **permanent water** (R1 distinguishes them), 255 excluded
                                  (no data, layover, shadow, outside the mapped AOI — no
                                  information there, docs/03 section 5.4)
``<out>``           2     uint16  flags (bit: meaning) — 0 NODATA, 1 LAYOVER, 2 SHADOW,
                                  3 PERMANENT_WATER, 4 OUTSIDE_AOI, 5 STEEP_SLOPE
                                  (> ``baseline.slope_max_deg``), 6 DENSE_FOREST (WorldCover 10),
                                  7 DENSE_URBAN (WorldCover 50), 8 LOW_CONFIDENCE, 9 OPTICAL_USED
``<out>_prob.tif``  1     float32 calibrated flood probability :math:`\sigma((z + c)/T)`; NaN = none
=================== ===== ======= =============================================================

Forest, urban and steep-slope pixels keep their classification but are **flagged, not
removed**: C-band does not see water under a closed canopy or between buildings
(``docs/01`` sections 7--8), so "dry" there means *not reliably mapped*, never *verified dry*
(``docs/09`` section 8.1). LOW_CONFIDENCE marks pixels whose decision flips if the cost ratio
changes by a factor 1.5 either way — :math:`p \in [1/(1 + 1.5r), 1/(1 + r/1.5)]`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PRODUCT_FLAGS", "build_flags", "main", "predict_stack", "write_product"]

LOGGER = logging.getLogger(__name__)

PRODUCT_FLAGS = {"NODATA": 0, "LAYOVER": 1, "SHADOW": 2, "PERMANENT_WATER": 3, "OUTSIDE_AOI": 4,
                 "STEEP_SLOPE": 5, "DENSE_FOREST": 6, "DENSE_URBAN": 7, "LOW_CONFIDENCE": 8, "OPTICAL_USED": 9}
CLASS_DRY, CLASS_FLOOD, CLASS_PERMANENT_WATER, CLASS_EXCLUDED = 0, 1, 2, 255
_WINDOW = 256
_CONFIDENCE_FACTOR = 1.5


def _window_records(stack_path: Path, height: int, width: int, has_optical: bool, meta: dict[str, Any]) -> list[Any]:
    from floodfusion.data.records import ChipRecord

    return [ChipRecord(chip_id=f"product_r{r:05d}_c{c:05d}", basin_id="product", event_id=str(meta.get("event_id", "")),
                       raster_path=stack_path, window=(c, r, _WINDOW, _WINDOW), positive_fraction=0.0,
                       valid_fraction=1.0, optical_available=has_optical,
                       relative_orbit=int(meta.get("relative_orbit", 0)), orbit_state=str(meta.get("orbit_state", "")))
            for r in range(0, height, _WINDOW) for c in range(0, width, _WINDOW)]


def predict_stack(model: Any, cfg: Any, stack_path: Path, *, temperature: float, offset: float, device: Any,
                  force_sar_only: bool = False, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probability ``(H, W)`` over the whole stack, and where optical was used."""
    import rasterio
    import torch

    from floodfusion.data.dataset import FloodChipDataset, build_dataloader, iter_batches_to_device
    from floodfusion.train import EVAL_CONTEXT_PX, _autocast, model_inputs

    with rasterio.open(stack_path) as src:
        height, width = src.height, src.width
    meta_path = stack_path.with_name("event.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    has_optical = bool(meta.get("has_optical", True)) and not force_sar_only
    records = _window_records(stack_path, height, width, has_optical, meta)
    data_cfg = replace(cfg.data, chip_source="raster")  # a product is never made of simulated pixels
    loader = build_dataloader(FloodChipDataset(records, cfg.norm, data_cfg, cfg.grid, context_px=EVAL_CONTEXT_PX),
                              data_cfg, batch_size=batch_size, shuffle=False, drop_last=False)
    prob = np.full((height, width), np.nan, dtype=np.float32)
    optical_used = np.zeros((height, width), dtype=bool)
    c = EVAL_CONTEXT_PX
    model.eval()
    with torch.inference_mode():
        for batch in iter_batches_to_device(loader, device, channels_last=cfg.train.channels_last):
            with _autocast(device, cfg):
                logits, _ = model(**model_inputs(batch, batch["optical_available"], force_sar_only=force_sar_only))
            z = logits.float()[:, 0, c:-c, c:-c]
            p = torch.sigmoid((z + offset) / temperature).cpu().numpy()
            valid = batch["valid"][:, 0, c:-c, c:-c].cpu().numpy()
            used = batch["optical_available"].cpu().numpy()
            for k, chip_id in enumerate(batch["chip_id"]):
                row, col = int(chip_id.split("_r")[1][:5]), int(chip_id.split("_c")[1][:5])
                h, w = min(_WINDOW, height - row), min(_WINDOW, width - col)
                prob[row:row + h, col:col + w] = np.where(valid[k, :h, :w], p[k, :h, :w], np.nan)
                optical_used[row:row + h, col:col + w] = bool(used[k])
    return prob, optical_used


def build_flags(validity: np.ndarray, slope_deg: np.ndarray, landcover: np.ndarray, prob: np.ndarray,
                optical_used: np.ndarray, *, slope_max_deg: float, tau: float, cost_ratio: float) -> np.ndarray:
    from floodfusion.data.encoding import WORLDCOVER_BUILT_UP, WORLDCOVER_TREE

    flags = (np.asarray(validity).astype(np.uint16) & 0x1F)
    lo, hi = 1.0 / (1.0 + _CONFIDENCE_FACTOR * cost_ratio), 1.0 / (1.0 + cost_ratio / _CONFIDENCE_FACTOR)
    if not lo < tau < hi:
        raise ValueError(f"tau={tau} lies outside its own low-confidence band [{lo:.3f}, {hi:.3f}]")
    for name, mask in (("STEEP_SLOPE", np.nan_to_num(slope_deg, nan=0.0) > slope_max_deg),
                       ("DENSE_FOREST", landcover == WORLDCOVER_TREE),
                       ("DENSE_URBAN", landcover == WORLDCOVER_BUILT_UP),
                       ("LOW_CONFIDENCE", np.isfinite(prob) & (prob >= lo) & (prob <= hi)),
                       ("OPTICAL_USED", optical_used)):
        flags |= (mask.astype(np.uint16) << PRODUCT_FLAGS[name])
    return flags


def product_classes(prob: np.ndarray, flags: np.ndarray, tau: float) -> np.ndarray:
    """0 dry / 1 flood / 2 permanent water / 255 excluded (no information)."""
    no_info = (flags & sum(1 << PRODUCT_FLAGS[k] for k in ("NODATA", "LAYOVER", "SHADOW", "OUTSIDE_AOI"))) > 0
    permanent = (flags & (1 << PRODUCT_FLAGS["PERMANENT_WATER"])) > 0
    classes = np.where(np.nan_to_num(prob, nan=0.0) > tau, CLASS_FLOOD, CLASS_DRY)
    classes = np.where(permanent, CLASS_PERMANENT_WATER, classes)
    return np.where(no_info | (~np.isfinite(prob) & ~permanent), CLASS_EXCLUDED, classes).astype(np.uint16)


def write_product(path: Path, prob: np.ndarray, flags: np.ndarray, *, tau: float, profile: dict[str, Any],
                  tags: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``<path>`` (class + flags, uint16) and ``<path stem>_prob.tif`` (float32)."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    base = {**profile, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER"}
    all_tags = {**{k: str(v) for k, v in tags.items()}, "flag_bits": json.dumps(PRODUCT_FLAGS), "tau_star": str(tau)}
    prob_path = path.with_name(f"{path.stem}_prob.tif")
    outputs = (
        (path, {**base, "count": 2, "dtype": "uint16", "nodata": None},
         ((product_classes(prob, flags, tau), "class_0dry_1flood_2permwater_255excluded"), (flags, "flags_bitfield"))),
        (prob_path, {**base, "count": 1, "dtype": "float32", "nodata": float("nan")},
         ((prob.astype(np.float32), "flood_probability_calibrated"),)),
    )
    for target, prof, bands in outputs:
        tmp = target.with_name(f".{target.name}.tmp")
        with rasterio.open(tmp, "w", **prof) as dst:
            for i, (data, desc) in enumerate(bands, start=1):
                dst.write(data.astype(prof["dtype"]), i)
                dst.set_band_description(i, desc)
            dst.update_tags(**all_tags)
        tmp.replace(target)
    return path, prob_path


def _resolve_run(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    """(directory holding best.pt, evaluation holding the calibration) — row 11 points at row 9."""
    from floodfusion.runs import load_run_record

    evaluation = run_dir / "eval_test.json"
    if not evaluation.is_file():
        raise SystemExit(f"{run_dir} has no eval_test.json: run floodfusion.evaluate first (it fits the calibration)")
    record = load_run_record(run_dir)
    weights = Path(record["source_run"]) if record.get("source_run") else run_dir
    return weights, json.loads(evaluation.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    import rasterio

    from floodfusion.data.encoding import ChipBand
    from floodfusion.evaluate import _load
    from floodfusion.train import resolve_device

    parser = argparse.ArgumentParser(description="Write the flood product GeoTIFF for one event stack.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="an evaluated run (for SAR-only products: the row-11 directory)")
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    weights_dir, evaluation = _resolve_run(args.run_dir)
    sar_only = bool(evaluation.get("force_sar_only"))
    calibration = evaluation["calibration"]
    device = resolve_device(args.device)
    cfg, _data, model, record, _epoch = _load(weights_dir, None, device, sar_only)
    prob, optical_used = predict_stack(model, cfg, args.stack, temperature=calibration["temperature"],
                                       offset=calibration["offset"], device=device, force_sar_only=sar_only)
    with rasterio.open(args.stack) as src:
        validity = src.read(int(ChipBand.VALIDITY) + 1)
        slope = src.read(int(ChipBand.SLOPE) + 1).astype(np.float64) / 100.0
        landcover = src.read(int(ChipBand.LANDCOVER) + 1)
        profile = {k: v for k, v in src.profile.items() if k in ("driver", "width", "height", "crs", "transform")}
    tau = cfg.eval.decision_threshold
    flags = build_flags(validity, slope, landcover, prob, optical_used, slope_max_deg=cfg.baseline.slope_max_deg,
                        tau=tau, cost_ratio=cfg.eval.cost_ratio)
    write_product(args.out, prob, flags, tau=tau, profile=profile, tags={
        "run": str(args.run_dir), "config_sha256": record["config_sha256"], "git_sha": record.get("git_sha"),
        "calibration": json.dumps(calibration), "sar_only": sar_only,
        "note": "snapshot at the S1 acquisition time; no depth; forest/urban/steep flagged as not reliably mapped",
    })
    flood = int(np.nansum(prob > tau))
    LOGGER.info("wrote %s: %d flood pixels (%.2f km^2) at tau*=%.2f", args.out, flood, flood * 1e-4, tau)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
