r"""floodfusion Live: a flood map of any place and date from Sentinel-1, made by a frozen model.

Run::

    uv run --frozen python -m floodfusion.live map --bbox 11.8 44.15 12.2 44.3 --date 2023-05-17 --out maps/forli

One map, tile by tile. The tiles are the CEMS benchmark's (0.2° × 0.15°, :func:`floodfusion.case_study.tiles_for`), so
the model reads exactly what it was scored on:

1. **Ingest** the Sentinel-1 pass of that date and its same-orbit pre-event reference, with no label: a new flood
   has no reference map (:func:`floodfusion.data.ingest.ingest_event`, Planetary Computer).
2. **Sidecars:** build the layers the model reads beside its stack (DEM, seasonal reference).
3. **Predict** with the frozen run, calibrated as in the blind test: windows without usable optical are read with the
   model's SAR-only calibration (:func:`floodfusion.case_study.recalibrate_without_optical`).
4. **Write** the GFM-style COG layers and STAC item (:func:`floodfusion.products.write_package`), plus the flood
   polygons in longitude/latitude (``flood.geojson``) for the live page.

The model is loaded from its run folder alone: ``run.json``, ``best.pt`` and ``eval_test.json``, plus the SAR-only
sibling's ``eval_test.json``. Those are the four files per seed that ``reports/v12_freeze.json`` pins; no training
data is needed. A tile without a usable pass is recorded in ``summary.json`` (``dropped`` / ``failed``), as the
benchmark records its tiles. A finished tile is not ingested again.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.case_study import tiles_for
from floodfusion.cems_benchmark import OPTICAL_WINDOW_DAYS
from floodfusion.data.archive import IngestSources as Sources
from floodfusion.data.http import redact
from floodfusion.data.ingest import EventConfig, ingest_event
from floodfusion.data.stac import NoMatchingPairError

__all__ = ["DEFAULT_RUN", "MAX_TILES", "Predictor", "Sources", "check_bbox", "flood_polygons", "load_predictor",
           "main", "map_area"]

LOGGER = logging.getLogger(__name__)

#: v1.2's frozen model, first seed (``reports/v12_freeze.json``).
DEFAULT_RUN = Path("runs/v12/final/all/all/row21_fusion_context_seasonal/seed42")
#: One map covers at most this many 0.2° × 0.15° tiles (about 1° × 1°): a larger box is a mistyped one.
MAX_TILES = 40
_SIDECAR_BUILDERS = ("dem", "seasonal")


# ---------------------------------------------------------------------------
# The frozen model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Predictor:
    """A frozen run, ready to predict a stack: network, calibrations and decision threshold."""

    run: Path
    cfg: Any
    model: Any
    calibration: Mapping[str, float]
    sar_calibration: Mapping[str, float] | None
    tau: float
    force_sar_only: bool
    record: Mapping[str, Any]

    @property
    def architecture(self) -> str:
        return str(self.cfg.model.architecture)

    def predict(self, stack: Path, device: Any) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated flood probability per pixel (NaN where nothing was read) and where optical was used."""
        from floodfusion.case_study import recalibrate_without_optical
        from floodfusion.predict import predict_stack

        prob, optical_used = predict_stack(self.model, self.cfg, stack, temperature=self.calibration["temperature"],
                                           offset=self.calibration["offset"], device=device,
                                           force_sar_only=self.force_sar_only)
        if self.sar_calibration is not None:
            prob = recalibrate_without_optical(prob, optical_used, self.calibration, self.sar_calibration)
        return prob, optical_used


def load_predictor(run_dir: Path, *, device: Any) -> Predictor:
    """The frozen run in ``run_dir``, from its own files only (no training data)."""
    import torch

    from floodfusion.case_study import sar_only_calibration
    from floodfusion.config import config_from_dict
    from floodfusion.models.zoo import build_model
    from floodfusion.runs import load_run_record

    run_dir = Path(run_dir)
    record = load_run_record(run_dir)
    if record.get("status") != "finished":
        raise ValueError(f"{run_dir}: run status is {record.get('status')!r}, not 'finished'")
    cfg = config_from_dict(record["config"])
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)  # never unpickle code
    if state.get("config_sha256") != record["config_sha256"]:
        raise ValueError(f"{run_dir / 'best.pt'} belongs to another configuration than its run.json")
    model = build_model(cfg.model, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    evaluation = json.loads((run_dir / "eval_test.json").read_text(encoding="utf-8"))
    return Predictor(run=run_dir, cfg=cfg, model=model, calibration=evaluation["calibration"],
                     sar_calibration=sar_only_calibration(run_dir),
                     tau=float(evaluation.get("tau_star", cfg.eval.decision_threshold)),
                     force_sar_only=bool(evaluation.get("force_sar_only")), record=record)


# ---------------------------------------------------------------------------
# One map
# ---------------------------------------------------------------------------
def check_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` in degrees, or ValueError."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must be west south east north, got {list(bbox)}")
    west, south, east, north = (float(v) for v in bbox)
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError(f"bbox must be west < east in [-180, 180] and south < north in [-90, 90], got {list(bbox)}")
    return west, south, east, north


def _build_sidecars(event_dir: Path, names: Sequence[str], client: Any) -> None:
    from floodfusion.data.seasonal import seasonal_reference
    from floodfusion.data.sidecars import build_dem

    for name in names:
        if name == "dem":
            build_dem(event_dir, client)
        elif name == "seasonal":
            seasonal_reference(event_dir, client)


def flood_polygons(prob: np.ndarray, *, tau: float, transform: Any, epsg: int) -> dict[str, Any]:
    """The pixels mapped as flood (``prob > tau``) as a GeoJSON FeatureCollection in longitude/latitude."""
    import shapely
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape

    mask = np.isfinite(prob) & (np.nan_to_num(prob, nan=0.0) > tau)
    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    features = []
    for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        polygon = shape(geometry)
        lonlat = shapely.transform(polygon, to_lonlat, interleaved=False)
        features.append({"type": "Feature", "geometry": mapping(lonlat),
                         "properties": {"area_km2": round(polygon.area / 1e6, 6)}})
    return {"type": "FeatureCollection", "features": features}


def _package_tile(stack: Path, prob: np.ndarray, optical_used: np.ndarray, predictor: Predictor,
                  out_dir: Path) -> dict[str, Any]:
    """The tile's COG layers, STAC item and flood polygons; the areas mapped."""
    from importlib.metadata import version

    import rasterio

    from floodfusion.data.encoding import ChipBand
    from floodfusion.predict import build_flags
    from floodfusion.products import write_package

    with rasterio.open(stack) as src:
        validity = src.read(int(ChipBand.VALIDITY) + 1)
        slope = src.read(int(ChipBand.SLOPE) + 1).astype(np.float64) / 100.0
        landcover = src.read(int(ChipBand.LANDCOVER) + 1)
        profile = {k: v for k, v in src.profile.items() if k in ("driver", "width", "height", "crs", "transform")}
        epsg, pixel_m2 = int(src.crs.to_epsg()), abs(src.transform.a * src.transform.e)
    cfg = predictor.cfg
    flags = build_flags(validity, slope, landcover, prob, optical_used, slope_max_deg=cfg.baseline.slope_max_deg,
                        tau=predictor.tau, cost_ratio=cfg.eval.cost_ratio)
    meta = json.loads(stack.with_name("event.json").read_text(encoding="utf-8"))
    write_package(out_dir, prob, flags, tau=predictor.tau, profile=profile, meta=meta,
                  weights=predictor.run / "best.pt",
                  provenance={"git_sha": predictor.record.get("git_sha"),
                              "config_sha256": predictor.record["config_sha256"], "run": str(predictor.run),
                              "calibration": dict(predictor.calibration), "sar_only": predictor.force_sar_only,
                              "version": version("floodfusion")})
    valid = validity == 0
    mapped = np.where(valid, prob, np.nan)
    (out_dir / "flood.geojson").write_text(json.dumps(flood_polygons(mapped, tau=predictor.tau,
                                                                     transform=profile["transform"], epsg=epsg)),
                                           encoding="utf-8")
    flood = valid & np.isfinite(prob) & (np.nan_to_num(prob, nan=0.0) > predictor.tau)
    return {"flood_km2": round(float(flood.sum()) * pixel_m2 / 1e6, 4),
            "valid_km2": round(float(valid.sum()) * pixel_m2 / 1e6, 4), "pass_time": meta.get("s1_post_datetime")}


def map_area(bbox: Sequence[float], when: date, out: Path, *, predictor: Any, name: str, cfg: Any, sources: Sources,
             device: Any, relative_orbit: int | None = None, orbit_state: str | None = None,
             reference_before: date | None = None) -> dict[str, Any]:
    """Map ``bbox`` on the Sentinel-1 pass of ``when``; write ``out/summary.json`` and return it.

    ``relative_orbit`` and ``orbit_state`` pin the pass; ``reference_before`` makes the pre-event reference predate it.
    A date that could show a blind flood is refused while blind floods are off (:mod:`floodfusion.live_rules`).
    """
    from floodfusion.live_rules import blind_floods_enabled, check_date

    check_date(when, enabled=blind_floods_enabled(), optical_window_days=OPTICAL_WINDOW_DAYS)
    tiles = tiles_for(check_bbox(bbox))
    if len(tiles) > MAX_TILES:
        raise ValueError(f"the box needs {len(tiles)} tiles (more than {MAX_TILES}): map a smaller box")
    from floodfusion.data.sidecars import sidecars_for

    names = sidecars_for(predictor.architecture)
    unsupported = sorted(set(names) - set(_SIDECAR_BUILDERS))
    if unsupported:
        raise ValueError(f"{predictor.architecture} reads sidecars {unsupported}, which a live map cannot build")
    out = Path(out)
    summary: dict[str, Any] = {"name": name, "bbox": list(check_bbox(bbox)), "date": when.isoformat(),
                               "run": str(getattr(predictor, "run", "")), "tau": predictor.tau,
                               "tiles": {}, "skipped": {}}
    for tile_id, tile_bbox in tiles:
        event = EventConfig(event_id=f"{name}_{tile_id}", bbox=tuple(tile_bbox), flood_start=when, flood_end=when,
                            optical_window_days=OPTICAL_WINDOW_DAYS, relative_orbit=relative_orbit,
                            orbit_state=orbit_state, reference_before=reference_before)
        event_dir = out / "events" / event.event_id
        if not (event_dir / "event.json").is_file():
            try:
                ingest_event(event, out, cfg, sources=sources)
            except NoMatchingPairError as exc:
                summary["skipped"][tile_id] = f"dropped: {exc}"
                continue
            except Exception as exc:  # noqa: BLE001 - one tile must not abort the map
                summary["skipped"][tile_id] = f"failed: {type(exc).__name__}: {redact(str(exc))[:300]}"
                continue
        _build_sidecars(event_dir, names, sources.client)
        stack = event_dir / "stack.tif"
        prob, optical_used = predictor.predict(stack, device)
        summary["tiles"][tile_id] = {"bbox": list(tile_bbox),
                                     **_package_tile(stack, prob, optical_used, predictor, out / "maps" / tile_id)}
        LOGGER.info("%s %s: %.2f km² of flood", name, tile_id, summary["tiles"][tile_id]["flood_km2"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="floodfusion Live: a flood map of any place and date.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("map", help="map one box on the Sentinel-1 pass of one date")
    one.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    one.add_argument("--date", type=date.fromisoformat, required=True, help="the pass date, YYYY-MM-DD (UTC)")
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--name", default="map", help="prefix of the tiles' stack folders")
    one.add_argument("--run", type=Path, default=DEFAULT_RUN, help="a frozen run folder")
    one.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="the ingest configuration")
    one.add_argument("--device", default="cpu")
    one.add_argument("--orbit", type=int, default=None, help="pin the pass: its relative orbit")
    one.add_argument("--orbit-state", choices=("ascending", "descending"), default=None)
    one.add_argument("--reference-before", type=date.fromisoformat, default=None,
                     help="the pre-event reference must predate this date")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from floodfusion.config import load_config
    from floodfusion.train import resolve_device

    device = resolve_device(args.device)
    summary = map_area(args.bbox, args.date, args.out, predictor=load_predictor(args.run, device=device),
                       name=args.name, cfg=load_config(args.config), sources=Sources.open(), device=device,
                       relative_orbit=args.orbit, orbit_state=args.orbit_state, reference_before=args.reference_before)
    total = sum(t["flood_km2"] for t in summary["tiles"].values())
    LOGGER.info("%d tiles mapped, %d skipped: %.2f km² of flood -> %s", len(summary["tiles"]), len(summary["skipped"]),
                total, args.out / "summary.json")
    return 0 if summary["tiles"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
