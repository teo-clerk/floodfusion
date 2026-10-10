r"""The physics-based synthetic benchmark: frozen event specifications, build and CLI.

Four events, three ~210 km^2 basins each (12 basins — the "K ~ 12" of ``docs/09`` section 3), chosen
to exercise every mechanism of the ``docs/01`` failure catalogue and the optical failure
modes of ``docs/02``:

=================== ========= ======================================================= =========
event               terrain   what it stresses                                        optical
=================== ========= ======================================================= =========
E1_lowland_po       lowland   calm floods, oxbow lake, town, runway (smooth dry),     35 % cloud
                              harvested fields (change false positives)
E2_alpine_valley    mountain  cast shadow and layover on the valley floor, flooded    none
                              riparian forest (double bounce), ascending geometry
E3_windy_plain      lowland   6 m/s wind: Bragg-roughened flood water, swath-edge     clear
                              nodata, sand bars
E4_hilly_forest     hilly     forest + town flooding, persistent cloud (75 %),        75 % cloud
                              cirrus and cloud-shadow false water
=================== ========= ======================================================= =========

**These parameters are frozen before any experiment runs and are never tuned toward an
expected ablation outcome.** The synthetic benchmark validates the *pipeline* — splits,
leakage, calibration, VRAM, the table — and its numbers are never reported as real-world
accuracy.

Run::

    python -m floodfusion.data.synthetic.benchmark --out data/synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floodfusion.config import DataConfig, GridConfig, NormalizationStats, load_config
from floodfusion.data.chipping import AnalysisGrid, extract_chips, write_event_stack
from floodfusion.data.records import ChipRecord, write_manifest
from floodfusion.data.synthetic.landscape import EventSpec
from floodfusion.data.synthetic.scene import render_event

__all__ = ["DEFAULT_EVENTS", "TINY_EVENTS", "BenchmarkResult", "build_benchmark", "main"]

LOGGER = logging.getLogger(__name__)

#: Shared geometry of the default events: 3 basins of 1024 x 2048 px (10.2 x 20.5 km, ~210
#: km^2 — a HydroBASINS level-8 size). Narrower basins lose most chips to the boundary
#: and buffer rules (docs/05 6.4: > 20 % dropped means basins too small for the chip).
_GEOMETRY = dict(n_basins=3, basin_width_px=1024, height_px=2048)
#: The stated, nominal acquisition time of every synthetic scene (products and STAC items need one).
NOMINAL_ACQUISITION = "2023-11-07T17:07:00+00:00"

DEFAULT_EVENTS: tuple[EventSpec, ...] = (
    EventSpec(
        "E1_lowland_po", seed=101, **_GEOMETRY, terrain="lowland", forest_cover=0.15, town=True, runway=True,
        wind_speed_ms=1.5, orbit_state="descending", relative_orbit=117, optical="cloudy",
        cloud_fraction=0.35, flood_depth_m=(1.2, 0.9, 1.6), harvest_fraction=0.10,
        origin_x=420_000.0, origin_y=5_060_000.0,
    ),
    EventSpec(
        "E2_alpine_valley", seed=202, **_GEOMETRY, terrain="mountain", forest_cover=0.6, riparian_forest=True,
        wind_speed_ms=1.0, orbit_state="ascending", relative_orbit=15, optical="none",
        flood_depth_m=(1.2, 1.6, 1.0), harvest_fraction=0.04,
        origin_x=520_000.0, origin_y=5_160_000.0,
    ),
    EventSpec(
        "E3_windy_plain", seed=303, **_GEOMETRY, terrain="lowland", forest_cover=0.1, wind_speed_ms=6.0,
        orbit_state="descending", relative_orbit=168, optical="clear", cloud_fraction=0.02,
        flood_depth_m=(1.5, 1.8, 1.2), nodata_border_px=40, harvest_fraction=0.08,
        origin_x=620_000.0, origin_y=4_960_000.0,
    ),
    EventSpec(
        "E4_hilly_forest", seed=404, **_GEOMETRY, terrain="hilly", forest_cover=0.5, riparian_forest=True,
        town=True, wind_speed_ms=2.5, orbit_state="ascending", relative_orbit=88,
        optical="cloudy", cloud_fraction=0.75, flood_depth_m=(1.0, 1.4, 1.6),
        harvest_fraction=0.06, origin_x=720_000.0, origin_y=5_100_000.0,
    ),
)

#: A miniature benchmark for tests and smoke runs: 2 events x 2 basins of 256 x 384 px.
TINY_EVENTS: tuple[EventSpec, ...] = (
    EventSpec("T1_lowland", seed=11, n_basins=2, basin_width_px=256, height_px=384,
              terrain="lowland", optical="clear", flood_depth_m=(1.2, 0.9),
              origin_x=400_000.0, origin_y=5_000_000.0),
    EventSpec("T2_hilly", seed=22, n_basins=2, basin_width_px=256, height_px=384,
              terrain="hilly", optical="none", orbit_state="ascending", relative_orbit=15,
              forest_cover=0.5, riparian_forest=True, flood_depth_m=(1.2, 1.6),
              origin_x=450_000.0, origin_y=5_000_000.0),
)


@dataclass(frozen=True)
class BenchmarkResult:
    root: Path
    manifest_path: Path
    basins_path: Path
    manifest_sha256: str
    records: tuple[ChipRecord, ...]
    events: tuple[dict[str, Any], ...]


def _write_basins(path: Path, basins: dict[str, tuple[str, Any]], epsg: int) -> None:
    import geopandas as gpd

    frame = gpd.GeoDataFrame(
        {"basin_id": list(basins), "event_id": [ev for ev, _ in basins.values()]},
        geometry=[geom for _, geom in basins.values()],
        crs=f"EPSG:{epsg}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{path.name}")
    frame.to_file(tmp, driver="GPKG", layer="basins")
    tmp.replace(path)


def build_benchmark(
    out_root: Path | str,
    *,
    specs: Sequence[EventSpec] = DEFAULT_EVENTS,
    data_cfg: DataConfig | None = None,
    grid_cfg: GridConfig | None = None,
    norm: NormalizationStats | None = None,
) -> BenchmarkResult:
    """Render every event, write its stack, chip it, and write the manifest + basins.

    Layout::

        <out_root>/events/<event_id>/stack.tif   16-band int16 analysis-ready stack
        <out_root>/events/<event_id>/event.json  provenance and per-event statistics
        <out_root>/basins.gpkg                   basin polygons (analysis CRS)
        <out_root>/chips/manifest.parquet        the immutable chip manifest
        <out_root>/chips/manifest.meta.json      its content hash + provenance
    """
    from floodfusion.utils.env import git_state

    root = Path(out_root)
    data_cfg = data_cfg or DataConfig()
    grid_cfg = grid_cfg or GridConfig()
    norm = norm or NormalizationStats()
    epsgs = {s.epsg for s in specs}
    if len(epsgs) != 1:
        raise ValueError("all synthetic events must share one CRS")
    records: list[ChipRecord] = []
    events: list[dict[str, Any]] = []
    basins: dict[str, tuple[str, Any]] = {}
    for spec in specs:
        started = time.time()
        rendered = render_event(spec, grid_cfg, norm)
        grid = AnalysisGrid(epsg=spec.epsg, transform=spec.transform,
                            width=spec.width_px, height=spec.height_px)
        event_dir = root / "events" / spec.event_id
        stack_path = write_event_stack(
            event_dir / "stack.tif", rendered.stack, grid,
            nodata=grid_cfg.nodata_int16,
            tags={"event_id": spec.event_id, "generator": "floodfusion.data.synthetic"},
        )
        geoms = rendered.basin_geometries()
        chips, counts = extract_chips(
            stack_path, event_id=spec.event_id, basins=geoms, data_cfg=data_cfg,
            grid_cfg=grid_cfg, relative_orbit=spec.relative_orbit, orbit_state=spec.orbit_state,
            has_optical=spec.optical != "none",
        )
        records.extend(chips)
        for bid, geom in geoms.items():
            basins[bid] = (spec.event_id, geom)
        meta = {**rendered.meta, "chip_counts": counts, "n_chips": len(chips),
                "render_seconds": round(time.time() - started, 2),
                # A synthetic scene has no acquisition; products need a time, so it is a stated nominal one.
                "s1_post_datetime": NOMINAL_ACQUISITION, "synthetic": True}
        (event_dir / "event.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        events.append(meta)
        LOGGER.info("%s rendered in %.1f s: %d chips, flood %.1f %% of valid pixels",
                    spec.event_id, meta["render_seconds"], len(chips), 100 * meta["flood_fraction_of_valid"])
    basins_path = root / "basins.gpkg"
    _write_basins(basins_path, basins, epsgs.pop())
    manifest_path = root / Path(data_cfg.chip_root).name / data_cfg.manifest_name
    provenance = {
        "generator": "floodfusion.data.synthetic.benchmark",
        "events": [e["event_id"] for e in events],
        "data_config": {"chip_size": data_cfg.chip_size, "train_stride": data_cfg.train_stride,
                        "eval_stride": data_cfg.eval_stride,
                        "min_valid_fraction": data_cfg.min_valid_fraction},
        **git_state(),
    }
    digest = write_manifest(records, manifest_path, provenance=provenance)
    return BenchmarkResult(root=root, manifest_path=manifest_path, basins_path=basins_path,
                           manifest_sha256=digest, records=tuple(records), events=tuple(events))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the physics-based synthetic flood benchmark.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--preset", choices=("default", "tiny"), default="default")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"),
                        help="config supplying the data/grid/norm sections (chip size, strides)")
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, overlays=args.overlay)
    specs = DEFAULT_EVENTS if args.preset == "default" else TINY_EVENTS
    started = time.time()
    result = build_benchmark(args.out, specs=specs, data_cfg=cfg.data, grid_cfg=cfg.grid, norm=cfg.norm)
    n_eval = sum(r.eval_grid for r in result.records)
    LOGGER.info("benchmark at %s: %d chips (%d on the evaluation grid), manifest %s… in %.0f s",
                result.root, len(result.records), n_eval, result.manifest_sha256[:12], time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
