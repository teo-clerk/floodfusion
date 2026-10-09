"""Render one synthetic event into the 16-band analysis-ready stack.

The terrain products are computed by the production code — :func:`floodfusion.masks.
horn_slope_aspect`, :func:`~floodfusion.masks.local_incidence_angle`,
:func:`~floodfusion.masks.cast_shadow_and_layover` and :func:`~floodfusion.masks.hand` (on
a 30 m version of the DEM, as for Copernicus GLO-30) — and the result is encoded by
:func:`floodfusion.data.encoding.encode_stack`, so the synthetic benchmark exercises
exactly the code paths real data does.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from floodfusion.config import GridConfig, NormalizationStats
from floodfusion.data.encoding import encode_stack
from floodfusion.data.synthetic.landscape import (
    EventSpec,
    build_landscape,
    flood_from_hand,
    generalise_label,
)
from floodfusion.data.synthetic.optical import simulate_optical
from floodfusion.data.synthetic.sar import simulate_sar
from floodfusion.masks import (
    build_validity_mask,
    cast_shadow_and_layover,
    hand,
    horn_slope_aspect,
    local_incidence_angle,
    permanent_water_from_occurrence,
)

__all__ = ["RenderedEvent", "basin_polygons", "incidence_field", "render_event"]

_DEM_SUPPORT_M = 30.0  # GLO-30 support scale: HAND is computed at 30 m, then interpolated


@dataclass
class RenderedEvent:
    spec: EventSpec
    stack: np.ndarray                 # (16, H, W) int16
    basins: dict[str, Any]            # basin_id -> shapely polygon (analysis CRS)
    meta: dict[str, Any]
    layers: dict[str, np.ndarray] | None = None

    def basin_geometries(self) -> dict[str, Any]:
        return dict(self.basins)


def basin_polygons(basin_index: np.ndarray, spec: EventSpec) -> dict[str, Any]:
    """Polygonise the basin raster (divides = equidistant lines between rivers)."""
    from affine import Affine
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    parts: dict[int, list[Any]] = {}
    for geom, value in features.shapes(basin_index.astype(np.int32), transform=Affine(*spec.transform)):
        parts.setdefault(int(value), []).append(shape(geom))
    return {f"{spec.event_id}_B{k}": unary_union(parts[k]) for k in sorted(parts)}


def incidence_field(spec: EventSpec) -> np.ndarray:
    """Ellipsoid incidence across the scene: near range on the sensor side.

    Descending passes have the sensor to the east (look azimuth ~103 deg), so incidence
    grows westward; ascending passes mirror it.
    """
    cols = np.arange(spec.width_px, dtype=np.float64) / max(spec.width_px - 1, 1)
    if spec.orbit_state == "descending":
        theta = spec.incidence_far_deg + (spec.incidence_near_deg - spec.incidence_far_deg) * cols
    else:
        theta = spec.incidence_near_deg + (spec.incidence_far_deg - spec.incidence_near_deg) * cols
    return np.broadcast_to(theta, (spec.height_px, spec.width_px)).copy()


def _hand_at_dem_support(dem: np.ndarray, pixel_m: float) -> np.ndarray:
    """HAND on a block-averaged 30 m DEM, bilinearly interpolated back to the grid."""
    from scipy import ndimage

    factor = max(1, int(round(_DEM_SUPPORT_M / pixel_m)))
    h, w = dem.shape
    ph, pw = math.ceil(h / factor) * factor, math.ceil(w / factor) * factor
    padded = np.pad(dem, ((0, ph - h), (0, pw - w)), mode="edge")
    coarse = padded.reshape(ph // factor, factor, pw // factor, factor).mean(axis=(1, 3))
    n_cells = coarse.size
    threshold = float(np.clip(n_cells // 400, 8, 600))
    hand_coarse = hand(coarse, threshold, pixel_m * factor)
    hand_coarse = np.nan_to_num(hand_coarse, nan=float(np.nanmax(hand_coarse)) if np.isfinite(hand_coarse).any() else 0.0)
    fine = ndimage.zoom(hand_coarse, factor, order=1, mode="nearest", grid_mode=True)
    return np.maximum(fine[:h, :w], 0.0)


def render_event(
    spec: EventSpec,
    grid_cfg: GridConfig,
    norm: NormalizationStats,
    *,
    keep_layers: bool = False,
) -> RenderedEvent:
    """Landscape -> terrain products -> flood -> SAR + optical -> validity -> int16 stack."""
    landscape = build_landscape(spec)
    px = spec.pixel_size_m
    phi = spec.look_azimuth_deg
    theta = incidence_field(spec)
    slope, aspect = horn_slope_aspect(landscape.dem, px, px)
    theta_loc = local_incidence_angle(slope, aspect, theta, phi)
    shadow, layover = cast_shadow_and_layover(landscape.dem, theta, phi, px)
    hand_m = _hand_at_dem_support(landscape.dem, px)

    flood_rng = np.random.default_rng([spec.seed, 1])
    flood = flood_from_hand(landscape, hand_m, flood_rng)
    landscape.flood_truth = flood
    # Flood = observed water AND NOT permanent water (docs/05 section 3.3) — re-applied after
    # generalisation, whose closing would otherwise bridge flood across the channel.
    permanent = permanent_water_from_occurrence(landscape.occurrence)
    label = generalise_label(flood, spec.label_generalisation_px) & ~permanent

    sar = simulate_sar(landscape, flood, theta, shadow, layover, np.random.default_rng([spec.seed, 2]))
    rho, scl, quality = simulate_optical(landscape, flood, np.random.default_rng([spec.seed, 3]))

    nodata = ~np.isfinite(sar["vv_post_db"])
    validity = build_validity_mask(layover, shadow, permanent, nodata, None, slope_max_deg=None)
    stack = encode_stack(
        vv_post_db=sar["vv_post_db"], vh_post_db=sar["vh_post_db"],
        vv_pre_db=sar["vv_pre_db"], vh_pre_db=sar["vh_pre_db"],
        reflectance=rho, slope_deg=slope, hand_m=hand_m, local_incidence_deg=theta_loc,
        label=label.astype(np.float64), validity=validity, optical_quality=quality,
        landcover=landscape.landcover, grid_cfg=grid_cfg, norm=norm,
    )

    basins = basin_polygons(landscape.basin_index, spec)
    valid = validity == 0
    meta = {
        "event_id": spec.event_id,
        "epsg": spec.epsg,
        "transform": list(spec.transform),
        "width": spec.width_px,
        "height": spec.height_px,
        "relative_orbit": spec.relative_orbit,
        "orbit_state": spec.orbit_state,
        "look_azimuth_deg": phi,
        "incidence_deg": [float(theta.min()), float(theta.max())],
        "has_optical": spec.optical != "none",
        "s1_post_id": f"SYNTH_S1_{spec.event_id}_POST",
        "s1_pre_id": f"SYNTH_S1_{spec.event_id}_PRE",
        "s2_id": f"SYNTH_S2_{spec.event_id}" if spec.optical != "none" else None,
        "dem_ids": [f"SYNTH_GLO30_{spec.event_id}"],
        "flood_fraction_of_valid": float((label & valid).sum() / max(valid.sum(), 1)),
        "permanent_water_fraction": float(permanent.mean()),
        "shadow_fraction": float(shadow.mean()),
        "layover_fraction": float(layover.mean()),
        "unusable_optical_fraction": float((quality < 0.5).mean()) if quality is not None else 1.0,
        "spec": asdict(spec),
    }
    layers = None
    if keep_layers:
        layers = {"dem": landscape.dem, "basin_index": landscape.basin_index, "flood_truth": flood, "label": label, "shadow": shadow,
                  "layover": layover, "slope": slope, "hand": hand_m, "theta_loc": theta_loc,
                  "permanent": permanent, "landcover": landscape.landcover, **sar}
        if scl is not None:
            layers["scl"] = scl
    return RenderedEvent(spec=spec, stack=stack, basins=basins, meta=meta, layers=layers)
