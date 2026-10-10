r"""Sentinel-2 L2A simulation: reflectance, clouds, cloud shadow, SCL and optical quality.

Reflectances (B3 green, B8 NIR, B11 SWIR-1) per surface, from ``docs/02`` sections 2.4
and 5:

========================== ======= ======= ======= =========================================
surface                    B3      B8      B11     note
========================== ======= ======= ======= =========================================
clear permanent water      0.05    0.02    0.008   NIR/SWIR absorbed within millimetres
turbid flood water         0.09    0.07    0.03    MNDWI = +0.5, NDWI only +0.13 (section 5.1)
cropland / grassland       0.07    0.30    0.21
forest                     0.04    0.31    0.14
built-up                   0.12    0.18    0.22
bare soil / sand           0.14    0.24    0.32    bright SWIR: MNDWI < 0 (unlike SAR!)
flooded forest             0.045   0.24    0.12    the canopy hides the water from above too
========================== ======= ======= ======= =========================================

**Cloud shadow** multiplies the bands by (0.55, 0.32, 0.26): diffuse skylight is
blue-rich, so the green band loses least and MNDWI *rises* toward false water — the
dominant optical false positive (``docs/02`` section 5.2). The SCL marks clouds (8, 9),
cirrus (10) and cloud shadow (3); following section 7.3 it is used **only** to build the
quality mask (classes {0, 1, 2, 3, 8, 9, 10}, dilated) and is never an input or a label.
"""

from __future__ import annotations

import numpy as np

from floodfusion.data.synthetic.landscape import LANDCOVER, Landscape, fbm
from floodfusion.masks import BAD_SCL_CLASSES, quality_from_scl

__all__ = ["BAD_SCL_CLASSES", "REFLECTANCE", "simulate_optical"]

REFLECTANCE: dict[str, tuple[float, float, float]] = {
    "water": (0.05, 0.02, 0.008),
    "flood": (0.09, 0.07, 0.03),
    "cropland": (0.07, 0.30, 0.21),
    "grassland": (0.06, 0.27, 0.19),
    "forest": (0.04, 0.31, 0.14),
    "built_up": (0.12, 0.18, 0.22),
    "bare": (0.14, 0.24, 0.32),
    "flooded_forest": (0.045, 0.24, 0.12),
    "flooded_built_up": (0.10, 0.12, 0.10),
}
_CLOUD = np.array([0.55, 0.58, 0.45])
_SHADOW_FACTOR = np.array([0.55, 0.32, 0.26])


def simulate_optical(
    landscape: Landscape,
    flood: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return ``(reflectance (3,H,W), scl (H,W), quality (H,W))`` or ``(None,)*3``."""
    spec = landscape.spec
    if spec.optical == "none":
        return None, None, None
    lc = landscape.landcover
    h, w = lc.shape
    rho = np.zeros((3, h, w))
    scl = np.full((h, w), 4, dtype=np.uint8)  # vegetation
    by_code = {
        LANDCOVER["tree"]: "forest", LANDCOVER["grassland"]: "grassland",
        LANDCOVER["cropland"]: "cropland", LANDCOVER["built_up"]: "built_up",
        LANDCOVER["bare"]: "bare", LANDCOVER["water"]: "water",
    }
    for code, name in by_code.items():
        rho[:, lc == code] = np.asarray(REFLECTANCE[name])[:, None]
    scl[(lc == LANDCOVER["bare"]) | (lc == LANDCOVER["built_up"])] = 5
    scl[landscape.permanent_water] = 6
    flooded_forest = flood & (lc == LANDCOVER["tree"])
    flooded_built = flood & (lc == LANDCOVER["built_up"])
    open_flood = flood & ~flooded_forest & ~flooded_built
    rho[:, open_flood] = np.asarray(REFLECTANCE["flood"])[:, None]
    rho[:, flooded_forest] = np.asarray(REFLECTANCE["flooded_forest"])[:, None]
    rho[:, flooded_built] = np.asarray(REFLECTANCE["flooded_built_up"])[:, None]
    scl[open_flood] = 6
    rho *= 1.0 + 0.05 * fbm((h, w), 2.0, rng)[None] + 0.02 * rng.standard_normal((3, h, w))

    fraction = spec.cloud_fraction if spec.optical == "cloudy" else min(spec.cloud_fraction, 0.03)
    if fraction > 0:
        field = fbm((h, w), 3.0, rng)
        threshold = np.quantile(field, 1.0 - fraction)
        cloud = field > threshold
        opacity = np.clip((field - threshold) / 0.4 + 0.6, 0.6, 1.0)
        rho = np.where(cloud[None], (1 - opacity) * rho + opacity * _CLOUD[:, None, None], rho)
        scl[cloud] = np.where(field[cloud] > threshold + 0.4, 9, 8)
        # Shadow: the cloud mask displaced away from the sun (low autumn sun from the SSE).
        dy, dx = -min(40, h // 8), -min(12, w // 20)
        shadow = np.roll(cloud, (dy, dx), axis=(0, 1)) & ~cloud
        rho = np.where(shadow[None], rho * _SHADOW_FACTOR[:, None, None], rho)
        scl[shadow] = 3
        if spec.optical == "cloudy":
            cirrus_field = fbm((h, w), 2.5, rng)
            cirrus = (cirrus_field > np.quantile(cirrus_field, 0.97)) & ~cloud & ~shadow
            rho = np.where(cirrus[None], rho + 0.05, rho)
            scl[cirrus] = 10
    rho = np.clip(rho, 0.0, 1.5)
    return rho, scl, quality_from_scl(scl)
