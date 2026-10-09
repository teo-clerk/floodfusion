r"""Sentinel-1 gamma0 simulation: surface physics, wind, terrain geometry and speckle.

Every number below is a physical claim from ``docs/01`` and can be defended as such:

=========================== ============== =============== ================================
surface                     VV (dB)        VH (dB)         mechanism (docs/01 section)
=========================== ============== =============== ================================
calm open water             -23            -29             specular Fresnel reflection (3)
cropland / grassland        -11 / -12.5    -18 / -19.5     surface + volume (5)
forest                      -7.5           -13             volume scattering (5)
built-up                    -1             -8              dihedral wall-ground (8)
bare soil / sand            -15            -23             smooth-ish surface (9.2)
runway / playa ("smooth")   -23.5          -30             specular although dry (9.2)
flooded forest              -4             -11.5           double bounce: BRIGHTER (7)
flooded built-up            +1             -7              double bounce + layover (8)
radar shadow                -25            -26             no illumination: NESZ (9.1)
=========================== ============== =============== ================================

**Wind** (section 6): open water is roughened by Bragg-resonant capillary-gravity waves,
:math:`\Delta\gamma^0 = 10\log_{10}(1 + (U/2.6)^{3.2} f)`, with :math:`f` a fetch factor
that leaves the sheltered upwind shore dark — the "detected crescent" morphology. At
5 m/s the flood signal shrinks by ~9 dB, i.e. below the speckle noise of a single pixel.

**Speckle** (section 10.2): intensity is multiplied by an independent Gamma(L, 1/L)
factor per date and polarisation, L = 4.4, so a single-pixel dB difference has
sigma = 3.07 dB with no change on the ground. **Thermal noise**: a residual zero-mean
noise term at the NESZ (-22 dB) after noise subtraction, so calm water in VH is
noise-limited (section 3.3, ``docs/04`` section 4.2).
"""

from __future__ import annotations

import numpy as np

from floodfusion.data.synthetic.landscape import LANDCOVER, Landscape, fbm

__all__ = ["SURFACE_DB", "simulate_sar"]

#: (VV, VH) gamma0 in dB at ~37 deg incidence.
SURFACE_DB: dict[str, tuple[float, float]] = {
    "water": (-23.0, -29.0),
    "cropland": (-11.0, -18.0),
    "grassland": (-12.5, -19.5),
    "forest": (-7.5, -13.0),
    "built_up": (-1.0, -8.0),
    "bare": (-15.0, -23.0),
    "smooth": (-23.5, -30.0),
    "flooded_forest": (-4.0, -11.5),
    "flooded_built_up": (1.0, -7.0),
    "shadow": (-25.0, -26.0),
}
_LAND_CLASS = {
    LANDCOVER["tree"]: "forest", LANDCOVER["grassland"]: "grassland",
    LANDCOVER["cropland"]: "cropland", LANDCOVER["built_up"]: "built_up",
    LANDCOVER["bare"]: "bare", LANDCOVER["water"]: "water",
}
_REFERENCE_INCIDENCE = 37.0
_LOOKS = 4.4
_NESZ_DB = -22.0
_LAYOVER_GAIN_DB = 6.0
_PRE_EVENT_WIND_MS = 1.5


def _fetch_m(water: np.ndarray, pixel_m: float) -> np.ndarray:
    """Distance to the upwind shore for a wind blowing from the west (along rows)."""
    h, w = water.shape
    cols = np.broadcast_to(np.arange(w), (h, w))
    last_land = np.maximum.accumulate(np.where(~water, cols, -1), axis=1)
    return (cols - last_land) * pixel_m


def _wind_gain_db(wind_ms: float, fetch_m: np.ndarray) -> np.ndarray:
    fetch_factor = 1.0 - np.exp(-fetch_m / 300.0)
    return 10.0 * np.log10(1.0 + (wind_ms / 2.6) ** 3.2 * fetch_factor)


def _class_db(classes: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vv = np.zeros(classes.shape)
    vh = np.zeros(classes.shape)
    for i, name in enumerate(names):
        sel = classes == i
        vv[sel], vh[sel] = SURFACE_DB[name]
    return vv, vh


def simulate_sar(
    landscape: Landscape,
    flood: np.ndarray,
    incidence_deg: np.ndarray,
    shadow: np.ndarray,
    layover: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """gamma0 VV/VH, pre- and post-event, in dB (NaN = nodata). See the module table."""
    spec = landscape.spec
    lc = landscape.landcover
    h, w = lc.shape
    names = list(SURFACE_DB)
    index = {name: i for i, name in enumerate(names)}

    pre_cls = np.full((h, w), index["grassland"])
    for code, name in _LAND_CLASS.items():
        pre_cls[lc == code] = index[name]
    pre_cls[landscape.smooth] = index["smooth"]
    post_cls = pre_cls.copy()
    post_cls[flood & (lc == LANDCOVER["tree"])] = index["flooded_forest"]
    post_cls[flood & (lc == LANDCOVER["built_up"])] = index["flooded_built_up"]
    open_flood = flood & (lc != LANDCOVER["tree"]) & (lc != LANDCOVER["built_up"])
    post_cls[open_flood] = index["water"]

    # Stable land heterogeneity (both dates) + per-parcel structure + date-specific change.
    n_parcels = int(landscape.parcels.max()) + 1
    parcel_offset = rng.normal(0.0, 1.2, n_parcels)[landscape.parcels]
    texture = 1.0 * fbm((h, w), 2.0, rng) + parcel_offset
    change_pre = rng.normal(0.0, 0.6, n_parcels)[landscape.parcels]
    change_post = rng.normal(0.0, 0.6, n_parcels)[landscape.parcels]

    out: dict[str, np.ndarray] = {}
    theta_term = incidence_deg - _REFERENCE_INCIDENCE
    for date, classes, change, wind in (
        ("pre", pre_cls, change_pre, _PRE_EVENT_WIND_MS),
        ("post", post_cls, change_post, spec.wind_speed_ms),
    ):
        vv, vh = _class_db(classes, names)
        water = classes == index["water"]
        land = ~water & (classes != index["smooth"])
        vv = vv + np.where(land, texture + change, 0.3 * texture)
        vh = vh + np.where(land, texture + change, 0.3 * texture)
        vv = vv + np.where(water, -0.35, -0.12) * theta_term
        vh = vh + np.where(water, -0.30, -0.10) * theta_term
        if date == "post":
            harvest = landscape.harvested & ~flood
            vv = vv - 4.0 * harvest
            vh = vh - 3.0 * harvest
        gain = _wind_gain_db(wind, _fetch_m(water, spec.pixel_size_m))
        vv = vv + np.where(water, gain, 0.0)
        vh = vh + np.where(water, 0.7 * gain, 0.0)
        # Terrain geometry, identical on both dates because the orbits match (docs/04 7).
        vv = np.where(layover, vv + _LAYOVER_GAIN_DB, vv)
        vh = np.where(layover, vh + _LAYOVER_GAIN_DB, vh)
        vv = np.where(shadow, SURFACE_DB["shadow"][0], vv)
        vh = np.where(shadow, SURFACE_DB["shadow"][1], vh)
        for pol, db in (("vv", vv), ("vh", vh)):
            power = 10.0 ** (db / 10.0) * rng.gamma(_LOOKS, 1.0 / _LOOKS, size=(h, w))
            residual = 10.0 ** (_NESZ_DB / 10.0) * 0.35 * (rng.gamma(1.0, 1.0, size=(h, w)) - 1.0)
            power = np.maximum(power + residual, 10.0 ** (-3.6))  # noise-limited floor
            value = 10.0 * np.log10(power)
            if spec.nodata_border_px > 0:
                value[:, : spec.nodata_border_px] = np.nan
            out[f"{pol}_{date}_db"] = value
    return out
