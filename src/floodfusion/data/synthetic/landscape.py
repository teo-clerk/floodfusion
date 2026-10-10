r"""Synthetic landscapes: terrain, basins, rivers, land cover, permanent water and floods.

The landscape is built so that every failure mechanism catalogued in ``docs/01`` section
12 and ``docs/02`` section 5 *can* occur, at a frequency set by the event specification —
and then the real terrain code (:mod:`floodfusion.masks`) is run on it, exactly as on a
Copernicus DEM. Nothing here peeks at the model or the expected results: the parameters
are physical (slopes, depths, wind speed, cloud cover) and are frozen in
:mod:`floodfusion.data.synthetic.benchmark` before any experiment runs.

Geometry
--------
``n_basins`` vertical strips, each ``basin_width_px`` wide, separated by ridges. Each
strip holds a meandering river flowing north with a floodplain, valley walls and fBm
roughness; ``mountain`` terrain adds massifs whose flanks are steep enough (> 51 deg at
10 m) to cast radar shadow onto the valley floor — the most dangerous false positive.
Basins are therefore real hydrological units: the ridges are their divides, and the
basin polygons are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LANDCOVER",
    "EventSpec",
    "Landscape",
    "build_landscape",
    "fbm",
    "look_azimuth_for",
]

#: ESA WorldCover codes used by the generator (and by the R4 caveat layers).
LANDCOVER = {"tree": 10, "grassland": 30, "cropland": 40, "built_up": 50, "bare": 60, "water": 80}

#: Look azimuth phi = bearing FROM target TO sensor = heading - 90 deg for a right-looking
#: SAR (docs/03 section 5.1; see tests/test_look_geometry.py for the convention).
_HEADING_DEG = {"descending": 193.0, "ascending": 347.0}


def look_azimuth_for(orbit_state: str) -> float:
    """Typical Sentinel-1 look azimuth (target -> sensor) for a pass direction."""
    try:
        return (_HEADING_DEG[orbit_state] - 90.0) % 360.0
    except KeyError as exc:
        raise ValueError(f"orbit_state must be 'ascending' or 'descending', got {orbit_state!r}") from exc


@dataclass(frozen=True)
class EventSpec:
    """Physical description of one synthetic flood event. Frozen before any experiment."""

    event_id: str
    seed: int
    n_basins: int = 3
    basin_width_px: int = 768
    height_px: int = 2048
    pixel_size_m: float = 10.0
    terrain: str = "lowland"          # lowland | hilly | mountain
    forest_cover: float = 0.3         # fraction of valley walls under forest
    riparian_forest: bool = False     # forest ON the floodplain -> flooded forest (docs/01 7)
    town: bool = False                # built-up block on the floodplain (docs/01 8)
    runway: bool = False              # smooth dry surface, dark on both dates (docs/01 9.2)
    sand_bars: bool = True            # bright in SWIR, dark-ish in SAR (docs/01 9.4)
    wind_speed_ms: float = 1.0        # Bragg roughening of open water (docs/01 6)
    orbit_state: str = "descending"
    relative_orbit: int = 117
    incidence_near_deg: float = 34.0
    incidence_far_deg: float = 38.0
    optical: str = "none"             # none | clear | cloudy
    cloud_fraction: float = 0.0
    flood_depth_m: tuple[float, ...] = (2.0,)
    harvest_fraction: float = 0.08    # cropland parcels darkened between dates
    nodata_border_px: int = 0         # GRD swath-edge nodata on the west side
    origin_x: float = 500_000.0
    origin_y: float = 5_000_000.0
    epsg: int = 32632
    label_generalisation_px: int = 2  # cartographic smoothing of the reference label

    def __post_init__(self) -> None:
        if self.terrain not in {"lowland", "hilly", "mountain"}:
            raise ValueError(f"terrain must be lowland, hilly or mountain, got {self.terrain!r}")
        if self.optical not in {"none", "clear", "cloudy"}:
            raise ValueError(f"optical must be none, clear or cloudy, got {self.optical!r}")
        if self.n_basins < 1 or self.basin_width_px < 16 or self.height_px < 16:
            raise ValueError("a landscape needs at least one basin of at least 16x16 px")
        look_azimuth_for(self.orbit_state)

    @property
    def width_px(self) -> int:
        return self.n_basins * self.basin_width_px

    @property
    def look_azimuth_deg(self) -> float:
        return look_azimuth_for(self.orbit_state)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        p = self.pixel_size_m
        return (p, 0.0, self.origin_x, 0.0, -p, self.origin_y)


@dataclass
class Landscape:
    """Everything the sensors will observe, on the 10 m grid."""

    spec: EventSpec
    dem: np.ndarray                 # (H, W) metres
    basin_index: np.ndarray         # (H, W) int, strip index
    river_distance_m: np.ndarray    # (H, W) horizontal distance to the river centreline
    permanent_water: np.ndarray     # (H, W) bool (JRC occurrence >= 80 %)
    occurrence: np.ndarray          # (H, W) JRC-like occurrence, percent
    landcover: np.ndarray           # (H, W) uint8 WorldCover codes
    smooth: np.ndarray              # (H, W) bool runway / playa
    parcels: np.ndarray             # (H, W) int parcel id (cropland/grassland)
    harvested: np.ndarray           # (H, W) bool parcels darkened between the dates
    flood_truth: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))


def fbm(shape: tuple[int, int], beta: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Brownian surface by spectral synthesis, normalised to zero mean, unit std."""
    h, w = shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amplitude = k ** (-beta / 2.0)
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=amplitude.shape)
    spectrum = amplitude * np.exp(1j * phase)
    field_ = np.fft.irfft2(spectrum, s=(h, w))
    std = field_.std()
    return (field_ - field_.mean()) / (std if std > 0 else 1.0)


_TERRAIN = {
    #            noise_m  wall_slope  long_slope  floodplain_m  channel_halfwidth_m
    "lowland":  (0.8,     0.012,      0.0006,     600.0,        25.0),
    "hilly":    (3.0,     0.10,       0.002,      320.0,        15.0),
    "mountain": (8.0,     0.30,       0.006,      200.0,        10.0),
}
#: Cross-slope of the floodplain towards the channel (m/m): HAND rises ~3 m per km.
_FLOODPLAIN_SLOPE = 0.003


def build_landscape(spec: EventSpec) -> Landscape:
    """Terrain, rivers, basins, land cover and permanent water (floods come later)."""
    rng = np.random.default_rng(spec.seed)
    h, w, px = spec.height_px, spec.width_px, spec.pixel_size_m
    rows, cols = np.mgrid[0:h, 0:w]
    noise_m, wall_slope, long_slope, fp_m, ch_m = _TERRAIN[spec.terrain]
    wb = spec.basin_width_px
    fp_m = min(fp_m, 0.22 * wb * px)
    ch_m = min(ch_m, 0.05 * wb * px)

    # Meandering centreline per basin: two sinusoids, random phases, amplitude ~14 % of width.
    centre = np.empty((spec.n_basins, h))
    y = np.arange(h, dtype=np.float64)
    for k in range(spec.n_basins):
        a1, a2 = 0.10 * wb, 0.04 * wb
        l1, l2 = rng.uniform(0.5, 0.9) * h, rng.uniform(0.15, 0.3) * h
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        centre[k] = (k + 0.5) * wb + a1 * np.sin(2 * np.pi * y / l1 + p1) + a2 * np.sin(2 * np.pi * y / l2 + p2)
    # Every cell belongs to the basin of its NEAREST river and its elevation follows the
    # distance to that river, so the divide is the (continuous) equidistant line between
    # two rivers — a real ridge crest, not a cliff along a straight strip boundary.
    distance_px = np.abs(cols[None, :, :] - centre[:, rows])  # (n_basins, H, W)
    basin_index = np.argmin(distance_px, axis=0)
    distance_m = np.take_along_axis(distance_px, basin_index[None], axis=0)[0] * px
    del distance_px

    # Valley cross-profile: a near-flat floodplain, then walls rising to the divides.
    # One base level for the whole scene: per-basin offsets would put a step at every divide.
    base = rng.uniform(80.0, 140.0)
    cross = _FLOODPLAIN_SLOPE * np.minimum(distance_m, fp_m) + wall_slope * np.maximum(distance_m - fp_m, 0.0)
    along = long_slope * rows * px  # rivers flow north: elevation rises southward
    roughness = noise_m * fbm((h, w), 3.2, rng)
    # Roughness fades towards the channel so the river stays the valley's lowest line.
    roughness *= np.clip(distance_m / max(fp_m, 1.0), 0.15, 1.0)
    dem = base + along + cross + roughness
    channel = distance_m < ch_m
    dem = np.where(channel, base + along - 1.5, dem)  # carved bed

    if spec.terrain in {"mountain", "hilly"}:
        # Mountain massifs peak at ~65 deg flanks: steeper than the 90 - theta ~ 53 deg a back
        # slope needs to fall out of the radar's line of sight (docs/03 section 5.2).
        height_m, sigma_m = (650.0, 180.0) if spec.terrain == "mountain" else (160.0, 260.0)
        for k in range(spec.n_basins):
            for _ in range(3 if spec.terrain == "mountain" else 1):
                side = rng.choice([-1.0, 1.0])
                offset = (fp_m + rng.uniform(1.2, 2.2) * sigma_m) / px
                cy = rng.uniform(0.2, 0.8) * h
                cx = centre[k, int(cy)] + side * offset
                r2 = ((rows - cy) ** 2 + (cols - cx) ** 2) * px**2
                # Not clipped to the basin: a massif straddling a divide stays a smooth
                # mountain; clipping it would build an artificial cliff at the boundary.
                dem += height_m * np.exp(-r2 / (2.0 * sigma_m**2))

    # Permanent water: the channel, plus an oxbow lake in lowland basin 0.
    permanent = channel.copy()
    occurrence = np.where(channel, 96.0, 0.0)
    if spec.terrain == "lowland" and h >= 256:
        # An oxbow on the floodplain (not on the valley wall, which would carve a cliff).
        ly, lx = 0.3 * h, centre[0, int(0.3 * h)] + 0.55 * fp_m / px
        ry, rx = 0.04 * h, 0.25 * fp_m / px
        lake = ((rows - ly) / ry) ** 2 + ((cols - lx) / rx) ** 2 < 1.0
        lake &= basin_index == 0
        seasonal = (((rows - ly) / (1.4 * ry)) ** 2 + ((cols - lx) / (1.4 * rx)) ** 2 < 1.0) & ~lake
        seasonal &= basin_index == 0
        dem = np.where(lake, np.minimum(dem, base + along + 0.2), dem)
        permanent |= lake
        occurrence = np.where(lake, 92.0, np.where(seasonal, 45.0, occurrence))

    landcover, smooth, parcels, harvested = _land_cover(spec, rng, rows, cols, distance_m, fp_m,
                                                        centre, basin_index, permanent)
    return Landscape(
        spec=spec, dem=dem.astype(np.float64), basin_index=basin_index,
        river_distance_m=distance_m, permanent_water=permanent, occurrence=occurrence,
        landcover=landcover, smooth=smooth, parcels=parcels, harvested=harvested,
    )


def _land_cover(spec, rng, rows, cols, distance_m, fp_m, centre, basin_index, permanent):
    h, w = rows.shape
    px = spec.pixel_size_m
    lc = np.full((h, w), LANDCOVER["grassland"], dtype=np.uint8)
    # Agricultural parcels: Voronoi cells of random seeds, ~25 ha each.
    n_seeds = max(4, int(h * w * px * px / 250_000.0))
    seeds_y = rng.uniform(0, h, n_seeds)
    seeds_x = rng.uniform(0, w, n_seeds)
    parcels = _voronoi_labels(h, w, seeds_y, seeds_x)
    is_crop_parcel = rng.random(n_seeds) < 0.65
    lc[is_crop_parcel[parcels]] = LANDCOVER["cropland"]
    harvested_parcel = is_crop_parcel & (rng.random(n_seeds) < spec.harvest_fraction / 0.65)
    harvested = harvested_parcel[parcels]

    walls = distance_m > fp_m
    forest_noise = fbm((h, w), 2.6, rng)
    threshold = np.quantile(forest_noise, 1.0 - spec.forest_cover) if spec.forest_cover > 0 else np.inf
    lc[walls & (forest_noise > threshold)] = LANDCOVER["tree"]
    if spec.riparian_forest:
        reach = (rows > 0.15 * h) & (rows < 0.75 * h)
        band = (distance_m > 0.15 * fp_m) & (distance_m < 0.75 * fp_m)
        lc[reach & band & (forest_noise > np.quantile(forest_noise, 0.35))] = LANDCOVER["tree"]

    smooth = np.zeros((h, w), dtype=bool)
    if spec.town and h >= 256:
        k = min(1, spec.n_basins - 1)
        r0, r1 = int(0.35 * h), int(0.52 * h)
        x0 = centre[k, int(0.43 * h)]
        town = (rows >= r0) & (rows < r1) & (cols > x0 + 60 / px) & (cols < x0 + 700 / px)
        lc[town & (basin_index == k)] = LANDCOVER["built_up"]
    if spec.runway and h >= 256:
        k = spec.n_basins - 1
        xr = centre[k, int(0.6 * h)] - 0.7 * fp_m / px
        runway = (rows > 0.55 * h) & (rows < 0.55 * h + 1500 / px) & (np.abs(cols - xr) < 22 / px)
        runway &= basin_index == k
        lc[runway] = LANDCOVER["bare"]
        smooth |= runway
    if spec.sand_bars:
        for k in range(spec.n_basins):
            for _ in range(2):
                cy = rng.uniform(0.1, 0.9) * h
                cx = centre[k, int(cy)] + rng.choice([-1, 1]) * 50 / px
                bar = ((rows - cy) / max(3.0, 0.02 * h)) ** 2 + ((cols - cx) / max(2.0, 4.0)) ** 2 < 1.0
                lc[bar & ~permanent] = LANDCOVER["bare"]
    lc[permanent] = LANDCOVER["water"]
    return lc, smooth, parcels, harvested & (lc == LANDCOVER["cropland"])


def _voronoi_labels(h: int, w: int, seeds_y: np.ndarray, seeds_x: np.ndarray) -> np.ndarray:
    """Nearest-seed labels (parcels) via a k-d tree — exact Voronoi cells on the pixel grid."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([seeds_y, seeds_x]))
    rr, cc = np.mgrid[0:h, 0:w]
    _, idx = tree.query(np.column_stack([rr.ravel(), cc.ravel()]))
    return idx.reshape(h, w)


def flood_from_hand(
    landscape: Landscape,
    hand_m: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flood extent: low HAND **and** hydraulically connected to the river.

    Per basin, water stands up to ``flood_depth_m`` above the drainage network (with a
    smooth +-0.3 m perturbation for a realistic, irregular shoreline). Only components
    touching the permanent channel flood — a depression on the far side of a divide does
    not fill just because its HAND is low (that is the point of HAND being measured along
    the flow path). Permanent water is never "flood" (``docs/05`` section 3.3).
    """
    from scipy import ndimage

    spec = landscape.spec
    depths = np.asarray(spec.flood_depth_m, dtype=np.float64)
    depth = depths[landscape.basin_index % depths.size]
    jitter = 0.3 * fbm(hand_m.shape, 2.8, rng)
    candidate = (hand_m < depth + jitter) & np.isfinite(hand_m)
    labels, n = ndimage.label(candidate | landscape.permanent_water, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros_like(candidate)
    touching = np.unique(labels[landscape.permanent_water])
    connected = np.isin(labels, touching[touching > 0])
    return connected & candidate & ~landscape.permanent_water


def generalise_label(truth: np.ndarray, radius_px: int) -> np.ndarray:
    """The reference label as a cartographer would draw it (``docs/05`` section 3.2).

    Closing then opening with a disk smooths the boundary, drops 1--2 px channels and
    islands, and fills small gaps: structured, non-random label error — the ceiling on the
    accuracy one can *measure*.
    """
    from scipy import ndimage

    if radius_px <= 0:
        return truth.copy()
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    disk = xx**2 + yy**2 <= radius_px**2
    closed = ndimage.binary_closing(truth, structure=disk, border_value=0)
    opened = ndimage.binary_opening(closed, structure=disk, border_value=0)
    labels, n = ndimage.label(opened)
    if n:
        sizes = ndimage.sum(opened, labels, index=np.arange(1, n + 1))
        small = np.isin(labels, np.nonzero(sizes < 4 * math.pi * radius_px**2)[0] + 1)
        opened &= ~small
    return opened
