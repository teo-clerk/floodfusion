r"""Terrain derivatives and the exclusion masks.

Terrain enters this project in four distinct roles, routinely conflated
(``docs/03`` section 1):

1. **Geometric correction** — placing SAR pixels at their true ground position.
2. **Radiometric correction** — normalising for illuminated area via
   :math:`\theta_{\mathrm{loc}}`.
3. **Validity masking** — where the measurement is meaningless (layover, shadow).
4. **Hydrological prior** — where water can physically be (slope, HAND).

Roles 1--2 change pixel values and belong to :mod:`floodfusion.data.preprocessing`.
Roles 3--4 are this module: they become channels of the input tensor and exclusion
layers of the output.

Radar shadow deserves special emphasis. It is the **most dangerous false positive in the
project**: darker than water (at the NESZ, :math:`\approx -25` dB), spatially coherent
and elongated so it looks morphologically plausible, and systematically located in
valleys where floods occur — so a hydrological prior does not remove it. It is also
**completely predictable** from the DEM and the acquisition geometry, both known before
inference. There is no excuse for a shadow false positive in a delivered product.

References
----------
``docs/03_dem_hand_and_terrain_effects.md`` sections 4, 5, 6, 8.
``docs/01_sar_hydrology_and_water_backscatter.md`` section 9.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from floodfusion.config import ValidityFlag

__all__ = [
    "BAD_SCL_CLASSES",
    "build_validity_mask",
    "cast_shadow_and_layover",
    "hand",
    "hand_on_dem_support",
    "horn_slope_aspect",
    "local_incidence_angle",
    "permanent_water_from_occurrence",
    "quality_from_scl",
    "validity_flag_counts",
]

#: Sentinel-2 SCL classes treated as unusable for the optical branch: no data, saturated,
#: dark area, cloud shadow, cloud medium/high probability, thin cirrus (docs/02 7.1).
#: SCL is used ONLY for this mask — never as an input or a label (docs/02 section 7.3).
BAD_SCL_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 8, 9, 10)

#: D8 neighbour offsets (row, col), clockwise from north.
_D8_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)


def horn_slope_aspect(
    elevation: np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Slope and downslope aspect by Horn's third-order finite difference.

    Parameters
    ----------
    elevation
        ``(H, W)`` elevation in **metres**.
    pixel_size_x_m, pixel_size_y_m
        Pixel size in **metres**. Separate arguments on purpose — see the warning.

    Returns
    -------
    slope_deg, aspect_deg
        ``(H, W)`` each. Aspect is the **downslope azimuth**, clockwise from north,
        which is the convention :func:`local_incidence_angle` assumes.

    Notes
    -----
    With the :math:`3\times3` neighbourhood indexed

    .. math::
        \begin{matrix} z_1 & z_2 & z_3 \\ z_4 & z_5 & z_6 \\ z_7 & z_8 & z_9 \end{matrix}

    .. math::

        \frac{\partial z}{\partial x} =
            \frac{(z_3 + 2z_6 + z_9) - (z_1 + 2z_4 + z_7)}{8\,\Delta x}, \qquad
        \frac{\partial z}{\partial y} =
            \frac{(z_7 + 2z_8 + z_9) - (z_1 + 2z_2 + z_3)}{8\,\Delta y}

    .. math::

        \alpha = \arctan\sqrt{(\partial_x z)^2 + (\partial_y z)^2}, \qquad
        A = \operatorname{atan2}(\partial_y z,\; -\partial_x z)

    .. warning::

       **Compute this on a metric CRS, never on a geographic one.** The numerators are
       elevation differences in metres; if :math:`\Delta x, \Delta y` are in *degrees*
       the ratio is metres per degree, which is not a slope. At 45 degrees N on a
       1-arcsecond grid, a 5 m rise across one pixel yields
       :math:`\arctan(1.8\times10^{4}) = 89.997` degrees — every pixel becomes a cliff.

       Two subtler variants bite even when people notice the first:

       - Converting with a single constant :math:`111\,320` m/degree is correct for
         latitude and wrong for longitude, where the spacing is
         :math:`111\,320\cos\varphi`. Slope magnitude is then wrong **and aspect is
         rotated** — plausible-looking and completely wrong.
       - Copernicus GLO-30 coarsens longitude sampling above 50 degrees N (1.5", 2",
         3", ...), so even the :math:`\cos\varphi` correction is insufficient unless you
         read :math:`\Delta x` from the file transform.

       Reproject to UTM, **then** compute. ``tests/test_terrain.py`` asserts the CRS is
       projected and that a negligible fraction of pixels exceed 60 degrees.

    Implementation notes
    --------------------
    1. Both finite differences use array slicing, not a Python loop.
    2. Borders are replicate-padded, computed, then trimmed (see the comment in the body).
    3. Aspect is in degrees, wrapped into ``[0, 360)``.
    4. A consumer should use aspect as :math:`(\sin A, \cos A)`, never as degrees: 359 and 1
       are adjacent directions but numerically maximally distant (``docs/03`` section 9).
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("elevation must be a 2-D array")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("pixel sizes must be positive distances in metres")
    # Border handling: replicate-pad by one cell, compute, then the output is trimmed
    # back to the input shape. Replicated borders bias the edge row/column towards
    # zero slope across the border, which is the conservative choice for an exclusion.
    zp = np.pad(z, 1, mode="edge")
    z1, z2, z3 = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    z4, z6 = zp[1:-1, :-2], zp[1:-1, 2:]
    z7, z8, z9 = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * pixel_size_x_m)
    # Rows increase SOUTHWARD, so this is (south - north): +ve when terrain rises to the south.
    dzdy = ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / (8.0 * pixel_size_y_m)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # atan2(dz/dy, -dz/dx) is a MATH angle (counter-clockwise from east); converting with
    # (90 - angle) mod 360 gives the compass bearing of steepest DESCENT (ESRI convention).
    aspect = np.mod(90.0 - np.degrees(np.arctan2(dzdy, -dzdx)), 360.0)
    aspect[aspect >= 360.0] = 0.0  # np.mod(-0.0, 360) can return 360.0
    aspect[slope == 0.0] = 0.0     # undefined on flats; any value works since sin(0) = 0
    return slope, aspect


def local_incidence_angle(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    incidence_deg: np.ndarray | float,
    look_azimuth_deg: np.ndarray | float,
) -> np.ndarray:
    r"""Local incidence angle between the terrain normal and the look direction.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` in degrees.

    Notes
    -----
    In a local East--North--Up frame, a surface with slope :math:`\alpha` and downslope
    aspect :math:`A` has outward unit normal
    :math:`\hat{\mathbf{n}} = (\sin\alpha\sin A,\ \sin\alpha\cos A,\ \cos\alpha)`, and
    the unit vector toward the sensor at incidence :math:`\theta` and look azimuth
    :math:`\phi` is
    :math:`\hat{\mathbf{i}} = (\sin\theta\sin\phi,\ \sin\theta\cos\phi,\ \cos\theta)`.
    Hence

    .. math::

        \cos\theta_{\mathrm{loc}} = \hat{\mathbf{n}}\cdot\hat{\mathbf{i}}
        = \cos\alpha\cos\theta + \sin\alpha\sin\theta\cos(\phi - A)

    Three sanity checks you should assert rather than trust:

    - :math:`\alpha = 0` gives :math:`\theta_{\mathrm{loc}} = \theta`;
    - slope facing the sensor (:math:`A = \phi`) gives
      :math:`\theta_{\mathrm{loc}} = \theta - \alpha`;
    - slope facing away gives :math:`\theta_{\mathrm{loc}} = \theta + \alpha`.

    **Look azimuth.** For a right-looking sensor, :math:`\phi` is the heading rotated
    :math:`+90` degrees. A descending pass heads roughly SSW and therefore looks WNW; an
    ascending pass looks ENE. This is why ascending and descending passes have
    mirror-image layover and shadow patterns, and why the pre-event reference must share
    the pass direction (``docs/04`` section 7).

    Implementation notes
    --------------------
    1. Everything is computed in radians internally.
    2. ``tests/test_terrain.py`` asserts the three checks above.
    3. ``incidence_deg`` may be a full array — it varies across the swath from 29 to
       46 degrees, and a scalar mid-swath value discards a real effect
       (``docs/01`` sections 4.2, 6.1).
    """
    alpha = np.radians(np.asarray(slope_deg, dtype=np.float64))
    aspect = np.radians(np.asarray(aspect_deg, dtype=np.float64))
    theta = np.radians(np.asarray(incidence_deg, dtype=np.float64))
    phi = np.radians(np.asarray(look_azimuth_deg, dtype=np.float64))
    cos_loc = np.cos(alpha) * np.cos(theta) + np.sin(alpha) * np.sin(theta) * np.cos(phi - aspect)
    theta_loc = np.degrees(np.arccos(np.clip(cos_loc, -1.0, 1.0)))
    return np.broadcast_to(theta_loc, np.broadcast_shapes(alpha.shape, theta.shape)).copy()


def cast_shadow_and_layover(
    elevation: np.ndarray,
    incidence_deg: np.ndarray | float,
    look_azimuth_deg: float,
    pixel_size_m: float,
    *,
    return_components: bool = False,
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    r"""Layover and shadow masks, including **cast** shadow.

    Returns
    -------
    shadow, layover
        ``(H, W)`` boolean.

    Notes
    -----
    The **local** test — :math:`\theta_{\mathrm{loc}} \le 0` for layover,
    :math:`\theta_{\mathrm{loc}} \ge 90` degrees for shadow — asks only about the facet
    at each pixel. It misses the case that matters most: **a tall ridge casting a shadow
    across flat terrain hundreds of metres beyond it**. That terrain has
    :math:`\alpha = 0`, hence :math:`\theta_{\mathrm{loc}} = \theta`, passes the local
    test, and receives no illumination at all.

    Cast shadow requires a horizon sweep along the ground-range direction::

        1. Resample or interpolate along rays aligned with the ground-range direction
           phi. Do NOT approximate phi by the nearest array axis -- a 20 degree error
           rotates every shadow.

        2. FOR each range profile, ordered by increasing distance from the sensor:
               max_elev_angle <- -infinity
               FOR each pixel p along the profile:
                   IF local_horizon_angle(p) < max_elev_angle:
                       shadow[p] <- True
                   ELSE:
                       shadow[p] <- False
                       max_elev_angle <- local_horizon_angle(p)

        3. Layover, symmetrically: a pixel is in layover if a FARTHER pixel has a
           SMALLER slant range. Compute slant range per pixel and flag any
           non-monotonicity along the profile.

    This is the standard horizon sweep and is :math:`O(N)` per profile — cheap.

    The per-mask attribution (``P1-M2.4``) reports the IoU change from the shadow mask
    alone. The fraction of true shadow the local test misses is large in mountainous
    terrain (``docs/03``).

    Implementation notes
    --------------------
    1. The local tests are those of :func:`local_incidence_angle`.
    2. The cast sweep runs along range rays: each pixel joins the ray of its azimuth
       coordinate, so :math:`\phi` is not snapped to an array axis and the elevation is not
       resampled twice.
    3. Local and cast masks are combined with a logical OR.
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("elevation must be a 2-D array")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be a positive distance in metres")
    theta = np.radians(np.broadcast_to(np.asarray(incidence_deg, dtype=np.float64), z.shape))
    if np.any((theta <= 0.0) | (theta >= np.pi / 2)):
        raise ValueError("incidence angles must lie strictly between 0 and 90 degrees")
    h_px, w_px = z.shape
    rows, cols = np.mgrid[0:h_px, 0:w_px]
    east = cols * pixel_size_m
    north = -rows * pixel_size_m
    # look_azimuth is the bearing FROM the target TO the sensor (docs/03 section 5.1), so
    # ground range increases along psi = phi + 180 deg (away from the sensor).
    psi = math.radians(look_azimuth_deg + 180.0)
    rng_coord = east * math.sin(psi) + north * math.cos(psi)
    az_coord = east * math.cos(psi) - north * math.sin(psi)
    # "Interpolate along rays": every pixel joins the range profile (ray) whose azimuth
    # coordinate it falls in, at its EXACT ground range — no rotation, no resampling of z,
    # and phi is not snapped to an array axis.
    ray = np.floor(az_coord / pixel_size_m + 0.5).astype(np.int64).ravel()
    r = rng_coord.ravel()
    zf = z.ravel()
    th = theta.ravel()
    # Shadow: the grazing ray from a nearer point q descends cot(theta) per metre of range,
    # so p is shadowed iff z_p + r_p cot(theta) < max_{q nearer} (z_q + r_q cot(theta)).
    horizon = zf + r / np.tan(th)
    # Layover: p is in layover iff a FARTHER point has a SMALLER slant range,
    # slant ~ r sin(theta) - z cos(theta) (far-field projection on the look direction).
    slant = r * np.sin(th) - zf * np.cos(th)
    order = np.lexsort((r, ray))           # by ray, then near -> far
    ray_sorted = ray[order]
    rank = np.concatenate(([0], np.cumsum(np.diff(ray_sorted) != 0)))
    eps = 1e-6

    # Segmented running max (near -> far) via a per-ray offset larger than any range of h.
    hs = horizon[order]
    big_h = (hs.max() - hs.min()) + 1.0
    h_off = hs + rank * big_h
    run_max = np.maximum.accumulate(h_off)
    prev_max = np.concatenate(([-np.inf], run_max[:-1]))
    shadow_sorted = h_off < prev_max - eps

    # Segmented running min (far -> near): traverse reversed, larger offsets for earlier rays.
    ss = slant[order][::-1]
    rank_rev = rank[::-1]
    big_s = (ss.max() - ss.min()) + 1.0
    s_off = ss + rank_rev * big_s
    run_min = np.minimum.accumulate(s_off)
    prev_min = np.concatenate(([np.inf], run_min[:-1]))
    layover_sorted = (s_off > prev_min + eps)[::-1]

    cast_shadow = np.zeros(zf.size, dtype=bool)
    cast_layover = np.zeros(zf.size, dtype=bool)
    cast_shadow[order] = shadow_sorted
    cast_layover[order] = layover_sorted
    cast_shadow = cast_shadow.reshape(z.shape)
    cast_layover = cast_layover.reshape(z.shape)

    # Local (facet) tests, OR-ed in: the slope component in the range plane, positive when
    # the facet faces the sensor. Layover iff alpha_r >= theta; shadow iff the back slope
    # exceeds the 90 - theta depression of the ray. (theta_loc from arccos can never be
    # <= 0, which is why the signed range component is used here.)
    slope, aspect = horn_slope_aspect(z, pixel_size_m, pixel_size_m)
    alpha_r = np.arctan(np.tan(np.radians(slope)) * np.cos(np.radians(look_azimuth_deg - aspect)))
    local_layover = alpha_r >= theta
    local_shadow = -alpha_r >= (np.pi / 2 - theta)

    shadow = cast_shadow | local_shadow
    layover = cast_layover | local_layover
    if return_components:
        return {
            "shadow": shadow, "layover": layover,
            "cast_shadow": cast_shadow, "cast_layover": cast_layover,
            "local_shadow": local_shadow, "local_layover": local_layover,
        }
    return shadow, layover


def hand(
    elevation: np.ndarray,
    accumulation_threshold: float,
    pixel_size_m: float,
) -> np.ndarray:
    r"""Height Above Nearest Drainage.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` in metres.

    Notes
    -----
    Absolute elevation says nothing about flood susceptibility: 200 m may be a valley
    floor in the Alps or a hilltop in the Netherlands. HAND assigns each cell the
    vertical drop to the drainage cell it flows to,
    :math:`\mathrm{HAND}(p) = z(p) - z(d(p))`, removing the regional slope.

    Algorithm (``docs/03`` section 8.2):

    1. **Condition** the DEM — fill sinks, or better *breach* depressions. Filling
       raises terrain and creates artificial flats; breaching cuts through barriers and
       better preserves real topography. Record which you used.
    2. **Flow direction** — D8 (simple, parallel artefacts on flats) or D-infinity
       (smoother, preferred).
    3. **Flow accumulation.**
    4. **Drainage network** = cells above ``accumulation_threshold``.
    5. For each cell, follow the flow direction downslope to the first network cell and
       take the elevation difference.

    .. warning::

       "Nearest" means **nearest along the flow path**, not Euclidean-nearest. A cell
       just over a ridge from a river drains to a *different*, possibly much lower,
       river. Using Euclidean distance assigns low HAND to the wrong side of a watershed
       divide — a common and serious bug.

    **Two DSM corruptions you must state as limitations** (``docs/03`` section 8.4). The
    Copernicus DEM is a *surface* model, not a terrain model:

    - **Canopy inflation.** Over forest, :math:`z(p)` is the canopy top, 15--35 m too
      high, while the drainage cell over open water is correct. HAND is inflated by
      roughly the tree height *exactly where flooded forest occurs*. This compounds with
      ``docs/01`` section 7 — C-band already under-detects flooded forest — in the
      **same direction**. Two independent mechanisms, both producing false negatives
      over forest.
    - **Urban flow blocking.** Buildings appear as solid walls, so step 1 fills city
      blocks into plateaus or breaches them arbitrarily and the flow directions are
      fictional.

    ``accumulation_threshold`` is a **real parameter**, not a constant: it sets drainage
    density and therefore every HAND value; results should be checked at two thresholds
    (``docs/03``).

    Implementation notes
    --------------------
    1. The five steps are implemented here (Priority-Flood with an epsilon fill, then D8),
       with no hydrology dependency.
    2. The network sees :math:`\log(1 + \mathrm{HAND})`: the informative range is 0--30 m and
       a log compresses the long hillslope tail.
    3. In the learned model HAND is a **soft prior** — an input channel. Only the physics
       baseline applies a hard cut (``hand_max_m``), and its attribution table measures the
       cost: a hard HAND mask makes dam breaks, urban pluvial flooding and coastal surge
       undetectable by construction.
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("elevation must be a 2-D array")
    if accumulation_threshold <= 0:
        raise ValueError("accumulation_threshold (in cells) must be positive")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be a positive distance in metres")
    filled = _priority_flood_epsilon(z)
    down = _d8_receivers(filled, pixel_size_m)
    accumulation = _flow_accumulation(filled, down)
    # Drainage network: cells above the accumulation threshold, plus every cell that
    # drains out of the domain (the domain edge is the outlet of the conditioned DEM).
    drainage = (accumulation.ravel() >= accumulation_threshold) | (down < 0)
    return _height_above_drainage(filled, down, drainage).reshape(z.shape)


def _priority_flood_epsilon(z: np.ndarray) -> np.ndarray:
    """Barnes et al. (2014) Priority-Flood + epsilon depression filling.

    Fills every pit to its spill level and imposes a strictly increasing epsilon gradient
    (one ``nextafter`` step) across flats, so every interior cell has a strictly lower
    neighbour and D8 is defined everywhere. **Filling** (not breaching) is the
    conditioning choice recorded here (docs/03 section 8.2). NaN cells are treated as
    outlets (nodata holes drain out).
    """
    h, w = z.shape
    filled = z.copy()
    closed = np.zeros((h, w), dtype=bool)
    heap: list[tuple[float, int, int, int]] = []
    counter = 0
    nan = ~np.isfinite(z)
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1) or nan[r, c]:
                closed[r, c] = True
                if nan[r, c]:
                    filled[r, c] = -np.inf
                heapq.heappush(heap, (filled[r, c], counter, r, c))
                counter += 1
    nextafter = math.nextafter
    inf = math.inf
    while heap:
        level, _, r, c = heapq.heappop(heap)
        for dr, dc in _D8_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not closed[nr, nc]:
                closed[nr, nc] = True
                if filled[nr, nc] <= level:
                    filled[nr, nc] = nextafter(level, inf)
                heapq.heappush(heap, (filled[nr, nc], counter, nr, nc))
                counter += 1
    return filled


def _d8_receivers(filled: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Flat index of each cell's steepest-descent D8 neighbour; -1 = drains off-domain."""
    h, w = filled.shape
    padded = np.pad(filled, 1, mode="constant", constant_values=-np.inf)
    best_drop = np.full((h, w), -np.inf)
    best_k = np.full((h, w), -1, dtype=np.int64)
    for k, (dr, dc) in enumerate(_D8_OFFSETS):
        dist = pixel_size_m * (math.sqrt(2.0) if dr and dc else 1.0)
        neighbour = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        with np.errstate(invalid="ignore"):
            drop = (filled - neighbour) / dist
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        best_k = np.where(better, k, best_k)
    rows, cols = np.mgrid[0:h, 0:w]
    offsets = np.array(_D8_OFFSETS)
    nr = rows + offsets[best_k, 0]
    nc = cols + offsets[best_k, 1]
    inside = (best_drop > 0) & (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w) & (best_k >= 0)
    inside &= np.isfinite(filled)
    receivers = np.where(inside, nr * w + nc, -1)
    return receivers.ravel()


def _flow_accumulation(filled: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Upslope cell count (incl. self), one sweep in topological (descending) order."""
    order = np.argsort(filled.ravel(), kind="stable")[::-1]
    acc = np.ones(filled.size, dtype=np.float64)
    down_list = down.tolist()
    acc_list = acc.tolist()
    for idx in order.tolist():
        receiver = down_list[idx]
        if receiver >= 0:
            acc_list[receiver] += acc_list[idx]
    return np.asarray(acc_list).reshape(filled.shape)


def _height_above_drainage(filled: np.ndarray, down: np.ndarray, drainage: np.ndarray) -> np.ndarray:
    """HAND along the flow path: z(p) - z(first drainage cell downstream of p)."""
    flat = filled.ravel()
    drain_level = np.full(flat.size, np.nan)
    order = np.argsort(flat, kind="stable")  # downstream cells (lower) first
    down_list = down.tolist()
    is_drain = drainage.ravel().tolist()
    level = drain_level.tolist()
    values = flat.tolist()
    for idx in order.tolist():
        if is_drain[idx] or down_list[idx] < 0:
            level[idx] = values[idx]
        else:
            level[idx] = level[down_list[idx]]
    hand_values = flat - np.asarray(level)
    hand_values[~np.isfinite(hand_values)] = np.nan
    return np.maximum(hand_values, 0.0, where=np.isfinite(hand_values), out=hand_values)


#: GLO-30 support scale: HAND is computed where the DEM carries information, not at 10 m.
DEM_SUPPORT_M = 30.0


def hand_on_dem_support(
    dem: np.ndarray,
    pixel_size_m: float,
    *,
    support_m: float = DEM_SUPPORT_M,
    accumulation_threshold: float | None = None,
) -> np.ndarray:
    """HAND computed at the DEM's native support (30 m), interpolated back to the grid.

    A 30 m DEM bilinearly resampled to 10 m contains no 10 m drainage information; routing
    flow on it only adds interpolation flats. So the 10 m DEM is block-averaged back to
    the support scale, HAND is computed there, and the result is interpolated bilinearly
    onto the analysis grid (``docs/03`` section 8). The same function serves the real
    ingest and the synthetic benchmark, so both are processed identically.

    ``accumulation_threshold`` (cells at support scale) defaults to
    ``clip(n_cells // 400, 8, 600)``: a drainage network dense enough to reach every
    valley of the AOI, sparse enough not to declare every hillslope gully a river.
    """
    from scipy import ndimage

    dem = np.asarray(dem, dtype=np.float64)
    factor = max(1, int(round(support_m / pixel_size_m)))
    h, w = dem.shape
    ph, pw = -(-h // factor) * factor, -(-w // factor) * factor
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem) if np.isfinite(dem).any() else 0.0)
    padded = np.pad(filled, ((0, ph - h), (0, pw - w)), mode="edge")
    coarse = padded.reshape(ph // factor, factor, pw // factor, factor).mean(axis=(1, 3))
    threshold = (float(np.clip(coarse.size // 400, 8, 600)) if accumulation_threshold is None
                 else float(accumulation_threshold))
    hand_coarse = hand(coarse, threshold, pixel_size_m * factor)
    if np.isfinite(hand_coarse).any():
        hand_coarse = np.nan_to_num(hand_coarse, nan=float(np.nanmax(hand_coarse)))
    else:
        hand_coarse = np.zeros_like(hand_coarse)
    fine = ndimage.zoom(hand_coarse, factor, order=1, mode="nearest", grid_mode=True)
    return np.maximum(fine[:h, :w], 0.0)


def build_validity_mask(
    layover: np.ndarray,
    shadow: np.ndarray,
    permanent_water: np.ndarray,
    nodata: np.ndarray,
    slope_deg: np.ndarray | None,
    *,
    slope_max_deg: float | None = 10.0,
    outside_aoi: np.ndarray | None = None,
) -> np.ndarray:
    """Pack the exclusions into a single ``uint8`` flag band.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` ``uint8`` of :class:`~floodfusion.config.ValidityFlag` bit flags. A
        pixel is valid for loss and metric iff its byte is exactly ``0``.

    Notes
    -----
    Bit flags rather than a collapsed boolean mask, so that error analysis can attribute
    an exclusion to its cause (``docs/09`` section 6) — "18 % of this basin was excluded
    for shadow" is a finding; "18 % was excluded" is not.

    The excluded pixels are removed from **both** the loss and the metric. Including
    them in the loss teaches the model to fit noise; including them in the metric makes
    the metric depend on terrain rather than skill.

    Implementation notes
    --------------------
    1. The flags are OR-ed together per :class:`~floodfusion.config.ValidityFlag`.
    2. :func:`validity_flag_counts` gives the per-flag pixel counts — the per-mask ablation
       (``P1-M2.4``) needs them, and they are the fastest way to notice that a mask is not
       being applied at all.
    3. Slope and HAND exclusions overlap substantially, so their individual IoU
       contributions do **not** sum to their joint contribution; attribution must report
       the joint effect too and not claim additivity.
    """
    layers = {
        ValidityFlag.LAYOVER: layover,
        ValidityFlag.SHADOW: shadow,
        ValidityFlag.PERMANENT_WATER: permanent_water,
        ValidityFlag.NODATA: nodata,
    }
    shape = np.shape(layover)
    mask = np.zeros(shape, dtype=np.uint8)
    for flag, layer in layers.items():
        layer = np.asarray(layer, dtype=bool)
        if layer.shape != shape:
            raise ValueError(f"{flag.name} layer shape {layer.shape} != {shape}")
        mask |= np.where(layer, np.uint8(flag), np.uint8(0))
    if outside_aoi is not None:
        mask |= np.where(np.asarray(outside_aoi, dtype=bool), np.uint8(ValidityFlag.OUTSIDE_AOI), np.uint8(0))
    # STEEP_SLOPE is optional: the stored training/metric VALIDITY band is built with
    # slope_max_deg=None (docs/05 section 4.1 lists only layover, shadow, permanent water,
    # nodata and out-of-AOI); the steep-slope exclusion belongs to the R4 product layer.
    if slope_max_deg is not None and slope_deg is not None:
        slope = np.asarray(slope_deg, dtype=np.float64)
        steep = np.isfinite(slope) & (slope > slope_max_deg)
        mask |= np.where(steep, np.uint8(ValidityFlag.STEEP_SLOPE), np.uint8(0))
    return mask


def validity_flag_counts(mask: np.ndarray) -> dict[str, int]:
    """Per-cause pixel counts of a validity band (``P1-M2.4``, ``docs/09`` section 6).

    Causes overlap (a shadowed pixel can also be steep), so the counts do not sum to the
    number of invalid pixels; ``"INVALID"`` reports that union separately.
    """
    m = np.asarray(mask, dtype=np.uint8)
    counts = {
        flag.name: int(np.count_nonzero(m & np.uint8(flag)))
        for flag in ValidityFlag if flag is not ValidityFlag.VALID
    }
    counts["INVALID"] = int(np.count_nonzero(m))
    counts["TOTAL"] = int(m.size)
    return counts


def permanent_water_from_occurrence(
    occurrence_percent: np.ndarray, threshold_percent: float = 80.0
) -> np.ndarray:
    """Permanent water from JRC Global Surface Water occurrence (``docs/05`` section 3.3).

    ``occurrence >= threshold`` (default 80 %). The JRC layer is an independent source —
    never Sentinel-2 SCL class 6 (``docs/02`` section 7.3). Values outside [0, 100]
    (JRC nodata is 255) are treated as "not permanent water".
    """
    occ = np.asarray(occurrence_percent, dtype=np.float64)
    if not 0.0 < threshold_percent <= 100.0:
        raise ValueError("threshold_percent must lie in (0, 100]")
    return np.isfinite(occ) & (occ >= threshold_percent) & (occ <= 100.0)


def quality_from_scl(scl: np.ndarray, *, dilate_px: int = 3, ramp_px: float = 5.0) -> np.ndarray:
    """Per-pixel optical quality in [0, 1] from the SCL cloud/shadow classes.

    The bad-class mask is **dilated** (cloud edges are under-detected by Sen2Cor) and the
    quality then ramps from 0 to 1 over ``ramp_px`` pixels with distance from it — a soft,
    per-pixel signal for the availability gate (``docs/07`` section 8.2), not a binary cut.
    """
    from scipy import ndimage

    bad = np.isin(np.asarray(scl), BAD_SCL_CLASSES)
    if dilate_px > 0:
        bad = ndimage.binary_dilation(bad, iterations=dilate_px)
    if not bad.any():
        return np.ones(np.shape(scl))
    distance = ndimage.distance_transform_edt(~bad)
    return np.clip(distance / ramp_px, 0.0, 1.0)
