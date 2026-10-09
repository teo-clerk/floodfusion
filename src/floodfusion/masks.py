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

import numpy as np

__all__ = [
    "cast_shadow_and_layover",
    "horn_slope_aspect",
    "local_incidence_angle",
    "hand",
    "build_validity_mask",
]


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

    TODO(student)
    -------------
    1. Implement the two finite differences with array slicing, not a Python loop.
    2. Handle borders explicitly — replicate-pad, then trim — and document the choice.
    3. Convert to degrees and wrap aspect into ``[0, 360)``.
    4. Feed aspect downstream as :math:`(\sin A, \cos A)`, never as degrees: 359 and 1
       are adjacent directions but numerically maximally distant
       (``docs/03`` section 9).
    """
    raise NotImplementedError("Implement Horn slope. See docs/03 section 4.")


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

    TODO(student)
    -------------
    1. Implement the formula, keeping everything in radians internally.
    2. Assert the three checks above in ``tests/test_terrain.py``.
    3. Support ``incidence_deg`` as a full array — it varies across the swath from 29 to
       46 degrees, and using a scalar mid-swath value discards a real effect
       (``docs/01`` sections 4.2, 6.1).
    """
    raise NotImplementedError("Implement the local incidence angle. See docs/03 section 5.1.")


def cast_shadow_and_layover(
    elevation: np.ndarray,
    incidence_deg: np.ndarray | float,
    look_azimuth_deg: float,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
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

    ``P1-M2.4`` requires you to report the IoU improvement from the shadow mask alone,
    and ``docs/03`` exercise 4 asks what fraction of true shadow the local test misses.
    Expect it to be large in mountainous terrain.

    TODO(student)
    -------------
    1. Implement the local tests from :func:`local_incidence_angle`.
    2. Implement the cast sweep. Rotating the array to align an axis with :math:`\phi`
       is the simplest correct approach; interpolating along rays avoids the double
       resampling but is fiddlier.
    3. Combine local and cast with a logical OR.
    4. Compare the two masks' pixel counts and report the difference.
    """
    raise NotImplementedError("Implement the masks. See docs/03 section 5.3.")


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
    density and therefore every HAND value. Report a sensitivity check at two thresholds
    (``docs/03`` exercise 6).

    TODO(student)
    -------------
    1. Implement or wrap the five steps. ``richdem`` or ``pysheds`` are legitimate
       dependencies here — hydrological conditioning is not the learning target.
    2. Feed the result downstream as :math:`\log(1 + \mathrm{HAND})`: the informative
       range is 0--30 m and a log compresses the long hillslope tail.
    3. Use it as a **soft prior** — an input channel — with a hard exclusion only at
       extreme values (say > 50 m). A hard HAND mask makes dam breaks, urban pluvial
       flooding and coastal surge undetectable by construction.
    """
    raise NotImplementedError("Implement HAND. See docs/03 section 8.")


def build_validity_mask(
    layover: np.ndarray,
    shadow: np.ndarray,
    permanent_water: np.ndarray,
    nodata: np.ndarray,
    slope_deg: np.ndarray,
    *,
    slope_max_deg: float = 10.0,
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

    TODO(student)
    -------------
    1. OR the flags together per :class:`~floodfusion.config.ValidityFlag`.
    2. Report the per-flag pixel counts — ``P1-M2.4``'s per-mask ablation needs them,
       and they are the fastest way to notice that a mask is not being applied at all.
    3. Note that slope and HAND exclusions overlap substantially, so their individual
       IoU contributions will **not** sum to their joint contribution. Report the joint
       effect too, and do not claim additivity.
    """
    raise NotImplementedError("Implement the validity mask. See docs/03 sections 5.4, 9.")
