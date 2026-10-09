r"""Basin-blocked splitting with buffering, and the frozen split file.

> **Your split determines your reported number more than your architecture does.**

Ablation row 13 of ``MILESTONES.md`` section 6 exists to measure exactly this: the same
model on a random chip split, which is expected to score **higher** than the blocked
split — not because it produces a better model, but because it produces a dishonest
measurement. If row 13 does not beat row 9, the blocking in this module is not working.

Why basins
----------
A hydrological basin is a natural unit of independence: its own drainage network, land
cover, terrain and flood dynamics. A model that transfers across basins has learned
something about water rather than about a particular river — and it matches the
operational question, since the next CEMS activation will be over a basin nobody has
seen.

References
----------
``docs/05_dataset_construction_and_splits.md`` section 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "SplitAssignment",
    "assign_basins",
    "empirical_variogram_range",
    "freeze_splits",
    "load_splits",
    "random_chip_split",
]


@dataclass(frozen=True)
class SplitAssignment:
    """The frozen split: explicit chip IDs and basin IDs per partition.

    Notes
    -----
    **Chip IDs are stored explicitly, not derived from a seed at runtime.** A split
    defined by ``random.seed(42)`` evaluated inside the training script is not a split;
    it is a function of your library versions and your iteration order, and it changes
    silently when either does. ``P1-M1``'s acceptance criterion requires a file on disk.

    Both the chip sets and the **basin sets** must be pairwise disjoint.
    ``tests/test_splits.py`` asserts both; the second assertion is the one that catches
    real bugs, because chip disjointness is trivially satisfied by a random split that
    leaks badly.
    """

    train_chips: tuple[str, ...]
    val_chips: tuple[str, ...]
    test_chips: tuple[str, ...]
    train_basins: tuple[str, ...]
    val_basins: tuple[str, ...]
    test_basins: tuple[str, ...]
    buffer_m: float
    strategy: str  # "leave_one_basin_out" | "random_chip" (row 13 only)


def empirical_variogram_range(
    values: Sequence[float],
    coordinates: Sequence[tuple[float, float]],
    *,
    max_lag_m: float = 20_000.0,
    n_bins: int = 40,
) -> float:
    r"""Estimate the spatial autocorrelation range from an empirical variogram.

    Parameters
    ----------
    values
        Sampled values of the variable — use :math:`\gamma^0_{\mathrm{VV}}`, and
        separately the label.
    coordinates
        Projected ``(x, y)`` in metres. Must be a **metric** CRS.
    max_lag_m, n_bins
        Lag binning.

    Returns
    -------
    float
        The lag at which the variogram flattens — beyond which samples are effectively
        uncorrelated.

    Notes
    -----
    .. math::

        \hat{\gamma}(h) = \frac{1}{2\,|N(h)|}
            \sum_{(i,j) \in N(h)} \bigl[z(\mathbf{x}_i) - z(\mathbf{x}_j)\bigr]^2

    over pairs separated by approximately :math:`h`. Plot :math:`\hat{\gamma}` against
    :math:`h`: it rises and then flattens, and the lag at which it flattens is the
    **range**.

    **The buffer width in** :func:`assign_basins` **must exceed this.** Do not guess it;
    ``P1-M1`` and ``docs/05`` exercise 1 require you to measure it.

    TODO(student)
    -------------
    1. Subsample pairs — all pairs is :math:`O(N^2)` and unnecessary; a few :math:`10^5`
       random pairs estimates the variogram well.
    2. Bin by lag and average the squared differences.
    3. Detect the flattening — fit a spherical or exponential model, or apply a simple
       threshold such as 95 % of the sill. Report which you used.
    4. Plot it. The plot goes in your ``P1-M6`` report.
    """
    raise NotImplementedError("Implement the variogram. See docs/05 section 6.2.")


def assign_basins(
    chip_basins: Mapping[str, str],
    chip_centroids: Mapping[str, tuple[float, float]],
    basin_boundaries: object,
    *,
    buffer_m: float,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> SplitAssignment:
    """Partition **basins** — never chips — into train, validation and test.

    Parameters
    ----------
    chip_basins
        ``chip_id -> basin_id``.
    chip_centroids
        ``chip_id -> (x, y)`` in the analysis CRS.
    basin_boundaries
        Basin polygons, e.g. a ``geopandas.GeoDataFrame`` from HydroBASINS.
    buffer_m
        Chips within this distance of a basin boundary are **dropped**. Must exceed the
        :func:`empirical_variogram_range` measurement.

    Returns
    -------
    SplitAssignment

    Notes
    -----
    The procedure (``docs/05`` section 6.4):

    1. Assign each chip a basin by the basin containing its **centroid**.
    2. Drop chips whose footprint crosses a basin boundary, or assign by majority
       overlap and **record the choice**.
    3. Apply the buffer: drop chips within ``buffer_m`` of a boundary.
    4. Partition **basins** into the three splits.
    5. Balance the folds by **flood-pixel count**, not by chip count — one basin can
       easily dominate the positive class, and a test split with almost no flood pixels
       produces a meaningless IoU with an enormous confidence interval.

    Four further channels of leakage survive correct basin blocking and are **not**
    handled here; you must handle them elsewhere (``docs/05`` section 6.5):

    - normalisation statistics computed over all data rather than train only;
    - model selection on the test set;
    - overlapping chips straddling a split boundary — **split by basin first, then
      chip**, not the reverse;
    - the permanent-water shortcut, which lets a model score well by memorising the
      river network from the DEM channels.

    TODO(student)
    -------------
    1. Implement steps 1--5.
    2. Assert basin sets are pairwise disjoint before returning.
    3. Report how many chips the buffer discarded. If it is more than ~20 %, either your
       basins are too small for the chip size or your variogram range is implausibly
       large — investigate rather than proceeding.
    """
    raise NotImplementedError("Implement basin assignment. See docs/05 section 6.4.")


def random_chip_split(
    chip_ids: Sequence[str],
    chip_basins: Mapping[str, str],
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> SplitAssignment:
    """A deliberately leaky random chip split — **ablation row 13 only**.

    Notes
    -----
    This function exists to be *wrong*, so that you can measure how wrong. It splits
    chips at random, so a test chip almost always has a training chip from the same
    river reach, with the same water, the same fields and the same acquisition. The
    model does not need to generalise; it needs to recognise.

    Row 13 is expected to score **higher** than row 9. The gap is your quantitative
    statement of how much published EO results can be inflated by an inappropriate
    split, and it is likely the single most valuable number in your report
    (``docs/09`` section 5.3).

    **Never use this for a headline number.** ``strategy`` is recorded on the returned
    object precisely so that ``evaluate.py`` can refuse to report it as one.

    TODO(student)
    -------------
    1. Shuffle chip IDs with the given seed and partition by fraction.
    2. Populate the basin tuples too — they will overlap, which is the point, and the
       test asserts that this strategy is the *only* one for which they do.
    """
    raise NotImplementedError("Implement the random split. See docs/09 section 5.3.")


def freeze_splits(assignment: SplitAssignment, path: Path | str) -> None:
    """Write the split to disk as JSON. Commit the result.

    TODO(student)
    -------------
    1. Serialise every field, including ``buffer_m`` and ``strategy``.
    2. Include the manifest's content hash so a split can never be silently paired with
       a regenerated dataset (``docs/05`` section 9).
    3. Write atomically — to a temporary file, then rename — so an interrupted write
       cannot leave a truncated split file that later loads as a valid but wrong split.
    """
    raise NotImplementedError("Implement split freezing. See docs/05 section 6.4 step 7.")


def load_splits(path: Path | str) -> SplitAssignment:
    """Read a frozen split, validating its invariants.

    Raises
    ------
    ValueError
        If chip sets or basin sets overlap, unless ``strategy == "random_chip"``.

    Notes
    -----
    Validating on **load** rather than only on creation means a hand-edited or
    accidentally regenerated split file cannot silently poison a run three weeks later.

    TODO(student)
    -------------
    1. Parse the JSON into a :class:`SplitAssignment`.
    2. Assert pairwise disjointness of both chip and basin sets.
    3. Allow basin overlap **only** when ``strategy == "random_chip"``, and emit a loud
       warning naming ablation row 13 when you do.
    """
    raise NotImplementedError("Implement split loading. See docs/05 section 6.4.")
