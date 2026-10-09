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

This module is torch-free. Its CLI (``python -m floodfusion.splits``) measures the
autocorrelation range, refuses a buffer that does not exceed it, freezes the
leave-one-basin-out split and the row-13 leaky split to JSON bound to the manifest hash,
and computes the normalisation statistics on the **training split only**.

References
----------
``docs/05_dataset_construction_and_splits.md`` section 6.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from floodfusion.data.records import ChipRecord, atomic_write_bytes

__all__ = [
    "SplitAssignment",
    "assign_basins",
    "compute_norm_stats",
    "empirical_variogram_range",
    "freeze_splits",
    "load_norm_stats",
    "load_splits",
    "random_chip_split",
    "test_pixels_covered_by_training",
]

LOGGER = logging.getLogger(__name__)

STRATEGY_LOBO = "leave_one_basin_out"
STRATEGY_RANDOM = "random_chip"
#: A dataset's own published split, used as published (e.g. Sen1Floods11's): its groups may
#: overlap by the publisher's design. Reported as within-group, never as generalisation.
STRATEGY_PUBLISHED = "published"


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

    ``manifest_sha256`` binds the split to one specific dataset: a split can never be
    silently paired with a regenerated manifest.
    """

    train_chips: tuple[str, ...]
    val_chips: tuple[str, ...]
    test_chips: tuple[str, ...]
    train_basins: tuple[str, ...]
    val_basins: tuple[str, ...]
    test_basins: tuple[str, ...]
    buffer_m: float
    strategy: str  # "leave_one_basin_out" | "random_chip" (row 13 only)
    manifest_sha256: str = ""

    def chips(self, split: str) -> tuple[str, ...]:
        return {"train": self.train_chips, "val": self.val_chips, "test": self.test_chips}[split]


def empirical_variogram_range(
    values: Sequence[float],
    coordinates: Sequence[tuple[float, float]],
    *,
    max_lag_m: float = 20_000.0,
    n_bins: int = 40,
    max_pairs: int = 200_000,
    sill_fraction: float = 0.95,
    seed: int = 0,
    return_curve: bool = False,
) -> float | tuple[float, dict[str, np.ndarray]]:
    r"""Estimate the spatial autocorrelation range from an empirical variogram.

    .. math::

        \hat{\gamma}(h) = \frac{1}{2\,|N(h)|}
            \sum_{(i,j) \in N(h)} \bigl[z(\mathbf{x}_i) - z(\mathbf{x}_j)\bigr]^2

    estimated from up to ``max_pairs`` random pairs (all pairs is :math:`O(N^2)` and
    unnecessary), binned by lag up to ``max_lag_m``. The **range** is the centre of the
    first lag bin whose semivariance reaches ``sill_fraction`` (95 %) of the sill, with the
    sill taken as the sample variance (the plateau of a second-order stationary field,
    :math:`\gamma(\infty) = C(0)`). That threshold rule is the method used and reported.

    **The buffer width in** :func:`assign_basins` **must exceed this.** Stationarity and
    isotropy are assumptions, so treat the result as a lower bound on the buffer you need
    (``docs/05`` section 6.2), never as a certificate.

    Returns
    -------
    float
        The range in metres (> 0; ``max_lag_m`` if the sill is never reached). With
        ``return_curve`` also ``{"lag", "gamma", "count", "sill"}`` for the report plot.
    """
    z = np.asarray(values, dtype=np.float64).ravel()
    xy = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    if z.size != xy.shape[0]:
        raise ValueError("values and coordinates must have the same length")
    ok = np.isfinite(z) & np.isfinite(xy).all(axis=1)
    z, xy = z[ok], xy[ok]
    if z.size < 3:
        raise ValueError("need at least three finite samples for a variogram")
    if max_lag_m <= 0 or n_bins < 2:
        raise ValueError("max_lag_m must be positive and n_bins >= 2")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, z.size, max_pairs)
    j = rng.integers(0, z.size, max_pairs)
    d = np.hypot(*(xy[i] - xy[j]).T)
    keep = (i != j) & (d > 0) & (d <= max_lag_m)
    d, sq = d[keep], 0.5 * (z[i[keep]] - z[j[keep]]) ** 2
    edges = np.linspace(0.0, max_lag_m, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    total = np.bincount(idx, weights=sq, minlength=n_bins)
    gamma = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    sill = float(np.var(z))
    bin_width = float(edges[1] - edges[0])
    if sill <= 0:
        rng_m = 0.5 * bin_width
    else:
        reached = np.nonzero(np.nan_to_num(gamma, nan=-np.inf) >= sill_fraction * sill)[0]
        rng_m = float(centres[reached[0]]) if reached.size else float(max_lag_m)
    rng_m = max(rng_m, 0.5 * bin_width)
    if return_curve:
        return rng_m, {"lag": centres, "gamma": gamma, "count": count, "sill": np.asarray(sill)}
    return rng_m


def _boundaries_as_dict(basin_boundaries: object) -> dict[str, Any] | None:
    """Accept ``None``, ``{basin_id: geometry}`` or a GeoDataFrame with a ``basin_id`` column."""
    if basin_boundaries is None:
        return None
    if isinstance(basin_boundaries, Mapping):
        return {str(k): v for k, v in basin_boundaries.items()}
    columns = getattr(basin_boundaries, "columns", None)
    if columns is not None and "basin_id" in columns:
        return {str(b): g for b, g in zip(basin_boundaries["basin_id"], basin_boundaries.geometry)}
    raise TypeError("basin_boundaries must be None, a mapping basin_id -> geometry, or a GeoDataFrame")


#: Up to this many basins the partition is searched exhaustively (3**13 ≈ 1.6 M candidates).
EXHAUSTIVE_MAX_BASINS = 13
#: A basin is "optical-bearing" if it holds at least this share of the optical weight.
OPTICAL_BEARING_SHARE = 0.05
#: The optical weight counts evaluation chips this clear — row 6's own definition.
OPTICAL_CLEAR_FOR_SPLITS = 0.9


def _partition_exhaustive(
    basins: Sequence[str], weights: Mapping[str, float], secondary: Mapping[str, float],
    targets: Mapping[str, float], seed: int,
) -> dict[str, str] | None:
    """Best partition over all 3**K assignments: shares of flood weight AND optical weight as
    close as possible to the targets, every split non-empty, and the optical case spread over
    the splits: a basin is *optical-bearing* if it holds at least
    :data:`OPTICAL_BEARING_SHARE` of the optical weight; with five or more such basins train
    and test need two each and validation one (with three or four, one each). Ties are broken
    by a seeded random key, so the result is deterministic."""
    k = len(basins)
    names = ("train", "val", "test")
    codes = np.indices((3,) * k).reshape(k, -1).T.astype(np.int8)  # (3**k, k)
    w = np.array([weights[b] for b in basins], dtype=np.float64)
    o = np.array([secondary.get(b, 0.0) for b in basins], dtype=np.float64)
    t = np.array([targets[n] for n in names])
    onehot = codes[:, :, None] == np.arange(3)[None, None, :]          # (N, k, 3)
    share_w = (onehot * w[None, :, None]).sum(1) / max(w.sum(), 1e-12)
    share_o = (onehot * o[None, :, None]).sum(1) / max(o.sum(), 1e-12)
    n_basins = onehot.sum(1)
    feasible = (n_basins >= 1).all(1)
    if o.sum() > 0:
        bearing = o >= OPTICAL_BEARING_SHARE * o.sum()
        n_optical = (onehot & bearing[None, :, None]).sum(1)
        n_bearing = int(bearing.sum())
        need = np.array([2, 1, 2]) if n_bearing >= 5 else np.array([1, 1, 1]) if n_bearing >= 3 else np.zeros(3)
        feasible &= (n_optical >= need[None, :]).all(1)
    if not feasible.any():
        return None
    cost = ((share_w - t) ** 2).sum(1) + ((share_o - t) ** 2).sum(1) if o.sum() > 0 else ((share_w - t) ** 2).sum(1)
    cost = np.where(feasible, cost, np.inf)
    best = np.flatnonzero(np.isclose(cost, cost.min(), rtol=0.0, atol=1e-12))
    pick = best[int(np.random.default_rng(seed).integers(best.size))]
    return {b: names[int(c)] for b, c in zip(basins, codes[pick])}


def _partition_basins(
    weights: Mapping[str, float], train_fraction: float, val_fraction: float, seed: int,
    secondary: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Greedy, deterministic partition of basins balanced by weight (flood pixels).

    Basins are visited heaviest first (ties broken by a seeded shuffle) and each goes to the
    split furthest below its target share. With ``secondary`` weights (optical-available
    evaluation area) the optical case must be spread over the splits too — a test set whose
    optical chips all sit in one basin cannot support any statement about what optical adds
    (row 9 vs row 11, row 6) — and two objectives defeat a greedy pass, so up to
    :data:`EXHAUSTIVE_MAX_BASINS` basins the partition is searched exhaustively
    (:func:`_partition_exhaustive`); beyond that the greedy pass sums both deficits. With
    three or more basins every split gets at least one.
    """
    targets = {"train": train_fraction, "val": val_fraction, "test": 1.0 - train_fraction - val_fraction}
    basins = sorted(weights)
    if secondary is not None and 3 <= len(basins) <= EXHAUSTIVE_MAX_BASINS:
        exact = _partition_exhaustive(basins, weights, secondary, targets, seed)
        if exact is not None:
            return exact
    rng = np.random.default_rng(seed)
    tiebreak = {b: float(r) for b, r in zip(basins, rng.random(len(basins)))}
    order = sorted(basins, key=lambda b: (-weights[b], tiebreak[b]))
    total = sum(weights.values()) or 1.0
    second = {b: float((secondary or {}).get(b, 0.0)) for b in basins}
    total_second = sum(second.values())
    assigned: dict[str, str] = {}
    load = dict.fromkeys(targets, 0.0)
    load_second = dict.fromkeys(targets, 0.0)
    empty = [s for s in ("test", "val", "train") if targets[s] > 0]

    def deficit(split: str, basin: str) -> float:
        primary = targets[split] - load[split] / total
        if total_second <= 0 or second[basin] <= 0:
            return primary
        return primary + (targets[split] - load_second[split] / total_second)

    for basin in order:
        remaining = len(order) - len(assigned)
        if len(empty) and remaining <= len(empty):
            split = empty[0]          # make sure no split stays empty
        else:
            split = max(targets, key=lambda s: deficit(s, basin))
        assigned[basin] = split
        load[split] += weights[basin]
        load_second[split] += second[basin]
        if split in empty:
            empty.remove(split)
    return assigned


def assign_basins(
    chip_basins: Mapping[str, str],
    chip_centroids: Mapping[str, tuple[float, float]],
    basin_boundaries: object,
    *,
    buffer_m: float,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 0,
    chip_size_m: float = 2560.0,
    basin_weights: Mapping[str, float] | None = None,
    optical_weights: Mapping[str, float] | None = None,
    report: dict[str, Any] | None = None,
) -> SplitAssignment:
    """Partition **basins** — never chips — into train, validation and test.

    Parameters
    ----------
    chip_basins
        ``chip_id -> basin_id`` (by centroid, from the manifest).
    chip_centroids
        ``chip_id -> (x, y)`` in the analysis CRS, metres.
    basin_boundaries
        Basin polygons (``{basin_id: geometry}`` or a GeoDataFrame with ``basin_id``), or
        ``None`` — then the distance to another basin is estimated from the nearest chip
        footprint of that basin.
    buffer_m
        Chips within this distance of a basin in **another split** are dropped. Must exceed
        :func:`empirical_variogram_range` (enforced by the CLI).
    chip_size_m
        Chip footprint side, for the crossing and buffer tests.
    basin_weights
        Per-basin balancing weight — flood-pixel counts, not chip counts; one basin can
        dominate the positive class (``docs/05`` section 6.4 step 6). Defaults to chip counts.
    optical_weights
        Optional per-basin optical-available area, balanced as a second objective so the
        optical case reaches every split (see :func:`_partition_basins`).
    report
        Optional dict filled with per-reason drop counts and the basin partition.

    Notes
    -----
    The procedure (``docs/05`` section 6.4):

    1. A chip belongs to the basin containing its **centroid** (checked against the
       polygons when given).
    2. Chips whose footprint crosses a basin boundary are **dropped** (recorded).
    3. **Basins** are partitioned, balanced by flood-pixel weight.
    4. Chips within ``buffer_m`` of a basin assigned to a **different** split are dropped.
       Buffering against *other-split* basins only (after partitioning) gives exactly the
       leakage guarantee of applying the buffer at every boundary before partitioning,
       without discarding chips between two basins of the same split — a documented
       deviation from the order of section 6.4.
    5. Basin and chip sets are asserted pairwise disjoint.

    Four further leakage channels survive correct blocking and are handled elsewhere
    (``docs/05`` section 6.5): normalisation statistics (:func:`compute_norm_stats`
    uses the training split only), model selection on test (the evaluator touches test
    once), overlapping chips across a split boundary (prevented by steps 2 and 4), and
    the permanent-water shortcut (permanent water is excluded from loss and metric).
    """
    from shapely.geometry import Point, box

    if buffer_m < 0:
        raise ValueError("buffer_m must be non-negative")
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1 or train_fraction + val_fraction > 1:
        raise ValueError("train/val fractions must be positive and sum to at most 1")
    info: dict[str, Any] = report if report is not None else {}
    chips = sorted(chip_basins)
    if not chips:
        info.update(n_chips=0, dropped={})
        return SplitAssignment((), (), (), (), (), (), float(buffer_m), STRATEGY_LOBO)

    geoms = _boundaries_as_dict(basin_boundaries)
    half = chip_size_m / 2.0
    dropped: Counter[str] = Counter()
    kept: list[str] = []
    for chip in chips:
        basin = str(chip_basins[chip])
        if geoms is not None:
            geom = geoms.get(basin)
            if geom is None:
                dropped["basin_without_polygon"] += 1
                continue
            cx, cy = chip_centroids[chip]
            if not geom.contains(Point(cx, cy)):
                dropped["centroid_outside_basin"] += 1
                continue
            if not geom.buffer(1e-6).contains(box(cx - half, cy - half, cx + half, cy + half)):
                dropped["footprint_crosses_boundary"] += 1
                continue
        kept.append(chip)

    basins = sorted({str(chip_basins[c]) for c in kept})
    if basin_weights is None:
        weights = Counter(str(chip_basins[c]) for c in kept)
        weight_map = {b: float(weights[b]) for b in basins}
    else:
        weight_map = {b: float(basin_weights.get(b, 0.0)) for b in basins}
    split_of_basin = (_partition_basins(weight_map, train_fraction, val_fraction, seed, secondary=optical_weights)
                      if basins else {})

    def split_of(chip: str) -> str:
        return split_of_basin[str(chip_basins[chip])]

    final: list[str] = []
    if geoms is not None:
        from shapely.ops import unary_union

        others = {
            s: unary_union([geoms[b] for b, sb in split_of_basin.items() if sb != s] or [box(0, 0, 0, 0)])
            for s in ("train", "val", "test")
        }
        for chip in kept:
            cx, cy = chip_centroids[chip]
            footprint = box(cx - half, cy - half, cx + half, cy + half)
            other = others[split_of(chip)]
            if not other.is_empty and footprint.distance(other) < buffer_m:
                dropped["within_buffer_of_other_split"] += 1
                continue
            final.append(chip)
    else:
        coords = {c: np.asarray(chip_centroids[c], dtype=np.float64) for c in kept}
        for chip in kept:
            mine = split_of(chip)
            gaps = [
                max(0.0, float(np.hypot(*(coords[chip] - coords[o]))) - chip_size_m)
                for o in kept if split_of(o) != mine
            ]
            if gaps and min(gaps) < buffer_m:
                dropped["within_buffer_of_other_split"] += 1
                continue
            final.append(chip)

    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for chip in final:
        by_split[split_of(chip)].append(chip)
    basins_by_split = {s: sorted(b for b, sb in split_of_basin.items() if sb == s) for s in by_split}
    assignment = SplitAssignment(
        train_chips=tuple(sorted(by_split["train"])),
        val_chips=tuple(sorted(by_split["val"])),
        test_chips=tuple(sorted(by_split["test"])),
        train_basins=tuple(basins_by_split["train"]),
        val_basins=tuple(basins_by_split["val"]),
        test_basins=tuple(basins_by_split["test"]),
        buffer_m=float(buffer_m),
        strategy=STRATEGY_LOBO,
    )
    _assert_disjoint(assignment, allow_basin_overlap=False)
    n_dropped = sum(dropped.values())
    fraction = n_dropped / len(chips)
    info.update(n_chips=len(chips), n_kept=len(final), dropped=dict(dropped),
                dropped_fraction=fraction, basin_split=split_of_basin)
    level = logging.WARNING if fraction > 0.20 else logging.INFO
    LOGGER.log(level, "assign_basins: dropped %d of %d chips (%.0f %%): %s%s", n_dropped, len(chips),
               100 * fraction, dict(dropped),
               " — more than 20 %: basins too small for the chip size, or an implausibly large "
               "buffer; investigate" if fraction > 0.20 else "")
    return assignment


def random_chip_split(
    chip_ids: Sequence[str],
    chip_basins: Mapping[str, str],
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 0,
    fixed_test_chips: Sequence[str] | None = None,
    fixed_val_chips: Sequence[str] | None = None,
    match_counts: Mapping[str, tuple[int, int]] | None = None,
    positive_chips: Sequence[str] | None = None,
) -> SplitAssignment:
    """A deliberately leaky random chip split — **ablation row 13 only**.

    Notes
    -----
    This function exists to be *wrong*, so that you can measure how wrong. It splits
    chips at random, so a test chip almost always has a training chip from the same
    river reach, with the same water, the same fields and the same acquisition. The
    model does not need to generalise; it needs to recognise.

    **The paired variant used by the ablation grid.** With ``fixed_test_chips`` the test
    set is held identical to row 9's, so the 13-vs-9 gap can be tested with a *paired*
    basin bootstrap; train and validation are drawn at random from every other chip —
    including chips from the test basins and stride-128 windows overlapping test tiles,
    exactly what a naive chip-then-split pipeline does. ``match_counts``
    (``{"train": (n, n_pos), "val": (n, n_pos)}``) subsamples them to row 9's sizes and
    positive-chip counts, so the only difference between rows 9 and 13 is *where* the
    training chips come from — not how many there are. With ``fixed_val_chips`` the
    validation set is row 9's too, so calibration, early stopping and checkpoint selection
    see the same chips and the **training split is the only variable** (a validation set
    drawn from the leaky pool would carry a different prior and confound the gap).

    Row 13 is expected to score **higher** than row 9. **Never use this for a headline
    number**: ``strategy`` is recorded so the evaluator can refuse to.
    """
    ids = sorted(set(chip_ids))
    rng = np.random.default_rng(seed)
    if fixed_test_chips is None:
        perm = [ids[i] for i in rng.permutation(len(ids))]
        n_train = int(round(train_fraction * len(ids)))
        n_val = int(round(val_fraction * len(ids)))
        train, val, test = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    else:
        test = sorted(set(fixed_test_chips))
        fixed_val = sorted(set(fixed_val_chips or ()))
        unknown = (set(test) | set(fixed_val)) - set(ids)
        if unknown:
            raise ValueError(f"fixed test/val chips missing from chip_ids: {sorted(unknown)[:5]}")
        if set(test) & set(fixed_val):
            raise ValueError("fixed validation and test chips overlap")
        pool = [c for c in ids if c not in set(test) and c not in set(fixed_val)]
        pool = [pool[i] for i in rng.permutation(len(pool))]
        if fixed_val_chips is not None:
            val = fixed_val
            if match_counts is not None:
                positives = set(positive_chips or ())
                n_total, n_pos = match_counts["train"]
                pos_pool = [c for c in pool if c in positives]
                neg_pool = [c for c in pool if c not in positives]
                take_pos = min(n_pos, len(pos_pool))
                train = pos_pool[:take_pos] + neg_pool[:min(n_total - take_pos, len(neg_pool))]
            else:
                train = pool
        elif match_counts is not None:
            positives = set(positive_chips or ())
            pos_pool = [c for c in pool if c in positives]
            neg_pool = [c for c in pool if c not in positives]
            train, val = [], []
            for name, bucket in (("train", train), ("val", val)):
                n_total, n_pos = match_counts[name]
                take_pos = min(n_pos, len(pos_pool))
                take_neg = min(n_total - take_pos, len(neg_pool))
                bucket.extend(pos_pool[:take_pos] + neg_pool[:take_neg])
                pos_pool, neg_pool = pos_pool[take_pos:], neg_pool[take_neg:]
        else:
            n_train = int(round(len(pool) * train_fraction / (train_fraction + val_fraction)))
            train, val = pool[:n_train], pool[n_train:]

    def basins_of(chips: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted({str(chip_basins[c]) for c in chips}))

    return SplitAssignment(
        train_chips=tuple(sorted(train)), val_chips=tuple(sorted(val)), test_chips=tuple(sorted(test)),
        train_basins=basins_of(train), val_basins=basins_of(val), test_basins=basins_of(test),
        buffer_m=0.0, strategy=STRATEGY_RANDOM,
    )


def _assert_disjoint(assignment: SplitAssignment, *, allow_basin_overlap: bool) -> None:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a, b in pairs:
        chips = set(getattr(assignment, f"{a}_chips")) & set(getattr(assignment, f"{b}_chips"))
        if chips:
            raise ValueError(f"{a} and {b} share chips {sorted(chips)[:5]}")
        basins = set(getattr(assignment, f"{a}_basins")) & set(getattr(assignment, f"{b}_basins"))
        if basins and not allow_basin_overlap:
            raise ValueError(
                f"{a} and {b} share basins {sorted(basins)[:5]} — spatial leakage. Only "
                "strategy='random_chip' (ablation row 13) or a dataset's own 'published' split "
                "may overlap (docs/05 section 6)."
            )


def freeze_splits(
    assignment: SplitAssignment,
    path: Path | str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the split to disk as JSON (atomically). Commit the result.

    Every field is serialised — including ``buffer_m``, ``strategy`` and the manifest
    hash — plus free-form ``metadata`` (variogram ranges, drop report). **No validation
    happens here**: invariants are checked by :func:`load_splits`, so a hand-edited or
    regenerated file is caught when it is *used*, weeks later, not only when written.
    """
    payload = {f.name: getattr(assignment, f.name) for f in fields(SplitAssignment)}
    payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in payload.items()}
    if metadata:
        payload["metadata"] = dict(metadata)
    atomic_write_bytes(Path(path), json.dumps(payload, indent=2, default=str).encode("utf-8"))


def load_splits(path: Path | str, *, expected_manifest_sha256: str | None = None) -> SplitAssignment:
    """Read a frozen split, validating its invariants.

    Raises
    ------
    ValueError
        If chip sets overlap; if basin sets overlap, unless ``strategy == "random_chip"``
        (allowed with a loud warning naming ablation row 13); or if the split is bound to
        a different manifest than ``expected_manifest_sha256``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(SplitAssignment)}
    missing = sorted(known - set(raw) - {"manifest_sha256"})
    if missing:
        raise ValueError(f"{path}: split file is missing fields {missing}")
    values = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    values["buffer_m"] = float(values["buffer_m"])
    assignment = SplitAssignment(**values)
    if assignment.strategy not in {STRATEGY_LOBO, STRATEGY_RANDOM, STRATEGY_PUBLISHED}:
        raise ValueError(f"{path}: unknown split strategy {assignment.strategy!r}")
    leaky = assignment.strategy in {STRATEGY_RANDOM, STRATEGY_PUBLISHED}
    _assert_disjoint(assignment, allow_basin_overlap=leaky)
    if assignment.strategy == STRATEGY_RANDOM:
        warnings.warn(
            f"{path}: strategy='random_chip' — a deliberately LEAKY split for ablation row 13 "
            "only. Its score measures recognition, not generalisation; never report it as a "
            "headline number (docs/09 section 5.3).",
            UserWarning, stacklevel=2,
        )
    elif assignment.strategy == STRATEGY_PUBLISHED:
        warnings.warn(
            f"{path}: strategy='published' — a dataset's own split whose groups overlap by design. "
            "Its score is a within-group score, reported only next to a group-blocked one.",
            UserWarning, stacklevel=2,
        )
    if expected_manifest_sha256 and assignment.manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"{path}: split is bound to manifest {assignment.manifest_sha256[:12] or '<none>'}… but the "
            f"dataset is {expected_manifest_sha256[:12]}… — regenerate the split for this dataset"
        )
    return assignment


def test_pixels_covered_by_training(assignment: SplitAssignment, records: Sequence[ChipRecord]) -> float:
    """Fraction of test-chip pixels that also lie inside some training window (literal leakage)."""
    by_id = {r.chip_id: r for r in records}
    covered = total = 0
    train_by_event: dict[str, list[ChipRecord]] = {}
    for chip in assignment.train_chips:
        r = by_id[chip]
        train_by_event.setdefault(r.event_id, []).append(r)
    for chip in assignment.test_chips:
        r = by_id[chip]
        col, row, w, h = r.window
        mask = np.zeros((h, w), dtype=bool)
        for t in train_by_event.get(r.event_id, ()):
            tc, tr, tw, th = t.window
            c0, c1 = max(col, tc), min(col + w, tc + tw)
            r0, r1 = max(row, tr), min(row + h, tr + th)
            if c0 < c1 and r0 < r1:
                mask[r0 - row:r1 - row, c0 - col:c1 - col] = True
        covered += int(mask.sum())
        total += w * h
    return covered / total if total else 0.0


test_pixels_covered_by_training.__test__ = False  # not a pytest test despite the name


def compute_norm_stats(
    records: Sequence[ChipRecord],
    base: Any,
    grid_cfg: Any,
    *,
    max_pixels_per_chip: int = 4096,
    seed: int = 0,
) -> Any:
    """Per-channel SAR dB mean/std over valid pixels of the given (training) chips.

    Computed on the **training split only** and frozen: statistics over train+val+test
    leak test information into training (``docs/05`` section 6.5 point 1). Overlapping
    training windows would weight some pixels twice, so only evaluation-grid
    (non-overlapping) chips are used when available. Returns ``base`` with
    ``sar_mean_db`` / ``sar_std_db`` replaced.
    """
    import rasterio
    from rasterio.windows import Window

    from floodfusion.data.encoding import N_CHIP_BANDS, decode_physical

    chosen = [r for r in records if r.eval_grid] or list(records)
    if not chosen:
        raise ValueError("compute_norm_stats needs at least one record")
    rng = np.random.default_rng(seed)
    sums = np.zeros(5)
    sq = np.zeros(5)
    n = 0
    by_path: dict[str, list[ChipRecord]] = {}
    for r in chosen:
        by_path.setdefault(str(r.raster_path), []).append(r)
    for path, recs in sorted(by_path.items()):
        with rasterio.open(path) as src:
            for r in recs:
                col, row, w, h = r.window
                raw = src.read(list(range(1, N_CHIP_BANDS + 1)), window=Window(col, row, w, h))
                phys = decode_physical(raw, grid_cfg, base)
                valid = (phys.validity == 0) & np.isfinite(phys.gamma0_db).all(axis=0)
                values = phys.gamma0_db[:, valid]
                if values.shape[1] > max_pixels_per_chip:
                    values = values[:, rng.choice(values.shape[1], max_pixels_per_chip, replace=False)]
                sums += values.sum(axis=1)
                sq += (values**2).sum(axis=1)
                n += values.shape[1]
    if n == 0:
        raise ValueError("no valid pixels in the training chips")
    mean = sums / n
    std = np.sqrt(np.maximum(sq / n - mean**2, 1e-12))
    return replace(base, sar_mean_db=tuple(float(v) for v in mean), sar_std_db=tuple(float(v) for v in std))


def load_norm_stats(path: Path | str, base: Any, *, expected_manifest_sha256: str | None = None) -> Any:
    """Apply a frozen ``norm_stats.json`` to a :class:`NormalizationStats`.

    With ``expected_manifest_sha256`` the statistics must have been computed on that dataset:
    statistics of another manifest would silently shift every input.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bound = raw.get("manifest_sha256")
    if expected_manifest_sha256 and bound and bound != expected_manifest_sha256:
        raise ValueError(f"{path}: statistics computed on manifest {bound[:12]}…, dataset is "
                         f"{expected_manifest_sha256[:12]}… — recompute them for this dataset")
    return replace(base, sar_mean_db=tuple(raw["sar_mean_db"]), sar_std_db=tuple(raw["sar_std_db"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _sample_pixels(records: Sequence[ChipRecord], grid_cfg: Any, norm: Any, n_per_event: int, seed: int):
    """Random valid pixels per event: (VV post dB, label, x, y) for the variogram."""
    import rasterio

    from floodfusion.data.encoding import ChipBand

    rng = np.random.default_rng(seed)
    out = []
    for path in sorted({str(r.raster_path) for r in records}):
        with rasterio.open(path) as src:
            vv = src.read(int(ChipBand.VV_POST) + 1)
            label = src.read(int(ChipBand.LABEL) + 1)
            validity = src.read(int(ChipBand.VALIDITY) + 1)
            transform = src.transform
        nodata = grid_cfg.nodata_int16
        rows, cols = np.nonzero((validity == 0) & (vv != nodata) & (label != nodata))
        if rows.size == 0:
            continue
        pick = rng.choice(rows.size, min(n_per_event, rows.size), replace=False)
        rows, cols = rows[pick], cols[pick]
        xs, ys = transform * (cols + 0.5, rows + 0.5)
        out.append((vv[rows, cols] / grid_cfg.db_scale, label[rows, cols].astype(float), np.asarray(xs), np.asarray(ys)))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze the LOBO split, the row-13 split and the train-only normalisation stats."""
    import geopandas as gpd

    from floodfusion.config import load_config
    from floodfusion.data.records import load_manifest, load_manifest_meta
    from floodfusion.utils.env import git_state

    parser = argparse.ArgumentParser(description="Basin-blocked splits, variogram check and norm stats.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/row09_fusion_crossattn.yaml"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixels-per-event", type=int, default=30_000)
    parser.add_argument("--allow-short-buffer", action="store_true",
                        help="record but do not fail when the buffer does not exceed the variogram range")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, overlays=args.overlay)
    chip_dir = args.data_root / Path(cfg.data.chip_root).name
    manifest_path = chip_dir / cfg.data.manifest_name
    records = load_manifest(manifest_path)
    manifest_sha = load_manifest_meta(manifest_path)["manifest_sha256"]
    basins = gpd.read_file(args.data_root / "basins.gpkg")

    # 1. Autocorrelation range (gamma0 VV and the label, separately — docs/05 exercise 1).
    samples = _sample_pixels(records, cfg.grid, cfg.norm, args.pixels_per_event, args.seed)
    ranges: dict[str, list[float]] = {"vv": [], "label": []}
    for vv, label, xs, ys in samples:
        coords = list(zip(xs, ys))
        ranges["vv"].append(empirical_variogram_range(vv, coords, max_lag_m=8000, n_bins=40, seed=args.seed))
        if np.var(label) > 0:
            ranges["label"].append(empirical_variogram_range(label, coords, max_lag_m=8000, n_bins=40, seed=args.seed))
    measured = {k: float(np.median(v)) if v else float("nan") for k, v in ranges.items()}
    governing = float(np.nanmax(list(measured.values())))
    buffer_m = cfg.data.basin_buffer_m
    LOGGER.info("variogram range (median over events): VV %.0f m, label %.0f m; buffer %.0f m",
                measured["vv"], measured["label"], buffer_m)
    if not buffer_m > governing:
        message = (f"basin_buffer_m={buffer_m:.0f} m does not exceed the measured autocorrelation "
                   f"range {governing:.0f} m (docs/05 section 6.2): raise data.basin_buffer_m")
        if not args.allow_short_buffer:
            LOGGER.error(message)
            return 2
        LOGGER.warning(message)

    # 2. Leave-one-basin-out with buffer; val/test keep only the non-overlapping eval grid.
    area = cfg.data.chip_size**2
    weights: Counter[str] = Counter()
    optical: Counter[str] = Counter()
    for r in records:
        if r.eval_grid:
            weights[r.basin_id] += r.positive_fraction * r.valid_fraction * area
            if r.optical_available and r.optical_clear_fraction >= OPTICAL_CLEAR_FOR_SPLITS:
                optical[r.basin_id] += area
    report: dict[str, Any] = {}
    lobo = assign_basins(
        {r.chip_id: r.basin_id for r in records},
        {r.chip_id: (r.centroid_x, r.centroid_y) for r in records},
        basins, buffer_m=buffer_m, chip_size_m=cfg.data.chip_size * cfg.grid.pixel_size_m,
        basin_weights={b: float(weights.get(b, 0.0)) for b in basins["basin_id"]},
        optical_weights={b: float(optical.get(b, 0.0)) for b in basins["basin_id"]},
        seed=args.seed, report=report,
    )
    by_id = {r.chip_id: r for r in records}
    lobo = replace(
        lobo,
        val_chips=tuple(c for c in lobo.val_chips if by_id[c].eval_grid),
        test_chips=tuple(c for c in lobo.test_chips if by_id[c].eval_grid),
        manifest_sha256=manifest_sha,
    )
    common = {"variogram_range_m": measured, "governing_range_m": governing, **git_state()}
    freeze_splits(lobo, chip_dir / "splits.json", metadata={**common, "assign_report": report})

    # 3. Normalisation statistics from the TRAINING split only.
    train_records = [by_id[c] for c in lobo.train_chips]
    norm = compute_norm_stats(train_records, cfg.norm, cfg.grid, seed=args.seed)
    atomic_write_bytes(chip_dir / cfg.data.norm_stats_file, json.dumps({
        "sar_mean_db": list(norm.sar_mean_db), "sar_std_db": list(norm.sar_std_db),
        "computed_on": "train split of splits.json (evaluation-grid chips)",
        "manifest_sha256": manifest_sha,
    }, indent=2).encode("utf-8"))

    # 4. Row 13: same test chips, random train/val from everything else, matched sizes.
    threshold = cfg.data.positive_fraction_threshold
    positives = [r.chip_id for r in records if r.positive_fraction > threshold]
    pos_set = set(positives)

    def counts(chips: Sequence[str]) -> tuple[int, int]:
        return len(chips), sum(c in pos_set for c in chips)

    leaky = random_chip_split(
        [r.chip_id for r in records], {r.chip_id: r.basin_id for r in records},
        seed=args.seed, fixed_test_chips=lobo.test_chips, fixed_val_chips=lobo.val_chips,
        match_counts={"train": counts(lobo.train_chips), "val": counts(lobo.val_chips)},
        positive_chips=positives,
    )
    leaky = replace(leaky, manifest_sha256=manifest_sha)
    leak = test_pixels_covered_by_training(leaky, records)
    freeze_splits(leaky, chip_dir / "splits_random_chip.json", metadata={
        **common, "test_pixels_covered_by_training": leak,
        "note": "ablation row 13 only: test AND validation chips identical to splits.json; "
                "training chips drawn at random from every other chip (the only variable)",
    })
    LOGGER.info("LOBO: %d/%d/%d chips, basins %s/%s/%s; row-13 leak: %.0f %% of test pixels inside training windows",
                len(lobo.train_chips), len(lobo.val_chips), len(lobo.test_chips),
                list(lobo.train_basins), list(lobo.val_basins), list(lobo.test_basins), 100 * leak)
    LOGGER.info("train-only SAR stats: mean %s std %s", np.round(norm.sar_mean_db, 2), np.round(norm.sar_std_db, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
