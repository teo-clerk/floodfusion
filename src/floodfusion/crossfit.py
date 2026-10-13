r"""Cross-fitted thresholds: calibrate each held-out event on the *other* held-out events.

In cross-validation every event is scored by the fold model that never saw it, at thresholds
fitted on that fold's single validation event. A threshold fitted on one event transfers badly
to another (v1.0: the SAR-only Spain fold scored 0.581 at the raw threshold, 0.271 calibrated).
Cross-fitting uses what the CV already produced: the held-out predictions of the other K - 1
events, each from its own fold model.

- :func:`crossfit_cost_threshold` — the cost-optimal threshold :math:`\arg\min_t\, r\,FN + FP`
  on the pooled confusion of the other events, applied to event k.
- :func:`crossfit_conformal` — conformal risk control on the other events' chips:
  :math:`\hat\lambda = \max\{t : \tfrac{n}{n+1}\hat R_n(t) + \tfrac{1}{n+1} \le \alpha\}` with
  :math:`\hat R_n` the mean per-chip false-negative rate over the :math:`n` pooled chips.
  This is chip-level control under pooling; it promises nothing per event. At the **event**
  level, with :math:`n = K - 1 = 9` events and a loss bounded by 1, the same bound needs
  :math:`\hat R = 0` for :math:`\alpha = 0.1` — vacuous (flag everything); :func:`event_level_bound`
  states that arithmetic.

Both use only stored confusion counts (``eval_test.json``), so they need no inference. Every
fold's probabilities are calibrated by its own model; cross-fitting moves the threshold only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from floodfusion.metrics import ConfusionMatrix

__all__ = ["MIN_FLOOD_PIXELS", "common_grid", "crossfit_conformal", "crossfit_cost_threshold", "event_level_bound"]

#: A chip enters the conformal calibration only with this many flood pixels (as ``evaluate``).
MIN_FLOOD_PIXELS = 100


def common_grid(curves: Mapping[str, tuple[Sequence[float], Sequence[ConfusionMatrix]]], *,
                digits: int = 4) -> tuple[list[float], dict[str, list[ConfusionMatrix]]]:
    """The thresholds every event was scored at, and each event's confusion at exactly those."""
    keyed = {e: {round(float(t), digits): cm for t, cm in zip(ts, cms)} for e, (ts, cms) in curves.items()}
    shared = sorted(set.intersection(*(set(k) for k in keyed.values()))) if keyed else []
    return shared, {e: [k[t] for t in shared] for e, k in keyed.items()}


def crossfit_cost_threshold(curves: Mapping[str, tuple[Sequence[float], Sequence[ConfusionMatrix]]],
                            cost_ratio: float) -> dict[str, dict[str, float]]:
    """Per event: the threshold minimising ``r FN + FP`` on the other events, and the event's IoU there."""
    grid, cms = common_grid(curves)
    out = {}
    for event in cms:
        others = [cms[e] for e in cms if e != event]
        if not others or not grid:
            continue
        pooled = [sum((o[i] for o in others), ConfusionMatrix(0, 0, 0, 0)) for i in range(len(grid))]
        best = int(np.argmin([cost_ratio * c.fn + c.fp for c in pooled]))
        cm = cms[event][best]
        out[event] = {"threshold": grid[best], "iou": cm.iou, "precision": cm.precision, "recall": cm.recall}
    return out


def _chip_fnr(cm: ConfusionMatrix) -> float:
    positives = cm.tp + cm.fn
    return cm.fn / positives if positives else math.nan


def crossfit_conformal(chips: Mapping[str, tuple[Sequence[float], Mapping[str, Sequence[ConfusionMatrix]]]],
                       alpha: float) -> dict[str, dict[str, float]]:
    """Per event: the conformal threshold fitted on the other events' chips, and what it gives on this one.

    ``chips[event] = (thresholds, {chip: [confusion at each threshold]})``. Returns the threshold,
    the calibration size, and on the event: mean per-chip FNR (chips with >= :data:`MIN_FLOOD_PIXELS`
    flood pixels), precision and IoU of the pooled confusion. ``feasible`` is False when no grid
    threshold satisfies the bound (flag everything).
    """
    curves = {e: (ts, [sum((c[i] for c in per.values()), ConfusionMatrix(0, 0, 0, 0)) for i in range(len(ts))])
              for e, (ts, per) in chips.items()}
    grid, _ = common_grid(curves)
    index = {e: [list(np.round(np.asarray(ts, dtype=float), 4)).index(t) for t in grid] for e, (ts, _) in chips.items()}
    out = {}
    for event, (_, per) in chips.items():
        calib = [[c[index[e][i]] for i in range(len(grid))] for e, (_, p) in chips.items() if e != event
                 for c in p.values() if c[0].tp + c[0].fn >= MIN_FLOOD_PIXELS]
        n = len(calib)
        if not n or not grid:
            continue
        risk = [float(np.mean([_chip_fnr(c[i]) for c in calib])) for i in range(len(grid))]
        ok = [i for i in range(len(grid)) if n / (n + 1) * risk[i] + 1 / (n + 1) <= alpha]
        own = [c for c in per.values() if c[0].tp + c[0].fn >= MIN_FLOOD_PIXELS]
        if not ok:
            out[event] = {"feasible": False, "n_calibration_chips": n}
            continue
        i = max(ok)                       # the largest threshold that keeps the bound: the most precise
        j = index[event][i]
        pooled = sum((c[j] for c in per.values()), ConfusionMatrix(0, 0, 0, 0))
        out[event] = {"feasible": True, "threshold": grid[i], "n_calibration_chips": n,
                      "mean_chip_fnr": float(np.nanmean([_chip_fnr(c[j]) for c in own])) if own else math.nan,
                      "precision": pooled.precision, "iou": pooled.iou}
    return out


def event_level_bound(n_events: int, alpha: float) -> float:
    """The largest calibration risk an event-level conformal bound tolerates: ``(alpha (n+1) - 1) / n``.

    Negative or zero means the bound is vacuous at this ``alpha`` with ``n`` calibration events.
    """
    return (alpha * (n_events + 1) - 1.0) / n_events
