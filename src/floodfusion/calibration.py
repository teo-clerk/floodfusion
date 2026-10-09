r"""Calibration: prior correction, temperature scaling, ECE (``docs/05`` 8.3, ``docs/09`` 4).

The model's scores are miscalibrated **by construction**, for two independent reasons:

1. **Class resampling** shifts the prior. Training saw a positive-pixel prior
   :math:`\pi_{\text{train}}` (the balanced sampler's), deployment sees
   :math:`\pi_{\text{deploy}}`. The correction is exact for a pure prior shift and is a
   single constant on the logit:

   .. math:: z \leftarrow z + \operatorname{logit}\pi_{\text{deploy}} - \operatorname{logit}\pi_{\text{train}}.

2. **Focal loss** reweights examples by their own confidence — a score-dependent
   distortion no constant removes. A temperature fitted on validation absorbs it:

   .. math:: p_{\text{cal}} = \sigma\bigl((z + c)/T^\star\bigr), \qquad
             T^\star = \arg\min_T \mathrm{NLL}_{\text{val}}.

If the one-parameter temperature on top of the analytic prior shift is insufficient — a
focal-trained model is typically *under*-confident (the loss stops pushing well-classified
pixels toward 0 or 1), so the fixed shift and one scale cannot both be right — the next step
of ``docs/09`` section 4.3 is **Platt scaling**, :math:`p = \sigma(a z + b)`, which contains
the temperature family. It is selected only if it lowers the validation NLL by more than
:data:`PLATT_MIN_RELATIVE_NLL_GAIN`: prefer the simplest map that works.

All of these maps are strictly monotone, so **ranking is unchanged**: AP, ROC-AUC and the best
achievable IoU over all thresholds are identical before and after. What changes is which
threshold value corresponds to which operating point — whether :math:`\tau^\star =
1/(1+r) = 0.25` means what ``docs/00`` section 6.2 says it means. Calibration does not
make the model better at discriminating; it makes the threshold interpretable.

Torch-free: everything here works on NumPy arrays of logits gathered at evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from floodfusion.metrics import expected_calibration_error

__all__ = [
    "CalibrationResult",
    "LogitReservoir",
    "apply_calibration",
    "calibrate_logits",
    "fit_platt",
    "fit_temperature",
    "logit",
    "prior_correction_offset",
]

_T_BOUNDS = (0.05, 20.0)
#: Platt scaling replaces the temperature only if it lowers validation NLL by more than this.
PLATT_MIN_RELATIVE_NLL_GAIN = 0.01


def logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"prior must lie in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def prior_correction_offset(pi_train: float, pi_deploy: float) -> float:
    """The constant :math:`c` added to every logit for a pure class-prior shift."""
    return logit(pi_deploy) - logit(pi_train)


def _nll(z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary NLL of logits ``z`` (numerically stable softplus form)."""
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    offset: float = 0.0,
    bounds: tuple[float, float] = _T_BOUNDS,
) -> float:
    r"""Scalar :math:`T^\star` minimising validation NLL of :math:`\sigma((z + c)/T)`.

    One parameter, optimised over :math:`\log T` with a bounded Brent search — it cannot
    meaningfully overfit the validation set. :math:`T > 1` softens an over-confident model.
    """
    from scipy.optimize import minimize_scalar

    z = np.asarray(logits, dtype=np.float64).ravel() + float(offset)
    y = np.asarray(labels, dtype=np.float64).ravel()
    if z.shape != y.shape or z.size == 0:
        raise ValueError("need equally sized, non-empty logits and labels")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0/1")
    if y.min() == y.max():
        raise ValueError("temperature is unidentifiable from a single class")
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    result = minimize_scalar(lambda log_t: _nll(z / math.exp(log_t), y), bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-4})
    return float(math.exp(result.x))


def fit_platt(logits: np.ndarray, labels: np.ndarray, *, init: tuple[float, float] = (1.0, 0.0)) -> tuple[float, float]:
    r"""Platt scaling :math:`p = \sigma(a z + b)` with :math:`a > 0` (monotone), by NLL.

    Optimised over :math:`(\log a, b)` with L-BFGS-B and the analytic gradient.
    """
    from scipy.optimize import minimize

    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if z.shape != y.shape or z.size == 0 or y.min() == y.max():
        raise ValueError("need equally sized logits and labels containing both classes")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = math.exp(theta[0]), theta[1]
        u = a * z + b
        residual = 0.5 * (1.0 + np.tanh(0.5 * u)) - y  # dNLL/du
        return _nll(u, y), np.array([float(np.mean(residual * z) * a), float(np.mean(residual))])

    start = np.array([math.log(max(init[0], 1e-3)), init[1]])
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      bounds=[(math.log(1.0 / _T_BOUNDS[1]), math.log(1.0 / _T_BOUNDS[0])), (-30.0, 30.0)])
    return float(math.exp(result.x[0])), float(result.x[1])


def apply_calibration(logits: np.ndarray, temperature: float, offset: float = 0.0) -> np.ndarray:
    """:math:`\\sigma((z + c)/T)` as float64 probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = (np.asarray(logits, dtype=np.float64) + offset) / temperature
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free sigmoid


@dataclass(frozen=True)
class CalibrationResult:
    r"""The selected map :math:`p = \sigma((z + c)/T)` and the ECE after each step (quantile bins).

    ``method`` is ``"temperature"`` (analytic prior offset + fitted T) or ``"platt"``
    (:math:`a = 1/T`, :math:`b = c/T` both fitted); either way ``temperature`` and ``offset``
    are the effective parameters every consumer applies.
    """

    method: str
    temperature: float
    offset: float
    pi_train: float
    pi_deploy: float
    ece_raw: float
    ece_prior: float
    ece_calibrated: float
    ece_temperature: float
    ece_platt: float
    nll_raw: float
    nll_temperature: float
    nll_platt: float
    nll_calibrated: float
    n_pixels: int
    reliability_raw: dict[str, Any]
    reliability_calibrated: dict[str, Any]

    def summary(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in (
            "method", "temperature", "offset", "pi_train", "pi_deploy", "ece_raw", "ece_prior",
            "ece_temperature", "ece_platt", "ece_calibrated", "nll_raw", "nll_temperature", "nll_platt",
            "nll_calibrated", "n_pixels")}


def calibrate_logits(logits: np.ndarray, labels: np.ndarray, *, pi_train: float, pi_deploy: float,
                     n_bins: int = 15) -> CalibrationResult:
    """Prior correction + temperature (``docs/09`` 4.3); Platt only if clearly better on NLL."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    offset = prior_correction_offset(pi_train, pi_deploy)
    t = fit_temperature(z, y, offset=offset)
    a, b = fit_platt(z, y, init=(1.0 / t, offset / t))
    nll_t, nll_p = _nll((z + offset) / t, y), _nll(a * z + b, y)
    use_platt = nll_p < (1.0 - PLATT_MIN_RELATIVE_NLL_GAIN) * nll_t
    temperature, effective_offset = (1.0 / a, b / a) if use_platt else (t, offset)
    ece_raw, rel_raw = expected_calibration_error(apply_calibration(z, 1.0), y, n_bins=n_bins)
    ece_prior, _ = expected_calibration_error(apply_calibration(z, 1.0, offset), y, n_bins=n_bins)
    ece_t, rel_t = expected_calibration_error(apply_calibration(z, t, offset), y, n_bins=n_bins)
    ece_p, rel_p = expected_calibration_error(apply_calibration(z, 1.0 / a, b / a), y, n_bins=n_bins)
    return CalibrationResult(
        method="platt" if use_platt else "temperature", temperature=temperature, offset=effective_offset,
        pi_train=pi_train, pi_deploy=pi_deploy, ece_raw=ece_raw, ece_prior=ece_prior,
        ece_calibrated=ece_p if use_platt else ece_t, ece_temperature=ece_t, ece_platt=ece_p,
        nll_raw=_nll(z, y), nll_temperature=nll_t, nll_platt=nll_p,
        nll_calibrated=nll_p if use_platt else nll_t, n_pixels=int(z.size),
        reliability_raw={k: np.asarray(v).tolist() for k, v in rel_raw.items()},
        reliability_calibrated={k: np.asarray(v).tolist() for k, v in (rel_p if use_platt else rel_t).items()},
    )


class LogitReservoir:
    """Uniform reservoir sample of (logit, label) pairs over valid pixels.

    Holding every validation logit would be the probability-map accumulation that causes
    the evaluation OOM of ``docs/08`` section 7.3; a fixed-size uniform sample is enough to
    fit one scalar and draw a reliability diagram. Algorithm R, vectorised per batch:
    after :math:`n` items each is retained with probability ``capacity / n``.
    """

    def __init__(self, capacity: int = 2_000_000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._z = np.empty(self.capacity, dtype=np.float32)
        self._y = np.empty(self.capacity, dtype=np.uint8)
        self.seen = 0

    def add(self, logits: np.ndarray, labels: np.ndarray) -> None:
        z = np.asarray(logits, dtype=np.float32).ravel()
        y = np.asarray(labels).ravel().astype(np.uint8)
        if z.shape != y.shape:
            raise ValueError("logits and labels differ in size")
        filled = min(self.seen, self.capacity)
        take = min(self.capacity - filled, z.size)
        if take:
            self._z[filled:filled + take] = z[:take]
            self._y[filled:filled + take] = y[:take]
        rest_z, rest_y = z[take:], y[take:]
        if rest_z.size:
            # Item i (global index n_i, 0-based) replaces a random slot w.p. capacity/(n_i+1).
            n_i = self.seen + take + np.arange(rest_z.size)
            slots = (self._rng.random(rest_z.size) * (n_i + 1)).astype(np.int64)
            hit = np.flatnonzero(slots < self.capacity)
            # Sequential Algorithm R lets a later item overwrite an earlier one on the same
            # slot; NumPy leaves duplicate-index assignment order unspecified, so keep the
            # LAST hit per slot explicitly.
            rev_slots = slots[hit][::-1]
            _, first_in_reversed = np.unique(rev_slots, return_index=True)
            winners = hit[::-1][first_in_reversed]
            self._z[slots[winners]] = rest_z[winners]
            self._y[slots[winners]] = rest_y[winners]
        self.seen += z.size

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = min(self.seen, self.capacity)
        return self._z[:n].copy(), self._y[:n].copy()
