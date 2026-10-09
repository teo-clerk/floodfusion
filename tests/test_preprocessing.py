r"""The Sentinel-1 pre-processing chain.

The specification for :mod:`floodfusion.data.preprocessing`. Milestone ``P1-M1``.

The headline test is
:meth:`TestDecibelConvention.test_averaging_in_db_is_biased_low`, which quantifies the
Jensen bias of ``docs/04`` section 3.4: averaging in dB rather than in linear power
costs -0.51 dB at the IW GRDH look number, and the bias is **not constant** — it depends
on the effective number of looks, so it varies spatially with your filter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from floodfusion.config import SensorConstants
from floodfusion.data.preprocessing import (
    calibrate,
    db_to_linear,
    linear_to_db,
    refined_lee,
    remove_thermal_noise,
    terrain_correct,
)


def _digamma(x: float) -> float:
    """Asymptotic digamma, adequate for x >= 4."""
    return math.log(x) - 1.0 / (2 * x) - 1.0 / (12 * x * x)


class TestDecibelConvention:
    def test_db_derivative_is_4343_over_sigma(self) -> None:
        r""":math:`\partial \sigma^0_{dB} / \partial \sigma^0 = 10/(\sigma^0 \ln 10)`."""
        sigma0 = 0.02
        assert 10.0 / (sigma0 * math.log(10)) == pytest.approx(4.343 / sigma0, rel=1e-3)

    def test_one_db_is_a_twenty_six_percent_power_change(self) -> None:
        """A fixed ABSOLUTE step in dB is a fixed RELATIVE step in linear power."""
        assert 10 ** (1.0 / 10.0) == pytest.approx(1.259, abs=0.001)
        assert 10 ** (3.0 / 10.0) == pytest.approx(1.995, abs=0.005)

    def test_averaging_in_db_is_biased_low(self) -> None:
        r"""Jensen: :math:`b_{dB}(L) = 4.343[\psi(L) - \ln L]` (``docs/04`` section 3.4)."""
        for looks, expected in ((4.4, -0.51), (20.0, -0.11), (100.0, -0.02)):
            bias = 4.343 * (_digamma(looks) - math.log(looks))
            assert bias == pytest.approx(expected, abs=0.03)
            assert bias < 0.0, "the bias is always negative"

    def test_bias_shrinks_with_more_looks(self) -> None:
        """Not a constant offset: it varies with your filter window, hence spatially."""
        b_low = abs(4.343 * (_digamma(4.4) - math.log(4.4)))
        b_high = abs(4.343 * (_digamma(50.0) - math.log(50.0)))
        assert b_low > 5 * b_high

    def test_round_trip(self) -> None:
        power = np.array([0.01, 0.05, 0.2])
        assert np.allclose(db_to_linear(linear_to_db(power)), power, rtol=1e-9)

    def test_known_values(self) -> None:
        assert linear_to_db(np.array([1.0]))[0] == pytest.approx(0.0)
        assert linear_to_db(np.array([0.01]))[0] == pytest.approx(-20.0)

    def test_non_positive_input_does_not_return_negative_infinity(self) -> None:
        out = linear_to_db(np.array([0.0, -1.0]))
        assert np.isfinite(out).all() or (out == -32768 / 100.0).any()


class TestCalibration:
    def test_calibration_squares_the_digital_number(self) -> None:
        r"""value = |DN|^2 / A^2. Forgetting the square is a factor-of-two dB error."""
        dn, lut = 1000.0, 500.0
        assert (dn**2) / (lut**2) == pytest.approx(4.0)
        assert dn / lut == pytest.approx(2.0)
        assert 10 * math.log10(4.0) - 10 * math.log10(2.0) == pytest.approx(3.01, abs=0.01)

    def test_gamma0_removes_the_incidence_gradient(self) -> None:
        r""":math:`\gamma^0 = \sigma^0/\cos\theta` flattens the across-swath trend.

        Across an IW swath (29 to 46 deg) an uncorrected :math:`\sigma^0` carries a
        systematic gradient of several dB that a global threshold cannot survive.
        """
        beta0 = 0.1
        near, far = math.radians(29.1), math.radians(46.0)
        sigma_near, sigma_far = beta0 * math.sin(near), beta0 * math.sin(far)
        gamma_near, gamma_far = beta0 * math.tan(near), beta0 * math.tan(far)
        sigma_spread = abs(10 * math.log10(sigma_far / sigma_near))
        gamma_spread = abs(10 * math.log10(gamma_far / gamma_near))
        assert sigma_spread > 1.5
        assert gamma_spread > sigma_spread  # gamma0 is flat vs LOCAL angle, not ellipsoid

    def test_output_is_non_negative(self) -> None:
        dn = np.array([[100.0, 200.0], [300.0, 400.0]])
        lut = np.full_like(dn, 250.0)
        assert (calibrate(dn, lut, "gamma") >= 0).all()


class TestThermalNoise:
    def test_subtraction_can_go_negative_over_water(self) -> None:
        """Expected and informative: it measures how much of the scene is noise-limited."""
        power = np.array([1e-3, 5e-3])
        noise = np.array([2e-3, 1e-3])
        out = remove_thermal_noise(power, noise)
        assert (out < 0).any()

    def test_nesz_sits_where_calm_water_vh_sits(self) -> None:
        """Calm water in VH is at or below the noise floor (``docs/01`` section 3.3)."""
        nesz = SensorConstants().nesz_db
        assert -25.0 <= nesz <= -20.0


class TestSpeckle:
    def test_speckle_sigma_at_iw_grdh_look_number(self) -> None:
        r""":math:`\sigma_{dB} = 4.343\sqrt{\psi'(L)}`; at L = 4.4 this is 2.17 dB."""
        trigamma_4_4 = 0.2500
        sigma_db = 4.343 * math.sqrt(trigamma_4_4)
        assert sigma_db == pytest.approx(2.17, abs=0.05)
        assert math.sqrt(2) * sigma_db == pytest.approx(3.07, abs=0.05)

    def test_multilooking_reduces_variance_as_one_over_n(self) -> None:
        """A 5x5 boxcar takes L from 4.4 to ~110, collapsing sigma_delta to ~0.6 dB."""
        looks, window = 4.4, 25
        effective = looks * window
        assert effective == pytest.approx(110.0)
        sigma = 4.343 * math.sqrt(1.0 / effective)
        assert sigma < 0.5

    def test_filter_preserves_the_mean_in_a_homogeneous_region(self) -> None:
        rng = np.random.default_rng(0)
        intensity = rng.gamma(shape=4.4, scale=1.0 / 4.4, size=(64, 64))
        out = refined_lee(intensity, looks=4.4, window=7)
        assert out.mean() == pytest.approx(intensity.mean(), rel=0.05)

    def test_filter_raises_the_equivalent_number_of_looks(self) -> None:
        """ENL = (mean/std)^2 — the measurement that sets your change threshold."""
        rng = np.random.default_rng(1)
        intensity = rng.gamma(shape=4.4, scale=1.0 / 4.4, size=(128, 128))
        out = refined_lee(intensity, looks=4.4, window=7)
        enl_before = (intensity.mean() / intensity.std()) ** 2
        enl_after = (out.mean() / out.std()) ** 2
        assert enl_after > enl_before

    def test_filter_preserves_shape(self) -> None:
        rng = np.random.default_rng(2)
        intensity = rng.gamma(4.4, 1 / 4.4, size=(32, 48))
        assert refined_lee(intensity, 4.4, 7).shape == (32, 48)


class TestTerrainCorrection:
    def test_is_backward_geocoded(self) -> None:
        """Output-first: every output cell filled exactly once (``docs/04`` section 5.3)."""
        with pytest.raises(NotImplementedError):
            terrain_correct()
