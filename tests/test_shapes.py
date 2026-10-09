"""The shape-contract helper. Given complete, so these tests pass on a fresh clone.

The characteristic bug in a multimodal segmentation network is not a crash; it is a
silent broadcast. A ``(B, 1, H, W)`` mask multiplied against ``(B, C, H, W)`` works, and
a ``(B, H, W, 1)`` mask *also* works, broadcasts differently, and trains a model that is
wrong with no exception raised. These tests exist to guarantee the tool that catches it
is itself correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from floodfusion.utils.shapes import ShapeError, assert_shape, check_finite, named_dims


class _Fake:
    """Minimal ``.shape`` carrier, so these tests need no array library."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class TestExactDimensions:
    def test_matching_shape_passes(self) -> None:
        assert_shape(_Fake((2, 5, 32, 32)), (2, 5, 32, 32), name="x")

    def test_mismatched_axis_raises_naming_the_axis(self) -> None:
        with pytest.raises(ShapeError, match="axis 1"):
            assert_shape(_Fake((2, 4, 32, 32)), (2, 5, 32, 32), name="x")

    def test_wrong_rank_raises(self) -> None:
        with pytest.raises(ShapeError, match="axes"):
            assert_shape(_Fake((2, 5, 32)), (2, 5, 32, 32), name="x")

    def test_error_message_contains_tensor_name(self) -> None:
        with pytest.raises(ShapeError, match="optical"):
            assert_shape(_Fake((2, 4)), (2, 5), name="optical")


class TestNamedDimensions:
    def test_first_use_binds(self) -> None:
        dims = named_dims()
        assert_shape(_Fake((2, 5, 32, 32)), ("B", 5, "H", "W"), dims, name="sar")
        assert dims == {"B": 2, "H": 32, "W": 32}

    def test_consistent_reuse_passes(self) -> None:
        dims = named_dims()
        assert_shape(_Fake((2, 5, 32, 32)), ("B", 5, "H", "W"), dims, name="sar")
        assert_shape(_Fake((2, 4, 32, 32)), ("B", 4, "H", "W"), dims, name="optical")
        assert_shape(_Fake((2, 1, 32, 32)), ("B", 1, "H", "W"), dims, name="label")

    def test_inconsistent_reuse_raises(self) -> None:
        """The whole point: a batch-size disagreement between modalities is caught."""
        dims = named_dims()
        assert_shape(_Fake((2, 5, 32, 32)), ("B", 5, "H", "W"), dims, name="sar")
        with pytest.raises(ShapeError, match="previously bound to 2"):
            assert_shape(_Fake((8, 4, 32, 32)), ("B", 4, "H", "W"), dims, name="optical")

    def test_transposed_spatial_axes_are_caught(self) -> None:
        """H and W bound separately, so a transpose on a non-square chip is detected."""
        dims = named_dims()
        assert_shape(_Fake((2, 5, 64, 32)), ("B", 5, "H", "W"), dims, name="sar")
        with pytest.raises(ShapeError):
            assert_shape(_Fake((2, 1, 32, 64)), ("B", 1, "H", "W"), dims, name="label")

    def test_independent_environments_do_not_interfere(self) -> None:
        a, b = named_dims(), named_dims()
        assert_shape(_Fake((2, 3)), ("B", 3), a, name="a")
        assert_shape(_Fake((8, 3)), ("B", 3), b, name="b")
        assert a["B"] == 2 and b["B"] == 8


class TestWildcards:
    def test_none_matches_any_size(self) -> None:
        assert_shape(_Fake((2, 99, 32, 32)), (2, None, 32, 32), name="x")

    def test_none_binds_nothing(self) -> None:
        dims = named_dims()
        assert_shape(_Fake((2, 99)), ("B", None), dims, name="x")
        assert dims == {"B": 2}

    def test_ellipsis_matches_leading_axes(self) -> None:
        assert_shape(_Fake((2, 3, 4, 5, 32)), ("...", 32), name="x")

    def test_ellipsis_still_checks_trailing_axes(self) -> None:
        with pytest.raises(ShapeError):
            assert_shape(_Fake((2, 3, 4, 5, 31)), ("...", 32), name="x")

    def test_ellipsis_with_too_few_axes_raises(self) -> None:
        with pytest.raises(ShapeError, match="at least"):
            assert_shape(_Fake((32,)), ("...", 16, 32), name="x")


class TestCheckFinite:
    def test_finite_tensor_passes(self) -> None:
        torch = pytest.importorskip("torch")
        check_finite(torch.zeros(4, 4), name="x")

    def test_nan_raises_and_reports_the_count(self) -> None:
        torch = pytest.importorskip("torch")
        x = torch.zeros(4, 4)
        x[0, 0] = float("nan")
        with pytest.raises(ShapeError, match="1 NaN"):
            check_finite(x, name="x")

    def test_inf_raises_and_is_distinguished_from_nan(self) -> None:
        """NaN and Inf have different causes; the message must separate them."""
        torch = pytest.importorskip("torch")
        x = torch.zeros(4, 4)
        x[0, 0] = float("inf")
        with pytest.raises(ShapeError, match="0 NaN and 1 Inf"):
            check_finite(x, name="x")

    def test_non_tensor_raises_type_error(self) -> None:
        pytest.importorskip("torch")
        with pytest.raises(TypeError):
            check_finite(np.zeros((2, 2)), name="x")


def test_works_with_numpy_arrays() -> None:
    """The helper is array-library agnostic — anything with ``.shape``."""
    dims = named_dims()
    assert_shape(np.zeros((2, 5, 32, 32)), ("B", 5, "H", "W"), dims, name="sar")
    assert dims["B"] == 2
