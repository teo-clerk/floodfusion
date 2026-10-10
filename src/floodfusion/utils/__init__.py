"""Utilities: shape contracts and VRAM instrumentation.

``shapes``: declarative tensor shape contracts with named dimensions, used everywhere.
``vram``: the analytic peak-memory model and its measured check (``P1-M4.1``).
``latency``: inference timing.  ``env``: the provenance a run records.
"""

from __future__ import annotations

from floodfusion.utils.shapes import ShapeError, assert_shape, check_finite, named_dims

__all__ = ["ShapeError", "assert_shape", "check_finite", "named_dims"]
