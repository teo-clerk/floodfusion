"""Utilities: shape contracts and VRAM instrumentation.

``shapes`` is **given to you complete and working**. It teaches an idiom — declarative
tensor shape contracts with named dimensions — and typing it out yourself would waste
your time. Use it everywhere.

``vram`` is a skeleton: the analytic memory model is milestone ``P1-M4.1`` and is a
learning target.
"""

from __future__ import annotations

from floodfusion.utils.shapes import ShapeError, assert_shape, check_finite, named_dims

__all__ = ["ShapeError", "assert_shape", "check_finite", "named_dims"]
