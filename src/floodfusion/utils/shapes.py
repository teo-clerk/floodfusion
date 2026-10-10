"""Declarative tensor shape contracts.

**This module is given to you complete and working.** It teaches an idiom rather than a
concept, so typing it out yourself would waste your time. Everything else in
``src/floodfusion/`` is yours to implement.

Motivation
----------
The characteristic bug in a multimodal segmentation network is not a crash. It is a
silent broadcast: a ``(B, 1, H, W)`` mask multiplied against a ``(B, C, H, W)`` tensor
works; a ``(B, H, W, 1)`` mask *also* works, broadcasts differently, and trains a model
that is wrong in a way no exception reveals. The fix is to state the contract at every
boundary and let it fail loudly.

Usage
-----
Named dimensions are bound on first sight and checked thereafter::

    dims = named_dims()
    assert_shape(sar,     ("B", 5, "H", "W"),  dims, name="sar")
    assert_shape(optical, ("B", 4, "H", "W"),  dims, name="optical")
    assert_shape(logits,  ("B", 1, "H", "W"),  dims, name="logits")

The first call binds ``B``, ``H`` and ``W``; the second and third check them. A mismatch
raises :class:`ShapeError` naming the tensor, the expected contract, the actual shape and
the specific axis that disagreed.

Integer entries are exact. ``None`` matches any size without binding. ``"..."`` as the
sole leading entry allows any number of leading axes.

Cost
----
Negligible: a few Python comparisons per call, no device synchronisation, no tensor
reads. :func:`check_finite` **does** synchronise, so it is opt-in and should be reserved
for debugging (see its docstring).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = ["ShapeError", "assert_shape", "check_finite", "named_dims"]


class ShapeError(AssertionError):
    """Raised when a tensor violates its declared shape contract."""


@runtime_checkable
class HasShape(Protocol):
    """Anything with a ``.shape`` — ``torch.Tensor``, ``numpy.ndarray``, and friends."""

    @property
    def shape(self) -> Any: ...


def named_dims() -> dict[str, int]:
    """Create a fresh binding environment for named dimensions.

    Returns
    -------
    dict[str, int]
        An empty dictionary, to be threaded through a sequence of
        :func:`assert_shape` calls that must agree with one another.
    """
    return {}


def assert_shape(
    tensor: HasShape,
    contract: Sequence[int | str | None],
    dims: dict[str, int] | None = None,
    *,
    name: str = "tensor",
) -> None:
    """Assert that ``tensor`` matches ``contract``, binding and checking named axes.

    Parameters
    ----------
    tensor
        Any object exposing ``.shape``.
    contract
        One entry per axis. Entries may be:

        - ``int``   — the axis must have exactly this size;
        - ``str``   — a named dimension, bound on first use and checked thereafter;
        - ``None``  — matches any size, binds nothing;
        - ``"..."`` — permitted only as the first entry; matches any number of
          leading axes.
    dims
        Binding environment from :func:`named_dims`. If ``None``, a throwaway
        environment is used and named dimensions are checked only within this call.
    name
        Name used in error messages. Always pass it; the default is useless in a
        traceback with six tensors.

    Raises
    ------
    ShapeError
        If the rank or any axis size disagrees with the contract, or if a named
        dimension conflicts with an earlier binding.

    Examples
    --------
    >>> import numpy as np
    >>> d = named_dims()
    >>> assert_shape(np.zeros((4, 5, 256, 256)), ("B", 5, "H", "W"), d, name="sar")
    >>> d["B"], d["H"]
    (4, 256)
    >>> assert_shape(np.zeros((8, 1, 256, 256)), ("B", 1, "H", "W"), d, name="logits")
    Traceback (most recent call last):
        ...
    floodfusion.utils.shapes.ShapeError: logits: axis 0 ...
    """
    if dims is None:
        dims = {}

    shape = tuple(int(s) for s in tensor.shape)
    contract = list(contract)

    if contract and contract[0] == "...":
        trailing = contract[1:]
        if len(shape) < len(trailing):
            raise ShapeError(
                f"{name}: expected at least {len(trailing)} axes for contract "
                f"{tuple(contract)}, got shape {shape}"
            )
        shape = shape[len(shape) - len(trailing) :]
        contract = trailing
        offset = len(tensor.shape) - len(trailing)
    else:
        offset = 0

    if len(shape) != len(contract):
        raise ShapeError(
            f"{name}: expected {len(contract)} axes for contract {tuple(contract)}, "
            f"got {len(shape)} axes with shape {tuple(tensor.shape)}"
        )

    for axis, (actual, expected) in enumerate(zip(shape, contract)):
        if expected is None:
            continue
        if isinstance(expected, int):
            if actual != expected:
                raise ShapeError(
                    f"{name}: axis {axis + offset} expected size {expected}, got "
                    f"{actual}. Full shape {tuple(tensor.shape)} against contract "
                    f"{tuple(contract)}."
                )
            continue
        # Named dimension.
        bound = dims.get(expected)
        if bound is None:
            dims[expected] = actual
        elif bound != actual:
            raise ShapeError(
                f"{name}: axis {axis + offset} is named {expected!r}, previously bound "
                f"to {bound}, but this tensor has {actual}. Full shape "
                f"{tuple(tensor.shape)} against contract {tuple(contract)}."
            )


def check_finite(tensor: Any, *, name: str = "tensor") -> None:
    """Raise if ``tensor`` contains ``NaN`` or ``Inf``.

    Parameters
    ----------
    tensor
        A ``torch.Tensor``.
    name
        Name used in the error message.

    Raises
    ------
    ShapeError
        If any element is non-finite. The message reports the counts of ``NaN`` and
        ``Inf`` separately, because they usually have different causes: ``NaN``
        typically comes from ``0/0`` or ``log(0)`` (an unscaled attention logit, a
        Dice denominator of zero), whereas ``Inf`` comes from overflow (an fp16
        reduction — see ``docs/08`` section 5.1).

    Warning
    -------
    This **synchronises the device**. Calling it every step will dominate your training
    loop. Use it while debugging a ``NaN`` (challenge C5 in ``docs/10`` section 3),
    then remove it.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: check_finite expects a torch.Tensor, got {type(tensor)}")

    n_nan = int(torch.isnan(tensor).sum().item())
    n_inf = int(torch.isinf(tensor).sum().item())
    if n_nan or n_inf:
        raise ShapeError(
            f"{name}: {n_nan} NaN and {n_inf} Inf among {tensor.numel()} elements "
            f"(shape {tuple(tensor.shape)}, dtype {tensor.dtype}). "
            "NaN usually means 0/0 or log(0); Inf usually means overflow. "
            "See docs/10 section 3, challenge C5."
        )
