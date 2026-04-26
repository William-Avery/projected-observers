"""4D-to-2D projection operators.

The bulk simulator runs a 4D CA ``X[x, y, z, w]``; downstream observers see
only a 2D projection ``Y[x, y]``.  Different choices of projection let us
probe how much hidden-axis structure leaks through.

All projections take a 4D array ``state_4d`` and reduce over the last two
axes (``z``, ``w``).  Convention: the first two axes are the "visible" plane.
"""

from __future__ import annotations

import numpy as np


def _check_4d(state_4d: np.ndarray) -> None:
    if state_4d.ndim != 4:
        raise ValueError(
            f"projection expects a 4D array, got {state_4d.ndim}D "
            f"(shape={state_4d.shape!r})"
        )


def mean_threshold_projection(
    state_4d: np.ndarray, theta: float = 0.5
) -> np.ndarray:
    """``Y(x, y) = 1 if mean_{z,w} X(x, y, z, w) > theta else 0``.

    Note the *strict* greater-than: a fibre that is exactly half-on with
    ``theta=0.5`` projects to 0.
    """
    _check_4d(state_4d)
    mean = state_4d.astype(np.float64).mean(axis=(2, 3))
    return (mean > float(theta)).astype(np.uint8)


def sum_projection(state_4d: np.ndarray) -> np.ndarray:
    """``Y(x, y) = sum_{z,w} X(x, y, z, w)`` as ``int32``."""
    _check_4d(state_4d)
    return state_4d.astype(np.int32).sum(axis=(2, 3)).astype(np.int32)


def parity_projection(state_4d: np.ndarray) -> np.ndarray:
    """``Y(x, y) = (sum_{z,w} X(x, y, z, w)) mod 2`` as ``uint8``."""
    _check_4d(state_4d)
    return (state_4d.astype(np.int32).sum(axis=(2, 3)) & 1).astype(np.uint8)


def max_projection(state_4d: np.ndarray) -> np.ndarray:
    """``Y(x, y) = max_{z,w} X(x, y, z, w)`` as ``uint8``."""
    _check_4d(state_4d)
    return state_4d.max(axis=(2, 3)).astype(np.uint8)


_DISPATCH = {
    "mean_threshold": lambda s, theta: mean_threshold_projection(s, theta),
    "sum": lambda s, theta: sum_projection(s),
    "parity": lambda s, theta: parity_projection(s),
    "max": lambda s, theta: max_projection(s),
}


def project(
    state_4d: np.ndarray,
    method: str = "mean_threshold",
    theta: float = 0.5,
) -> np.ndarray:
    """Dispatch to the named projection.

    ``method`` is one of ``{"mean_threshold", "sum", "parity", "max"}``.
    ``theta`` is only consulted by ``"mean_threshold"``.

    Returns a 2D array of shape ``(Nx, Ny)``.  ``"sum"`` returns ``int32``;
    the others return ``uint8``.
    """
    try:
        fn = _DISPATCH[method]
    except KeyError as e:
        raise ValueError(
            f"unknown projection method {method!r}; "
            f"valid: {sorted(_DISPATCH)}"
        ) from e
    return fn(state_4d, theta)
