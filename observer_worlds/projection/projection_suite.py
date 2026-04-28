"""Registry of 4D->2D projection operators used by Follow-up Topic 1.

Wraps the existing implementations in
:mod:`observer_worlds.worlds.projection` and adds two new ones
(``random_linear`` and ``multi_channel``) plus a no-op
``learned_projection`` placeholder.

Design notes
------------

* Every projection is a callable
  ``state_4d -> projected_2d_or_multichannel`` whose first two axes
  match the substrate's ``(Nx, Ny)`` shape.
* Some projections do **not** have a natural threshold margin. The
  ``threshold_margin_supported`` field on :class:`ProjectionSpec` tells
  downstream analysis to mark threshold-related metrics as N/A rather
  than running the audit logic where it does not apply.
* For ``random_linear`` and ``multi_channel`` the projection depends
  on a seed (the random projection matrix). The seed is part of the
  ``params`` dict so a run can be reproduced.
* This module imports lazily; it does not perform expensive work at
  import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from observer_worlds.worlds.projection import (
    max_projection,
    mean_threshold_projection,
    parity_projection,
    sum_projection,
)


# ---------------------------------------------------------------------------
# Per-projection metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionSpec:
    """Metadata + callable for one projection method.

    Parameters
    ----------
    name
        Stable identifier used in artifact filenames and analyses.
    fn
        ``fn(state_4d, **params) -> np.ndarray``. Must return either a
        ``(Nx, Ny)`` array (single-channel) or a ``(Nx, Ny, C)`` array
        (multi-channel; only used by ``multi_channel_projection``).
    threshold_margin_supported
        Whether the projection has a natural threshold margin (i.e.
        whether ``near_threshold_fraction`` is meaningful for it).
    output_kind
        ``"binary"`` (``uint8`` 0/1), ``"count"`` (int), ``"continuous"``
        (float), or ``"multi_channel"``.
    default_params
        Default parameters bound to ``fn`` when the suite is
        instantiated; per-call overrides remain possible via
        :meth:`ProjectionSuite.project`.
    docstring
        One-line description used in summaries.
    """
    name: str
    fn: Callable[..., np.ndarray]
    threshold_margin_supported: bool
    output_kind: str
    default_params: Mapping[str, object] = field(default_factory=dict)
    docstring: str = ""


# ---------------------------------------------------------------------------
# Built-in projection implementations
# ---------------------------------------------------------------------------


def _sum_threshold_projection(state_4d: np.ndarray, theta: int = 1) -> np.ndarray:
    """``Y(x, y) = 1 if sum_{z, w} X(x, y, z, w) >= theta else 0``.

    Distinct from ``mean_threshold`` (which compares the *fraction*
    against ``theta``) and from ``sum_projection`` (which returns the
    raw count). The natural threshold margin is the integer count
    ``s - theta``, which is meaningful.
    """
    if state_4d.ndim != 4:
        raise ValueError(
            f"sum_threshold expects a 4D array, got {state_4d.ndim}D"
        )
    s = state_4d.astype(np.int32).sum(axis=(2, 3))
    return (s >= int(theta)).astype(np.uint8)


def random_linear_weights(nz: int, nw: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic ``(nz, nw)`` weight matrix for ``random_linear`` /
    ``random_linear_projection``. Exposed so callers (e.g. the
    invisible-perturbation generator) can reproduce the projection's
    weights without going through ``_random_linear_projection``."""
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((int(nz), int(nw))).astype(np.float32)


def _random_linear_projection(
    state_4d: np.ndarray, *, seed: int = 0,
) -> np.ndarray:
    """Project to ``(Nx, Ny)`` continuous values via a random linear
    combination of the ``(z, w)`` fibre.

    The hidden-axis weights are drawn once per ``seed`` from a
    standard normal and reused across all timesteps. Output is a
    ``float32`` array; binarisation, if any, is applied downstream.
    """
    if state_4d.ndim != 4:
        raise ValueError(
            f"random_linear expects a 4D array, got {state_4d.ndim}D"
        )
    nz, nw = state_4d.shape[2:]
    weights = random_linear_weights(nz, nw, seed=seed)
    # Einsum keeps it on the active backend (numpy here; cupy users
    # should call the suite with cupy arrays directly — the einsum
    # dispatches via the array's __array_function__).
    return np.einsum("xyzw,zw->xy", state_4d.astype(np.float32), weights)


def multi_channel_masks(
    nz: int, nw: int, *, n_channels: int = 4, seed: int = 0,
) -> np.ndarray:
    """Deterministic ``(n_channels, nz, nw)`` random binary masks for
    ``multi_channel_projection``. Exposed for the invisible-
    perturbation generator's signature-based pair-swap strategy."""
    rng = np.random.default_rng(int(seed))
    masks = []
    for c in range(int(n_channels)):
        m = rng.integers(0, 2, size=(int(nz), int(nw))).astype(np.float32)
        if m.sum() == 0:
            m[0, 0] = 1.0
        masks.append(m)
    return np.stack(masks, axis=0)


def _multi_channel_projection(
    state_4d: np.ndarray, *, n_channels: int = 4, seed: int = 0,
) -> np.ndarray:
    """Project to ``(Nx, Ny, n_channels)`` via ``n_channels`` independent
    random binary masks over ``(z, w)``.

    Channel ``c`` is the mean over the masked subset of fibres,
    thresholded at ``0.5``. Returns ``uint8`` of shape
    ``(Nx, Ny, n_channels)``.
    """
    if state_4d.ndim != 4:
        raise ValueError(
            f"multi_channel expects a 4D array, got {state_4d.ndim}D"
        )
    nz, nw = state_4d.shape[2:]
    masks = multi_channel_masks(nz, nw, n_channels=n_channels, seed=seed)
    out_channels = []
    for c in range(int(n_channels)):
        m = masks[c]
        weighted = (state_4d.astype(np.float32) * m).sum(axis=(2, 3))
        weighted /= m.sum()
        out_channels.append((weighted > 0.5).astype(np.uint8))
    return np.stack(out_channels, axis=-1)


def _learned_projection_placeholder(state_4d: np.ndarray, **params) -> np.ndarray:
    """Placeholder for a future learned projection. Not implemented."""
    raise NotImplementedError(
        "learned_projection is reserved for future work; "
        "use one of the analytic projections in the meantime."
    )


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------


class ProjectionSuite:
    """Registry of named :class:`ProjectionSpec` entries.

    Construct with :func:`default_suite` for the standard six
    projections; register additional methods with :meth:`register`.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ProjectionSpec] = {}

    def register(self, spec: ProjectionSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"projection {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ProjectionSpec:
        try:
            return self._specs[name]
        except KeyError as e:
            raise KeyError(
                f"unknown projection {name!r}; available: {self.names()}"
            ) from e

    def project(
        self,
        name: str,
        state_4d: np.ndarray,
        **overrides,
    ) -> np.ndarray:
        spec = self.get(name)
        params = {**spec.default_params, **overrides}
        return spec.fn(state_4d, **params)

    def supports_threshold_margin(self, name: str) -> bool:
        return self.get(name).threshold_margin_supported

    def output_kind(self, name: str) -> str:
        return self.get(name).output_kind


def default_suite() -> ProjectionSuite:
    """The six projections evaluated in Follow-up Topic 1.

    `learned_projection` is intentionally **not** in the default suite;
    it is a placeholder for later work and would raise
    :class:`NotImplementedError` if invoked.
    """
    s = ProjectionSuite()
    s.register(ProjectionSpec(
        name="mean_threshold",
        fn=mean_threshold_projection,
        threshold_margin_supported=True,
        output_kind="binary",
        default_params={"theta": 0.5},
        docstring="Y(x,y) = 1 if mean_{z,w} X(x,y,z,w) > theta.",
    ))
    s.register(ProjectionSpec(
        name="sum_threshold",
        fn=_sum_threshold_projection,
        threshold_margin_supported=True,
        output_kind="binary",
        default_params={"theta": 1},
        docstring="Y(x,y) = 1 if sum_{z,w} X(x,y,z,w) >= theta.",
    ))
    s.register(ProjectionSpec(
        name="max_projection",
        fn=max_projection,
        threshold_margin_supported=False,
        output_kind="binary",
        default_params={},
        docstring="Y(x,y) = max_{z,w} X(x,y,z,w).",
    ))
    s.register(ProjectionSpec(
        name="parity_projection",
        fn=parity_projection,
        threshold_margin_supported=False,
        output_kind="binary",
        default_params={},
        docstring="Y(x,y) = (sum_{z,w} X) mod 2.",
    ))
    s.register(ProjectionSpec(
        name="random_linear_projection",
        fn=_random_linear_projection,
        threshold_margin_supported=False,
        output_kind="continuous",
        default_params={"seed": 0},
        docstring="Y = einsum('xyzw,zw->xy', X, W); W ~ N(0,1) per seed.",
    ))
    s.register(ProjectionSpec(
        name="multi_channel_projection",
        fn=_multi_channel_projection,
        threshold_margin_supported=False,
        output_kind="multi_channel",
        default_params={"n_channels": 4, "seed": 0},
        docstring="Y[..., c] = thresholded mean over a random subset of (z, w).",
    ))
    return s
