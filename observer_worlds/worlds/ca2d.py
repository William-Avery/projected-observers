"""2D binary cellular automaton with Moore-r1 neighbourhood and periodic BCs.

This is the small 2D baseline used by sanity checks (e.g. Conway's Life) and
by the comparison experiments that pit a flat 2D CA against a 4D-projected
2D image.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from observer_worlds.worlds.rules import BSRule


# Moore-r1 in 2D has 3**2 - 1 = 8 neighbours.
_MAX_NEIGHBOURS_2D = 8


def _neighbour_kernel_2d() -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    return kernel


def update_2d_numpy(state: np.ndarray, rule: BSRule) -> np.ndarray:
    """Reference 2D CA update using ``scipy.ndimage.convolve`` (periodic)."""
    if state.ndim != 2:
        raise ValueError(f"update_2d_numpy expects a 2D array, got {state.ndim}D")

    kernel = _neighbour_kernel_2d()
    counts = convolve(state.astype(np.int32), kernel, mode="wrap")

    birth_lut, survival_lut = rule.to_lookup_tables(_MAX_NEIGHBOURS_2D)
    alive = state.astype(bool)
    new_alive = np.where(alive, survival_lut[counts], birth_lut[counts])
    return new_alive.astype(np.uint8)


class CA2D:
    """2D Moore-r1 binary CA with periodic boundaries (used by the 2D baseline)."""

    def __init__(self, shape: tuple[int, int], rule: BSRule) -> None:
        if len(shape) != 2:
            raise ValueError(f"CA2D requires a 2D shape, got {shape!r}")
        self.shape: tuple[int, int] = tuple(int(s) for s in shape)  # type: ignore[assignment]
        self.rule = rule
        self._state: np.ndarray = np.zeros(self.shape, dtype=np.uint8)

    def initialize_random(
        self, density: float, rng: np.random.Generator
    ) -> None:
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"density must be in [0,1], got {density}")
        self._state = (rng.random(self.shape) < density).astype(np.uint8)

    def step(self) -> None:
        self._state = update_2d_numpy(self._state, self.rule)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        if value.shape != self.shape:
            raise ValueError(
                f"state shape mismatch: expected {self.shape}, got {value.shape}"
            )
        self._state = np.ascontiguousarray(value, dtype=np.uint8)
