"""4D binary cellular automaton with Moore-r1 neighbourhood and periodic BCs.

Provides:

* :func:`update_4d_numpy` - reference implementation using ``scipy.ndimage.convolve``.
* :func:`update_4d_numba` - Numba-accelerated implementation (falls back to numpy
  if numba is unavailable, but raises if explicitly requested).
* :class:`CA4D` - a thin stateful wrapper that picks a backend and steps in place.

The Moore-r1 neighbourhood in 4D has ``3**4 - 1 = 80`` cells, so neighbour counts
fit in a ``uint8``.  States are stored as ``uint8`` arrays of zeros and ones.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from observer_worlds.worlds.rules import BSRule


# ---------------------------------------------------------------------------
# Optional numba import.  We must not blow up at import time if numba is
# missing, since the numpy backend is fully usable on its own.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import-time guard
    import numba
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - import-time guard
    HAS_NUMBA = False


# Bootstrap CUDA_PATH from the nvidia-cuda-nvrtc-cu12 wheel layout *before*
# importing cupy. cupy caches the resolved CUDA path on first import (in
# `_setup_win32_dll_directory`), so we have to set the env var ahead of
# `import cupy`. See ``_cuda_bootstrap.bootstrap_cuda_path`` for details.
from observer_worlds.worlds._cuda_bootstrap import bootstrap_cuda_path as _bootstrap_cuda_path

_bootstrap_cuda_path()

try:  # pragma: no cover - import-time guard
    import cupy as cp
except ImportError:  # pragma: no cover - import-time guard
    cp = None  # type: ignore[assignment]


# 4D Moore-r1 neighbourhood has 3**4 - 1 = 80 neighbours.
_MAX_NEIGHBOURS_4D = 80


# ---------------------------------------------------------------------------
# Numpy reference implementation
# ---------------------------------------------------------------------------


def _neighbour_kernel_4d() -> np.ndarray:
    """Return a (3,3,3,3) int32 kernel that sums over neighbours (centre = 0)."""
    kernel = np.ones((3, 3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1, 1] = 0
    return kernel


def update_4d_numpy(state: np.ndarray, rule: BSRule) -> np.ndarray:
    """Reference 4D CA update using ``scipy.ndimage.convolve`` (periodic).

    Parameters
    ----------
    state:
        4D ``uint8`` array of zeros and ones.
    rule:
        Birth/survival rule.

    Returns
    -------
    np.ndarray
        New 4D ``uint8`` state of the same shape.
    """
    if state.ndim != 4:
        raise ValueError(f"update_4d_numpy expects a 4D array, got {state.ndim}D")

    kernel = _neighbour_kernel_4d()
    counts = convolve(state.astype(np.int32), kernel, mode="wrap")

    birth_lut, survival_lut = rule.to_lookup_tables(_MAX_NEIGHBOURS_4D)
    # Index lookup tables by count.  Counts are in [0, 80], guaranteed in range.
    alive = state.astype(bool)
    new_alive = np.where(alive, survival_lut[counts], birth_lut[counts])
    return new_alive.astype(np.uint8)


# ---------------------------------------------------------------------------
# Numba implementation
# ---------------------------------------------------------------------------


# We must define the jitted core function only when numba is importable;
# otherwise we leave a sentinel and raise at call time / construction time.
if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _update_4d_numba_core(
        state: np.ndarray,
        birth_lut: np.ndarray,
        survival_lut: np.ndarray,
    ) -> np.ndarray:
        """Numba kernel: 4 nested loops, periodic BCs by modulo arithmetic.

        ``birth_lut`` and ``survival_lut`` are bool arrays of shape (81,).
        """
        nx, ny, nz, nw = state.shape
        out = np.empty_like(state)
        for x in prange(nx):
            for y in range(ny):
                for z in range(nz):
                    for w in range(nw):
                        count = 0
                        # Sum over the 3x3x3x3 = 81 cube; skip the centre.
                        for dx in range(-1, 2):
                            xi = (x + dx) % nx
                            for dy in range(-1, 2):
                                yi = (y + dy) % ny
                                for dz in range(-1, 2):
                                    zi = (z + dz) % nz
                                    for dw in range(-1, 2):
                                        if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                                            continue
                                        wi = (w + dw) % nw
                                        count += state[xi, yi, zi, wi]

                        if state[x, y, z, w]:
                            out[x, y, z, w] = 1 if survival_lut[count] else 0
                        else:
                            out[x, y, z, w] = 1 if birth_lut[count] else 0
        return out


def update_4d_numba(state: np.ndarray, rule: BSRule) -> np.ndarray:
    """Numba-accelerated 4D CA update.

    Pure-Python wrapper that builds the lookup tables (numba can't consume
    a :class:`BSRule` dataclass directly) and then dispatches into a jitted
    core with parallel ``prange`` on the outer-most axis.

    Falls back to nothing: if numba isn't installed this raises ``RuntimeError``.
    """
    if state.ndim != 4:
        raise ValueError(f"update_4d_numba expects a 4D array, got {state.ndim}D")
    if not HAS_NUMBA:
        raise RuntimeError(
            "numba is not installed; install it or use update_4d_numpy / "
            "CA4D(..., backend='numpy')."
        )

    birth_lut, survival_lut = rule.to_lookup_tables(_MAX_NEIGHBOURS_4D)
    # The jitted kernel needs concrete ndarray dtypes; pass uint8 state and
    # bool LUTs (numba handles bool natively).
    state_u8 = np.ascontiguousarray(state, dtype=np.uint8)
    return _update_4d_numba_core(state_u8, birth_lut, survival_lut)


# ---------------------------------------------------------------------------
# Stateful wrapper
# ---------------------------------------------------------------------------


class CA4D:
    """4D Moore-r1 binary cellular automaton with periodic boundaries.

    Neighbourhood has ``3**4 - 1 = 80`` cells.

    Parameters
    ----------
    shape:
        4-tuple ``(Nx, Ny, Nz, Nw)``.
    rule:
        Birth/survival rule.
    backend:
        ``"numba"`` (default), ``"numpy"``, or ``"cuda"``.  ``"numba"`` requires
        the optional numba dependency; ``"cuda"`` requires cupy. If a backend
        is requested whose dependency is missing, a clear ``RuntimeError`` is
        raised at construction time.
    """

    def __init__(
        self,
        shape: tuple[int, int, int, int],
        rule: BSRule,
        backend: str = "numba",
    ) -> None:
        if len(shape) != 4:
            raise ValueError(f"CA4D requires a 4D shape, got {shape!r}")
        if backend not in {"numba", "numpy", "cuda"}:
            raise ValueError(
                f"backend must be 'numba', 'numpy', or 'cuda', got {backend!r}"
            )
        if backend == "numba" and not HAS_NUMBA:
            raise RuntimeError(
                "CA4D was constructed with backend='numba' but numba is not "
                "installed.  Install numba or pass backend='numpy'."
            )
        if backend == "cuda":
            from observer_worlds.worlds.ca4d_cuda import HAS_CUPY
            if not HAS_CUPY:
                raise RuntimeError(
                    "CA4D was constructed with backend='cuda' but cupy is "
                    "not installed.  Run `pip install cupy-cuda12x` or use "
                    "backend='numba'."
                )

        self.shape: tuple[int, int, int, int] = tuple(int(s) for s in shape)  # type: ignore[assignment]
        self.rule = rule
        self.backend = backend
        self._state = np.zeros(self.shape, dtype=np.uint8)
        if backend == "numba":
            self._update = update_4d_numba
        elif backend == "cuda":
            from observer_worlds.worlds.ca4d_cuda import update_4d_cuda
            self._update = update_4d_cuda
        else:
            self._update = update_4d_numpy

    # ---- state management -------------------------------------------------

    def initialize_random(
        self, density: float, rng: np.random.Generator
    ) -> None:
        """Set ``self.state`` to a Bernoulli(density) sample."""
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"density must be in [0,1], got {density}")
        host = (rng.random(self.shape) < density).astype(np.uint8)
        if self.backend == "cuda":
            self._state = cp.asarray(host)
        else:
            self._state = host

    def step(self) -> None:
        """Advance ``self.state`` by one timestep (in place via reassignment)."""
        self._state = self._update(self._state, self.rule)

    @property
    def state(self) -> np.ndarray:
        """4D ``uint8`` array of shape ``self.shape``.

        Returns a host (numpy) view. For the cuda backend, this triggers
        a device->host copy. Use ``state_device`` to keep the device array.
        """
        if self.backend == "cuda":
            return cp.asnumpy(self._state)
        return self._state

    @property
    def state_device(self):
        """Raw underlying state. cupy ndarray for cuda; numpy ndarray
        otherwise. Use this in tight loops to avoid host<->device copies."""
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        if value.shape != self.shape:
            raise ValueError(
                f"state shape mismatch: expected {self.shape}, got {value.shape}"
            )
        if self.backend == "cuda":
            self._state = cp.ascontiguousarray(cp.asarray(value, dtype=cp.uint8))
        else:
            self._state = np.ascontiguousarray(value, dtype=np.uint8)
