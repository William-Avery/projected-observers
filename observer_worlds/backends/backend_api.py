"""Backend API for batched 4D-CA work (numpy reference, cupy GPU).

This module defines a thin abstract interface used by the Stage G1
benchmark harness and equivalence tests. It does **not** replace the
existing per-runner ``--backend numpy|numba|cuda`` flag wired through
:class:`observer_worlds.worlds.ca4d.CA4D`. The legacy flag stays for
all production runners; this layer is additive and is only consumed by
new GPU code paths (G1 benchmark harness; later: GPU-aware HCE
runners, once equivalence is proven).

Design rules (Stage G1):

* Keep state arrays on-backend across the timestep loop. Only compact
  metric arrays cross the device boundary.
* CPU and GPU primitives must be bit-identical for binary projections
  on tiny deterministic inputs; for continuous projections they must
  match within a documented tolerance.
* CuPy is optional. If unavailable, ``get_backend("cupy")`` raises
  ``RuntimeError`` (not ``ImportError``); ``is_cupy_available()``
  returns ``False`` and tests skip cleanly.
* All hidden / visible perturbation primitives in this layer are
  **deterministic** (no RNG): they pick the first ON / first OFF cell
  in each fibre. This preserves the count-preserving / count-changing
  invariants exactly while enabling exact CPU/GPU equivalence tests.
  Production runners that require randomized strategies continue to
  use the CPU implementations in
  :mod:`observer_worlds.projection.invisible_perturbations` and
  :mod:`observer_worlds.projection.visible_perturbations`; G1 does not
  alter those.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def is_cupy_available() -> bool:
    """Return True iff cupy is importable and a CUDA device is reachable."""
    try:
        from observer_worlds.worlds._cuda_bootstrap import bootstrap_cuda_path
        bootstrap_cuda_path()
        import cupy as cp  # noqa: F401
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_backend(name: str, *, device: int = 0) -> "Backend":
    """Return a :class:`Backend` instance for ``name``.

    Parameters
    ----------
    name
        ``"numpy"`` or ``"cupy"``.
    device
        CUDA device id (only consulted when ``name == "cupy"``).

    Raises
    ------
    ValueError
        Unknown backend name.
    RuntimeError
        ``name == "cupy"`` but cupy is not installed / no CUDA device.
    """
    if name == "numpy":
        from observer_worlds.backends.numpy_backend import NumpyBackend
        return NumpyBackend()
    if name == "cupy":
        if not is_cupy_available():
            raise RuntimeError(
                "cupy backend requested but cupy is not installed or no "
                "CUDA device is reachable. Install cupy-cuda12x and ensure "
                "an NVIDIA GPU is visible."
            )
        from observer_worlds.backends.cupy_backend import CupyBackend
        return CupyBackend(device=device)
    raise ValueError(
        f"unknown backend {name!r}; valid names: 'numpy', 'cupy'"
    )


@dataclass
class GpuMemoryEstimate:
    """Memory footprint estimate for a batched rollout."""
    bytes_per_state: int
    n_state_arrays: int
    bytes_per_projection_frame: int
    n_projection_frames: int
    total_bytes: int
    fits_in_target_gb: bool


def estimate_max_safe_batch_size(
    *,
    grid: Sequence[int],
    n_perturbation_conditions: int,
    n_projection_frames_per_state: int,
    target_gb: float,
    state_dtype_bytes: int = 1,        # uint8
    projection_dtype_bytes: int = 1,   # uint8 for binary projections
    safety_factor: float = 0.6,        # leave room for kernel scratch
) -> int:
    """Estimate the largest batch size B such that
    ``B * n_perturbation_conditions`` state arrays + the per-frame
    projection scratch fit in ``target_gb * safety_factor`` GiB.

    Returns at least 1.
    """
    Nx, Ny, Nz, Nw = (int(g) for g in grid)
    bytes_per_state = Nx * Ny * Nz * Nw * state_dtype_bytes
    bytes_per_proj = Nx * Ny * projection_dtype_bytes
    per_b = (
        n_perturbation_conditions * bytes_per_state
        + n_projection_frames_per_state * bytes_per_proj
    )
    budget = int(target_gb * safety_factor * (1024 ** 3))
    if per_b <= 0:
        return 1
    b = max(1, budget // per_b)
    return int(b)


class Backend:
    """Common interface implemented by NumpyBackend and CupyBackend.

    Methods take and return *batched* arrays in this layer's canonical
    shape ``(B, Nx, Ny, Nz, Nw)`` for state and ``(B, Nx, Ny[, C])``
    for projections. All array creation and arithmetic uses the
    backend's ``xp`` namespace; callers should not assume numpy.
    """

    name: str = "abstract"
    is_gpu: bool = False
    xp: Any = None

    # ---- transfer helpers -------------------------------------------------

    def asarray(self, a, dtype=None):
        """Move ``a`` onto this backend (no-op if already there)."""
        raise NotImplementedError

    def asnumpy(self, a):
        """Copy ``a`` to a host numpy array (no-op for numpy backend)."""
        raise NotImplementedError

    # ---- batched 4D CA step -----------------------------------------------

    def step_4d_batch(self, states, birth_luts, surv_luts):
        """Advance ``states`` (shape ``(B, Nx, Ny, Nz, Nw)``) one timestep.

        ``birth_luts`` / ``surv_luts`` are ``(B, 81)`` uint8 arrays.
        Returns a new state array of the same shape and dtype.
        """
        raise NotImplementedError

    # ---- batched projection -----------------------------------------------

    def project_batch(self, states, *, method: str, **params):
        """Apply the named projection to each of B states.

        Returns shape ``(B, Nx, Ny)`` for single-channel projections
        and ``(B, Nx, Ny, C)`` for ``multi_channel_projection``.
        """
        raise NotImplementedError

    # ---- perturbations ----------------------------------------------------

    def apply_hidden_perturbations_batch(
        self, states, candidate_masks, *, n_swaps_per_fibre: int = 1,
    ):
        """Deterministic count-preserving swap.

        For each ``(x, y)`` in ``candidate_masks[b]``, take ``k`` swap
        pairs in the ``(z, w)`` fibre by toggling the first ``k`` ON
        cells (in raster order) with the first ``k`` OFF cells. The
        per-fibre count is unchanged, so all count-based projections
        (``mean_threshold``, ``sum_threshold``, ``max_projection``,
        ``parity_projection``) see exactly the same projection.

        ``states`` shape ``(B, Nx, Ny, Nz, Nw)`` uint8.
        ``candidate_masks`` shape ``(B, Nx, Ny)`` uint8/bool.

        Returns a new state array of the same shape and dtype.
        """
        raise NotImplementedError

    def apply_visible_perturbations_batch(
        self, states, candidate_masks, *, n_flips_per_fibre: int = 1,
    ):
        """Deterministic count-changing flip.

        Flips the first ``k`` cells in raster order in each masked
        fibre. Almost-always changes the projection (count is no
        longer preserved). Used as the visible-perturbation control.

        Returns a new state array of the same shape and dtype.
        """
        raise NotImplementedError

    # ---- HCE rollout primitives ------------------------------------------

    def compute_candidate_local_l1_batch(
        self, projected_perturbed, projected_original, candidate_masks,
    ):
        """L1 divergence over the candidate region, per batch element.

        Returns a 1D array of length B. For multi-channel projections
        the L1 is summed across channels.
        """
        raise NotImplementedError

    def rollout_hce_batch(
        self,
        states_orig,
        states_pert,
        birth_luts,
        surv_luts,
        candidate_masks,
        *,
        horizons: Sequence[int],
        projection: str,
        projection_params: dict | None = None,
    ):
        """Rollout original + perturbed in lockstep, project at each
        horizon, return per-batch per-horizon candidate-local L1.

        States are kept on-backend through the entire loop; only the
        compact ``(B, len(horizons))`` metric array is produced. Caller
        is responsible for transferring it to host with :meth:`asnumpy`.
        """
        raise NotImplementedError
