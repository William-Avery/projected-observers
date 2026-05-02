"""CuPy backend for batched 4D-CA work.

Reuses the existing batched RawKernel from
:mod:`observer_worlds.worlds.ca4d_batch` for ``step_4d_batch``. All
other primitives are pure CuPy implementations sharing the helper
functions in :mod:`observer_worlds.backends.numpy_backend` (they take
``xp`` as an argument so they work on either array library).

The contract for Stage G1:

* States stay on the device through the timestep loop.
* The only CPU-bound work is RNG-driven weight / mask generation in
  ``random_linear_projection`` and ``multi_channel_projection``;
  weights are generated once on host and uploaded (this matches the
  CPU reference and keeps RNG deterministic across backends).
* Perturbations are deterministic (no RNG) so CPU/GPU equivalence is
  bit-exact for binary projections.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from observer_worlds.backends.backend_api import Backend
from observer_worlds.backends.numpy_backend import (
    _candidate_local_l1_xp,
    _hidden_swap_xp,
    _project_batch_xp,
    _visible_flip_xp,
)
from observer_worlds.worlds._cuda_bootstrap import bootstrap_cuda_path

bootstrap_cuda_path()

import cupy as cp  # noqa: E402


# Reuse the existing batched RawKernel (compiled once and cached).
from observer_worlds.worlds.ca4d_batch import _compile_batch_kernel  # noqa: E402


class CupyBackend(Backend):
    name = "cupy"
    is_gpu = True

    def __init__(self, *, device: int = 0) -> None:
        self.device_id = int(device)
        cp.cuda.Device(self.device_id).use()
        self.xp = cp

    # ---- transfer helpers -------------------------------------------------

    def asarray(self, a, dtype=None):
        return cp.asarray(a, dtype=dtype)

    def asnumpy(self, a):
        if isinstance(a, np.ndarray):
            return a
        return cp.asnumpy(a)

    # ---- batched 4D CA step (RawKernel) ----------------------------------

    def step_4d_batch(self, states, birth_luts, surv_luts):
        if states.ndim != 5:
            raise ValueError(
                f"step_4d_batch expects (B, Nx, Ny, Nz, Nw); got "
                f"shape {states.shape!r}"
            )
        states_d = cp.ascontiguousarray(states, dtype=cp.uint8)
        b_luts_d = cp.ascontiguousarray(birth_luts, dtype=cp.uint8)
        s_luts_d = cp.ascontiguousarray(surv_luts, dtype=cp.uint8)
        B, Nx, Ny, Nz, Nw = (int(d) for d in states_d.shape)
        if b_luts_d.shape != (B, 81) or s_luts_d.shape != (B, 81):
            raise ValueError(
                f"luts must be (B, 81); got birth {b_luts_d.shape!r}, "
                f"surv {s_luts_d.shape!r}"
            )
        out_d = cp.empty_like(states_d)
        total = B * Nx * Ny * Nz * Nw
        block = 256
        grid = (total + block - 1) // block
        kernel = _compile_batch_kernel()
        kernel(
            (grid,), (block,),
            (
                states_d, out_d, b_luts_d, s_luts_d,
                np.int32(B), np.int32(Nx), np.int32(Ny),
                np.int32(Nz), np.int32(Nw),
            ),
        )
        return out_d

    # ---- batched projection -----------------------------------------------

    def project_batch(self, states, *, method: str, **params):
        return _project_batch_xp(self.xp, states, method=method, **params)

    # ---- perturbations ----------------------------------------------------

    def apply_hidden_perturbations_batch(
        self, states, candidate_masks, *, n_swaps_per_fibre: int = 1,
    ):
        return _hidden_swap_xp(
            self.xp, states, candidate_masks,
            n_swaps_per_fibre=n_swaps_per_fibre,
        )

    def apply_visible_perturbations_batch(
        self, states, candidate_masks, *, n_flips_per_fibre: int = 1,
    ):
        return _visible_flip_xp(
            self.xp, states, candidate_masks,
            n_flips_per_fibre=n_flips_per_fibre,
        )

    # ---- HCE rollout primitives ------------------------------------------

    def compute_candidate_local_l1_batch(
        self, projected_perturbed, projected_original, candidate_masks,
    ):
        return _candidate_local_l1_xp(
            self.xp, projected_perturbed, projected_original, candidate_masks,
        )

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
        if projection_params is None:
            projection_params = {}
        max_h = int(max(horizons))
        h_set = sorted(set(int(h) for h in horizons))
        h_index = {h: i for i, h in enumerate(h_set)}

        # Make sure all inputs live on device for the loop.
        cur_o = cp.ascontiguousarray(states_orig, dtype=cp.uint8)
        cur_p = cp.ascontiguousarray(states_pert, dtype=cp.uint8)
        b_d = cp.ascontiguousarray(birth_luts, dtype=cp.uint8)
        s_d = cp.ascontiguousarray(surv_luts, dtype=cp.uint8)
        masks_d = cp.ascontiguousarray(candidate_masks)

        B = int(cur_o.shape[0])
        out = cp.zeros((B, len(h_set)), dtype=cp.float32)

        for t in range(1, max_h + 1):
            cur_o = self.step_4d_batch(cur_o, b_d, s_d)
            cur_p = self.step_4d_batch(cur_p, b_d, s_d)
            if t in h_index:
                proj_o = self.project_batch(
                    cur_o, method=projection, **projection_params,
                )
                proj_p = self.project_batch(
                    cur_p, method=projection, **projection_params,
                )
                out[:, h_index[t]] = self.compute_candidate_local_l1_batch(
                    proj_p, proj_o, masks_d,
                )
        return out
