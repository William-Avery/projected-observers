"""GPU-side helper for the projection-robustness HCE rollout phase.

This module is the GPU counterpart of the perturbed-rollout body inside
:func:`observer_worlds.experiments._followup_projection.measure_candidate_under_projection`.
The CPU side of the runner remains responsible for:

* Substrate rollout, projection-stream construction, binarisation.
* Tracker + candidate selection.
* Construction of ``s_hidden_r`` / ``s_far_r`` via
  :func:`make_projection_invisible_perturbation` (so the RNG path
  matches the CPU runner exactly — this is the load-bearing condition
  for bit-equivalent HCE numbers).

The GPU side here only does the perturbed-rollout-to-each-horizon and
the candidate-local L1 measurement. States are uploaded once per
chunk and remain device-resident for the entire timestep loop; only
the compact ``(B, len(horizons))`` HCE / far_HCE matrices come back.

The ``measure_batch_on_gpu`` API is intentionally narrow: it operates
on a single (chunk, projection) group. The outer runner is responsible
for grouping work by projection and chunking by ``--gpu-batch-size``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from observer_worlds.backends.backend_api import Backend
from observer_worlds.backends.numpy_backend import (
    _candidate_local_l1_mean_xp,
    _project_batch_xp,
)


def measure_batch_on_gpu(
    *,
    backend: Backend,
    states_orig: np.ndarray,
    states_hidden: np.ndarray,
    states_far: np.ndarray,
    birth_luts: np.ndarray,
    surv_luts: np.ndarray,
    candidate_local_masks: np.ndarray,
    horizons: Sequence[int],
    avail_steps: np.ndarray,
    projection: str,
    projection_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single (projection, chunk) batch through the device.

    Parameters
    ----------
    backend
        Backend instance (numpy or cupy). Both are accepted; the numpy
        backend serves as the equivalence reference.
    states_orig, states_hidden, states_far
        ``(B, Nx, Ny, Nz, Nw)`` ``uint8`` host arrays. ``orig`` is the
        unperturbed substrate state at the candidate's peak frame;
        ``hidden`` is the projection-invisible perturbation in the
        candidate bbox; ``far`` is the same-shape perturbation outside.
    birth_luts, surv_luts
        ``(B, 81)`` ``uint8`` host arrays. Per-item rule LUTs.
    candidate_local_masks
        ``(B, Nx, Ny)`` ``uint8``/bool host array. The interior /
        peak-mask of each candidate, used as the L1-mean mask.
    horizons
        Sorted unique list of horizons to project at. Returned matrices
        have ``len(horizons)`` columns in the same order.
    avail_steps
        ``(B,)`` int — per-item upper bound on horizon
        (``state_stream_len - 1 - peak_frame``). Returned entries
        beyond this are reported as ``np.nan``.
    projection
        Projection name ("mean_threshold", "sum_threshold", ...).
    projection_params
        Optional kwargs forwarded to :func:`_project_batch_xp`.

    Returns
    -------
    (HCE, far_HCE)
        Each ``(B, len(horizons))`` ``float32`` numpy array. Entries
        where ``horizon > avail_steps[b]`` are ``np.nan``.
    """
    if projection_params is None:
        projection_params = {}
    B = int(states_orig.shape[0])
    if states_hidden.shape != states_orig.shape or states_far.shape != states_orig.shape:
        raise ValueError(
            "states_orig / states_hidden / states_far must have identical "
            f"shapes; got {states_orig.shape!r}, {states_hidden.shape!r}, "
            f"{states_far.shape!r}"
        )
    if birth_luts.shape != (B, 81) or surv_luts.shape != (B, 81):
        raise ValueError(
            f"luts must be (B, 81); got birth {birth_luts.shape!r}, "
            f"surv {surv_luts.shape!r}"
        )
    if candidate_local_masks.shape[0] != B:
        raise ValueError(
            f"candidate_local_masks first dim must be B={B}; got "
            f"{candidate_local_masks.shape!r}"
        )
    h_set = sorted(set(int(h) for h in horizons))
    h_index = {h: i for i, h in enumerate(h_set)}
    if not h_set:
        raise ValueError("horizons must be a non-empty sequence")
    max_h = h_set[-1]

    xp = backend.xp
    # Stack the three condition slabs into one (3B, Nx, Ny, Nz, Nw)
    # batch and one (3B, 81) LUT block. We reuse the per-item LUT for
    # all three conditions of that item (same rule). This keeps the
    # rollout for orig/hidden/far in lockstep on a single kernel.
    stacked_states = np.concatenate(
        [states_orig, states_hidden, states_far], axis=0,
    )
    stacked_luts_b = np.concatenate([birth_luts] * 3, axis=0)
    stacked_luts_s = np.concatenate([surv_luts] * 3, axis=0)

    # Single device upload (the no-copy contract: nothing inside the
    # timestep loop touches the host).
    cur = backend.asarray(stacked_states)
    bl = backend.asarray(stacked_luts_b)
    sl = backend.asarray(stacked_luts_s)
    masks_d = backend.asarray(candidate_local_masks)

    # On-device output matrices.
    hce_d = xp.full((B, len(h_set)), float("nan"), dtype=xp.float32)
    far_d = xp.full((B, len(h_set)), float("nan"), dtype=xp.float32)

    # Per-item validity mask for each horizon column.
    # (B, len(h_set)) host bool — uploaded once.
    h_arr = np.asarray(h_set, dtype=np.int32)
    valid_h = (h_arr[None, :] <= np.asarray(avail_steps, dtype=np.int32)[:, None])
    valid_h_d = backend.asarray(valid_h)

    for t in range(1, max_h + 1):
        cur = backend.step_4d_batch(cur, bl, sl)
        if t in h_index:
            proj = backend.project_batch(
                cur, method=projection, **projection_params,
            )
            # Slice the three condition slabs.
            proj_orig = proj[:B]
            proj_hidden = proj[B:2 * B]
            proj_far = proj[2 * B:3 * B]
            l1_h = _candidate_local_l1_mean_xp(
                xp, proj_hidden, proj_orig, masks_d,
            )
            l1_f = _candidate_local_l1_mean_xp(
                xp, proj_far, proj_orig, masks_d,
            )
            col = h_index[t]
            # Apply per-item validity mask: if avail_steps[b] < t, leave NaN.
            v = valid_h_d[:, col]
            hce_d[:, col] = xp.where(v, l1_h, hce_d[:, col])
            far_d[:, col] = xp.where(v, l1_f, far_d[:, col])

    # Compact metric matrices off-device.
    return backend.asnumpy(hce_d), backend.asnumpy(far_d)


def project_batch_host_to_xp(backend: Backend, states_h: np.ndarray, *,
                              method: str, **params):
    """Convenience: project a host states array via the backend.

    Used by the equivalence audit where one side keeps things on host
    (numpy backend) and the other uses cupy.
    """
    states_d = backend.asarray(states_h)
    return backend.asnumpy(
        _project_batch_xp(backend.xp, states_d, method=method, **params),
    )
