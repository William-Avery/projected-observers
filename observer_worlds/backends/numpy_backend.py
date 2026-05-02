"""NumPy reference backend for the batched 4D-CA backend API.

This is the source of truth: every primitive here matches the
scientific definitions used by Stage 5 / Stage 6 production code. The
cupy backend is required to match this implementation bit-for-bit on
binary primitives and within tolerance on continuous ones.

State convention: ``(B, Nx, Ny, Nz, Nw)`` uint8.
Projection convention: ``(B, Nx, Ny)`` for single-channel,
``(B, Nx, Ny, C)`` for multi-channel.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import convolve

from observer_worlds.backends.backend_api import Backend


_NEIGH_KERNEL_4D = np.ones((3, 3, 3, 3), dtype=np.int32)
_NEIGH_KERNEL_4D[1, 1, 1, 1] = 0


def _step_4d_numpy_single(
    state: np.ndarray, birth_lut: np.ndarray, surv_lut: np.ndarray,
) -> np.ndarray:
    """One CA step on a single (Nx, Ny, Nz, Nw) state. Mirrors
    :func:`observer_worlds.worlds.ca4d.update_4d_numpy` exactly."""
    counts = convolve(state.astype(np.int32), _NEIGH_KERNEL_4D, mode="wrap")
    alive = state.astype(bool)
    new = np.where(alive, surv_lut[counts], birth_lut[counts])
    return new.astype(np.uint8)


class NumpyBackend(Backend):
    name = "numpy"
    is_gpu = False

    def __init__(self) -> None:
        self.xp = np

    # ---- transfer helpers -------------------------------------------------

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=dtype)

    def asnumpy(self, a):
        return np.asarray(a)

    # ---- batched 4D CA step -----------------------------------------------

    def step_4d_batch(self, states, birth_luts, surv_luts):
        if states.ndim != 5:
            raise ValueError(
                f"step_4d_batch expects (B, Nx, Ny, Nz, Nw); got "
                f"shape {states.shape!r}"
            )
        B = int(states.shape[0])
        if birth_luts.shape != (B, 81) or surv_luts.shape != (B, 81):
            raise ValueError(
                f"luts must be (B, 81); got birth {birth_luts.shape!r}, "
                f"surv {surv_luts.shape!r}"
            )
        out = np.empty_like(states)
        for b in range(B):
            out[b] = _step_4d_numpy_single(
                states[b], birth_luts[b], surv_luts[b],
            )
        return out

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

        B = int(states_orig.shape[0])
        out = self.xp.zeros((B, len(h_set)), dtype=self.xp.float32)

        cur_o = states_orig
        cur_p = states_pert
        for t in range(1, max_h + 1):
            cur_o = self.step_4d_batch(cur_o, birth_luts, surv_luts)
            cur_p = self.step_4d_batch(cur_p, birth_luts, surv_luts)
            if t in h_index:
                proj_o = self.project_batch(
                    cur_o, method=projection, **projection_params,
                )
                proj_p = self.project_batch(
                    cur_p, method=projection, **projection_params,
                )
                out[:, h_index[t]] = self.compute_candidate_local_l1_batch(
                    proj_p, proj_o, candidate_masks,
                )
        return out


# ---------------------------------------------------------------------------
# Pure-xp helpers shared by both backends
# ---------------------------------------------------------------------------


def _project_batch_xp(xp, states, *, method: str, **params):
    """Apply a 4D->2D projection to every batch element using ``xp``.

    ``states`` has shape ``(B, Nx, Ny, Nz, Nw)``. Reduces over axes
    ``(3, 4)``. Semantics match
    :mod:`observer_worlds.worlds.projection` and
    :mod:`observer_worlds.projection.projection_suite`.
    """
    if states.ndim != 5:
        raise ValueError(
            f"project_batch expects (B, Nx, Ny, Nz, Nw); got "
            f"shape {states.shape!r}"
        )
    if method == "mean_threshold":
        theta = float(params.get("theta", 0.5))
        mean = states.astype(xp.float64).mean(axis=(3, 4))
        return (mean > theta).astype(xp.uint8)
    if method == "sum_threshold":
        theta = int(params.get("theta", 1))
        s = states.astype(xp.int32).sum(axis=(3, 4))
        return (s >= theta).astype(xp.uint8)
    if method == "max_projection":
        return states.max(axis=(3, 4)).astype(xp.uint8)
    if method == "parity_projection":
        s = states.astype(xp.int32).sum(axis=(3, 4))
        return (s & 1).astype(xp.uint8)
    if method == "random_linear_projection":
        seed = int(params.get("seed", 0))
        Nz, Nw = int(states.shape[3]), int(states.shape[4])
        weights = _random_linear_weights_xp(xp, Nz, Nw, seed=seed)
        # Avoid xp.einsum: cupy routes einsum through cuBLAS, but the
        # default install only ships nvrtc. Elementwise mul + sum is
        # marginally less efficient but uses only RawKernel-class ops.
        weighted = states.astype(xp.float32) * weights[None, None, None, :, :]
        return weighted.sum(axis=(3, 4))
    if method == "multi_channel_projection":
        seed = int(params.get("seed", 0))
        n_channels = int(params.get("n_channels", 4))
        Nz, Nw = int(states.shape[3]), int(states.shape[4])
        masks = _multi_channel_masks_xp(xp, Nz, Nw, n_channels=n_channels, seed=seed)
        # masks: (C, Nz, Nw); states: (B, Nx, Ny, Nz, Nw); result: (B, Nx, Ny, C)
        # Loop over channels (small, typically 4) instead of a 6D temp
        # or einsum. Each channel does one (B, Nx, Ny) reduction.
        states_f = states.astype(xp.float32)
        out_channels = []
        for c in range(int(n_channels)):
            m = masks[c]
            denom = float(m.sum())
            w = (states_f * m[None, None, None, :, :]).sum(axis=(3, 4)) / denom
            out_channels.append((w > 0.5).astype(xp.uint8))
        return xp.stack(out_channels, axis=-1)
    raise ValueError(f"unknown projection method {method!r}")


def _random_linear_weights_xp(xp, nz: int, nw: int, *, seed: int = 0):
    """Generate weights on host (numpy) and transfer to ``xp``.

    The reference implementation uses ``np.random.default_rng`` for
    deterministic cross-backend equivalence (cupy's RNG has a different
    bit stream from numpy's even with identical seeds).
    """
    rng = np.random.default_rng(int(seed))
    w = rng.standard_normal((int(nz), int(nw))).astype(np.float32)
    if xp is np:
        return w
    return xp.asarray(w)


def _multi_channel_masks_xp(xp, nz: int, nw: int, *, n_channels: int = 4, seed: int = 0):
    """Generate masks on host (numpy) and transfer to ``xp``."""
    rng = np.random.default_rng(int(seed))
    masks = np.empty((int(n_channels), int(nz), int(nw)), dtype=np.float32)
    for c in range(int(n_channels)):
        m = rng.integers(0, 2, size=(int(nz), int(nw))).astype(np.float32)
        if m.sum() == 0:
            m[0, 0] = 1.0
        masks[c] = m
    if xp is np:
        return masks
    return xp.asarray(masks)


def _hidden_swap_xp(xp, states, candidate_masks, *, n_swaps_per_fibre: int = 1):
    """Deterministic count-preserving swap.

    For each fibre ``(b, x, y)`` where ``candidate_masks[b, x, y]`` is
    truthy, scan the flattened ``(z, w)`` fibre in raster order and
    swap (toggle) the first ``n_swaps_per_fibre`` ON cells with the
    first ``n_swaps_per_fibre`` OFF cells. If the fibre has fewer
    than ``n_swaps_per_fibre`` of either, take ``min(n_on, n_off,
    n_swaps_per_fibre)`` swaps.

    Implemented in pure xp via a vectorized argsort trick: we sort the
    fibre's values descending so all ON cells come first, and pair the
    first ``k`` ONs with the first ``k`` OFFs in the original raster
    order. Toggling both members of a swap pair preserves the count.
    """
    if states.ndim != 5:
        raise ValueError(
            f"states must be (B, Nx, Ny, Nz, Nw); got {states.shape!r}"
        )
    if candidate_masks.shape != states.shape[:3]:
        raise ValueError(
            f"candidate_masks must be (B, Nx, Ny); got "
            f"{candidate_masks.shape!r} vs states {states.shape!r}"
        )
    B, Nx, Ny, Nz, Nw = (int(d) for d in states.shape)
    fibre_size = Nz * Nw
    out = states.copy()

    # Flatten to (B*Nx*Ny, Nz*Nw) for vectorized fibre ops.
    flat = out.reshape(B * Nx * Ny, fibre_size)
    mask_flat = candidate_masks.reshape(B * Nx * Ny).astype(bool)
    if not bool(mask_flat.any()):
        return out

    sub = flat[mask_flat]                              # (M, F)
    M = int(sub.shape[0])
    # ON positions in raster order, then OFF positions in raster order.
    is_on = sub.astype(xp.bool_)
    # cumulative ON-rank along the fibre: 1-indexed for ON cells, 0 for OFF.
    on_rank = xp.where(is_on, xp.cumsum(is_on, axis=1), 0)
    off_rank = xp.where(~is_on, xp.cumsum(~is_on, axis=1), 0)

    k = int(n_swaps_per_fibre)
    # toggle ON cells whose on_rank in [1, k] AND OFF cells whose off_rank in [1, k]
    swap_mask = (
        (is_on & (on_rank >= 1) & (on_rank <= k))
        | (~is_on & (off_rank >= 1) & (off_rank <= k))
    )
    # Also enforce: only swap as many as both sides have. We need to
    # cap k per fibre at min(total_on, total_off). Recompute swap_mask
    # with the per-fibre cap.
    total_on = is_on.sum(axis=1)                       # (M,)
    total_off = fibre_size - total_on
    per_fibre_k = xp.minimum(xp.minimum(total_on, total_off), k)  # (M,)
    swap_mask = (
        (is_on & (on_rank >= 1) & (on_rank <= per_fibre_k[:, None]))
        | (~is_on & (off_rank >= 1) & (off_rank <= per_fibre_k[:, None]))
    )

    sub = xp.where(swap_mask, 1 - sub, sub).astype(states.dtype)
    flat[mask_flat] = sub
    return flat.reshape(B, Nx, Ny, Nz, Nw)


def _visible_flip_xp(xp, states, candidate_masks, *, n_flips_per_fibre: int = 1):
    """Deterministic flip of the first ``k`` cells (raster order) in
    each masked fibre. Almost-always changes the projection.
    """
    if states.ndim != 5:
        raise ValueError(
            f"states must be (B, Nx, Ny, Nz, Nw); got {states.shape!r}"
        )
    if candidate_masks.shape != states.shape[:3]:
        raise ValueError(
            f"candidate_masks must be (B, Nx, Ny); got "
            f"{candidate_masks.shape!r}"
        )
    B, Nx, Ny, Nz, Nw = (int(d) for d in states.shape)
    fibre_size = Nz * Nw
    out = states.copy()
    flat = out.reshape(B * Nx * Ny, fibre_size)
    mask_flat = candidate_masks.reshape(B * Nx * Ny).astype(bool)
    if not bool(mask_flat.any()):
        return out
    k = min(int(n_flips_per_fibre), fibre_size)
    sub = flat[mask_flat]
    sub[:, :k] = 1 - sub[:, :k]
    flat[mask_flat] = sub
    return flat.reshape(B, Nx, Ny, Nz, Nw)


def _candidate_local_l1_xp(xp, projected_perturbed, projected_original, candidate_masks):
    """Sum |perturbed - original| over candidate region, per batch element.

    Both inputs ``(B, Nx, Ny)`` or ``(B, Nx, Ny, C)``. ``candidate_masks``
    is ``(B, Nx, Ny)``. Returns ``(B,)`` float32.
    """
    diff = xp.abs(
        projected_perturbed.astype(xp.float32)
        - projected_original.astype(xp.float32)
    )
    # Broadcast mask over channels if present.
    while diff.ndim > candidate_masks.ndim:
        candidate_masks = candidate_masks[..., None]
    masked = diff * candidate_masks.astype(xp.float32)
    # Sum over all but batch.
    axes = tuple(range(1, masked.ndim))
    return masked.sum(axis=axes).astype(xp.float32)
