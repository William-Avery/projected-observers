"""Causality observer-likeness score.

Functional question: does intervening on cells *belonging to* the candidate
(its interior or boundary 4D fibers) influence the projected future *more*
than an intervention of comparable magnitude on the candidate's environment?

We compare four paired forward rollouts of the 4D CA from a single 4D
snapshot, all sharing the same dynamics but with different interventions:

* ``internal``       -- flip a fraction of 4D cells under the candidate's
                        interior 2D footprint.
* ``boundary``       -- flip a fraction of 4D cells under the candidate's
                        boundary 2D footprint.
* ``environment``    -- flip a fraction of 4D cells under the candidate's
                        environment shell footprint.
* ``hidden_shuffle`` -- shuffle the (z, w) fiber under the *interior*
                        footprint; cell counts are preserved per (x, y) but
                        any coherent hidden structure is destroyed.

For each intervention type we measure mean L1 divergence of the projected
2D state versus the unperturbed baseline over an n-step window, then
combine into a single causality score:

    causality_score = D_internal + D_boundary - 2 * D_environment

This is positive when targeted, agent-localised interventions cause
disproportionately large effects relative to environment perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observer_worlds.worlds import CA4D, BSRule, project


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CausalityResult:
    track_id: int | None
    n_steps: int
    # Per-intervention mean L1 divergence over the rollout window.
    divergence_internal: float
    divergence_boundary: float
    divergence_environment: float
    divergence_hidden_shuffle: float
    # Summary score: how much more do candidate-targeted interventions
    # affect the future, relative to comparable environment perturbations.
    causality_score: float
    valid: bool
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast_mask_2d_to_4d(
    mask_2d: np.ndarray, shape_4d: tuple[int, int, int, int]
) -> np.ndarray:
    """Return a 4D bool mask where ``mask_4d[x, y, z, w] = mask_2d[x, y]``."""
    if mask_2d.shape != shape_4d[:2]:
        raise ValueError(
            f"mask_2d shape {mask_2d.shape} does not match 4D leading shape "
            f"{shape_4d[:2]}"
        )
    mask_2d_b = np.asarray(mask_2d, dtype=bool)
    return np.broadcast_to(mask_2d_b[:, :, None, None], shape_4d).copy()


def apply_flip_intervention(
    state_4d: np.ndarray,
    mask_2d: np.ndarray,
    flip_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of ``state_4d`` with a random ``flip_fraction`` of
    4D cells inside the broadcasted ``mask_2d`` XORed with 1.

    If ``flip_fraction <= 0`` or the mask is empty, the returned array is an
    untouched copy.
    """
    if state_4d.ndim != 4:
        raise ValueError(f"state_4d must be 4D, got shape {state_4d.shape}")
    out = state_4d.copy()
    if flip_fraction <= 0.0:
        return out

    mask_4d = _broadcast_mask_2d_to_4d(mask_2d, state_4d.shape)  # type: ignore[arg-type]
    flat_idx = np.flatnonzero(mask_4d.ravel())
    n_target = flat_idx.size
    if n_target == 0:
        return out

    n_flip = int(round(n_target * float(flip_fraction)))
    n_flip = max(0, min(n_flip, n_target))
    if n_flip == 0:
        return out

    chosen = rng.choice(flat_idx, size=n_flip, replace=False)
    flat_out = out.ravel()
    # XOR with 1 to flip 0<->1 in the uint8/bool state.
    flat_out[chosen] = flat_out[chosen] ^ np.uint8(1)
    return flat_out.reshape(state_4d.shape)


def apply_hidden_shuffle_intervention(
    state_4d: np.ndarray,
    mask_2d: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of ``state_4d`` where, for each ``(x, y)`` in
    ``mask_2d``, the ``(z, w)`` fiber is randomly permuted in place.

    Cell counts are preserved per ``(x, y)``; hidden-axis structure is
    destroyed.
    """
    if state_4d.ndim != 4:
        raise ValueError(f"state_4d must be 4D, got shape {state_4d.shape}")
    out = state_4d.copy()
    fiber_shape = state_4d.shape[2:]
    coords = np.argwhere(np.asarray(mask_2d, dtype=bool))
    for x, y in coords:
        flat = out[x, y].ravel().copy()
        rng.shuffle(flat)
        out[x, y] = flat.reshape(fiber_shape)
    return out


def rollout(
    state_4d: np.ndarray,
    rule: BSRule,
    n_steps: int,
    *,
    backend: str = "numpy",
    projection_method: str = "mean_threshold",
    projection_theta: float = 0.5,
) -> np.ndarray:
    """Run the 4D CA forward for ``n_steps``, returning projected frames.

    Returns a ``uint8`` array of shape ``(n_steps, Nx, Ny)`` containing the
    projected 2D state after each step (the initial frame is *not* included).
    """
    if state_4d.ndim != 4:
        raise ValueError(f"state_4d must be 4D, got shape {state_4d.shape}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")

    nx, ny, _, _ = state_4d.shape
    frames = np.empty((n_steps, nx, ny), dtype=np.uint8)
    if n_steps == 0:
        return frames

    ca = CA4D(shape=state_4d.shape, rule=rule, backend=backend)  # type: ignore[arg-type]
    ca.state = state_4d.copy()
    for t in range(n_steps):
        ca.step()
        proj = project(
            ca.state, method=projection_method, theta=projection_theta
        )
        # Coerce to uint8 in case the projection returned int32 (e.g. "sum").
        frames[t] = proj.astype(np.uint8)
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _mean_l1_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over time of mean per-cell L1 distance between two stacks.

    Both arrays have shape ``(T, Nx, Ny)``.  Returns the average over ``T``
    of ``mean(|a[t] - b[t]|)``, which equals ``mean(|a - b|)`` for equal-T
    inputs but is computed in two stages for clarity.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] == 0:
        return 0.0
    a_i = a.astype(np.int16)
    b_i = b.astype(np.int16)
    per_t = np.abs(a_i - b_i).reshape(a.shape[0], -1).mean(axis=1)
    return float(per_t.mean())


def compute_causality_score(
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask_2d: np.ndarray,
    boundary_mask_2d: np.ndarray,
    env_mask_2d: np.ndarray,
    *,
    n_steps: int = 10,
    flip_fraction: float = 0.5,
    projection_method: str = "mean_threshold",
    projection_theta: float = 0.5,
    backend: str = "numpy",
    seed: int = 0,
    track_id: int | None = None,
) -> CausalityResult:
    """Run paired rollouts under four intervention types and measure
    divergence of the projected 2D future relative to the unperturbed
    baseline.

    See module docstring for the conceptual model.
    """
    if snapshot_4d.ndim != 4:
        raise ValueError(
            f"snapshot_4d must be 4D, got shape {snapshot_4d.shape}"
        )

    if (
        int(np.asarray(interior_mask_2d, dtype=bool).sum()) == 0
        or int(np.asarray(boundary_mask_2d, dtype=bool).sum()) == 0
        or int(np.asarray(env_mask_2d, dtype=bool).sum()) == 0
    ):
        return CausalityResult(
            track_id=track_id,
            n_steps=n_steps,
            divergence_internal=float("nan"),
            divergence_boundary=float("nan"),
            divergence_environment=float("nan"),
            divergence_hidden_shuffle=float("nan"),
            causality_score=float("nan"),
            valid=False,
            reason="empty_mask",
        )

    # Use independent RNG streams per intervention so that, e.g., the
    # boundary intervention's randomness does not depend on the internal
    # one's draw size.  Spawning child generators from a single seed keeps
    # the function deterministic w.r.t. ``seed``.
    parent_rng = np.random.default_rng(seed)
    rng_internal, rng_boundary, rng_env, rng_shuffle = [
        np.random.default_rng(s) for s in parent_rng.integers(0, 2**63 - 1, size=4)
    ]

    rollout_kwargs = dict(
        backend=backend,
        projection_method=projection_method,
        projection_theta=projection_theta,
    )

    # Baseline (unperturbed) rollout.
    baseline = rollout(snapshot_4d, rule, n_steps, **rollout_kwargs)

    # Per-intervention rollouts.
    state_internal = apply_flip_intervention(
        snapshot_4d, interior_mask_2d, flip_fraction, rng_internal
    )
    state_boundary = apply_flip_intervention(
        snapshot_4d, boundary_mask_2d, flip_fraction, rng_boundary
    )
    state_env = apply_flip_intervention(
        snapshot_4d, env_mask_2d, flip_fraction, rng_env
    )
    state_shuffle = apply_hidden_shuffle_intervention(
        snapshot_4d, interior_mask_2d, rng_shuffle
    )

    y_internal = rollout(state_internal, rule, n_steps, **rollout_kwargs)
    y_boundary = rollout(state_boundary, rule, n_steps, **rollout_kwargs)
    y_env = rollout(state_env, rule, n_steps, **rollout_kwargs)
    y_shuffle = rollout(state_shuffle, rule, n_steps, **rollout_kwargs)

    d_internal = _mean_l1_divergence(baseline, y_internal)
    d_boundary = _mean_l1_divergence(baseline, y_boundary)
    d_env = _mean_l1_divergence(baseline, y_env)
    d_shuffle = _mean_l1_divergence(baseline, y_shuffle)

    score = d_internal + d_boundary - 2.0 * d_env

    return CausalityResult(
        track_id=track_id,
        n_steps=n_steps,
        divergence_internal=d_internal,
        divergence_boundary=d_boundary,
        divergence_environment=d_env,
        divergence_hidden_shuffle=d_shuffle,
        causality_score=float(score),
        valid=True,
        reason="ok",
    )
