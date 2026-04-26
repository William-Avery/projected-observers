"""M6B — extended hidden-state interventions.

These build on M6's `apply_hidden_shuffle_intervention` (a within-column
permutation that preserves both per-column count *and* mean-threshold
projection). M6B adds:

  * ``apply_one_time_scramble_intervention`` — replace each candidate
    column's z,w fiber with a *fresh* uniform-random arrangement at the
    same active count. Stronger than permutation: destroys any spatial
    correlation among hidden bits while still preserving projection.
  * ``apply_fiber_replacement_intervention`` — swap each candidate
    column's z,w fiber with a fiber from a different (x,y) location
    that has the same active count (and therefore same projected
    value). Tests whether the *specific* hidden microstate matters
    versus the count.
  * ``apply_far_hidden_intervention`` — translate the candidate's
    interior mask to a far region (antipodal under periodic boundaries)
    and apply a hidden shuffle there. Localization control: if the HCE
    is candidate-local, the far variant should produce noticeably less
    candidate-footprint divergence.
  * ``apply_sham_intervention`` — identity. Establishes the numerical
    floor (and confirms paired rollouts of identical states give zero
    divergence by construction).

All non-sham interventions on candidate-interior cells preserve
``project(state, "mean_threshold", theta=0.5)`` by construction. This
is the central invariant the regression tests guard.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sham
# ---------------------------------------------------------------------------


def apply_sham_intervention(
    state: np.ndarray,
    mask_2d: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Identity. Returns ``state.copy()``."""
    return state.copy()


# ---------------------------------------------------------------------------
# One-time scramble (fresh random fill at same active count per column)
# ---------------------------------------------------------------------------


def apply_one_time_scramble_intervention(
    state: np.ndarray,
    mask_2d: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace every (x,y) column under ``mask_2d`` with a *new* random
    z,w arrangement that has the same active count.

    Vs. ``apply_hidden_shuffle_intervention``: shuffle is a
    permutation of the existing bits (so cells often agree by luck);
    one_time_scramble draws a fresh uniform sample at matched count
    (so on average ~half the bits flip).
    """
    out = state.copy()
    Nz, Nw = state.shape[2], state.shape[3]
    n_total = Nz * Nw
    coords = np.argwhere(mask_2d)
    for x, y in coords:
        column = state[x, y]
        n_active = int(column.sum())
        flat = np.zeros(n_total, dtype=state.dtype)
        if n_active > 0:
            picks = rng.choice(n_total, size=n_active, replace=False)
            flat[picks] = 1
        out[x, y] = flat.reshape(Nz, Nw)
    return out


# ---------------------------------------------------------------------------
# Fiber replacement (swap candidate fibers with bucket-matched fibers)
# ---------------------------------------------------------------------------


def apply_fiber_replacement_intervention(
    state: np.ndarray,
    mask_2d: np.ndarray,
    rng: np.random.Generator,
    *,
    match_strategy: str = "exact_count",
) -> np.ndarray:
    """For each (x,y) under ``mask_2d``, replace its z,w fiber with the
    fiber from a different (x',y') location whose column is
    bucket-matched on ``match_strategy``.

    ``match_strategy`` options:
      * ``"exact_count"``: donor must have the same active count
        (preserves projection AND count). May fail to find a donor for
        rare counts; falls back to ``"projection_value"``.
      * ``"projection_value"``: donor projects to the same mean-threshold
        bin (above or below 0.5). Preserves projection; count may differ
        slightly.

    Donors are drawn from the complement of ``mask_2d`` so the swap is
    truly *replacement* (not an internal permutation).
    """
    out = state.copy()
    Nx, Ny, Nz, Nw = state.shape
    n_total = Nz * Nw
    threshold = n_total / 2.0
    counts = state.reshape(Nx, Ny, -1).sum(axis=-1)

    # Build donor pools by exact count and by projection value, restricted
    # to non-mask cells.
    not_mask = ~mask_2d
    by_count: dict[int, list[tuple[int, int]]] = {}
    above_pool: list[tuple[int, int]] = []
    below_pool: list[tuple[int, int]] = []
    coords_iter = np.argwhere(not_mask)
    for x, y in coords_iter:
        c = int(counts[x, y])
        by_count.setdefault(c, []).append((int(x), int(y)))
        if c > threshold:
            above_pool.append((int(x), int(y)))
        else:
            below_pool.append((int(x), int(y)))

    candidate_coords = np.argwhere(mask_2d)
    for x, y in candidate_coords:
        c = int(counts[x, y])
        donor: tuple[int, int] | None = None
        if match_strategy == "exact_count" and c in by_count and by_count[c]:
            pool = by_count[c]
            donor = pool[rng.integers(0, len(pool))]
        if donor is None:
            # Projection-value fallback.
            pool = above_pool if c > threshold else below_pool
            if pool:
                donor = pool[rng.integers(0, len(pool))]
        if donor is None:
            # Last resort: leave column unchanged.
            continue
        dx, dy = donor
        out[x, y] = state[dx, dy].copy()
    return out


# ---------------------------------------------------------------------------
# Far hidden intervention (localization control)
# ---------------------------------------------------------------------------


def build_far_mask(
    mask_2d: np.ndarray,
    *,
    Nx: int | None = None,
    Ny: int | None = None,
) -> np.ndarray:
    """Return a translated copy of ``mask_2d`` shifted by half the grid
    in both axes (using periodic wrap), so it lies as far as possible
    from the original mask.

    If the input mask is empty, returns an empty mask.
    If the translated mask intersects the original (e.g. mask spans
    most of the grid), the intersection is removed from the far mask
    so localization remains meaningful.
    """
    if Nx is None or Ny is None:
        Nx, Ny = mask_2d.shape
    if not mask_2d.any():
        return np.zeros_like(mask_2d)
    rows, cols = np.where(mask_2d)
    far_rows = (rows + Nx // 2) % Nx
    far_cols = (cols + Ny // 2) % Ny
    far_mask = np.zeros_like(mask_2d)
    far_mask[far_rows, far_cols] = True
    far_mask &= ~mask_2d  # no overlap with the original
    return far_mask


def apply_far_hidden_intervention(
    state: np.ndarray,
    mask_2d: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a hidden-invisible shuffle on the *far* mask (translated
    half-grid from the candidate's footprint). Returns
    ``(perturbed_state, far_mask)`` so the caller can verify
    localization (footprint divergence in the far mask should be near
    zero for the candidate's region).
    """
    from observer_worlds.metrics.causality_score import (
        apply_hidden_shuffle_intervention,
    )

    Nx, Ny = state.shape[0], state.shape[1]
    far_mask = build_far_mask(mask_2d, Nx=Nx, Ny=Ny)
    if not far_mask.any():
        return state.copy(), far_mask
    out = apply_hidden_shuffle_intervention(state, far_mask, rng)
    return out, far_mask


# ---------------------------------------------------------------------------
# Convenience: registry mapping intervention name → callable
# ---------------------------------------------------------------------------


# Returned by intervention-name lookup. Callable signature:
#   (state, mask_2d, rng) -> state'
# `apply_far_hidden_intervention` is special: returns (state', far_mask).
INTERVENTION_NAMES_M6B: tuple[str, ...] = (
    "sham",
    "hidden_invisible_local",       # M6's z,w shuffle (re-exported for clarity)
    "one_time_scramble_local",
    "fiber_replacement_local",
    "hidden_invisible_far",
    "visible_match_count",          # M5/M6 control (bit-matched random flip)
)
