"""Sensory-vs-active classification of boundary cells.

For each cell that ever appears in a track's boundary mask we ask: does
information flow inward (sensory-like) or outward (active-like) through
this cell?

We use a simple correlation-based proxy:

    sensory_corr(r,c) = corr( e[i],     s_rc[i+1] )   # env at t -> cell at t+1
    active_corr(r,c)  = corr( s_rc[i],  e[i+1]    )   # cell at t -> env at t+1

where ``e[t] = features.env_count[t]`` is a scalar environment summary and
``s_rc[t]`` is the boundary cell's (bool) state at time ``t``.  The pairs
``i, i+1`` are restricted to *contiguous* observed frames.

A cell is classified by which directional correlation has the larger
absolute value (negative correlation is still informative directionally).
Cells with zero variance in their state -- always-on or always-off -- are
skipped.

The track-level summary is the fraction of classified cells that came out
sensory-like (``sensory_fraction``) plus its complement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observer_worlds.metrics.features import TrackFeatures


@dataclass
class BoundaryClassification:
    track_id: int
    sensory_fraction: float     # fraction of boundary cells classified as sensory
    active_fraction: float      # complement
    n_boundary_cells_total: int  # total cells classified across all frames
    valid: bool
    reason: str


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns ``NaN`` if either input has zero variance."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    cm = np.corrcoef(a, b)
    val = float(cm[0, 1])
    if not np.isfinite(val):
        return float("nan")
    return val


def classify_boundary(
    features: TrackFeatures,
    *,
    min_samples: int = 8,
) -> BoundaryClassification:
    """Classify each boundary cell as sensory-like vs active-like.

    See module docstring for the full description.
    """
    track_id = features.track_id
    n_obs = features.n_obs

    if n_obs < min_samples:
        return BoundaryClassification(
            track_id=track_id,
            sensory_fraction=float("nan"),
            active_fraction=float("nan"),
            n_boundary_cells_total=0,
            valid=False,
            reason="too_short",
        )

    pairs = features.contiguous_pairs()
    if pairs.size == 0:
        return BoundaryClassification(
            track_id=track_id,
            sensory_fraction=float("nan"),
            active_fraction=float("nan"),
            n_boundary_cells_total=0,
            valid=False,
            reason="too_short",
        )

    if not features.boundary_masks:
        return BoundaryClassification(
            track_id=track_id,
            sensory_fraction=float("nan"),
            active_fraction=float("nan"),
            n_boundary_cells_total=0,
            valid=False,
            reason="no_boundary_cells",
        )

    # Union mask: any cell that was a boundary cell in *any* observed frame.
    union = np.zeros_like(features.boundary_masks[0], dtype=bool)
    for m in features.boundary_masks:
        union |= m.astype(bool)

    if not np.any(union):
        return BoundaryClassification(
            track_id=track_id,
            sensory_fraction=float("nan"),
            active_fraction=float("nan"),
            n_boundary_cells_total=0,
            valid=False,
            reason="no_boundary_cells",
        )

    e = features.env_count.astype(np.float64)
    e_t = e[pairs]
    e_tp1 = e[pairs + 1]

    # Pre-stack masks as (T_obs, H, W) for efficient cell extraction.
    mask_stack = np.stack(
        [m.astype(bool) for m in features.boundary_masks], axis=0
    )  # (T_obs, H, W)

    cells = np.argwhere(union)  # (n_cells, 2)
    sensory_count = 0
    active_count = 0

    for r, c in cells:
        s_rc = mask_stack[:, r, c].astype(np.float64)
        s_t = s_rc[pairs]
        s_tp1 = s_rc[pairs + 1]

        sensory_corr = _safe_corr(e_t, s_tp1)   # env at t -> cell at t+1
        active_corr = _safe_corr(s_t, e_tp1)    # cell at t -> env at t+1

        sens_nan = not np.isfinite(sensory_corr)
        act_nan = not np.isfinite(active_corr)
        if sens_nan and act_nan:
            continue

        # If only one direction has variance, classify by that one.
        if sens_nan:
            active_count += 1
            continue
        if act_nan:
            sensory_count += 1
            continue

        if abs(sensory_corr) > abs(active_corr):
            sensory_count += 1
        else:
            active_count += 1

    total = sensory_count + active_count
    if total == 0:
        # Cells exist but none had useful variance to classify.  Spec says
        # 0.5 if neither.  Still mark as valid with the cell count so callers
        # can see the situation.
        return BoundaryClassification(
            track_id=track_id,
            sensory_fraction=0.5,
            active_fraction=0.5,
            n_boundary_cells_total=0,
            valid=True,
            reason="ok",
        )

    sensory_fraction = sensory_count / total
    active_fraction = active_count / total

    return BoundaryClassification(
        track_id=track_id,
        sensory_fraction=float(sensory_fraction),
        active_fraction=float(active_fraction),
        n_boundary_cells_total=total,
        valid=True,
        reason="ok",
    )
