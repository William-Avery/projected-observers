"""Resilience observer-likeness score.

Functional question: when we damage the candidate by perturbing its interior
4D fibers, does the projected 2D structure recover?

We run two paired forward rollouts of the 4D CA from a single 4D snapshot:

* baseline: unperturbed evolution.
* perturbed: a flip-intervention applied to the candidate's interior 4D
  fibers, then evolved for the same number of steps.

We compare the *final* projected frames inside a region of interest equal
to the union of ``interior_mask`` and a small dilation of it (so a recovering
structure is allowed to drift slightly).  Four components combine into a
single resilience score:

* survival          -- does any active cell remain inside the ROI?
* area_recovery     -- how much of the baseline area is recovered?
* centroid_continuity -- exp(-||centroid_pert - centroid_orig|| / scale)
* shape_similarity  -- IoU of perturbed and baseline final masks within ROI.

The final score is a (by default equal-weighted) average of the four
components and lies in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from observer_worlds.metrics.causality_score import (
    apply_flip_intervention,
    rollout,
)
from observer_worlds.worlds import BSRule


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResilienceResult:
    track_id: int | None
    n_steps: int
    flip_fraction: float
    survival: float
    area_recovery: float
    centroid_continuity: float
    shape_similarity: float
    resilience_score: float
    valid: bool
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_WEIGHTS: dict[str, float] = {
    "survival": 0.25,
    "area_recovery": 0.25,
    "centroid_continuity": 0.25,
    "shape_similarity": 0.25,
}


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Centroid (cx, cy) of an active 2D bool mask, or None if empty."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    cy = float(coords[:, 0].mean())
    cx = float(coords[:, 1].mean())
    return (cy, cx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_resilience_score(
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask_2d: np.ndarray,
    *,
    n_steps: int = 20,
    flip_fraction: float = 0.5,
    weights: dict[str, float] | None = None,
    projection_method: str = "mean_threshold",
    projection_theta: float = 0.5,
    backend: str = "numpy",
    seed: int = 0,
    track_id: int | None = None,
) -> ResilienceResult:
    """Perturb the interior 4D fibers, run forward, compare to unperturbed.

    See module docstring for the components and their combination.
    """
    if snapshot_4d.ndim != 4:
        raise ValueError(
            f"snapshot_4d must be 4D, got shape {snapshot_4d.shape}"
        )

    interior_mask_2d = np.asarray(interior_mask_2d, dtype=bool)
    if int(interior_mask_2d.sum()) == 0:
        return ResilienceResult(
            track_id=track_id,
            n_steps=n_steps,
            flip_fraction=flip_fraction,
            survival=float("nan"),
            area_recovery=float("nan"),
            centroid_continuity=float("nan"),
            shape_similarity=float("nan"),
            resilience_score=float("nan"),
            valid=False,
            reason="empty_mask",
        )

    if weights is None:
        weights = _DEFAULT_WEIGHTS

    rollout_kwargs = dict(
        backend=backend,
        projection_method=projection_method,
        projection_theta=projection_theta,
    )

    rng = np.random.default_rng(seed)

    # Baseline rollout.  We only need the final projected frame for the
    # resilience computation, but ``rollout`` returns the whole window.
    if n_steps <= 0:
        # Degenerate: no evolution requested.  Use the projected initial
        # state by stepping zero times.  We just bypass the rollout call.
        from observer_worlds.worlds import project as _project

        y_orig_final = _project(
            snapshot_4d,
            method=projection_method,
            theta=projection_theta,
        ).astype(np.uint8)
        perturbed_state = apply_flip_intervention(
            snapshot_4d, interior_mask_2d, flip_fraction, rng
        )
        y_pert_final = _project(
            perturbed_state,
            method=projection_method,
            theta=projection_theta,
        ).astype(np.uint8)
    else:
        baseline = rollout(snapshot_4d, rule, n_steps, **rollout_kwargs)
        y_orig_final = baseline[-1].astype(np.uint8)

        perturbed_state = apply_flip_intervention(
            snapshot_4d, interior_mask_2d, flip_fraction, rng
        )
        perturbed = rollout(perturbed_state, rule, n_steps, **rollout_kwargs)
        y_pert_final = perturbed[-1].astype(np.uint8)

    # ROI: interior union with a small dilation so the structure may drift.
    roi = binary_dilation(interior_mask_2d, iterations=1)

    orig_in_roi = (y_orig_final.astype(bool)) & roi
    pert_in_roi = (y_pert_final.astype(bool)) & roi

    orig_area = int(orig_in_roi.sum())
    pert_area = int(pert_in_roi.sum())

    # Component 1: survival.
    survival = 1.0 if pert_area > 0 else 0.0

    # Component 2: area_recovery.
    area_recovery = float(min(1.0, pert_area / max(1, orig_area)))

    # Component 3: centroid_continuity.
    c_orig = _centroid(orig_in_roi)
    c_pert = _centroid(pert_in_roi)
    if c_orig is None or c_pert is None:
        centroid_continuity = 0.0
    else:
        d = float(np.hypot(c_orig[0] - c_pert[0], c_orig[1] - c_pert[1]))
        scale = float(np.sqrt(max(1, int(roi.sum()))))
        centroid_continuity = float(np.exp(-d / scale))

    # Component 4: shape_similarity (IoU within ROI).
    inter = int(np.logical_and(orig_in_roi, pert_in_roi).sum())
    union = int(np.logical_or(orig_in_roi, pert_in_roi).sum())
    if union == 0:
        # Both empty within ROI -- treat as identical (no structure to
        # disagree about).  This pairs with area_recovery, which is also
        # 1.0 in this case (0/max(1,0) -> 0/1 = 0; but min(1, 0)=0).
        # Empty/empty -> use IoU=1.0 since they're identical.
        shape_similarity = 1.0
    else:
        shape_similarity = inter / union

    # Combine.  Use explicit weight names so callers can override partially.
    components = {
        "survival": survival,
        "area_recovery": area_recovery,
        "centroid_continuity": centroid_continuity,
        "shape_similarity": shape_similarity,
    }
    total_weight = float(sum(weights.get(k, 0.0) for k in components))
    if total_weight <= 0.0:
        score = 0.0
    else:
        score = float(
            sum(weights.get(k, 0.0) * v for k, v in components.items())
            / total_weight
        )

    return ResilienceResult(
        track_id=track_id,
        n_steps=n_steps,
        flip_fraction=flip_fraction,
        survival=float(survival),
        area_recovery=float(area_recovery),
        centroid_continuity=float(centroid_continuity),
        shape_similarity=float(shape_similarity),
        resilience_score=score,
        valid=True,
        reason="ok",
    )
