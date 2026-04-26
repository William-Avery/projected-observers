"""M8C — large-grid mechanism validation.

M8B suggested M7's hidden support is interior-dominant whole-body
coupling among thick candidates, but at grid 48×48 the antipode
translation used as the far control was only ~24 cells from the
candidate centroid. That is short relative to candidate radius (most
thick candidates have radius ~3 cells, so far_distance / radius ≈ 8 —
not bad, but not great). And 36% of M7 thick candidates classified
as `global_chaotic` at that scale. M8C addresses both:

1. Bigger grids (default 96×96×8×8) so the antipode is meaningfully
   distant from the candidate.
2. **Adaptive far-mask selection** that searches multiple candidate
   translations and picks one with (a) distance ≥ max(32,
   5×candidate_radius), (b) no overlap with the candidate's
   environment shell, and (c) similar projected and hidden activity
   if possible. Candidates with no valid far mask are flagged
   `far_control_invalid` and excluded from locality / global-chaos
   classification.

Reuses M8B's region-aware response measurement and v2 classifier;
this module only adds far-control selection and the measurement loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

from observer_worlds.detection.morphology import (
    MorphologyResult,
    classify_morphology,
    shell_masks_strict,
)
from observer_worlds.experiments._m8_mechanism import (
    _l1_full,
    _rollout_proj,
)
from observer_worlds.experiments._m8b_spatial import (
    M8BCandidateResult,
    classify_mechanism_v2,
    measure_all_regions,
    measure_emergence_and_pathway,
    measure_region_effect,
)
from observer_worlds.metrics.causality_score import (
    apply_hidden_shuffle_intervention,
)
from observer_worlds.worlds import BSRule


# ---------------------------------------------------------------------------
# Far-control geometry
# ---------------------------------------------------------------------------


@dataclass
class FarControlInfo:
    """Per-candidate far-control diagnostics."""

    candidate_radius: float
    candidate_diameter: float
    far_control_translation: tuple[int, int] | None
    far_control_distance: float
    far_control_distance_over_radius: float
    far_control_valid: bool
    far_control_min_distance_required: float
    far_control_projected_activity_diff: float
    far_control_hidden_activity_diff: float
    rejection_reason: str = ""


def _candidate_extent(mask: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """Return (radius, diameter, centroid_yx).

    Radius = max distance from centroid to any candidate cell.
    Diameter = max pairwise distance between candidate cells (or 2*radius
    as a cheaper proxy when the mask is small)."""
    rows, cols = np.where(mask)
    if rows.size == 0: return 0.0, 0.0, (0.0, 0.0)
    cy = float(rows.mean()); cx = float(cols.mean())
    dists = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    radius = float(dists.max())
    if rows.size <= 200:
        # Direct pairwise.
        pts = np.stack([rows, cols], axis=1)
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        diameter = float(d.max())
    else:
        diameter = 2.0 * radius
    return radius, diameter, (cy, cx)


def select_far_mask(
    candidate_mask: np.ndarray,
    snapshot_4d: np.ndarray | None = None,
    *,
    min_distance: float | None = None,
    min_distance_floor: int = 32,
    min_distance_radius_mult: float = 5.0,
    n_translation_candidates: int = 16,
    rng_seed: int = 0,
) -> tuple[np.ndarray, FarControlInfo]:
    """Search a small grid of translations for a far-mask that

    1. is the candidate translated by some (dx, dy);
    2. has periodic-distance from the candidate centroid
       ≥ ``max(min_distance_floor, min_distance_radius_mult × radius)``
       (or the user-specified ``min_distance``);
    3. has zero overlap with the candidate's environment shell.

    Among valid candidates, prefer the translation that minimizes the
    sum of |projected_activity_diff| + |hidden_activity_diff| (so the
    far-control region looks like the candidate region in raw activity).

    Returns (far_mask, FarControlInfo). If no valid translation is
    found, returns a zero mask + ``far_control_valid=False``.
    """
    Nx, Ny = candidate_mask.shape
    radius, diameter, (cy, cx) = _candidate_extent(candidate_mask)
    if min_distance is None:
        min_distance = max(min_distance_floor,
                           min_distance_radius_mult * radius)

    # Avoid env-shell overlap: build a generous environment shell.
    env = ndi.binary_dilation(candidate_mask, iterations=3)

    # Reference activity for matching.
    if snapshot_4d is not None:
        proj = (snapshot_4d.mean(axis=(2, 3)) > 0.5).astype(np.uint8)
        cand_proj_act = float(proj[candidate_mask].mean()) if candidate_mask.any() else 0.0
        cand_hidden_act = float(snapshot_4d[candidate_mask].mean()) \
            if candidate_mask.any() else 0.0
    else:
        proj = None
        cand_proj_act = 0.0
        cand_hidden_act = 0.0

    rng = np.random.default_rng(rng_seed)
    # Enumerate translations on a coarse grid first (quadrants), then
    # add randomized offsets for diversity.
    candidates = []
    for dy in (-Ny // 2, Ny // 4, -Ny // 4, Ny // 2):
        for dx in (-Nx // 2, Nx // 4, -Nx // 4, Nx // 2):
            candidates.append((int(dy), int(dx)))
    for _ in range(max(0, n_translation_candidates - len(candidates))):
        dy = int(rng.integers(-Ny // 2, Ny // 2))
        dx = int(rng.integers(-Nx // 2, Nx // 2))
        candidates.append((dy, dx))

    best = None
    best_score = float("inf")
    for dy, dx in candidates:
        translated = np.roll(candidate_mask, shift=(dy, dx), axis=(0, 1))
        # Distance between centroids on the periodic grid.
        rows, cols = np.where(translated)
        if rows.size == 0: continue
        ty = float(rows.mean()); tx = float(cols.mean())
        # Min over periodic wraparound.
        ddy = min(abs(ty - cy), Nx - abs(ty - cy))
        ddx = min(abs(tx - cx), Ny - abs(tx - cx))
        dist = float(np.sqrt(ddy ** 2 + ddx ** 2))
        if dist < min_distance: continue
        if (translated & env).any(): continue
        # Activity match.
        if proj is not None:
            tr_proj_act = float(proj[translated].mean()) if translated.any() else 0.0
            tr_hidden_act = float(snapshot_4d[translated].mean()) \
                if translated.any() else 0.0
            score = (abs(tr_proj_act - cand_proj_act)
                     + abs(tr_hidden_act - cand_hidden_act))
        else:
            tr_proj_act = 0.0; tr_hidden_act = 0.0
            score = 0.0
        if score < best_score:
            best = (translated, dy, dx, dist, tr_proj_act, tr_hidden_act)
            best_score = score

    if best is None:
        info = FarControlInfo(
            candidate_radius=radius, candidate_diameter=diameter,
            far_control_translation=None, far_control_distance=0.0,
            far_control_distance_over_radius=0.0,
            far_control_valid=False,
            far_control_min_distance_required=min_distance,
            far_control_projected_activity_diff=0.0,
            far_control_hidden_activity_diff=0.0,
            rejection_reason="no translation passed distance + env-overlap test",
        )
        return np.zeros_like(candidate_mask), info

    far_mask, dy, dx, dist, tr_proj_act, tr_hidden_act = best
    info = FarControlInfo(
        candidate_radius=radius, candidate_diameter=diameter,
        far_control_translation=(int(dy), int(dx)),
        far_control_distance=float(dist),
        far_control_distance_over_radius=float(dist / max(radius, 1.0)),
        far_control_valid=True,
        far_control_min_distance_required=min_distance,
        far_control_projected_activity_diff=float(abs(tr_proj_act - cand_proj_act)),
        far_control_hidden_activity_diff=float(abs(tr_hidden_act - cand_hidden_act)),
    )
    return far_mask.astype(bool), info


# ---------------------------------------------------------------------------
# M8C result dataclass + per-candidate measurement
# ---------------------------------------------------------------------------


@dataclass
class M8CCandidateResult:
    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    snapshot_t: int
    candidate_area: int
    candidate_lifetime: int
    observer_score: float | None
    near_threshold_fraction: float

    morphology: MorphologyResult
    far_control: FarControlInfo

    region_effects: dict
    far_effect: object   # RegionEffect
    first_visible_effect_time: int
    hidden_to_visible_conversion_time: int
    fraction_hidden_at_end: float
    fraction_visible_at_end: float

    mechanism_label: str
    mechanism_confidence: float
    supporting_metrics: dict


def measure_candidate_m8c(
    *,
    snapshot_4d: np.ndarray,
    candidate_mask_2d: np.ndarray,
    rule: BSRule,
    rule_id: str,
    rule_source: str,
    seed: int,
    candidate_id: int,
    snapshot_t: int,
    candidate_area: int,
    candidate_lifetime: int,
    observer_score: float | None,
    near_threshold_fraction: float,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    rng_seed: int,
    region_shell_widths: tuple[int, ...] = (1, 2, 3),
    min_far_distance_floor: int = 32,
    min_far_distance_radius_mult: float = 5.0,
) -> M8CCandidateResult:
    morph = classify_morphology(candidate_mask_2d)
    far_mask, far_info = select_far_mask(
        candidate_mask_2d, snapshot_4d=snapshot_4d,
        min_distance_floor=min_far_distance_floor,
        min_distance_radius_mult=min_far_distance_radius_mult,
        rng_seed=rng_seed,
    )

    headline_h = horizons[len(horizons) // 2]

    # Reuse M8B's `measure_all_regions` for interior/boundary/env/whole, then
    # override the far_effect using our M8C-selected far_mask.
    region_effects, _ = measure_all_regions(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizon=headline_h,
        n_replicates=n_replicates, backend=backend, rng_seed=rng_seed,
        region_shell_widths=region_shell_widths,
    )
    if far_info.far_control_valid:
        frames_orig = _rollout_proj(snapshot_4d, rule, headline_h,
                                    backend=backend)
        far_effect = measure_region_effect(
            snapshot_4d=snapshot_4d, rule=rule,
            region_mask_2d=far_mask, candidate_mask_2d=candidate_mask_2d,
            region_name="far_validated", horizon=headline_h,
            n_replicates=n_replicates, backend=backend,
            rng_seed=rng_seed + 99, frames_orig=frames_orig,
        )
    else:
        # Use a zero-effect placeholder so downstream code still has the
        # shape, but flag classifier to skip global_chaotic.
        from observer_worlds.experiments._m8b_spatial import RegionEffect
        far_effect = RegionEffect(
            region_name="far_invalid", n_perturbed_cells_2d=0,
            n_flipped_cells_4d=0,
            region_hidden_effect=0.0, region_local_divergence=0.0,
            region_global_divergence=0.0, region_response_fraction=0.0,
            region_effect_per_cell=0.0, region_effect_per_flipped_cell=0.0,
        )

    first_visible, conv, frac_hid, frac_vis = measure_emergence_and_pathway(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizons=horizons,
        backend=backend, rng_seed=rng_seed + 100,
    )

    # Apply v2 classifier, but if far_control is invalid, prevent it
    # from labeling the candidate global_chaotic (it cannot tell).
    if far_info.far_control_valid:
        label, conf, metrics = classify_mechanism_v2(
            morphology=morph, region_effects=region_effects,
            far_effect=far_effect,
            first_visible_effect_time=first_visible,
            fraction_hidden_at_end=frac_hid,
            fraction_visible_at_end=frac_vis,
            near_threshold_fraction=near_threshold_fraction,
        )
    else:
        # Invent a sentinel "tiny" far_effect for the classifier so it
        # doesn't shortcut to global_chaotic; instead post-tag the
        # result.
        from observer_worlds.experiments._m8b_spatial import RegionEffect
        sentinel_far = RegionEffect(
            region_name="far_synthetic_zero", n_perturbed_cells_2d=0,
            n_flipped_cells_4d=0,
            region_hidden_effect=0.0, region_local_divergence=0.0,
            region_global_divergence=0.0, region_response_fraction=0.0,
            region_effect_per_cell=0.0, region_effect_per_flipped_cell=0.0,
        )
        label, conf, metrics = classify_mechanism_v2(
            morphology=morph, region_effects=region_effects,
            far_effect=sentinel_far,
            first_visible_effect_time=first_visible,
            fraction_hidden_at_end=frac_hid,
            fraction_visible_at_end=frac_vis,
            near_threshold_fraction=near_threshold_fraction,
        )
        # If classifier had returned global_chaotic, that cannot be
        # supported without a valid far control; soft-replace.
        if label == "global_chaotic":
            label = "unclear"
            conf = 0.0
        metrics = {**metrics, "far_control_valid": False}
    metrics = {**metrics,
               "far_control_distance": far_info.far_control_distance,
               "far_control_distance_over_radius":
                   far_info.far_control_distance_over_radius,
               "far_control_valid": far_info.far_control_valid,
               "candidate_radius": far_info.candidate_radius,
               "candidate_diameter": far_info.candidate_diameter}
    return M8CCandidateResult(
        rule_id=rule_id, rule_source=rule_source, seed=seed,
        candidate_id=candidate_id, snapshot_t=snapshot_t,
        candidate_area=candidate_area, candidate_lifetime=candidate_lifetime,
        observer_score=observer_score,
        near_threshold_fraction=near_threshold_fraction,
        morphology=morph, far_control=far_info,
        region_effects=region_effects, far_effect=far_effect,
        first_visible_effect_time=first_visible,
        hidden_to_visible_conversion_time=conv,
        fraction_hidden_at_end=frac_hid, fraction_visible_at_end=frac_vis,
        mechanism_label=label, mechanism_confidence=conf,
        supporting_metrics=metrics,
    )
