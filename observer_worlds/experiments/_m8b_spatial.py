"""M8B — region-aware response maps + per-cell-normalized mediation +
v2 mechanism classifier.

The defining change over M8: every region (interior, boundary,
environment, far, whole-candidate) is probed *independently* and the
effect is normalized by the number of perturbed and number of actually-
flipped cells. M8's classifier missed environment-coupling because the
response map only probed interior ∪ boundary; M8B fixes that.

Classification rules differ for thick vs thin morphologies. For thin
candidates, the boundary-vs-interior decomposition is structurally
impossible — those candidates get `candidate_local_thin` instead of
`boundary_mediated`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from observer_worlds.detection.morphology import (
    MorphologyResult,
    classify_morphology,
    far_mask,
    shell_masks_strict,
)
from observer_worlds.experiments._m8_mechanism import (
    _l1_full,
    _l1_local,
    _rollout_proj,
    _rollout_proj_capturing_4d,
)
from observer_worlds.metrics.causality_score import (
    apply_hidden_shuffle_intervention,
)
from observer_worlds.worlds import BSRule


M8B_MECHANISM_CLASSES: tuple[str, ...] = (
    "boundary_mediated",
    "interior_reservoir",
    "environment_coupled",
    "whole_body_hidden_support",
    "delayed_hidden_channel",
    "global_chaotic",
    "threshold_mediated",
    "candidate_local_thin",
    "environment_coupled_thin",
    "unclear",
)


# ---------------------------------------------------------------------------
# Region effect dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegionEffect:
    region_name: str
    n_perturbed_cells_2d: int       # cells in the 2D region mask
    n_flipped_cells_4d: int         # cells whose 4D value actually changed at t=0 (count over all reps)
    region_hidden_effect: float            # mean L1 over full grid at horizon
    region_local_divergence: float         # L1 within candidate
    region_global_divergence: float        # L1 over full grid (alias for hidden_effect)
    region_response_fraction: float        # share of total response budget
    region_effect_per_cell: float          # hidden_effect / (n_perturbed_cells_2d * Nz * Nw)
    region_effect_per_flipped_cell: float  # hidden_effect / n_flipped_cells_4d


@dataclass
class M8BCandidateResult:
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
    region_effects: dict           # region_name -> RegionEffect
    far_effect: RegionEffect
    first_visible_effect_time: int
    hidden_to_visible_conversion_time: int
    fraction_hidden_at_end: float
    fraction_visible_at_end: float
    mechanism_label: str
    mechanism_confidence: float
    supporting_metrics: dict


# ---------------------------------------------------------------------------
# Region-aware response measurement
# ---------------------------------------------------------------------------


def measure_region_effect(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    region_mask_2d: np.ndarray,
    candidate_mask_2d: np.ndarray,
    region_name: str,
    horizon: int,
    n_replicates: int,
    backend: str,
    rng_seed: int,
    frames_orig: np.ndarray | None = None,
) -> RegionEffect:
    """Apply count-preserving hidden-invisible perturbation to the
    region's hidden fibers and measure response at the horizon.

    Per-cell normalization uses the 2D region size × Nz × Nw; per-
    flipped-cell normalization uses how many 4D cells actually changed
    value at t=0 (averaged across replicates).
    """
    Nx, Ny, Nz, Nw = snapshot_4d.shape
    n_perturbed_2d = int(region_mask_2d.sum())
    if n_perturbed_2d == 0:
        return RegionEffect(
            region_name=region_name, n_perturbed_cells_2d=0,
            n_flipped_cells_4d=0,
            region_hidden_effect=0.0, region_local_divergence=0.0,
            region_global_divergence=0.0, region_response_fraction=0.0,
            region_effect_per_cell=0.0, region_effect_per_flipped_cell=0.0,
        )
    if frames_orig is None:
        frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend=backend)

    rng = np.random.default_rng(rng_seed)
    full_l1: list[float] = []
    local_l1: list[float] = []
    flipped_counts: list[int] = []
    for _ in range(n_replicates):
        sub = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        perturbed = apply_hidden_shuffle_intervention(
            snapshot_4d, region_mask_2d, sub
        )
        # Count 4D cells that actually changed.
        flipped_counts.append(int((perturbed != snapshot_4d).sum()))
        f_int = _rollout_proj(perturbed, rule, horizon, backend=backend)
        full_l1.append(_l1_full(frames_orig[-1], f_int[-1]))
        local_l1.append(_l1_local(frames_orig[-1], f_int[-1], candidate_mask_2d))

    mean_full = float(np.mean(full_l1)) if full_l1 else 0.0
    mean_local = float(np.mean(local_l1)) if local_l1 else 0.0
    mean_flipped = float(np.mean(flipped_counts)) if flipped_counts else 0.0
    n_cells_4d = n_perturbed_2d * Nz * Nw
    per_cell = mean_full / max(n_cells_4d, 1)
    per_flipped = mean_full / max(mean_flipped, 1.0)
    return RegionEffect(
        region_name=region_name, n_perturbed_cells_2d=n_perturbed_2d,
        n_flipped_cells_4d=int(round(mean_flipped)),
        region_hidden_effect=mean_full, region_local_divergence=mean_local,
        region_global_divergence=mean_full,
        region_response_fraction=0.0,  # filled in by caller after summing
        region_effect_per_cell=per_cell,
        region_effect_per_flipped_cell=per_flipped,
    )


def measure_all_regions(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    candidate_mask_2d: np.ndarray,
    horizon: int,
    n_replicates: int,
    backend: str,
    rng_seed: int,
    region_shell_widths: tuple[int, ...] = (1, 2, 3),
) -> tuple[dict, RegionEffect]:
    """Probe interior, boundary, environment, whole-candidate, and far
    regions. Returns (region_effects_dict, far_effect).

    For environment shells the function probes each width in
    `region_shell_widths` and reports the *primary* environment (first
    width) plus widths under the keys ``environment_w<k>``. The "primary"
    width is what the classifier consumes; the alternative widths feed
    `plot_environment_coupling_by_shell_width`.
    """
    shells = shell_masks_strict(candidate_mask_2d, erosion_radius=1,
                                env_dilation=region_shell_widths[0])
    far = far_mask(candidate_mask_2d)
    frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend=backend)

    effects: dict[str, RegionEffect] = {}
    rng_offsets = {
        "interior": 1, "boundary": 2, "environment": 3,
        "whole": 4, "far": 5,
    }
    for region in ("interior", "boundary", "environment", "whole"):
        m = shells[region]
        effects[region] = measure_region_effect(
            snapshot_4d=snapshot_4d, rule=rule,
            region_mask_2d=m, candidate_mask_2d=candidate_mask_2d,
            region_name=region, horizon=horizon,
            n_replicates=n_replicates, backend=backend,
            rng_seed=rng_seed + rng_offsets[region],
            frames_orig=frames_orig,
        )
    far_effect = measure_region_effect(
        snapshot_4d=snapshot_4d, rule=rule,
        region_mask_2d=far, candidate_mask_2d=candidate_mask_2d,
        region_name="far", horizon=horizon,
        n_replicates=n_replicates, backend=backend,
        rng_seed=rng_seed + rng_offsets["far"],
        frames_orig=frames_orig,
    )
    # Alternative environment shell widths for the by-shell-width plot.
    for w in region_shell_widths[1:]:
        env_w_mask = shell_masks_strict(
            candidate_mask_2d, erosion_radius=1, env_dilation=w,
        )["environment"]
        effects[f"environment_w{w}"] = measure_region_effect(
            snapshot_4d=snapshot_4d, rule=rule,
            region_mask_2d=env_w_mask, candidate_mask_2d=candidate_mask_2d,
            region_name=f"environment_w{w}", horizon=horizon,
            n_replicates=n_replicates, backend=backend,
            rng_seed=rng_seed + 6 + w,
            frames_orig=frames_orig,
        )

    # Fill response_fraction across the four primary regions.
    primaries = ("interior", "boundary", "environment", "whole")
    total = sum(effects[r].region_hidden_effect for r in primaries)
    if total > 1e-12:
        for r in primaries:
            effects[r].region_response_fraction = (
                effects[r].region_hidden_effect / total
            )
    return effects, far_effect


# ---------------------------------------------------------------------------
# Pathway tracing for hidden-channel detection
# ---------------------------------------------------------------------------


def measure_emergence_and_pathway(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    candidate_mask_2d: np.ndarray,
    horizons: list[int],
    backend: str,
    rng_seed: int,
    epsilon: float = 1e-3,
) -> tuple[int, int, float, float]:
    """Run a single pathway trace + emergence probe on the candidate's
    interior+boundary fibers.

    Returns (first_visible_effect_time, hidden_to_visible_conversion_time,
    fraction_hidden_at_end, fraction_visible_at_end).
    """
    H_max = max(horizons)
    rng = np.random.default_rng(rng_seed)
    perturbed = apply_hidden_shuffle_intervention(
        snapshot_4d, candidate_mask_2d, rng,
    )
    proj_orig, full_orig = _rollout_proj_capturing_4d(
        snapshot_4d, rule, H_max, backend=backend,
    )
    proj_int, full_int = _rollout_proj_capturing_4d(
        perturbed, rule, H_max, backend=backend,
    )

    first_visible = -1
    for t in range(H_max):
        local = _l1_local(proj_orig[t], proj_int[t], candidate_mask_2d)
        if local > epsilon:
            first_visible = t + 1
            break
    conv = -1
    for t in range(H_max):
        if (proj_orig[t] != proj_int[t]).any():
            conv = t + 1
            break
    total_4d = full_orig[0].size
    total_2d = proj_orig[0].size
    frac_hidden = float((full_orig[-1] != full_int[-1]).sum() / total_4d)
    frac_visible = float((proj_orig[-1] != proj_int[-1]).sum() / total_2d)
    return first_visible, conv, frac_hidden, frac_visible


# ---------------------------------------------------------------------------
# v2 classifier
# ---------------------------------------------------------------------------


def classify_mechanism_v2(
    *,
    morphology: MorphologyResult,
    region_effects: dict,
    far_effect: RegionEffect,
    first_visible_effect_time: int,
    fraction_hidden_at_end: float,
    fraction_visible_at_end: float,
    near_threshold_fraction: float,
) -> tuple[str, float, dict]:
    """Returns (label, confidence, supporting_metrics).

    The contract: for ``thin_candidate`` and ``degenerate``
    morphologies, this function NEVER returns ``boundary_mediated``,
    ``interior_reservoir``, or ``whole_body_hidden_support`` — those
    require the morphology gate to pass.
    """
    cand_e = (region_effects["interior"].region_hidden_effect
              + region_effects["boundary"].region_hidden_effect)
    interior_pc = region_effects["interior"].region_effect_per_cell
    boundary_pc = region_effects["boundary"].region_effect_per_cell
    env_pc = region_effects["environment"].region_effect_per_cell
    whole_pc = region_effects["whole"].region_effect_per_cell
    far_e = far_effect.region_hidden_effect

    metrics = {
        "interior_effect_per_cell": interior_pc,
        "boundary_effect_per_cell": boundary_pc,
        "environment_effect_per_cell": env_pc,
        "whole_effect_per_cell": whole_pc,
        "far_hidden_effect": far_e,
        "candidate_hidden_effect": cand_e,
        "interior_hidden_effect": region_effects["interior"].region_hidden_effect,
        "boundary_hidden_effect": region_effects["boundary"].region_hidden_effect,
        "environment_hidden_effect": region_effects["environment"].region_hidden_effect,
        "whole_hidden_effect": region_effects["whole"].region_hidden_effect,
        "first_visible_effect_time": first_visible_effect_time,
        "fraction_hidden_at_end": fraction_hidden_at_end,
        "fraction_visible_at_end": fraction_visible_at_end,
        "near_threshold_fraction": near_threshold_fraction,
        "morphology_class": morphology.morphology_class,
    }

    # Threshold check applies to all morphology classes.
    if (near_threshold_fraction > 0.5
            and interior_pc <= 0.5 * boundary_pc + 1e-9
            and morphology.morphology_class != "degenerate"):
        return "threshold_mediated", float(min(1.0, near_threshold_fraction)), metrics

    # Global chaos check applies to all classes.
    if cand_e > 0:
        if far_e >= 0.7 * cand_e:
            return "global_chaotic", float(min(1.0, far_e / max(cand_e, 1e-9))), metrics

    # Thin / degenerate: cannot distinguish boundary from interior.
    if morphology.morphology_class in ("thin_candidate", "degenerate"):
        # If environment per-cell effect dominates, surface it.
        if env_pc > 1.5 * max(interior_pc, boundary_pc, 1e-9) and env_pc > 0:
            return "environment_coupled_thin", 0.6, metrics
        # If candidate effect is positive, label thin-local.
        if cand_e > far_e:
            return "candidate_local_thin", 0.6, metrics
        return "unclear", 0.0, metrics

    # Thick candidates: full classification.
    # Environment-dominant (per-cell, robust to size differences).
    if env_pc > 1.5 * max(interior_pc, boundary_pc) and env_pc > 0:
        return "environment_coupled", float(min(
            1.0, env_pc / max(max(interior_pc, boundary_pc), 1e-9) - 1.0,
        )), metrics
    # Boundary-dominant.
    if boundary_pc > 1.5 * interior_pc and boundary_pc > 0:
        return "boundary_mediated", float(min(
            1.0, boundary_pc / max(interior_pc, 1e-9) - 1.0,
        )), metrics
    # Interior-dominant.
    if interior_pc > 1.5 * boundary_pc and interior_pc > 0:
        return "interior_reservoir", float(min(
            1.0, interior_pc / max(boundary_pc, 1e-9) - 1.0,
        )), metrics
    # Boundary and interior similar but candidate effect substantial.
    if cand_e > 1.5 * far_e and cand_e > 0:
        return "whole_body_hidden_support", 0.6, metrics

    # Delayed hidden channel.
    if (first_visible_effect_time >= 5
            and fraction_hidden_at_end > 0.5 * max(fraction_visible_at_end, 1e-9)):
        return "delayed_hidden_channel", 0.6, metrics

    return "unclear", 0.0, metrics


# ---------------------------------------------------------------------------
# Per-candidate measurement (called by the M8B CLI for each candidate
# loaded from the large-candidate search CSV)
# ---------------------------------------------------------------------------


def measure_candidate_m8b(
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
) -> M8BCandidateResult:
    morph = classify_morphology(candidate_mask_2d)
    headline_h = horizons[len(horizons) // 2]
    region_effects, far_effect = measure_all_regions(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizon=headline_h,
        n_replicates=n_replicates, backend=backend, rng_seed=rng_seed,
        region_shell_widths=region_shell_widths,
    )
    first_visible, conv, frac_hid, frac_vis = measure_emergence_and_pathway(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizons=horizons,
        backend=backend, rng_seed=rng_seed + 100,
    )
    label, confidence, metrics = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far_effect,
        first_visible_effect_time=first_visible,
        fraction_hidden_at_end=frac_hid,
        fraction_visible_at_end=frac_vis,
        near_threshold_fraction=near_threshold_fraction,
    )
    return M8BCandidateResult(
        rule_id=rule_id, rule_source=rule_source, seed=seed,
        candidate_id=candidate_id, snapshot_t=snapshot_t,
        candidate_area=candidate_area, candidate_lifetime=candidate_lifetime,
        observer_score=observer_score,
        near_threshold_fraction=near_threshold_fraction,
        morphology=morph, region_effects=region_effects, far_effect=far_effect,
        first_visible_effect_time=first_visible,
        hidden_to_visible_conversion_time=conv,
        fraction_hidden_at_end=frac_hid, fraction_visible_at_end=frac_vis,
        mechanism_label=label, mechanism_confidence=confidence,
        supporting_metrics=metrics,
    )
