"""M8D — global-chaotic decomposition.

M8B and M8C both showed a persistent ~1/3 minority of M7 thick
candidates that classify as `global_chaotic` (far-region effect ≥ 70%
of candidate-body effect). Strengthening far-control geometry from
M8B → M8C dropped that rate from 36% to 33%, a small change. This
module asks **what those candidates actually are**:

  A. **true global instability** — effect is flat across distance, the
     world is just dynamically chaotic.
  B. **broad hidden coupling** — effect decays slightly with distance
     but stays well above a system-level background.
  C. **background-sensitive world** — far effect is no larger than
     a random-perturbation background; candidate body still wins.
  D. **far-control artifact** — only antipodal far is hot, near-far
     and mid-far are clean (still a geometry issue).
  E. **threshold/volatility-mediated** — same features that drive
     M6C's threshold class also drive global_chaotic.
  F. **unresolved** — none of the above fits cleanly.

Implements five analyses per global_chaotic candidate:
    Part A: multi-distance far probes (body, env, 2r, 5r, 10r, antipode, random×5)
    Part B: random-baseline background sensitivity at the world level
    Part C: feature audit (M6C hidden features) vs interior / whole-body controls
    Part D: stabilization variants (lower count, shorter horizon, threshold filter, local-window rollout)
    Part E: 6-subclass relabeling
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

from observer_worlds.analysis.hidden_features import candidate_hidden_features
from observer_worlds.detection.morphology import classify_morphology
from observer_worlds.experiments._m8_mechanism import (
    _l1_full,
    _l1_local,
    _rollout_proj,
)
from observer_worlds.experiments._m8b_spatial import (
    M8B_MECHANISM_CLASSES,
    RegionEffect,
    classify_mechanism_v2,
    measure_all_regions,
    measure_emergence_and_pathway,
    measure_region_effect,
)
from observer_worlds.experiments._m8c_validation import (
    FarControlInfo,
    _candidate_extent,
    select_far_mask,
)
from observer_worlds.metrics.causality_score import (
    apply_hidden_shuffle_intervention,
)
from observer_worlds.worlds import BSRule


M8D_GLOBAL_SUBCLASSES: tuple[str, ...] = (
    "global_instability",
    "broad_hidden_coupling",
    "background_sensitive_world",
    "far_control_artifact",
    "threshold_volatility_artifact",
    "unresolved_global",
)


# All M8B labels remain valid; M8D only refines `global_chaotic`.
M8D_MECHANISM_CLASSES: tuple[str, ...] = (
    "boundary_mediated",
    "interior_reservoir",
    "environment_coupled",
    "whole_body_hidden_support",
    "delayed_hidden_channel",
    "threshold_mediated",
    "candidate_local_thin",
    "environment_coupled_thin",
    "unclear",
) + M8D_GLOBAL_SUBCLASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_disc_mask_at(
    grid_shape: tuple[int, int], cy: float, cx: float, radius: float,
) -> np.ndarray:
    Nx, Ny = grid_shape
    yy, xx = np.ogrid[0:Nx, 0:Ny]
    dy = yy - cy; dx = xx - cx
    # Use periodic distance.
    dy = np.minimum(np.abs(dy), Nx - np.abs(dy))
    dx = np.minimum(np.abs(dx), Ny - np.abs(dx))
    return (dy * dy + dx * dx <= radius * radius)


def _periodic_dist(pa: tuple[float, float], pb: tuple[float, float],
                   shape: tuple[int, int]) -> float:
    Nx, Ny = shape
    dy = abs(pa[0] - pb[0]); dx = abs(pa[1] - pb[1])
    dy = min(dy, Nx - dy); dx = min(dx, Ny - dx)
    return float(np.sqrt(dy * dy + dx * dx))


def _translate_mask_at_distance(
    mask: np.ndarray, *, target_distance: float, rng_seed: int = 0,
    n_attempts: int = 32,
) -> tuple[np.ndarray | None, float]:
    """Find a translation of `mask` whose centroid sits roughly
    `target_distance` cells from the original centroid (periodic
    distance) with no overlap with the original mask. Returns the
    closest match or (None, 0) on failure."""
    Nx, Ny = mask.shape
    rows, cols = np.where(mask)
    if rows.size == 0: return None, 0.0
    cy = float(rows.mean()); cx = float(cols.mean())
    rng = np.random.default_rng(rng_seed)
    best = None; best_diff = float("inf")
    for _ in range(n_attempts):
        dy = int(rng.integers(-Nx // 2, Nx // 2))
        dx = int(rng.integers(-Ny // 2, Ny // 2))
        translated = np.roll(mask, shift=(dy, dx), axis=(0, 1))
        if (translated & mask).any(): continue
        tr = np.where(translated)
        if tr[0].size == 0: continue
        ty = float(tr[0].mean()); tx = float(tr[1].mean())
        d = _periodic_dist((cy, cx), (ty, tx), (Nx, Ny))
        diff = abs(d - target_distance)
        if diff < best_diff:
            best_diff = diff; best = (translated, d)
    if best is None: return None, 0.0
    return best


# ---------------------------------------------------------------------------
# Part A: multi-distance probes
# ---------------------------------------------------------------------------


@dataclass
class DistanceEffect:
    name: str
    distance: float
    distance_over_radius: float
    n_perturbed_2d: int
    raw_effect: float
    effect_per_cell: float


def measure_multi_distance_effects(
    *, snapshot_4d, rule, candidate_mask_2d, horizon, n_replicates,
    backend, rng_seed,
) -> list[DistanceEffect]:
    """Probe candidate body, env shell, and translated copies at
    multiples of candidate_radius. Always include antipode + random×5."""
    Nx, Ny = candidate_mask_2d.shape
    radius, _, _ = _candidate_extent(candidate_mask_2d)
    radius = max(radius, 1.0)
    frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend=backend)
    rng = np.random.default_rng(rng_seed)

    def _probe(name, mask, distance):
        if mask is None or not mask.any():
            return DistanceEffect(name=name, distance=distance,
                                  distance_over_radius=distance / radius,
                                  n_perturbed_2d=0, raw_effect=0.0,
                                  effect_per_cell=0.0)
        eff = measure_region_effect(
            snapshot_4d=snapshot_4d, rule=rule, region_mask_2d=mask,
            candidate_mask_2d=candidate_mask_2d, region_name=name,
            horizon=horizon, n_replicates=n_replicates, backend=backend,
            rng_seed=int(rng.integers(0, 2**63 - 1)),
            frames_orig=frames_orig,
        )
        return DistanceEffect(
            name=name, distance=distance,
            distance_over_radius=distance / radius,
            n_perturbed_2d=eff.n_perturbed_cells_2d,
            raw_effect=eff.region_hidden_effect,
            effect_per_cell=eff.region_effect_per_cell,
        )

    out = []
    # Body.
    out.append(_probe("body", candidate_mask_2d, 0.0))
    # Environment shell (dilation 1, exclude candidate).
    env = ndi.binary_dilation(candidate_mask_2d, iterations=2) & ~candidate_mask_2d
    out.append(_probe("env_shell", env, 1.5))
    # Translated copies at 2×, 5×, 10× radius.
    for factor in (2, 5, 10):
        target = factor * radius
        if target >= max(Nx, Ny) // 2: continue
        m, d = _translate_mask_at_distance(
            candidate_mask_2d, target_distance=target,
            rng_seed=int(rng.integers(0, 2**63 - 1)),
        )
        if m is not None:
            out.append(_probe(f"far_{factor}r", m, d))
    # Antipode.
    antipode = np.roll(candidate_mask_2d, shift=(Nx // 2, Ny // 2), axis=(0, 1))
    cy, cx = _candidate_extent(candidate_mask_2d)[2]
    rows, cols = np.where(antipode)
    if rows.size > 0:
        ay = float(rows.mean()); ax = float(cols.mean())
        d_ant = _periodic_dist((cy, cx), (ay, ax), (Nx, Ny))
    else:
        d_ant = 0.0
    out.append(_probe("antipode", antipode, d_ant))
    # 5 random translations.
    for k in range(5):
        dy = int(rng.integers(-Nx // 2, Nx // 2))
        dx = int(rng.integers(-Ny // 2, Ny // 2))
        rand_m = np.roll(candidate_mask_2d, shift=(dy, dx), axis=(0, 1))
        rrows, rcols = np.where(rand_m)
        if rrows.size == 0: continue
        ry = float(rrows.mean()); rx = float(rcols.mean())
        d_r = _periodic_dist((cy, cx), (ry, rx), (Nx, Ny))
        out.append(_probe(f"random_{k}", rand_m, d_r))
    return out


def fit_decay_curve(effects: list[DistanceEffect]) -> dict:
    """Linear fit of effect vs distance over the non-body probes."""
    pts = [(e.distance, e.effect_per_cell) for e in effects if e.name != "body"
           and e.n_perturbed_2d > 0]
    if len(pts) < 3:
        return {"slope": 0.0, "intercept": 0.0, "floor": 0.0,
                "n_points": len(pts)}
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if xs.std() < 1e-9:
        return {"slope": 0.0, "intercept": float(ys.mean()),
                "floor": float(ys.min()), "n_points": len(pts)}
    slope, intercept = np.polyfit(xs, ys, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "floor": float(ys.min()), "n_points": len(pts)}


# ---------------------------------------------------------------------------
# Part B: background sensitivity baseline
# ---------------------------------------------------------------------------


def measure_background_sensitivity(
    *, snapshot_4d, rule, candidate_mask_2d, horizon, n_samples, sample_size,
    backend, rng_seed,
) -> dict:
    """Sample `n_samples` random non-candidate hidden perturbations, each
    of size ~`sample_size` cells in the 2D plane. Return the distribution
    of the resulting full-grid L1 effects."""
    Nx, Ny = candidate_mask_2d.shape
    rng = np.random.default_rng(rng_seed)
    frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend=backend)
    excluded = ndi.binary_dilation(candidate_mask_2d, iterations=2)
    available = ~excluded

    samples = []
    for _ in range(n_samples):
        # Sample sample_size random (x,y) cells from available.
        candidates = np.argwhere(available)
        if candidates.shape[0] == 0: break
        idxs = rng.choice(candidates.shape[0],
                         size=min(sample_size, candidates.shape[0]),
                         replace=False)
        m = np.zeros((Nx, Ny), dtype=bool)
        for i in idxs:
            m[candidates[i, 0], candidates[i, 1]] = True
        eff = measure_region_effect(
            snapshot_4d=snapshot_4d, rule=rule, region_mask_2d=m,
            candidate_mask_2d=candidate_mask_2d, region_name="background",
            horizon=horizon, n_replicates=1, backend=backend,
            rng_seed=int(rng.integers(0, 2**63 - 1)),
            frames_orig=frames_orig,
        )
        samples.append(eff.region_hidden_effect)
    if not samples:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "n_samples": 0,
                "samples": []}
    arr = np.array(samples)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "n_samples": int(arr.size),
        "samples": [float(x) for x in arr],
    }


# ---------------------------------------------------------------------------
# Part C: feature audit (uses existing M6C features)
# ---------------------------------------------------------------------------


def audit_features(snapshot_4d, candidate_mask) -> dict:
    f = candidate_hidden_features(snapshot_4d, candidate_mask)
    keep = (
        "near_threshold_fraction", "mean_threshold_margin",
        "hidden_temporal_persistence", "hidden_volatility",
        "mean_hidden_entropy", "hidden_spatial_autocorrelation",
        "mean_active_fraction", "hidden_heterogeneity",
    )
    return {k: float(f.get(k, 0.0)) for k in keep}


# ---------------------------------------------------------------------------
# Part D: stabilization variants
# ---------------------------------------------------------------------------


def _local_window_rollout(
    snapshot_4d, rule, horizon, *, candidate_mask, window_dilation, backend,
    perturbed_state,
):
    """Rollout where everything outside the candidate's local window is
    frozen to the original state at every step.

    Approximation: we run a normal forward rollout but at each step
    overwrite cells outside the local window with the corresponding
    cells from the unperturbed rollout.
    """
    from observer_worlds.worlds import CA4D
    Nx, Ny, Nz, Nw = snapshot_4d.shape
    window_2d = ndi.binary_dilation(candidate_mask, iterations=window_dilation)
    window_4d = window_2d[..., None, None] & np.ones(
        (Nx, Ny, Nz, Nw), dtype=bool,
    )

    # Reference rollout (no perturbation).
    ca_ref = CA4D(shape=snapshot_4d.shape, rule=rule, backend=backend)
    ca_ref.state = snapshot_4d.copy()

    # Local-window rollout (perturbed inside, frozen outside).
    ca_loc = CA4D(shape=snapshot_4d.shape, rule=rule, backend=backend)
    ca_loc.state = perturbed_state.copy()

    proj_ref = np.empty((horizon, Nx, Ny), dtype=np.uint8)
    proj_loc = np.empty((horizon, Nx, Ny), dtype=np.uint8)
    for t in range(horizon):
        ca_ref.step(); ca_loc.step()
        # Replace outside-window cells with the reference state.
        ca_loc.state[~window_4d] = ca_ref.state[~window_4d]
        proj_ref[t] = (ca_ref.state.mean(axis=(2, 3)) > 0.5).astype(np.uint8)
        proj_loc[t] = (ca_loc.state.mean(axis=(2, 3)) > 0.5).astype(np.uint8)
    return proj_ref, proj_loc


def stabilization_variants(
    *, snapshot_4d, rule, candidate_mask_2d, horizon, n_replicates,
    backend, rng_seed, window_dilation: int = 5,
) -> dict:
    """Run 5 stabilization variants and report the resulting
    per-cell candidate effect vs far effect (using a quick antipode far
    mask). The variant counts how each variant affects the
    candidate-vs-far separation."""
    rng = np.random.default_rng(rng_seed)
    Nx, Ny = candidate_mask_2d.shape

    far_antipode = np.roll(candidate_mask_2d, shift=(Nx // 2, Ny // 2),
                           axis=(0, 1))
    out: dict[str, dict] = {}

    # 1. Baseline (full perturbation, full horizon).
    eff_baseline_body = measure_region_effect(
        snapshot_4d=snapshot_4d, rule=rule,
        region_mask_2d=candidate_mask_2d,
        candidate_mask_2d=candidate_mask_2d, region_name="baseline_body",
        horizon=horizon, n_replicates=n_replicates, backend=backend,
        rng_seed=int(rng.integers(0, 2**63 - 1)),
    )
    eff_baseline_far = measure_region_effect(
        snapshot_4d=snapshot_4d, rule=rule,
        region_mask_2d=far_antipode,
        candidate_mask_2d=candidate_mask_2d, region_name="baseline_far",
        horizon=horizon, n_replicates=n_replicates, backend=backend,
        rng_seed=int(rng.integers(0, 2**63 - 1)),
    )
    body_b = eff_baseline_body.region_hidden_effect
    far_b = eff_baseline_far.region_hidden_effect
    out["baseline"] = {
        "body": body_b, "far": far_b,
        "body_minus_far": body_b - far_b,
        "global_chaotic_label_would_fire": far_b >= 0.7 * max(body_b, 1e-9),
    }

    # 2. Shorter horizon (half).
    short_h = max(1, horizon // 2)
    eff_short_body = measure_region_effect(
        snapshot_4d=snapshot_4d, rule=rule,
        region_mask_2d=candidate_mask_2d,
        candidate_mask_2d=candidate_mask_2d, region_name="short_body",
        horizon=short_h, n_replicates=n_replicates, backend=backend,
        rng_seed=int(rng.integers(0, 2**63 - 1)),
    )
    eff_short_far = measure_region_effect(
        snapshot_4d=snapshot_4d, rule=rule, region_mask_2d=far_antipode,
        candidate_mask_2d=candidate_mask_2d, region_name="short_far",
        horizon=short_h, n_replicates=n_replicates, backend=backend,
        rng_seed=int(rng.integers(0, 2**63 - 1)),
    )
    out["short_horizon"] = {
        "body": eff_short_body.region_hidden_effect,
        "far": eff_short_far.region_hidden_effect,
        "global_chaotic_label_would_fire":
            eff_short_far.region_hidden_effect
            >= 0.7 * max(eff_short_body.region_hidden_effect, 1e-9),
    }

    # 3. Threshold-filtered (skip if near-threshold).
    feats = candidate_hidden_features(snapshot_4d, candidate_mask_2d)
    near_th = float(feats.get("near_threshold_fraction", 0.0))
    out["threshold_filtered"] = {
        "near_threshold_fraction": near_th,
        "filter_eligible": near_th < 0.10,
        "body": body_b if near_th < 0.10 else 0.0,
        "far": far_b if near_th < 0.10 else 0.0,
    }

    # 4. Local-window rollout.
    try:
        sub = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        perturbed = apply_hidden_shuffle_intervention(
            snapshot_4d, candidate_mask_2d, sub,
        )
        proj_ref, proj_loc = _local_window_rollout(
            snapshot_4d, rule, horizon,
            candidate_mask=candidate_mask_2d, window_dilation=window_dilation,
            backend=backend, perturbed_state=perturbed,
        )
        body_lw = _l1_local(proj_ref[-1], proj_loc[-1], candidate_mask_2d)
        # Far effect under local window: by construction, far cells are
        # frozen to the reference state; far effect should drop sharply.
        far_lw = _l1_full(proj_ref[-1], proj_loc[-1])
        out["local_window"] = {
            "body": float(body_lw), "far": float(far_lw),
            "global_chaotic_label_would_fire":
                far_lw >= 0.7 * max(body_lw, 1e-9),
        }
    except Exception as e:
        out["local_window"] = {
            "body": 0.0, "far": 0.0, "error": str(e),
            "global_chaotic_label_would_fire": False,
        }

    # 5. Activity-normalized: scale by candidate area.
    area = max(int(candidate_mask_2d.sum()), 1)
    out["activity_normalized"] = {
        "body_per_cell": body_b / area,
        "far_per_cell": far_b / area,
    }

    return out


# ---------------------------------------------------------------------------
# Part E: 6-subclass relabeling
# ---------------------------------------------------------------------------


def relabel_global_chaotic(
    *, distance_effects: list[DistanceEffect], background: dict,
    feature_audit: dict, stabilization: dict, body_effect: float,
    far_effect: float,
) -> tuple[str, float, dict]:
    """Apply the 6-subclass classifier, returning (label, confidence,
    metrics)."""
    metrics = {
        "decay_slope": fit_decay_curve(distance_effects)["slope"],
        "decay_floor": fit_decay_curve(distance_effects)["floor"],
        "decay_intercept": fit_decay_curve(distance_effects)["intercept"],
        "background_mean": background["mean"],
        "background_p95": background["p95"],
        "near_threshold_fraction": feature_audit.get("near_threshold_fraction", 0.0),
        "hidden_volatility": feature_audit.get("hidden_volatility", 0.0),
        "body_effect": body_effect,
        "far_effect": far_effect,
    }
    body_over_bg = body_effect / max(background["mean"], 1e-9)
    far_over_bg = far_effect / max(background["mean"], 1e-9)
    metrics["body_over_background"] = body_over_bg
    metrics["far_over_background"] = far_over_bg

    # Local-window check: if far drops sharply under local-window
    # rollout, it was propagation through the world — reclassify.
    lw = stabilization.get("local_window", {})
    bl = stabilization.get("baseline", {})
    if (lw and bl
            and not lw.get("global_chaotic_label_would_fire", True)
            and bl.get("global_chaotic_label_would_fire", False)):
        return "broad_hidden_coupling", 0.7, metrics

    # Threshold-volatility artifact: high near-threshold fraction or
    # high volatility AND threshold filter would remove the candidate.
    if (feature_audit.get("near_threshold_fraction", 0.0) > 0.4
            or feature_audit.get("hidden_volatility", 0.0) > 0.5):
        return "threshold_volatility_artifact", 0.6, metrics

    # Far-control artifact: only antipode is hot.
    far_probes = [e for e in distance_effects
                  if e.name in ("far_2r", "far_5r", "far_10r")
                  and e.n_perturbed_2d > 0]
    antipode = next((e for e in distance_effects if e.name == "antipode"), None)
    if far_probes and antipode and antipode.effect_per_cell > 0:
        far_mean = float(np.mean([e.effect_per_cell for e in far_probes]))
        if antipode.effect_per_cell > 1.5 * max(far_mean, 1e-9):
            return "far_control_artifact", 0.6, metrics

    # Background-sensitive world: far effect ≈ background distribution.
    if (background["n_samples"] > 0
            and far_effect <= 1.5 * background["p95"]
            and body_over_bg > 1.5):
        return "background_sensitive_world", 0.6, metrics

    # Decay analysis.
    fit = fit_decay_curve(distance_effects)
    if fit["n_points"] >= 3:
        decay = fit["slope"]
        # Effect decays with distance.
        if decay < -1e-7:
            return "broad_hidden_coupling", 0.6, metrics
        # Effect is flat with distance.
        if abs(decay) <= 1e-7:
            return "global_instability", 0.6, metrics

    return "unresolved_global", 0.0, metrics


# ---------------------------------------------------------------------------
# Per-candidate end-to-end M8D measurement
# ---------------------------------------------------------------------------


@dataclass
class M8DCandidateResult:
    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    snapshot_t: int
    candidate_area: int
    candidate_lifetime: int
    near_threshold_fraction: float

    morphology: object
    far_control: FarControlInfo

    region_effects: dict
    far_effect: object
    distance_effects: list

    background_mean: float
    background_p95: float
    background_p99: float
    body_over_background: float
    far_over_background: float

    feature_audit: dict
    stabilization: dict

    # Original M8B/M8C label.
    base_mechanism_label: str
    base_mechanism_confidence: float

    # If base label was global_chaotic, the relabel; else equals base.
    final_mechanism_label: str
    final_mechanism_confidence: float
    relabel_metrics: dict


def measure_candidate_m8d(
    *, snapshot_4d, candidate_mask_2d, rule, rule_id, rule_source, seed,
    candidate_id, snapshot_t, candidate_area, candidate_lifetime,
    near_threshold_fraction, horizons, n_replicates, backend, rng_seed,
    region_shell_widths=(1, 2, 3),
    background_n_samples: int = 16, background_sample_size: int = 8,
    stabilization_window_dilation: int = 5,
    min_far_distance_floor: int = 24,
    min_far_distance_radius_mult: float = 5.0,
) -> M8DCandidateResult:
    morph = classify_morphology(candidate_mask_2d)
    headline_h = horizons[len(horizons) // 2]

    # Reuse M8B's region-aware measurement.
    region_effects, _ = measure_all_regions(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizon=headline_h,
        n_replicates=n_replicates, backend=backend, rng_seed=rng_seed,
        region_shell_widths=region_shell_widths,
    )
    far_mask, far_info = select_far_mask(
        candidate_mask_2d, snapshot_4d=snapshot_4d,
        min_distance_floor=min_far_distance_floor,
        min_distance_radius_mult=min_far_distance_radius_mult,
        rng_seed=rng_seed + 7,
    )
    if far_info.far_control_valid:
        far_eff = measure_region_effect(
            snapshot_4d=snapshot_4d, rule=rule, region_mask_2d=far_mask,
            candidate_mask_2d=candidate_mask_2d, region_name="far_validated",
            horizon=headline_h, n_replicates=n_replicates, backend=backend,
            rng_seed=rng_seed + 11,
        )
    else:
        far_eff = RegionEffect(
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
    base_label, base_conf, _ = classify_mechanism_v2(
        morphology=morph, region_effects=region_effects, far_effect=far_eff,
        first_visible_effect_time=first_visible,
        fraction_hidden_at_end=frac_hid, fraction_visible_at_end=frac_vis,
        near_threshold_fraction=near_threshold_fraction,
    )

    # Always run the multi-distance probe + background sample (for
    # comparison curves with non-global candidates).
    dist_effects = measure_multi_distance_effects(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizon=headline_h,
        n_replicates=n_replicates, backend=backend,
        rng_seed=rng_seed + 200,
    )
    background = measure_background_sensitivity(
        snapshot_4d=snapshot_4d, rule=rule,
        candidate_mask_2d=candidate_mask_2d, horizon=headline_h,
        n_samples=background_n_samples,
        sample_size=background_sample_size,
        backend=backend, rng_seed=rng_seed + 300,
    )
    feats = audit_features(snapshot_4d, candidate_mask_2d)

    body_eff_obj = next((e for e in dist_effects if e.name == "body"), None)
    body_effect_raw = body_eff_obj.raw_effect if body_eff_obj else 0.0
    far_effect_raw = far_eff.region_hidden_effect

    # Stabilization runs for global candidates only (cost).
    if base_label == "global_chaotic":
        stab = stabilization_variants(
            snapshot_4d=snapshot_4d, rule=rule,
            candidate_mask_2d=candidate_mask_2d, horizon=headline_h,
            n_replicates=n_replicates, backend=backend,
            rng_seed=rng_seed + 400,
            window_dilation=stabilization_window_dilation,
        )
        final_label, final_conf, relabel_metrics = relabel_global_chaotic(
            distance_effects=dist_effects, background=background,
            feature_audit=feats, stabilization=stab,
            body_effect=body_effect_raw, far_effect=far_effect_raw,
        )
    else:
        stab = {}
        final_label, final_conf = base_label, base_conf
        relabel_metrics = {}

    bg_mean = background["mean"]
    body_over_bg = body_effect_raw / max(bg_mean, 1e-9)
    far_over_bg = far_effect_raw / max(bg_mean, 1e-9)

    return M8DCandidateResult(
        rule_id=rule_id, rule_source=rule_source, seed=seed,
        candidate_id=candidate_id, snapshot_t=snapshot_t,
        candidate_area=candidate_area, candidate_lifetime=candidate_lifetime,
        near_threshold_fraction=near_threshold_fraction,
        morphology=morph, far_control=far_info,
        region_effects=region_effects, far_effect=far_eff,
        distance_effects=dist_effects,
        background_mean=bg_mean, background_p95=background["p95"],
        background_p99=background["p99"],
        body_over_background=body_over_bg,
        far_over_background=far_over_bg,
        feature_audit=feats, stabilization=stab,
        base_mechanism_label=base_label,
        base_mechanism_confidence=base_conf,
        final_mechanism_label=final_label,
        final_mechanism_confidence=final_conf,
        relabel_metrics=relabel_metrics,
    )
