"""M8 — mechanism discovery.

Decomposes how hidden 4D state causally supports projected 2D candidate
dynamics. Six analyses per candidate:

  1. **Per-column response map** — apply hidden_invisible perturbation
     to each (x, y) column individually, measure future local
     divergence, build a 2D heatmap over the candidate footprint.
  2. **Hidden-to-visible emergence timing** — measure divergence at
     dense short horizons to find when hidden state first surfaces.
  3. **Pathway tracing** — track the 4D XOR difference mass over
     rollout time, separating hidden vs visible components.
  4. **Mediation analysis** — apply targeted perturbations to
     interior / boundary / environment / far regions and compare
     effect sizes.
  5. **Feature dynamics** — compute M6C hidden features as a time
     series; identify which features lead visible divergence.
  6. **Mechanism classifier** — rule-based assignment of one of 7
     classes per candidate (boundary_mediated, interior_reservoir,
     environment_coupled, global_chaotic, threshold_mediated,
     delayed_hidden_channel, unclear).

The core constraint M6B/M6C/M7/M7B verified — hidden_invisible
perturbations preserve the 2D projection at t=0 by construction —
remains the foundation. Every measurement here flows from that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi

from observer_worlds.analysis.hidden_features import candidate_hidden_features
from observer_worlds.experiments._m6b_interventions import (
    apply_far_hidden_intervention,
    apply_sham_intervention,
)
from observer_worlds.metrics.causality_score import (
    apply_flip_intervention,
    apply_hidden_shuffle_intervention,
)
from observer_worlds.worlds import CA4D, BSRule, project


MECHANISM_CLASSES: tuple[str, ...] = (
    "boundary_mediated",
    "interior_reservoir",
    "environment_coupled",
    "global_chaotic",
    "threshold_mediated",
    "delayed_hidden_channel",
    "unclear",
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ResponseMap:
    """Per-column response map for one candidate at one horizon."""

    candidate_id: int
    horizon: int
    grid_shape: tuple[int, int]
    interior_mask: np.ndarray            # bool, full-grid
    response_grid: np.ndarray            # float, full-grid; 0 outside interior

    # Aggregated metrics (interior, boundary, environment shells of the candidate).
    interior_response_fraction: float = 0.0
    boundary_response_fraction: float = 0.0
    environment_response_fraction: float = 0.0
    response_concentration: float = 0.0       # max(response) / mean(response)
    response_centroid_distance_to_boundary: float = 0.0
    response_entropy: float = 0.0


@dataclass
class EmergenceTiming:
    """Hidden-to-visible emergence timing for one candidate."""

    candidate_id: int
    horizons: list[int]
    full_grid_l1_per_horizon: list[float]
    local_l1_per_horizon: list[float]
    first_visible_effect_time: int = -1   # first horizon where local_l1 > epsilon
    peak_effect_time: int = -1
    effect_growth_rate: float = 0.0       # slope from h=1 to peak
    effect_decay_rate: float = 0.0        # slope from peak to last
    auc_full: float = 0.0
    auc_local: float = 0.0


@dataclass
class PathwayTrace:
    """Per-step 4D / 2D divergence mass for one candidate × one
    intervention replicate."""

    candidate_id: int
    n_steps: int
    hidden_mass_per_step: list[int]       # number of differing 4D cells at each step
    visible_mass_per_step: list[int]      # number of differing projected cells
    hidden_to_visible_conversion_time: int = -1
    spread_radius_4d: list[float] = field(default_factory=list)
    spread_radius_2d: list[float] = field(default_factory=list)
    fraction_hidden_at_end: float = 0.0
    fraction_visible_at_end: float = 0.0


@dataclass
class MediationResult:
    """Per-target mediation effect sizes for one candidate."""

    candidate_id: int
    interior_hidden_effect: float = 0.0
    boundary_hidden_effect: float = 0.0
    environment_hidden_effect: float = 0.0
    far_hidden_effect: float = 0.0
    visible_boundary_effect: float = 0.0
    visible_environment_effect: float = 0.0
    boundary_mediation_index: float = 0.0
    candidate_locality_index: float = 0.0


@dataclass
class FeatureDynamics:
    """Time series of hidden features for one candidate × intervention."""

    candidate_id: int
    horizons: list[int]
    feature_names: list[str]
    deltas: dict             # feature_name → list[float] of feature(intervened) - feature(orig) per horizon
    visible_div_per_horizon: list[float]
    leading_features: list[tuple[str, int, float]]  # (feature_name, lag, lag_corr)


@dataclass
class MechanismLabel:
    """Per-candidate mechanism classification."""

    candidate_id: int
    rule_id: str
    rule_source: str
    seed: int
    label: str
    confidence: float
    supporting_metrics: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project(state, theta=0.5):
    return project(state, method="mean_threshold", theta=theta)


def _l1_full(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum() / a.size)


def _l1_local(a, b, mask):
    if not mask.any(): return 0.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff[mask].sum() / mask.sum())


def _rollout_proj_capturing_4d(
    state, rule, n_steps, *, backend, theta=0.5,
):
    """Run a CA4D rollout from `state`. Return both per-step projected
    frames and per-step 4D states (so the pathway tracer can compute
    XOR mass)."""
    Nx, Ny = state.shape[0], state.shape[1]
    ca = CA4D(shape=state.shape, rule=rule, backend=backend)
    ca.state = state.copy()
    proj = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    full4d = np.empty((n_steps, *state.shape), dtype=np.uint8)
    for t in range(n_steps):
        ca.step()
        proj[t] = _project(ca.state, theta)
        full4d[t] = ca.state.copy()
    return proj, full4d


def _rollout_proj(state, rule, n_steps, *, backend, theta=0.5):
    Nx, Ny = state.shape[0], state.shape[1]
    ca = CA4D(shape=state.shape, rule=rule, backend=backend)
    ca.state = state.copy()
    out = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    for t in range(n_steps):
        ca.step()
        out[t] = _project(ca.state, theta)
    return out


def _shell_masks(interior_mask: np.ndarray, *, env_dilation: int = 4
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (boundary_mask, env_mask) given the interior mask.

    boundary = interior XOR erosion(interior, 1)
    env = (dilation(interior, env_dilation) XOR dilation(interior, 1)) AND ~interior
    """
    if not interior_mask.any():
        z = np.zeros_like(interior_mask)
        return z, z
    interior_eroded = ndi.binary_erosion(interior_mask, iterations=1)
    boundary = interior_mask & ~interior_eroded
    if not boundary.any():
        boundary = interior_mask  # fallback for thin candidates
    outer = ndi.binary_dilation(interior_mask, iterations=env_dilation)
    inner = ndi.binary_dilation(interior_mask, iterations=1)
    env = outer & ~inner & ~interior_mask
    return boundary, env


# ---------------------------------------------------------------------------
# 1. Response map
# ---------------------------------------------------------------------------


def compute_response_map(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    horizon: int,
    n_replicates: int,
    backend: str,
    rng_seed: int,
) -> ResponseMap:
    """For each (x,y) in the interior, perturb only that column's
    z,w fiber with a hidden_invisible shuffle, run forward, and record
    the full-grid local divergence at the chosen horizon.

    ``backend`` accepts:
      * ``"numba"`` / ``"numpy"`` / ``"cuda"`` — serial per-column rollouts
        on the corresponding CA4D backend.
      * ``"cuda-batched"`` — all per-column probes evolved together in a
        single CA4DBatch kernel launch (one chunk per ``max_chunk``
        elements). Aggregate metrics are byte-identical to the serial
        path because (a) the CUDA single-step is bit-identical to numba
        and (b) the per-(coord, rep) RNG draw order is preserved.
    """
    if backend == "cuda-batched":
        return _compute_response_map_cuda_batched(
            snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
            candidate_id=candidate_id, horizon=horizon,
            n_replicates=n_replicates, rng_seed=rng_seed,
        )
    return _compute_response_map_serial(
        snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
        candidate_id=candidate_id, horizon=horizon,
        n_replicates=n_replicates, rng_seed=rng_seed,
        backend=backend,
    )


def _compute_response_map_serial(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    horizon: int,
    n_replicates: int,
    rng_seed: int,
    backend: str,
) -> ResponseMap:
    """Serial per-column response map (canonical CPU reference)."""
    Nx, Ny = snapshot_4d.shape[0], snapshot_4d.shape[1]
    response = np.zeros((Nx, Ny), dtype=np.float64)

    if not interior_mask.any():
        return ResponseMap(
            candidate_id=candidate_id, horizon=horizon, grid_shape=(Nx, Ny),
            interior_mask=interior_mask, response_grid=response,
        )

    # Unperturbed rollout (one-shot, reused for all per-column probes).
    frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend=backend)
    parent_rng = np.random.default_rng(rng_seed)

    boundary, env = _shell_masks(interior_mask)
    # Combine interior + boundary (shells overlap if thin).
    probe_mask = interior_mask | boundary

    coords = np.argwhere(probe_mask)
    for x, y in coords:
        # Single-column mask containing only (x, y).
        col_mask = np.zeros_like(interior_mask)
        col_mask[x, y] = True

        responses_at_h: list[float] = []
        for rep in range(n_replicates):
            r = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
            perturbed = apply_hidden_shuffle_intervention(snapshot_4d, col_mask, r)
            f_int = _rollout_proj(perturbed, rule, horizon, backend=backend)
            local = _l1_local(frames_orig[-1], f_int[-1], interior_mask)
            responses_at_h.append(local)
        response[x, y] = float(np.mean(responses_at_h)) if responses_at_h else 0.0

    return _aggregate_response_map(
        response=response, interior_mask=interior_mask,
        boundary=boundary, env=env,
        candidate_id=candidate_id, horizon=horizon,
        grid_shape=(Nx, Ny),
    )


def _compute_response_map_cuda_batched(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    horizon: int,
    n_replicates: int,
    rng_seed: int,
    max_chunk: int = 128,
) -> ResponseMap:
    """Batched per-column response map.

    Builds a ``(B, *snapshot_4d.shape)`` host array — one batch element
    per ``(coord, rep)`` pair, in the same order as the serial path —
    then evolves them all via :func:`evolve_chunked` in a single kernel
    launch (chunked at ``max_chunk`` to fit VRAM). The final 4D states
    are projected to 2D and the local L1 metric is computed against the
    one-shot unperturbed reference rollout.

    Byte-identity invariant: because (a) the CUDA single-step is
    bit-identical to numba and (b) we draw one child RNG per
    ``(coord, rep)`` in the same order as the serial path, the resulting
    ``response_grid`` should match the serial reference exactly.
    """
    from observer_worlds.worlds.ca4d_batch import evolve_chunked

    Nx, Ny = snapshot_4d.shape[0], snapshot_4d.shape[1]
    response = np.zeros((Nx, Ny), dtype=np.float64)

    if not interior_mask.any():
        return ResponseMap(
            candidate_id=candidate_id, horizon=horizon, grid_shape=(Nx, Ny),
            interior_mask=interior_mask, response_grid=response,
        )

    # Unperturbed reference rollout — single CPU pass, reused as the
    # comparison baseline for every probe. Backend "numba" is the
    # canonical CPU reference; the CUDA single-step is bit-identical so
    # this is also what the GPU would produce.
    frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend="numba")
    parent_rng = np.random.default_rng(rng_seed)

    boundary, env = _shell_masks(interior_mask)
    probe_mask = interior_mask | boundary

    coords = np.argwhere(probe_mask)
    n_coords = len(coords)
    B = n_coords * n_replicates

    if B == 0:
        return _aggregate_response_map(
            response=response, interior_mask=interior_mask,
            boundary=boundary, env=env,
            candidate_id=candidate_id, horizon=horizon,
            grid_shape=(Nx, Ny),
        )

    # Build (B, *shape) host array of perturbed initial states. Order:
    # coord-major, replicate-minor — same as the serial loop.
    states = np.empty((B, *snapshot_4d.shape), dtype=np.uint8)
    b = 0
    for x, y in coords:
        col_mask = np.zeros_like(interior_mask)
        col_mask[x, y] = True
        for _rep in range(n_replicates):
            r = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
            states[b] = apply_hidden_shuffle_intervention(snapshot_4d, col_mask, r)
            b += 1

    # Single batched kernel launch (chunked at max_chunk for VRAM).
    final_states = evolve_chunked(
        shape=snapshot_4d.shape,
        rules=[rule] * B,
        initial_states_host=states,
        n_steps=horizon,
        max_chunk=max_chunk,
    )

    # Project each final 4D state and compute the local L1 metric, then
    # average across replicates per coord.
    ref_proj = frames_orig[-1]
    b = 0
    for x, y in coords:
        responses_at_h: list[float] = []
        for _rep in range(n_replicates):
            f_int_proj = _project(final_states[b])
            responses_at_h.append(
                _l1_local(ref_proj, f_int_proj, interior_mask)
            )
            b += 1
        response[x, y] = (
            float(np.mean(responses_at_h)) if responses_at_h else 0.0
        )

    return _aggregate_response_map(
        response=response, interior_mask=interior_mask,
        boundary=boundary, env=env,
        candidate_id=candidate_id, horizon=horizon,
        grid_shape=(Nx, Ny),
    )


def _aggregate_response_map(
    *,
    response: np.ndarray,
    interior_mask: np.ndarray,
    boundary: np.ndarray,
    env: np.ndarray,
    candidate_id: int,
    horizon: int,
    grid_shape: tuple[int, int],
) -> ResponseMap:
    """Compute aggregate metrics from a filled response grid and return a
    :class:`ResponseMap`. Shared by serial and cuda-batched paths."""
    total = response.sum()
    interior_resp = response[interior_mask].sum()
    bnd_resp = response[boundary].sum()
    env_resp = response[env].sum()

    interior_frac = float(interior_resp / total) if total > 1e-12 else 0.0
    bnd_frac = float(bnd_resp / total) if total > 1e-12 else 0.0
    env_frac = float(env_resp / total) if total > 1e-12 else 0.0

    nonzero = response[response > 1e-12]
    if nonzero.size > 0:
        concentration = float(nonzero.max() / nonzero.mean())
    else:
        concentration = 0.0

    # Response entropy (over normalized non-zero values).
    if total > 1e-12:
        p = response[response > 1e-12] / total
        entropy = float(-(p * np.log(p + 1e-12)).sum())
    else:
        entropy = 0.0

    # Centroid distance to boundary.
    if total > 1e-12 and boundary.any():
        rows, cols = np.where(response > 1e-12)
        weights = response[rows, cols]
        cy = float((rows * weights).sum() / weights.sum())
        cx = float((cols * weights).sum() / weights.sum())
        b_rows, b_cols = np.where(boundary)
        if b_rows.size > 0:
            dists = np.sqrt((b_rows - cy) ** 2 + (b_cols - cx) ** 2)
            cent_dist = float(dists.min())
        else:
            cent_dist = 0.0
    else:
        cent_dist = 0.0

    return ResponseMap(
        candidate_id=candidate_id, horizon=horizon, grid_shape=grid_shape,
        interior_mask=interior_mask, response_grid=response,
        interior_response_fraction=interior_frac,
        boundary_response_fraction=bnd_frac,
        environment_response_fraction=env_frac,
        response_concentration=concentration,
        response_centroid_distance_to_boundary=cent_dist,
        response_entropy=entropy,
    )


# ---------------------------------------------------------------------------
# 2. Emergence timing
# ---------------------------------------------------------------------------


def compute_emergence_timing(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    rng_seed: int,
    epsilon: float = 1e-3,
) -> EmergenceTiming:
    """Apply a single hidden_invisible perturbation, run forward, measure
    divergence at every horizon."""
    H_max = max(horizons)
    frames_orig = _rollout_proj(snapshot_4d, rule, H_max, backend=backend)
    parent_rng = np.random.default_rng(rng_seed)

    full_per_h = {h: [] for h in horizons}
    local_per_h = {h: [] for h in horizons}
    for rep in range(n_replicates):
        r = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
        perturbed = apply_hidden_shuffle_intervention(snapshot_4d, interior_mask, r)
        f_int = _rollout_proj(perturbed, rule, H_max, backend=backend)
        for h in horizons:
            full_per_h[h].append(_l1_full(frames_orig[h - 1], f_int[h - 1]))
            local_per_h[h].append(_l1_local(frames_orig[h - 1], f_int[h - 1],
                                            interior_mask))

    full_means = [float(np.mean(full_per_h[h])) for h in horizons]
    local_means = [float(np.mean(local_per_h[h])) for h in horizons]

    first_visible = -1
    for i, h in enumerate(horizons):
        if local_means[i] > epsilon:
            first_visible = h
            break
    peak_idx = int(np.argmax(full_means)) if full_means else 0
    peak_t = horizons[peak_idx] if full_means else -1
    growth = (full_means[peak_idx] / max(peak_t, 1)) if full_means else 0.0
    decay = (
        (full_means[-1] - full_means[peak_idx]) / max(horizons[-1] - peak_t, 1)
        if full_means and peak_idx < len(full_means) - 1 else 0.0
    )
    auc_full = float(sum(full_means))
    auc_local = float(sum(local_means))

    return EmergenceTiming(
        candidate_id=candidate_id, horizons=list(horizons),
        full_grid_l1_per_horizon=full_means,
        local_l1_per_horizon=local_means,
        first_visible_effect_time=first_visible,
        peak_effect_time=peak_t,
        effect_growth_rate=float(growth),
        effect_decay_rate=float(decay),
        auc_full=auc_full, auc_local=auc_local,
    )


# ---------------------------------------------------------------------------
# 3. Pathway tracing
# ---------------------------------------------------------------------------


def compute_pathway_trace(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    n_steps: int,
    backend: str,
    rng_seed: int,
    visible_emergence_threshold: int = 1,
) -> PathwayTrace:
    """Track XOR mass between original and intervened 4D rollouts at
    each step. Separately track hidden-mass (4D cells differing) vs
    visible-mass (projected cells differing)."""
    rng = np.random.default_rng(rng_seed)
    perturbed = apply_hidden_shuffle_intervention(snapshot_4d, interior_mask, rng)
    proj_orig, full_orig = _rollout_proj_capturing_4d(
        snapshot_4d, rule, n_steps, backend=backend
    )
    proj_int, full_int = _rollout_proj_capturing_4d(
        perturbed, rule, n_steps, backend=backend
    )

    hidden_mass = []
    visible_mass = []
    spread_4d = []
    spread_2d = []
    for t in range(n_steps):
        diff_4d = full_orig[t] != full_int[t]
        diff_2d = proj_orig[t] != proj_int[t]
        hidden_mass.append(int(diff_4d.sum()))
        visible_mass.append(int(diff_2d.sum()))
        # Spread radii: distance from candidate centroid.
        if interior_mask.any():
            rows, cols = np.where(interior_mask)
            cy = float(rows.mean()); cx = float(cols.mean())
            dr_4d, dc_4d = np.where(diff_4d.any(axis=(2, 3)))
            if dr_4d.size > 0:
                d4 = np.sqrt((dr_4d - cy) ** 2 + (dc_4d - cx) ** 2)
                spread_4d.append(float(d4.max()))
            else:
                spread_4d.append(0.0)
            dr_2d, dc_2d = np.where(diff_2d)
            if dr_2d.size > 0:
                d2 = np.sqrt((dr_2d - cy) ** 2 + (dc_2d - cx) ** 2)
                spread_2d.append(float(d2.max()))
            else:
                spread_2d.append(0.0)
        else:
            spread_4d.append(0.0); spread_2d.append(0.0)

    conv_t = -1
    for t, vm in enumerate(visible_mass):
        if vm >= visible_emergence_threshold:
            conv_t = t + 1
            break
    total_4d_cells = full_orig[0].size
    total_2d_cells = proj_orig[0].size
    return PathwayTrace(
        candidate_id=candidate_id, n_steps=n_steps,
        hidden_mass_per_step=hidden_mass,
        visible_mass_per_step=visible_mass,
        hidden_to_visible_conversion_time=conv_t,
        spread_radius_4d=spread_4d, spread_radius_2d=spread_2d,
        fraction_hidden_at_end=(hidden_mass[-1] / total_4d_cells) if hidden_mass else 0.0,
        fraction_visible_at_end=(visible_mass[-1] / total_2d_cells) if visible_mass else 0.0,
    )


# ---------------------------------------------------------------------------
# 4. Mediation analysis
# ---------------------------------------------------------------------------


def compute_mediation(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    horizon: int,
    n_replicates: int,
    backend: str,
    rng_seed: int,
) -> MediationResult:
    """Apply hidden_invisible interventions to each shell (interior,
    boundary, environment, far) and visible_flip to boundary/env, then
    measure full-grid divergence at the horizon."""
    boundary, env = _shell_masks(interior_mask)
    Nz, Nw = snapshot_4d.shape[2], snapshot_4d.shape[3]
    parent_rng = np.random.default_rng(rng_seed)
    frames_orig = _rollout_proj(snapshot_4d, rule, horizon, backend=backend)

    def _hidden_effect_on_mask(m):
        if not m.any(): return 0.0
        vals = []
        for _ in range(n_replicates):
            r = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
            p = apply_hidden_shuffle_intervention(snapshot_4d, m, r)
            f = _rollout_proj(p, rule, horizon, backend=backend)
            vals.append(_l1_full(frames_orig[-1], f[-1]))
        return float(np.mean(vals))

    def _visible_effect_on_mask(m):
        if not m.any(): return 0.0
        vals = []
        for _ in range(n_replicates):
            r = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
            p = apply_flip_intervention(snapshot_4d, m, 0.5, r)
            f = _rollout_proj(p, rule, horizon, backend=backend)
            vals.append(_l1_full(frames_orig[-1], f[-1]))
        return float(np.mean(vals))

    interior_e = _hidden_effect_on_mask(interior_mask)
    boundary_e = _hidden_effect_on_mask(boundary)
    env_e = _hidden_effect_on_mask(env)
    # Far: translated mask.
    far_state = snapshot_4d.copy()
    rng = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))
    far_perturbed, far_mask = apply_far_hidden_intervention(snapshot_4d, interior_mask, rng)
    f = _rollout_proj(far_perturbed, rule, horizon, backend=backend)
    far_e = _l1_full(frames_orig[-1], f[-1])

    visible_b = _visible_effect_on_mask(boundary)
    visible_env = _visible_effect_on_mask(env)

    bmi = (boundary_e / (interior_e + boundary_e + 1e-9)) if (interior_e + boundary_e) > 0 else 0.0
    cli = float(interior_e + boundary_e - far_e)

    return MediationResult(
        candidate_id=candidate_id,
        interior_hidden_effect=interior_e,
        boundary_hidden_effect=boundary_e,
        environment_hidden_effect=env_e,
        far_hidden_effect=far_e,
        visible_boundary_effect=visible_b,
        visible_environment_effect=visible_env,
        boundary_mediation_index=float(bmi),
        candidate_locality_index=cli,
    )


# ---------------------------------------------------------------------------
# 5. Feature dynamics
# ---------------------------------------------------------------------------


# A small subset of M6C features that we re-evaluate per-step. Re-running
# the full 24-feature extraction at every horizon would dominate runtime.
_DYNAMIC_FEATURE_NAMES = (
    "mean_active_fraction",
    "mean_threshold_margin",
    "near_threshold_fraction",
    "mean_hidden_entropy",
    "hidden_heterogeneity",
)


def compute_feature_dynamics(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    candidate_id: int,
    horizons: list[int],
    backend: str,
    rng_seed: int,
) -> FeatureDynamics:
    """Track hidden features in original vs perturbed rollouts; compute
    leading indicators."""
    rng = np.random.default_rng(rng_seed)
    perturbed = apply_hidden_shuffle_intervention(snapshot_4d, interior_mask, rng)
    H_max = max(horizons)
    proj_orig, full_orig = _rollout_proj_capturing_4d(
        snapshot_4d, rule, H_max, backend=backend
    )
    proj_int, full_int = _rollout_proj_capturing_4d(
        perturbed, rule, H_max, backend=backend
    )

    deltas: dict[str, list[float]] = {n: [] for n in _DYNAMIC_FEATURE_NAMES}
    visible_div = []
    for t in range(H_max):
        f_orig = candidate_hidden_features(full_orig[t], interior_mask)
        f_int = candidate_hidden_features(full_int[t], interior_mask)
        for fn in _DYNAMIC_FEATURE_NAMES:
            deltas[fn].append(float(f_int.get(fn, 0.0) - f_orig.get(fn, 0.0)))
        visible_div.append(_l1_full(proj_orig[t], proj_int[t]))

    # Leading-feature analysis: for each feature, compute lag-correlation
    # between feature_delta[t] and visible_div[t+lag] for lag in {1,2,3}.
    leading: list[tuple[str, int, float]] = []
    vis_arr = np.array(visible_div)
    for fn in _DYNAMIC_FEATURE_NAMES:
        f_arr = np.array(deltas[fn])
        for lag in (1, 2, 3, 5):
            if H_max - lag < 3: continue
            # corr(feature[0:H-lag], visible[lag:H])
            x = f_arr[:H_max - lag]
            y = vis_arr[lag:H_max]
            if x.std() < 1e-12 or y.std() < 1e-12:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            leading.append((fn, lag, corr))
    # Sort by absolute correlation.
    leading.sort(key=lambda t: -abs(t[2]))

    # Also store per-horizon (subsample to the requested horizons).
    return FeatureDynamics(
        candidate_id=candidate_id, horizons=list(horizons),
        feature_names=list(_DYNAMIC_FEATURE_NAMES),
        deltas={fn: [deltas[fn][h - 1] if h - 1 < H_max else 0.0
                    for h in horizons]
               for fn in _DYNAMIC_FEATURE_NAMES},
        visible_div_per_horizon=[visible_div[h - 1] if h - 1 < H_max else 0.0
                                for h in horizons],
        leading_features=leading[:5],
    )


# ---------------------------------------------------------------------------
# 6. Mechanism classifier
# ---------------------------------------------------------------------------


def classify_mechanism(
    *,
    rmap: ResponseMap,
    timing: EmergenceTiming,
    pathway: PathwayTrace,
    mediation: MediationResult,
    near_threshold_fraction: float,
    candidate_id: int,
    rule_id: str,
    rule_source: str,
    seed: int,
) -> MechanismLabel:
    """Apply spec'd rule-based classification."""
    metrics = {
        "boundary_response_fraction": rmap.boundary_response_fraction,
        "interior_response_fraction": rmap.interior_response_fraction,
        "environment_response_fraction": rmap.environment_response_fraction,
        "first_visible_effect_time": timing.first_visible_effect_time,
        "hidden_to_visible_conversion_time": pathway.hidden_to_visible_conversion_time,
        "fraction_hidden_at_end": pathway.fraction_hidden_at_end,
        "fraction_visible_at_end": pathway.fraction_visible_at_end,
        "boundary_mediation_index": mediation.boundary_mediation_index,
        "candidate_locality_index": mediation.candidate_locality_index,
        "far_hidden_effect": mediation.far_hidden_effect,
        "interior_hidden_effect": mediation.interior_hidden_effect,
        "boundary_hidden_effect": mediation.boundary_hidden_effect,
        "near_threshold_fraction": near_threshold_fraction,
    }

    # Rule 1: threshold-mediated
    if (near_threshold_fraction > 0.5
            and mediation.interior_hidden_effect <= 0.5 * mediation.boundary_hidden_effect + 1e-6):
        # If most hidden effect comes through near-threshold cells, flag it.
        # (Heuristic; M6C threshold-audit gives the cleaner version.)
        return MechanismLabel(
            candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
            seed=seed, label="threshold_mediated",
            confidence=min(1.0, near_threshold_fraction),
            supporting_metrics=metrics,
        )

    # Rule 2: global_chaotic — far effect is comparable to candidate effect
    cand_e = max(mediation.interior_hidden_effect + mediation.boundary_hidden_effect,
                 1e-9)
    if mediation.far_hidden_effect > 0.7 * cand_e:
        return MechanismLabel(
            candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
            seed=seed, label="global_chaotic",
            confidence=float(min(1.0, mediation.far_hidden_effect / cand_e)),
            supporting_metrics=metrics,
        )

    # Rule 3: boundary_mediated
    if rmap.boundary_response_fraction > 0.6:
        return MechanismLabel(
            candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
            seed=seed, label="boundary_mediated",
            confidence=float(rmap.boundary_response_fraction),
            supporting_metrics=metrics,
        )

    # Rule 4: interior_reservoir
    if rmap.interior_response_fraction > 0.6:
        return MechanismLabel(
            candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
            seed=seed, label="interior_reservoir",
            confidence=float(rmap.interior_response_fraction),
            supporting_metrics=metrics,
        )

    # Rule 5: environment_coupled
    if rmap.environment_response_fraction > 0.4:
        return MechanismLabel(
            candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
            seed=seed, label="environment_coupled",
            confidence=float(rmap.environment_response_fraction),
            supporting_metrics=metrics,
        )

    # Rule 6: delayed_hidden_channel
    # Hidden mass grows before visible mass; first_visible_effect_time is
    # delayed; fraction_hidden_at_end > fraction_visible_at_end.
    if (timing.first_visible_effect_time >= 5
            and pathway.fraction_hidden_at_end > 0.5 * pathway.fraction_visible_at_end):
        return MechanismLabel(
            candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
            seed=seed, label="delayed_hidden_channel",
            confidence=0.6,
            supporting_metrics=metrics,
        )

    return MechanismLabel(
        candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
        seed=seed, label="unclear", confidence=0.0,
        supporting_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Per-candidate end-to-end runner
# ---------------------------------------------------------------------------


@dataclass
class M8CandidateResult:
    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    snapshot_t: int
    candidate_area: float
    candidate_lifetime: int
    observer_score: float | None
    near_threshold_fraction: float

    response_map: ResponseMap
    timing: EmergenceTiming
    pathway: PathwayTrace
    mediation: MediationResult
    feature_dynamics: FeatureDynamics
    mechanism: MechanismLabel


def measure_candidate_m8(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    rule_id: str,
    rule_source: str,
    seed: int,
    candidate_id: int,
    snapshot_t: int,
    candidate_area: float,
    candidate_lifetime: int,
    observer_score: float | None,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    rng_seed: int,
) -> M8CandidateResult:
    if not interior_mask.any():
        raise ValueError("empty interior mask")

    feats = candidate_hidden_features(snapshot_4d, interior_mask)
    near_thresh = float(feats.get("near_threshold_fraction", 0.0))

    headline_h = horizons[len(horizons) // 2]
    rmap = compute_response_map(
        snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
        candidate_id=candidate_id, horizon=headline_h,
        n_replicates=n_replicates, backend=backend, rng_seed=rng_seed,
    )
    timing = compute_emergence_timing(
        snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
        candidate_id=candidate_id, horizons=horizons,
        n_replicates=n_replicates, backend=backend, rng_seed=rng_seed + 1,
    )
    pathway = compute_pathway_trace(
        snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
        candidate_id=candidate_id, n_steps=max(horizons),
        backend=backend, rng_seed=rng_seed + 2,
    )
    mediation = compute_mediation(
        snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
        candidate_id=candidate_id, horizon=headline_h,
        n_replicates=n_replicates, backend=backend, rng_seed=rng_seed + 3,
    )
    feat_dyn = compute_feature_dynamics(
        snapshot_4d=snapshot_4d, rule=rule, interior_mask=interior_mask,
        candidate_id=candidate_id, horizons=horizons,
        backend=backend, rng_seed=rng_seed + 4,
    )
    mech = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pathway, mediation=mediation,
        near_threshold_fraction=near_thresh, candidate_id=candidate_id,
        rule_id=rule_id, rule_source=rule_source, seed=seed,
    )
    return M8CandidateResult(
        rule_id=rule_id, rule_source=rule_source, seed=seed,
        candidate_id=candidate_id, snapshot_t=snapshot_t,
        candidate_area=candidate_area, candidate_lifetime=candidate_lifetime,
        observer_score=observer_score,
        near_threshold_fraction=near_thresh,
        response_map=rmap, timing=timing, pathway=pathway,
        mediation=mediation, feature_dynamics=feat_dyn, mechanism=mech,
    )


# ---------------------------------------------------------------------------
# Top-level runner using the same candidate-selection machinery as M6C.
# ---------------------------------------------------------------------------


def run_m8_for_rule_seed(
    *,
    rule, rule_id: str, rule_source: str, seed: int,
    grid_shape, timesteps, max_candidates: int, horizons: list[int],
    n_replicates: int, backend: str, workdir, progress=None,
) -> list[M8CandidateResult]:
    """Simulate one (rule, seed), select top-K candidates, run full
    M8 measurement on each."""
    from observer_worlds.detection import GreedyTracker
    from observer_worlds.experiments._pipeline import (
        compute_full_metrics, detect_and_track, simulate_4d_to_zarr,
    )
    from observer_worlds.metrics import score_persistence
    from observer_worlds.storage import ZarrRunStore
    from observer_worlds.utils import seeded_rng
    from observer_worlds.utils.config import (
        DetectionConfig, OutputConfig, ProjectionConfig, RunConfig, WorldConfig,
    )

    bs = rule.to_bsrule()
    cfg = RunConfig(
        world=WorldConfig(
            nx=grid_shape[0], ny=grid_shape[1], nz=grid_shape[2], nw=grid_shape[3],
            timesteps=timesteps, initial_density=rule.initial_density,
            rule_birth=tuple(int(x) for x in bs.birth),
            rule_survival=tuple(int(x) for x in bs.survival),
            backend=backend,
        ),
        projection=ProjectionConfig(method="mean_threshold", theta=0.5),
        detection=DetectionConfig(),
        output=OutputConfig(save_4d_snapshots=True,
                            snapshot_interval=max(1, timesteps // 6)),
        seed=seed,
        label=f"m8_{rule_id}_seed{seed}",
    )
    rundir = workdir / f"{rule_id}_seed{seed}"
    rundir.mkdir(parents=True, exist_ok=True)
    store = ZarrRunStore(
        rundir, timesteps=timesteps,
        shape_2d=(grid_shape[0], grid_shape[1]),
        save_4d_snapshots=True, shape_4d=grid_shape,
    )
    rng_init = seeded_rng(seed)
    simulate_4d_to_zarr(cfg, store, rng_init)
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)
    candidates = score_persistence(
        tracks, grid_shape=(grid_shape[0], grid_shape[1]), config=cfg.detection
    )
    observer_scores, _ = compute_full_metrics(
        cfg, tracks, candidates, store, rollout_steps=4, world_kind="4d",
    )
    snap_times = store.list_snapshots()

    score_by_id = {o.track_id: o for o in observer_scores}
    track_by_id = {t.track_id: t for t in tracks}
    ranked = sorted(observer_scores, key=lambda o: -o.combined)

    results: list[M8CandidateResult] = []
    for idx, obs in enumerate(ranked):
        if len(results) >= max_candidates:
            break
        tr = track_by_id.get(obs.track_id)
        if tr is None: continue
        snap_t = None
        for st in reversed(snap_times):
            if tr.birth_frame <= st <= tr.last_frame:
                snap_t = st; break
        if snap_t is None: continue
        if snap_t in tr.frames: i = tr.frames.index(snap_t)
        else:
            nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
            i = tr.frames.index(nearest)
        interior = tr.interior_history[i]
        if not interior.any(): interior = tr.mask_history[i]
        if not interior.any() or int(interior.sum()) < 4:
            continue
        if progress:
            progress(f"    cand {len(results)+1}/{max_candidates} "
                     f"track={tr.track_id} snap_t={snap_t} "
                     f"interior={int(interior.sum())}")
        snapshot_4d = store.read_snapshot_4d(snap_t)
        res = measure_candidate_m8(
            snapshot_4d=snapshot_4d, rule=bs, interior_mask=interior,
            rule_id=rule_id, rule_source=rule_source, seed=seed,
            candidate_id=tr.track_id, snapshot_t=snap_t,
            candidate_area=float(np.mean(tr.area_history)) if tr.area_history else 0.0,
            candidate_lifetime=int(tr.age),
            observer_score=float(obs.combined),
            horizons=horizons, n_replicates=n_replicates,
            backend=backend, rng_seed=seed * 101 + tr.track_id * 31 + idx,
        )
        results.append(res)
    return results


def _run_m8_for_parallel(
    item: tuple, shared: dict
) -> list[M8CandidateResult]:
    """Worker dispatch for ``parallel_sweep``.

    Unpacks ``(rule, rule_id, rule_source, seed)`` and calls
    ``run_m8_for_rule_seed`` with the per-sweep ``shared`` params. Catches
    per-task exceptions and logs them, returning ``[]`` so a single bad
    (rule, seed) doesn't sink the whole sweep — matches the original
    serial behaviour.
    """
    import sys
    rule, rule_id, rule_source, seed = item
    try:
        return run_m8_for_rule_seed(
            rule=rule, rule_id=rule_id, rule_source=rule_source,
            seed=seed,
            grid_shape=shared["grid_shape"],
            timesteps=shared["timesteps"],
            max_candidates=shared["max_candidates"],
            horizons=shared["horizons"],
            n_replicates=shared["n_replicates"],
            backend=shared["backend"],
            workdir=shared["workdir"],
            progress=None,
        )
    except Exception as e:
        print(
            f"    m8 error rule={rule_id} src={rule_source} seed={seed}: {e}",
            file=sys.stderr,
        )
        return []


class _M8ParallelTask:
    """Picklable task dispatcher for ``parallel_sweep``.

    joblib loky pickles the callable by reference (qualified name) and
    re-imports it in the worker; nested closures aren't picklable. This
    class binds the per-sweep ``shared`` dict to a top-level callable.
    """

    def __init__(self, shared: dict) -> None:
        self.shared = shared

    def __call__(self, item: tuple) -> list[M8CandidateResult]:
        return _run_m8_for_parallel(item, self.shared)


def run_m8_mechanism_discovery(
    *,
    rules: list[tuple],          # (rule, rule_id, rule_source)
    seeds: list[int],
    grid_shape, timesteps,
    max_candidates: int, horizons: list[int],
    n_replicates: int, backend: str,
    workdir, progress=None,
    n_workers: int | None = None,
) -> list[M8CandidateResult]:
    """Driver. Returns a flat list of ``M8CandidateResult`` across all
    (rule, seed, candidate).

    Work is parallelized over (rule, seed) tuples via ``parallel_sweep``.
    Default ``n_workers`` is ``cpu_count - 2``; on machines with <= 2
    cores this resolves to 1 and the serial path is used. Pass
    ``n_workers=1`` explicitly to force serial.
    """
    from pathlib import Path

    from observer_worlds.parallel import parallel_sweep

    items: list[tuple] = [
        (rule, rule_id, rule_source, seed)
        for rule, rule_id, rule_source in rules
        for seed in seeds
    ]

    shared = {
        "grid_shape": grid_shape,
        "timesteps": timesteps,
        "max_candidates": max_candidates,
        "horizons": horizons,
        "n_replicates": n_replicates,
        "backend": backend,
        # Normalize to Path so workers (and the serial path) can do
        # ``workdir / f"{rule_id}_seed{seed}"`` even if a string was
        # passed in.
        "workdir": Path(workdir),
    }

    # Forward live per-task lines to stderr when a progress callback is
    # provided -- otherwise long sweeps go silent for hours.
    verbose = 10 if progress is not None else 0

    t0 = time.time()
    flat_results = parallel_sweep(
        items,
        _M8ParallelTask(shared),
        n_workers=n_workers,
        progress=progress,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    if progress is not None:
        progress(
            f"  m8 sweep wall time {elapsed:.0f}s "
            f"({len(items)} runs across {len(rules)} rules x "
            f"{len(seeds)} seeds)"
        )

    out: list[M8CandidateResult] = [
        res for sublist in flat_results for res in sublist
    ]
    return out
