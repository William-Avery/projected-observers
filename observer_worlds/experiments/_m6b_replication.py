"""M6B — replication core.

Multi-rule × multi-seed × multi-candidate × multi-intervention ×
multi-horizon × multi-replicate runner. Records the right primary
quantities so HCE-as-ratio cannot dominate interpretation when
``initial_projection_delta == 0`` (which is true by construction for
every hidden-invisible intervention).

Each row in the output is one (rule, seed, candidate, condition,
intervention_type, replicate, horizon) measurement.

Candidates are sampled per (rule, seed) by three modes:
  * ``"top_observer"``: highest combined observer_score
  * ``"top_lifetime"``: largest track age
  * ``"random_eligible"``: uniform sample from eligible candidates

Eligible = persistence-filter candidate with non-empty interior fiber.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from observer_worlds.detection import GreedyTracker, classify_boundary, extract_components
from observer_worlds.experiments._m4b_sweep import hidden_shuffle_mutator
from observer_worlds.experiments._m6b_interventions import (
    apply_far_hidden_intervention,
    apply_fiber_replacement_intervention,
    apply_one_time_scramble_intervention,
    apply_sham_intervention,
    build_far_mask,
)
from observer_worlds.experiments._pipeline import (
    compute_full_metrics,
    detect_and_track,
    simulate_4d_to_zarr,
)
from observer_worlds.metrics import score_persistence
from observer_worlds.metrics.causality_score import (
    apply_flip_intervention,
    apply_hidden_shuffle_intervention,
)
from observer_worlds.search import FractionalRule
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import RunConfig, seeded_rng
from observer_worlds.utils.config import (
    DetectionConfig,
    OutputConfig,
    ProjectionConfig,
    WorldConfig,
)
from observer_worlds.worlds import CA4D, BSRule, project


# ---------------------------------------------------------------------------
# Conditions and intervention types
# ---------------------------------------------------------------------------


# `condition` describes which simulation produced the snapshot.
CONDITIONS_M6B: tuple[str, ...] = (
    "coherent_4d",
    "per_step_hidden_shuffled_4d",
)


# `intervention_type` describes what perturbation is applied at the snapshot.
# Within each condition, every intervention is applied (subject to feasibility).
INTERVENTION_TYPES_M6B: tuple[str, ...] = (
    "sham",
    "hidden_invisible_local",
    "one_time_scramble_local",
    "fiber_replacement_local",
    "hidden_invisible_far",
    "visible_match_count",
)


# Selection modes for candidates per (rule, seed).
CANDIDATE_SELECTION_MODES: tuple[str, ...] = (
    "top_observer",
    "top_lifetime",
    "random_eligible",
)


EPSILON_HCE = 1e-6


# ---------------------------------------------------------------------------
# Per-row measurement
# ---------------------------------------------------------------------------


@dataclass
class M6BRow:
    """One paired-rollout measurement at one horizon under one
    intervention. Mirrors the spec's CSV column list."""

    rule_id: str
    rule_source: str
    seed: int
    candidate_id: int
    candidate_selection_mode: str
    condition: str
    intervention_type: str
    replicate: int
    horizon: int

    initial_projection_delta: float = 0.0
    future_projection_divergence: float = 0.0     # full-grid L1 at horizon
    local_future_divergence: float = 0.0          # candidate-footprint L1 at horizon
    global_future_divergence: float = 0.0         # alias for future_projection_divergence
    hidden_causal_dependence: float = 0.0         # future / (initial + epsilon)
    hidden_vs_visible_ratio: float = 0.0          # future / visible_future (per replicate)
    hidden_vs_sham_delta: float = 0.0             # future - sham_future
    hidden_vs_far_delta: float = 0.0              # local_future - far_local_future
    survival_original: float = 0.0                # active cells in interior at horizon (unperturbed)
    survival_intervened: float = 0.0              # active cells in interior at horizon (intervened)
    survival_delta: float = 0.0
    trajectory_divergence: float = 0.0            # AUC of full-grid L1 over horizon
    recovery_delta: float = 0.0                   # local_future divergence at last step minus min over horizon

    candidate_area: float = 0.0
    candidate_lifetime: int = 0
    observer_score: float | None = None
    morphology_class: str = "unspecified"

    n_flips_applied: int = 0
    flip_fraction_for_visible: float = 0.0
    snapshot_t: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_2d(state: np.ndarray, theta: float = 0.5) -> np.ndarray:
    return project(state, method="mean_threshold", theta=theta)


def _l1_full(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum() / a.size)


def _l1_local(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff[mask].sum() / mask.sum())


def _rollout(
    state: np.ndarray, rule: BSRule, n_steps: int, *,
    backend: str, projection_theta: float,
) -> np.ndarray:
    """Run a CA4D rollout from ``state`` for n_steps; return projected
    frames of shape (n_steps, Nx, Ny). Step 0 is post-step-1."""
    Nx, Ny = state.shape[0], state.shape[1]
    ca = CA4D(shape=state.shape, rule=rule, backend=backend)
    ca.state = state.copy()
    out = np.empty((n_steps, Nx, Ny), dtype=np.uint8)
    for t in range(n_steps):
        ca.step()
        out[t] = _project_2d(ca.state, projection_theta)
    return out


def _safe_int(rng: np.random.Generator, hi: int) -> int:
    if hi <= 0:
        return 0
    return int(rng.integers(0, hi))


# ---------------------------------------------------------------------------
# Per-candidate runner: applies all interventions × all replicates × all horizons.
# ---------------------------------------------------------------------------


def _run_one_candidate(
    *,
    snapshot_4d: np.ndarray,
    rule: BSRule,
    interior_mask: np.ndarray,
    rule_id: str,
    rule_source: str,
    seed: int,
    candidate_id: int,
    candidate_selection_mode: str,
    condition: str,
    snapshot_t: int,
    candidate_area: float,
    candidate_lifetime: int,
    observer_score: float | None,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    rng_seed: int,
    projection_theta: float = 0.5,
) -> list[M6BRow]:
    """For one candidate at one snapshot, run all interventions × all
    replicates × all horizons in one shot.  Per replicate, runs
    max(horizons) steps once per intervention and reads off all horizons.
    Returns one M6BRow per (intervention, replicate, horizon).
    """
    if not interior_mask.any():
        return []

    parent_rng = np.random.default_rng(rng_seed)
    Nx, Ny = snapshot_4d.shape[0], snapshot_4d.shape[1]
    Nz, Nw = snapshot_4d.shape[2], snapshot_4d.shape[3]
    H_max = max(horizons)
    proj0 = _project_2d(snapshot_4d, projection_theta)
    interior_size = int(interior_mask.sum())

    # Far-mask is fixed per snapshot (deterministic from interior_mask).
    far_mask = build_far_mask(interior_mask, Nx=Nx, Ny=Ny)

    # Unperturbed rollout reused as "sham original" reference.
    frames_orig = _rollout(snapshot_4d, rule, H_max,
                           backend=backend, projection_theta=projection_theta)

    # Pre-compute survival_original at each horizon (active cells in interior).
    survival_orig = np.array([
        int((frames_orig[h - 1].astype(bool) & interior_mask).sum()) for h in horizons
    ], dtype=np.float64)

    rows: list[M6BRow] = []

    # We need the visible_match_count per replicate to match the n_flips of
    # hidden_invisible_local (the canonical hidden perturbation). Compute
    # those flips up front per replicate so visible_match_count is matched.

    # Per replicate, generate intervention states ONCE and reuse across horizons.
    for rep in range(n_replicates):
        rep_rng = np.random.default_rng(int(parent_rng.integers(0, 2**63 - 1)))

        # First: hidden_invisible_local. Determines the bit-flip count we'll
        # use for visible_match_count.
        rng_h = np.random.default_rng(int(rep_rng.integers(0, 2**63 - 1)))
        state_hi = apply_hidden_shuffle_intervention(snapshot_4d, interior_mask, rng_h)
        n_flips_hi = int(((state_hi != snapshot_4d) &
                          interior_mask[:, :, None, None]).sum())
        # initial projection delta (should be 0 by invariant for all hidden interventions)
        initial_delta_hi = _l1_full(proj0, _project_2d(state_hi, projection_theta))

        rng_s = np.random.default_rng(int(rep_rng.integers(0, 2**63 - 1)))
        state_scramble = apply_one_time_scramble_intervention(
            snapshot_4d, interior_mask, rng_s
        )
        n_flips_scramble = int(((state_scramble != snapshot_4d) &
                                interior_mask[:, :, None, None]).sum())
        initial_delta_scramble = _l1_full(proj0, _project_2d(state_scramble, projection_theta))

        rng_f = np.random.default_rng(int(rep_rng.integers(0, 2**63 - 1)))
        state_fiber = apply_fiber_replacement_intervention(
            snapshot_4d, interior_mask, rng_f
        )
        n_flips_fiber = int(((state_fiber != snapshot_4d) &
                             interior_mask[:, :, None, None]).sum())
        initial_delta_fiber = _l1_full(proj0, _project_2d(state_fiber, projection_theta))

        rng_far = np.random.default_rng(int(rep_rng.integers(0, 2**63 - 1)))
        state_far, _ = apply_far_hidden_intervention(snapshot_4d, interior_mask, rng_far)
        # NB: far perturbation flips bits in far_mask cells; "initial delta" still
        # measures full-grid projection (which should be ~0 since shuffle preserves
        # projection in those columns too).
        initial_delta_far = _l1_full(proj0, _project_2d(state_far, projection_theta))

        # Visible: match n_flips_hi.
        rng_vis = np.random.default_rng(int(rep_rng.integers(0, 2**63 - 1)))
        total_interior_cells = max(interior_size * Nz * Nw, 1)
        flip_fraction = float(n_flips_hi) / total_interior_cells
        state_vis = apply_flip_intervention(
            snapshot_4d, interior_mask, flip_fraction, rng_vis
        )
        initial_delta_vis = _l1_full(proj0, _project_2d(state_vis, projection_theta))

        rng_sham = np.random.default_rng(int(rep_rng.integers(0, 2**63 - 1)))
        state_sham = apply_sham_intervention(snapshot_4d, interior_mask, rng_sham)
        # sham initial delta is exactly 0 (identity).

        # Roll out each.
        frames_by_intv: dict[str, np.ndarray] = {
            "sham": _rollout(state_sham, rule, H_max,
                             backend=backend, projection_theta=projection_theta),
            "hidden_invisible_local": _rollout(state_hi, rule, H_max,
                                               backend=backend, projection_theta=projection_theta),
            "one_time_scramble_local": _rollout(state_scramble, rule, H_max,
                                                backend=backend, projection_theta=projection_theta),
            "fiber_replacement_local": _rollout(state_fiber, rule, H_max,
                                                backend=backend, projection_theta=projection_theta),
            "hidden_invisible_far": _rollout(state_far, rule, H_max,
                                             backend=backend, projection_theta=projection_theta),
            "visible_match_count": _rollout(state_vis, rule, H_max,
                                            backend=backend, projection_theta=projection_theta),
        }
        initial_deltas = {
            "sham": 0.0,
            "hidden_invisible_local": initial_delta_hi,
            "one_time_scramble_local": initial_delta_scramble,
            "fiber_replacement_local": initial_delta_fiber,
            "hidden_invisible_far": initial_delta_far,
            "visible_match_count": initial_delta_vis,
        }
        n_flips_per_intv = {
            "sham": 0,
            "hidden_invisible_local": n_flips_hi,
            "one_time_scramble_local": n_flips_scramble,
            "fiber_replacement_local": n_flips_fiber,
            "hidden_invisible_far": n_flips_hi,
            "visible_match_count": n_flips_hi,
        }

        # Build rows per (intervention, horizon).
        for intv, frames_int in frames_by_intv.items():
            # Per-step full-grid L1 (recomputed for trajectory_divergence and recovery).
            full_l1_traj = np.array([
                _l1_full(frames_orig[t], frames_int[t]) for t in range(H_max)
            ], dtype=np.float64)
            local_l1_traj = np.array([
                _l1_local(frames_orig[t], frames_int[t], interior_mask)
                for t in range(H_max)
            ], dtype=np.float64)
            # Lookups for cross-intervention deltas (use sham + far + visible).
            sham_full = np.array([
                _l1_full(frames_orig[t], frames_by_intv["sham"][t]) for t in range(H_max)
            ], dtype=np.float64)
            far_local = np.array([
                _l1_local(frames_orig[t], frames_by_intv["hidden_invisible_far"][t],
                          interior_mask)
                for t in range(H_max)
            ], dtype=np.float64)
            vis_full = np.array([
                _l1_full(frames_orig[t], frames_by_intv["visible_match_count"][t])
                for t in range(H_max)
            ], dtype=np.float64)

            for h_idx, h in enumerate(horizons):
                idx = h - 1  # frames_orig[0] is post-step-1
                future_full = float(full_l1_traj[idx])
                future_local = float(local_l1_traj[idx])
                survival_int = int((frames_int[idx].astype(bool) & interior_mask).sum())
                trajectory_divergence = float(full_l1_traj[: idx + 1].sum())
                # recovery_delta: how much did local divergence "come back down"
                # relative to its peak by horizon h?  positive = candidate
                # is recovering, negative = still escalating.
                if idx > 0:
                    peak = float(local_l1_traj[: idx + 1].max())
                    recovery = peak - float(local_l1_traj[idx])
                else:
                    recovery = 0.0
                vis_at_h = float(vis_full[idx])
                hi_vs_vis = future_full / vis_at_h if vis_at_h > 1e-12 else 0.0
                vs_sham = future_full - float(sham_full[idx])
                vs_far = future_local - float(far_local[idx])
                init_delta = initial_deltas[intv]
                hce = future_full / (init_delta + EPSILON_HCE)

                rows.append(M6BRow(
                    rule_id=rule_id, rule_source=rule_source, seed=seed,
                    candidate_id=candidate_id,
                    candidate_selection_mode=candidate_selection_mode,
                    condition=condition,
                    intervention_type=intv, replicate=rep, horizon=h,
                    initial_projection_delta=init_delta,
                    future_projection_divergence=future_full,
                    local_future_divergence=future_local,
                    global_future_divergence=future_full,
                    hidden_causal_dependence=float(hce),
                    hidden_vs_visible_ratio=hi_vs_vis,
                    hidden_vs_sham_delta=vs_sham,
                    hidden_vs_far_delta=vs_far,
                    survival_original=float(survival_orig[h_idx]),
                    survival_intervened=float(survival_int),
                    survival_delta=float(survival_int - survival_orig[h_idx]),
                    trajectory_divergence=trajectory_divergence,
                    recovery_delta=recovery,
                    candidate_area=candidate_area,
                    candidate_lifetime=candidate_lifetime,
                    observer_score=observer_score,
                    morphology_class=_morphology_class(candidate_area, Nx * Ny),
                    n_flips_applied=n_flips_per_intv[intv],
                    flip_fraction_for_visible=flip_fraction,
                    snapshot_t=snapshot_t,
                ))
    return rows


def _morphology_class(area: float, grid_size: int) -> str:
    """Coarse area-based morphology bucket. Useful for the candidate-property
    correlations the spec asks for."""
    frac = area / max(grid_size, 1)
    if frac < 0.005:
        return "tiny"
    if frac < 0.02:
        return "small"
    if frac < 0.10:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _select_candidates(
    *,
    tracks: list,
    observer_scores: list,
    snap_times: list[int],
    max_per_mode: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """Return a list of (track, snapshot_t, mode, observer_score) tuples.

    Sampling rules:
      * top_observer: highest combined observer_score, eligible
      * top_lifetime: largest age, eligible (drawn from all tracks, not
        just observer-scored)
      * random_eligible: uniform sample from all tracks with snapshot
        in lifetime + non-empty interior

    Eligibility = has snapshot in lifetime AND
                  some frame's interior mask is non-empty (with M6's lenient
                  fallback to mask_history if eroded interior is empty).
    """
    track_by_id = {t.track_id: t for t in tracks}

    def _eligible_snapshot(track):
        for st in reversed(snap_times):
            if track.birth_frame <= st <= track.last_frame:
                # Find the closest observed frame.
                if st in track.frames:
                    i = track.frames.index(st)
                else:
                    nearest = min(track.frames, key=lambda f: abs(f - st))
                    i = track.frames.index(nearest)
                interior = track.interior_history[i]
                if not interior.any():
                    interior = track.mask_history[i]
                if interior.any():
                    return st, interior
        return None, None

    selected: list[tuple] = []

    # top_observer
    score_by_id = {o.track_id: o.combined for o in observer_scores}
    by_obs = sorted(observer_scores, key=lambda o: -o.combined)
    n_obs = 0
    for o in by_obs:
        if n_obs >= max_per_mode:
            break
        tr = track_by_id.get(o.track_id)
        if tr is None:
            continue
        st, interior = _eligible_snapshot(tr)
        if st is None:
            continue
        selected.append((tr, st, interior, "top_observer", float(o.combined)))
        n_obs += 1

    # top_lifetime
    by_age = sorted(tracks, key=lambda t: -t.age)
    n_life = 0
    for tr in by_age:
        if n_life >= max_per_mode:
            break
        st, interior = _eligible_snapshot(tr)
        if st is None:
            continue
        selected.append((tr, st, interior, "top_lifetime",
                         float(score_by_id.get(tr.track_id, 0.0))
                         if tr.track_id in score_by_id else None))
        n_life += 1

    # random_eligible (uniform over eligible tracks)
    eligible = []
    for tr in tracks:
        st, interior = _eligible_snapshot(tr)
        if st is not None:
            eligible.append((tr, st, interior))
    if eligible:
        rng.shuffle(eligible)
        for tr, st, interior in eligible[:max_per_mode]:
            selected.append((tr, st, interior, "random_eligible",
                             float(score_by_id.get(tr.track_id, 0.0))
                             if tr.track_id in score_by_id else None))
    return selected


# ---------------------------------------------------------------------------
# Per-(rule, seed) runner: simulate, detect, score, sample candidates, run
# all interventions for each candidate.
# ---------------------------------------------------------------------------


def _build_run_config_for_rule(
    rule: FractionalRule, *, grid_shape, timesteps, backend, seed, label,
) -> RunConfig:
    bs = rule.to_bsrule()
    return RunConfig(
        world=WorldConfig(
            nx=grid_shape[0], ny=grid_shape[1], nz=grid_shape[2], nw=grid_shape[3],
            timesteps=timesteps,
            initial_density=rule.initial_density,
            rule_birth=tuple(int(x) for x in bs.birth),
            rule_survival=tuple(int(x) for x in bs.survival),
            backend=backend,
        ),
        projection=ProjectionConfig(method="mean_threshold", theta=0.5),
        detection=DetectionConfig(),
        output=OutputConfig(save_4d_snapshots=True,
                            snapshot_interval=max(1, timesteps // 6)),
        seed=seed,
        label=label,
    )


def run_replication_for_rule_seed(
    *,
    rule: FractionalRule,
    rule_id: str,
    rule_source: str,
    seed: int,
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    max_candidates_per_mode: int,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    condition: str = "coherent_4d",
    workdir_for_zarr=None,
    progress: Callable[[str], None] | None = None,
) -> list[M6BRow]:
    """Simulate one (rule, seed), select candidates, run all interventions
    for each candidate. Returns flat list of M6BRow."""
    cfg = _build_run_config_for_rule(
        rule, grid_shape=grid_shape, timesteps=timesteps,
        backend=backend, seed=seed,
        label=f"m6b_{rule_id}_seed{seed}_{condition}",
    )

    # Use a temporary in-memory-style ZarrRunStore. Allocate a tmp dir.
    import tempfile
    if workdir_for_zarr is None:
        workdir = tempfile.mkdtemp(prefix="m6b_")
    else:
        from pathlib import Path
        workdir = Path(workdir_for_zarr) / f"{rule_id}_seed{seed}_{condition}"
        workdir.mkdir(parents=True, exist_ok=True)
    store = ZarrRunStore(
        workdir, timesteps=timesteps,
        shape_2d=(grid_shape[0], grid_shape[1]),
        save_4d_snapshots=True, shape_4d=grid_shape,
    )
    rng_init = seeded_rng(seed)
    if condition == "per_step_hidden_shuffled_4d":
        mut_rng = np.random.default_rng(seed * 7919 + 1)
        def _mutator(state, t, _rng):
            return hidden_shuffle_mutator(state, t, mut_rng)
        simulate_4d_to_zarr(cfg, store, rng_init, state_mutator=_mutator)
    else:
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

    rng_sel = np.random.default_rng(seed * 31)
    selected = _select_candidates(
        tracks=tracks, observer_scores=observer_scores,
        snap_times=snap_times, max_per_mode=max_candidates_per_mode,
        rng=rng_sel,
    )
    rule_4d_bs = BSRule(birth=cfg.world.rule_birth, survival=cfg.world.rule_survival)

    rows: list[M6BRow] = []
    for cand_idx, (tr, snap_t, interior, mode, obs_score) in enumerate(selected):
        snapshot_4d = store.read_snapshot_4d(snap_t)
        if progress is not None:
            progress(f"    candidate {cand_idx+1}/{len(selected)} "
                     f"track={tr.track_id} mode={mode} interior={int(interior.sum())}")
        cand_rows = _run_one_candidate(
            snapshot_4d=snapshot_4d, rule=rule_4d_bs,
            interior_mask=interior,
            rule_id=rule_id, rule_source=rule_source, seed=seed,
            candidate_id=tr.track_id,
            candidate_selection_mode=mode,
            condition=condition,
            snapshot_t=snap_t,
            candidate_area=float(np.mean(tr.area_history)) if tr.area_history else 0.0,
            candidate_lifetime=int(tr.age),
            observer_score=obs_score,
            horizons=list(horizons), n_replicates=n_replicates,
            backend=backend,
            rng_seed=seed * 1009 + tr.track_id * 31 + cand_idx,
        )
        rows.extend(cand_rows)

    # Cleanup (we keep the zarr stores as scratch; the user's outputs/m6b_*
    # dir points to the experiment-level CSVs not these per-rule simulation dirs).
    return rows


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _run_replication_for_parallel(
    item: tuple[FractionalRule, str, str, int, str], shared: dict
) -> list[M6BRow]:
    rule, rule_id, rule_source, seed, condition = item
    return run_replication_for_rule_seed(
        rule=rule, rule_id=rule_id, rule_source=rule_source,
        seed=seed,
        grid_shape=shared["grid_shape"],
        timesteps=shared["timesteps"],
        max_candidates_per_mode=shared["max_candidates_per_mode"],
        horizons=shared["horizons"],
        n_replicates=shared["n_replicates"],
        backend=shared["backend"],
        condition=condition,
        workdir_for_zarr=shared["workdir_for_zarr"],
        progress=None,
    )


class _M6BParallelTask:
    """Picklable task dispatcher for parallel_sweep.

    joblib loky pickles the callable by reference (qualified name) and
    re-imports it in the worker; nested closures aren't picklable. This
    class binds the per-sweep ``shared`` dict to a top-level callable.
    """

    def __init__(self, shared: dict) -> None:
        self.shared = shared

    def __call__(
        self, item: tuple[FractionalRule, str, str, int, str]
    ) -> list[M6BRow]:
        return _run_replication_for_parallel(item, self.shared)


def run_m6b_replication(
    *,
    rules: list[tuple[FractionalRule, str, str]],   # (rule, rule_id, rule_source)
    seeds: list[int],
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    max_candidates_per_mode: int,
    horizons: list[int],
    n_replicates: int,
    backend: str,
    include_per_step_shuffled: bool = True,
    workdir_for_zarr=None,
    progress: Callable[[str], None] | None = None,
    n_workers: int | None = None,
) -> list[M6BRow]:
    """Driver. Returns a flat list of M6BRow across all
    (rule, seed, condition, candidate, intervention, replicate, horizon).

    Work is parallelized over (rule, seed, condition) tuples via
    ``parallel_sweep``. Default ``n_workers`` is ``cpu_count - 2``; on
    machines with <= 2 cores this resolves to 1 and the serial path is
    used. Pass ``n_workers=1`` explicitly to force serial.
    """
    from observer_worlds.parallel import parallel_sweep

    conditions = ["coherent_4d"]
    if include_per_step_shuffled:
        conditions.append("per_step_hidden_shuffled_4d")

    items: list[tuple[FractionalRule, str, str, int, str]] = [
        (rule, rule_id, rule_source, seed, condition)
        for rule, rule_id, rule_source in rules
        for seed in seeds
        for condition in conditions
    ]

    shared = {
        "grid_shape": grid_shape,
        "timesteps": timesteps,
        "max_candidates_per_mode": max_candidates_per_mode,
        "horizons": horizons,
        "n_replicates": n_replicates,
        "backend": backend,
        "workdir_for_zarr": workdir_for_zarr,
    }

    # Forward live per-task lines to stderr when a progress callback is
    # provided -- otherwise long sweeps go silent for hours.
    verbose = 10 if progress is not None else 0

    t0 = time.time()
    flat_results = parallel_sweep(
        items,
        _M6BParallelTask(shared),
        n_workers=n_workers,
        progress=progress,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    if progress is not None:
        progress(
            f"  m6b sweep wall time {elapsed:.0f}s "
            f"({len(items)} runs across {len(rules)} rules x "
            f"{len(seeds)} seeds x {len(conditions)} conditions)"
        )

    rows: list[M6BRow] = [row for sublist in flat_results for row in sublist]
    return rows
