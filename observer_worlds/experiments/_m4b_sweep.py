"""M4B observer-metric sweep core.

Runs paired (rule, seed) evaluations across three conditions:

    A. coherent_4d   -- the 4D CA evolved normally
    B. shuffled_4d   -- the 4D CA where z,w fibers are permuted into ca.state
                        every step (the projection-only no-op was M3's bug)
    C. matched_2d    -- a 2D CA baseline (Conway's Life by default)

Each condition is run **in memory** (no per-run zarr store) so 300+ runs
fit in a reasonable wall-clock budget.  Snapshots needed for causality /
resilience are kept in a per-condition dict.

For every (rule, seed) the sweep produces three :class:`ConditionResult`
objects bundled into a :class:`PairedRecord`.  The recorded summaries
include many track-count-aware aggregates (top-K, percentiles,
lifetime-weighted mean) so that comparisons aren't confounded by the fact
that the shuffled condition often produces more tracks than the coherent
one.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from observer_worlds.detection import (
    GreedyTracker,
    classify_boundary,
    extract_components,
)
from observer_worlds.metrics import (
    collect_raw_scores,
    compute_causality_score,
    compute_memory_score,
    compute_observer_scores,
    compute_resilience_score,
    compute_selfhood_score,
    compute_time_score,
    extract_track_features,
    score_persistence,
)
from observer_worlds.search.rules import FractionalRule
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import CA2D, CA4D, BSRule, project


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


CONDITION_NAMES: tuple[str, str, str] = ("coherent_4d", "shuffled_4d", "matched_2d")


@dataclass
class ConditionResult:
    """All numerical outputs for a single (rule, seed, condition) run."""

    rule_idx: int
    seed: int
    condition: str
    rule_dict: dict           # FractionalRule.to_dict() (4D conditions); a stub for 2D

    # Run-level diagnostics.
    n_tracks: int = 0
    n_candidates: int = 0
    mean_active: float = 0.0
    late_active: float = 0.0
    activity_variance: float = 0.0
    mean_frame_to_frame_change: float = 0.0
    max_component_lifetime: int = 0
    mean_component_lifetime: float = 0.0

    # Observer-score aggregates.
    max_score: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    p90_score: float = 0.0
    p95_score: float = 0.0
    p99_score: float = 0.0
    top3_mean_score: float = 0.0
    top5_mean_score: float = 0.0
    top10_mean_score: float = 0.0
    lifetime_weighted_mean_score: float = 0.0
    area_weighted_mean_score: float = 0.0
    score_per_track: float = 0.0  # sum(scores) / max(num_tracks, 1)

    # Best-candidate per-component breakdown (None if missing).
    best_track_id: int | None = None
    best_age: int | None = None
    best_persistence: float | None = None
    best_time_score: float | None = None
    best_memory_score: float | None = None
    best_selfhood_score: float | None = None
    best_causality_score: float | None = None
    best_resilience_score: float | None = None

    # All combined scores (for percentile + bootstrap downstream).
    all_combined_scores: list[float] = field(default_factory=list)
    # Per-candidate ages and areas for area/lifetime-weighted aggregates.
    all_ages: list[int] = field(default_factory=list)
    all_mean_areas: list[float] = field(default_factory=list)

    # Frame-sequence hash (sha1 of first 32 frames concatenated) for
    # regression-checking that conditions actually differ.
    projected_hash: str = ""

    # Captured frames (kept for video extraction; only the first
    # ``video_frames_kept`` are retained to bound memory).
    frames_for_video: np.ndarray | None = None

    sim_time_seconds: float = 0.0
    metric_time_seconds: float = 0.0


@dataclass
class PairedRecord:
    """All three conditions for one (rule, seed) pair."""

    rule_idx: int
    seed: int
    rule_dict: dict
    coherent_4d: ConditionResult
    shuffled_4d: ConditionResult
    matched_2d: ConditionResult


# ---------------------------------------------------------------------------
# Hidden-shuffle mutator (writes back into ca.state through _pipeline,
# but for in-memory pipelines we apply it directly).
# ---------------------------------------------------------------------------


def hidden_shuffle_mutator(
    state: np.ndarray, t: int, rng: np.random.Generator, *, shuffle_every: int = 1
) -> np.ndarray:
    """Vectorized z,w permutation per (x,y) column. Preserves per-column counts."""
    if shuffle_every > 1 and (t % shuffle_every != 0):
        return state
    Nx, Ny, Nz, Nw = state.shape
    flat = state.reshape(Nx * Ny, Nz * Nw)
    keys = rng.random(flat.shape)
    order = np.argsort(keys, axis=1)
    out = np.take_along_axis(flat, order, axis=1)
    return out.reshape(Nx, Ny, Nz, Nw).astype(state.dtype, copy=False)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _simulate_4d_capturing(
    rule: BSRule,
    *,
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    initial_density: float,
    seed: int,
    backend: str,
    projection_method: str,
    projection_theta: float,
    snapshot_at: list[int],
    state_mutator: Callable | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Simulate a 4D CA in memory, optionally with a state-mutator that
    writes back into ``ca.state``.  Returns ``(frames_2d, snapshots,
    active_history)``.
    """
    rng = seeded_rng(seed)
    ca = CA4D(shape=grid_shape, rule=rule, backend=backend)
    ca.initialize_random(density=initial_density, rng=rng)

    Nx, Ny = grid_shape[0], grid_shape[1]
    frames = np.empty((timesteps, Nx, Ny), dtype=np.uint8)
    active = np.empty(timesteps, dtype=np.float32)
    snapshots: dict[int, np.ndarray] = {}

    snap_set = set(int(t) for t in snapshot_at)

    def _project(state: np.ndarray) -> np.ndarray:
        return project(state, method=projection_method, theta=projection_theta)

    # t=0
    if state_mutator is not None:
        ca.state = state_mutator(ca.state.copy(), 0, rng)
    Y0 = _project(ca.state)
    frames[0] = Y0
    active[0] = float(Y0.mean())
    if 0 in snap_set:
        snapshots[0] = ca.state.copy()

    for t in range(1, timesteps):
        ca.step()
        if state_mutator is not None:
            ca.state = state_mutator(ca.state.copy(), t, rng)
        Y = _project(ca.state)
        frames[t] = Y
        active[t] = float(Y.mean())
        if t in snap_set:
            snapshots[t] = ca.state.copy()

    return frames, snapshots, active


def _simulate_2d_capturing(
    rule_2d: BSRule,
    *,
    grid_shape: tuple[int, int],
    timesteps: int,
    initial_density: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = seeded_rng(seed)
    ca = CA2D(shape=grid_shape, rule=rule_2d)
    ca.initialize_random(density=initial_density, rng=rng)
    Nx, Ny = grid_shape
    frames = np.empty((timesteps, Nx, Ny), dtype=np.uint8)
    active = np.empty(timesteps, dtype=np.float32)
    frames[0] = ca.state
    active[0] = float(ca.state.mean())
    for t in range(1, timesteps):
        ca.step()
        frames[t] = ca.state
        active[t] = float(ca.state.mean())
    return frames, active


# ---------------------------------------------------------------------------
# Metric computation (in memory, no zarr store)
# ---------------------------------------------------------------------------


def _detect_and_track(frames: np.ndarray, detection_config: DetectionConfig) -> list:
    tracker = GreedyTracker(config=detection_config)
    for t in range(frames.shape[0]):
        comps = extract_components(frames[t], frame_idx=t, config=detection_config)
        tracker.update(t, comps)
    return tracker.finalize()


def _compute_metrics_in_memory(
    tracks: list,
    candidates: list,
    grid_shape: tuple[int, int],
    *,
    seed: int,
    snapshots: dict[int, np.ndarray] | None,
    rule_4d: BSRule | None,
    rollout_steps: int,
    backend: str,
    world_kind: str,
):
    """In-memory variant of `_pipeline.compute_full_metrics` that takes a
    snapshot dict instead of a `ZarrRunStore`.

    Returns ``(observer_scores, per_candidate)`` exactly like the pipeline
    helper.
    """
    candidate_ids = {c.track_id for c in candidates if c.is_candidate}
    track_by_id = {t.track_id: t for t in tracks}
    cand_tracks = [track_by_id[i] for i in candidate_ids if i in track_by_id]
    raw_per_track: list[dict] = []
    per_candidate: dict[int, dict] = {}

    available_snapshot_times = sorted(snapshots.keys()) if snapshots else []
    for tr in cand_tracks:
        feats = extract_track_features(tr)
        time_res = compute_time_score(feats, seed=seed)
        mem_res = compute_memory_score(feats, seed=seed)
        self_res = compute_selfhood_score(feats, seed=seed)
        bnd_res = classify_boundary(feats)

        causal_res = None
        resil_res = None
        if world_kind == "4d" and snapshots and rule_4d is not None:
            # Pick the latest snapshot inside the track's lifetime.
            snap_t = None
            for t in reversed(available_snapshot_times):
                if tr.birth_frame <= t <= tr.last_frame:
                    snap_t = t
                    break
            if snap_t is not None:
                # Mask lookup at snapshot frame (or nearest observed).
                if snap_t in tr.frames:
                    i = tr.frames.index(snap_t)
                else:
                    nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
                    i = tr.frames.index(nearest)
                interior = tr.interior_history[i]
                boundary = tr.boundary_history[i]
                env = tr.env_history[i]
                if interior.any() and boundary.any() and env.any():
                    try:
                        snapshot_4d = snapshots[snap_t]
                        causal_res = compute_causality_score(
                            snapshot_4d, rule_4d, interior, boundary, env,
                            n_steps=rollout_steps, backend=backend,
                            seed=seed, track_id=tr.track_id,
                        )
                        resil_res = compute_resilience_score(
                            snapshot_4d, rule_4d, interior,
                            n_steps=rollout_steps, backend=backend,
                            seed=seed, track_id=tr.track_id,
                        )
                    except Exception:
                        pass  # Causality is optional in M4B summaries.

        per_candidate[tr.track_id] = {
            "time": time_res, "memory": mem_res, "selfhood": self_res,
            "boundary": bnd_res, "causality": causal_res, "resilience": resil_res,
        }
        raw_per_track.append(collect_raw_scores(
            track_id=tr.track_id,
            time=time_res, memory=mem_res, selfhood=self_res,
            causality=causal_res, resilience=resil_res,
        ))

    observer_scores = compute_observer_scores(raw_per_track)
    return observer_scores, per_candidate


# ---------------------------------------------------------------------------
# Per-condition orchestration
# ---------------------------------------------------------------------------


def _projected_hash(frames: np.ndarray, n_frames: int = 32) -> str:
    """SHA-1 of the first n_frames concatenated. Used to verify conditions
    actually produce different projected sequences."""
    n = min(n_frames, frames.shape[0])
    return hashlib.sha1(frames[:n].tobytes()).hexdigest()


def _summarize_condition(
    result: ConditionResult,
    tracks: list,
    observer_scores: list,
    per_candidate: dict,
    frames: np.ndarray,
    active: np.ndarray,
    *,
    video_frames_kept: int,
) -> None:
    """Populate a ConditionResult from raw outputs.  Mutates ``result``."""
    result.n_tracks = len(tracks)
    candidate_ids = {o.track_id for o in observer_scores}
    result.n_candidates = len(candidate_ids)

    # Activity stats.
    result.mean_active = float(active.mean()) if active.size else 0.0
    half = active.size // 2
    result.late_active = float(active[half:].mean()) if active.size else 0.0
    result.activity_variance = float(active.var()) if active.size else 0.0
    if frames.shape[0] >= 2:
        xor = np.bitwise_xor(frames[1:], frames[:-1])
        result.mean_frame_to_frame_change = float(xor.mean())

    # Component lifetimes.
    if tracks:
        ages = [t.age for t in tracks]
        result.max_component_lifetime = int(max(ages))
        result.mean_component_lifetime = float(np.mean(ages))

    # Observer-score aggregates.
    combined = [o.combined for o in observer_scores]
    result.all_combined_scores = combined
    if combined:
        arr = np.asarray(combined)
        result.max_score = float(arr.max())
        result.mean_score = float(arr.mean())
        result.median_score = float(np.median(arr))
        result.p90_score = float(np.percentile(arr, 90))
        result.p95_score = float(np.percentile(arr, 95))
        result.p99_score = float(np.percentile(arr, 99))
        sorted_desc = np.sort(arr)[::-1]
        result.top3_mean_score = float(sorted_desc[: min(3, arr.size)].mean())
        result.top5_mean_score = float(sorted_desc[: min(5, arr.size)].mean())
        result.top10_mean_score = float(sorted_desc[: min(10, arr.size)].mean())
        result.score_per_track = float(arr.sum() / max(result.n_tracks, 1))

        # Weighted means.
        track_by_id = {t.track_id: t for t in tracks}
        ages = []
        areas = []
        for o in observer_scores:
            tr = track_by_id.get(o.track_id)
            if tr is None:
                ages.append(0); areas.append(0.0)
            else:
                ages.append(int(tr.age))
                areas.append(float(np.mean(tr.area_history)) if tr.area_history else 0.0)
        result.all_ages = ages
        result.all_mean_areas = areas
        ages_arr = np.asarray(ages, dtype=np.float64)
        areas_arr = np.asarray(areas, dtype=np.float64)
        if ages_arr.sum() > 0:
            result.lifetime_weighted_mean_score = float((arr * ages_arr).sum() / ages_arr.sum())
        if areas_arr.sum() > 0:
            result.area_weighted_mean_score = float((arr * areas_arr).sum() / areas_arr.sum())

        # Best candidate.
        best = max(observer_scores, key=lambda o: o.combined)
        result.best_track_id = best.track_id
        tr = track_by_id.get(best.track_id)
        result.best_age = int(tr.age) if tr is not None else None
        breakdown = per_candidate.get(best.track_id, {})

        def _val(res, attr):
            if res is None or not getattr(res, "valid", True):
                return None
            return float(getattr(res, attr))

        result.best_time_score = _val(breakdown.get("time"), "time_score")
        result.best_memory_score = _val(breakdown.get("memory"), "memory_score")
        self_res = breakdown.get("selfhood")
        if self_res is not None and getattr(self_res, "valid", False):
            result.best_selfhood_score = float(self_res.selfhood_score)
            result.best_persistence = float(self_res.persistence)
        result.best_causality_score = _val(breakdown.get("causality"), "causality_score")
        result.best_resilience_score = _val(breakdown.get("resilience"), "resilience_score")

    # Frame hash + retained frames for videos.
    result.projected_hash = _projected_hash(frames)
    if video_frames_kept > 0:
        keep = min(video_frames_kept, frames.shape[0])
        result.frames_for_video = frames[:keep].copy()


def run_one_condition(
    *,
    condition: str,
    rule_idx: int,
    seed: int,
    rule_4d: FractionalRule | None,
    rule_2d: BSRule | None,
    grid_shape_4d: tuple[int, int, int, int],
    grid_shape_2d: tuple[int, int],
    timesteps: int,
    initial_density_4d: float,
    initial_density_2d: float,
    detection_config: DetectionConfig,
    backend: str,
    rollout_steps: int,
    video_frames_kept: int,
    snapshot_at: list[int],
) -> ConditionResult:
    """Run a single (condition, rule_idx, seed) and return the populated result."""
    if condition not in CONDITION_NAMES:
        raise ValueError(condition)

    rule_dict = rule_4d.to_dict() if rule_4d is not None else {"family": "2d_life"}
    result = ConditionResult(
        rule_idx=rule_idx, seed=seed, condition=condition, rule_dict=rule_dict
    )

    t0 = time.time()
    if condition == "coherent_4d":
        bsrule = rule_4d.to_bsrule()
        frames, snapshots, active = _simulate_4d_capturing(
            bsrule,
            grid_shape=grid_shape_4d,
            timesteps=timesteps,
            initial_density=rule_4d.initial_density,
            seed=seed,
            backend=backend,
            projection_method="mean_threshold",
            projection_theta=0.5,
            snapshot_at=snapshot_at,
            state_mutator=None,
        )
        rule_for_metrics = bsrule
        world_kind = "4d"
    elif condition == "shuffled_4d":
        bsrule = rule_4d.to_bsrule()
        # Use a fresh RNG inside the mutator that's seeded from the run seed.
        mutator_rng = np.random.default_rng(seed * 7919 + 1)

        def _mutator(state, t, _rng):
            # Use mutator_rng so the shuffle is deterministic from `seed`,
            # independent of any other RNG draws.
            return hidden_shuffle_mutator(state, t, mutator_rng)

        frames, snapshots, active = _simulate_4d_capturing(
            bsrule,
            grid_shape=grid_shape_4d,
            timesteps=timesteps,
            initial_density=rule_4d.initial_density,
            seed=seed,
            backend=backend,
            projection_method="mean_threshold",
            projection_theta=0.5,
            snapshot_at=snapshot_at,
            state_mutator=_mutator,
        )
        rule_for_metrics = bsrule
        world_kind = "4d"
    else:  # matched_2d
        frames, active = _simulate_2d_capturing(
            rule_2d,
            grid_shape=grid_shape_2d,
            timesteps=timesteps,
            initial_density=initial_density_2d,
            seed=seed,
        )
        snapshots = None
        rule_for_metrics = None
        world_kind = "2d"

    result.sim_time_seconds = time.time() - t0

    t1 = time.time()
    tracks = _detect_and_track(frames, detection_config)
    candidates = score_persistence(
        tracks,
        grid_shape=(frames.shape[1], frames.shape[2]),
        config=detection_config,
    )
    observer_scores, per_candidate = _compute_metrics_in_memory(
        tracks, candidates,
        grid_shape=(frames.shape[1], frames.shape[2]),
        seed=seed,
        snapshots=snapshots,
        rule_4d=rule_for_metrics,
        rollout_steps=rollout_steps,
        backend=backend,
        world_kind=world_kind,
    )
    result.metric_time_seconds = time.time() - t1

    _summarize_condition(
        result, tracks, observer_scores, per_candidate, frames, active,
        video_frames_kept=video_frames_kept,
    )
    return result


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _run_one_condition_for_parallel(
    item: tuple[int, int, str], shared: dict
) -> ConditionResult:
    ri, seed, cond = item
    rules = shared["rules"]
    rule = rules[ri]
    return run_one_condition(
        condition=cond, rule_idx=ri, seed=seed,
        rule_4d=rule, rule_2d=shared["rule_2d"],
        grid_shape_4d=shared["grid_shape_4d"],
        grid_shape_2d=shared["grid_shape_2d"],
        timesteps=shared["timesteps"],
        initial_density_4d=rule.initial_density,
        initial_density_2d=shared["initial_density_2d"],
        detection_config=shared["detection_config"],
        backend=shared["backend"],
        rollout_steps=shared["rollout_steps"],
        video_frames_kept=shared["video_frames_kept"],
        snapshot_at=shared["snapshot_at"],
    )


class _ParallelTask:
    """Picklable task dispatcher for parallel_sweep.

    joblib loky pickles the callable by reference (qualified name) and
    re-imports it in the worker; nested closures aren't picklable. This
    class binds the per-sweep ``shared`` dict to a top-level callable.
    """

    def __init__(self, shared: dict) -> None:
        self.shared = shared

    def __call__(self, item: tuple[int, int, str]) -> ConditionResult:
        return _run_one_condition_for_parallel(item, self.shared)


def run_sweep(
    *,
    rules: list[FractionalRule],
    seeds: list[int],
    grid_shape_4d: tuple[int, int, int, int],
    grid_shape_2d: tuple[int, int],
    timesteps: int,
    initial_density_2d: float,
    detection_config: DetectionConfig,
    backend: str,
    rollout_steps: int,
    rule_2d: BSRule,
    video_frames_kept: int = 0,
    snapshots_per_run: int = 2,
    progress: Callable[[str], None] | None = None,
    n_workers: int | None = None,
) -> list[PairedRecord]:
    """Run the full (rules x seeds x conditions) sweep.

    Work is parallelized over (rule_idx, seed, condition) triples via
    ``parallel_sweep``. Default ``n_workers`` is ``cpu_count - 2``; on
    machines with <= 2 cores this resolves to 1 and the serial path is
    used. Pass ``n_workers=1`` explicitly to force serial.
    """
    from observer_worlds.parallel import parallel_sweep

    snapshot_at = [
        int(timesteps * k / (snapshots_per_run + 1))
        for k in range(1, snapshots_per_run + 1)
    ]

    # Flatten work items.
    items: list[tuple[int, int, str]] = [
        (ri, seed, cond)
        for ri in range(len(rules))
        for seed in seeds
        for cond in CONDITION_NAMES
    ]

    shared = {
        "rules": rules,
        "rule_2d": rule_2d,
        "grid_shape_4d": grid_shape_4d,
        "grid_shape_2d": grid_shape_2d,
        "timesteps": timesteps,
        "initial_density_2d": initial_density_2d,
        "detection_config": detection_config,
        "backend": backend,
        "rollout_steps": rollout_steps,
        "video_frames_kept": video_frames_kept,
        "snapshot_at": snapshot_at,
    }

    # Forward live per-task lines to stderr when a progress callback is
    # provided -- otherwise long sweeps go silent for hours.
    verbose = 10 if progress is not None else 0

    t0 = time.time()
    flat_results = parallel_sweep(
        items,
        _ParallelTask(shared),
        n_workers=n_workers,
        progress=progress,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    if progress is not None:
        progress(
            f"  sweep wall time {elapsed:.0f}s "
            f"({len(items)} runs across {len(rules)} rules x "
            f"{len(seeds)} seeds x {len(CONDITION_NAMES)} conditions)"
        )

    # Regroup flat list -> PairedRecord per (rule_idx, seed).
    by_pair: dict[tuple[int, int], dict[str, ConditionResult]] = {}
    for (ri, seed, cond), result in zip(items, flat_results):
        by_pair.setdefault((ri, seed), {})[cond] = result

    records: list[PairedRecord] = []
    for ri, rule in enumerate(rules):
        for seed in seeds:
            triple = by_pair[(ri, seed)]
            records.append(PairedRecord(
                rule_idx=ri, seed=seed, rule_dict=rule.to_dict(),
                coherent_4d=triple["coherent_4d"],
                shuffled_4d=triple["shuffled_4d"],
                matched_2d=triple["matched_2d"],
            ))
    return records


# ---------------------------------------------------------------------------
# Summary metrics that the stats module compares across conditions
# ---------------------------------------------------------------------------


SUMMARY_METRICS: tuple[str, ...] = (
    "max_score",
    "mean_score",
    "median_score",
    "p90_score",
    "p95_score",
    "p99_score",
    "top3_mean_score",
    "top5_mean_score",
    "top10_mean_score",
    "lifetime_weighted_mean_score",
    "area_weighted_mean_score",
    "score_per_track",
    "n_candidates",
    "n_tracks",
    "max_component_lifetime",
    "mean_component_lifetime",
)


def metrics_dict(result: ConditionResult) -> dict[str, float]:
    """Pull the SUMMARY_METRICS out of a ConditionResult as a flat dict."""
    return {m: float(getattr(result, m)) for m in SUMMARY_METRICS}
