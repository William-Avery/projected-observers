"""Observer-metric-guided rule search (M4C).

Where M4A (viability search) used cheap proxies — persistent-component
count, target activity, etc. — M4C runs the **full M2 metric suite**
(time / memory / selfhood / causality / resilience → observer_score) on
each candidate rule and selects rules that produce candidates with high
observer scores.

The fitness function defaults to ``lifetime_weighted_mean_score`` because:

  * it is a single number per run (the search loop needs a scalar),
  * it naturally resists the **track-count confound** that M4B exposed
    (long-lived candidates dominate, regardless of how many short-lived
    ones a chaotic rule also produces), and
  * it rewards the joint property "many high-quality observers that
    persist for a long time", which is exactly what we want.

Other fitness modes (``top5_mean``, ``score_per_track``, ``composite``)
are exposed for cross-validation or for exploring the impact of the
fitness choice on which rules win.

This module provides random search; the evolutionary ``(μ+λ)`` loop and
CLI are layered on top in ``observer_search_evolve.py`` and
``experiments/run_search_observer_rules.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from observer_worlds.detection import GreedyTracker, classify_boundary, extract_components
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
from observer_worlds.search.fitness import simulate_4d_in_memory
from observer_worlds.search.rules import FractionalRule, sample_random_fractional_rule
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import BSRule


# ---------------------------------------------------------------------------
# Fitness modes
# ---------------------------------------------------------------------------


# A FitnessFn maps a metric-summary dict to a scalar fitness value.
FitnessFn = Callable[[dict[str, float]], float]


def fitness_lifetime_weighted(s: dict[str, float]) -> float:
    return float(s.get("lifetime_weighted_mean_score", 0.0))


def fitness_top5_mean(s: dict[str, float]) -> float:
    return float(s.get("top5_mean_score", 0.0))


def fitness_score_per_track(s: dict[str, float]) -> float:
    return float(s.get("score_per_track", 0.0))


def fitness_composite(s: dict[str, float]) -> float:
    """0.5 * top5_mean + 0.5 * lifetime_weighted_mean."""
    return 0.5 * s.get("top5_mean_score", 0.0) + 0.5 * s.get("lifetime_weighted_mean_score", 0.0)


FITNESS_MODES: dict[str, FitnessFn] = {
    "lifetime_weighted": fitness_lifetime_weighted,
    "top5_mean": fitness_top5_mean,
    "score_per_track": fitness_score_per_track,
    "composite": fitness_composite,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ObserverFitnessReport:
    """Outcome of evaluating one (rule, seeds) on observer fitness."""

    rule: FractionalRule
    fitness: float
    fitness_mode: str
    n_seeds: int

    # Aggregated summary metrics (mean across seeds).
    mean_n_tracks: float = 0.0
    mean_n_candidates: float = 0.0
    mean_max_score: float = 0.0
    mean_top5_mean_score: float = 0.0
    mean_p95_score: float = 0.0
    mean_lifetime_weighted_mean_score: float = 0.0
    mean_score_per_track: float = 0.0
    mean_late_active: float = 0.0
    mean_max_component_lifetime: float = 0.0

    # Per-seed scalars (length n_seeds) for variance inspection.
    per_seed_fitness: list[float] = field(default_factory=list)
    per_seed_n_candidates: list[int] = field(default_factory=list)
    per_seed_n_tracks: list[int] = field(default_factory=list)

    # Diagnostics.
    aborted_seeds: int = 0  # how many seeds produced no candidates
    sim_time_seconds: float = 0.0

    # Source/identity for reproducibility.
    seeds_used: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-rule, single-seed evaluation
# ---------------------------------------------------------------------------


def _summary_for_run(
    tracks: list,
    observer_scores: list,
    n_tracks: int,
    late_active: float,
) -> dict[str, float]:
    """Reduce a single run's outputs into the SUMMARY_METRICS-style dict
    that the fitness functions consume.

    Keys mirror those in `experiments._m4b_sweep.SUMMARY_METRICS` so any
    future refactor can share the helper."""
    track_by_id = {t.track_id: t for t in tracks}
    if not observer_scores:
        return {
            "n_tracks": float(n_tracks),
            "n_candidates": 0.0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "p90_score": 0.0,
            "p95_score": 0.0,
            "p99_score": 0.0,
            "top3_mean_score": 0.0,
            "top5_mean_score": 0.0,
            "top10_mean_score": 0.0,
            "lifetime_weighted_mean_score": 0.0,
            "area_weighted_mean_score": 0.0,
            "score_per_track": 0.0,
            "max_component_lifetime": float(max((t.age for t in tracks), default=0)),
            "late_active": late_active,
        }
    scores = np.asarray([o.combined for o in observer_scores], dtype=np.float64)
    sorted_desc = np.sort(scores)[::-1]
    ages = np.asarray(
        [int(track_by_id[o.track_id].age) for o in observer_scores], dtype=np.float64
    )
    areas = np.asarray(
        [
            float(np.mean(track_by_id[o.track_id].area_history))
            if track_by_id[o.track_id].area_history else 0.0
            for o in observer_scores
        ],
        dtype=np.float64,
    )
    out = {
        "n_tracks": float(n_tracks),
        "n_candidates": float(len(observer_scores)),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "p90_score": float(np.percentile(scores, 90)),
        "p95_score": float(np.percentile(scores, 95)),
        "p99_score": float(np.percentile(scores, 99)),
        "top3_mean_score": float(sorted_desc[: min(3, scores.size)].mean()),
        "top5_mean_score": float(sorted_desc[: min(5, scores.size)].mean()),
        "top10_mean_score": float(sorted_desc[: min(10, scores.size)].mean()),
        "lifetime_weighted_mean_score": float((scores * ages).sum() / ages.sum())
        if ages.sum() > 0 else 0.0,
        "area_weighted_mean_score": float((scores * areas).sum() / areas.sum())
        if areas.sum() > 0 else 0.0,
        "score_per_track": float(scores.sum() / max(n_tracks, 1)),
        "max_component_lifetime": float(max((t.age for t in tracks), default=0)),
        "late_active": late_active,
    }
    return out


def _evaluate_single_seed(
    rule: FractionalRule,
    *,
    seed: int,
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    detection_config: DetectionConfig,
    backend: str,
    rollout_steps: int,
    snapshots_per_run: int,
    early_abort: bool,
) -> dict[str, float]:
    """Run a single (rule, seed) coherent_4d simulation, compute the full
    M2 metric suite, return a summary dict."""
    bsrule = rule.to_bsrule()
    snapshot_at = [
        int(timesteps * k / (snapshots_per_run + 1))
        for k in range(1, snapshots_per_run + 1)
    ]

    rng = seeded_rng(seed)
    from observer_worlds.worlds import CA4D, project
    ca = CA4D(shape=grid_shape, rule=bsrule, backend=backend)
    ca.initialize_random(density=rule.initial_density, rng=rng)

    Nx, Ny = grid_shape[0], grid_shape[1]
    frames = np.empty((timesteps, Nx, Ny), dtype=np.uint8)
    active = np.empty(timesteps, dtype=np.float32)
    snapshots: dict[int, np.ndarray] = {}
    snap_set = set(snapshot_at)

    Y0 = project(ca.state, method="mean_threshold", theta=0.5)
    frames[0] = Y0
    active[0] = float(Y0.mean())
    if 0 in snap_set:
        snapshots[0] = ca.state.copy()
    for t in range(1, timesteps):
        ca.step()
        Y = project(ca.state, method="mean_threshold", theta=0.5)
        frames[t] = Y
        active[t] = float(Y.mean())
        if t in snap_set:
            snapshots[t] = ca.state.copy()
        if early_abort and t >= 10:
            recent = active[max(0, t - 5): t + 1]
            if recent.max() < 1e-4 or recent.min() > 0.99:
                # Trivial dynamics — bail and return zero summary.
                return _summary_for_run(
                    tracks=[], observer_scores=[], n_tracks=0,
                    late_active=float(active[t]),
                )

    # Detect + track.
    tracker = GreedyTracker(config=detection_config)
    for t in range(timesteps):
        comps = extract_components(frames[t], frame_idx=t, config=detection_config)
        tracker.update(t, comps)
    tracks = tracker.finalize()
    candidates = score_persistence(
        tracks, grid_shape=(Nx, Ny), config=detection_config
    )

    # Run the full metric suite.
    candidate_ids = {c.track_id for c in candidates if c.is_candidate}
    track_by_id = {t.track_id: t for t in tracks}
    cand_tracks = [track_by_id[i] for i in candidate_ids if i in track_by_id]
    raw_per_track = []
    available_snap_t = sorted(snapshots.keys())
    for tr in cand_tracks:
        feats = extract_track_features(tr)
        time_res = compute_time_score(feats, seed=seed)
        mem_res = compute_memory_score(feats, seed=seed)
        self_res = compute_selfhood_score(feats, seed=seed)
        bnd_res = classify_boundary(feats)
        causal_res = None
        resil_res = None
        snap_t = None
        for t_ in reversed(available_snap_t):
            if tr.birth_frame <= t_ <= tr.last_frame:
                snap_t = t_; break
        if snap_t is not None:
            i = tr.frames.index(snap_t) if snap_t in tr.frames else \
                tr.frames.index(min(tr.frames, key=lambda f: abs(f - snap_t)))
            interior = tr.interior_history[i]
            boundary = tr.boundary_history[i]
            env = tr.env_history[i]
            if interior.any() and boundary.any() and env.any():
                try:
                    snapshot_4d = snapshots[snap_t]
                    causal_res = compute_causality_score(
                        snapshot_4d, bsrule, interior, boundary, env,
                        n_steps=rollout_steps, backend=backend,
                        seed=seed, track_id=tr.track_id,
                    )
                    resil_res = compute_resilience_score(
                        snapshot_4d, bsrule, interior,
                        n_steps=rollout_steps, backend=backend,
                        seed=seed, track_id=tr.track_id,
                    )
                except Exception:
                    pass
        raw_per_track.append(collect_raw_scores(
            track_id=tr.track_id,
            time=time_res, memory=mem_res, selfhood=self_res,
            causality=causal_res, resilience=resil_res,
        ))
    observer_scores = compute_observer_scores(raw_per_track)

    return _summary_for_run(
        tracks=tracks, observer_scores=observer_scores,
        n_tracks=len(tracks),
        late_active=float(active[active.size // 2:].mean()),
    )


def evaluate_observer_fitness(
    rule: FractionalRule,
    *,
    n_seeds: int = 3,
    base_seed: int = 0,
    grid_shape: tuple[int, int, int, int] = (32, 32, 4, 4),
    timesteps: int = 200,
    detection_config: DetectionConfig | None = None,
    backend: str = "numba",
    fitness_mode: str = "lifetime_weighted",
    rollout_steps: int = 6,
    snapshots_per_run: int = 2,
    early_abort: bool = True,
) -> ObserverFitnessReport:
    """Evaluate a fractional rule on observer-fitness across multiple seeds.

    Returns an :class:`ObserverFitnessReport` with the mean-across-seeds
    fitness and a battery of diagnostics.
    """
    detection_config = detection_config or DetectionConfig()
    fitness_fn = FITNESS_MODES.get(fitness_mode)
    if fitness_fn is None:
        raise ValueError(f"Unknown fitness_mode {fitness_mode!r}; "
                         f"options: {list(FITNESS_MODES)}")

    seeds = [base_seed + i for i in range(n_seeds)]

    t0 = time.time()
    summaries: list[dict[str, float]] = []
    for s in seeds:
        summaries.append(_evaluate_single_seed(
            rule, seed=s,
            grid_shape=grid_shape, timesteps=timesteps,
            detection_config=detection_config, backend=backend,
            rollout_steps=rollout_steps, snapshots_per_run=snapshots_per_run,
            early_abort=early_abort,
        ))
    sim_time = time.time() - t0

    per_seed_fit = [fitness_fn(s) for s in summaries]
    per_seed_nc = [int(s["n_candidates"]) for s in summaries]
    per_seed_nt = [int(s["n_tracks"]) for s in summaries]

    def _mean(key):
        return float(np.mean([s[key] for s in summaries]))

    return ObserverFitnessReport(
        rule=rule,
        fitness=float(np.mean(per_seed_fit)),
        fitness_mode=fitness_mode,
        n_seeds=n_seeds,
        mean_n_tracks=float(np.mean(per_seed_nt)),
        mean_n_candidates=float(np.mean(per_seed_nc)),
        mean_max_score=_mean("max_score"),
        mean_top5_mean_score=_mean("top5_mean_score"),
        mean_p95_score=_mean("p95_score"),
        mean_lifetime_weighted_mean_score=_mean("lifetime_weighted_mean_score"),
        mean_score_per_track=_mean("score_per_track"),
        mean_late_active=_mean("late_active"),
        mean_max_component_lifetime=_mean("max_component_lifetime"),
        per_seed_fitness=per_seed_fit,
        per_seed_n_candidates=per_seed_nc,
        per_seed_n_tracks=per_seed_nt,
        aborted_seeds=sum(1 for nc in per_seed_nc if nc == 0),
        sim_time_seconds=sim_time,
        seeds_used=seeds,
    )


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


def random_search_observer(
    *,
    n_rules: int,
    n_seeds: int,
    base_seed: int,
    sampler_seed: int,
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    detection_config: DetectionConfig | None = None,
    backend: str = "numba",
    fitness_mode: str = "lifetime_weighted",
    rollout_steps: int = 6,
    snapshots_per_run: int = 2,
    progress: Callable[[str], None] | None = None,
) -> list[ObserverFitnessReport]:
    """Random search over fractional rules; fitness via the M2 metric suite."""
    rng = seeded_rng(sampler_seed)
    reports: list[ObserverFitnessReport] = []
    t0 = time.time()
    for i in range(n_rules):
        rule = sample_random_fractional_rule(rng)
        rep = evaluate_observer_fitness(
            rule,
            n_seeds=n_seeds, base_seed=base_seed,
            grid_shape=grid_shape, timesteps=timesteps,
            detection_config=detection_config, backend=backend,
            fitness_mode=fitness_mode, rollout_steps=rollout_steps,
            snapshots_per_run=snapshots_per_run,
        )
        reports.append(rep)
        if progress is not None:
            elapsed = time.time() - t0
            best = max(reports, key=lambda r: r.fitness)
            progress(
                f"  [{i+1}/{n_rules}] elapsed={elapsed:.0f}s  "
                f"best so far: fitness={best.fitness:+.3f} "
                f"(top5={best.mean_top5_mean_score:+.3f}, "
                f"lwm={best.mean_lifetime_weighted_mean_score:+.3f}, "
                f"n_cand={best.mean_n_candidates:.0f})"
            )
    reports.sort(key=lambda r: -r.fitness)
    return reports
