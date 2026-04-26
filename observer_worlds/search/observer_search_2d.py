"""2D variant of the observer-metric-guided rule search (M4D).

This is the 2D analogue of :mod:`observer_worlds.search.observer_search`.
It runs the same M2 metric pipeline (time / memory / selfhood ->
observer_score) on a 2D :class:`CA2D` instead of the 4D ``CA4D``.

The 4D-specific scores (``causality`` and ``resilience``) are skipped:
they require simulating the 4D bulk inside a candidate's interior /
boundary / environment fibers, which has no natural meaning for a flat
2D world.  ``compute_observer_scores`` already handles missing
components by redistributing the weights, so the resulting combined
score is well-defined.

The output of this module is consumed by ``run_m4d_holdout_validation.py``
to give the 2D baseline a fair, optimized rule (rather than just
Conway's Life) to compare against the M4A/M4C 4D winners.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from observer_worlds.detection import GreedyTracker, classify_boundary, extract_components
from observer_worlds.metrics import (
    collect_raw_scores,
    compute_memory_score,
    compute_observer_scores,
    compute_selfhood_score,
    compute_time_score,
    extract_track_features,
    score_persistence,
)
from observer_worlds.search.observer_search import (
    FITNESS_MODES,
    ObserverFitnessReport,
    _summary_for_run,
)
from observer_worlds.search.rules import FractionalRule, sample_random_fractional_rule
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import BSRule, CA2D


# Max neighbour count for the 2D Moore-r1 neighbourhood (3**2 - 1).
DEFAULT_MAX_COUNT_2D: int = 8


# ---------------------------------------------------------------------------
# Single-rule, single-seed evaluation
# ---------------------------------------------------------------------------


def _evaluate_single_seed_2d(
    rule: FractionalRule,
    *,
    seed: int,
    grid_shape: tuple[int, int] = (32, 32),
    timesteps: int = 200,
    detection_config: DetectionConfig,
    early_abort: bool = True,
) -> dict[str, float]:
    """Run one 2D scout, full M2 metric suite (sans causality + resilience),
    and return the summary dict consumed by the fitness functions.

    Mirrors :func:`observer_worlds.search.observer_search._evaluate_single_seed`
    but uses :class:`CA2D` (no projection step needed -- the CA already lives
    in 2D) and skips the 4D-only causality / resilience scores.
    """
    bsrule = rule.to_bsrule(max_count=DEFAULT_MAX_COUNT_2D)
    rng = seeded_rng(seed)
    ca = CA2D(shape=grid_shape, rule=bsrule)
    ca.initialize_random(density=rule.initial_density, rng=rng)

    Nx, Ny = grid_shape
    frames = np.empty((timesteps, Nx, Ny), dtype=np.uint8)
    active = np.empty(timesteps, dtype=np.float32)
    frames[0] = ca.state
    active[0] = float(ca.state.mean())
    for t in range(1, timesteps):
        ca.step()
        frames[t] = ca.state
        active[t] = float(ca.state.mean())
        if early_abort and t >= 10:
            recent = active[max(0, t - 5): t + 1]
            if recent.max() < 1e-4 or recent.min() > 0.99:
                # Trivial dynamics -- bail with a zero summary.
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

    # Run the (subset of the) M2 metric suite on the candidate tracks.
    candidate_ids = {c.track_id for c in candidates if c.is_candidate}
    track_by_id = {t.track_id: t for t in tracks}
    cand_tracks = [track_by_id[i] for i in candidate_ids if i in track_by_id]
    raw_per_track: list[dict[str, float | None]] = []
    for tr in cand_tracks:
        feats = extract_track_features(tr)
        time_res = compute_time_score(feats, seed=seed)
        mem_res = compute_memory_score(feats, seed=seed)
        self_res = compute_selfhood_score(feats, seed=seed)
        # boundary classification kept for parity with the 4D variant; result
        # currently unused since causality/resilience are skipped in 2D.
        _ = classify_boundary(feats)
        raw_per_track.append(collect_raw_scores(
            track_id=tr.track_id,
            time=time_res, memory=mem_res, selfhood=self_res,
            causality=None, resilience=None,
        ))
    observer_scores = compute_observer_scores(raw_per_track)

    return _summary_for_run(
        tracks=tracks,
        observer_scores=observer_scores,
        n_tracks=len(tracks),
        late_active=float(active[active.size // 2:].mean()),
    )


# ---------------------------------------------------------------------------
# Multi-seed fitness evaluation
# ---------------------------------------------------------------------------


def evaluate_observer_fitness_2d(
    rule: FractionalRule,
    *,
    n_seeds: int = 3,
    base_seed: int = 0,
    grid_shape: tuple[int, int] = (32, 32),
    timesteps: int = 200,
    detection_config: DetectionConfig | None = None,
    fitness_mode: str = "lifetime_weighted",
    early_abort: bool = True,
) -> ObserverFitnessReport:
    """2D variant of :func:`evaluate_observer_fitness`.

    Returns the same :class:`ObserverFitnessReport` dataclass.  ``aborted_seeds``
    counts the seeds that produced zero observer candidates.
    """
    detection_config = detection_config or DetectionConfig()
    fitness_fn = FITNESS_MODES.get(fitness_mode)
    if fitness_fn is None:
        raise ValueError(
            f"Unknown fitness_mode {fitness_mode!r}; options: {list(FITNESS_MODES)}"
        )

    seeds = [base_seed + i for i in range(n_seeds)]

    t0 = time.time()
    summaries: list[dict[str, float]] = []
    for s in seeds:
        summaries.append(_evaluate_single_seed_2d(
            rule, seed=s,
            grid_shape=grid_shape, timesteps=timesteps,
            detection_config=detection_config, early_abort=early_abort,
        ))
    sim_time = time.time() - t0

    per_seed_fit = [fitness_fn(s) for s in summaries]
    per_seed_nc = [int(s["n_candidates"]) for s in summaries]
    per_seed_nt = [int(s["n_tracks"]) for s in summaries]

    def _mean(key: str) -> float:
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


def random_search_observer_2d(
    *,
    n_rules: int,
    n_seeds: int,
    base_seed: int,
    sampler_seed: int,
    grid_shape: tuple[int, int],
    timesteps: int,
    detection_config: DetectionConfig | None = None,
    fitness_mode: str = "lifetime_weighted",
    progress: Callable[[str], None] | None = None,
) -> list[ObserverFitnessReport]:
    """Random search over fractional rules using the 2D pipeline.

    Returns the reports sorted by descending fitness.
    """
    rng = seeded_rng(sampler_seed)
    reports: list[ObserverFitnessReport] = []
    t0 = time.time()
    for i in range(n_rules):
        rule = sample_random_fractional_rule(rng)
        rep = evaluate_observer_fitness_2d(
            rule,
            n_seeds=n_seeds, base_seed=base_seed,
            grid_shape=grid_shape, timesteps=timesteps,
            detection_config=detection_config, fitness_mode=fitness_mode,
        )
        reports.append(rep)
        if progress is not None:
            elapsed = time.time() - t0
            best = max(reports, key=lambda r: r.fitness)
            progress(
                f"  [{i+1}/{n_rules}] elapsed={elapsed:.0f}s  "
                f"best fitness={best.fitness:+.3f} "
                f"(n_cand={best.mean_n_candidates:.0f}, "
                f"n_tracks={best.mean_n_tracks:.0f})"
            )
    reports.sort(key=lambda r: -r.fitness)
    return reports


# ---------------------------------------------------------------------------
# (mu + lam) evolutionary search
# ---------------------------------------------------------------------------


def _history_entry_2d(
    generation: int, population: list[ObserverFitnessReport]
) -> dict:
    fits = np.asarray([r.fitness for r in population], dtype=np.float64)
    return {
        "generation": int(generation),
        "best_fitness": float(fits.max()) if fits.size else 0.0,
        "mean_fitness": float(fits.mean()) if fits.size else 0.0,
        "median_fitness": float(np.median(fits)) if fits.size else 0.0,
        "population_size": int(fits.size),
    }


def evolutionary_search_observer_2d(
    *,
    n_generations: int,
    mu: int,
    lam: int,
    n_seeds: int,
    base_seed: int,
    sampler_seed: int,
    grid_shape: tuple[int, int],
    timesteps: int,
    detection_config: DetectionConfig | None = None,
    fitness_mode: str = "lifetime_weighted",
    sigmas: dict[str, float] | None = None,
    initial_population: list[FractionalRule] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ObserverFitnessReport], list[dict]]:
    """``(mu + lam)`` evolutionary search over 2D fractional rules.

    Mirrors :func:`evolutionary_search_observer` but evaluates each rule
    via :func:`evaluate_observer_fitness_2d`.  Reuses
    :func:`mutate_fractional_rule` from ``observer_evolve`` since the
    mutator is rule-family-only and dimension-agnostic.

    Returns
    -------
    (final_population, history)
        ``final_population`` is the final population sorted by descending
        fitness.  ``history`` has ``n_generations + 1`` entries
        (generation 0 plus one per evolved generation) with keys
        ``generation``, ``best_fitness``, ``mean_fitness``,
        ``median_fitness``, ``population_size``.
    """
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    if n_generations < 0:
        raise ValueError(f"n_generations must be >= 0, got {n_generations}")

    # Local import to avoid a circular import at module load time and to make
    # the rule-family-only nature of the mutator obvious.
    from observer_worlds.search.observer_evolve import mutate_fractional_rule

    sampler_rng = seeded_rng(sampler_seed)
    detection_config = detection_config or DetectionConfig()

    def _eval(rule: FractionalRule) -> ObserverFitnessReport:
        return evaluate_observer_fitness_2d(
            rule,
            n_seeds=n_seeds, base_seed=base_seed,
            grid_shape=grid_shape, timesteps=timesteps,
            detection_config=detection_config, fitness_mode=fitness_mode,
        )

    # ---------------- Generation 0 (initial population)
    t0 = time.time()
    if initial_population is None:
        seed_rules = [sample_random_fractional_rule(sampler_rng) for _ in range(mu)]
    else:
        seed_rules = list(initial_population[:mu])
        while len(seed_rules) < mu:
            seed_rules.append(sample_random_fractional_rule(sampler_rng))

    population: list[ObserverFitnessReport] = []
    for i, rule in enumerate(seed_rules):
        rep = _eval(rule)
        population.append(rep)
        if progress is not None:
            elapsed = time.time() - t0
            progress(
                f"  [gen 0 init {i+1}/{mu}] elapsed={elapsed:.0f}s  "
                f"fitness={rep.fitness:+.3f}  ({rep.rule.short_repr()})"
            )
    population.sort(key=lambda r: -r.fitness)
    history: list[dict] = [_history_entry_2d(0, population)]
    if progress is not None:
        h = history[-1]
        progress(
            f"  [gen 0 done] best={h['best_fitness']:+.3f}  "
            f"mean={h['mean_fitness']:+.3f}  median={h['median_fitness']:+.3f}"
        )

    # ---------------- evolved generations
    for gen in range(1, n_generations + 1):
        parent_indices = sampler_rng.integers(0, len(population), size=lam)
        offspring_rules: list[FractionalRule] = []
        for pi in parent_indices:
            parent = population[int(pi)].rule
            child = mutate_fractional_rule(parent, sampler_rng, sigmas=sigmas)
            offspring_rules.append(child)

        offspring_reports: list[ObserverFitnessReport] = []
        for j, child_rule in enumerate(offspring_rules):
            rep = _eval(child_rule)
            offspring_reports.append(rep)
            if progress is not None:
                elapsed = time.time() - t0
                progress(
                    f"  [gen {gen} child {j+1}/{lam}] elapsed={elapsed:.0f}s  "
                    f"fitness={rep.fitness:+.3f}  ({rep.rule.short_repr()})"
                )

        combined = population + offspring_reports
        combined.sort(key=lambda r: -r.fitness)
        population = combined[:mu]
        history.append(_history_entry_2d(gen, population))
        if progress is not None:
            h = history[-1]
            progress(
                f"  [gen {gen} done] best={h['best_fitness']:+.3f}  "
                f"mean={h['mean_fitness']:+.3f}  median={h['median_fitness']:+.3f}"
            )

    population.sort(key=lambda r: -r.fitness)
    return population, history
