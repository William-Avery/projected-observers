"""(μ + λ) evolutionary search over fractional 4D rules with observer
fitness (M4C).

The :func:`mutate_fractional_rule` helper applies independent Gaussian
noise to the 5 rule floats and clips them back to the documented sample
ranges (and the absolute ``birth_max <= 0.80`` / ``survive_max <= 0.80``
caps from the M4A spec).  We mutate the *width* of each interval rather
than the maximum directly so the ``max >= min`` invariant is naturally
preserved.

The :func:`evolutionary_search_observer` driver runs a textbook
``(μ + λ)`` loop on top of :func:`evaluate_observer_fitness` from
``observer_search``: at each generation, ``λ`` parents are sampled
uniformly with replacement from the current population, mutated, and
evaluated.  The current population and the offspring are then merged and
truncated to the top ``μ`` by fitness.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from observer_worlds.search.observer_search import (
    ObserverFitnessReport,
    evaluate_observer_fitness,
)
from observer_worlds.search.rules import (
    FractionalRule,
    SAMPLE_RANGES,
    sample_random_fractional_rule,
)
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig


# Default per-parameter Gaussian mutation sigmas. The 5 floats are
# birth_min, birth_width, survive_min, survive_width, initial_density.
# (We mutate width = max - min, not max directly, so the constraint
# max - min >= 0 is naturally enforced.)
DEFAULT_MUTATION_SIGMAS: dict[str, float] = {
    "birth_min": 0.04,
    "birth_width": 0.04,
    "survive_min": 0.04,
    "survive_width": 0.05,
    "initial_density": 0.04,
}


def mutate_fractional_rule(
    rule: FractionalRule,
    rng: np.random.Generator,
    *,
    sigmas: dict[str, float] | None = None,
    sample_ranges: dict[str, tuple[float, float]] | None = None,
    max_value: float = 0.80,
) -> FractionalRule:
    """Apply Gaussian noise to each of the 5 rule floats and clip to the
    documented sample ranges (and the absolute ``max_value`` cap on
    ``birth_max`` / ``survive_max``).

    We mutate the *interval widths* (``birth_max - birth_min`` and
    ``survive_max - survive_min``) rather than the max values themselves
    so the ``max >= min`` invariant is preserved by construction.
    """
    sig = {**DEFAULT_MUTATION_SIGMAS, **(sigmas or {})}
    ranges = {**SAMPLE_RANGES, **(sample_ranges or {})}

    bw = float(rule.birth_max - rule.birth_min)
    sw = float(rule.survive_max - rule.survive_min)

    # 1) Apply Gaussian noise.
    bmin = float(rule.birth_min) + float(rng.normal(0.0, sig["birth_min"]))
    bw_n = bw + float(rng.normal(0.0, sig["birth_width"]))
    smin = float(rule.survive_min) + float(rng.normal(0.0, sig["survive_min"]))
    sw_n = sw + float(rng.normal(0.0, sig["survive_width"]))
    dens = float(rule.initial_density) + float(rng.normal(0.0, sig["initial_density"]))

    # 2) Clip each parameter to its documented sample range.
    def _clip(name: str, v: float) -> float:
        lo, hi = ranges[name]
        return float(min(hi, max(lo, v)))

    bmin = _clip("birth_min", bmin)
    bw_n = _clip("birth_width", bw_n)
    smin = _clip("survive_min", smin)
    sw_n = _clip("survive_width", sw_n)
    dens = _clip("initial_density", dens)

    # 3) Reconstruct max values, then enforce the absolute max_value cap by
    # *shrinking the width* (preserves max >= min).
    bmax = bmin + bw_n
    smax = smin + sw_n
    if bmax > max_value:
        bmax = max_value
        if bmax < bmin:
            # bmin clipped above max_value (only possible if the sample range
            # itself extends past max_value).  Pull bmin back down.
            bmin = bmax
    if smax > max_value:
        smax = max_value
        if smax < smin:
            smin = smax

    return FractionalRule(
        birth_min=bmin,
        birth_max=bmax,
        survive_min=smin,
        survive_max=smax,
        initial_density=dens,
    )


# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------


def _history_entry(generation: int, population: list[ObserverFitnessReport]) -> dict:
    fits = np.asarray([r.fitness for r in population], dtype=np.float64)
    return {
        "generation": int(generation),
        "best_fitness": float(fits.max()) if fits.size else 0.0,
        "mean_fitness": float(fits.mean()) if fits.size else 0.0,
        "median_fitness": float(np.median(fits)) if fits.size else 0.0,
        "population_size": int(fits.size),
    }


def evolutionary_search_observer(
    *,
    n_generations: int,
    mu: int,
    lam: int,
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
    sigmas: dict[str, float] | None = None,
    initial_population: list[FractionalRule] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ObserverFitnessReport], list[dict]]:
    """``(μ + λ)`` evolutionary search over fractional 4D rules.

    Generation 0 either uses the supplied ``initial_population`` (truncated
    or padded to ``mu`` rules) or samples ``mu`` rules with
    :func:`sample_random_fractional_rule`.  Each subsequent generation
    samples ``lam`` parents uniformly with replacement, mutates each one,
    evaluates the offspring, then keeps the top ``mu`` of (population +
    offspring) by fitness.

    All rule evaluations re-use the same ``base_seed..base_seed + n_seeds - 1``
    seeds for the M2 metric pipeline, matching the convention in
    :func:`random_search_observer`.

    Returns
    -------
    (final_population, history)
        ``final_population`` is the final population sorted by descending
        fitness.  ``history`` has one dict per generation (``n_generations + 1``
        entries: generation 0 plus one per evolved generation) with keys
        ``generation``, ``best_fitness``, ``mean_fitness``,
        ``median_fitness``, ``population_size``.
    """
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    if n_generations < 0:
        raise ValueError(f"n_generations must be >= 0, got {n_generations}")

    sampler_rng = seeded_rng(sampler_seed)
    detection_config = detection_config or DetectionConfig()

    def _eval(rule: FractionalRule) -> ObserverFitnessReport:
        return evaluate_observer_fitness(
            rule,
            n_seeds=n_seeds, base_seed=base_seed,
            grid_shape=grid_shape, timesteps=timesteps,
            detection_config=detection_config, backend=backend,
            fitness_mode=fitness_mode, rollout_steps=rollout_steps,
            snapshots_per_run=snapshots_per_run,
        )

    # ---------------- Generation 0 (initial population)
    t0 = time.time()
    if initial_population is None:
        seed_rules = [sample_random_fractional_rule(sampler_rng) for _ in range(mu)]
    else:
        # Use the first mu rules; if too few, top up with random samples.
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
    history: list[dict] = [_history_entry(0, population)]
    if progress is not None:
        h = history[-1]
        progress(
            f"  [gen 0 done] best={h['best_fitness']:+.3f}  "
            f"mean={h['mean_fitness']:+.3f}  median={h['median_fitness']:+.3f}"
        )

    # ---------------- evolved generations
    for gen in range(1, n_generations + 1):
        # 1) Pick lam parents with replacement and mutate each.
        parent_indices = sampler_rng.integers(0, len(population), size=lam)
        offspring_rules: list[FractionalRule] = []
        for pi in parent_indices:
            parent = population[int(pi)].rule
            child = mutate_fractional_rule(parent, sampler_rng, sigmas=sigmas)
            offspring_rules.append(child)

        # 2) Evaluate offspring.
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

        # 3) Combine + truncate.
        combined = population + offspring_reports
        combined.sort(key=lambda r: -r.fitness)
        population = combined[:mu]
        history.append(_history_entry(gen, population))
        if progress is not None:
            h = history[-1]
            progress(
                f"  [gen {gen} done] best={h['best_fitness']:+.3f}  "
                f"mean={h['mean_fitness']:+.3f}  median={h['median_fitness']:+.3f}"
            )

    population.sort(key=lambda r: -r.fitness)
    return population, history
