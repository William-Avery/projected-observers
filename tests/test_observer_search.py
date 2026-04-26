"""Tests for M4C observer-metric-guided rule search."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.search import (
    DEFAULT_MUTATION_SIGMAS,
    FITNESS_MODES,
    FractionalRule,
    SAMPLE_RANGES,
    evaluate_observer_fitness,
    evolutionary_search_observer,
    mutate_fractional_rule,
    random_search_observer,
)


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------


def _is_in_range(value: float, lo: float, hi: float, tol: float = 1e-9) -> bool:
    return (lo - tol) <= value <= (hi + tol)


def test_mutate_stays_in_range():
    rng = np.random.default_rng(42)
    rule = FractionalRule(0.20, 0.30, 0.20, 0.40, 0.20)
    for _ in range(100):
        m = mutate_fractional_rule(rule, rng)
        assert _is_in_range(m.birth_min, *SAMPLE_RANGES["birth_min"])
        assert _is_in_range(m.survive_min, *SAMPLE_RANGES["survive_min"])
        assert _is_in_range(m.initial_density, *SAMPLE_RANGES["initial_density"])
        assert m.birth_max <= 0.80 + 1e-9
        assert m.survive_max <= 0.80 + 1e-9
        # The post-init invariants must hold.
        assert m.birth_max >= m.birth_min
        assert m.survive_max >= m.survive_min


def test_mutate_changes_values():
    """At least 4 of 5 fields should differ after a single mutation under
    default sigmas (with high probability)."""
    rng = np.random.default_rng(7)
    rule = FractionalRule(0.20, 0.30, 0.20, 0.40, 0.20)
    n_changed_at_least_4 = 0
    for _ in range(20):
        m = mutate_fractional_rule(rule, rng)
        diffs = sum([
            m.birth_min != rule.birth_min,
            m.birth_max != rule.birth_max,
            m.survive_min != rule.survive_min,
            m.survive_max != rule.survive_max,
            m.initial_density != rule.initial_density,
        ])
        if diffs >= 4:
            n_changed_at_least_4 += 1
    # Under default sigmas this should essentially always be true.
    assert n_changed_at_least_4 >= 18


def test_mutate_widths_nonnegative():
    """Even narrow widths must stay non-negative after mutation (post-init
    catches violations)."""
    rng = np.random.default_rng(11)
    # Very narrow widths.
    rule = FractionalRule(0.30, 0.305, 0.30, 0.305, 0.20)
    for _ in range(50):
        m = mutate_fractional_rule(rule, rng)
        assert m.birth_max >= m.birth_min
        assert m.survive_max >= m.survive_min


# ---------------------------------------------------------------------------
# Fitness modes registry
# ---------------------------------------------------------------------------


def test_fitness_modes_are_listed():
    expected = {"lifetime_weighted", "top5_mean", "score_per_track", "composite"}
    assert set(FITNESS_MODES) == expected
    for fn in FITNESS_MODES.values():
        # Each value should be a callable accepting a dict.
        assert callable(fn)
        # Shouldn't crash on an empty-ish dict; should default to 0.
        out = fn({})
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# Single-rule fitness evaluation
# ---------------------------------------------------------------------------


def test_evaluate_observer_fitness_returns_report():
    rule = FractionalRule(0.15, 0.26, 0.09, 0.38, 0.15)
    rep = evaluate_observer_fitness(
        rule, n_seeds=1, base_seed=1000,
        grid_shape=(8, 8, 2, 2), timesteps=20,
        backend="numba", fitness_mode="lifetime_weighted",
        rollout_steps=2, snapshots_per_run=1,
    )
    assert rep.n_seeds == 1
    assert len(rep.per_seed_fitness) == 1
    assert np.isfinite(rep.fitness)
    assert rep.fitness_mode == "lifetime_weighted"


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


def test_random_search_observer_returns_sorted():
    reports = random_search_observer(
        n_rules=3, n_seeds=1,
        base_seed=1000, sampler_seed=7,
        grid_shape=(8, 8, 2, 2), timesteps=20,
        backend="numba", fitness_mode="lifetime_weighted",
        rollout_steps=2, snapshots_per_run=1,
    )
    assert len(reports) == 3
    fits = [r.fitness for r in reports]
    assert fits == sorted(fits, reverse=True), "expected descending sort"


# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------


def test_evolutionary_search_finds_finite_results():
    reports, history = evolutionary_search_observer(
        n_generations=2, mu=3, lam=3,
        n_seeds=1, base_seed=1000, sampler_seed=11,
        grid_shape=(8, 8, 2, 2), timesteps=20,
        backend="numba", fitness_mode="lifetime_weighted",
        rollout_steps=2, snapshots_per_run=1,
    )
    assert len(reports) == 3
    fits = [r.fitness for r in reports]
    assert fits == sorted(fits, reverse=True)
    for r in reports:
        assert np.isfinite(r.fitness)
    # gen 0 + 2 evolved generations.
    assert len(history) == 3


def test_evolve_history_format():
    _, history = evolutionary_search_observer(
        n_generations=1, mu=2, lam=2,
        n_seeds=1, base_seed=1000, sampler_seed=13,
        grid_shape=(8, 8, 2, 2), timesteps=20,
        backend="numba", fitness_mode="lifetime_weighted",
        rollout_steps=2, snapshots_per_run=1,
    )
    keys_required = {
        "generation", "best_fitness", "mean_fitness",
        "median_fitness", "population_size",
    }
    for entry in history:
        assert keys_required.issubset(entry.keys())
