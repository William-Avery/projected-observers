"""Tests for the 2D variant of M4C observer-metric-guided rule search (M4D)."""

from __future__ import annotations

import numpy as np

from observer_worlds.search import (
    DEFAULT_MAX_COUNT_2D,
    FractionalRule,
    evaluate_observer_fitness_2d,
    evolutionary_search_observer_2d,
    random_search_observer_2d,
)
from observer_worlds.worlds import BSRule


# ---------------------------------------------------------------------------
# Single-rule fitness evaluation
# ---------------------------------------------------------------------------


def test_evaluate_observer_fitness_2d_returns_report():
    """Life-ish fractional rule on a tiny grid yields a finite ObserverFitnessReport."""
    rule = FractionalRule(0.30, 0.40, 0.20, 0.40, 0.30)
    rep = evaluate_observer_fitness_2d(
        rule, n_seeds=1, base_seed=1000,
        grid_shape=(8, 8), timesteps=20,
        fitness_mode="lifetime_weighted",
    )
    assert rep.n_seeds == 1
    assert len(rep.per_seed_fitness) == 1
    assert np.isfinite(rep.fitness)
    assert rep.fitness_mode == "lifetime_weighted"
    assert rep.seeds_used == [1000]


# ---------------------------------------------------------------------------
# 2D BSRule conversion via to_bsrule(max_count=8)
# ---------------------------------------------------------------------------


def test_to_bsrule_2d_for_life_like_rule():
    """A FractionalRule tightened to Life's exact rho-bands maps to B3/S23."""
    rule = FractionalRule(3 / 8, 3 / 8, 2 / 8, 3 / 8, 0.30)
    bs = rule.to_bsrule(max_count=DEFAULT_MAX_COUNT_2D)
    assert bs == BSRule(birth=(3,), survival=(2, 3))


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


def test_random_search_observer_2d_returns_sorted():
    reports = random_search_observer_2d(
        n_rules=3, n_seeds=1,
        base_seed=1000, sampler_seed=7,
        grid_shape=(8, 8), timesteps=20,
        fitness_mode="lifetime_weighted",
    )
    assert len(reports) == 3
    fits = [r.fitness for r in reports]
    assert fits == sorted(fits, reverse=True), "expected descending sort"
    for r in reports:
        assert np.isfinite(r.fitness)


# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------


def test_evolutionary_search_observer_2d_returns_history():
    reports, history = evolutionary_search_observer_2d(
        n_generations=2, mu=3, lam=3,
        n_seeds=1, base_seed=1000, sampler_seed=11,
        grid_shape=(8, 8), timesteps=20,
        fitness_mode="lifetime_weighted",
    )
    assert len(reports) == 3
    fits = [r.fitness for r in reports]
    assert fits == sorted(fits, reverse=True)
    # gen 0 + 2 evolved generations.
    assert len(history) == 3
    keys_required = {
        "generation", "best_fitness", "mean_fitness",
        "median_fitness", "population_size",
    }
    for entry in history:
        assert keys_required.issubset(entry.keys())


# ---------------------------------------------------------------------------
# Early-abort behaviour
# ---------------------------------------------------------------------------


def test_2d_evaluate_aborts_cleanly_on_dieoff():
    """A near-empty birth/survive band with low density should die out fast,
    yielding 0 candidates per seed and fitness == 0.0."""
    rule = FractionalRule(0.95, 0.99, 0.95, 0.99, 0.05)
    rep = evaluate_observer_fitness_2d(
        rule, n_seeds=2, base_seed=1000,
        grid_shape=(8, 8), timesteps=20,
        fitness_mode="lifetime_weighted",
    )
    assert rep.aborted_seeds >= 1
    assert rep.fitness == 0.0
