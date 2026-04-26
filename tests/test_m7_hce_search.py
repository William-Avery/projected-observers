"""Tests for M7: HCE-aware fitness, search loops, seed-split protocol,
and holdout-validation interpretation rules.

Combines what the spec lists as five separate test files:
  test_m7_hce_fitness
  test_hce_search_4d
  test_m7_seed_splits
  test_m7_holdout_validation
  test_m7_anti_artifact_penalties
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observer_worlds.search.hce_search_4d import (
    DEFAULT_M7_SCALES,
    DEFAULT_M7_WEIGHTS,
    M7Fitness,
    _PerSeedAggregate,
    _aggregate_to_fitness,
    _normalize,
    evaluate_rule_m7,
    evolutionary_search_hce,
    random_search_hce,
)
from observer_worlds.search import FractionalRule


# ---------------------------------------------------------------------------
# Fitness composition
# ---------------------------------------------------------------------------


def _agg(*, observer=0.0, lifetime=0.0, vs_sham=0.0, vs_far=0.0,
         near_thresh=0.0, excess_global=0.0, fragility=0.0,
         init_proj=0.0, n=1, n_pre=1, n_deg=0):
    a = _PerSeedAggregate()
    a.n_candidates = n
    a.n_total_candidates_pre_filter = n_pre
    a.n_degenerate = n_deg
    a.observer_scores = [observer] * n
    a.lifetimes = [lifetime] * n
    a.hidden_vs_sham_deltas = [vs_sham] * n
    a.hidden_vs_far_deltas = [vs_far] * n
    a.near_threshold_fracs = [near_thresh] * n
    a.excess_global_divs = [excess_global] * n
    a.fragilities = [fragility] * n
    a.initial_proj_deltas = [init_proj] * n
    return a


def _rule():
    return FractionalRule(0.20, 0.30, 0.20, 0.40, 0.20)


def test_default_weights_sum_to_documented_values():
    expected = {"obs", "hce", "local", "life", "recovery",
                "thresh", "global", "fragile", "degenerate"}
    assert set(DEFAULT_M7_WEIGHTS) == expected
    # Specific defaults from spec:
    assert DEFAULT_M7_WEIGHTS["hce"] == 2.0
    assert DEFAULT_M7_WEIGHTS["local"] == 2.0
    assert DEFAULT_M7_WEIGHTS["thresh"] == 1.5


def test_fitness_increases_with_hce():
    a_low = _agg(vs_sham=0.0)
    a_high = _agg(vs_sham=0.10)
    f_low = _aggregate_to_fitness(_rule(), [a_low],
                                  weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    f_high = _aggregate_to_fitness(_rule(), [a_high],
                                   weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    assert f_high.fitness > f_low.fitness


def test_fitness_decreases_with_threshold_artifact():
    a_clean = _agg(vs_sham=0.05, near_thresh=0.0)
    a_thresh = _agg(vs_sham=0.05, near_thresh=0.5)
    f_clean = _aggregate_to_fitness(_rule(), [a_clean],
                                   weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    f_thresh = _aggregate_to_fitness(_rule(), [a_thresh],
                                    weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    assert f_clean.fitness > f_thresh.fitness


def test_fitness_decreases_with_fragility():
    a_robust = _agg(vs_sham=0.05, fragility=0.0)
    a_fragile = _agg(vs_sham=0.05, fragility=1.0)
    f_robust = _aggregate_to_fitness(_rule(), [a_robust],
                                    weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    f_fragile = _aggregate_to_fitness(_rule(), [a_fragile],
                                     weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    assert f_robust.fitness > f_fragile.fitness


def test_fitness_penalizes_global_chaos():
    a_local = _agg(vs_sham=0.05, excess_global=0.0)
    a_global = _agg(vs_sham=0.05, excess_global=0.10)
    f_local = _aggregate_to_fitness(_rule(), [a_local],
                                   weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    f_global = _aggregate_to_fitness(_rule(), [a_global],
                                    weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    assert f_local.fitness > f_global.fitness


def test_fitness_penalizes_degenerate_candidates():
    a_clean = _agg(vs_sham=0.05, n=10, n_pre=10, n_deg=0)
    a_dirty = _agg(vs_sham=0.05, n=10, n_pre=20, n_deg=10)
    f_clean = _aggregate_to_fitness(_rule(), [a_clean],
                                   weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    f_dirty = _aggregate_to_fitness(_rule(), [a_dirty],
                                   weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    assert f_clean.fitness > f_dirty.fitness


def test_fitness_hard_penalty_on_nonzero_initial_projection_delta():
    """Non-zero init delta indicates a bug; should drag fitness toward zero."""
    a_ok = _agg(vs_sham=0.05, init_proj=0.0)
    a_buggy = _agg(vs_sham=0.05, init_proj=0.05)
    f_ok = _aggregate_to_fitness(_rule(), [a_ok],
                                weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    f_buggy = _aggregate_to_fitness(_rule(), [a_buggy],
                                   weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    # Hard penalty is 5.0 * mean_init_delta = 5.0 * 0.05 = 0.25.
    assert f_ok.fitness - f_buggy.fitness >= 0.20


def test_fitness_zero_when_no_aggs():
    f = _aggregate_to_fitness(_rule(), [],
                             weights=DEFAULT_M7_WEIGHTS, scales=DEFAULT_M7_SCALES)
    assert f.fitness == 0.0
    assert f.n_seeds == 0


def test_normalize_clips_to_range():
    assert _normalize(100.0, scale=1.0) == 3.0   # clipped at +3
    assert _normalize(-100.0, scale=1.0) == -3.0
    assert _normalize(0.5, scale=1.0) == 0.5


def test_normalize_zero_scale_returns_zero():
    assert _normalize(1.0, scale=0.0) == 0.0


# ---------------------------------------------------------------------------
# Real evaluation on a tiny grid
# ---------------------------------------------------------------------------


def test_evaluate_rule_m7_returns_finite():
    rule = FractionalRule(0.15, 0.26, 0.09, 0.38, 0.15)
    fit = evaluate_rule_m7(
        rule, seeds=[2000],
        grid_shape=(8, 8, 2, 2), timesteps=20,
        max_candidates=2, horizons=[3], n_replicates=1,
        backend="numpy",
    )
    assert isinstance(fit, M7Fitness)
    assert np.isfinite(fit.fitness)
    # Initial projection delta MUST be near zero (regression invariant).
    assert fit.mean_initial_projection_delta < 1e-3


def test_random_search_hce_returns_sorted():
    reports = random_search_hce(
        n_rules=2, train_seeds=[2000],
        grid_shape=(8, 8, 2, 2), timesteps=20,
        max_candidates=2, horizons=[3], n_replicates=1,
        backend="numpy", sampler_seed=0,
    )
    assert len(reports) == 2
    fits = [r.fitness for r in reports]
    assert fits == sorted(fits, reverse=True)


def test_evolutionary_search_hce_runs():
    reports, history = evolutionary_search_hce(
        n_generations=1, mu=2, lam=2,
        train_seeds=[2000],
        grid_shape=(8, 8, 2, 2), timesteps=20,
        max_candidates=2, horizons=[3], n_replicates=1,
        backend="numpy", sampler_seed=0,
    )
    assert len(reports) == 2
    assert len(history) == 2  # gen 0 + 1 evolved
    for h in history:
        for k in ("generation", "best_fitness", "mean_fitness",
                  "best_observer_score", "best_hidden_vs_sham",
                  "best_hidden_vs_far", "best_near_threshold_fraction"):
            assert k in h


# ---------------------------------------------------------------------------
# Seed-split disjointness (the spec's protocol)
# ---------------------------------------------------------------------------


def test_seed_disjointness_check_passes_when_disjoint():
    """The CLI build_arg_parser → _check_seed_disjointness should not raise
    when the three base seeds are far apart."""
    import argparse
    from observer_worlds.experiments.evolve_4d_hce_rules import (
        _check_seed_disjointness,
    )
    args = argparse.Namespace(
        train_base_seed=1000, train_seeds=5,
        validation_base_seed=4000, validation_seeds=5,
        test_base_seed=3000,
    )
    tr, val, _ = _check_seed_disjointness(args)
    assert set(tr) & set(val) == set()


def test_seed_disjointness_check_fails_on_overlap():
    import argparse
    from observer_worlds.experiments.evolve_4d_hce_rules import (
        _check_seed_disjointness,
    )
    args = argparse.Namespace(
        train_base_seed=1000, train_seeds=5,
        validation_base_seed=1003, validation_seeds=5,  # overlaps train
        test_base_seed=3000,
    )
    with pytest.raises(SystemExit):
        _check_seed_disjointness(args)


# ---------------------------------------------------------------------------
# Holdout interpretation rules
# ---------------------------------------------------------------------------


def _stub_summary(*, m7_obs=0.5, m7_vs_sham=0.05, m7_local=0.10, m7_global=0.10,
                 m7_near=0.05,
                 m4c_obs=0.5, m4c_vs_sham=0.05,
                 m7_audit_all=0.05, m7_audit_far=0.05):
    return {
        "headline_horizon": 10,
        "aggregates": {
            "M7_HCE_optimized": {
                "n": 1, "n_unique_candidates": 10, "mean_observer": m7_obs,
                "mean_lifetime": 50.0,
                "mean_future_div": m7_global, "mean_local_div": m7_local,
                "mean_vs_sham": m7_vs_sham, "mean_vs_far": 0.05,
                "mean_HCE": m7_global, "mean_near_threshold": m7_near,
            },
            "M4C_observer_optimized": {
                "n": 1, "n_unique_candidates": 10, "mean_observer": m4c_obs,
                "mean_lifetime": 50.0,
                "mean_future_div": 0.05, "mean_local_div": 0.05,
                "mean_vs_sham": m4c_vs_sham, "mean_vs_far": 0.05,
                "mean_HCE": 0.05, "mean_near_threshold": 0.20,
            },
        },
        "threshold_audit": {
            "M7_HCE_optimized": [
                {"filter": "all_candidates", "n_candidates": 10,
                 "mean_future_div": m7_audit_all, "mean_vs_sham": m7_audit_all,
                 "mean_vs_far": 0.05,
                 "fraction_future_div_gt_zero": 0.8 if m7_audit_far > 0 else 0.4},
                {"filter": "mean_threshold_margin>0.10", "n_candidates": 8,
                 "mean_future_div": m7_audit_far, "mean_vs_sham": m7_audit_far,
                 "mean_vs_far": 0.05,
                 "fraction_future_div_gt_zero": 0.8 if m7_audit_far > 0 else 0.4},
            ],
        },
        "test_seeds": [3000, 3001],
    }


def test_holdout_interp_finds_non_threshold_hce():
    from observer_worlds.experiments.run_m7_hce_holdout_validation import _interpret
    s = _stub_summary(m7_audit_all=0.05, m7_audit_far=0.04)  # most of effect survives
    msgs = _interpret(s, 10)
    assert any("non-threshold-mediated" in m for m in msgs)


def test_holdout_interp_flags_threshold_artifact():
    from observer_worlds.experiments.run_m7_hce_holdout_validation import _interpret
    s = _stub_summary(m7_audit_all=0.10, m7_audit_far=0.01)  # huge drop under filter
    msgs = _interpret(s, 10)
    assert any("threshold sensitivity" in m for m in msgs)


def test_holdout_interp_flags_observer_collapse():
    from observer_worlds.experiments.run_m7_hce_holdout_validation import _interpret
    # M7 has higher HCE but observer collapsed.
    s = _stub_summary(m7_obs=0.10, m7_vs_sham=0.10, m4c_obs=0.50, m4c_vs_sham=0.05)
    msgs = _interpret(s, 10)
    assert any("hidden sensitivity rather than observer-like" in m for m in msgs)


def test_holdout_interp_flags_balanced_win():
    from observer_worlds.experiments.run_m7_hce_holdout_validation import _interpret
    # Both observer and HCE improved.
    s = _stub_summary(m7_obs=0.55, m7_vs_sham=0.10, m4c_obs=0.50, m4c_vs_sham=0.05)
    msgs = _interpret(s, 10)
    assert any("both observer-like projected structure and hidden causal dependence" in m
              for m in msgs)
