"""Tests for M7B production-scale holdout validation.

Combines what the spec lists as five separate test files:
  test_m7b_production_manifest
  test_m7b_stats
  test_m7b_threshold_filtered_success
  test_m7b_interpretation_rules
  test_m7b_seed_disjointness
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from observer_worlds.analysis.m7b_stats import (
    ComparisonResult,
    cliffs_delta,
    cluster_bootstrap_by_groups,
    cohens_d_independent,
    compute_generalization_gap,
    multi_level_bootstrap,
    permutation_test_mean_diff,
    rank_biserial,
    select_interpretations,
    sign_test_p,
)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def test_cliffs_delta_perfect_dominance():
    a = np.array([5, 6, 7, 8])
    b = np.array([1, 2, 3, 4])
    assert cliffs_delta(a, b) == pytest.approx(1.0)


def test_cliffs_delta_anti_dominance():
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    assert cliffs_delta(a, b) == pytest.approx(-1.0)


def test_cliffs_delta_handles_empty():
    assert cliffs_delta(np.array([]), np.array([1.0])) == 0.0


def test_rank_biserial_matches_cliffs_delta_when_no_ties():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 1.0, size=20)
    b = rng.normal(0.0, 1.0, size=20)
    cd = cliffs_delta(a, b)
    rb = rank_biserial(a, b)
    # Without ties they are mathematically identical.
    assert abs(cd - rb) < 0.05


def test_cohens_d_zero_for_identical_distributions():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cohens_d_independent(a, b) == 0.0


def test_cohens_d_handles_constant_input():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0])
    # Pooled SD == 0 → guard returns 0.0.
    assert cohens_d_independent(a, b) == 0.0


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_brackets_true_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=50)
    groups = np.repeat(np.arange(5), 10)
    m, lo, hi = cluster_bootstrap_by_groups(
        values, groups, n_boot=500, seed=42,
    )
    assert lo < m < hi
    assert lo < 0.5 < hi


def test_cluster_bootstrap_handles_single_group():
    values = np.array([0.1, 0.2, 0.3])
    groups = np.array([0, 0, 0])
    m, lo, hi = cluster_bootstrap_by_groups(
        values, groups, n_boot=100, seed=0,
    )
    assert m == lo == hi == pytest.approx(values.mean())


def test_multi_level_bootstrap_returns_three_levels():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, 60)
    rule_ids = np.array([f"r{i % 6}" for i in range(60)])
    seeds = np.array([s for _ in range(6) for s in range(10)])
    out = multi_level_bootstrap(values, rule_ids, seeds, n_boot=200)
    for k in ("by_rule", "by_seed", "by_rule_and_seed"):
        assert k in out
        m, lo, hi = out[k]
        assert lo < m < hi


# ---------------------------------------------------------------------------
# Permutation + sign test
# ---------------------------------------------------------------------------


def test_permutation_p_small_for_strong_effect():
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = permutation_test_mean_diff(a, b, n_permutations=200, seed=0)
    assert p < 0.05


def test_permutation_p_large_for_no_effect():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 30)
    b = rng.normal(0.0, 1.0, 30)
    p = permutation_test_mean_diff(a, b, n_permutations=200, seed=0)
    assert p > 0.05


def test_sign_test_p_high_for_zero_diffs():
    p = sign_test_p(np.array([0.0, 0.0, 0.0]))
    assert p > 0.05


# ---------------------------------------------------------------------------
# Generalization gap
# ---------------------------------------------------------------------------


def test_generalization_gap_basic():
    train = {"r1": 10.0, "r2": 8.0}
    val = {"r1": 9.0, "r2": 7.5}
    test = {"r1": 7.0, "r2": 6.0}
    gap = compute_generalization_gap(train, val, test)
    assert gap["n"] == 2
    assert gap["mean_train"] == pytest.approx(9.0)
    # train→test drop = (7+6)/2 - (10+8)/2 = 6.5 - 9.0 = -2.5
    assert gap["mean_train_to_test_drop"] == pytest.approx(-2.5)


def test_generalization_gap_no_overlap():
    train = {"r1": 10.0}
    val = {"r2": 9.0}
    test = {"r3": 8.0}
    gap = compute_generalization_gap(train, val, test)
    assert gap["n"] == 0


# ---------------------------------------------------------------------------
# Frozen manifest
# ---------------------------------------------------------------------------


def test_file_hash_returns_consistent_sha256(tmp_path: Path):
    from observer_worlds.experiments.run_m7b_production_holdout import _file_hash
    p = tmp_path / "test.txt"
    p.write_text("hello world")
    h1 = _file_hash(str(p))
    h2 = _file_hash(str(p))
    assert h1["sha256"] == h2["sha256"]
    assert h1["sha256"] == \
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_file_hash_returns_none_for_missing_path():
    from observer_worlds.experiments.run_m7b_production_holdout import _file_hash
    assert _file_hash("/nonexistent/path") is None
    assert _file_hash(None) is None


def test_autodetect_seed_splits_finds_config(tmp_path: Path):
    """Walking up from m7_rules path should find the M7 evolve config.json
    that has train_seeds / validation_seeds."""
    from observer_worlds.experiments.run_m7b_production_holdout import (
        _autodetect_m7_seed_splits,
    )
    # Create fake M7 evolve dir with config.json + nested top_hce_rules.json.
    evolve_dir = tmp_path / "m7_evolve_20260426"
    evolve_dir.mkdir()
    (evolve_dir / "config.json").write_text(json.dumps({
        "train_seeds": [1000, 1001, 1002],
        "validation_seeds": [4000, 4001],
    }))
    rules_path = evolve_dir / "top_hce_rules.json"
    rules_path.write_text("[]")
    auto = _autodetect_m7_seed_splits(str(rules_path))
    assert auto.get("m7_train_seeds") == [1000, 1001, 1002]
    assert auto.get("m7_validation_seeds") == [4000, 4001]


def test_autodetect_seed_splits_returns_empty_when_no_config(tmp_path: Path):
    from observer_worlds.experiments.run_m7b_production_holdout import (
        _autodetect_m7_seed_splits,
    )
    rules_path = tmp_path / "rules.json"
    rules_path.write_text("[]")
    auto = _autodetect_m7_seed_splits(str(rules_path))
    assert auto == {} or "m7_train_seeds" not in auto


# ---------------------------------------------------------------------------
# Seed disjointness
# ---------------------------------------------------------------------------


def test_seed_disjointness_passes_when_disjoint():
    from observer_worlds.experiments.run_m7b_production_holdout import (
        check_seed_disjointness,
    )
    err = check_seed_disjointness(
        test_seeds=[5000, 5001],
        autodetected={"m7_train_seeds": [1000], "m7_validation_seeds": [4000]},
    )
    assert err is None


def test_seed_disjointness_fails_when_test_overlaps_train():
    from observer_worlds.experiments.run_m7b_production_holdout import (
        check_seed_disjointness,
    )
    err = check_seed_disjointness(
        test_seeds=[1000, 5001],
        autodetected={"m7_train_seeds": [1000], "m7_validation_seeds": [4000]},
    )
    assert err is not None
    assert "test ∩ train" in err


def test_seed_disjointness_fails_when_train_overlaps_validation():
    from observer_worlds.experiments.run_m7b_production_holdout import (
        check_seed_disjointness,
    )
    err = check_seed_disjointness(
        test_seeds=[5000],
        autodetected={"m7_train_seeds": [1000], "m7_validation_seeds": [1000]},
    )
    assert err is not None
    assert "train ∩ validation" in err


def test_seed_disjointness_passes_when_autodetected_empty():
    """No M7 evolve config available → no overlap to detect."""
    from observer_worlds.experiments.run_m7b_production_holdout import (
        check_seed_disjointness,
    )
    err = check_seed_disjointness(test_seeds=[5000, 5001], autodetected={})
    assert err is None


# ---------------------------------------------------------------------------
# Interpretation rules
# ---------------------------------------------------------------------------


def _stub_comparison(*, mean_diff: float, ci_low: float, ci_high: float,
                    perm_p: float = 0.005) -> ComparisonResult:
    return ComparisonResult(
        metric="x", source_a="A", source_b="B", n_a=10, n_b=10,
        mean_a=mean_diff + 1.0, mean_b=1.0, mean_diff=mean_diff,
        bootstrap_by_rule=(mean_diff, ci_low, ci_high),
        bootstrap_by_seed=(mean_diff, ci_low, ci_high),
        bootstrap_by_rule_and_seed=(mean_diff, ci_low, ci_high),
        cliffs_delta=0.5, rank_biserial=0.5, cohens_d=0.5,
        permutation_p=perm_p, win_rate_a=0.7,
    )


def test_interpret_strong_success():
    """M7 beats both M4C and M4A on HCE significantly, observer not lost."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "observer_score": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "hidden_vs_far_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[
            {"filter": "all_candidates", "mean_future_div": 0.30,
             "mean_vs_sham": 0.30, "mean_vs_far": 0.10,
             "fraction_future_div_gt_zero": 0.9, "n_candidates": 30},
            {"filter": "mean_threshold_margin>0.10",
             "mean_future_div": 0.27, "mean_vs_sham": 0.27,
             "mean_vs_far": 0.10, "fraction_future_div_gt_zero": 0.85,
             "n_candidates": 25},
        ],
    )
    assert any("supports the core claim" in m for m in msgs)


def test_interpret_partial_success():
    """M7 beats M4C on HCE but loses observer significantly."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "observer_score": _stub_comparison(
                mean_diff=-0.20, ci_low=-0.30, ci_high=-0.10
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[
            {"filter": "all_candidates", "mean_future_div": 0.30,
             "mean_vs_sham": 0.30, "mean_vs_far": 0.10,
             "fraction_future_div_gt_zero": 0.9, "n_candidates": 30},
            {"filter": "mean_threshold_margin>0.10",
             "mean_future_div": 0.27, "mean_vs_sham": 0.27,
             "mean_vs_far": 0.10, "fraction_future_div_gt_zero": 0.85,
             "n_candidates": 25},
        ],
    )
    assert any("partially traded off" in m for m in msgs)


def test_interpret_failure_threshold_artifact():
    """M7 future_div drops to near zero under threshold filter."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "observer_score": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[
            {"filter": "all_candidates", "mean_future_div": 0.30,
             "mean_vs_sham": 0.30, "mean_vs_far": 0.10,
             "fraction_future_div_gt_zero": 0.9, "n_candidates": 30},
            {"filter": "mean_threshold_margin>0.10",
             "mean_future_div": 0.05, "mean_vs_sham": 0.05,
             "mean_vs_far": 0.01, "fraction_future_div_gt_zero": 0.30,
             "n_candidates": 25},
        ],
    )
    assert any("exploited projection-threshold sensitivity" in m for m in msgs)


def test_interpret_not_replicated():
    """M7 has zero or negative mean diff vs M4C on HCE."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=-0.05, ci_low=-0.10, ci_high=0.0
            ),
            "observer_score": _stub_comparison(
                mean_diff=0.0, ci_low=-0.05, ci_high=0.05
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.05, ci_low=0.0, ci_high=0.10
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[
            {"filter": "all_candidates", "mean_future_div": 0.20,
             "mean_vs_sham": 0.20, "mean_vs_far": 0.10,
             "fraction_future_div_gt_zero": 0.7, "n_candidates": 30},
            {"filter": "mean_threshold_margin>0.10",
             "mean_future_div": 0.15, "mean_vs_sham": 0.15,
             "mean_vs_far": 0.08, "fraction_future_div_gt_zero": 0.6,
             "n_candidates": 25},
        ],
    )
    assert any("did not replicate" in m for m in msgs)


def test_interpret_local_not_global():
    """When M7 vs M4C on hidden_vs_far_delta is positive, the
    'effect is candidate-local' message is appended."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "observer_score": _stub_comparison(
                mean_diff=0.05, ci_low=0.0, ci_high=0.10
            ),
            "hidden_vs_far_delta": _stub_comparison(
                mean_diff=0.05, ci_low=0.01, ci_high=0.10
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[],
    )
    assert any("candidate-local" in m for m in msgs)


# ---------------------------------------------------------------------------
# Threshold-filtered success criterion
# ---------------------------------------------------------------------------


def test_threshold_filtered_success_classifies_high_retention():
    """Audit with retention >= 50% under strict filter should NOT trigger
    threshold-failure interpretation."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "observer_score": _stub_comparison(
                mean_diff=0.0, ci_low=-0.05, ci_high=0.05
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[
            {"filter": "all_candidates", "mean_future_div": 0.20,
             "mean_vs_sham": 0.20, "mean_vs_far": 0.05,
             "fraction_future_div_gt_zero": 0.85, "n_candidates": 30},
            {"filter": "mean_threshold_margin>0.10",
             "mean_future_div": 0.15, "mean_vs_sham": 0.15,
             "mean_vs_far": 0.04, "fraction_future_div_gt_zero": 0.80,
             "n_candidates": 25},
        ],
    )
    # Should NOT include the threshold-exploit message (retention 75%).
    assert not any("exploited projection-threshold" in m for m in msgs)


def test_threshold_filtered_success_classifies_low_retention():
    """Retention < 30% triggers the threshold-failure interpretation."""
    msgs = select_interpretations(
        m7_vs_m4c={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
            "observer_score": _stub_comparison(
                mean_diff=0.0, ci_low=-0.05, ci_high=0.05
            ),
        },
        m7_vs_m4a={
            "hidden_vs_sham_delta": _stub_comparison(
                mean_diff=0.10, ci_low=0.05, ci_high=0.15
            ),
        },
        m7_vs_2d_observer=None,
        m7_threshold_audit=[
            {"filter": "all_candidates", "mean_future_div": 0.30,
             "mean_vs_sham": 0.30, "mean_vs_far": 0.05,
             "fraction_future_div_gt_zero": 0.9, "n_candidates": 30},
            {"filter": "mean_threshold_margin>0.10",
             "mean_future_div": 0.05, "mean_vs_sham": 0.05,
             "mean_vs_far": 0.01, "fraction_future_div_gt_zero": 0.30,
             "n_candidates": 25},
        ],
    )
    assert any("exploited projection-threshold" in m for m in msgs)


# ---------------------------------------------------------------------------
# Frozen manifest produced by build_frozen_manifest
# ---------------------------------------------------------------------------


def test_build_frozen_manifest_has_all_required_keys(tmp_path: Path):
    import argparse
    from observer_worlds.experiments.run_m7b_production_holdout import (
        build_frozen_manifest,
    )
    args = argparse.Namespace(
        m7_rules="/tmp/m7.json", m4c_rules="/tmp/m4c.json",
        m4a_rules="/tmp/m4a.json", optimized_2d_rules=None,
        n_rules_per_source=10, timesteps=500, grid=[64, 64, 8, 8],
        max_candidates=40, hce_replicates=5, horizons=[5, 10, 20, 40, 80],
        backend="numpy", n_bootstrap=2000, n_permutations=2000,
        test_seeds=[5000, 5001],
    )
    manifest = build_frozen_manifest(args, tmp_path,
                                    autodetected_seeds={"m7_train_seeds": [1000]})
    # Spec-required keys.
    for key in ("timestamp_utc", "git", "command", "python_version",
                "input_rule_files", "output_dir", "test_seeds", "config"):
        assert key in manifest, f"manifest missing key {key}"
    assert "commit" in manifest["git"]
    assert "dirty" in manifest["git"]
    assert manifest["test_seeds"] == [5000, 5001]
    assert manifest["m7_train_seeds"] == [1000]
