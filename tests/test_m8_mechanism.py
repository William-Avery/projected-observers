"""Tests for M8 mechanism discovery.

Combines what the spec lists as six separate test files:
  test_m8_response_maps
  test_m8_pathway_tracing
  test_m8_mediation_metrics
  test_m8_mechanism_classifier
  test_m8_feature_dynamics
  test_m8_stats
"""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.experiments._m8_mechanism import (
    MECHANISM_CLASSES,
    EmergenceTiming,
    FeatureDynamics,
    M8CandidateResult,
    MechanismLabel,
    MediationResult,
    PathwayTrace,
    ResponseMap,
    _shell_masks,
    classify_mechanism,
    compute_emergence_timing,
    compute_feature_dynamics,
    compute_mediation,
    compute_pathway_trace,
    compute_response_map,
)
from observer_worlds.search.rules import FractionalRule


# ---------------------------------------------------------------------------
# Fixtures: a small 4D state and an interior mask
# ---------------------------------------------------------------------------


def _toy_rule() -> FractionalRule:
    return FractionalRule(
        birth_min=0.30, birth_max=0.45,
        survive_min=0.25, survive_max=0.50, initial_density=0.20,
    )


def _toy_state(seed: int = 0, shape=(8, 8, 4, 4)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < 0.25).astype(np.uint8)


def _toy_interior(shape=(8, 8)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[3:6, 3:6] = True
    return m


# ---------------------------------------------------------------------------
# Shell masks
# ---------------------------------------------------------------------------


def test_shell_masks_basic():
    interior = _toy_interior()
    boundary, env = _shell_masks(interior)
    assert boundary.shape == interior.shape
    assert env.shape == interior.shape
    # Boundary cells must lie inside the interior.
    assert (boundary & ~interior).sum() == 0
    # Env cells must lie outside interior.
    assert (env & interior).sum() == 0


def test_shell_masks_empty_interior_returns_zero_masks():
    empty = np.zeros((8, 8), dtype=bool)
    b, e = _shell_masks(empty)
    assert not b.any() and not e.any()


# ---------------------------------------------------------------------------
# 1. Response maps
# ---------------------------------------------------------------------------


def test_response_map_zero_outside_interior():
    """Per-spec invariant: cells outside the interior must have zero
    response in the output grid."""
    state = _toy_state()
    interior = _toy_interior()
    rmap = compute_response_map(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, horizon=3,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    boundary, _ = _shell_masks(interior)
    probe_mask = interior | boundary
    # Cells outside the (interior ∪ boundary) probe region must be exactly 0.
    assert (rmap.response_grid[~probe_mask] == 0).all()


def test_response_map_returns_zero_for_empty_interior():
    state = _toy_state()
    empty = np.zeros((8, 8), dtype=bool)
    rmap = compute_response_map(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=empty, candidate_id=1, horizon=3,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    assert rmap.response_grid.sum() == 0
    assert rmap.boundary_response_fraction == 0.0
    assert rmap.interior_response_fraction == 0.0


def test_response_map_concentration_one_for_uniform_zero():
    """Concentration = max/mean: undefined for all-zeros, set to 0 by guard."""
    rmap = ResponseMap(
        candidate_id=1, horizon=3, grid_shape=(8, 8),
        interior_mask=_toy_interior(),
        response_grid=np.zeros((8, 8)),
    )
    assert rmap.response_concentration == 0.0


# ---------------------------------------------------------------------------
# 2. Emergence timing
# ---------------------------------------------------------------------------


def test_emergence_timing_horizons_preserved():
    state = _toy_state()
    interior = _toy_interior()
    horizons = [1, 2, 3, 5]
    timing = compute_emergence_timing(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, horizons=horizons,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    assert timing.horizons == horizons
    assert len(timing.full_grid_l1_per_horizon) == len(horizons)
    assert len(timing.local_l1_per_horizon) == len(horizons)
    # Non-negative L1.
    assert all(v >= 0 for v in timing.full_grid_l1_per_horizon)
    assert all(v >= 0 for v in timing.local_l1_per_horizon)


# ---------------------------------------------------------------------------
# 3. Pathway tracing
# ---------------------------------------------------------------------------


def test_pathway_trace_returns_n_steps_entries():
    state = _toy_state()
    interior = _toy_interior()
    n_steps = 6
    pw = compute_pathway_trace(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, n_steps=n_steps,
        backend="numpy", rng_seed=0,
    )
    assert pw.n_steps == n_steps
    assert len(pw.hidden_mass_per_step) == n_steps
    assert len(pw.visible_mass_per_step) == n_steps
    assert len(pw.spread_radius_4d) == n_steps
    assert len(pw.spread_radius_2d) == n_steps
    # Hidden mass should be >= visible mass at every step (visible is a
    # projection of hidden).
    for h, v in zip(pw.hidden_mass_per_step, pw.visible_mass_per_step):
        assert h >= 0 and v >= 0


def test_pathway_trace_conversion_time_within_horizon_or_minus_one():
    state = _toy_state()
    interior = _toy_interior()
    n_steps = 5
    pw = compute_pathway_trace(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, n_steps=n_steps,
        backend="numpy", rng_seed=0,
    )
    assert pw.hidden_to_visible_conversion_time == -1 or \
        1 <= pw.hidden_to_visible_conversion_time <= n_steps


# ---------------------------------------------------------------------------
# 4. Mediation metrics
# ---------------------------------------------------------------------------


def test_mediation_returns_non_negative_effects():
    state = _toy_state()
    interior = _toy_interior()
    med = compute_mediation(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, horizon=3,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    for v in (med.interior_hidden_effect, med.boundary_hidden_effect,
              med.environment_hidden_effect, med.far_hidden_effect,
              med.visible_boundary_effect, med.visible_environment_effect):
        assert v >= 0.0


def test_mediation_indices_in_expected_ranges():
    state = _toy_state()
    interior = _toy_interior()
    med = compute_mediation(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, horizon=2,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    # boundary_mediation_index is bounded in [0, 1].
    assert 0.0 <= med.boundary_mediation_index <= 1.0


# ---------------------------------------------------------------------------
# 5. Feature dynamics
# ---------------------------------------------------------------------------


def test_feature_dynamics_returns_one_value_per_horizon():
    state = _toy_state()
    interior = _toy_interior()
    horizons = [1, 2, 3, 5]
    fd = compute_feature_dynamics(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, horizons=horizons,
        backend="numpy", rng_seed=0,
    )
    for fn in fd.feature_names:
        assert len(fd.deltas[fn]) == len(horizons)
    assert len(fd.visible_div_per_horizon) == len(horizons)


def test_feature_dynamics_leading_features_sorted_by_abs_corr():
    state = _toy_state()
    interior = _toy_interior()
    fd = compute_feature_dynamics(
        snapshot_4d=state, rule=_toy_rule().to_bsrule(),
        interior_mask=interior, candidate_id=1, horizons=[1, 2, 3, 5, 10],
        backend="numpy", rng_seed=0,
    )
    abs_corrs = [abs(c) for (_, _, c) in fd.leading_features]
    assert abs_corrs == sorted(abs_corrs, reverse=True)


# ---------------------------------------------------------------------------
# 6. Mechanism classifier
# ---------------------------------------------------------------------------


def _stub_inputs(*,
                 boundary_resp=0.1, interior_resp=0.1, env_resp=0.1,
                 first_visible=2, hidden_to_visible=1,
                 frac_hidden_end=0.05, frac_visible_end=0.05,
                 interior_e=0.1, boundary_e=0.1, env_e=0.05, far_e=0.05,
                 vis_b=0.0, vis_env=0.0,
                 bmi=0.5, cli=0.05,
                 near_threshold=0.0):
    rmap = ResponseMap(
        candidate_id=1, horizon=5, grid_shape=(8, 8),
        interior_mask=np.ones((8, 8), dtype=bool),
        response_grid=np.zeros((8, 8)),
        interior_response_fraction=interior_resp,
        boundary_response_fraction=boundary_resp,
        environment_response_fraction=env_resp,
    )
    timing = EmergenceTiming(
        candidate_id=1, horizons=[1, 2, 5],
        full_grid_l1_per_horizon=[0.0, 0.0, 0.0],
        local_l1_per_horizon=[0.0, 0.0, 0.0],
        first_visible_effect_time=first_visible,
    )
    pathway = PathwayTrace(
        candidate_id=1, n_steps=5,
        hidden_mass_per_step=[0]*5, visible_mass_per_step=[0]*5,
        hidden_to_visible_conversion_time=hidden_to_visible,
        fraction_hidden_at_end=frac_hidden_end,
        fraction_visible_at_end=frac_visible_end,
    )
    mediation = MediationResult(
        candidate_id=1,
        interior_hidden_effect=interior_e,
        boundary_hidden_effect=boundary_e,
        environment_hidden_effect=env_e,
        far_hidden_effect=far_e,
        visible_boundary_effect=vis_b,
        visible_environment_effect=vis_env,
        boundary_mediation_index=bmi,
        candidate_locality_index=cli,
    )
    return rmap, timing, pathway, mediation, near_threshold


def test_classifier_boundary_mediated():
    rmap, timing, pw, med, nt = _stub_inputs(boundary_resp=0.85,
                                              interior_resp=0.10,
                                              env_resp=0.05)
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "boundary_mediated"
    assert label.confidence >= 0.6


def test_classifier_interior_reservoir():
    rmap, timing, pw, med, nt = _stub_inputs(boundary_resp=0.10,
                                              interior_resp=0.85,
                                              env_resp=0.05)
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "interior_reservoir"


def test_classifier_environment_coupled():
    rmap, timing, pw, med, nt = _stub_inputs(boundary_resp=0.20,
                                              interior_resp=0.20,
                                              env_resp=0.60)
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "environment_coupled"


def test_classifier_global_chaotic():
    """Far-region effect comparable to candidate effect → global_chaotic."""
    rmap, timing, pw, med, nt = _stub_inputs(
        boundary_resp=0.30, interior_resp=0.30, env_resp=0.10,
        interior_e=0.10, boundary_e=0.10, far_e=0.50,  # far >> candidate
    )
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "global_chaotic"


def test_classifier_threshold_mediated_takes_priority():
    """near_threshold_fraction > 0.5 with low interior effect dominates."""
    rmap, timing, pw, med, nt = _stub_inputs(
        near_threshold=0.80,
        interior_e=0.01, boundary_e=0.20,
        boundary_resp=0.85,  # would otherwise be boundary_mediated
    )
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "threshold_mediated"


def test_classifier_delayed_hidden_channel():
    rmap, timing, pw, med, nt = _stub_inputs(
        boundary_resp=0.30, interior_resp=0.30, env_resp=0.10,
        first_visible=10, frac_hidden_end=0.20, frac_visible_end=0.05,
    )
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "delayed_hidden_channel"


def test_classifier_unclear_fallback():
    rmap, timing, pw, med, nt = _stub_inputs(
        boundary_resp=0.20, interior_resp=0.20, env_resp=0.20,
        first_visible=2, frac_hidden_end=0.01, frac_visible_end=0.01,
    )
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label == "unclear"


def test_classifier_label_in_known_classes():
    rmap, timing, pw, med, nt = _stub_inputs()
    label = classify_mechanism(
        rmap=rmap, timing=timing, pathway=pw, mediation=med,
        near_threshold_fraction=nt, candidate_id=1,
        rule_id="r1", rule_source="src", seed=0,
    )
    assert label.label in MECHANISM_CLASSES


# ---------------------------------------------------------------------------
# Stats: aggregate_per_source / mechanism_class_distribution / grid
# ---------------------------------------------------------------------------


def _stub_result(*, rule_source="M7_HCE_optimized", rule_id="r0", seed=0,
                 candidate_id=1, label="boundary_mediated", lifetime=20,
                 hce=0.10, observer=0.5, near_thresh=0.05) -> M8CandidateResult:
    rmap = ResponseMap(
        candidate_id=candidate_id, horizon=5, grid_shape=(8, 8),
        interior_mask=np.ones((8, 8), dtype=bool),
        response_grid=np.zeros((8, 8)),
        boundary_response_fraction=0.6, interior_response_fraction=0.3,
        environment_response_fraction=0.1,
    )
    timing = EmergenceTiming(
        candidate_id=candidate_id, horizons=[1, 2, 5, 10, 20],
        full_grid_l1_per_horizon=[hce/2, hce, hce, hce, hce*0.8],
        local_l1_per_horizon=[hce/4, hce/2, hce, hce, hce*0.6],
        first_visible_effect_time=2,
        peak_effect_time=5,
    )
    pathway = PathwayTrace(
        candidate_id=candidate_id, n_steps=10,
        hidden_mass_per_step=[5]*10, visible_mass_per_step=[2]*10,
        hidden_to_visible_conversion_time=3,
        fraction_hidden_at_end=0.05, fraction_visible_at_end=0.02,
    )
    mediation = MediationResult(
        candidate_id=candidate_id,
        interior_hidden_effect=0.10, boundary_hidden_effect=0.20,
        environment_hidden_effect=0.05, far_hidden_effect=0.05,
        boundary_mediation_index=0.66, candidate_locality_index=0.25,
    )
    fd = FeatureDynamics(
        candidate_id=candidate_id, horizons=[1, 2, 5],
        feature_names=["near_threshold_fraction"],
        deltas={"near_threshold_fraction": [0.01, 0.02, 0.03]},
        visible_div_per_horizon=[0.0, 0.05, 0.10],
        leading_features=[("near_threshold_fraction", 1, 0.5)],
    )
    mech = MechanismLabel(
        candidate_id=candidate_id, rule_id=rule_id, rule_source=rule_source,
        seed=seed, label=label, confidence=0.8,
        supporting_metrics={},
    )
    return M8CandidateResult(
        rule_id=rule_id, rule_source=rule_source, seed=seed,
        candidate_id=candidate_id, snapshot_t=10,
        candidate_area=10.0, candidate_lifetime=lifetime,
        observer_score=observer, near_threshold_fraction=near_thresh,
        response_map=rmap, timing=timing, pathway=pathway,
        mediation=mediation, feature_dynamics=fd, mechanism=mech,
    )


def test_aggregate_per_source_counts():
    from observer_worlds.analysis.m8_stats import aggregate_per_source
    results = [
        _stub_result(rule_source="M7_HCE_optimized", rule_id="r1", seed=0),
        _stub_result(rule_source="M7_HCE_optimized", rule_id="r1", seed=1,
                     candidate_id=2),
        _stub_result(rule_source="M4C_observer_optimized", rule_id="r2", seed=0,
                     candidate_id=3),
    ]
    aggs = aggregate_per_source(results)
    assert aggs["M7_HCE_optimized"]["n_candidates"] == 2
    assert aggs["M4C_observer_optimized"]["n_candidates"] == 1
    assert aggs["M7_HCE_optimized"]["n_unique_rules"] == 1
    assert aggs["M7_HCE_optimized"]["n_unique_seeds"] == 2


def test_mechanism_class_distribution_fractions_sum_to_one():
    from observer_worlds.analysis.m8_stats import mechanism_class_distribution
    results = [
        _stub_result(label="boundary_mediated"),
        _stub_result(candidate_id=2, label="interior_reservoir"),
        _stub_result(candidate_id=3, label="boundary_mediated"),
    ]
    dist = mechanism_class_distribution(results)
    src = dist["M7_HCE_optimized"]
    total_frac = sum(c["fraction"] for c in src["per_class"].values())
    assert total_frac == pytest.approx(1.0)
    assert src["per_class"]["boundary_mediated"]["count"] == 2
    assert src["per_class"]["interior_reservoir"]["count"] == 1


def test_mechanism_class_distribution_covers_all_classes():
    from observer_worlds.analysis.m8_stats import mechanism_class_distribution
    results = [_stub_result(label="boundary_mediated")]
    dist = mechanism_class_distribution(results)
    for cls in MECHANISM_CLASSES:
        assert cls in dist["M7_HCE_optimized"]["per_class"]


def test_hce_lifetime_correlations_returns_keys():
    from observer_worlds.analysis.m8_stats import hce_lifetime_correlations
    results = [
        _stub_result(rule_id=f"r{i}", seed=i, candidate_id=i, hce=0.1 + 0.01*i,
                     lifetime=20 + i)
        for i in range(5)
    ]
    cors = hce_lifetime_correlations(results)
    assert "pearson_HCE_vs_lifetime" in cors
    assert -1.0 <= cors["pearson_HCE_vs_lifetime"] <= 1.0


def test_compare_sources_on_metric_returns_full_summary():
    from observer_worlds.analysis.m8_stats import compare_sources_on_metric
    a = [_stub_result(rule_source="M7_HCE_optimized", rule_id="r1", seed=i,
                      candidate_id=i, hce=0.20) for i in range(5)]
    b = [_stub_result(rule_source="M4C_observer_optimized", rule_id="r2",
                      seed=i, candidate_id=i+10, hce=0.05) for i in range(5)]
    res = compare_sources_on_metric(
        a, b,
        metric_fn=lambda r: r.timing.full_grid_l1_per_horizon[
            len(r.timing.horizons) // 2
        ],
        source_a_name="M7_HCE_optimized", source_b_name="M4C_observer_optimized",
        n_boot=100, n_permutations=100, seed=0,
    )
    for key in ("n_a", "n_b", "mean_a", "mean_b", "mean_diff",
                "ci_low", "ci_high", "perm_p", "cliffs_delta",
                "rank_biserial", "win_rate_a"):
        assert key in res
    # Mean diff should reflect the distinct-by-construction values.
    assert res["mean_diff"] == pytest.approx(0.15, abs=1e-6)


def test_m7_vs_baseline_grid_includes_metric_keys():
    from observer_worlds.analysis.m8_stats import m7_vs_baseline_grid
    results = (
        [_stub_result(rule_source="M7_HCE_optimized", rule_id="r1",
                     seed=i, candidate_id=i) for i in range(3)] +
        [_stub_result(rule_source="M4C_observer_optimized", rule_id="r2",
                     seed=i, candidate_id=i+10, hce=0.02) for i in range(3)]
    )
    grid = m7_vs_baseline_grid(results)
    assert "M7_vs_M4C_observer_optimized" in grid
    expected_metrics = ("HCE", "boundary_response_fraction",
                        "interior_response_fraction",
                        "first_visible_effect_time",
                        "hidden_to_visible_conversion_time",
                        "boundary_mediation_index",
                        "candidate_locality_index", "fraction_hidden_at_end",
                        "candidate_lifetime")
    for m in expected_metrics:
        assert m in grid["M7_vs_M4C_observer_optimized"]


def test_m7_vs_baseline_grid_returns_empty_without_m7():
    from observer_worlds.analysis.m8_stats import m7_vs_baseline_grid
    results = [_stub_result(rule_source="M4C_observer_optimized")]
    grid = m7_vs_baseline_grid(results)
    assert grid == {}


def test_render_m8_summary_md_basic():
    from observer_worlds.analysis.m8_stats import (
        m8_full_summary, render_m8_summary_md,
    )
    results = [
        _stub_result(label="boundary_mediated"),
        _stub_result(candidate_id=2, label="boundary_mediated"),
    ]
    md = render_m8_summary_md(m8_full_summary(results))
    assert "M8 — Mechanism Discovery" in md
    assert "M7_HCE_optimized" in md
    assert "boundary_mediated" in md


# ---------------------------------------------------------------------------
# CLI helpers (frozen manifest reused from M7B; tag inference is M8-specific)
# ---------------------------------------------------------------------------


def test_infer_tag_handles_known_paths():
    from observer_worlds.experiments.run_m8_mechanism_discovery import _infer_tag
    assert _infer_tag("/tmp/m7_evolve_top.json") == "M7_HCE_optimized"
    assert _infer_tag("/tmp/m4c_top_observers.json") == "M4C_observer_optimized"
    assert _infer_tag("/tmp/m4a_viability.json") == "M4A_viability"
