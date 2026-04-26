"""Tests for M8D global-chaotic decomposition.

Combines what the spec lists as four separate test files:
  test_m8d_distance_controls
  test_m8d_background_sensitivity
  test_m8d_global_subclasses
  test_m8d_stabilization
"""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.detection.morphology import MorphologyResult
from observer_worlds.experiments._m8c_validation import FarControlInfo
from observer_worlds.experiments._m8d_decomposition import (
    M8D_GLOBAL_SUBCLASSES,
    M8D_MECHANISM_CLASSES,
    DistanceEffect,
    M8DCandidateResult,
    _make_disc_mask_at,
    _periodic_dist,
    _translate_mask_at_distance,
    fit_decay_curve,
    measure_background_sensitivity,
    measure_multi_distance_effects,
    relabel_global_chaotic,
)
from observer_worlds.search.rules import FractionalRule


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def test_periodic_dist_handles_wraparound():
    # On a 10x10 grid, points (1,1) and (9,9) are 2*sqrt(2) ~ 2.83 apart
    # under periodic distance (not 8*sqrt(2) ~ 11.3).
    d = _periodic_dist((1, 1), (9, 9), (10, 10))
    assert 2.5 < d < 3.0


def test_translate_mask_at_distance_finds_translation():
    m = np.zeros((48, 48), dtype=bool); m[10:13, 10:13] = True
    translated, dist = _translate_mask_at_distance(
        m, target_distance=20.0, rng_seed=0,
    )
    assert translated is not None
    assert (translated & m).sum() == 0
    assert translated.sum() == m.sum()
    # Distance should be reasonable; within 10 cells of target.
    assert abs(dist - 20.0) < 12.0


def test_make_disc_mask_periodic():
    m = _make_disc_mask_at((20, 20), 0.0, 0.0, 2.0)
    # Origin disc should wrap onto opposite corners on the 20x20 grid.
    assert m[0, 0]
    assert m.sum() > 0


# ---------------------------------------------------------------------------
# multi-distance probe
# ---------------------------------------------------------------------------


def _toy_state(seed: int = 0, shape=(8, 8, 4, 4)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < 0.25).astype(np.uint8)


def test_multi_distance_effects_includes_required_probes():
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45, survive_min=0.25, survive_max=0.50,
        initial_density=0.20,
    ).to_bsrule()
    mask = np.zeros((8, 8), dtype=bool); mask[2:6, 2:6] = True
    effects = measure_multi_distance_effects(
        snapshot_4d=state, rule=rule, candidate_mask_2d=mask, horizon=2,
        n_replicates=1, backend="numpy", rng_seed=0,
    )
    names = {e.name for e in effects}
    assert "body" in names
    assert "env_shell" in names
    assert "antipode" in names
    assert any(n.startswith("random_") for n in names)


def test_fit_decay_curve_handles_few_points():
    eff = [
        DistanceEffect("body", 0.0, 0.0, 4, 0.05, 0.005),
        DistanceEffect("env_shell", 1.5, 1.0, 8, 0.04, 0.002),
    ]
    fit = fit_decay_curve(eff)
    # Excludes "body" → only 1 non-body point, < 3 → returns zeros.
    assert fit["n_points"] <= 1
    assert fit["slope"] == 0.0


def test_fit_decay_curve_negative_slope_for_decreasing():
    eff = [
        DistanceEffect("env_shell", 1.5, 1.0, 8, 0.04, 0.005),
        DistanceEffect("far_2r", 4.0, 2.0, 8, 0.03, 0.003),
        DistanceEffect("far_5r", 10.0, 5.0, 8, 0.02, 0.001),
        DistanceEffect("antipode", 30.0, 15.0, 8, 0.01, 0.0005),
    ]
    fit = fit_decay_curve(eff)
    assert fit["slope"] < 0.0
    assert fit["n_points"] >= 3


# ---------------------------------------------------------------------------
# Background sensitivity
# ---------------------------------------------------------------------------


def test_background_sensitivity_returns_distribution():
    state = _toy_state()
    rule = FractionalRule(
        birth_min=0.30, birth_max=0.45, survive_min=0.25, survive_max=0.50,
        initial_density=0.20,
    ).to_bsrule()
    mask = np.zeros((8, 8), dtype=bool); mask[2:6, 2:6] = True
    bg = measure_background_sensitivity(
        snapshot_4d=state, rule=rule, candidate_mask_2d=mask, horizon=2,
        n_samples=4, sample_size=4, backend="numpy", rng_seed=0,
    )
    assert "mean" in bg and "p95" in bg and "n_samples" in bg
    assert bg["n_samples"] >= 1
    assert bg["p95"] >= bg["mean"]


# ---------------------------------------------------------------------------
# Relabel classifier
# ---------------------------------------------------------------------------


def _stub_dist_effects(slope=0.0, body=0.10, env=0.08, near=0.06, far=0.04,
                       antipode=0.04):
    """Construct distance effects so that fit_decay_curve returns the
    requested slope (approximately)."""
    return [
        DistanceEffect("body", 0.0, 0.0, 9, body, body / (9 * 16)),
        DistanceEffect("env_shell", 1.5, 0.5, 12, env, env / (12 * 16)),
        DistanceEffect("far_2r", 6.0, 2.0, 9, near, near / (9 * 16)),
        DistanceEffect("far_5r", 15.0, 5.0, 9, far, far / (9 * 16)),
        DistanceEffect("antipode", 32.0, 10.0, 9, antipode,
                       antipode / (9 * 16)),
    ]


def test_relabel_returns_known_subclass():
    effects = _stub_dist_effects()
    bg = {"mean": 0.01, "p95": 0.02, "p99": 0.03, "n_samples": 16,
          "samples": []}
    feats = {"near_threshold_fraction": 0.10, "hidden_volatility": 0.20,
             "mean_threshold_margin": 0.30, "hidden_temporal_persistence": 0.5,
             "mean_hidden_entropy": 0.5, "hidden_spatial_autocorrelation": 0.0,
             "mean_active_fraction": 0.3, "hidden_heterogeneity": 0.4}
    label, conf, metrics = relabel_global_chaotic(
        distance_effects=effects, background=bg, feature_audit=feats,
        stabilization={}, body_effect=0.10, far_effect=0.06,
    )
    assert label in M8D_GLOBAL_SUBCLASSES


def test_relabel_threshold_volatility_artifact():
    effects = _stub_dist_effects()
    bg = {"mean": 0.01, "p95": 0.02, "p99": 0.03, "n_samples": 16,
          "samples": []}
    feats = {"near_threshold_fraction": 0.60, "hidden_volatility": 0.20,
             "mean_threshold_margin": 0.30, "hidden_temporal_persistence": 0.5,
             "mean_hidden_entropy": 0.5, "hidden_spatial_autocorrelation": 0.0,
             "mean_active_fraction": 0.3, "hidden_heterogeneity": 0.4}
    label, _, _ = relabel_global_chaotic(
        distance_effects=effects, background=bg, feature_audit=feats,
        stabilization={}, body_effect=0.10, far_effect=0.06,
    )
    assert label == "threshold_volatility_artifact"


def test_relabel_far_control_artifact_when_only_antipode_hot():
    effects = _stub_dist_effects(near=0.005, far=0.002, antipode=0.05)
    bg = {"mean": 0.001, "p95": 0.002, "p99": 0.003, "n_samples": 16,
          "samples": []}
    feats = {"near_threshold_fraction": 0.05, "hidden_volatility": 0.10,
             "mean_threshold_margin": 0.30, "hidden_temporal_persistence": 0.5,
             "mean_hidden_entropy": 0.5, "hidden_spatial_autocorrelation": 0.0,
             "mean_active_fraction": 0.3, "hidden_heterogeneity": 0.4}
    label, _, _ = relabel_global_chaotic(
        distance_effects=effects, background=bg, feature_audit=feats,
        stabilization={}, body_effect=0.10, far_effect=0.05,
    )
    assert label == "far_control_artifact"


def test_relabel_local_window_removes_global():
    """If local_window stabilization variant no longer fires global,
    relabel as broad_hidden_coupling."""
    effects = _stub_dist_effects()
    bg = {"mean": 0.001, "p95": 0.002, "p99": 0.003, "n_samples": 16,
          "samples": []}
    feats = {"near_threshold_fraction": 0.05, "hidden_volatility": 0.10,
             "mean_threshold_margin": 0.30, "hidden_temporal_persistence": 0.5,
             "mean_hidden_entropy": 0.5, "hidden_spatial_autocorrelation": 0.0,
             "mean_active_fraction": 0.3, "hidden_heterogeneity": 0.4}
    stabilization = {
        "baseline": {"global_chaotic_label_would_fire": True},
        "local_window": {"global_chaotic_label_would_fire": False},
    }
    label, _, _ = relabel_global_chaotic(
        distance_effects=effects, background=bg, feature_audit=feats,
        stabilization=stabilization, body_effect=0.10, far_effect=0.06,
    )
    assert label == "broad_hidden_coupling"


def test_m8d_classes_includes_subclasses():
    for cls in M8D_GLOBAL_SUBCLASSES:
        assert cls in M8D_MECHANISM_CLASSES


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _stub_morph(cls="thick_candidate", area=30):
    return MorphologyResult(
        morphology_class=cls, area=area,
        erosion1_interior_size=area // 4, erosion2_interior_size=area // 8,
        boundary_size=area // 2, environment_size=area,
        can_separate_boundary_from_interior=(
            cls in ("thick_candidate", "very_thick_candidate")
        ),
        can_classify_environment_coupled=True,
    )


def _stub_far_info():
    return FarControlInfo(
        candidate_radius=4.0, candidate_diameter=8.0,
        far_control_translation=(20, 20),
        far_control_distance=30.0,
        far_control_distance_over_radius=7.5,
        far_control_valid=True,
        far_control_min_distance_required=20.0,
        far_control_projected_activity_diff=0.05,
        far_control_hidden_activity_diff=0.05,
    )


def _stub_far_eff(raw=0.05):
    from observer_worlds.experiments._m8b_spatial import RegionEffect
    return RegionEffect(
        region_name="far_validated", n_perturbed_cells_2d=10,
        n_flipped_cells_4d=80,
        region_hidden_effect=raw, region_local_divergence=raw,
        region_global_divergence=raw, region_response_fraction=0.25,
        region_effect_per_cell=raw / 160.0,
        region_effect_per_flipped_cell=raw / 80.0,
    )


def _stub_region_effects():
    from observer_worlds.experiments._m8b_spatial import RegionEffect
    e = lambda pc, raw: RegionEffect(
        region_name="x", n_perturbed_cells_2d=10, n_flipped_cells_4d=80,
        region_hidden_effect=raw, region_local_divergence=raw,
        region_global_divergence=raw, region_response_fraction=0.25,
        region_effect_per_cell=pc, region_effect_per_flipped_cell=raw / 80.0,
    )
    return {
        "interior": e(0.001, 0.01), "boundary": e(0.001, 0.01),
        "environment": e(0.0001, 0.005), "whole": e(0.001, 0.05),
    }


def _stub_m8d_result(*, source="M7_HCE_optimized", base="global_chaotic",
                     final="global_instability", body_over_bg=2.0,
                     candidate_id=1, seed=0,
                     features=None, distance_effects=None):
    return M8DCandidateResult(
        rule_id="r0", rule_source=source, seed=seed,
        candidate_id=candidate_id, snapshot_t=10,
        candidate_area=30, candidate_lifetime=50,
        near_threshold_fraction=0.05,
        morphology=_stub_morph(),
        far_control=_stub_far_info(),
        region_effects=_stub_region_effects(),
        far_effect=_stub_far_eff(),
        distance_effects=distance_effects or _stub_dist_effects(),
        background_mean=0.01, background_p95=0.02, background_p99=0.03,
        body_over_background=body_over_bg, far_over_background=1.5,
        feature_audit=(features or {
            "near_threshold_fraction": 0.05, "hidden_volatility": 0.10,
            "mean_threshold_margin": 0.30, "hidden_temporal_persistence": 0.5,
            "mean_hidden_entropy": 0.5, "hidden_spatial_autocorrelation": 0.0,
            "mean_active_fraction": 0.3, "hidden_heterogeneity": 0.4,
        }),
        stabilization={"baseline": {"global_chaotic_label_would_fire": True},
                       "local_window": {"global_chaotic_label_would_fire": True}},
        base_mechanism_label=base, base_mechanism_confidence=0.5,
        final_mechanism_label=final, final_mechanism_confidence=0.6,
        relabel_metrics={},
    )


def test_aggregate_per_source_global_count():
    from observer_worlds.analysis.m8d_stats import aggregate_per_source
    rs = [
        _stub_m8d_result(base="global_chaotic", candidate_id=1),
        _stub_m8d_result(base="global_chaotic", candidate_id=2, seed=1),
        _stub_m8d_result(base="interior_reservoir", candidate_id=3, seed=2),
    ]
    a = aggregate_per_source(rs)["M7_HCE_optimized"]
    assert a["n_thick"] == 3
    assert a["n_global_base"] == 2
    assert a["global_chaotic_base_thick_fraction"] == pytest.approx(2/3)


def test_global_subclass_distribution_fractions():
    from observer_worlds.analysis.m8d_stats import global_subclass_distribution
    rs = [
        _stub_m8d_result(base="global_chaotic",
                         final="global_instability", candidate_id=i, seed=i)
        for i in range(3)
    ] + [
        _stub_m8d_result(base="global_chaotic",
                         final="threshold_volatility_artifact",
                         candidate_id=i + 100, seed=i + 100)
        for i in range(2)
    ]
    d = global_subclass_distribution(rs)["M7_HCE_optimized"]
    assert d["n_global_base"] == 5
    total_frac = sum(d["per_class"][c]["fraction"]
                     for c in M8D_GLOBAL_SUBCLASSES)
    assert total_frac == pytest.approx(1.0)


def test_stabilization_reclassification_rates_local_window():
    from observer_worlds.analysis.m8d_stats import (
        stabilization_reclassification_rates,
    )
    rs = [_stub_m8d_result(base="global_chaotic", candidate_id=i, seed=i)
          for i in range(4)]
    # All baseline=fire, local_window=fire → no_longer_fires_fraction = 0.
    out = stabilization_reclassification_rates(rs)["M7_HCE_optimized"]
    bv = out["by_variant"]
    assert bv["baseline"]["no_longer_fires_fraction"] == 0.0
    assert bv["local_window"]["no_longer_fires_fraction"] == 0.0


def test_select_interpretations_global_instability():
    from observer_worlds.analysis.m8d_stats import (
        m8d_full_summary, select_interpretations,
    )
    rs = [
        _stub_m8d_result(base="global_chaotic",
                        final="global_instability",
                        candidate_id=i, seed=i)
        for i in range(5)
    ]
    msgs = select_interpretations(m8d_full_summary(rs))
    assert any("system-wide hidden instability" in m for m in msgs)


def test_select_interpretations_no_global():
    from observer_worlds.analysis.m8d_stats import (
        m8d_full_summary, select_interpretations,
    )
    rs = [
        _stub_m8d_result(base="interior_reservoir",
                        final="interior_reservoir",
                        candidate_id=i, seed=i)
        for i in range(3)
    ]
    msgs = select_interpretations(m8d_full_summary(rs))
    assert any("No M7 global_chaotic" in m for m in msgs)


def test_render_m8d_summary_md_basic():
    from observer_worlds.analysis.m8d_stats import (
        m8d_full_summary, render_m8d_summary_md,
    )
    rs = [
        _stub_m8d_result(base="global_chaotic",
                        final="global_instability",
                        candidate_id=i, seed=i) for i in range(3)
    ]
    md = render_m8d_summary_md(m8d_full_summary(rs))
    assert "M8D" in md
    assert "Global subclass distribution" in md
    assert "global_instability" in md


def test_cli_arg_parser_quick():
    from observer_worlds.experiments.run_m8d_global_chaos_decomposition \
        import build_arg_parser, _quick
    args = _quick(build_arg_parser().parse_args([
        "--m7-rules", "/tmp/x.json", "--quick",
    ]))
    assert args.timesteps <= 100
    assert args.grid == [32, 32, 4, 4]
    assert args.background_n_samples == 4
