"""Tests for M6 hidden-causal-dependence experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observer_worlds.experiments._m6_hidden_causal import (
    HiddenCausalReport,
    HiddenCausalTrajectory,
    PERTURBATION_TYPES,
    aggregate_hce_stats,
    compare_hce_paired,
    run_hidden_causal_experiment,
)
from observer_worlds.metrics.causality_score import (
    apply_hidden_shuffle_intervention,
)
from observer_worlds.search import FractionalRule
from observer_worlds.worlds import CA4D, project


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _viable_snapshot_8x8x2x2(seed: int = 1000):
    rule = FractionalRule(0.15, 0.26, 0.09, 0.38, 0.15)
    bsrule = rule.to_bsrule()
    ca = CA4D(shape=(8, 8, 2, 2), rule=bsrule, backend="numpy")
    ca.initialize_random(0.15, np.random.default_rng(seed))
    for _ in range(15):
        ca.step()
    return ca.state.copy(), bsrule


def _disc_mask_8x8(half_size: int = 2):
    interior = np.zeros((8, 8), dtype=bool)
    cy = cx = 4
    interior[cy - half_size:cy + half_size, cx - half_size:cx + half_size] = True
    return interior


# ---------------------------------------------------------------------------
# Construction-property tests
# ---------------------------------------------------------------------------


def test_hidden_invisible_preserves_projection_at_t0():
    """The defining invariant: hidden_invisible perturbation must leave
    project(state) byte-identical at t=0. This is what makes HCE > 0
    impossible in 2D systems."""
    snapshot, _ = _viable_snapshot_8x8x2x2()
    interior = _disc_mask_8x8()
    rng = np.random.default_rng(42)
    perturbed = apply_hidden_shuffle_intervention(snapshot, interior, rng)
    p_orig = project(snapshot, method="mean_threshold", theta=0.5)
    p_pert = project(perturbed, method="mean_threshold", theta=0.5)
    assert np.array_equal(p_orig, p_pert), \
        "hidden_shuffle must preserve mean-threshold projection at t=0"


def test_hidden_invisible_actually_changes_4d_state():
    """The perturbation must actually do something — not just leave the
    state untouched. It should differ in (z,w) arrangement somewhere."""
    snapshot, _ = _viable_snapshot_8x8x2x2()
    interior = _disc_mask_8x8()
    rng = np.random.default_rng(7)
    perturbed = apply_hidden_shuffle_intervention(snapshot, interior, rng)
    # Inside the interior columns, at least one cell should differ (with high
    # probability for a non-trivial column).
    interior_4d = interior[:, :, None, None]
    diffs_in_interior = ((snapshot != perturbed) & interior_4d).sum()
    assert diffs_in_interior > 0


# ---------------------------------------------------------------------------
# Runner contract
# ---------------------------------------------------------------------------


def test_run_hidden_causal_basic_contract():
    snapshot, rule = _viable_snapshot_8x8x2x2()
    interior = _disc_mask_8x8()
    report = run_hidden_causal_experiment(
        snapshot, rule, interior,
        track_id=0, track_age=15, snapshot_t=15, observer_score=0.5,
        n_steps=6, n_replicates=3, backend="numpy", seed=42,
    )
    assert isinstance(report, HiddenCausalReport)
    assert report.n_steps == 6
    assert report.n_replicates == 3
    assert isinstance(report.hidden_invisible, HiddenCausalTrajectory)
    assert isinstance(report.visible_match_count, HiddenCausalTrajectory)
    assert len(report.hidden_invisible.full_grid_l1_mean) == 6
    assert len(report.visible_match_count.full_grid_l1_mean) == 6
    # All scalars finite.
    for v in (report.HCE, report.visible_final_l1, report.hce_to_visible_ratio,
              report.hce_immediate_check):
        assert np.isfinite(v)


def test_run_hidden_causal_degenerate_mask_returns_empty():
    snapshot, rule = _viable_snapshot_8x8x2x2()
    empty = np.zeros((8, 8), dtype=bool)
    report = run_hidden_causal_experiment(
        snapshot, rule, empty,
        track_id=0, track_age=15, snapshot_t=15,
        n_steps=5, n_replicates=2, backend="numpy",
    )
    assert report.interior_size == 0
    assert report.HCE == 0.0
    assert report.hidden_invisible.n_replicates == 0


def test_HCE_is_finite_and_immediate_is_small():
    """Sanity: HCE itself can be anywhere (>=0 typically), but the
    immediate-divergence check must be tiny because the perturbation
    preserves projection at t=0 (so divergence at step 1 = downstream
    effect of one CA step on a permuted state, not a direct flip)."""
    snapshot, rule = _viable_snapshot_8x8x2x2()
    interior = _disc_mask_8x8()
    rep = run_hidden_causal_experiment(
        snapshot, rule, interior,
        track_id=0, track_age=15, snapshot_t=15,
        n_steps=8, n_replicates=4, backend="numpy", seed=1,
    )
    assert np.isfinite(rep.HCE)
    # The immediate (step 1) divergence under hidden_invisible reflects
    # how much one CA step differs given identical projection at t=0
    # but different hidden state. It should be small relative to a
    # bit-matched visible perturbation at the same step.
    assert rep.hce_immediate_check < rep.visible_match_count.full_grid_l1_mean[0] + 1e-9


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _fake_report(track_id: int, hce: float, vis: float) -> HiddenCausalReport:
    hi = HiddenCausalTrajectory(
        perturbation_type="hidden_invisible", n_steps=5, n_replicates=3,
        full_grid_l1_mean=[0.0, 0.05, 0.1, 0.15, hce],
        full_grid_l1_std=[0.0] * 5,
        candidate_footprint_l1_mean=[0.0, 0.05, 0.1, 0.15, hce],
        candidate_footprint_l1_std=[0.0] * 5,
        mean_immediate_l1=0.0, mean_final_l1=hce,
        mean_auc=hce * 5, mean_n_flips=5.0,
    )
    vt = HiddenCausalTrajectory(
        perturbation_type="visible_match_count", n_steps=5, n_replicates=3,
        full_grid_l1_mean=[vis, vis, vis, vis, vis],
        full_grid_l1_std=[0.0] * 5,
        candidate_footprint_l1_mean=[vis] * 5,
        candidate_footprint_l1_std=[0.0] * 5,
        mean_immediate_l1=vis, mean_final_l1=vis,
        mean_auc=vis * 5, mean_n_flips=5.0,
    )
    return HiddenCausalReport(
        track_id=track_id, snapshot_t=10, track_age=20,
        observer_score=0.5, interior_size=4, n_steps=5, n_replicates=3,
        flip_fraction_for_visible=0.5,
        hidden_invisible=hi, visible_match_count=vt,
        HCE=hce, visible_final_l1=vis,
        hce_to_visible_ratio=(hce / vis if vis > 0 else 0.0),
        hce_immediate_check=0.001,
    )


def test_aggregate_hce_stats_basic():
    reports = [_fake_report(i, hce=0.05 + i * 0.01, vis=0.10) for i in range(5)]
    stats = aggregate_hce_stats(reports)
    assert stats["n_candidates"] == 5
    assert stats["mean_HCE"] == pytest.approx(np.mean([r.HCE for r in reports]))
    assert stats["fraction_hce_positive"] == 1.0
    # Sign test should report low p-value for 5/5 positives.
    assert stats["one_sample_p_hce_gt_zero"] < 0.1


def test_aggregate_hce_stats_zero_when_all_zero():
    reports = [_fake_report(i, hce=0.0, vis=0.05) for i in range(5)]
    stats = aggregate_hce_stats(reports)
    assert stats["mean_HCE"] == 0.0
    assert stats["fraction_hce_positive"] == 0.0


def test_compare_hce_paired_sign_correctness():
    coh = [_fake_report(i, hce=0.10, vis=0.05) for i in range(4)]
    sh = [_fake_report(i, hce=0.04, vis=0.05) for i in range(4)]
    paired = compare_hce_paired(coh, sh)
    assert paired["n_paired"] == 4
    # diff = 0.10 - 0.04 = +0.06 for all 4
    assert paired["mean_diff_coh_minus_shuf"] == pytest.approx(0.06)
    assert paired["n_coherent_wins"] == 4
    # Sign test on 4/4 positives ~ p = 2 * 1/16 = 0.125 (small N)
    assert paired["sign_test_p"] < 0.5


def test_compare_hce_paired_falls_back_to_rank_pairing():
    """When track IDs don't overlap (the common case for coherent vs
    shuffled simulations), rank-pairing should kick in."""
    coh = [_fake_report(i, hce=0.10 + i * 0.01, vis=0.05) for i in range(3)]
    sh = [_fake_report(i + 100, hce=0.05, vis=0.05) for i in range(3)]
    paired = compare_hce_paired(coh, sh)
    assert paired["n_paired"] == 3
    assert paired["comparison_strategy"] == "rank_pairing"
    # All coh HCE > sh HCE in this fixture, regardless of pairing strategy.
    assert paired["mean_diff_coh_minus_shuf"] > 0


def test_compare_hce_paired_id_strategy_when_ids_overlap():
    coh = [_fake_report(i, hce=0.10, vis=0.05) for i in range(3)]
    sh = [_fake_report(i, hce=0.05, vis=0.05) for i in range(3)]
    paired = compare_hce_paired(coh, sh)
    assert paired["comparison_strategy"] == "id_pairing"


def test_compare_hce_paired_returns_full_keys_on_empty():
    paired = compare_hce_paired([], [])
    for k in ("n_paired", "n_coherent_wins", "n_shuffled_wins",
              "comparison_strategy", "mean_diff_coh_minus_shuf"):
        assert k in paired


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------


def test_m6_plots_render_to_files(tmp_path: Path):
    from observer_worlds.analysis import write_all_m6_plots
    coh = [_fake_report(i, hce=0.05 + i * 0.01, vis=0.08) for i in range(4)]
    sh = [_fake_report(i, hce=0.03 + i * 0.005, vis=0.08) for i in range(4)]
    write_all_m6_plots({"coherent": coh, "shuffled": sh}, tmp_path)
    expected = [
        "aggregate_divergence_coherent.png",
        "aggregate_divergence_shuffled.png",
        "hce_distribution_coherent.png",
        "hce_distribution_shuffled.png",
        "hce_vs_visible_coherent.png",
        "hce_vs_visible_shuffled.png",
        "hce_coherent_vs_shuffled_paired.png",
        "hce_boxplot_by_condition.png",
    ]
    for name in expected:
        f = tmp_path / name
        assert f.exists() and f.stat().st_size > 0, name


def test_m6_plots_handle_empty(tmp_path: Path):
    from observer_worlds.analysis import write_all_m6_plots
    write_all_m6_plots({"coherent": []}, tmp_path)
    assert (tmp_path / "aggregate_divergence_coherent.png").exists()
