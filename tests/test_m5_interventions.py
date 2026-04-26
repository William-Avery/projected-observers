"""Tests for M5 intervention runner + plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observer_worlds.experiments._m5_interventions import (
    INTERVENTION_TYPES,
    CandidateInterventionReport,
    InterventionTrajectory,
    aggregate_intervention_summaries,
    run_candidate_interventions,
)
from observer_worlds.search import FractionalRule
from observer_worlds.worlds import CA4D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _viable_snapshot_8x8x2x2(seed: int = 1000):
    """Run a viable rule for a few steps to produce a non-trivial 4D state."""
    rule = FractionalRule(0.15, 0.26, 0.09, 0.38, 0.15)
    bsrule = rule.to_bsrule()
    ca = CA4D(shape=(8, 8, 2, 2), rule=bsrule, backend="numpy")
    ca.initialize_random(0.15, np.random.default_rng(seed))
    for _ in range(15):
        ca.step()
    return ca.state.copy(), bsrule


def _disc_masks_8x8():
    interior = np.zeros((8, 8), dtype=bool)
    interior[3:5, 3:5] = True
    boundary = np.zeros((8, 8), dtype=bool)
    boundary[2:6, 2:6] = True
    boundary &= ~interior
    env = np.zeros((8, 8), dtype=bool)
    env[1:7, 1:7] = True
    env &= ~interior
    env &= ~boundary
    return interior, boundary, env


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


def test_run_candidate_interventions_returns_all_types():
    snapshot, rule = _viable_snapshot_8x8x2x2()
    interior, boundary, env = _disc_masks_8x8()
    report = run_candidate_interventions(
        snapshot, rule, interior, boundary, env,
        track_id=0, track_age=10, snapshot_t=15, observer_score=0.4,
        n_steps=6, flip_fraction=0.5, backend="numpy", seed=42,
    )
    assert set(report.trajectories.keys()) == set(INTERVENTION_TYPES)
    for kind in INTERVENTION_TYPES:
        traj = report.trajectories[kind]
        assert traj.n_steps == 6
        assert len(traj.full_grid_l1) == 6
        assert len(traj.candidate_footprint_l1) == 6


def test_run_candidate_interventions_degenerate_masks():
    snapshot, rule = _viable_snapshot_8x8x2x2()
    empty = np.zeros((8, 8), dtype=bool)
    interior, boundary, env = _disc_masks_8x8()
    report = run_candidate_interventions(
        snapshot, rule, empty, boundary, env,
        track_id=0, track_age=10, snapshot_t=15,
        n_steps=6, flip_fraction=0.5, backend="numpy", seed=0,
    )
    assert report.trajectories == {}
    assert report.intervention_summary == {}


def test_intervention_trajectory_summary_fields_finite():
    snapshot, rule = _viable_snapshot_8x8x2x2()
    interior, boundary, env = _disc_masks_8x8()
    report = run_candidate_interventions(
        snapshot, rule, interior, boundary, env,
        track_id=0, track_age=10, snapshot_t=15,
        n_steps=5, flip_fraction=0.5, backend="numpy", seed=1,
    )
    for traj in report.trajectories.values():
        assert np.isfinite(traj.mean_full_grid_l1)
        assert np.isfinite(traj.mean_candidate_footprint_l1)
        assert np.isfinite(traj.auc_full_grid_l1)
        assert np.isfinite(traj.final_area_ratio)


def test_intervention_n_steps_matches_trajectory_length():
    snapshot, rule = _viable_snapshot_8x8x2x2()
    interior, boundary, env = _disc_masks_8x8()
    for n in (3, 7, 12):
        report = run_candidate_interventions(
            snapshot, rule, interior, boundary, env,
            track_id=0, track_age=10, snapshot_t=15,
            n_steps=n, flip_fraction=0.5, backend="numpy", seed=2,
        )
        for traj in report.trajectories.values():
            assert len(traj.full_grid_l1) == n
            assert len(traj.candidate_active_orig) == n


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _fake_trajectory(kind: str, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    full = rng.uniform(0.0, 1.0, n_steps)
    cand = rng.uniform(0.0, 1.0, n_steps)
    t = InterventionTrajectory(
        intervention_type=kind, snapshot_t=10, n_steps=n_steps,
        flip_fraction=0.5,
        full_grid_l1=full.tolist(),
        candidate_footprint_l1=cand.tolist(),
        candidate_active_orig=[5] * n_steps,
        candidate_active_intervened=[3] * n_steps,
    )
    t.mean_full_grid_l1 = float(full.mean())
    t.mean_candidate_footprint_l1 = float(cand.mean())
    t.auc_full_grid_l1 = float(full.sum())
    t.final_survival = True
    t.final_area_ratio = 0.6
    return t


def _fake_report(track_id: int, seed: int) -> CandidateInterventionReport:
    rep = CandidateInterventionReport(
        track_id=track_id, track_age=20, snapshot_t=10, observer_score=0.5,
        n_steps=4, flip_fraction=0.5,
        interior_size=6, boundary_size=8, env_size=20,
    )
    for i, kind in enumerate(INTERVENTION_TYPES):
        traj = _fake_trajectory(kind, n_steps=4, seed=seed + i)
        rep.trajectories[kind] = traj
        rep.intervention_summary[kind] = {
            "mean_full_grid_l1": traj.mean_full_grid_l1,
            "mean_candidate_footprint_l1": traj.mean_candidate_footprint_l1,
            "auc_full_grid_l1": traj.auc_full_grid_l1,
            "final_survival": float(traj.final_survival),
            "final_area_ratio": traj.final_area_ratio,
        }
    return rep


def test_aggregate_intervention_summaries_groups_by_type():
    reports = [_fake_report(0, 100), _fake_report(1, 200)]
    agg = aggregate_intervention_summaries(reports)
    assert set(agg.keys()) == set(INTERVENTION_TYPES)
    for kind in INTERVENTION_TYPES:
        assert "mean_full_grid_l1" in agg[kind]
        # Mean of 2 known values per kind.
        expected = float(np.mean([
            reports[0].intervention_summary[kind]["mean_full_grid_l1"],
            reports[1].intervention_summary[kind]["mean_full_grid_l1"],
        ]))
        assert agg[kind]["mean_full_grid_l1"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------


def test_plots_render_to_files(tmp_path: Path):
    from observer_worlds.analysis import write_all_m5_plots
    reports = [_fake_report(i, 100 + i * 10) for i in range(3)]
    write_all_m5_plots(reports, tmp_path, per_candidate_max=2)
    expected = [
        "aggregate_divergence_full_grid.png",
        "aggregate_divergence_candidate_footprint.png",
        "intervention_heatmap_full_grid.png",
        "intervention_heatmap_candidate_footprint.png",
        "intervention_summary_bars.png",
    ]
    for name in expected:
        f = tmp_path / name
        assert f.exists() and f.stat().st_size > 0, name
    per_dir = tmp_path / "per_candidate"
    assert per_dir.is_dir()
    pngs = list(per_dir.glob("*.png"))
    # 2 candidates × 2 plots each = 4 PNGs.
    assert len(pngs) >= 2


def test_plots_render_with_empty_reports(tmp_path: Path):
    from observer_worlds.analysis import write_all_m5_plots
    write_all_m5_plots([], tmp_path)
    assert (tmp_path / "aggregate_divergence_full_grid.png").exists()
    assert (tmp_path / "intervention_summary_bars.png").exists()
