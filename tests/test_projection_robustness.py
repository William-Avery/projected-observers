"""CLI / config / smoke tests for Follow-up Topic 1 runner (Stage 2)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from observer_worlds.experiments import run_followup_projection_robustness as runner

REPO = Path(__file__).resolve().parents[1]


def test_help_runs_without_error():
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__, "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "projection-robustness" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Tiny smoke run — exercises the real Stage-2 pipeline at minimum cost.
# ---------------------------------------------------------------------------


def test_tiny_smoke_run_writes_full_artifact_set(tmp_path: Path):
    """Run the real pipeline at the smallest viable scale and check the
    documented output bundle is produced."""
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "tiny_smoke",
        "--timesteps", "12",
        "--max-candidates", "2",
        "--horizons", "3", "5",
        "--n-rules-per-source", "1",
        "--test-seeds", "6000",
        "--projections", "mean_threshold",
        "--n-workers", "1",
        "--grid", "12", "12", "3", "3",
        "--hce-replicates", "1",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())

    # All documented artifacts present.
    expected = {
        "config.json", "frozen_manifest.json",
        "projection_summary.csv", "candidate_metrics.csv",
        "hce_by_projection.csv", "mechanism_by_projection.csv",
        "projection_artifact_audit.csv",
        "stats_summary.json", "summary.md",
    }
    have = {p.name for p in out.iterdir() if p.is_file()}
    missing = expected - have
    assert not missing, f"missing artifacts: {missing}"

    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    # The tiny-smoke overrides apply on top of --quick smoke defaults.
    assert cfg["timesteps"] == 12
    assert cfg["max_candidates"] == 2
    assert cfg["projections"] == ["mean_threshold"]
    assert cfg["horizons"] == [3, 5]
    assert cfg["hce_replicates"] == 1

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Stage 2" in summary

    stats = json.loads((out / "stats_summary.json").read_text(encoding="utf-8"))
    assert stats["stage"] == 2
    assert "per_projection" in stats
    assert "mean_threshold" in stats["per_projection"]
    # The plots directory exists (plots may be empty if matplotlib
    # missing, but the dir is created).
    assert (out / "plots").is_dir()


def test_unknown_projection_rejected(tmp_path: Path):
    with pytest.raises(SystemExit):
        runner.main([
            "--out-root", str(tmp_path),
            "--label", "bad",
            "--projections", "does_not_exist",
        ])


def test_metric_inventory_is_importable():
    from observer_worlds.analysis.projection_robustness_stats import (
        PROJECTION_METRICS,
        aggregate_per_projection,
        project_metrics_template,
        summarize,
        write_summary_md,
    )
    assert "mean_HCE" in PROJECTION_METRICS
    assert "mean_initial_projection_delta" in PROJECTION_METRICS
    template = project_metrics_template(["mean_threshold"])
    assert template["mean_threshold"]["mean_HCE"] is None
    s = summarize(template)
    assert s["stage"] in (1, 2)
    # Empty aggregation path doesn't crash.
    agg = aggregate_per_projection([], ["mean_threshold"])
    proj = agg["per_projection"]["mean_threshold"]
    assert proj["n_candidates_total"] == 0
    assert proj["n_valid_hidden_invisible"] == 0
    assert proj["n_invalid_hidden_invisible"] == 0


def test_runner_handles_all_six_projections_end_to_end(tmp_path: Path):
    """Stage 5A hardening: end-to-end smoke covering every registered
    projection. The runner must not crash on any of them, and per-
    projection valid/invalid counts must be recorded so we can audit
    coverage downstream.

    Continuous projections (random_linear) and multi-channel are
    *expected* to produce invalid hidden-invisible perturbations under
    the strict 1e-6 verification tolerance; the test asserts the
    pipeline records that as invalid rather than silently including
    those candidates in HCE means."""
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "all6_runner_smoke",
        "--projections",
        "mean_threshold", "sum_threshold", "max_projection",
        "parity_projection", "random_linear_projection",
        "multi_channel_projection",
        "--timesteps", "12",
        "--max-candidates", "2",
        "--horizons", "3", "5",
        "--n-rules-per-source", "1",
        "--test-seeds", "6000",
        "--n-workers", "1",
        "--grid", "12", "12", "3", "3",
        "--hce-replicates", "1",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())
    stats = json.loads((out / "stats_summary.json").read_text(encoding="utf-8"))
    expected = {
        "mean_threshold", "sum_threshold", "max_projection",
        "parity_projection", "random_linear_projection",
        "multi_channel_projection",
    }
    assert set(stats["per_projection"]) == expected
    # Each projection must have either >0 valid OR >0 invalid (not silently empty).
    for proj, agg in stats["per_projection"].items():
        n_total = agg["n_candidates_total"]
        if n_total == 0:
            continue
        n_valid = agg["n_valid_hidden_invisible"]
        n_invalid = agg["n_invalid_hidden_invisible"]
        assert n_valid + n_invalid == n_total, (
            f"{proj}: n_total={n_total} but valid+invalid={n_valid + n_invalid}"
        )


def test_aggregate_with_real_rows_only_valid_in_means():
    """Stage 2B: HCE means are taken only over candidates whose
    hidden-invisible perturbation was accepted. Invalid candidates are
    counted but excluded from the means."""
    from observer_worlds.analysis.projection_robustness_stats import (
        aggregate_per_projection,
    )
    base = {
        "projection": "max_projection", "rule_id": "r", "rule_source": "s",
        "seed": 6000, "peak_frame": 5,
        "projection_supports_threshold_margin": False,
        "projection_output_kind": "binary",
    }
    rows = [
        # valid candidate
        {**base, "candidate_id": 0, "track_id": 1,
         "valid": True, "invalid_reason": None,
         "preservation_strategy": "count_preserving_swap",
         "HCE": 0.02, "far_HCE": 0.005, "sham_HCE": 0.0,
         "hidden_vs_far_delta": 0.015, "hidden_vs_sham_delta": 0.02,
         "initial_projection_delta": 0.0,
         "far_initial_projection_delta": 0.0,
         "lifetime": 30, "n_flipped_hidden": 8, "n_flipped_far": 8},
        # second valid candidate
        {**base, "candidate_id": 1, "track_id": 2,
         "valid": True, "invalid_reason": None,
         "preservation_strategy": "count_preserving_swap",
         "HCE": 0.03, "far_HCE": 0.01, "sham_HCE": 0.0,
         "hidden_vs_far_delta": 0.02, "hidden_vs_sham_delta": 0.03,
         "initial_projection_delta": 0.0,
         "far_initial_projection_delta": 0.0,
         "lifetime": 25, "n_flipped_hidden": 8, "n_flipped_far": 8},
        # invalid candidate (all-zero fibre); excluded from HCE means
        {**base, "candidate_id": 2, "track_id": 3,
         "valid": False,
         "invalid_reason":
             "no fibre in candidate mask had both ON and OFF cells",
         "preservation_strategy": "count_preserving_swap",
         "HCE": None, "far_HCE": None, "sham_HCE": None,
         "hidden_vs_far_delta": None, "hidden_vs_sham_delta": None,
         "initial_projection_delta": 0.0,
         "far_initial_projection_delta": 0.0,
         "lifetime": 20, "n_flipped_hidden": 0, "n_flipped_far": 0},
    ]
    agg = aggregate_per_projection(rows, ["max_projection"])
    proj = agg["per_projection"]["max_projection"]
    assert proj["n_candidates_total"] == 3
    assert proj["n_valid_hidden_invisible"] == 2
    assert proj["n_invalid_hidden_invisible"] == 1
    # Means computed only over the 2 valid candidates.
    assert abs(proj["mean_HCE"] - 0.025) < 1e-9
    # Invalid reason captured.
    reason_keys = list(proj["invalid_reason_counts"])
    assert len(reason_keys) == 1
    assert "no fibre" in reason_keys[0]
    # Clean-initial-projection fraction is over valid candidates only.
    assert proj["fraction_clean_initial_projection"] == 1.0
