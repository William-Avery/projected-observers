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
    # Stage-1-shape dict still works through the compat shim.
    s = summarize(template)
    assert s["stage"] in (1, 2)
    # Empty aggregation path doesn't crash.
    agg = aggregate_per_projection([], ["mean_threshold"])
    assert agg["per_projection"]["mean_threshold"]["n_candidates"] == 0


def test_aggregate_with_real_rows():
    from observer_worlds.analysis.projection_robustness_stats import (
        aggregate_per_projection,
    )
    rows = [
        {"projection": "mean_threshold", "candidate_id": 0, "track_id": 1,
         "HCE": 0.02, "far_HCE": 0.005, "sham_HCE": 0.0,
         "hidden_vs_far_delta": 0.015, "hidden_vs_sham_delta": 0.02,
         "initial_projection_delta": 0.0, "lifetime": 30,
         "rule_id": "r", "rule_source": "s", "seed": 6000,
         "peak_frame": 5, "far_initial_projection_delta": 0.0,
         "projection_supports_threshold_margin": True,
         "projection_output_kind": "binary"},
        {"projection": "mean_threshold", "candidate_id": 1, "track_id": 2,
         "HCE": 0.03, "far_HCE": 0.01, "sham_HCE": 0.0,
         "hidden_vs_far_delta": 0.02, "hidden_vs_sham_delta": 0.03,
         "initial_projection_delta": 0.0, "lifetime": 25,
         "rule_id": "r", "rule_source": "s", "seed": 6000,
         "peak_frame": 4, "far_initial_projection_delta": 0.0,
         "projection_supports_threshold_margin": True,
         "projection_output_kind": "binary"},
    ]
    agg = aggregate_per_projection(rows, ["mean_threshold"])
    proj = agg["per_projection"]["mean_threshold"]
    assert proj["n_candidates"] == 2
    assert abs(proj["mean_HCE"] - 0.025) < 1e-9
    assert proj["fraction_clean_initial_projection"] == 1.0
