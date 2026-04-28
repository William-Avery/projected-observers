"""CLI / config smoke tests for Follow-up Topic 1 runner (Stage 1)."""
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


def test_quick_smoke_writes_config_and_summary(tmp_path: Path):
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "smoke_test",
    ])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    out = runs[0]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    # Smoke defaults applied.
    assert cfg["n_rules_per_source"] == 1
    assert cfg["timesteps"] == 100
    assert cfg["max_candidates"] == 5
    assert cfg["projections"] == [
        "mean_threshold", "max_projection", "parity_projection",
    ]
    assert cfg["horizons"] == [5, 10]
    assert cfg["hce_replicates"] == 1
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Stage 1" in summary


def test_unknown_projection_rejected(tmp_path: Path):
    with pytest.raises(SystemExit):
        runner.main([
            "--out-root", str(tmp_path),
            "--label", "bad",
            "--projections", "does_not_exist",
        ])


def test_full_default_uses_all_six_projections(tmp_path: Path):
    rc = runner.main([
        "--out-root", str(tmp_path),
        "--label", "full_default",
        "--timesteps", "10",  # keep it light
        "--n-rules-per-source", "1",
        "--test-seeds", "6000",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert len(cfg["projections"]) == 6


def test_metric_inventory_is_importable():
    from observer_worlds.analysis.projection_robustness_stats import (
        PROJECTION_METRICS, project_metrics_template, summarize,
    )
    assert "HCE" in PROJECTION_METRICS
    assert "boundary_and_interior_co_mediated_fraction" in PROJECTION_METRICS
    template = project_metrics_template(["mean_threshold"])
    assert template["mean_threshold"]["HCE"] is None
    summary = summarize(template)
    assert summary["stage"] == 1
