"""CLI / config smoke tests for Follow-up Topic 2 runner (Stage 1)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from observer_worlds.experiments import run_followup_hidden_identity_swap as runner

REPO = Path(__file__).resolve().parents[1]


def test_help_runs_without_error():
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__, "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "hidden identity swap" in result.stdout.lower()


def test_quick_smoke_writes_config_and_summary(tmp_path: Path):
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "smoke_test",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["max_pairs"] == 10
    assert cfg["timesteps"] == 100
    assert cfg["matching_mode"] in runner.MATCHING_MODES


def test_matching_mode_is_validated(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__,
         "--matching-mode", "definitely_not_a_mode",
         "--out-root", str(tmp_path), "--label", "bad"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode != 0


def test_metric_inventory_is_importable():
    from observer_worlds.analysis.identity_swap_stats import (
        IDENTITY_METRICS, summarize,
    )
    assert "hidden_identity_pull" in IDENTITY_METRICS
    assert "projection_preservation_error" in IDENTITY_METRICS
    s = summarize([])
    assert s["stage"] == 1
    assert s["n_records"] == 0
