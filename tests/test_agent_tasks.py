"""CLI / task-registry smoke tests for Follow-up Topic 3 runner (Stage 1)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from observer_worlds.environments import (
    KNOWN_TASKS, TaskSpec, available_tasks, get_task,
)
from observer_worlds.experiments import run_followup_agent_tasks as runner

REPO = Path(__file__).resolve().parents[1]


def test_help_runs_without_error():
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__, "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "agent-task" in result.stdout.lower()


def test_quick_smoke_writes_config_and_summary(tmp_path: Path):
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "smoke_test",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["tasks"] == ["repair", "memory"]
    assert cfg["timesteps"] == 100


def test_unknown_task_rejected_at_argparse(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__,
         "--tasks", "definitely_not_a_task",
         "--out-root", str(tmp_path), "--label", "bad"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


def test_three_tasks_registered():
    assert set(available_tasks()) == {"repair", "foraging", "memory"}


def test_each_task_has_primary_metric():
    for name in available_tasks():
        spec = get_task(name)
        assert isinstance(spec, TaskSpec)
        assert spec.primary_metric, f"{name} missing primary metric"
        # Stage-1 evaluators are intentionally None.
        assert spec.evaluator is None


def test_repair_requires_perturbation():
    assert get_task("repair").requires_perturbation is True


def test_foraging_requires_resource_field():
    assert get_task("foraging").requires_resource_field is True


def test_memory_requires_environmental_cue():
    assert get_task("memory").requires_environmental_cue is True


def test_get_task_unknown_raises():
    with pytest.raises(KeyError):
        get_task("nonexistent")


def test_metric_inventory_is_importable():
    from observer_worlds.analysis.agent_task_stats import (
        REGRESSION_MODELS, TASK_METRICS, summarize,
    )
    assert "task_score" in TASK_METRICS
    assert "HCE" in TASK_METRICS
    assert "task_score ~ HCE + observer_score" in REGRESSION_MODELS
    s = summarize([])
    assert s["stage"] == 1
