"""Smoke tests for the M-perf profiler skeleton."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from observer_worlds.perf import Profiler


REPO = Path(__file__).resolve().parents[1]


def test_phase_accumulates_wall_time():
    p = Profiler(label="t")
    with p.phase("a"):
        time.sleep(0.01)
    with p.phase("a"):
        time.sleep(0.01)
    with p.phase("b"):
        time.sleep(0.005)
    rep = p.report()
    assert rep["phases_seconds"]["a"] >= 0.015
    assert rep["phases_seconds"]["b"] >= 0.004
    assert rep["phase_unaccounted_seconds"] >= 0


def test_count_and_throughput_in_report():
    p = Profiler(label="t")
    with p.phase("sim"):
        p.count("timesteps", 100)
        p.count("candidates", 5)
    rep = p.report()
    assert rep["counts"]["timesteps"] == 100
    assert rep["counts"]["candidates"] == 5
    assert "timesteps_per_second" in rep["throughput"]


def test_write_json_roundtrip(tmp_path: Path):
    p = Profiler(label="round")
    with p.phase("x"):
        p.count("rollouts", 3)
    out = tmp_path / "perf.json"
    written = p.write_json(out)
    assert written == out
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["label"] == "round"
    assert rep["counts"]["rollouts"] == 3


def test_memory_snapshot_optional_no_crash():
    p = Profiler()
    # Returns a dict if psutil available, else None. Either way, no
    # exception.
    rec = p.snapshot_memory(tag="test")
    assert rec is None or isinstance(rec, dict)


def test_gpu_snapshot_optional_no_crash():
    p = Profiler()
    rec = p.snapshot_gpu_memory(tag="test")
    assert rec is None or isinstance(rec, dict)


def test_profile_experiment_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "observer_worlds.perf.profile_experiment",
         "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "experiment" in result.stdout.lower()


def test_profile_experiment_skeleton_runs(tmp_path: Path):
    """Stage 1: harness runs, writes plan + perf json, does NOT execute
    the underlying experiment."""
    result = subprocess.run(
        [sys.executable, "-m", "observer_worlds.perf.profile_experiment",
         "--experiment", "projection_robustness",
         "--quick", "--out-root", str(tmp_path)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    plan = tmp_path / "plan_projection_robustness.json"
    perf = tmp_path / "perf_projection_robustness_skeleton.json"
    assert plan.exists()
    assert perf.exists()
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    assert plan_data["experiment"] == "projection_robustness"
    assert plan_data["stage"] == 1
