"""CLI / task-registry / workhorse / smoke tests for Follow-up Topic 3."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_help_runs_without_error():
    result = subprocess.run(
        [sys.executable, "-m", runner.__name__, "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "agent-task" in result.stdout.lower()


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
        REGRESSION_MODELS, TASK_METRICS, aggregate_agent_task_results,
    )
    assert "task_score" in TASK_METRICS
    assert "HCE" in TASK_METRICS
    assert "task_score ~ HCE + observer_score" in REGRESSION_MODELS
    s = aggregate_agent_task_results([], [])
    assert s["stage"] == 4


# ---------------------------------------------------------------------------
# Workhorse: task evaluators on a tiny constructed candidate
# ---------------------------------------------------------------------------


def _tiny_cic():
    """Build a CandidateInCell with a real CA4D substrate so the
    evaluators can run (they call _rollout_perturbed which instantiates
    a CA4D)."""
    import numpy as np
    from observer_worlds.experiments._followup_identity_swap import (
        CandidateInCell,
    )
    from observer_worlds.experiments._followup_projection import (
        CandidateRef, initial_4d_state, run_substrate,
    )
    from observer_worlds.search.rules import FractionalRule

    rule = FractionalRule(
        birth_min=0.4, birth_max=0.6,
        survive_min=0.3, survive_max=0.7,
        initial_density=0.5,
    )
    bs = rule.to_bsrule()
    grid = (10, 10, 3, 3)
    state0 = initial_4d_state(grid, 0.5, seed=42)
    stream = run_substrate(bs, state0, 8, backend="numpy")
    peak_mask = np.zeros((10, 10), dtype=np.uint8)
    peak_mask[3:6, 3:6] = 1
    peak_interior = peak_mask.copy()
    cand = CandidateRef(
        candidate_id=0, track_id=0, peak_frame=2, peak_mask=peak_mask,
        peak_interior=peak_interior, peak_bbox=(3, 3, 5, 5), lifetime=5,
    )
    return CandidateInCell(
        cell_id="r|42", rule_id="r", rule_source="src", seed=42,
        cand=cand, state_at_peak=stream[2].copy(), state_stream=stream,
    ), bs


def test_repair_evaluator_returns_one_trial_per_horizon():
    from observer_worlds.experiments._followup_agent_tasks import evaluate_repair
    cic, bs = _tiny_cic()
    trials = evaluate_repair(
        cic=cic, rule_bs=bs, projection_name="mean_threshold",
        horizons=(2, 5), backend="numpy",
    )
    assert len(trials) == 2
    for t in trials:
        assert t.task_name == "repair"
        assert t.task_score is not None
        assert t.repair_score is not None


def test_foraging_evaluator_returns_one_trial_per_horizon():
    from observer_worlds.experiments._followup_agent_tasks import evaluate_foraging
    cic, bs = _tiny_cic()
    trials = evaluate_foraging(
        cic=cic, rule_bs=bs, projection_name="mean_threshold",
        horizons=(2, 5), backend="numpy",
    )
    assert len(trials) == 2
    for t in trials:
        assert t.task_name == "foraging"
        assert t.resource_contact_score is not None
        assert t.movement_toward_resource is not None


def test_memory_evaluator_returns_one_trial_per_horizon():
    import numpy as np
    from observer_worlds.experiments._followup_agent_tasks import evaluate_memory
    cic, bs = _tiny_cic()
    rng = np.random.default_rng(0)
    trials = evaluate_memory(
        cic=cic, rule_bs=bs, projection_name="mean_threshold",
        horizons=(2, 5), backend="numpy", rng=rng,
    )
    assert len(trials) == 2
    for t in trials:
        assert t.task_name == "memory"
        assert t.cue_memory_score is not None


def test_run_tasks_for_candidate_records_hidden_intervention_delta():
    """Stage 5A: hidden_intervention_task_delta is computed for repair
    and memory; foraging is intentionally skipped (smoke quality)."""
    import numpy as np
    from observer_worlds.experiments._followup_agent_tasks import (
        run_tasks_for_candidate,
    )
    cic, bs = _tiny_cic()
    rng = np.random.default_rng(0)
    trials = run_tasks_for_candidate(
        cic=cic, rule_bs=bs, projection_name="mean_threshold",
        horizons=(2, 5), backend="numpy",
        tasks=["repair", "memory"], rng=rng,
        measure_hidden_intervention_delta=True,
    )
    # 2 tasks × 2 horizons = 4 trials; each carries a delta when the
    # perturbation was accepted.
    assert len(trials) == 4
    deltas = [t.hidden_intervention_task_delta for t in trials]
    # Either all trials get a delta (perturbation accepted) or none
    # (perturbation rejected); never partial.
    if any(d is not None for d in deltas):
        assert all(d is not None for d in deltas), (
            f"partial deltas: {deltas}"
        )


def test_run_tasks_disable_delta_returns_none_for_delta():
    import numpy as np
    from observer_worlds.experiments._followup_agent_tasks import (
        run_tasks_for_candidate,
    )
    cic, bs = _tiny_cic()
    rng = np.random.default_rng(0)
    trials = run_tasks_for_candidate(
        cic=cic, rule_bs=bs, projection_name="mean_threshold",
        horizons=(2,), backend="numpy",
        tasks=["repair"], rng=rng,
        measure_hidden_intervention_delta=False,
    )
    for t in trials:
        assert t.hidden_intervention_task_delta is None


def test_run_tasks_for_candidate_stamps_hce_and_observer_score():
    import numpy as np
    from observer_worlds.experiments._followup_agent_tasks import (
        run_tasks_for_candidate,
    )
    cic, bs = _tiny_cic()
    rng = np.random.default_rng(0)
    trials = run_tasks_for_candidate(
        cic=cic, rule_bs=bs, projection_name="mean_threshold",
        horizons=(2, 5), backend="numpy",
        tasks=["repair", "memory"], rng=rng,
    )
    # 2 tasks × 2 horizons = 4 trials.
    assert len(trials) == 4
    for t in trials:
        # HCE and observer_score are set per-candidate, identical across
        # the candidate's trials.
        assert t.hce is not None
        assert t.observer_score is not None
        assert t.observer_score == 5  # lifetime proxy


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_tiny_smoke_writes_full_artifact_set(tmp_path: Path):
    rc = runner.main([
        "--quick",
        "--out-root", str(tmp_path),
        "--label", "tiny_tasks",
        "--n-rules-per-source", "1",
        "--test-seeds", "6000", "6001", "6002",
        "--timesteps", "20",
        "--max-candidates", "3",
        "--horizons", "3", "5",
        "--projection", "mean_threshold",
        "--tasks", "repair", "memory",
        "--n-workers", "1",
        "--grid", "12", "12", "3", "3",
    ])
    assert rc == 0
    out = next(tmp_path.iterdir())
    expected = {
        "config.json", "frozen_manifest.json",
        "task_trials.csv", "task_scores.csv", "hce_task_joined.csv",
        "regression_results.json", "stats_summary.json", "summary.md",
    }
    have = {p.name for p in out.iterdir() if p.is_file()}
    assert expected.issubset(have), f"missing: {expected - have}"
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["tasks"] == ["repair", "memory"]
    summary = json.loads(
        (out / "stats_summary.json").read_text(encoding="utf-8"),
    )
    assert summary["stage"] == 4
    assert "per_task" in summary
