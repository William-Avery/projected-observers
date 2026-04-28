"""Agent-task environments — Stage 1 skeleton.

Defines the metadata for the three task environments that Follow-up
Topic 3 will use. Stage 4 will implement the actual task evaluators
(perturbation + recovery for ``repair``, resource-field dynamics for
``foraging``, transient-cue propagation for ``memory``).

These are *functional agency* tasks — survival, recovery, contact,
delayed response. They are not claims about consciousness; see
``docs/FOLLOWUP_RESEARCH_ROADMAP.md`` for the framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class TaskSpec:
    """Metadata for one task environment.

    Parameters
    ----------
    name
        Stable task identifier used in artifact filenames.
    description
        Short human-readable description.
    primary_metric
        The single number reported as ``task_score`` per trial.
    secondary_metrics
        Auxiliary metrics also recorded per trial.
    requires_resource_field
        ``True`` if the task injects a resource field into the world.
    requires_perturbation
        ``True`` if the task perturbs the candidate at trial start.
    requires_environmental_cue
        ``True`` if the task injects a transient external cue.
    evaluator
        Callable that, in Stage 4, will take a candidate snapshot +
        the rule + a trial seed and return a per-trial result dict.
        ``None`` in Stage 1.
    """
    name: str
    description: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    requires_resource_field: bool = False
    requires_perturbation: bool = False
    requires_environmental_cue: bool = False
    evaluator: Callable | None = None


KNOWN_TASKS: Mapping[str, TaskSpec] = {
    "repair": TaskSpec(
        name="repair",
        description=(
            "Perturb a candidate; measure recovery of shape, area, "
            "trajectory, observer_score over a fixed horizon."
        ),
        primary_metric="recovery_score",
        secondary_metrics=(
            "shape_recovery", "area_recovery", "trajectory_recovery",
            "survival_time", "perturbation_resilience",
        ),
        requires_perturbation=True,
        evaluator=None,
    ),
    "foraging": TaskSpec(
        name="foraging",
        description=(
            "Inject a resource field; score for movement toward and "
            "contact with the resource over the rollout."
        ),
        primary_metric="task_score",
        secondary_metrics=(
            "resource_contact_time", "movement_toward_resource",
            "survival_time",
        ),
        requires_resource_field=True,
        evaluator=None,
    ),
    "memory": TaskSpec(
        name="memory",
        description=(
            "Inject a transient environmental cue; remove it; later "
            "measure whether the candidate's response reflects the cue."
        ),
        primary_metric="cue_memory_score",
        secondary_metrics=("survival_time", "post_cue_response_amplitude"),
        requires_environmental_cue=True,
        evaluator=None,
    ),
}


def available_tasks() -> list[str]:
    return sorted(KNOWN_TASKS)


def get_task(name: str) -> TaskSpec:
    try:
        return KNOWN_TASKS[name]
    except KeyError as e:
        raise KeyError(
            f"unknown task {name!r}; available: {available_tasks()}"
        ) from e
