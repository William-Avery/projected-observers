"""Stats helpers for Follow-up Topic 3 (Stage 1 skeleton)."""
from __future__ import annotations

TASK_METRICS: tuple[str, ...] = (
    "task_score",
    "survival_time",
    "recovery_score",
    "perturbation_resilience",
    "cue_memory_score",
    "resource_contact_time",
    "movement_toward_resource",
    "HCE",
    "observer_score",
    "hidden_intervention_effect_on_task_score",
)

REGRESSION_MODELS: tuple[str, ...] = (
    "task_score ~ HCE",
    "task_score ~ observer_score",
    "task_score ~ HCE + observer_score",
    "task_score ~ HCE + observer_score + mechanism_class",
)


def summarize(trials: list) -> dict:
    return {
        "stage": 1,
        "metrics_recorded": list(TASK_METRICS),
        "regression_models_planned": list(REGRESSION_MODELS),
        "n_trials": len(trials),
    }
