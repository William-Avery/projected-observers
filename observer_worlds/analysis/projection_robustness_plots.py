"""Plotting stubs for Follow-up Topic 1 (Stage 1 skeleton).

Stage 2 will implement the plot generators listed in
``docs/FOLLOWUP_RESEARCH_ROADMAP.md``. For now this module records the
expected plot filenames so the runner and tests can reference them.
"""
from __future__ import annotations

PLOT_FILENAMES: tuple[str, ...] = (
    "hce_by_projection.png",
    "hidden_vs_far_by_projection.png",
    "observer_score_by_projection.png",
    "candidate_count_by_projection.png",
    "mechanism_distribution_by_projection.png",
    "hce_within_revised_class_by_projection.png",
    "initial_projection_delta_by_projection.png",
)


def write_all(*_, **__) -> None:
    """Stage-1 stub. Stage 2 will write the plots in ``PLOT_FILENAMES``."""
    raise NotImplementedError(
        "projection-robustness plots are Stage-2 work; not implemented in "
        "Stage 1 skeleton."
    )
