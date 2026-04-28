"""Stats helpers for Follow-up Topic 1 (Stage 1 skeleton).

Stage 2 will implement the per-projection HCE / mechanism aggregation.
For now this module exposes only the metric inventory and a stub
summary builder, both of which are useful for the test surface and
for the experiment runner to import without circulars.
"""
from __future__ import annotations

from typing import Iterable


# Metric names recorded per projection (per-source, per-class, etc.).
PROJECTION_METRICS: tuple[str, ...] = (
    "n_candidates",
    "observer_score",
    "HCE",
    "hidden_vs_sham_delta",
    "hidden_vs_far_delta",
    "initial_projection_delta",
    "near_threshold_fraction",  # only when supported
    "boundary_and_interior_co_mediated_fraction",
    "global_chaotic_fraction",
    "HCE_within_co_mediated",
    "HCE_within_global_chaotic",
)


def project_metrics_template(projections: Iterable[str]) -> dict:
    """Empty-result template: ``{projection_name: {metric: None}}``.

    Used by the runner to allocate the result dict before the sweep.
    """
    return {
        proj: {m: None for m in PROJECTION_METRICS}
        for proj in projections
    }


def summarize(per_projection: dict) -> dict:
    """Stage-1 stub. Returns the input as-is plus a header.

    Stage 2 will compute cross-projection comparisons here.
    """
    return {
        "stage": 1,
        "metrics_recorded": list(PROJECTION_METRICS),
        "per_projection": per_projection,
    }
