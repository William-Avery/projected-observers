"""Stats helpers for Follow-up Topic 2 (Stage 1 skeleton)."""
from __future__ import annotations

IDENTITY_METRICS: tuple[str, ...] = (
    "projection_preservation_error",
    "future_similarity_to_host",
    "future_similarity_to_donor",
    "centroid_trajectory_distance",
    "area_trajectory_distance",
    "shape_trajectory_distance",
    "lifetime_change",
    "future_divergence",
    "identity_follow_hidden_score",
    "identity_follow_visible_score",
    "hidden_identity_pull",  # = donor_similarity - host_similarity
)


def summarize(records: list) -> dict:
    return {
        "stage": 1,
        "metrics_recorded": list(IDENTITY_METRICS),
        "n_records": len(records),
    }
