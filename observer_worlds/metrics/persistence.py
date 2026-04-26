"""Persistence-based observer-candidate scoring (M1).

Each :class:`Track` is summarised by a :class:`CandidateScore`; tracks meeting
the size/age/variation criteria flagged by the :class:`DetectionConfig` are
treated as observer candidates by the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from observer_worlds.detection.tracking import Track
from observer_worlds.utils.config import DetectionConfig


@dataclass
class CandidateScore:
    """Output of the persistence-based observer-candidate filter.

    M1 only computes a few summary stats.  M2+ adds time/memory/causality/etc.
    """

    track_id: int
    age: int
    length: int
    is_candidate: bool
    boundedness: float
    internal_variation: float
    mean_area: float
    max_area: int
    reasons: list[str] = field(default_factory=list)


def _score_one(
    track: Track,
    grid_shape: tuple[int, int],
    config: DetectionConfig,
) -> CandidateScore:
    nx, ny = grid_shape
    grid_cells = max(int(nx) * int(ny), 1)

    if track.area_history:
        areas = np.asarray(track.area_history, dtype=np.float64)
        mean_area = float(areas.mean())
        max_area = int(areas.max())
        std_area = float(areas.std())
        boundedness = 1.0 / (1.0 + std_area / (mean_area + 1e-6))
    else:
        mean_area = 0.0
        max_area = 0
        std_area = 0.0
        boundedness = 0.0

    if track.interior_history:
        interior_counts = np.asarray(
            [int(m.sum()) for m in track.interior_history], dtype=np.float64
        )
        internal_variation = float(interior_counts.std())
    else:
        internal_variation = 0.0

    age = track.age
    length = track.length

    reasons: list[str] = []
    if age < config.min_age:
        reasons.append("too_short")
    if grid_cells > 0 and (max_area / grid_cells) > config.max_area_fraction:
        reasons.append("whole_grid")
    if mean_area < config.min_area:
        reasons.append("too_small")
    if internal_variation <= 0.0:
        reasons.append("no_internal_variation")

    is_candidate = not reasons

    return CandidateScore(
        track_id=track.track_id,
        age=age,
        length=length,
        is_candidate=is_candidate,
        boundedness=boundedness,
        internal_variation=internal_variation,
        mean_area=mean_area,
        max_area=max_area,
        reasons=reasons,
    )


def score_persistence(
    tracks: list[Track],
    grid_shape: tuple[int, int],
    config: DetectionConfig,
) -> list[CandidateScore]:
    """Compute one :class:`CandidateScore` per track.

    A track is a candidate iff:

    - ``age >= config.min_age``
    - ``max_area / (Nx * Ny) <= config.max_area_fraction`` (not "whole grid")
    - ``mean_area >= config.min_area``
    - ``internal_variation > 0`` (non-trivial internal state)

    Boundedness ``= 1 / (1 + std(area) / (mean(area) + 1e-6))`` -- closer to
    1.0 means more stable area.
    """
    return [_score_one(t, grid_shape, config) for t in tracks]


def filter_observer_candidates(
    tracks: list[Track],
    grid_shape: tuple[int, int],
    config: DetectionConfig,
) -> list[CandidateScore]:
    """Return only the entries from :func:`score_persistence` that are candidates."""
    return [s for s in score_persistence(tracks, grid_shape, config) if s.is_candidate]
