"""Shared per-track feature extraction.

All metric modules consume the same :class:`TrackFeatures` representation:
for every observed timestep of a :class:`Track`, we produce a small
fixed-dimensional feature vector each for the candidate's interior, boundary,
and environment.  Higher-level metrics (time / memory / selfhood) then
predict one set of features from another.

Tracks may have gaps (a missed frame within ``max_gap`` is tolerated).
``TrackFeatures.frames`` records the actual frame index for each row, so
metrics that need contiguous (t, t+1) pairs can filter via
:meth:`TrackFeatures.contiguous_pairs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from observer_worlds.detection.tracking import Track


# Names of the columns in each composite feature vector.  Kept as module-level
# constants so callers (and tests) can introspect.
INTERNAL_FEATURE_NAMES: tuple[str, ...] = (
    "interior_count",
    "area",
    "interior_density",   # interior_count / max(area, 1)
    "centroid_y",
    "centroid_x",
    "centroid_vy",
    "centroid_vx",
    "bbox_h",
    "bbox_w",
    "compactness",        # 4*pi*area / max(perim^2, 1)
)

SENSORY_FEATURE_NAMES: tuple[str, ...] = (
    "env_count",
    "env_density",        # env_count / env_size
    "env_size",
)

BOUNDARY_FEATURE_NAMES: tuple[str, ...] = (
    "boundary_count",
    "boundary_density",   # boundary_count / boundary_size
    "boundary_size",
)


@dataclass
class TrackFeatures:
    """Per-time feature arrays for a single :class:`Track`.

    All ``*_history``-derived arrays have shape ``(T_obs,)`` where
    ``T_obs == len(track.frames)``.  Composite feature matrices have shape
    ``(T_obs, D_*)``.
    """

    track_id: int
    frames: np.ndarray             # (T_obs,) int — actual frame indices

    # Scalar columns (all length T_obs).
    interior_count: np.ndarray
    boundary_count: np.ndarray
    env_count: np.ndarray
    boundary_size: np.ndarray
    env_size: np.ndarray
    area: np.ndarray
    centroid_y: np.ndarray
    centroid_x: np.ndarray
    centroid_vy: np.ndarray        # diff of centroid_y; 0 at first row
    centroid_vx: np.ndarray
    bbox_h: np.ndarray
    bbox_w: np.ndarray
    perimeter: np.ndarray          # boundary_count is a fine proxy
    compactness: np.ndarray

    # Composite vectors.
    internal_features: np.ndarray  # (T_obs, len(INTERNAL_FEATURE_NAMES))
    sensory_features: np.ndarray   # (T_obs, len(SENSORY_FEATURE_NAMES))
    boundary_features: np.ndarray  # (T_obs, len(BOUNDARY_FEATURE_NAMES))

    # Cell-level data, retained for the boundaries module and causality.
    boundary_masks: list[np.ndarray] = field(default_factory=list)
    interior_masks: list[np.ndarray] = field(default_factory=list)
    env_masks: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers

    @property
    def n_obs(self) -> int:
        return int(self.frames.shape[0])

    def contiguous_pairs(self) -> np.ndarray:
        """Return an int array of indices ``i`` such that ``frames[i+1] == frames[i] + 1``.

        Use this in metrics that predict ``X_{t+1}`` from ``X_t`` to avoid
        spanning gaps in the track.
        """
        if self.n_obs < 2:
            return np.empty(0, dtype=np.int64)
        diffs = np.diff(self.frames)
        return np.flatnonzero(diffs == 1).astype(np.int64)

    def contiguous_triples(self) -> np.ndarray:
        """Indices ``i`` with ``frames[i+1] == frames[i]+1`` and
        ``frames[i+2] == frames[i]+2``."""
        if self.n_obs < 3:
            return np.empty(0, dtype=np.int64)
        d1 = np.diff(self.frames[:-1])
        d2 = np.diff(self.frames[1:])
        return np.flatnonzero((d1 == 1) & (d2 == 1)).astype(np.int64)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_track_features(track: "Track") -> TrackFeatures:
    """Build a :class:`TrackFeatures` from a :class:`Track`.

    All scalar columns are computed from the per-frame ``*_history`` mask
    lists.  Composite vectors are stacked by name in the order documented in
    :data:`INTERNAL_FEATURE_NAMES` etc.
    """
    n = len(track.frames)
    if n == 0:
        return _empty_features(track.track_id)

    frames = np.asarray(track.frames, dtype=np.int64)
    area = np.asarray(track.area_history, dtype=np.float64)
    centroids = np.asarray(track.centroid_history, dtype=np.float64)  # (n, 2)
    centroid_y = centroids[:, 0]
    centroid_x = centroids[:, 1]

    # Cell counts via mask sums.
    interior_count = np.array(
        [int(m.sum()) for m in track.interior_history], dtype=np.float64
    )
    boundary_count = np.array(
        [int(m.sum()) for m in track.boundary_history], dtype=np.float64
    )
    # env_count distinguishes "active cells in the env shell" from "shell
    # size".  The Component populates env_active_count = (env_mask & frame).sum().
    # For backwards compatibility (old tracks lacking env_active_history),
    # fall back to the shell mask sum.
    if getattr(track, "env_active_history", None):
        env_count = np.array(track.env_active_history, dtype=np.float64)
    else:
        env_count = np.array(
            [int(m.sum()) for m in track.env_history], dtype=np.float64
        )
    env_size = np.array(
        [int(m.sum()) for m in track.env_history], dtype=np.float64
    )

    # Boundary cells in our convention are a subset of mask, so every cell
    # in boundary_mask is active by construction; ``boundary_size`` and
    # ``boundary_count`` are intentionally equal.  We keep the field so the
    # composite vector has consistent shape.
    boundary_size = boundary_count.copy()
    # Avoid divide-by-zero downstream.
    bnd_size_safe = np.maximum(boundary_size, 1.0)
    env_size_safe = np.maximum(env_size, 1.0)
    area_safe = np.maximum(area, 1.0)

    # Velocity (forward diff; first row is 0).
    centroid_vy = np.zeros(n, dtype=np.float64)
    centroid_vx = np.zeros(n, dtype=np.float64)
    if n >= 2:
        # Velocity is only meaningful between contiguous frames; we compute
        # naive diffs and zero out non-contiguous transitions.
        dy = np.diff(centroid_y)
        dx = np.diff(centroid_x)
        df = np.diff(frames)
        contig = (df == 1)
        dy = np.where(contig, dy, 0.0)
        dx = np.where(contig, dx, 0.0)
        centroid_vy[1:] = dy
        centroid_vx[1:] = dx

    # BBox dims.
    bbox_h = np.zeros(n, dtype=np.float64)
    bbox_w = np.zeros(n, dtype=np.float64)
    for i, bb in enumerate(track.bbox_history):
        rmin, cmin, rmax, cmax = bb
        bbox_h[i] = float(max(rmax - rmin, 0))
        bbox_w[i] = float(max(cmax - cmin, 0))

    perimeter = boundary_count.copy()  # cheap proxy
    compactness = (4.0 * np.pi * area) / np.maximum(perimeter, 1.0) ** 2
    interior_density = interior_count / area_safe
    boundary_density = boundary_count / bnd_size_safe  # always 1 with current masks
    env_density = env_count / env_size_safe

    internal = np.stack(
        [
            interior_count,
            area,
            interior_density,
            centroid_y,
            centroid_x,
            centroid_vy,
            centroid_vx,
            bbox_h,
            bbox_w,
            compactness,
        ],
        axis=1,
    )
    sensory = np.stack([env_count, env_density, env_size], axis=1)
    boundary = np.stack(
        [boundary_count, boundary_density, boundary_size], axis=1
    )

    return TrackFeatures(
        track_id=track.track_id,
        frames=frames,
        interior_count=interior_count,
        boundary_count=boundary_count,
        env_count=env_count,
        boundary_size=boundary_size,
        env_size=env_size,
        area=area,
        centroid_y=centroid_y,
        centroid_x=centroid_x,
        centroid_vy=centroid_vy,
        centroid_vx=centroid_vx,
        bbox_h=bbox_h,
        bbox_w=bbox_w,
        perimeter=perimeter,
        compactness=compactness,
        internal_features=internal,
        sensory_features=sensory,
        boundary_features=boundary,
        boundary_masks=list(track.boundary_history),
        interior_masks=list(track.interior_history),
        env_masks=list(track.env_history),
        masks=list(track.mask_history),
    )


def _empty_features(track_id: int) -> TrackFeatures:
    z = np.empty(0, dtype=np.float64)
    iz = np.empty(0, dtype=np.int64)
    z2_int = np.empty((0, len(INTERNAL_FEATURE_NAMES)), dtype=np.float64)
    z2_sen = np.empty((0, len(SENSORY_FEATURE_NAMES)), dtype=np.float64)
    z2_bnd = np.empty((0, len(BOUNDARY_FEATURE_NAMES)), dtype=np.float64)
    return TrackFeatures(
        track_id=track_id,
        frames=iz,
        interior_count=z, boundary_count=z, env_count=z,
        boundary_size=z, env_size=z,
        area=z, centroid_y=z, centroid_x=z, centroid_vy=z, centroid_vx=z,
        bbox_h=z, bbox_w=z, perimeter=z, compactness=z,
        internal_features=z2_int, sensory_features=z2_sen, boundary_features=z2_bnd,
    )
