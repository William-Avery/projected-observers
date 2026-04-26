"""Greedy frame-to-frame component tracker.

The tracker links per-frame :class:`Component` instances into
:class:`Track`-objects.  Matching is greedy by IoU first, then a centroid
distance fallback.  Missed frames within ``config.max_gap`` are tolerated; the
``frames`` history can have gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from observer_worlds.detection.components import Component
from observer_worlds.utils.config import DetectionConfig


@dataclass
class Track:
    """A tracked structure across multiple frames.

    History lists are parallel: ``history[i]`` corresponds to frame
    ``frames[i]``.  The tracker only appends to a track on frames where it was
    matched; if a track misses ``k`` frames and then matches again, ``frames``
    will have a gap.
    """

    track_id: int
    birth_frame: int
    last_frame: int
    frames: list[int] = field(default_factory=list)
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    area_history: list[int] = field(default_factory=list)
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    mask_history: list[np.ndarray] = field(default_factory=list)
    interior_history: list[np.ndarray] = field(default_factory=list)
    boundary_history: list[np.ndarray] = field(default_factory=list)
    env_history: list[np.ndarray] = field(default_factory=list)
    env_active_history: list[int] = field(default_factory=list)

    @property
    def age(self) -> int:
        """``last_frame - birth_frame + 1`` -- total span (incl. missed frames)."""
        return self.last_frame - self.birth_frame + 1

    @property
    def length(self) -> int:
        """Number of frames the track was actually observed (``len(frames)``)."""
        return len(self.frames)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two full-grid bool masks."""
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class GreedyTracker:
    """Greedy frame-to-frame component tracker.

    For each new frame:
      1. For each currently-live track (last seen within ``max_gap`` frames),
         compute IoU between its most-recent mask and each new component mask.
      2. Assign component to track if ``iou >= iou_threshold`` (greedy by IoU,
         highest first).  Components / tracks may be left unassigned.
      3. For unassigned tracks, also match by centroid distance ``<=
         centroid_distance_threshold`` (still constrained by ``max_gap``).
      4. Components without any match start new tracks.
      5. Tracks unmatched for ``> max_gap`` frames are no longer considered
         live (but remain in :meth:`finalize`'s output).
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 0

    # ---------------------------------------------------------------- helpers

    def _live_track_ids(self, frame_idx: int) -> list[int]:
        """Tracks whose ``last_frame >= frame_idx - max_gap``."""
        cutoff = frame_idx - self.config.max_gap
        return [
            tid
            for tid, tr in self._tracks.items()
            if tr.last_frame >= cutoff
        ]

    def _start_track(self, comp: Component, frame_idx: int) -> None:
        tid = self._next_track_id
        self._next_track_id += 1
        track = Track(
            track_id=tid,
            birth_frame=frame_idx,
            last_frame=frame_idx,
        )
        self._append_to_track(track, comp, frame_idx)
        self._tracks[tid] = track

    def _append_to_track(
        self, track: Track, comp: Component, frame_idx: int
    ) -> None:
        track.last_frame = frame_idx
        track.frames.append(frame_idx)
        track.centroid_history.append(comp.centroid)
        track.area_history.append(comp.area)
        track.bbox_history.append(comp.bbox)
        track.mask_history.append(comp.mask)
        track.interior_history.append(comp.interior_mask)
        track.boundary_history.append(comp.boundary_mask)
        track.env_history.append(comp.environment_mask)
        track.env_active_history.append(int(getattr(comp, "env_active_count", 0)))

    # ----------------------------------------------------------------- update

    def update(self, frame_idx: int, components: list[Component]) -> None:
        """Process one frame.  Updates internal track dict in place."""
        live_ids = self._live_track_ids(frame_idx)

        if not components:
            return

        if not live_ids:
            for comp in components:
                self._start_track(comp, frame_idx)
            return

        n_tracks = len(live_ids)
        n_comps = len(components)

        # IoU matrix.
        iou_mat = np.zeros((n_tracks, n_comps), dtype=np.float64)
        for i, tid in enumerate(live_ids):
            last_mask = self._tracks[tid].mask_history[-1]
            for j, comp in enumerate(components):
                iou_mat[i, j] = _iou(last_mask, comp.mask)

        track_assigned = [False] * n_tracks
        comp_assigned = [False] * n_comps

        # Greedy IoU pass.
        # Sort all (i, j) pairs by descending IoU.
        idx_pairs = [(i, j) for i in range(n_tracks) for j in range(n_comps)]
        idx_pairs.sort(key=lambda ij: -iou_mat[ij[0], ij[1]])
        thresh = self.config.iou_threshold
        for i, j in idx_pairs:
            if track_assigned[i] or comp_assigned[j]:
                continue
            if iou_mat[i, j] < thresh:
                break  # rest are even smaller
            tid = live_ids[i]
            self._append_to_track(self._tracks[tid], components[j], frame_idx)
            track_assigned[i] = True
            comp_assigned[j] = True

        # Centroid-distance fallback for still-unassigned (track, component).
        unassigned_tracks = [i for i, a in enumerate(track_assigned) if not a]
        unassigned_comps = [j for j, a in enumerate(comp_assigned) if not a]
        if unassigned_tracks and unassigned_comps:
            cd_thresh = self.config.centroid_distance_threshold
            dist_pairs: list[tuple[float, int, int]] = []
            for i in unassigned_tracks:
                tid = live_ids[i]
                last_centroid = self._tracks[tid].centroid_history[-1]
                for j in unassigned_comps:
                    d = _centroid_distance(last_centroid, components[j].centroid)
                    if d <= cd_thresh:
                        dist_pairs.append((d, i, j))
            dist_pairs.sort(key=lambda t: t[0])
            for _, i, j in dist_pairs:
                if track_assigned[i] or comp_assigned[j]:
                    continue
                tid = live_ids[i]
                self._append_to_track(self._tracks[tid], components[j], frame_idx)
                track_assigned[i] = True
                comp_assigned[j] = True

        # Unmatched components start new tracks.
        for j, a in enumerate(comp_assigned):
            if not a:
                self._start_track(components[j], frame_idx)

    # --------------------------------------------------------------- finalize

    def finalize(self) -> list[Track]:
        """Return all tracks (active + retired), ordered by ``track_id``."""
        return [self._tracks[i] for i in sorted(self._tracks)]
