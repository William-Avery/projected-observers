"""Tests for `observer_worlds.detection.tracking`."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.detection.components import extract_components
from observer_worlds.detection.tracking import GreedyTracker, Track
from observer_worlds.utils.config import DetectionConfig


# --------------------------------------------------------------------- helpers


def _disc_frame(
    grid_shape: tuple[int, int], center: tuple[int, int], radius: int = 2
) -> np.ndarray:
    nx, ny = grid_shape
    cr, cc = center
    rows, cols = np.ogrid[:nx, :ny]
    return ((rows - cr) ** 2 + (cols - cc) ** 2 <= radius**2).astype(np.uint8)


def _empty_frame(grid_shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(grid_shape, dtype=np.uint8)


def _run(tracker: GreedyTracker, frames: list[np.ndarray], cfg: DetectionConfig) -> list[Track]:
    for i, fr in enumerate(frames):
        comps = extract_components(fr, frame_idx=i, config=cfg)
        tracker.update(i, comps)
    return tracker.finalize()


# ----------------------------------------------------------------------- tests


def test_single_object_two_frames() -> None:
    cfg = DetectionConfig()
    frames = [
        _disc_frame((16, 16), center=(8, 8), radius=2),
        _disc_frame((16, 16), center=(8, 9), radius=2),
    ]
    tracks = _run(GreedyTracker(cfg), frames, cfg)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.length == 2
    assert tr.age == 2
    assert tr.frames == [0, 1]
    assert tr.centroid_history[0] == pytest.approx((8.0, 8.0))
    assert tr.centroid_history[1] == pytest.approx((8.0, 9.0))


def test_object_disappears_then_reappears_within_max_gap() -> None:
    cfg = DetectionConfig(max_gap=2)
    frames = [
        _disc_frame((16, 16), center=(8, 8), radius=2),
        _empty_frame((16, 16)),
        _disc_frame((16, 16), center=(8, 8), radius=2),
    ]
    tracks = _run(GreedyTracker(cfg), frames, cfg)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.frames == [0, 2]
    assert tr.length == 2
    assert tr.age == 3
    assert tr.birth_frame == 0
    assert tr.last_frame == 2


def test_object_disappears_too_long_starts_new_track() -> None:
    cfg = DetectionConfig(max_gap=2)
    grid = (16, 16)
    frames = [
        _disc_frame(grid, center=(8, 8), radius=2),  # 0
        _empty_frame(grid),                          # 1
        _empty_frame(grid),                          # 2
        _empty_frame(grid),                          # 3
        _empty_frame(grid),                          # 4
        _empty_frame(grid),                          # 5
        _disc_frame(grid, center=(8, 8), radius=2),  # 6
    ]
    tracks = _run(GreedyTracker(cfg), frames, cfg)
    assert len(tracks) == 2
    assert tracks[0].birth_frame == 0
    assert tracks[0].last_frame == 0
    assert tracks[1].birth_frame == 6
    assert tracks[1].last_frame == 6


def test_two_independent_objects() -> None:
    cfg = DetectionConfig()
    grid = (24, 24)

    def two(c1: tuple[int, int], c2: tuple[int, int]) -> np.ndarray:
        f1 = _disc_frame(grid, c1, radius=2)
        f2 = _disc_frame(grid, c2, radius=2)
        return np.maximum(f1, f2)

    frames = [
        two((6, 6), (18, 18)),
        two((6, 7), (18, 19)),
    ]
    tracks = _run(GreedyTracker(cfg), frames, cfg)
    assert len(tracks) == 2
    for tr in tracks:
        assert tr.length == 2
        assert tr.age == 2

    # Verify no cross-assignment: each track's centroids stay near its
    # starting region.
    for tr in tracks:
        r0, c0 = tr.centroid_history[0]
        r1, c1 = tr.centroid_history[1]
        # Movement between frames is at most ~1 cell.
        assert abs(r1 - r0) < 2.0
        assert abs(c1 - c0) < 2.0


def test_centroid_fallback_match() -> None:
    """Shift a small disc enough that IoU < threshold but centroid distance is small."""
    # Tiny radius-1 disc (5 cells, "+"-shape).  Shift by 2 cells: the new disc
    # shares only 1 cell with the old one -> IoU = 1/9 ~= 0.11 < 0.3.  But the
    # centroids are 2 cells apart, well within the default centroid threshold
    # of 5.0.
    cfg = DetectionConfig(min_area=1, iou_threshold=0.3, centroid_distance_threshold=5.0)
    grid = (16, 16)
    f0 = _disc_frame(grid, center=(8, 8), radius=1)
    f1 = _disc_frame(grid, center=(8, 10), radius=1)

    # Sanity: confirm the IoU is indeed below threshold.
    inter = int(np.logical_and(f0.astype(bool), f1.astype(bool)).sum())
    union = int(np.logical_or(f0.astype(bool), f1.astype(bool)).sum())
    assert inter / union < cfg.iou_threshold

    tracks = _run(GreedyTracker(cfg), [f0, f1], cfg)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.length == 2
    assert tr.frames == [0, 1]
    assert tr.centroid_history[0] == pytest.approx((8.0, 8.0))
    assert tr.centroid_history[1] == pytest.approx((8.0, 10.0))
