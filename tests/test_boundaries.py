"""Tests for the sensory-vs-active boundary classifier."""

from __future__ import annotations

import numpy as np

from observer_worlds.detection.boundaries import (
    BoundaryClassification,
    classify_boundary,
)
from observer_worlds.metrics.features import TrackFeatures


def _make_features(
    track_id: int,
    n_frames: int,
    boundary_masks: list[np.ndarray],
    env_count: np.ndarray,
) -> TrackFeatures:
    """Construct a TrackFeatures with the bare minimum the classifier needs.

    We only populate ``frames``, ``env_count``, and ``boundary_masks`` --
    everything else gets zero-filled stubs of the correct shape.
    """
    n = n_frames
    z = np.zeros(n, dtype=np.float64)
    return TrackFeatures(
        track_id=track_id,
        frames=np.arange(n, dtype=np.int64),
        interior_count=z.copy(),
        boundary_count=z.copy(),
        env_count=np.asarray(env_count, dtype=np.float64),
        boundary_size=z.copy(),
        env_size=z.copy(),
        area=z.copy() + 10.0,
        centroid_y=z.copy(),
        centroid_x=z.copy(),
        centroid_vy=z.copy(),
        centroid_vx=z.copy(),
        bbox_h=z.copy(),
        bbox_w=z.copy(),
        perimeter=z.copy(),
        compactness=z.copy(),
        internal_features=np.zeros((n, 1), dtype=np.float64),
        sensory_features=np.zeros((n, 1), dtype=np.float64),
        boundary_features=np.zeros((n, 1), dtype=np.float64),
        boundary_masks=boundary_masks,
        interior_masks=[],
        env_masks=[],
        masks=[],
    )


def test_too_short_invalid():
    n = 3
    masks = [np.zeros((4, 4), dtype=bool) for _ in range(n)]
    masks[0][1, 1] = True
    masks[1][1, 1] = True
    masks[2][1, 1] = True
    feat = _make_features(0, n, masks, env_count=np.array([1.0, 2.0, 3.0]))
    result = classify_boundary(feat, min_samples=8)
    assert isinstance(result, BoundaryClassification)
    assert result.valid is False
    assert result.reason == "too_short"


def test_no_boundary_cells_invalid():
    n = 30
    masks = [np.zeros((4, 4), dtype=bool) for _ in range(n)]
    env_count = np.arange(n, dtype=np.float64)
    feat = _make_features(1, n, masks, env_count=env_count)
    result = classify_boundary(feat, min_samples=8)
    assert result.valid is False
    assert result.reason == "no_boundary_cells"


def test_pure_sensory_cell():
    """Designate one cell whose state at t equals (env_count[t-1] > median).

    That cell is purely sensory: its state is *driven* by the previous
    env_count.  Other cells flicker randomly.  The classifier should
    classify a majority of cells as sensory.
    """
    rng = np.random.default_rng(42)
    n = 30
    H, W = 5, 5
    env_count = rng.standard_normal(n).astype(np.float64)
    median = float(np.median(env_count))

    # The "sensory" cell is at (1, 1).
    sensory_states = np.zeros(n, dtype=bool)
    # State at t depends on env at t-1; t=0 has no t-1, leave it False.
    for t in range(1, n):
        sensory_states[t] = env_count[t - 1] > median

    # Build boundary masks.  Each frame: include (1,1) per the sensory
    # rule, plus a couple of random "noise" cells with random independent
    # state to make the union non-trivial.  Noise cells are uncorrelated
    # with both env and sensory rule, so they should split roughly evenly.
    masks: list[np.ndarray] = []
    for t in range(n):
        m = np.zeros((H, W), dtype=bool)
        if sensory_states[t]:
            m[1, 1] = True
        # A small handful of noise cells.
        m[3, 3] = bool(rng.integers(0, 2))
        m[3, 4] = bool(rng.integers(0, 2))
        m[4, 4] = bool(rng.integers(0, 2))
        masks.append(m)

    feat = _make_features(2, n, masks, env_count=env_count)
    result = classify_boundary(feat, min_samples=8)
    assert result.valid is True
    assert result.n_boundary_cells_total >= 1
    # The sensory cell dominates; majority of classifiable cells should be
    # sensory-like.
    assert result.sensory_fraction > 0.55


def test_pure_active_cell():
    """Cells drive env_count one step ahead.

    Several boundary cells share an active-like state pattern that gets
    written into ``env_count[t+1]``.  The classifier should label them as
    active-like, dominating the cell count.  A small number of independent
    noise cells split however they will and remain in the minority.
    """
    rng = np.random.default_rng(123)
    n = 30
    H, W = 6, 6

    # A cohort of "active driver" cells, all following the same hidden
    # pattern -- so they all classify as active and dominate the count.
    n_active_cells = 5
    driver_pattern = rng.integers(0, 2, size=n).astype(bool)

    # env_count[t] reflects driver_pattern[t-1] (with small noise).
    env_count = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        env_count[t] = 5.0 * float(driver_pattern[t - 1]) + 0.05 * rng.standard_normal()

    # Independent noise cells -- a couple, so they can't outvote the driver
    # cohort even in a worst-case split.
    n_noise_cells = 2
    noise_states = rng.integers(0, 2, size=(n, n_noise_cells)).astype(bool)

    active_positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)]
    noise_positions = [(4, 4), (4, 5)]
    assert len(active_positions) == n_active_cells
    assert len(noise_positions) == n_noise_cells

    masks: list[np.ndarray] = []
    for t in range(n):
        m = np.zeros((H, W), dtype=bool)
        if driver_pattern[t]:
            for r, c in active_positions:
                m[r, c] = True
        for k, (r, c) in enumerate(noise_positions):
            if noise_states[t, k]:
                m[r, c] = True
        masks.append(m)

    feat = _make_features(3, n, masks, env_count=env_count)
    result = classify_boundary(feat, min_samples=8)
    assert result.valid is True
    assert result.n_boundary_cells_total >= n_active_cells
    assert result.active_fraction > 0.55
