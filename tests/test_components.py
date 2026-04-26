"""Tests for `observer_worlds.detection.components`."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.detection.components import Component, extract_components
from observer_worlds.utils.config import DetectionConfig


# --------------------------------------------------------------------- helpers


def _disc(grid_shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    """Solid Euclidean disc of given integer radius on a uint8 grid."""
    nx, ny = grid_shape
    cr, cc = center
    rows, cols = np.ogrid[:nx, :ny]
    return ((rows - cr) ** 2 + (cols - cc) ** 2 <= radius**2).astype(np.uint8)


def _square(
    grid_shape: tuple[int, int], top_left: tuple[int, int], size: int
) -> np.ndarray:
    nx, ny = grid_shape
    arr = np.zeros((nx, ny), dtype=np.uint8)
    r0, c0 = top_left
    arr[r0 : r0 + size, c0 : c0 + size] = 1
    return arr


# ----------------------------------------------------------------------- tests


def test_single_disc_extracts_one_component() -> None:
    """One disc -> one component with the right area and (symmetric) centroid."""
    cfg = DetectionConfig()
    frame = _disc((16, 16), center=(8, 8), radius=2)
    comps = extract_components(frame, frame_idx=0, config=cfg)

    assert len(comps) == 1
    comp = comps[0]
    assert isinstance(comp, Component)
    assert comp.frame == 0
    assert comp.component_id == 0
    # Radius-2 Euclidean disc has 13 active cells (centre + 4 axial pairs at
    # distance 1, 4 axial cells at distance 2, plus the 4 diagonal-1 cells).
    assert comp.area == int(frame.sum())
    # The disc is symmetric about (8, 8).
    assert comp.centroid == pytest.approx((8.0, 8.0), abs=1e-9)
    # bbox should be inclusive-exclusive: rows 6..10 active -> (6, 6, 11, 11).
    rmin, cmin, rmax, cmax = comp.bbox
    assert (rmin, cmin) == (6, 6)
    assert (rmax, cmax) == (11, 11)


def test_below_min_area_filtered() -> None:
    """A 2x2 active block under min_area=5 is filtered out."""
    cfg = DetectionConfig(min_area=5)
    frame = _square((16, 16), top_left=(4, 4), size=2)
    comps = extract_components(frame, frame_idx=0, config=cfg)
    assert comps == []


def test_two_separate_components() -> None:
    """Two non-touching blobs produce two components."""
    cfg = DetectionConfig(min_area=1)
    frame = np.zeros((20, 20), dtype=np.uint8)
    frame[2:5, 2:5] = 1  # 3x3 blob
    frame[12:15, 12:15] = 1  # 3x3 blob, far away
    comps = extract_components(frame, frame_idx=3, config=cfg)
    assert len(comps) == 2
    # Each component's frame index is propagated.
    assert {c.frame for c in comps} == {3}
    centroids = sorted(c.centroid for c in comps)
    assert centroids[0] == pytest.approx((3.0, 3.0))
    assert centroids[1] == pytest.approx((13.0, 13.0))


def test_boundary_environment_shells() -> None:
    """Shell relationships hold and env shell width = env_dilation - boundary_dilation."""
    cfg = DetectionConfig(boundary_dilation=1, environment_dilation=4)
    frame = _disc((16, 16), center=(8, 8), radius=2)
    comps = extract_components(frame, frame_idx=0, config=cfg)
    assert len(comps) == 1
    comp = comps[0]

    # Interior is strictly inside the mask.
    assert np.all(comp.interior_mask <= comp.mask)
    assert not np.array_equal(comp.interior_mask, comp.mask)

    # boundary == mask XOR interior, and they partition mask.
    assert np.array_equal(comp.boundary_mask, comp.mask ^ comp.interior_mask)
    assert np.array_equal(comp.boundary_mask | comp.interior_mask, comp.mask)
    assert not np.any(comp.boundary_mask & comp.interior_mask)

    # Environment is disjoint from the mask itself.
    assert not np.any(comp.environment_mask & comp.mask)

    # The environment shell sits at distances strictly greater than
    # ``boundary_dilation`` (we explicitly carved out the dilate-by-bd region)
    # and at most ``environment_dilation`` from the original mask.
    import scipy.ndimage as ndi

    structure = ndi.generate_binary_structure(2, cfg.connectivity)
    inside = ndi.binary_dilation(
        comp.mask, structure=structure, iterations=cfg.boundary_dilation
    )
    outside = ndi.binary_dilation(
        comp.mask, structure=structure, iterations=cfg.environment_dilation
    )
    expected_env = outside & ~inside & ~comp.mask
    assert np.array_equal(comp.environment_mask, expected_env)


def test_connectivity_4_vs_8() -> None:
    """Diagonal pair of cells: 2 components under conn=1, 1 under conn=2."""
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[3, 3] = 1
    frame[4, 4] = 1

    cfg4 = DetectionConfig(connectivity=1, min_area=1)
    comps4 = extract_components(frame, frame_idx=0, config=cfg4)
    assert len(comps4) == 2

    cfg8 = DetectionConfig(connectivity=2, min_area=1)
    comps8 = extract_components(frame, frame_idx=0, config=cfg8)
    assert len(comps8) == 1
    assert comps8[0].area == 2
