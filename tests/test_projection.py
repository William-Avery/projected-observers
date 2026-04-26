"""Tests for the 4D-to-2D projection operators."""

from __future__ import annotations

import numpy as np

from observer_worlds.worlds.projection import (
    max_projection,
    mean_threshold_projection,
    parity_projection,
    project,
    sum_projection,
)


def test_mean_threshold_zeros() -> None:
    state = np.zeros((4, 4, 3, 3), dtype=np.uint8)
    out = mean_threshold_projection(state, theta=0.5)
    assert out.shape == (4, 4)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.zeros((4, 4), dtype=np.uint8))


def test_mean_threshold_ones() -> None:
    state = np.ones((4, 4, 3, 3), dtype=np.uint8)
    out = mean_threshold_projection(state, theta=0.5)
    assert out.shape == (4, 4)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.ones((4, 4), dtype=np.uint8))


def test_mean_threshold_below_threshold() -> None:
    """A (z, w) fibre that is exactly half-on must project to 0 (strict >)."""
    state = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    # At (x=0, y=0): exactly 2 of 4 (z,w) cells are on -> mean = 0.5.
    state[0, 0, 0, 0] = 1
    state[0, 0, 1, 1] = 1
    # At (x=1, y=1): all 4 on -> mean = 1.0.
    state[1, 1, :, :] = 1

    out = mean_threshold_projection(state, theta=0.5)
    assert out[0, 0] == 0, "exactly half-on must NOT cross strict > threshold"
    assert out[1, 1] == 1
    # The other two columns are all zeros -> 0.
    assert out[0, 1] == 0
    assert out[1, 0] == 0


def test_sum_projection_known() -> None:
    """Hand-built (z, w) sums."""
    state = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    state[0, 0, 0, 0] = 1  # sum at (0,0) = 1
    state[0, 1, :, :] = 1  # sum at (0,1) = 4
    state[1, 0, 0, 1] = 1
    state[1, 0, 1, 0] = 1  # sum at (1,0) = 2
    # (1,1) untouched -> 0.

    out = sum_projection(state)
    assert out.dtype == np.int32
    assert out.shape == (2, 2)
    expected = np.array([[1, 4], [2, 0]], dtype=np.int32)
    assert np.array_equal(out, expected)


def test_max_projection_known() -> None:
    state = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    state[0, 1, 1, 0] = 1
    state[1, 1, 0, 1] = 1

    out = max_projection(state)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2)
    expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_parity_projection_known() -> None:
    state = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    state[0, 0, 0, 0] = 1  # sum 1 -> parity 1
    state[0, 1, :, :] = 1  # sum 4 -> parity 0
    state[1, 0, 0, 1] = 1
    state[1, 0, 1, 0] = 1  # sum 2 -> parity 0
    state[1, 1, 0, 0] = 1
    state[1, 1, 0, 1] = 1
    state[1, 1, 1, 0] = 1  # sum 3 -> parity 1

    out = parity_projection(state)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2)
    expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_project_dispatch() -> None:
    """``project(method='mean_threshold')`` must agree with the direct call."""
    rng = np.random.default_rng(42)
    state = (rng.random((5, 5, 3, 3)) < 0.4).astype(np.uint8)

    via_dispatch = project(state, method="mean_threshold", theta=0.5)
    direct = mean_threshold_projection(state, theta=0.5)
    assert np.array_equal(via_dispatch, direct)

    # Sanity: also dispatch for the others.
    assert np.array_equal(project(state, method="sum"), sum_projection(state))
    assert np.array_equal(
        project(state, method="parity"), parity_projection(state)
    )
    assert np.array_equal(project(state, method="max"), max_projection(state))
