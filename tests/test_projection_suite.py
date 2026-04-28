"""Tests for the projection suite registry (Follow-up Topic 1, Stage 1)."""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.projection import ProjectionSpec, ProjectionSuite, default_suite
from observer_worlds.projection.projection_suite import (
    _multi_channel_projection,
    _random_linear_projection,
    _sum_threshold_projection,
)


# Small 4D state for fast deterministic tests.
def _state(seed: int = 0, shape=(8, 8, 4, 4)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=shape).astype(np.uint8)


# ---------------------------------------------------------------------------
# Default suite contains the six expected projections
# ---------------------------------------------------------------------------


def test_default_suite_has_six_projections():
    s = default_suite()
    assert set(s.names()) == {
        "mean_threshold",
        "sum_threshold",
        "max_projection",
        "parity_projection",
        "random_linear_projection",
        "multi_channel_projection",
    }


def test_default_suite_each_spec_callable():
    s = default_suite()
    state = _state()
    for name in s.names():
        out = s.project(name, state)
        assert isinstance(out, np.ndarray), f"{name} must return ndarray"
        # First two dims always match input.
        assert out.shape[:2] == state.shape[:2], (
            f"{name} output shape {out.shape[:2]} != input {state.shape[:2]}"
        )


# ---------------------------------------------------------------------------
# Threshold-margin metadata flag
# ---------------------------------------------------------------------------


def test_threshold_margin_supported_only_for_threshold_projections():
    s = default_suite()
    assert s.supports_threshold_margin("mean_threshold") is True
    assert s.supports_threshold_margin("sum_threshold") is True
    # The remaining four do not have a natural threshold margin.
    for name in (
        "max_projection",
        "parity_projection",
        "random_linear_projection",
        "multi_channel_projection",
    ):
        assert s.supports_threshold_margin(name) is False, (
            f"{name} should not claim threshold margin support"
        )


def test_output_kind_is_documented_for_each_projection():
    s = default_suite()
    expected = {
        "mean_threshold": "binary",
        "sum_threshold": "binary",
        "max_projection": "binary",
        "parity_projection": "binary",
        "random_linear_projection": "continuous",
        "multi_channel_projection": "multi_channel",
    }
    for name, kind in expected.items():
        assert s.output_kind(name) == kind


# ---------------------------------------------------------------------------
# Per-projection numerical sanity checks
# ---------------------------------------------------------------------------


def test_sum_threshold_basic():
    state = np.zeros((4, 4, 2, 2), dtype=np.uint8)
    state[0, 0, 0, 0] = 1  # one voxel hot
    out = _sum_threshold_projection(state, theta=1)
    assert out.shape == (4, 4)
    assert out[0, 0] == 1
    assert out[1, 1] == 0
    # theta=2: same voxel, sum = 1 < 2 -> 0
    out2 = _sum_threshold_projection(state, theta=2)
    assert out2[0, 0] == 0


def test_random_linear_is_deterministic_per_seed():
    state = _state(seed=7)
    a = _random_linear_projection(state, seed=42)
    b = _random_linear_projection(state, seed=42)
    np.testing.assert_array_equal(a, b)
    c = _random_linear_projection(state, seed=43)
    assert not np.array_equal(a, c)
    assert a.dtype == np.float32


def test_multi_channel_returns_uint8_with_expected_channels():
    state = _state(seed=3)
    out = _multi_channel_projection(state, n_channels=4, seed=0)
    assert out.shape == (state.shape[0], state.shape[1], 4)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Registry semantics
# ---------------------------------------------------------------------------


def test_get_unknown_projection_raises_keyerror():
    s = default_suite()
    with pytest.raises(KeyError):
        s.get("does_not_exist")


def test_register_rejects_duplicate_names():
    s = ProjectionSuite()
    spec = ProjectionSpec(
        name="dummy",
        fn=lambda x, **kw: x[..., 0, 0],
        threshold_margin_supported=False,
        output_kind="binary",
    )
    s.register(spec)
    with pytest.raises(ValueError):
        s.register(spec)


def test_project_passes_overrides_to_underlying_fn():
    s = default_suite()
    # Build a sparse state so theta=1 vs theta=2 actually differ:
    # one (z,w) voxel hot at (0,0); two voxels hot at (1,1).
    state = np.zeros((4, 4, 2, 2), dtype=np.uint8)
    state[0, 0, 0, 0] = 1
    state[1, 1, 0, 0] = 1
    state[1, 1, 1, 1] = 1
    a = s.project("sum_threshold", state, theta=1)
    b = s.project("sum_threshold", state, theta=2)
    # theta=1 -> both (0,0) and (1,1) on; theta=2 -> only (1,1) on.
    assert a[0, 0] == 1 and b[0, 0] == 0
    assert a[1, 1] == 1 and b[1, 1] == 1
    assert not np.array_equal(a, b)
