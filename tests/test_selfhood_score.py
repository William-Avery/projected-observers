"""Tests for the Markov-blanket-style selfhood score."""

from __future__ import annotations

import numpy as np

from observer_worlds.metrics.features import TrackFeatures
from observer_worlds.metrics.selfhood_score import (
    SelfhoodScoreResult,
    compute_selfhood_score,
)


def make_synthetic_features(track_id, frames, internal, sensory, boundary):
    """Build a TrackFeatures from explicit composite arrays."""
    n = len(frames)
    z = np.zeros(n, dtype=np.float64)
    return TrackFeatures(
        track_id=track_id,
        frames=np.asarray(frames, dtype=np.int64),
        interior_count=z.copy(),
        boundary_count=z.copy(),
        env_count=z.copy(),
        boundary_size=z.copy(),
        env_size=z.copy(),
        area=z.copy() + 10.0,  # nonzero so boundedness is finite
        centroid_y=z.copy(),
        centroid_x=z.copy(),
        centroid_vy=z.copy(),
        centroid_vx=z.copy(),
        bbox_h=z.copy(),
        bbox_w=z.copy(),
        perimeter=z.copy(),
        compactness=z.copy(),
        internal_features=np.asarray(internal, dtype=np.float64),
        sensory_features=np.asarray(sensory, dtype=np.float64),
        boundary_features=np.asarray(boundary, dtype=np.float64),
    )


def test_too_short_returns_invalid():
    rng = np.random.default_rng(42)
    n = 3
    feat = make_synthetic_features(
        track_id=0,
        frames=list(range(n)),
        internal=rng.standard_normal((n, 2)),
        sensory=rng.standard_normal((n, 2)),
        boundary=rng.standard_normal((n, 2)),
    )
    result = compute_selfhood_score(feat, min_samples=8)
    assert isinstance(result, SelfhoodScoreResult)
    assert result.valid is False
    assert result.reason == "too_short"
    assert np.isnan(result.selfhood_score)


def test_perfect_boundary_mediation():
    """I = f(B) deterministically; E independent of everything.

    Boundary should explain I almost perfectly, env should add nothing on
    top, so selfhood_score should be close to 1.0.
    """
    rng = np.random.default_rng(42)
    n = 60
    B = rng.standard_normal((n, 2))
    # I is a fixed linear function of B.
    W = np.array([[1.0, -0.5], [0.7, 1.3]])
    I = B @ W
    # E is independent of both.
    E = rng.standard_normal((n, 2))

    feat = make_synthetic_features(
        track_id=1,
        frames=list(range(n)),
        internal=I,
        sensory=E,
        boundary=B,
    )
    result = compute_selfhood_score(feat, min_samples=8)
    assert result.valid is True
    assert result.boundary_predictability > 0.8
    assert result.extra_env_given_boundary < 0.2
    assert result.selfhood_score > 0.6


def test_no_boundary_mediation():
    """All three (I, B, E) independent.

    Boundary doesn't predict I; env doesn't either.  selfhood_score is
    bounded by boundary_predictability which should be near zero.
    """
    rng = np.random.default_rng(42)
    n = 60
    feat = make_synthetic_features(
        track_id=2,
        frames=list(range(n)),
        internal=rng.standard_normal((n, 2)),
        sensory=rng.standard_normal((n, 2)),
        boundary=rng.standard_normal((n, 2)),
    )
    result = compute_selfhood_score(feat, min_samples=8)
    assert result.valid is True
    # Not predictive -- should clip near 0.
    assert result.boundary_predictability < 0.3
    assert result.selfhood_score < 0.3


def test_environment_helps_beyond_boundary():
    """I depends mostly on E; B is weakly correlated.

    extra_env_given_boundary should be > 0, dragging selfhood_score down
    below the boundary_predictability.
    """
    rng = np.random.default_rng(42)
    n = 80
    E = rng.standard_normal((n, 2))
    B = 0.1 * rng.standard_normal((n, 2))
    # I depends strongly on E, with a tiny B coupling.
    W_E = np.array([[1.0, -0.3], [0.5, 0.9]])
    W_B = np.array([[0.05, 0.0], [0.0, 0.05]])
    I = E @ W_E + B @ W_B + 0.01 * rng.standard_normal((n, 2))

    feat = make_synthetic_features(
        track_id=3,
        frames=list(range(n)),
        internal=I,
        sensory=E,
        boundary=B,
    )
    result = compute_selfhood_score(feat, min_samples=8)
    assert result.valid is True
    # E provides far more information about I than B alone does.
    assert result.direct_env_predictability > result.boundary_predictability
    assert result.extra_env_given_boundary > 0.0
    # Selfhood is dragged below boundary_predictability.
    assert result.selfhood_score <= result.boundary_predictability


def test_persistence_stable_when_internal_constant():
    n = 50
    # Tiny noise on a constant vector -- consecutive cosines should be ~1.
    base = np.array([1.0, 2.0, 3.0])
    I = np.tile(base, (n, 1)) + 1e-6 * np.random.default_rng(0).standard_normal((n, 3))
    rng = np.random.default_rng(42)
    feat = make_synthetic_features(
        track_id=4,
        frames=list(range(n)),
        internal=I,
        sensory=rng.standard_normal((n, 2)),
        boundary=rng.standard_normal((n, 2)),
    )
    result = compute_selfhood_score(feat, min_samples=8)
    assert result.persistence > 0.99


def test_persistence_unstable_when_internal_random():
    rng = np.random.default_rng(42)
    n = 60
    # iid zero-mean random vectors -- consecutive cosines distribute around
    # 0, mean is well below 1.
    I = rng.standard_normal((n, 5))
    feat = make_synthetic_features(
        track_id=5,
        frames=list(range(n)),
        internal=I,
        sensory=rng.standard_normal((n, 2)),
        boundary=rng.standard_normal((n, 2)),
    )
    result = compute_selfhood_score(feat, min_samples=8)
    assert result.persistence < 0.6
