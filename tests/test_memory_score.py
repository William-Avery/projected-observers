"""Tests for `observer_worlds.metrics.memory_score`."""

from __future__ import annotations

import numpy as np

from observer_worlds.metrics.features import TrackFeatures
from observer_worlds.metrics.memory_score import (
    MemoryScoreResult,
    compute_memory_score,
)


# --------------------------------------------------------------------- helpers


def make_synthetic_features(track_id, frames, internal, sensory, boundary):
    """Return a TrackFeatures with the given (T, D_*) arrays.
    Other fields filled with zeros/empties as needed.
    """
    n = len(frames)
    z = np.zeros(n, dtype=np.float64)
    return TrackFeatures(
        track_id=track_id,
        frames=np.asarray(frames, dtype=np.int64),
        interior_count=z.copy(), boundary_count=z.copy(), env_count=z.copy(),
        boundary_size=z.copy(), env_size=z.copy(),
        area=z.copy(), centroid_y=z.copy(), centroid_x=z.copy(),
        centroid_vy=z.copy(), centroid_vx=z.copy(),
        bbox_h=z.copy(), bbox_w=z.copy(), perimeter=z.copy(), compactness=z.copy(),
        internal_features=np.asarray(internal, dtype=np.float64),
        sensory_features=np.asarray(sensory, dtype=np.float64),
        boundary_features=np.asarray(boundary, dtype=np.float64),
    )


# ----------------------------------------------------------------------- tests


def test_returns_invalid_when_too_short():
    """Track too short for min_samples -> valid=False, reason='too_short'."""
    n = 4
    rng = np.random.default_rng(0)
    internal = rng.standard_normal((n, 2))
    sensory = rng.standard_normal((n, 2))
    boundary = rng.standard_normal((n, 1))

    feats = make_synthetic_features(
        track_id=11, frames=list(range(n)),
        internal=internal, sensory=sensory, boundary=boundary,
    )

    result = compute_memory_score(feats, horizon=1, min_samples=8)
    assert isinstance(result, MemoryScoreResult)
    assert result.track_id == 11
    assert result.valid is False
    assert result.reason == "too_short"


def test_useful_memory_yields_positive_score():
    """Internal state I_t encodes S_{t-1}, and the future S_{t+1} is a
    function of S_{t-1}.  So (S_t, I_t) >> S_t alone."""
    rng = np.random.default_rng(42)
    n = 100
    # Build a hidden process u_t (iid noise) of length n+1.  Define:
    #   S_t      := a noisy unrelated signal (no info about future)
    #   I_t      := [u_{t-1}, extra_noise_t]  -> internal state that has
    #               memory of the past
    #   future S_{t+1} := f(u_{t-1}) + small noise
    # so the "memory" channel is u_{t-1}, which I_t sees but S_t does not.
    #
    # Concretely we set sensory_features[t+1, 0] = g(u_{t-1}) so the model
    # using S_t alone cannot recover u_{t-1}, but the model using
    # (S_t, I_t) can read u_{t-1} directly from I_t[:, 0].
    u = rng.standard_normal(n + 2)  # padded so we can index t-1 and t+1

    S = np.zeros((n, 2), dtype=np.float64)
    I = np.zeros((n, 2), dtype=np.float64)
    for t in range(n):
        # S_t carries no information about the future memory channel.
        S[t, 0] = rng.standard_normal()
        S[t, 1] = rng.standard_normal()
        # I_t encodes the past hidden state u_{t-1}.
        I[t, 0] = u[t]                       # this is u at index t == "past" for t+1
        I[t, 1] = rng.standard_normal()

    # Overwrite the *first* sensory channel of S_{t+1} (i.e. S[t+1, 0])
    # to be a function of u[t] = I[t, 0] plus small noise.  Since this
    # depends on u[t] which is in I_t but not in S_t, only the augmented
    # model can predict it well.
    target_signal = 1.5 * u[: n] + 0.1 * rng.standard_normal(n)
    # Place it at sensory_features[t+1, 0] for t in [0, n-2].  The
    # _contiguous_horizon_indices function with horizon=1 will then read
    # S_target = S[t+1] for t in valid range.
    # We build a fresh sensory series with the desired structure:
    S_new = S.copy()
    S_new[1:, 0] = target_signal[: n - 1]

    boundary = np.zeros((n, 1), dtype=np.float64)
    feats = make_synthetic_features(
        track_id=1, frames=list(range(n)),
        internal=I, sensory=S_new, boundary=boundary,
    )

    result = compute_memory_score(
        feats, horizon=1, min_samples=8, cv_splits=3, seed=42,
    )

    assert result.valid is True
    assert result.reason == "ok"
    assert result.memory_score > 0.05, (
        f"expected positive memory score; got {result.memory_score}, "
        f"err_S={result.error_s_only}, err_SI={result.error_s_plus_i}"
    )


def test_no_useful_memory_yields_near_zero_score():
    """I_t is iid noise; S_t and S_{t+1} are autocorrelated only with each
    other.  Adding I_t cannot help -- score should be small."""
    rng = np.random.default_rng(42)
    n = 80

    # Sensory autoregressive: S_{t+1} = phi * S_t + eps.
    phi = 0.7
    S = np.zeros((n, 2), dtype=np.float64)
    S[0] = rng.standard_normal(2)
    for t in range(n - 1):
        S[t + 1] = phi * S[t] + 0.5 * rng.standard_normal(2)

    # Internal: pure iid noise, no information about future.
    I = rng.standard_normal((n, 2))
    boundary = np.zeros((n, 1), dtype=np.float64)

    feats = make_synthetic_features(
        track_id=2, frames=list(range(n)),
        internal=I, sensory=S, boundary=boundary,
    )

    result = compute_memory_score(
        feats, horizon=1, min_samples=8, cv_splits=3, seed=42,
    )

    assert result.valid is True
    assert abs(result.memory_score) < 0.5, (
        f"expected near-zero memory score; got {result.memory_score}"
    )
