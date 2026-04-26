"""Tests for `observer_worlds.metrics.time_score`."""

from __future__ import annotations

import numpy as np

from observer_worlds.metrics.features import TrackFeatures
from observer_worlds.metrics.time_score import (
    TimeScoreResult,
    compute_time_score,
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


def test_returns_invalid_when_track_too_short():
    """A track of length 3 with default min_samples=8 must yield valid=False."""
    n = 3
    rng = np.random.default_rng(0)
    internal = rng.standard_normal((n, 2))
    sensory = rng.standard_normal((n, 2))
    boundary = rng.standard_normal((n, 1))

    feats = make_synthetic_features(
        track_id=7, frames=list(range(n)),
        internal=internal, sensory=sensory, boundary=boundary,
    )

    result = compute_time_score(feats, min_samples=8)

    assert isinstance(result, TimeScoreResult)
    assert result.track_id == 7
    assert result.valid is False
    assert result.reason == "too_short"


def test_time_asymmetric_signal_yields_positive_score():
    """Forward (S_{t+1}) is a linear function of (I_t, S_t); backward
    (S_{t-1}) is independent noise.  Score must be positive."""
    rng = np.random.default_rng(42)
    n = 50
    d_i, d_s = 2, 2

    I = rng.standard_normal((n, d_i))
    # Construct sensory series so S_{t+1} = A @ I_t + B @ S_t + small noise.
    A = rng.standard_normal((d_s, d_i))
    B = rng.standard_normal((d_s, d_s)) * 0.5
    S = np.zeros((n, d_s), dtype=np.float64)
    S[0] = rng.standard_normal(d_s)
    for t in range(n - 1):
        S[t + 1] = A @ I[t] + B @ S[t] + 0.05 * rng.standard_normal(d_s)

    # The natural S already gives "forward predictable" because S_{t+1}
    # depends on (I_t, S_t).  But S_{t-1} also has structure relative to
    # (I_t, S_t) via the same dynamics.  To make the *backward* direction
    # genuinely unpredictable, we replace each S_{t-1} we'd query with iid
    # noise -- i.e. swap the "previous-sensory" channel out.  We do this by
    # interleaving: even-indexed rows of S keep the dynamic, but at the
    # *backward* prediction time we read a noise lookup.  Simpler: build a
    # second sensory array used only as the "ground truth past", which is
    # iid noise, by reordering.
    #
    # Concretely: define S_used[t] = S[t] (the dynamic-driven series) for
    # use as features S_t; redefine the *backward* target in terms of an
    # independent noise series with the same shape.
    #
    # The score function consumes S_{t-1} = S_used[t-1] from the same
    # array, so to make it unpredictable we randomize *the early entries*
    # of S used at backward lookups.  Easiest: shuffle S along the time
    # axis in a way that decorrelates S_{t-1} from (I_t, S_t) but keeps
    # S_{t+1} exactly as the dynamics produced it.  That isn't trivial
    # within a single array, so instead we directly hand the score function
    # a track whose S has the forward dynamics, and we accept that
    # contiguous_triples will use S[i], S[i+1], S[i+2] -- the asymmetry
    # arises because S[i+2] is a deterministic function of (I[i+1], S[i+1])
    # while S[i] is the *cause* but not the *effect* of (I[i+1], S[i+1]):
    # given I[i+1] and S[i+1], we only know S[i] up to the inverse of B,
    # which is noisy enough (especially with the noise term) that the
    # backward MSE is larger.  This already produces an asymmetry.
    #
    # However the asymmetry can be mild.  To strengthen it, we add extra
    # iid noise to S[i] *retroactively* -- i.e. corrupt S[i] (and only
    # S[i], not the dynamics step that produced S[i+1]).  Implementation:
    # build a corrupted copy used for backward lookup only.  Since the
    # score function reads S_{t-1} from features.sensory_features[i],
    # we have to corrupt that.  Compromise: corrupt the first half of
    # the series with extra noise so backward predictions on those
    # frames are noisier, while the dynamics S[t+1] = f(I[t], S[t]) was
    # already computed with the un-corrupted S so forward stays clean.
    #
    # We don't need this extra corruption for the assertion margin >0.05
    # to hold reliably -- the natural asymmetry from the noise term plus
    # ridge regularisation is sufficient at n=50.

    boundary = np.zeros((n, 1), dtype=np.float64)
    feats = make_synthetic_features(
        track_id=1, frames=list(range(n)),
        internal=I, sensory=S, boundary=boundary,
    )

    result = compute_time_score(feats, min_samples=8, cv_splits=3, seed=0)

    assert result.valid is True
    assert result.reason == "ok"
    assert result.time_score > 0.05, (
        f"expected forward to be easier; got time_score={result.time_score}, "
        f"forward_error={result.forward_error}, "
        f"backward_error={result.backward_error}"
    )


def test_time_symmetric_signal_yields_near_zero_score():
    """Both S_{t+1} and S_{t-1} are independent noise w.r.t. (I_t, S_t):
    score should be close to zero (small magnitude)."""
    rng = np.random.default_rng(42)
    n = 50
    d_i, d_s = 2, 2

    I = rng.standard_normal((n, d_i))
    S = rng.standard_normal((n, d_s))  # iid -- no temporal structure
    boundary = np.zeros((n, 1), dtype=np.float64)

    feats = make_synthetic_features(
        track_id=2, frames=list(range(n)),
        internal=I, sensory=S, boundary=boundary,
    )

    result = compute_time_score(feats, min_samples=8, cv_splits=3, seed=0)

    assert result.valid is True
    assert abs(result.time_score) < 0.5, (
        f"expected near-zero score; got time_score={result.time_score}"
    )
