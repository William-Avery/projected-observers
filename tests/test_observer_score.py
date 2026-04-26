"""Tests for the combined observer-likeness score."""

from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.metrics.observer_score import (
    DEFAULT_WEIGHTS,
    ObserverScore,
    compute_observer_scores,
)


def _raw(track_id, time=None, memory=None, selfhood=None, causality=None, resilience=None):
    return {
        "track_id": track_id,
        "time": time,
        "memory": memory,
        "selfhood": selfhood,
        "causality": causality,
        "resilience": resilience,
    }


def test_empty_input_returns_empty():
    assert compute_observer_scores([]) == []


def test_single_track_single_component_returns_zero():
    """One track, one component, no population to normalize against -> 0."""
    out = compute_observer_scores([_raw(0, time=0.5)])
    assert len(out) == 1
    assert out[0].time_raw == 0.5
    assert out[0].combined == 0.0  # no variance to normalize against
    assert out[0].n_components_used == 1


def test_two_tracks_higher_raw_yields_higher_combined():
    raw = [
        _raw(0, time=0.1, memory=0.1, selfhood=0.1, causality=0.1, resilience=0.1),
        _raw(1, time=0.9, memory=0.9, selfhood=0.9, causality=0.9, resilience=0.9),
    ]
    out = compute_observer_scores(raw)
    by_id = {o.track_id: o for o in out}
    assert by_id[1].combined > by_id[0].combined
    # Symmetric z-scoring: track 1 should be roughly the negation of track 0.
    assert by_id[0].combined == pytest.approx(-by_id[1].combined, abs=1e-9)


def test_missing_components_redistribute_weight():
    """Track with only 'time' valid should still get a finite combined score."""
    raw = [
        _raw(0, time=0.1),
        _raw(1, time=0.9),
        _raw(2, time=0.5),
    ]
    out = compute_observer_scores(raw)
    by_id = {o.track_id: o for o in out}
    # Each track used only the 'time' component.
    for o in out:
        assert o.n_components_used == 1
        assert o.weights_used["time"] == DEFAULT_WEIGHTS["time"]
        assert o.weights_used["memory"] == 0.0
    assert by_id[1].combined > by_id[2].combined > by_id[0].combined


def test_zero_variance_component_contributes_zero():
    """If all tracks have the same raw value for a component, that component's
    z-score is 0 across the population."""
    raw = [
        _raw(0, time=0.5, memory=0.1),
        _raw(1, time=0.5, memory=0.9),
    ]
    out = compute_observer_scores(raw)
    # 'time' should have z=0 for both (no variance); only 'memory' contributes.
    by_id = {o.track_id: o for o in out}
    assert by_id[0].time_normalized == 0.0
    assert by_id[1].time_normalized == 0.0
    # Combined should differ purely from memory.
    assert by_id[1].combined > by_id[0].combined


def test_zscore_clipped_at_three_sigma():
    """A wild outlier should be clipped to +/- 3 in the normalized score."""
    raw = [
        _raw(0, time=v) for v in [0.0] * 10 + [1000.0]
    ]
    out = compute_observer_scores(raw)
    # The outlier track should be at z=+3 (not >>3).
    outlier = max(out, key=lambda o: o.time_raw or 0.0)
    assert outlier.time_normalized == pytest.approx(3.0, abs=1e-9)


def test_custom_weights_change_ranking():
    """Weighting one component much higher should let it dominate the combined score."""
    raw = [
        _raw(0, time=0.9, memory=0.1),
        _raw(1, time=0.1, memory=0.9),
    ]
    # Default: equal weights -> tied.
    out_default = compute_observer_scores(raw)
    by_id_d = {o.track_id: o for o in out_default}
    assert abs(by_id_d[0].combined + by_id_d[1].combined) < 1e-9

    # Heavy memory weight -> track 1 (high memory) wins.
    out_mem = compute_observer_scores(raw, weights={"memory": 10.0})
    by_id_m = {o.track_id: o for o in out_mem}
    assert by_id_m[1].combined > by_id_m[0].combined
