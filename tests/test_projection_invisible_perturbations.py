"""Tests for projection-specific hidden-invisible perturbations
(Stage 2B)."""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.projection import (
    default_suite,
    make_projection_invisible_perturbation,
)


def _state(seed: int = 0, shape=(8, 8, 4, 4), density: float = 0.4
           ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(np.uint8)


def _full_mask(shape_2d) -> np.ndarray:
    return np.ones(shape_2d, dtype=bool)


def _project(name: str, X: np.ndarray, **kwargs) -> np.ndarray:
    return default_suite().project(name, X, **kwargs)


# ---------------------------------------------------------------------------
# Count-preserving strategy: the four binary projections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", [
    "mean_threshold", "sum_threshold", "max_projection", "parity_projection",
])
def test_count_preserving_strategy_keeps_projection_unchanged(projection):
    rng = np.random.default_rng(7)
    X = _state(seed=42)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name=projection,
        rng=rng, max_attempts=10, target_flip_fraction=0.25,
    )
    assert report["accepted"] is True
    assert report["preservation_strategy"] == "count_preserving_swap"
    # Verify the projection actually didn't change.
    p_un = _project(projection, X)
    p_pe = _project(projection, perturbed)
    np.testing.assert_array_equal(p_un, p_pe)
    # And the perturbation actually flipped some bits.
    assert report["n_flipped"] >= 2  # at least one swap = two toggles.


def test_max_projection_does_not_flip_last_active_bit_off():
    # Build a state where the candidate fibre has exactly one ON cell.
    X = np.zeros((4, 4, 2, 2), dtype=np.uint8)
    X[0, 0, 0, 0] = 1   # the one ON bit at fibre (0, 0)
    cm = np.zeros(X.shape[:2], dtype=bool); cm[0, 0] = True
    rng = np.random.default_rng(0)
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name="max_projection",
        rng=rng, target_flip_fraction=0.5,
    )
    assert report["accepted"] is True
    p_un = _project("max_projection", X)
    p_pe = _project("max_projection", perturbed)
    # max(fibre) was 1 before; must remain 1 after.
    assert p_un[0, 0] == 1 and p_pe[0, 0] == 1
    # Total ON count in the fibre is preserved.
    assert int(perturbed[0, 0].sum()) == int(X[0, 0].sum()) == 1


def test_max_projection_zero_fiber_is_marked_invalid_or_unchanged():
    # All-zero fibre -> count-preserving swap can't engage; reported invalid.
    X = np.zeros((4, 4, 2, 2), dtype=np.uint8)
    cm = np.zeros(X.shape[:2], dtype=bool); cm[1, 1] = True
    rng = np.random.default_rng(0)
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name="max_projection",
        rng=rng,
    )
    assert report["accepted"] is False
    assert report["invalid_reason"] is not None
    # State is unchanged when invalid.
    np.testing.assert_array_equal(X, perturbed)
    # Crucially: projection is still unchanged.
    np.testing.assert_array_equal(
        _project("max_projection", X),
        _project("max_projection", perturbed),
    )


def test_parity_projection_uses_count_preserving_strategy_and_preserves_parity():
    rng = np.random.default_rng(3)
    X = _state(seed=11, density=0.4)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name="parity_projection",
        rng=rng, target_flip_fraction=0.5,
    )
    assert report["accepted"] is True
    p_un = _project("parity_projection", X)
    p_pe = _project("parity_projection", perturbed)
    np.testing.assert_array_equal(p_un, p_pe)
    # And per-fibre count is preserved (which guarantees parity).
    counts_un = X.sum(axis=(2, 3))
    counts_pe = perturbed.sum(axis=(2, 3))
    np.testing.assert_array_equal(counts_un, counts_pe)


def test_mean_threshold_preserves_thresholded_value_per_column():
    rng = np.random.default_rng(5)
    X = _state(seed=99, density=0.55)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name="mean_threshold",
        rng=rng, target_flip_fraction=0.3,
    )
    assert report["accepted"] is True
    p_un = _project("mean_threshold", X)
    p_pe = _project("mean_threshold", perturbed)
    np.testing.assert_array_equal(p_un, p_pe)
    # initial_projection_delta == 0 within float tolerance.
    assert report["initial_projection_delta"] < 1e-6


# ---------------------------------------------------------------------------
# Verification-based: random_linear and multi_channel
# ---------------------------------------------------------------------------


def test_random_linear_with_no_valid_perturbation_reports_failure():
    """A continuous projection cannot generally be preserved by random
    bit flips; verification should exhaust attempts and report
    failure with the best-delta available."""
    rng = np.random.default_rng(0)
    X = _state(seed=21)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name="random_linear_projection",
        rng=rng, max_attempts=5, target_flip_fraction=0.25,
        verification_tolerance=1e-9,  # impossibly strict
    )
    # With strict tolerance and continuous output, no attempt should
    # produce exactly zero delta.
    assert report["accepted"] is False
    assert report["preservation_strategy"] == "verification"
    assert "verification failed" in (report["invalid_reason"] or "")
    assert report["attempts_used"] >= 1


def test_multi_channel_verification_can_reject_when_channels_differ():
    rng = np.random.default_rng(0)
    X = _state(seed=44)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=cm, projection_name="multi_channel_projection",
        rng=rng, max_attempts=5, target_flip_fraction=0.25,
        verification_tolerance=0.0,  # zero tolerance — strict
    )
    # The reported strategy is verification-based.
    assert report["preservation_strategy"] == "verification"
    # Either accepted (a flip set happened to preserve the channels)
    # or rejected with a clear reason.
    if report["accepted"]:
        p_un = _project("multi_channel_projection", X)
        p_pe = _project("multi_channel_projection", perturbed)
        np.testing.assert_array_equal(p_un, p_pe)
    else:
        assert "verification failed" in (report["invalid_reason"] or "")


# ---------------------------------------------------------------------------
# Empty / pathological inputs
# ---------------------------------------------------------------------------


def test_empty_candidate_mask_returns_invalid_unchanged():
    rng = np.random.default_rng(0)
    X = _state(seed=2)
    empty = np.zeros(X.shape[:2], dtype=bool)
    perturbed, report = make_projection_invisible_perturbation(
        X, candidate_mask=empty, projection_name="mean_threshold", rng=rng,
    )
    assert report["accepted"] is False
    assert report["invalid_reason"] == "empty candidate mask"
    np.testing.assert_array_equal(X, perturbed)


def test_unknown_projection_raises():
    with pytest.raises(ValueError):
        make_projection_invisible_perturbation(
            _state(), candidate_mask=_full_mask((8, 8)),
            projection_name="not_a_thing",
        )


# ---------------------------------------------------------------------------
# Integration: pipeline-level initial_projection_delta = 0
# ---------------------------------------------------------------------------


def test_integration_smoke_initial_projection_delta_is_zero_for_count_based():
    """End-to-end: count-preserving strategy yields exact zero
    initial_projection_delta on every accepted perturbation."""
    rng = np.random.default_rng(0)
    X = _state(seed=7, density=0.4)
    cm = _full_mask(X.shape[:2])
    for projection in ("mean_threshold", "sum_threshold",
                       "max_projection", "parity_projection"):
        perturbed, report = make_projection_invisible_perturbation(
            X, candidate_mask=cm, projection_name=projection,
            rng=rng, target_flip_fraction=0.25,
        )
        if report["accepted"]:
            assert report["initial_projection_delta"] < 1e-6, (
                f"{projection}: delta {report['initial_projection_delta']} "
                f"not within tolerance"
            )
