"""Tests for projection-changing (visible) perturbations (Stage 5B)."""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.projection import (
    default_suite, make_projection_visible_perturbation,
)


def _state(seed: int = 0, shape=(8, 8, 4, 4), density: float = 0.4
           ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(np.uint8)


def _full_mask(shape_2d) -> np.ndarray:
    return np.ones(shape_2d, dtype=bool)


def _project(name: str, X: np.ndarray) -> np.ndarray:
    return default_suite().project(name, X)


# ---------------------------------------------------------------------------
# Visible perturbation actually changes the projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", [
    "mean_threshold", "sum_threshold", "max_projection",
    "parity_projection", "random_linear_projection",
    "multi_channel_projection",
])
def test_visible_perturbation_changes_projection(projection):
    rng = np.random.default_rng(7)
    X = _state(seed=42)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_visible_perturbation(
        X, candidate_mask=cm, projection_name=projection,
        rng=rng, target_visible_fraction=0.2,
        min_projection_delta=1e-3, max_attempts=5,
    )
    if report["valid"]:
        # Reported visible delta meets the requested minimum.
        assert report["visible_projection_delta"] >= 1e-3
        # And the actual projection differs.
        p_un = _project(projection, X)
        p_pe = _project(projection, perturbed)
        if p_un.dtype.kind in "iu":
            assert not np.array_equal(p_un, p_pe), (
                f"{projection}: projection unchanged after visible "
                f"perturbation"
            )
        else:
            assert float(np.abs(p_un - p_pe).mean()) > 0.0


def test_visible_perturbation_empty_mask_invalid():
    X = _state()
    empty = np.zeros(X.shape[:2], dtype=bool)
    rng = np.random.default_rng(0)
    perturbed, report = make_projection_visible_perturbation(
        X, candidate_mask=empty, projection_name="mean_threshold", rng=rng,
    )
    assert report["valid"] is False
    assert report["invalid_reason"] == "empty candidate mask"
    np.testing.assert_array_equal(X, perturbed)


def test_visible_perturbation_unknown_projection_raises():
    with pytest.raises(ValueError):
        make_projection_visible_perturbation(
            _state(), candidate_mask=_full_mask((8, 8)),
            projection_name="not_a_thing",
        )


def test_visible_perturbation_reaches_min_delta_via_added_rounds():
    """When the initial batch underperforms, the algorithm adds more
    fibres until min_projection_delta is met."""
    # Choose a setup where flipping 10% of cells initially doesn't
    # quite hit a strict delta, but adding more rounds reaches it.
    rng = np.random.default_rng(3)
    X = _state(seed=99, density=0.5)
    cm = _full_mask(X.shape[:2])
    perturbed, report = make_projection_visible_perturbation(
        X, candidate_mask=cm, projection_name="mean_threshold",
        rng=rng, target_visible_fraction=0.05,
        min_projection_delta=0.05, max_attempts=20,
    )
    if report["valid"]:
        assert report["visible_projection_delta"] >= 0.05
        assert report["attempts_used"] >= 1


def test_visible_perturbation_records_strategy_and_counts():
    rng = np.random.default_rng(0)
    X = _state(seed=21)
    cm = _full_mask(X.shape[:2])
    _, report = make_projection_visible_perturbation(
        X, candidate_mask=cm, projection_name="mean_threshold", rng=rng,
    )
    assert report["strategy"] == "fibre_xor_visible"
    assert "n_flipped" in report
    assert "n_fibres_flipped" in report
    assert "min_projection_delta_requested" in report
