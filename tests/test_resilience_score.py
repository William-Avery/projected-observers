"""Tests for `observer_worlds.metrics.resilience_score`."""

from __future__ import annotations

import numpy as np

from observer_worlds.metrics.resilience_score import (
    ResilienceResult,
    compute_resilience_score,
)
from observer_worlds.worlds.rules import BSRule


_TEST_RULE = BSRule(
    birth=tuple(range(20, 30)), survival=tuple(range(15, 35))
)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def test_empty_mask_invalid() -> None:
    rng = np.random.default_rng(0)
    snap = (rng.random((6, 6, 2, 2)) < 0.3).astype(np.uint8)
    interior = np.zeros((6, 6), dtype=bool)

    result = compute_resilience_score(
        snap,
        rule=_TEST_RULE,
        interior_mask_2d=interior,
        n_steps=5,
        backend="numpy",
        seed=0,
    )

    assert isinstance(result, ResilienceResult)
    assert result.valid is False
    assert result.reason == "empty_mask"


# ---------------------------------------------------------------------------
# Zero perturbation -> max resilience
# ---------------------------------------------------------------------------


def test_zero_perturbation_perfect_resilience() -> None:
    """flip_fraction=0 -> perturbed and baseline rollouts are identical, so
    every component should saturate at its maximum.
    """
    rng = np.random.default_rng(3)
    snap = (rng.random((8, 8, 2, 2)) < 0.4).astype(np.uint8)
    interior = np.zeros((8, 8), dtype=bool)
    interior[3:6, 3:6] = True

    result = compute_resilience_score(
        snap,
        rule=_TEST_RULE,
        interior_mask_2d=interior,
        n_steps=5,
        flip_fraction=0.0,
        backend="numpy",
        seed=0,
    )

    assert result.valid is True
    assert result.reason == "ok"
    # Identical rollouts -> all components at their max.
    assert result.shape_similarity == 1.0
    assert result.area_recovery == 1.0
    assert result.centroid_continuity == 1.0
    # Survival is 1.0 if any active cell is in ROI; with an unperturbed
    # rollout it matches whatever the baseline produced.  We can only
    # *guarantee* that survival(perturbed) == survival(baseline), so the
    # resilience score should equal the survival contribution + 0.75
    # from the other three components.
    expected_lo = 0.75
    assert result.resilience_score >= expected_lo - 1e-9
    # And of course never exceed 1.
    assert result.resilience_score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Full destruction -> lower resilience than no perturbation
# ---------------------------------------------------------------------------


def test_full_destruction_low_resilience() -> None:
    """interior_mask = entire grid, flip_fraction=1.0 -> every interior 4D
    cell flipped.  Resilience should be strictly less than the unperturbed
    case for the same snapshot.
    """
    rng = np.random.default_rng(5)
    snap = (rng.random((6, 6, 2, 2)) < 0.4).astype(np.uint8)
    interior = np.ones((6, 6), dtype=bool)

    res_unperturbed = compute_resilience_score(
        snap,
        rule=_TEST_RULE,
        interior_mask_2d=interior,
        n_steps=5,
        flip_fraction=0.0,
        backend="numpy",
        seed=0,
    )
    res_destroyed = compute_resilience_score(
        snap,
        rule=_TEST_RULE,
        interior_mask_2d=interior,
        n_steps=5,
        flip_fraction=1.0,
        backend="numpy",
        seed=0,
    )

    assert res_unperturbed.valid is True
    assert res_destroyed.valid is True
    assert res_destroyed.resilience_score < res_unperturbed.resilience_score, (
        f"expected destroyed score < unperturbed score; "
        f"got destroyed={res_destroyed.resilience_score}, "
        f"unperturbed={res_unperturbed.resilience_score}"
    )
