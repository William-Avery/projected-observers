"""Tests for `observer_worlds.metrics.causality_score`."""

from __future__ import annotations

import numpy as np

from observer_worlds.metrics.causality_score import (
    CausalityResult,
    apply_flip_intervention,
    apply_hidden_shuffle_intervention,
    compute_causality_score,
    rollout,
)
from observer_worlds.worlds.rules import BSRule


# A non-trivial 4D rule: covers a chunk of the count distribution so we get
# *some* propagation under random initialization in tiny grids.
_TEST_RULE = BSRule(
    birth=tuple(range(20, 30)), survival=tuple(range(15, 35))
)


# ---------------------------------------------------------------------------
# Validity / API contract
# ---------------------------------------------------------------------------


def test_empty_mask_invalid() -> None:
    """All-False interior mask -> result.valid is False, reason='empty_mask'."""
    rng = np.random.default_rng(0)
    snap = (rng.random((4, 4, 2, 2)) < 0.3).astype(np.uint8)
    interior = np.zeros((4, 4), dtype=bool)
    boundary = np.zeros((4, 4), dtype=bool)
    boundary[0, 0] = True
    env = np.zeros((4, 4), dtype=bool)
    env[1, 1] = True

    result = compute_causality_score(
        snap,
        rule=_TEST_RULE,
        interior_mask_2d=interior,
        boundary_mask_2d=boundary,
        env_mask_2d=env,
        n_steps=2,
        backend="numpy",
        seed=0,
    )

    assert isinstance(result, CausalityResult)
    assert result.valid is False
    assert result.reason == "empty_mask"


def test_unperturbed_baseline_matches_itself() -> None:
    """rollout() is deterministic: calling twice with the same input gives
    bitwise-identical projected frames."""
    rng = np.random.default_rng(1)
    snap = (rng.random((6, 6, 2, 2)) < 0.3).astype(np.uint8)
    a = rollout(snap, _TEST_RULE, n_steps=5, backend="numpy")
    b = rollout(snap, _TEST_RULE, n_steps=5, backend="numpy")
    assert a.shape == (5, 6, 6)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Core dynamics: an interior intervention should perturb the future
# ---------------------------------------------------------------------------


def test_internal_intervention_changes_future() -> None:
    """With a small grid and a propagating rule, flipping interior 4D cells
    should produce strictly positive divergence vs. the baseline."""
    rng = np.random.default_rng(7)
    snap = (rng.random((8, 8, 2, 2)) < 0.4).astype(np.uint8)

    interior = np.zeros((8, 8), dtype=bool)
    interior[3:5, 3:5] = True
    boundary = np.zeros((8, 8), dtype=bool)
    boundary[2, 2:6] = True
    boundary[5, 2:6] = True
    boundary[2:6, 2] = True
    boundary[2:6, 5] = True
    env = np.zeros((8, 8), dtype=bool)
    env[0, :] = True
    env[7, :] = True
    env[:, 0] = True
    env[:, 7] = True

    result = compute_causality_score(
        snap,
        rule=_TEST_RULE,
        interior_mask_2d=interior,
        boundary_mask_2d=boundary,
        env_mask_2d=env,
        n_steps=5,
        flip_fraction=1.0,  # maximize the chance of *some* propagation
        backend="numpy",
        seed=42,
    )

    assert result.valid is True
    assert result.reason == "ok"
    assert result.divergence_internal > 0.0, (
        f"expected positive internal divergence; got {result.divergence_internal}"
    )


# ---------------------------------------------------------------------------
# Helper functions: hidden-shuffle preserves counts
# ---------------------------------------------------------------------------


def test_hidden_shuffle_preserves_counts() -> None:
    """For every (x, y) where mask_2d is True, the (z, w) fiber sum is
    preserved by the shuffle intervention."""
    rng = np.random.default_rng(0)
    state = (rng.random((6, 6, 2, 2)) < 0.5).astype(np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True

    out = apply_hidden_shuffle_intervention(
        state, mask, np.random.default_rng(99)
    )

    # Same shape, dtype, total mass.
    assert out.shape == state.shape
    assert out.dtype == state.dtype

    # Per-(x, y) fiber sums preserved everywhere; in fact preserved
    # universally because shuffle of unmasked fibers is the identity.
    sums_before = state.sum(axis=(2, 3))
    sums_after = out.sum(axis=(2, 3))
    assert np.array_equal(sums_before, sums_after), (
        f"per-fiber sums changed: max diff = "
        f"{int(np.abs(sums_before - sums_after).max())}"
    )

    # Outside the mask, the state must be untouched.
    outside = ~mask
    assert np.array_equal(state[outside], out[outside])


# ---------------------------------------------------------------------------
# Helper functions: flip count matches fraction
# ---------------------------------------------------------------------------


def test_flip_intervention_count_matches_fraction() -> None:
    """With a mask covering N=20 4D cells and flip_fraction=0.5, exactly
    round(N*0.5)=10 cells differ from the original."""
    state = np.zeros((4, 4, 2, 2), dtype=np.uint8)
    # Build a mask covering exactly 5 (x, y) cells -> 5 * 2 * 2 = 20 4D cells.
    mask = np.zeros((4, 4), dtype=bool)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    for x, y in coords:
        mask[x, y] = True
    n_target = int(mask.sum()) * state.shape[2] * state.shape[3]
    assert n_target == 20

    out = apply_flip_intervention(
        state, mask, flip_fraction=0.5, rng=np.random.default_rng(0)
    )

    diff = (state ^ out).astype(bool)
    assert int(diff.sum()) == 10, (
        f"expected 10 flips out of {n_target}, got {int(diff.sum())}"
    )

    # All flips must be inside the broadcasted mask.
    mask_4d = np.broadcast_to(mask[:, :, None, None], state.shape)
    assert np.array_equal(diff, diff & mask_4d)


def test_flip_intervention_zero_fraction_is_identity() -> None:
    """flip_fraction=0 -> result is an unmodified copy (different ndarray)."""
    rng = np.random.default_rng(0)
    state = (rng.random((4, 4, 2, 2)) < 0.4).astype(np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    out = apply_flip_intervention(
        state, mask, flip_fraction=0.0, rng=np.random.default_rng(1)
    )
    assert np.array_equal(state, out)
    # And it must be a copy, not the same buffer.
    assert out is not state
