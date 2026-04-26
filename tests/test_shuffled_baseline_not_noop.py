"""Regression tests for the M3 "shuffled-4D is a no-op" bug.

The original M3 implementation of the shuffled-hidden baseline applied
the hidden-state mutator to a *copy* of ``ca.state`` before projection,
but never wrote the mutated state back into ``ca``.  Because
``mean_threshold`` projection only looks at per-(x, y) column counts --
which the shuffle preserves by construction -- the projected sequences
of the coherent and shuffled runs were byte-identical, defeating the
whole point of the baseline.

These tests pin the fix in two layers:

1. The in-memory helper :func:`_simulate_4d_capturing` actually diverges
   between coherent and shuffled runs.
2. The production pipeline :func:`simulate_4d_to_zarr` re-injects the
   mutated state into ``ca.state`` so successive steps evolve from the
   shuffled lattice.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from observer_worlds.experiments._m4b_sweep import (
    _simulate_4d_capturing,
    hidden_shuffle_mutator,
)
from observer_worlds.experiments._pipeline import simulate_4d_to_zarr
from observer_worlds.search import FractionalRule
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils.config import RunConfig


# ---------------------------------------------------------------------------
# Mutator unit tests
# ---------------------------------------------------------------------------


def test_hidden_shuffle_preserves_per_column_counts():
    """Shuffling z,w fibers must not change the per-(x, y) active count."""
    rng = np.random.default_rng(0)
    state = np.zeros((4, 4, 2, 2), dtype=np.uint8)
    # Seed a few columns with mixed counts so we have something meaningful.
    state[0, 0] = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    state[1, 2] = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    state[2, 1] = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    state[3, 3] = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    mutated = hidden_shuffle_mutator(state.copy(), t=0, rng=rng)

    assert mutated.shape == state.shape
    np.testing.assert_array_equal(
        mutated.sum(axis=(2, 3)),
        state.sum(axis=(2, 3)),
        err_msg="per-(x, y) column counts must be preserved by the shuffle",
    )


def test_hidden_shuffle_changes_arrangement_when_possible():
    """For a column with both 0s and 1s, the shuffle must vary the
    arrangement across trials -- not always return the input verbatim."""
    state = np.zeros((1, 1, 4, 4), dtype=np.uint8)
    # 8 ones / 8 zeros => many distinct permutations.
    flat = np.array([1] * 8 + [0] * 8, dtype=np.uint8)
    state[0, 0] = flat.reshape(4, 4)

    rng = np.random.default_rng(42)
    distinct: set[bytes] = set()
    n_trials = 50
    for _ in range(n_trials):
        m = hidden_shuffle_mutator(state.copy(), t=0, rng=rng)
        # Count must still be preserved.
        assert int(m.sum()) == int(state.sum())
        distinct.add(bytes(m[0, 0].tobytes()))

    assert len(distinct) > 1, (
        "shuffled column was identical across all trials; the shuffle is "
        "not actually permuting"
    )


# ---------------------------------------------------------------------------
# In-memory simulator divergence
# ---------------------------------------------------------------------------


_VIABLE_RULE = FractionalRule(0.15, 0.26, 0.09, 0.38, 0.15)


def _hash_first_n(frames: np.ndarray, n: int = 16) -> str:
    n = min(n, frames.shape[0])
    return hashlib.sha1(frames[:n].tobytes()).hexdigest()


def test_simulate_capturing_coherent_vs_shuffled_diverge():
    """Coherent and shuffled in-memory runs of the same rule+seed must
    produce different projected sequences.  If the shuffle were a no-op
    (the M3 bug), the SHA-1 of the first 16 frames would match exactly."""
    bsrule = _VIABLE_RULE.to_bsrule()

    coh_frames, _, _ = _simulate_4d_capturing(
        bsrule,
        grid_shape=(16, 16, 4, 4),
        timesteps=30,
        initial_density=_VIABLE_RULE.initial_density,
        seed=1234,
        backend="numba",
        projection_method="mean_threshold",
        projection_theta=0.5,
        snapshot_at=[],
        state_mutator=None,
    )

    mutator_rng = np.random.default_rng(99)

    def _mutator(state, t, _rng):
        return hidden_shuffle_mutator(state, t, mutator_rng)

    shuf_frames, _, _ = _simulate_4d_capturing(
        bsrule,
        grid_shape=(16, 16, 4, 4),
        timesteps=30,
        initial_density=_VIABLE_RULE.initial_density,
        seed=1234,
        backend="numba",
        projection_method="mean_threshold",
        projection_theta=0.5,
        snapshot_at=[],
        state_mutator=_mutator,
    )

    h_coh = _hash_first_n(coh_frames, 16)
    h_shuf = _hash_first_n(shuf_frames, 16)
    assert h_coh != h_shuf, (
        "coherent and shuffled in-memory runs produced byte-identical "
        "projected frames; the M3 shuffled-baseline no-op bug has "
        "regressed"
    )


def test_simulate_capturing_writes_back_into_ca_state():
    """Re-statement of the same divergence using a different seed pair
    to guard against the hash collision being coincidental.  Documents
    the property: identical rule+seed but with vs without a shuffle
    mutator must yield different projected frames once the rule has had
    a chance to evolve."""
    bsrule = _VIABLE_RULE.to_bsrule()
    grid_shape = (16, 16, 4, 4)
    seed = 4321

    no_mut_frames, _, _ = _simulate_4d_capturing(
        bsrule,
        grid_shape=grid_shape,
        timesteps=30,
        initial_density=_VIABLE_RULE.initial_density,
        seed=seed,
        backend="numba",
        projection_method="mean_threshold",
        projection_theta=0.5,
        snapshot_at=[],
        state_mutator=None,
    )

    mutator_rng = np.random.default_rng(seed * 7919 + 1)

    def _mutator(state, t, _rng):
        return hidden_shuffle_mutator(state, t, mutator_rng)

    mut_frames, _, _ = _simulate_4d_capturing(
        bsrule,
        grid_shape=grid_shape,
        timesteps=30,
        initial_density=_VIABLE_RULE.initial_density,
        seed=seed,
        backend="numba",
        projection_method="mean_threshold",
        projection_theta=0.5,
        snapshot_at=[],
        state_mutator=_mutator,
    )

    # Final frame must differ -- if the mutator only operated on a copy,
    # mean_threshold projection would yield byte-identical frames.
    assert not np.array_equal(no_mut_frames[-1], mut_frames[-1]), (
        "final projected frame is identical with and without the shuffle "
        "mutator; the mutator is not affecting CA evolution"
    )

    # Whole-trace SHA-1 should also differ.
    h_no = hashlib.sha1(no_mut_frames.tobytes()).hexdigest()
    h_mut = hashlib.sha1(mut_frames.tobytes()).hexdigest()
    assert h_no != h_mut


# ---------------------------------------------------------------------------
# Pipeline-level regression test
# ---------------------------------------------------------------------------


def _build_run_config_for_pipeline() -> RunConfig:
    cfg = RunConfig()
    cfg.world.nx = 8
    cfg.world.ny = 8
    cfg.world.nz = 2
    cfg.world.nw = 2
    cfg.world.timesteps = 5
    cfg.world.initial_density = _VIABLE_RULE.initial_density
    # Use the BSRule derived from the viable FractionalRule used elsewhere
    # in this file -- it actually keeps the lattice alive on the tiny grid.
    bsrule = _VIABLE_RULE.to_bsrule()
    cfg.world.rule_birth = tuple(int(b) for b in bsrule.birth)
    cfg.world.rule_survival = tuple(int(s) for s in bsrule.survival)
    cfg.world.backend = "numpy"
    cfg.output.save_4d_snapshots = False
    cfg.output.save_gif = False
    return cfg


def test_shuffled_baseline_via_pipeline_writes_back_to_ca(tmp_path):
    """Strongest regression test: drives the production pipeline.

    1. Run with an identity mutator that just records the input it sees
       at each timestep.  The recorded inputs at consecutive timesteps
       must differ -- proving the CA actually evolves between calls and
       the pipeline calls the mutator on the post-step state, not on a
       single frozen array.
    2. Run again with a real shuffler.  The frames produced must differ
       from the identity-mutator run, proving the mutator's *output*
       affects subsequent evolution (i.e. the mutated state is written
       back into ``ca.state``).
    """
    cfg = _build_run_config_for_pipeline()

    # ---- Pass 1: identity mutator that records the inputs it sees. -------
    seen_states: list[np.ndarray] = []

    def identity_recording_mutator(state, t, _rng):
        seen_states.append(np.asarray(state).copy())
        return state

    store_id = ZarrRunStore(
        tmp_path / "run_identity",
        timesteps=cfg.world.timesteps,
        shape_2d=(cfg.world.nx, cfg.world.ny),
    )
    rng_id = np.random.default_rng(0)
    simulate_4d_to_zarr(cfg, store_id, rng_id, state_mutator=identity_recording_mutator)

    # The pipeline calls the mutator once per timestep (T calls total).
    assert len(seen_states) == cfg.world.timesteps, (
        f"expected {cfg.world.timesteps} mutator calls, saw {len(seen_states)}"
    )

    # Consecutive seen states must differ -- if they didn't, the CA isn't
    # being stepped between mutator calls, which would mask the bug.
    diffs_observed = 0
    for a, b in zip(seen_states, seen_states[1:]):
        if not np.array_equal(a, b):
            diffs_observed += 1
    assert diffs_observed >= 1, (
        "all recorded mutator-input states were identical across timesteps; "
        "the CA is not actually evolving between mutator calls"
    )

    frames_id = store_id.read_frames_2d()

    # ---- Pass 2: real shuffler that should diverge from the identity run. -
    shuffler_rng = np.random.default_rng(2024)

    def shuffler_mutator(state, t, _rng):
        return hidden_shuffle_mutator(state, t, shuffler_rng)

    store_shuf = ZarrRunStore(
        tmp_path / "run_shuffled",
        timesteps=cfg.world.timesteps,
        shape_2d=(cfg.world.nx, cfg.world.ny),
    )
    rng_shuf = np.random.default_rng(0)
    simulate_4d_to_zarr(cfg, store_shuf, rng_shuf, state_mutator=shuffler_mutator)

    frames_shuf = store_shuf.read_frames_2d()

    assert frames_id.shape == frames_shuf.shape

    # If the shuffler's output were ignored (the M3 bug), mean_threshold
    # projection would produce byte-identical frames in both runs.
    h_id = hashlib.sha1(frames_id.tobytes()).hexdigest()
    h_shuf = hashlib.sha1(frames_shuf.tobytes()).hexdigest()
    assert h_id != h_shuf, (
        "pipeline produced byte-identical projected frames with identity "
        "and shuffler mutators; ca.state is not being updated with the "
        "mutator's output (M3 bug regressed)"
    )
