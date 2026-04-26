"""tests/cuda/test_ca4d_batch_parity.py

Batched evolution of K identical-rule grids must produce per-grid
trajectories byte-identical to K independent CA4D runs."""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.utils import seeded_rng
from observer_worlds.worlds import CA4D, BSRule
from observer_worlds.worlds.ca4d_batch import CA4DBatch


def test_batch_matches_independent_runs():
    shape = (12, 12, 4, 4)
    rule = BSRule(birth=(3, 4, 5), survival=(2, 3, 4, 5))
    seeds = [1, 7, 13, 42]
    densities = [0.3, 0.3, 0.4, 0.4]
    n_steps = 20

    # Independent runs.
    indep = []
    for s, d in zip(seeds, densities):
        ca = CA4D(shape=shape, rule=rule, backend="cuda")
        ca.initialize_random(density=d, rng=seeded_rng(s))
        for _ in range(n_steps):
            ca.step()
        indep.append(ca.state)  # host copy via property

    # Batched run.
    batch = CA4DBatch.from_rules(
        shape=shape,
        rules=[rule] * len(seeds),
        seeds=seeds,
        initial_density=densities,
    )
    for _ in range(n_steps):
        batch.step()
    for b, expected in enumerate(indep):
        np.testing.assert_array_equal(batch.state_at(b), expected)


def test_batch_supports_per_batch_rules():
    """Different rules in different batch slots must each evolve under
    their own rule."""
    shape = (8, 8, 4, 4)
    r0 = BSRule(birth=(3,), survival=(2, 3))
    r1 = BSRule(birth=(3, 4, 5, 6), survival=(2, 3, 4, 5, 6, 7))
    seeds = [101, 102]

    batch = CA4DBatch.from_rules(
        shape=shape, rules=[r0, r1], seeds=seeds, initial_density=[0.3, 0.3],
    )
    for _ in range(10):
        batch.step()

    expected = []
    for s, r in zip(seeds, [r0, r1]):
        ca = CA4D(shape=shape, rule=r, backend="cuda")
        ca.initialize_random(density=0.3, rng=seeded_rng(s))
        for _ in range(10):
            ca.step()
        expected.append(ca.state)

    np.testing.assert_array_equal(batch.state_at(0), expected[0])
    np.testing.assert_array_equal(batch.state_at(1), expected[1])


def test_batch_from_states_explicit_initial():
    """from_states accepts an explicit (B, Nx, Ny, Nz, Nw) host array."""
    shape = (8, 8, 4, 4)
    rule = BSRule(birth=(3,), survival=(2, 3))
    rng = seeded_rng(99)
    init = (rng.random((3, *shape)) < 0.3).astype(np.uint8)

    batch = CA4DBatch.from_states(states_host=init, rules=[rule] * 3)
    for _ in range(5):
        batch.step()

    expected = []
    for b in range(3):
        ca = CA4D(shape=shape, rule=rule, backend="cuda")
        ca.state = init[b]
        for _ in range(5):
            ca.step()
        expected.append(ca.state)

    for b in range(3):
        np.testing.assert_array_equal(batch.state_at(b), expected[b])
