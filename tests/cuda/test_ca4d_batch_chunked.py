"""tests/cuda/test_ca4d_batch_chunked.py

evolve_chunked must produce per-grid results identical to a single big
batched run when chunk_size >= K, and identical to independent CA4D runs
when split across multiple chunks.
"""
from __future__ import annotations

import numpy as np

from observer_worlds.utils import seeded_rng
from observer_worlds.worlds import CA4D, BSRule
from observer_worlds.worlds.ca4d_batch import CA4DBatch, evolve_chunked


def test_chunked_run_matches_independent_runs():
    shape = (8, 8, 4, 4)
    rule = BSRule(birth=(3, 4, 5), survival=(2, 3, 4, 5))
    K = 6
    seeds = list(range(100, 100 + K))
    n_steps = 5

    init = np.empty((K, *shape), dtype=np.uint8)
    for k, s in enumerate(seeds):
        init[k] = (seeded_rng(s).random(shape) < 0.3).astype(np.uint8)

    # Chunked run with max_chunk=2 forces 3 chunks.
    chunked = evolve_chunked(
        shape=shape, rules=[rule] * K,
        initial_states_host=init, n_steps=n_steps, max_chunk=2,
    )

    # Independent reference runs.
    expected = np.empty((K, *shape), dtype=np.uint8)
    for k in range(K):
        ca = CA4D(shape=shape, rule=rule, backend="cuda")
        ca.state = init[k]
        for _ in range(n_steps):
            ca.step()
        expected[k] = ca.state

    np.testing.assert_array_equal(chunked, expected)


def test_chunked_run_unchunked_matches_single_batch():
    """max_chunk=None (single batch) and max_chunk=K (one chunk of size K)
    must produce identical output."""
    shape = (8, 8, 4, 4)
    rule = BSRule(birth=(3, 4, 5), survival=(2, 3, 4, 5))
    K = 4
    seeds = list(range(200, 200 + K))
    n_steps = 5

    init = np.empty((K, *shape), dtype=np.uint8)
    for k, s in enumerate(seeds):
        init[k] = (seeded_rng(s).random(shape) < 0.3).astype(np.uint8)

    a = evolve_chunked(
        shape=shape, rules=[rule] * K,
        initial_states_host=init, n_steps=n_steps,
    )
    b = evolve_chunked(
        shape=shape, rules=[rule] * K,
        initial_states_host=init, n_steps=n_steps, max_chunk=K,
    )
    np.testing.assert_array_equal(a, b)
