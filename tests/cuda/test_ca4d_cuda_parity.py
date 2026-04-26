"""tests/cuda/test_ca4d_cuda_parity.py

CUDA backend produces statistically equivalent rollouts to the numba
canonical path. Per spec: per-step mean active fraction must agree
within 1% absolute on a 16x16x4x4 grid over 50 steps."""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.utils import seeded_rng
from observer_worlds.worlds import CA4D, BSRule


@pytest.fixture
def small_rule() -> BSRule:
    # Generous birth/survival: lots of activity to expose any kernel bug.
    return BSRule(birth=(3, 4, 5), survival=(2, 3, 4, 5))


def _step_n(backend: str, shape, rule, seed: int, n_steps: int) -> list[float]:
    """Return per-step mean active fraction over n_steps."""
    ca = CA4D(shape=shape, rule=rule, backend=backend)
    ca.initialize_random(density=0.30, rng=seeded_rng(seed))
    out = []
    for _ in range(n_steps):
        ca.step()
        out.append(float(np.mean(ca.state)))
    return out


def test_cuda_per_step_active_matches_numba(small_rule):
    shape = (16, 16, 4, 4)
    seed = 1234
    n_steps = 50
    cpu = _step_n("numba", shape, small_rule, seed, n_steps)
    gpu = _step_n("cuda", shape, small_rule, seed, n_steps)

    assert len(cpu) == len(gpu) == n_steps
    cpu_arr = np.asarray(cpu)
    gpu_arr = np.asarray(gpu)
    abs_diff = np.abs(cpu_arr - gpu_arr)
    assert abs_diff.max() < 0.01, (cpu, gpu, abs_diff.max())


def test_cuda_initial_state_matches_numba(small_rule):
    """Same seed + density -> same initial state on both backends.
    initialize_random uses host-side numpy randomness regardless of backend."""
    shape = (8, 8, 4, 4)
    seed = 99
    ca_cpu = CA4D(shape=shape, rule=small_rule, backend="numba")
    ca_gpu = CA4D(shape=shape, rule=small_rule, backend="cuda")
    ca_cpu.initialize_random(density=0.30, rng=seeded_rng(seed))
    ca_gpu.initialize_random(density=0.30, rng=seeded_rng(seed))
    np.testing.assert_array_equal(ca_cpu.state, ca_gpu.state)


def test_cuda_single_step_bit_identical_to_numba(small_rule):
    """For a fixed initial state, one CUDA step must produce the same
    bytes as one numba step. The CA update is deterministic - there's no
    floating-point in the neighbor sum (uint8) or LUT lookup (bool), so
    parity here should be exact, not just statistical."""
    shape = (8, 8, 4, 4)
    seed = 42

    ca_cpu = CA4D(shape=shape, rule=small_rule, backend="numba")
    ca_gpu = CA4D(shape=shape, rule=small_rule, backend="cuda")
    ca_cpu.initialize_random(density=0.40, rng=seeded_rng(seed))
    ca_gpu.initialize_random(density=0.40, rng=seeded_rng(seed))

    ca_cpu.step()
    ca_gpu.step()

    np.testing.assert_array_equal(ca_cpu.state, ca_gpu.state)
