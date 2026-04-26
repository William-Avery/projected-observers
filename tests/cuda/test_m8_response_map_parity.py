"""tests/cuda/test_m8_response_map_parity.py

The cuda-batched response map must produce ResponseMap aggregates within
numerical tolerance of the serial CPU version. Per spec Q3-C: each
*_response_fraction must agree within 5% absolute (bounded in [0,1]).

In practice they should agree byte-identically because both paths use
the same per-column shuffle RNG order and the cuda CA step is
bit-identical to numba.
"""
from __future__ import annotations

import numpy as np
import pytest

from observer_worlds.experiments._m8_mechanism import compute_response_map
from observer_worlds.utils import seeded_rng
from observer_worlds.worlds import BSRule


def _make_snapshot(shape, rule, seed, n_steps=10):
    from observer_worlds.worlds import CA4D
    ca = CA4D(shape=shape, rule=rule, backend="numba")
    ca.initialize_random(density=0.30, rng=seeded_rng(seed))
    for _ in range(n_steps):
        ca.step()
    return ca.state.copy()


def _make_interior_mask(Nx, Ny):
    m = np.zeros((Nx, Ny), dtype=bool)
    m[Nx // 2 - 2 : Nx // 2 + 3, Ny // 2 - 2 : Ny // 2 + 3] = True
    return m


def test_response_map_cuda_batched_matches_cpu():
    shape = (12, 12, 4, 4)
    rule = BSRule(birth=(3, 4, 5), survival=(2, 3, 4, 5))
    snap = _make_snapshot(shape, rule, seed=7)
    interior = _make_interior_mask(shape[0], shape[1])

    common = dict(
        snapshot_4d=snap, rule=rule, interior_mask=interior,
        candidate_id=0, horizon=8, n_replicates=2, rng_seed=12345,
    )
    cpu = compute_response_map(backend="numba", **common)
    gpu = compute_response_map(backend="cuda-batched", **common)

    # Per spec Q3-C: 5% absolute tolerance on bounded [0,1] aggregates.
    assert abs(cpu.interior_response_fraction - gpu.interior_response_fraction) < 0.05
    assert abs(cpu.boundary_response_fraction - gpu.boundary_response_fraction) < 0.05
    assert abs(cpu.environment_response_fraction - gpu.environment_response_fraction) < 0.05

    # Tighter check on the response_grid itself (which feeds all the
    # aggregates). Should be very close.
    np.testing.assert_allclose(
        cpu.response_grid, gpu.response_grid,
        atol=1e-9, rtol=1e-6,
        err_msg="cuda-batched response grid diverges from CPU reference",
    )
