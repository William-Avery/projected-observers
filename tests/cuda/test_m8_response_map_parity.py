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


def _make_snapshot(shape, rule, seed, *, density=0.30, n_steps=10):
    from observer_worlds.worlds import CA4D
    ca = CA4D(shape=shape, rule=rule, backend="numba")
    ca.initialize_random(density=density, rng=seeded_rng(seed))
    for _ in range(n_steps):
        ca.step()
    return ca.state.copy()


def _make_interior_mask(Nx, Ny):
    m = np.zeros((Nx, Ny), dtype=bool)
    m[Nx // 2 - 2 : Nx // 2 + 3, Ny // 2 - 2 : Ny // 2 + 3] = True
    return m


def test_response_map_cuda_batched_matches_cpu():
    shape = (12, 12, 4, 4)
    # A vigorous rule from the M4A viability leaderboard (top entry).
    # Density 0.147 produces sustained activity on a 12x12x4x4 grid; the
    # generic life-like rules all die out at this small grid size.
    rule = BSRule(
        birth=(12, 13, 14, 15, 16, 17, 18, 19, 20),
        survival=tuple(range(8, 31)),
    )
    snap = _make_snapshot(shape, rule, seed=7, density=0.147, n_steps=10)
    interior = _make_interior_mask(shape[0], shape[1])

    common = dict(
        snapshot_4d=snap, rule=rule, interior_mask=interior,
        candidate_id=0, horizon=8, n_replicates=2, rng_seed=12345,
    )
    cpu = compute_response_map(backend="numba", **common)
    gpu = compute_response_map(backend="cuda-batched", **common)

    # Sanity: the CPU reference must produce a non-trivial response grid;
    # otherwise the parity comparison is "0 == 0" and proves nothing.
    assert cpu.response_grid.sum() > 0.0, (
        "CPU response grid is all zeros -- the rule died out before the "
        "horizon. The parity test would not actually exercise the kernel."
    )

    # Per spec Q3-C: 5% absolute tolerance on bounded [0,1] aggregates.
    assert abs(cpu.interior_response_fraction - gpu.interior_response_fraction) < 0.05
    assert abs(cpu.boundary_response_fraction - gpu.boundary_response_fraction) < 0.05
    assert abs(cpu.environment_response_fraction - gpu.environment_response_fraction) < 0.05

    # Tighter check on the response_grid itself (which feeds all the
    # aggregates). Should be very close (byte-identical in practice
    # given the cuda CA step is bit-identical to numba and the per-column
    # shuffle RNG order matches between paths).
    np.testing.assert_allclose(
        cpu.response_grid, gpu.response_grid,
        atol=1e-9, rtol=1e-6,
        err_msg="cuda-batched response grid diverges from CPU reference",
    )
