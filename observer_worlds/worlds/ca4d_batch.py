"""Batched 4D Moore-r1 CA on CUDA.

K independent grids of identical spatial shape are evolved together in
one kernel launch. Each grid carries its own birth/survival LUT, so K
*different* rules can run in parallel.

State shape: (B, Nx, Ny, Nz, Nw), uint8, on device.
LUTs shape:  (B, 81), uint8 (0/1), on device.

Use this class for: M8 response-map probes (B = #interior_columns x
n_replicates), rule-search fitness (B = #rules x #seeds), and any other
workload with K small independent grids.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from observer_worlds.worlds import _cuda_bootstrap as _bootstrap

_bootstrap.bootstrap_cuda_path()

from observer_worlds.worlds.rules import BSRule

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


_MAX_NEIGHBOURS_4D = 80


_BATCH_KERNEL_SRC = r"""
extern "C" __global__
void update_4d_batched(const unsigned char* __restrict__ in,
                       unsigned char* __restrict__ out,
                       const unsigned char* __restrict__ birth_lut,
                       const unsigned char* __restrict__ surv_lut,
                       int B, int Nx, int Ny, int Nz, int Nw) {
    long long per_grid = (long long)Nx * Ny * Nz * Nw;
    long long total = (long long)B * per_grid;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b = (int)(idx / per_grid);
    long long li = idx - (long long)b * per_grid;

    int w =  li % Nw;
    long long t1 = li / Nw;
    int z =  t1 % Nz;
    long long t2 = t1 / Nz;
    int y =  t2 % Ny;
    int x =  (int)(t2 / Ny);

    long long base = (long long)b * per_grid;

    int count = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        int xi = (x + dx + Nx) % Nx;
        for (int dy = -1; dy <= 1; ++dy) {
            int yi = (y + dy + Ny) % Ny;
            for (int dz = -1; dz <= 1; ++dz) {
                int zi = (z + dz + Nz) % Nz;
                for (int dw = -1; dw <= 1; ++dw) {
                    if (dx == 0 && dy == 0 && dz == 0 && dw == 0) continue;
                    int wi = (w + dw + Nw) % Nw;
                    long long nidx = base + ((long long)xi * Ny + yi) * Nz * Nw
                                    + (long long)zi * Nw + wi;
                    count += in[nidx];
                }
            }
        }
    }

    unsigned char alive = in[idx];
    int lut_off = b * 81;
    out[idx] = alive ? surv_lut[lut_off + count] : birth_lut[lut_off + count];
}
"""

_BATCH_KERNEL = None


def _compile_batch_kernel():
    global _BATCH_KERNEL
    if _BATCH_KERNEL is None:
        _BATCH_KERNEL = cp.RawKernel(_BATCH_KERNEL_SRC, "update_4d_batched")
    return _BATCH_KERNEL


class CA4DBatch:
    """K independent 4D CAs on device, stepped together."""

    def __init__(
        self,
        *,
        shape: tuple[int, int, int, int],
        state,
        birth_lut,
        surv_lut,
    ) -> None:
        if not HAS_CUPY:
            raise RuntimeError("cupy required for CA4DBatch")
        self.shape: tuple[int, int, int, int] = tuple(int(s) for s in shape)  # type: ignore[assignment]
        self._state = state
        self._birth_lut = birth_lut
        self._surv_lut = surv_lut
        self.B = int(state.shape[0])

    @classmethod
    def from_rules(
        cls,
        *,
        shape: tuple[int, int, int, int],
        rules: Sequence[BSRule],
        seeds: Sequence[int],
        initial_density: Sequence[float],
    ) -> "CA4DBatch":
        if not HAS_CUPY:
            raise RuntimeError("cupy required for CA4DBatch")
        if len(rules) != len(seeds) or len(rules) != len(initial_density):
            raise ValueError(
                "rules, seeds, initial_density must have equal length"
            )
        B = len(rules)
        Nx, Ny, Nz, Nw = shape

        host_state = np.empty((B, Nx, Ny, Nz, Nw), dtype=np.uint8)
        for b in range(B):
            rng = np.random.default_rng(int(seeds[b]))
            host_state[b] = (
                rng.random((Nx, Ny, Nz, Nw)) < float(initial_density[b])
            ).astype(np.uint8)

        host_birth = np.zeros((B, 81), dtype=np.uint8)
        host_surv = np.zeros((B, 81), dtype=np.uint8)
        for b in range(B):
            bl, sl = rules[b].to_lookup_tables(_MAX_NEIGHBOURS_4D)
            host_birth[b] = bl.astype(np.uint8)
            host_surv[b] = sl.astype(np.uint8)

        return cls(
            shape=shape,
            state=cp.asarray(host_state),
            birth_lut=cp.asarray(host_birth),
            surv_lut=cp.asarray(host_surv),
        )

    @classmethod
    def from_states(
        cls,
        *,
        states_host: np.ndarray,
        rules: Sequence[BSRule],
    ) -> "CA4DBatch":
        """Construct a batch from explicit initial states (e.g. for the M8
        response-map use case where each batch element is a copy of the same
        snapshot with a different per-column shuffle applied)."""
        if not HAS_CUPY:
            raise RuntimeError("cupy required for CA4DBatch")
        B = states_host.shape[0]
        if len(rules) != B:
            raise ValueError(
                f"rules length {len(rules)} != batch size {B}"
            )

        host_birth = np.zeros((B, 81), dtype=np.uint8)
        host_surv = np.zeros((B, 81), dtype=np.uint8)
        for b in range(B):
            bl, sl = rules[b].to_lookup_tables(_MAX_NEIGHBOURS_4D)
            host_birth[b] = bl.astype(np.uint8)
            host_surv[b] = sl.astype(np.uint8)

        return cls(
            shape=tuple(int(x) for x in states_host.shape[1:]),
            state=cp.asarray(np.ascontiguousarray(states_host, dtype=np.uint8)),
            birth_lut=cp.asarray(host_birth),
            surv_lut=cp.asarray(host_surv),
        )

    def step(self) -> None:
        """Advance all B grids one timestep with one kernel launch."""
        Nx, Ny, Nz, Nw = self.shape
        out = cp.empty_like(self._state)
        total = self.B * Nx * Ny * Nz * Nw
        block = 256
        grid = (total + block - 1) // block
        kernel = _compile_batch_kernel()
        kernel(
            (grid,), (block,),
            (self._state, out, self._birth_lut, self._surv_lut,
             np.int32(self.B), np.int32(Nx), np.int32(Ny),
             np.int32(Nz), np.int32(Nw)),
        )
        self._state = out

    @property
    def state(self):
        """Device-resident (B, Nx, Ny, Nz, Nw) array."""
        return self._state

    def state_at(self, b: int) -> np.ndarray:
        """Host copy of one batch element."""
        return cp.asnumpy(self._state[b])

    def states_host(self) -> np.ndarray:
        """Host copy of all batch elements (B, Nx, Ny, Nz, Nw)."""
        return cp.asnumpy(self._state)


def evolve_chunked(
    *,
    shape: tuple[int, int, int, int],
    rules: Sequence[BSRule],
    initial_states_host: np.ndarray,
    n_steps: int,
    max_chunk: int | None = None,
) -> np.ndarray:
    """Evolve K independent grids for ``n_steps`` and return host states.

    Splits the work into chunks of size ``max_chunk`` so very large K
    fits in VRAM. On ``cupy.cuda.OutOfMemoryError`` the chunk size is
    halved and the failing chunk retried. Returns ``(K, *shape)`` host
    array of final states.

    This is the recommended caller for M8 response-map probes and
    rule-search fitness eval when K may be large.
    """
    if not HAS_CUPY:
        raise RuntimeError("cupy required")
    K = int(initial_states_host.shape[0])
    if K != len(rules):
        raise ValueError(f"rules length {len(rules)} != K {K}")

    chunk = K if max_chunk is None else int(max_chunk)
    out = np.empty((K, *shape), dtype=np.uint8)
    i = 0
    while i < K:
        end = min(K, i + chunk)
        try:
            sub = CA4DBatch.from_states(
                states_host=initial_states_host[i:end],
                rules=list(rules[i:end]),
            )
            for _ in range(n_steps):
                sub.step()
            out[i:end] = sub.states_host()
            del sub
            cp.get_default_memory_pool().free_all_blocks()
            i = end
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            if chunk <= 1:
                raise RuntimeError(
                    "batch does not fit even at chunk=1; "
                    "shrink grid or n_steps"
                )
            chunk = max(1, chunk // 2)
    return out
