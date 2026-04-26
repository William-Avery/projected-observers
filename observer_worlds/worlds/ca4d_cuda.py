"""4D Moore-r1 CA on CUDA via cupy RawKernel.

The kernel mirrors :func:`observer_worlds.worlds.ca4d._update_4d_numba_core`:
each thread computes the next state of one (x, y, z, w) cell by summing
its 80 neighbors with periodic boundaries, then applying the birth /
survival LUT. State is uint8; LUTs are uint8 (0/1) of size 81.
"""

from __future__ import annotations

import numpy as np

from observer_worlds.worlds._cuda_bootstrap import bootstrap_cuda_path
from observer_worlds.worlds.rules import BSRule

# Must run before `import cupy` (cupy caches CUDA_PATH on first import).
bootstrap_cuda_path()


try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # pragma: no cover
    HAS_CUPY = False


_MAX_NEIGHBOURS_4D = 80


_KERNEL_SRC = r"""
extern "C" __global__
void update_4d(const unsigned char* __restrict__ in,
               unsigned char* __restrict__ out,
               const unsigned char* __restrict__ birth_lut,
               const unsigned char* __restrict__ surv_lut,
               int Nx, int Ny, int Nz, int Nw) {
    long long total = (long long)Nx * Ny * Nz * Nw;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w =  idx % Nw;
    long long t1 = idx / Nw;
    int z =  t1 % Nz;
    long long t2 = t1 / Nz;
    int y =  t2 % Ny;
    int x =  (int)(t2 / Ny);

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
                    long long nidx = ((long long)xi * Ny + yi) * Nz * Nw
                                    + (long long)zi * Nw + wi;
                    count += in[nidx];
                }
            }
        }
    }

    unsigned char alive = in[idx];
    out[idx] = alive ? surv_lut[count] : birth_lut[count];
}
"""


_KERNEL = None  # lazy compile


def _compile():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_KERNEL_SRC, "update_4d")
    return _KERNEL


def update_4d_cuda(state, rule: BSRule):
    """One CA step on device. ``state`` may be a cupy or numpy ndarray;
    return value is a fresh cupy ndarray on device."""
    if not HAS_CUPY:  # pragma: no cover
        raise RuntimeError(
            "cupy is not installed; install with `pip install cupy-cuda12x`."
        )
    if state.ndim != 4:
        raise ValueError(f"update_4d_cuda expects a 4D array, got {state.ndim}D")

    if not isinstance(state, cp.ndarray):
        state_d = cp.asarray(state, dtype=cp.uint8)
    else:
        state_d = cp.ascontiguousarray(state, dtype=cp.uint8)

    Nx, Ny, Nz, Nw = state_d.shape

    birth_lut, surv_lut = rule.to_lookup_tables(_MAX_NEIGHBOURS_4D)
    # uint8 LUTs (cupy bool ABI is sometimes finicky inside RawKernels).
    birth_d = cp.asarray(birth_lut.astype(np.uint8))
    surv_d = cp.asarray(surv_lut.astype(np.uint8))

    out_d = cp.empty_like(state_d)

    total = Nx * Ny * Nz * Nw
    block = 256
    grid = (total + block - 1) // block

    kernel = _compile()
    kernel((grid,), (block,),
           (state_d, out_d, birth_d, surv_d,
            np.int32(Nx), np.int32(Ny), np.int32(Nz), np.int32(Nw)))
    return out_d
