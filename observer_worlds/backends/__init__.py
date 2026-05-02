"""Pluggable array backends for batched 4D-CA work.

Two implementations:

* :class:`observer_worlds.backends.numpy_backend.NumpyBackend` — CPU
  reference; the source of truth for scientific results.
* :class:`observer_worlds.backends.cupy_backend.CupyBackend` — CuPy /
  CUDA implementation; subject to per-primitive equivalence audit
  before being used for any production claim.

Public entry points:

* :func:`get_backend(name, device=0)` — return an initialized backend.
* :func:`is_cupy_available()` — True iff cupy + CUDA reachable.
* :func:`estimate_max_safe_batch_size(...)` — memory planner used by
  the GPU benchmark harness.
"""
from observer_worlds.backends.backend_api import (
    Backend,
    GpuMemoryEstimate,
    estimate_max_safe_batch_size,
    get_backend,
    is_cupy_available,
)

__all__ = [
    "Backend",
    "GpuMemoryEstimate",
    "estimate_max_safe_batch_size",
    "get_backend",
    "is_cupy_available",
]
