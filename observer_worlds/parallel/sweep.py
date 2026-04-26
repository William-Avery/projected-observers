"""Process-parallel sweep over independent work items.

A work item is anything picklable; the caller-supplied ``fn`` is invoked
once per item and its return value is collected. Order is preserved.

Workers initialize with ``NUMBA_NUM_THREADS=1`` so that numba's intra-step
``prange`` does not oversubscribe physical cores when several workers are
each running CA simulations.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence


def _worker_init() -> None:
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _default_n_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu - 2)


def parallel_sweep(
    items: Sequence,
    fn: Callable,
    *,
    n_workers: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list:
    """Map ``fn`` over ``items`` in parallel processes, preserving order.

    Parameters
    ----------
    items:
        Picklable work items. Each item is passed positionally to ``fn``.
    fn:
        Callable invoked once per item. Must be importable from a module
        (lambdas / closures are not picklable across processes).
    n_workers:
        Number of worker processes. Defaults to ``cpu_count - 2``.
        ``n_workers == 1`` runs serially in-process (no joblib overhead).
    progress:
        Optional callback. Called with a human-readable string after the
        sweep completes; not called per-item to avoid pickling overhead.
    """
    n = _default_n_workers() if n_workers is None else int(n_workers)
    if n == 1 or len(items) <= 1:
        # Serial path: avoids joblib import + spawn overhead for tiny sweeps.
        results = [fn(item) for item in items]
        if progress is not None:
            progress(f"  serial sweep complete: {len(items)} items")
        return results

    from joblib import Parallel, delayed

    results = Parallel(
        n_jobs=n,
        backend="loky",
        verbose=0,
        # `loky` initializer fires once per worker process.
        # joblib doesn't expose initializer in its public API for loky in older
        # versions; we set env vars at module import below as a fallback.
    )(delayed(fn)(item) for item in items)

    if progress is not None:
        progress(f"  parallel sweep complete: {len(items)} items, n_workers={n}")
    return list(results)


# Set the thread limits at module import too, so a worker that imports
# `parallel.sweep` early (or via dependency) gets the limits even if
# joblib doesn't run our explicit initializer.
_worker_init()
