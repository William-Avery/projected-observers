"""Process-parallel sweep over independent work items.

A work item is anything picklable; the caller-supplied ``fn`` is invoked
once per item and its return value is collected. Order is preserved.

When ``parallel_sweep`` runs with more than one worker, the env vars
``NUMBA_NUM_THREADS`` / ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` are set
to ``"1"`` in the parent process before joblib spawns its loky workers.
Loky workers inherit ``os.environ`` at spawn time, so setting these in
the parent is what propagates the pin to children. This avoids
oversubscription when several workers each run numba's intra-step
``prange``.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence


def _pin_thread_limits() -> None:
    """Set thread-count env vars to 1 in the current process.

    Loky workers inherit ``os.environ`` at spawn, so calling this in the
    parent before ``joblib.Parallel(...)`` is what propagates the pin to
    children. Has no effect on numba threads in the *current* process if
    numba was already imported (numba reads the env at import time).
    """
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
        Picklable, sized sequence of work items. Each item is passed
        positionally to ``fn``.
    fn:
        Callable invoked once per item. Must be importable from a module
        (lambdas / closures are not picklable across processes).
    n_workers:
        Number of worker processes. Defaults to ``cpu_count - 2``.
        ``n_workers == 1`` runs serially in-process (no joblib overhead).
    progress:
        Optional completion callback. Called once with a human-readable
        string when the sweep finishes. Per-item progress is not emitted;
        callers needing live updates for long sweeps should use joblib's
        ``verbose`` setting directly or wrap ``fn`` to log on each call.
    """
    n = _default_n_workers() if n_workers is None else int(n_workers)
    if n == 1 or len(items) <= 1:
        # Serial path: avoids joblib import + spawn overhead for tiny sweeps.
        results = [fn(item) for item in items]
        if progress is not None:
            progress(f"  serial sweep complete: {len(items)} items")
        return results

    # Pin thread limits in the parent so loky workers inherit them at
    # spawn. Done here (not at module import) to avoid a surprising
    # side effect for callers who only import the module.
    _pin_thread_limits()

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n, backend="loky", verbose=0)(
        delayed(fn)(item) for item in items
    )

    if progress is not None:
        progress(f"  parallel sweep complete: {len(items)} items, n_workers={n}")
    return list(results)
