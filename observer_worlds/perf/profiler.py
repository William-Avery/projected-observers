"""Lightweight phase-timing profiler for follow-up experiments.

Usage::

    from observer_worlds.perf import Profiler

    prof = Profiler(label="projection_robustness_smoke")
    with prof.phase("substrate"):
        run_substrate(...)
        prof.count("timesteps", T)
    with prof.phase("projection"):
        for proj in suite.names():
            project_one(...)
            prof.count("projections", 1)
    with prof.phase("stats"):
        compute_stats(...)
    prof.snapshot_memory()              # optional, requires psutil
    prof.snapshot_gpu_memory()          # optional, requires cupy
    prof.write_json(out_path)

The profiler is intentionally minimal:

* It never claims to be a sampling profiler. Phase timings are
  cumulative wall-time spent inside ``prof.phase(name)`` blocks.
* ``psutil`` and ``cupy`` are imported lazily; both are optional. If
  unavailable, the corresponding snapshots silently no-op.
* No threading / async support — phases must be entered and exited on
  the calling thread.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class Profiler:
    """Phase timings + counters + optional resource snapshots."""

    def __init__(self, label: str = "experiment") -> None:
        self.label = label
        self._t_start = time.perf_counter()
        self._wall_start = _dt.datetime.now(_dt.timezone.utc)
        self._phases: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._memory_snapshots: list[dict] = []
        self._gpu_snapshots: list[dict] = []

    # -- phases -------------------------------------------------------

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Accumulate wall-clock time spent inside the block under ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (
                time.perf_counter() - t0
            )

    def add_phase_seconds(self, name: str, seconds: float) -> None:
        """Externally record time for a phase (e.g. measured in a subprocess)."""
        self._phases[name] = self._phases.get(name, 0.0) + float(seconds)

    # -- counters -----------------------------------------------------

    def count(self, name: str, n: int = 1) -> None:
        self._counts[name] = self._counts.get(name, 0) + int(n)

    # -- optional resource snapshots ----------------------------------

    def snapshot_memory(self, tag: str = "") -> dict | None:
        """Record process RSS (and optionally VMS). No-op if ``psutil``
        is unavailable."""
        try:
            import psutil  # type: ignore
        except ImportError:
            return None
        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        rec = {
            "tag": tag,
            "t_seconds_since_start": time.perf_counter() - self._t_start,
            "rss_bytes": int(info.rss),
            "vms_bytes": int(getattr(info, "vms", 0)),
        }
        self._memory_snapshots.append(rec)
        return rec

    def snapshot_gpu_memory(self, tag: str = "") -> dict | None:
        """Record CuPy default memory pool stats. No-op if cupy is
        unavailable or no GPU is initialised."""
        try:
            import cupy  # type: ignore
        except ImportError:
            return None
        try:
            mempool = cupy.get_default_memory_pool()
            used = int(mempool.used_bytes())
            total = int(mempool.total_bytes())
        except Exception:  # noqa: BLE001 - cupy can raise many things
            return None
        rec = {
            "tag": tag,
            "t_seconds_since_start": time.perf_counter() - self._t_start,
            "cupy_pool_used_bytes": used,
            "cupy_pool_total_bytes": total,
        }
        self._gpu_snapshots.append(rec)
        return rec

    # -- reporting ----------------------------------------------------

    def total_seconds(self) -> float:
        return time.perf_counter() - self._t_start

    def report(self) -> dict:
        total = self.total_seconds()
        sum_phases = sum(self._phases.values())
        # Throughput counters of interest, computed on demand.
        throughput: dict[str, float] = {}
        for k, n in self._counts.items():
            if total > 0:
                throughput[f"{k}_per_second"] = n / total
        return {
            "label": self.label,
            "started_at_utc": self._wall_start.replace(microsecond=0)
                .isoformat().replace("+00:00", "Z"),
            "total_seconds": total,
            "phases_seconds": dict(self._phases),
            "phase_total_seconds": sum_phases,
            "phase_unaccounted_seconds": max(0.0, total - sum_phases),
            "counts": dict(self._counts),
            "throughput": throughput,
            "memory_snapshots": list(self._memory_snapshots),
            "gpu_snapshots": list(self._gpu_snapshots),
            "system": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
                "cpu_count": os.cpu_count(),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the report as JSON. Returns the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
        return p

    # -- compatibility helpers ---------------------------------------

    def reset(self) -> None:
        """Reset all phases / counters. ``label`` and ``_t_start``
        remain. Useful when reusing a profiler across stages."""
        self._phases.clear()
        self._counts.clear()
        self._memory_snapshots.clear()
        self._gpu_snapshots.clear()
        self._t_start = time.perf_counter()
        self._wall_start = _dt.datetime.now(_dt.timezone.utc)
