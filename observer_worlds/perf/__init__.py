"""Performance utilities for follow-up experiments.

Exposes :class:`Profiler`, the lightweight phase-timer that follow-up
experiment runners use to record per-phase wall time, throughput
counters, and optional process-RSS / GPU-memory snapshots.

Importing this package does NOT import ``psutil`` or ``cupy``; those
are looked up lazily by the profiler when it is asked for memory or
GPU stats.
"""
from __future__ import annotations

from .profiler import Profiler

__all__ = ["Profiler"]
