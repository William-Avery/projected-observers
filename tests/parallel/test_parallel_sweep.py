"""tests/parallel/test_parallel_sweep.py"""
from __future__ import annotations

import os

import numpy as np
import pytest

from observer_worlds.parallel import parallel_sweep


def _square(x: int) -> int:
    return x * x


def test_parallel_sweep_preserves_order():
    items = list(range(50))
    out = parallel_sweep(items, _square, n_workers=4)
    assert out == [i * i for i in items]


def test_parallel_sweep_serial_fallback_for_n_workers_1():
    items = [1, 2, 3]
    out = parallel_sweep(items, _square, n_workers=1)
    assert out == [1, 4, 9]


def _check_numba_threads_env(_) -> str:
    return os.environ.get("NUMBA_NUM_THREADS", "<unset>")


def test_parallel_sweep_workers_have_numba_threads_pinned_to_1():
    items = list(range(8))
    out = parallel_sweep(items, _check_numba_threads_env, n_workers=4)
    assert all(v == "1" for v in out), out
