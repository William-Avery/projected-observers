"""Performance test configuration.

Two gating mechanisms:

* ``--perf-long`` — opts in to long-form gates (M7B-class production
  baseline, ~30 min). Without this flag, tests marked
  ``@pytest.mark.perf_long`` skip.

* ``--perf-gate`` — promotes warnings to hard failures. With
  ``--perf-long`` alone, a regression past the tolerance threshold
  emits a warning and the test still passes. With both flags, the
  same regression fails the test.

Short gates (the moderate M8 smoke in ``test_m8_quick_perf.py``) run
by default whenever cupy is available, independent of these flags.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--perf-long",
        action="store_true",
        default=False,
        help="Run long-form performance tests (M7B-class production baseline, ~30 min).",
    )
    parser.addoption(
        "--perf-gate",
        action="store_true",
        default=False,
        help="In long-form perf tests, treat tolerance regressions as hard failures rather than warnings.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--perf-long"):
        return
    skip = pytest.mark.skip(reason="long perf test; pass --perf-long to run")
    for item in items:
        if "perf_long" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def perf_gate_strict(request) -> bool:
    """Whether to treat perf regressions as hard failures (--perf-gate)."""
    return bool(request.config.getoption("--perf-gate"))
