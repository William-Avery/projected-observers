"""Performance test configuration.

Long-form gates (M7B reference at production scale, etc.) are gated
behind ``--perf-long`` so they don't block the default test loop.
Short gates (the moderate M8 smoke) run by default whenever cupy is
available.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--perf-long",
        action="store_true",
        default=False,
        help="Run long-form performance tests (M7B reference, ~30 min).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--perf-long"):
        return
    skip = pytest.mark.skip(reason="long perf test; pass --perf-long to run")
    for item in items:
        if "perf_long" in item.keywords:
            item.add_marker(skip)
