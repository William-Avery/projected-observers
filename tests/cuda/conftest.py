"""Auto-skip the cuda test directory when cupy / a GPU is unavailable."""
from __future__ import annotations

import pytest


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return cupy.cuda.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):  # noqa: D401
    if _cupy_available():
        return
    skip = pytest.mark.skip(reason="cupy / CUDA not available")
    for item in items:
        if "tests/cuda" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)
