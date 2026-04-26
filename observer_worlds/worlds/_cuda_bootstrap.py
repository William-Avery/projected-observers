"""Helper: set CUDA_PATH from the nvidia-cuda-nvrtc-cu12 pip wheel.

cupy auto-detects CUDA_PATH on Windows by scanning a few well-known wheel
layouts. The detection only handles the CUDA 13+ unified layout
(``site-packages/nvidia/cuXX/...``); for the CUDA 12.x splayed layout used
by ``nvidia-cuda-nvrtc-cu12`` (``site-packages/nvidia/cuda_nvrtc/bin/...``)
it returns None, and any subsequent ``RawKernel`` compile fails with
"Failed to auto-detect CUDA root directory" even though the dll is present
and loadable through ``cuda.pathfinder``.

cupy caches the detected path on first import (inside
``_setup_win32_dll_directory``), so this bootstrap must run *before*
``import cupy``. Importers in ``ca4d.py`` and ``ca4d_cuda.py`` call this
prior to their cupy imports.
"""

from __future__ import annotations

import os


def bootstrap_cuda_path() -> None:
    """If CUDA_PATH is unset and ``nvidia-cuda-nvrtc-cu12`` is installed,
    point CUDA_PATH at the wheel's directory."""
    if os.environ.get("CUDA_PATH"):
        return
    try:
        import nvidia.cuda_nvrtc as _nvrtc_pkg  # type: ignore[import-not-found]
    except ImportError:
        return
    # nvidia.cuda_nvrtc is a namespace package; __file__ may be None. Use
    # __path__ entries (list of dirs the namespace covers).
    paths = list(getattr(_nvrtc_pkg, "__path__", []))
    if not paths:
        return
    pkg_dir = paths[0]
    if os.path.isdir(os.path.join(pkg_dir, "bin")):
        os.environ["CUDA_PATH"] = pkg_dir
