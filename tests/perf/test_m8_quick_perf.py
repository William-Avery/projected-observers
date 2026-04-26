"""tests/perf/test_m8_quick_perf.py

Moderate-scale M8 perf gate: ``run_m8_mechanism_discovery --quick``
on the cuda-batched backend must complete in under 9 seconds end-to-end.

This is a **smoke gate** — not the full T1 production gate (which
requires a 30-min M7B reference run, see test_m7b_reference.py). The
smoke gate catches obvious regressions in CA stepping or batched
response-map dispatch without blocking the default test loop.

Skipped when cupy / a GPU is unavailable.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


def _cuda_available() -> bool:
    try:
        import cupy
        return cupy.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="cuda required for perf gate")
def test_m8_quick_under_9s_cuda_batched(tmp_path: Path):
    """M8 --quick on cuda-batched: end-to-end under 9 seconds.

    The wall time includes Python startup, numba JIT, cuda kernel
    compile, and joblib worker spawn — so this is a pessimistic
    measurement. Empirical baseline at refactor time: 8.5s.
    """
    baselines = json.loads(
        (REPO / "tests" / "perf" / "baselines.json").read_text()
    )
    gate = float(baselines["m8_quick_cuda_batched_seconds"])

    rules_dir = REPO / "release" / "rules"
    cmd = [
        sys.executable, "-m",
        "observer_worlds.experiments.run_m8_mechanism_discovery",
        "--m7-rules", str(rules_dir / "m7_top_hce_rules.json"),
        "--m4c-rules", str(rules_dir / "m4c_evolve_leaderboard.json"),
        "--m4a-rules", str(rules_dir / "m4a_search_leaderboard.json"),
        "--quick",
        "--backend", "cuda-batched",
        "--label", "perf_gate_m8_quick",
        "--out-root", str(tmp_path),
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    elapsed = time.time() - t0

    assert result.returncode == 0, (
        f"M8 quick run failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout[-2000:]}\n\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert elapsed < gate, (
        f"M8 --quick cuda-batched took {elapsed:.1f}s; gate is {gate:.1f}s. "
        f"Last stdout: {result.stdout.strip().splitlines()[-3:] if result.stdout else '(empty)'}"
    )


@pytest.mark.perf_long
@pytest.mark.skipif(not _cuda_available(), reason="cuda required")
def test_m7b_reference_under_30min_placeholder(tmp_path: Path):
    """Long-form M7B reference gate: production config under 30 min.

    Currently a placeholder. To enable: run the M7B reference config
    (5 rules per source x 50 test seeds x T=500 x 32x32x4x4) on the
    14900k + 3080 Ti, record the wall time as
    ``m7b_reference_seconds_observed`` in baselines.json, then replace
    this skip with the actual gate.
    """
    pytest.skip("M7B reference baseline not yet captured at production scale")
