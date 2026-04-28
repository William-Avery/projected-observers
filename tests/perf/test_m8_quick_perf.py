"""tests/perf/test_m8_quick_perf.py

Moderate-scale M8 perf gate: ``run_m8_mechanism_discovery --quick``
on the cuda-batched backend must (a) complete in under the wall-time
gate, (b) actually measure candidates, and (c) write a non-empty
summary.md.

The candidate-count and summary-non-empty checks exist because the
original wall-time-only gate silently passed when every cell errored
on a stale validator (commit ``cbc0a79`` fixed that bug). A perf gate
that only measures wall time can't tell "fast and correct" from
"fast because everything errored immediately".

This is a **smoke gate** — not the full T1 production gate (which
requires a 30-min M7B reference run, see test_m7b_reference.py).

Skipped when cupy / a GPU is unavailable.
"""
from __future__ import annotations

import json
import re
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
def test_m8_quick_under_gate_cuda_batched(tmp_path: Path):
    """M8 --quick on cuda-batched: under the wall-time gate, with
    candidates actually measured and summary.md actually written."""
    baselines = json.loads(
        (REPO / "tests" / "perf" / "baselines.json").read_text(encoding="utf-8")
    )
    fast_gate = baselines["fast_perf_gates"]["m8_quick_cuda_batched"]
    gate = float(fast_gate["wall_time_seconds_max"])
    min_candidates = int(fast_gate["min_candidates"])

    rules_dir = REPO / "release" / "rules"
    label = "perf_gate_m8_quick"
    # Pin n_workers to a small value for the gate. The default
    # (cpu_count - 2) is 30 on the captured machine, and 30 worker
    # processes each instantiating their own cupy context on a single
    # 12 GB GPU triggers Windows "access violation" crashes. The
    # smoke runs only 6 cells, so 4 workers is plenty.
    cmd = [
        sys.executable, "-m",
        "observer_worlds.experiments.run_m8_mechanism_discovery",
        "--m7-rules", str(rules_dir / "m7_top_hce_rules.json"),
        "--m4c-rules", str(rules_dir / "m4c_evolve_leaderboard.json"),
        "--m4a-rules", str(rules_dir / "m4a_search_leaderboard.json"),
        "--quick",
        "--backend", "cuda-batched",
        "--n-workers", "4",
        "--label", label,
        "--out-root", str(tmp_path),
    ]
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    elapsed = time.time() - t0

    # 1. Process must exit clean.
    assert result.returncode == 0, (
        f"M8 quick run failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout[-2000:]}\n\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )

    # 2. Run must complete under the wall-time gate.
    assert elapsed < gate, (
        f"M8 --quick cuda-batched took {elapsed:.1f}s; gate is {gate:.1f}s. "
        f"Last stdout: {result.stdout.strip().splitlines()[-3:] if result.stdout else '(empty)'}"
    )

    # 3. Candidates must actually be measured (catches the
    #    "every cell errored silently" failure mode).
    match = re.search(r"Measured (\d+) candidates", result.stdout)
    assert match, (
        f"could not find 'Measured N candidates' line in stdout. "
        f"Last stdout:\n{result.stdout[-1500:]}"
    )
    n_candidates = int(match.group(1))
    assert n_candidates >= min_candidates, (
        f"M8 --quick measured {n_candidates} candidates; gate requires "
        f">= {min_candidates}. Likely a regression in the simulation or "
        f"detection pipeline. Last stdout:\n{result.stdout[-1500:]}"
    )

    # 4. Stderr must not contain per-cell error lines from the parallel
    #    sweep dispatcher (e.g. "m8 error rule=..."). One bad cell can
    #    happen on a flaky rule, but more than half the cells erroring
    #    is a regression.
    err_lines = [ln for ln in result.stderr.splitlines()
                 if "m8 error rule=" in ln]
    # _quick runs 6 cells (3 sources x 2 seeds); allow up to 1 flake.
    assert len(err_lines) <= 1, (
        f"{len(err_lines)} M8 cells errored:\n"
        + "\n".join(err_lines[:5])
    )

    # 5. summary.md must exist and be non-trivially populated.
    out_dirs = list(tmp_path.glob(f"{label}_*"))
    assert len(out_dirs) == 1, f"expected one run dir, found {out_dirs}"
    summary = out_dirs[0] / "summary.md"
    assert summary.exists(), f"summary.md missing at {summary}"
    summary_text = summary.read_text(encoding="utf-8")
    assert len(summary_text) > 200, (
        f"summary.md suspiciously small ({len(summary_text)} bytes); "
        f"first 500 chars: {summary_text[:500]}"
    )
    # Must include the headline section.
    assert "Mechanism Discovery" in summary_text, (
        f"summary.md missing expected header. Content:\n{summary_text[:500]}"
    )


# The long-form M7B-class production gate lives in test_m7b_reference.py;
# this file holds only the fast automatic gate.
