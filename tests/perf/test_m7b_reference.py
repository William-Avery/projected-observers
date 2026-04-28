"""Long-form M7B-class production performance gate.

Skipped by default. Pass ``--perf-long`` to actually run the experiment
(~30 min on the captured-machine numpy backend; faster on cuda-batched).

Behavior:

* Reads the captured baseline at ``production_baselines/<variant>``
  in ``tests/perf/baselines.json``. ``--variant`` defaults to the
  numpy reference baseline.
* Re-runs the experiment with the same config and backend.
* Compares the new sweep wall time to the baseline.
* If the regression exceeds ``tolerance_policy.warn_regression_fraction``
  (currently 20%), the test issues a warning. With ``--perf-gate`` the
  warning becomes a hard assertion failure.
* Always asserts the run produced a non-zero candidate count and exited
  cleanly — those are correctness gates, not perf gates.

Refresh a baseline with::

    python tests/perf/capture_m7b_baseline.py \\
        --variant m8_m7b_class_numpy --backend numpy --update-baseline

Then commit the updated ``baselines.json``.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
BASELINES_PATH = REPO / "tests" / "perf" / "baselines.json"

DEFAULT_VARIANT = "m8_m7b_class_numpy"

pytestmark = pytest.mark.perf_long


def _load_baselines() -> dict:
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


def _backend_available(backend: str) -> bool:
    if backend in ("numpy", "numba"):
        return True
    if backend in ("cuda", "cuda-batched"):
        try:
            import cupy
            return cupy.cuda.is_available()
        except Exception:
            return False
    return False


_RE_SWEEP = re.compile(r"m8 sweep wall time (\d+(?:\.\d+)?)s")
_RE_CANDS = re.compile(r"Measured (\d+) candidates")


def _parse(stdout: str) -> tuple[int | None, int | None]:
    sweep = None
    cands = None
    m = _RE_SWEEP.search(stdout)
    if m:
        sweep = int(round(float(m.group(1))))
    m = _RE_CANDS.search(stdout)
    if m:
        cands = int(m.group(1))
    return sweep, cands


@pytest.mark.parametrize("variant", [DEFAULT_VARIANT])
def test_m7b_class_baseline_holds(
    variant: str, tmp_path: Path, perf_gate_strict: bool,
):
    """Re-run the captured baseline and check the sweep wall time
    against the recorded value within tolerance."""
    baselines = _load_baselines()
    prod = baselines.get("production_baselines", {})
    baseline = prod.get(variant)
    if baseline is None:
        pytest.skip(
            f"no captured baseline at production_baselines/{variant}; "
            f"capture one with tests/perf/capture_m7b_baseline.py"
        )

    backend = baseline["config"]["backend"]
    if not _backend_available(backend):
        pytest.skip(f"backend {backend!r} not available on this machine")

    cfg = baseline["config"]
    cmd = [
        sys.executable, "-m",
        "observer_worlds.experiments.run_m8_mechanism_discovery",
        "--m7-rules", str(REPO / "release" / "rules" / "m7_top_hce_rules.json"),
        "--n-rules-per-source", str(cfg["n_rules_per_source"]),
        "--test-seeds", *[
            str(s) for s in range(
                int(cfg["test_seeds_range"].split("..")[0]),
                int(cfg["test_seeds_range"].split("..")[1]) + 1,
            )
        ],
        "--timesteps", str(cfg["timesteps"]),
        "--grid", *[str(g) for g in cfg["grid"]],
        "--max-candidates", str(cfg["max_candidates"]),
        "--hce-replicates", str(cfg["hce_replicates"]),
        "--horizons", *[str(h) for h in cfg["horizons"]],
        "--backend", cfg["backend"],
        "--label", f"perf_check_{variant}",
        "--out-root", str(tmp_path),
    ]
    if cfg.get("n_workers") is not None:
        cmd += ["--n-workers", str(cfg["n_workers"])]

    # Generous timeout: 1.5x the baseline's total wall time, min 1 hour.
    base_total = int(baseline["results"].get("wall_time_seconds_total_approx",
                                              baseline["results"]["wall_time_seconds_sweep"]))
    timeout = max(3600, int(base_total * 1.5) + 300)

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, cwd=str(REPO),
    )
    elapsed = time.time() - t0

    # --- Correctness gates (always assertions) --------------------------
    assert result.returncode == 0, (
        f"M7B-class run failed (rc={result.returncode}):\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n\nstderr tail:\n{result.stderr[-2000:]}"
    )
    sweep_seconds, n_candidates = _parse(result.stdout)
    assert sweep_seconds is not None, (
        f"could not parse 'm8 sweep wall time' from stdout. "
        f"Tail:\n{result.stdout[-2000:]}"
    )
    assert n_candidates is not None and n_candidates > 0, (
        f"M7B-class run measured no candidates (n_candidates={n_candidates}). "
        f"Tail:\n{result.stdout[-2000:]}"
    )

    # --- Perf tolerance check ------------------------------------------
    base_sweep = int(baseline["results"]["wall_time_seconds_sweep"])
    tolerance = float(baselines["tolerance_policy"]["warn_regression_fraction"])
    threshold = base_sweep * (1.0 + tolerance)
    regression = (sweep_seconds - base_sweep) / base_sweep

    msg = (
        f"M7B-class baseline check: variant={variant} backend={backend}\n"
        f"  baseline sweep: {base_sweep}s (captured at {baseline['captured_at_utc']})\n"
        f"  observed sweep: {sweep_seconds}s\n"
        f"  delta: {regression:+.1%} (tolerance: +{tolerance:.0%})\n"
        f"  candidates: {n_candidates} (baseline: {baseline['results']['candidates_measured']})\n"
        f"  total wall: {elapsed:.0f}s"
    )
    print("\n" + msg)

    if sweep_seconds > threshold:
        if perf_gate_strict:
            pytest.fail(f"perf regression past tolerance:\n{msg}")
        warnings.warn(f"perf regression past tolerance (warning only — pass --perf-gate to fail):\n{msg}")
