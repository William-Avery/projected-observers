"""Stage G5A / G5B — tests for the discovery parallelism benchmark
and the candidate-discovery profiler.

Coverage (lightweight; no real production runs are launched here):

* CLI ``--help`` works on both new scripts.
* The phase-instrumented worker returns no full state stream in the
  per-cell payload (matches the production payload contract).
* ``CellProfile.per_phase_seconds`` covers all five logical phases.
* Deterministic merge ordering: profiler output rows are sorted by
  ``(rule_source, rule_id, seed)``.
* Benchmark workload builder is deterministic across calls.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI --help smoke tests
# ---------------------------------------------------------------------------


def test_benchmark_discovery_parallelism_help_works():
    rc = subprocess.run(
        [sys.executable, "-m",
         "observer_worlds.perf.benchmark_discovery_parallelism", "--help"],
        cwd=str(REPO),
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    for needle in (
        "--worker-counts", "--backends", "--max-consecutive-failures",
        "--n-rules-per-source", "--seeds", "--projections",
    ):
        assert needle in out, f"missing flag {needle!r} in --help"


def test_profile_candidate_discovery_help_works():
    rc = subprocess.run(
        [sys.executable, "-m",
         "observer_worlds.perf.profile_candidate_discovery", "--help"],
        cwd=str(REPO),
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    for needle in (
        "--n-workers", "--cprofile-top-n", "--projections", "--label",
    ):
        assert needle in out, f"missing flag {needle!r} in --help"


# ---------------------------------------------------------------------------
# Phase-instrumented worker — payload + phase coverage
# ---------------------------------------------------------------------------


def _tiny_cfg():
    return {
        "test_seeds": [7000],
        "timesteps": 50,
        "grid": [16, 16, 4, 4],
        "max_candidates": 3,
        "hce_replicates": 1,
        "horizons": [5],
        "projections": ["mean_threshold"],
        "cpu_discovery_backend": "numpy",
    }


def _tiny_rule_record():
    from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
    rules = load_top_rules(
        REPO / "release" / "rules" / "m7_top_hce_rules.json", 1,
    )
    return {"rule": rules[0], "rule_id": "M7_test_rank01",
            "rule_source": "M7_HCE_optimized"}


def test_phase_instrumented_payload_has_no_state_stream(tmp_path):
    """Same contract as the production worker: the returned profile and
    scaffolds must not contain numpy state arrays."""
    from observer_worlds.perf._phase_instrumented_discover import (
        discover_one_cell_profiled, CellProfile,
    )
    rec = _tiny_rule_record()
    cfg = _tiny_cfg()
    prof, scaffolds = discover_one_cell_profiled(
        rule_record=rec, seed=7000, cfg=cfg, scratch_dir=str(tmp_path),
    )
    assert isinstance(prof, CellProfile)
    # CellProfile fields are scalars / dicts / floats. Check explicitly.
    for v in prof.per_phase_seconds.values():
        assert isinstance(v, float)
    for proj_dict in prof.per_phase_per_projection.values():
        for v in proj_dict.values():
            assert isinstance(v, float)
    # Scaffolds: per-replicate horizon lists are floats (or unset);
    # never numpy arrays.
    for sc in scaffolds.values():
        for r in sc.hce_per_replicate_per_horizon:
            for v in r:
                assert isinstance(v, float)


def test_phase_instrumented_covers_five_phases(tmp_path):
    """All five logical phases must appear in per_phase_seconds and
    have non-negative timing."""
    from observer_worlds.perf._phase_instrumented_discover import (
        discover_one_cell_profiled,
    )
    prof, _ = discover_one_cell_profiled(
        rule_record=_tiny_rule_record(),
        seed=7000, cfg=_tiny_cfg(), scratch_dir=str(tmp_path),
    )
    expected = {
        "substrate_rollout", "projection_stream",
        "candidate_detection", "perturbation_construction",
        "npz_write", "total",
    }
    assert set(prof.per_phase_seconds.keys()) == expected
    for ph, sec in prof.per_phase_seconds.items():
        assert sec >= 0.0, f"phase {ph!r} has negative time"
    # total >= sum of named phases (small overhead allowed).
    named_sum = sum(v for ph, v in prof.per_phase_seconds.items()
                    if ph != "total")
    assert prof.per_phase_seconds["total"] >= named_sum - 0.1


# ---------------------------------------------------------------------------
# Workload builder determinism (G5A)
# ---------------------------------------------------------------------------


def test_benchmark_build_workload_deterministic():
    from observer_worlds.perf.benchmark_discovery_parallelism import (
        _build_workload,
    )
    cfg = {
        "rules_json": str(REPO / "release" / "rules" / "m7_top_hce_rules.json"),
        "m4c_rules": None, "m4a_rules": None,
        "n_rules_per_source": 1,
        "test_seeds": [7000, 7001, 7002],
    }
    a = _build_workload(cfg)
    b = _build_workload(cfg)
    # Same length and same identity tuples (rule_id, seed).
    assert [(rec["rule_id"], seed) for rec, seed in a] == \
           [(rec["rule_id"], seed) for rec, seed in b]
    assert len(a) == 3  # 1 rule x 3 seeds


def test_profiler_summary_csv_sorted(tmp_path):
    """Run the profiler in-process on a tiny workload and verify
    profile_summary.csv is sorted by (rule_source, rule_id, seed)."""
    from observer_worlds.perf import profile_candidate_discovery as g5b
    rc = g5b.main([
        "--rules-from", str(REPO / "release" / "rules" / "m7_top_hce_rules.json"),
        "--n-rules-per-source", "1",
        "--seeds", "7000,7001",
        "--timesteps", "30",
        "--grid", "16", "16", "4", "4",
        "--max-candidates", "2",
        "--hce-replicates", "1",
        "--horizons", "5",
        "--projections", "mean_threshold",
        "--n-workers", "1",
        "--cprofile-top-n", "10",
        "--out-root", str(tmp_path),
        "--label", "g5b_test",
    ])
    assert rc == 0, "profiler returned non-zero"
    out = next(tmp_path.glob("g5b_test_*"))
    csv_path = out / "profile_summary.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    keys = [(r["rule_source"], r["rule_id"], int(r["seed"])) for r in rows]
    assert keys == sorted(keys), \
        f"profile_summary.csv not sorted: {keys}"
    # And the artifacts the brief asks for all exist.
    assert (out / "phase_timing_by_projection.csv").exists()
    assert (out / "phase_timing_by_source.csv").exists()
    assert (out / "profile_summary.md").exists()
    assert (out / "hot_functions.txt").exists()
