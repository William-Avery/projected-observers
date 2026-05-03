"""Lightweight smoke tests for the G4-Linux bootstrap script and its
Python helper.

These tests don't (and can't) actually run G4 — that requires a Linux
GPU host. They only verify that the scripts respond to ``--help``,
that the Python comparator's argparse + index logic is correct on a
synthetic minimal CSV pair, and that the comparator returns the
expected exit code on the bit-identical case.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI --help smoke
# ---------------------------------------------------------------------------


def test_g4_compare_to_stage6c_help_works():
    """Python comparator must respond to --help with rc 0 and a usage
    line that names the required flags."""
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "g4_compare_to_stage6c.py"),
         "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    for needle in ("--g4-run-dir", "--stage6c-baseline-dir",
                   "--rtl-tol", "--out-json", "--out-md"):
        assert needle in out, f"missing flag {needle!r} in --help"


def test_g4_linux_bootstrap_help_works():
    """Bootstrap shell script must respond to --help with rc 0."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available; can't smoke-test the shell script")
    script = REPO / "scripts" / "g4_linux_bootstrap.sh"
    if not script.exists():
        pytest.skip("bootstrap script missing")
    rc = subprocess.run(
        [bash, str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout + rc.stderr
    # The usage block at the top of the script names these flags.
    for needle in (
        "--stage6c-baseline-dir", "--n-workers", "--gpu-batch-size",
        "--skip-smoke", "--skip-full", "--smoke-only",
    ):
        assert needle in out, f"missing flag {needle!r} in --help"


# ---------------------------------------------------------------------------
# Comparator end-to-end on a tiny synthetic pair
# ---------------------------------------------------------------------------


_CSV_HEADER = (
    "rule_id,rule_source,seed,projection,n_candidates,"
    "projection_supports_threshold_margin,projection_output_kind,"
    "candidate_id,track_id,peak_frame,lifetime,valid,"
    "invalid_reason,preservation_strategy,HCE,far_HCE,sham_HCE,"
    "hidden_vs_far_delta,hidden_vs_sham_delta,"
    "initial_projection_delta,far_initial_projection_delta,"
    "n_flipped_hidden,n_flipped_far"
)


def _row(rule_id="M7_test_rank01", seed=7000, projection="mean_threshold",
         cid=0, hce=0.123456, far=0.045678, init=0.0):
    return (
        f"{rule_id},M7_HCE_optimized,{seed},{projection},1,True,binary,"
        f"{cid},10,42,5,True,,count_preserving_swap,{hce},{far},0.0,"
        f"{hce - far},{hce},{init},0.0,2,2"
    )


def _make_run_dir(tmp_path: Path, name: str, rows: list[str]) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "candidate_metrics.csv").write_text(
        _CSV_HEADER + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return d


def test_comparator_rc0_on_bit_identical_inputs(tmp_path):
    """Two CSVs with identical rows -> rc 0, overall_pass=True."""
    rows = [_row(cid=0, hce=0.5, far=0.3),
            _row(cid=1, hce=0.7, far=0.2)]
    bl = _make_run_dir(tmp_path, "baseline", rows)
    g4 = _make_run_dir(tmp_path, "g4", rows)
    rc = subprocess.run(
        [sys.executable,
         str(REPO / "scripts" / "g4_compare_to_stage6c.py"),
         "--g4-run-dir", str(g4),
         "--stage6c-baseline-dir", str(bl),
         "--skip-posthoc"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert '"overall_pass": true' in rc.stdout


def test_comparator_rc2_on_binary_mismatch(tmp_path):
    """Different binary HCE values -> rc 2."""
    rows_bl = [_row(cid=0, hce=0.5)]
    rows_g4 = [_row(cid=0, hce=0.500001)]
    bl = _make_run_dir(tmp_path, "baseline", rows_bl)
    g4 = _make_run_dir(tmp_path, "g4", rows_g4)
    rc = subprocess.run(
        [sys.executable,
         str(REPO / "scripts" / "g4_compare_to_stage6c.py"),
         "--g4-run-dir", str(g4),
         "--stage6c-baseline-dir", str(bl),
         "--skip-posthoc"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 2, rc.stdout + rc.stderr


def test_comparator_continuous_within_tol_passes(tmp_path):
    """random_linear_projection with delta below --rtl-tol -> rc 0."""
    rows_bl = [_row(projection="random_linear_projection", hce=2.0, far=1.0)]
    rows_g4 = [_row(projection="random_linear_projection",
                    hce=2.0 + 5e-7, far=1.0 + 5e-7)]
    bl = _make_run_dir(tmp_path, "baseline", rows_bl)
    g4 = _make_run_dir(tmp_path, "g4", rows_g4)
    rc = subprocess.run(
        [sys.executable,
         str(REPO / "scripts" / "g4_compare_to_stage6c.py"),
         "--g4-run-dir", str(g4),
         "--stage6c-baseline-dir", str(bl),
         "--rtl-tol", "1e-6",
         "--skip-posthoc"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr


def test_comparator_continuous_above_tol_fails(tmp_path):
    """random_linear_projection with delta above --rtl-tol -> rc 2."""
    rows_bl = [_row(projection="random_linear_projection", hce=2.0, far=1.0)]
    rows_g4 = [_row(projection="random_linear_projection",
                    hce=2.0 + 1e-3, far=1.0)]
    bl = _make_run_dir(tmp_path, "baseline", rows_bl)
    g4 = _make_run_dir(tmp_path, "g4", rows_g4)
    rc = subprocess.run(
        [sys.executable,
         str(REPO / "scripts" / "g4_compare_to_stage6c.py"),
         "--g4-run-dir", str(g4),
         "--stage6c-baseline-dir", str(bl),
         "--rtl-tol", "1e-6",
         "--skip-posthoc"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 2, rc.stdout + rc.stderr


def test_comparator_emits_json_and_md(tmp_path):
    rows = [_row(cid=0)]
    bl = _make_run_dir(tmp_path, "baseline", rows)
    g4 = _make_run_dir(tmp_path, "g4", rows)
    j = tmp_path / "out.json"
    m = tmp_path / "out.md"
    rc = subprocess.run(
        [sys.executable,
         str(REPO / "scripts" / "g4_compare_to_stage6c.py"),
         "--g4-run-dir", str(g4),
         "--stage6c-baseline-dir", str(bl),
         "--out-json", str(j), "--out-md", str(m),
         "--skip-posthoc"],
        capture_output=True, text=True, timeout=30,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert j.exists() and j.stat().st_size > 0
    assert m.exists() and m.stat().st_size > 0
    md = m.read_text(encoding="utf-8")
    assert "G4 vs Stage 6C equivalence report" in md
    assert "row-by-row" in md.lower()
