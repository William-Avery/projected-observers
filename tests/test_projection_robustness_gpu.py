"""Stage G2 — tests for the GPU-aware projection-robustness runner.

Skip cleanly when cupy/CUDA is unavailable.

Coverage:

1. CLI ``--help`` works.
2. GPU runner skips gracefully when cupy is missing
   (``--backend cupy`` returns rc != 0 with a clear error).
3. Tiny deterministic CPU/GPU rollout-batch HCE equivalence
   (``measure_batch_on_gpu`` against the numpy backend).
4. Output schema matches the CPU runner — same artifact files.
5. ``stats_summary.json`` includes the GPU metadata block.
6. No CPU/GPU copy occurs inside the timestep loop (audited via the
   ``state_transfer_policy`` field on the GPU metadata block).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from observer_worlds.backends import get_backend, is_cupy_available
from observer_worlds.experiments._followup_projection_gpu import (
    measure_batch_on_gpu,
)
from observer_worlds.experiments import (
    run_followup_projection_robustness_gpu as gpu_runner,
)


HAS_CUPY = is_cupy_available()
needs_cupy = pytest.mark.skipif(
    not HAS_CUPY, reason="cupy + CUDA device not available; GPU tests skipped"
)


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. CLI help
# ---------------------------------------------------------------------------


def test_cli_help_works():
    rc = subprocess.run(
        [sys.executable, "-m",
         "observer_worlds.experiments.run_followup_projection_robustness_gpu",
         "--help"],
        cwd=str(REPO),
        capture_output=True, text=True, timeout=60,
    )
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    for needle in [
        "--backend", "--gpu-batch-size", "--gpu-memory-target-gb",
        "--gpu-device", "--equivalence-audit", "--projections",
    ]:
        assert needle in out, f"missing flag {needle!r} in --help output"


# ---------------------------------------------------------------------------
# 2. Cupy-unavailable path
# ---------------------------------------------------------------------------


def test_cupy_unavailable_returns_nonzero_with_clear_message():
    """When cupy is missing, ``--backend cupy`` must exit non-zero with
    a clear message, not crash with a stack trace."""
    if HAS_CUPY:
        pytest.skip("cupy is installed; this guard only fires when it isn't")
    rc = subprocess.run(
        [sys.executable, "-m",
         "observer_worlds.experiments.run_followup_projection_robustness_gpu",
         "--backend", "cupy", "--n-rules-per-source", "1",
         "--seeds", "7000", "--timesteps", "10", "--horizons", "1",
         "--projections", "mean_threshold"],
        cwd=str(REPO),
        capture_output=True, text=True, timeout=60,
    )
    assert rc.returncode != 0
    assert "cupy" in (rc.stderr + rc.stdout).lower()


# ---------------------------------------------------------------------------
# 3. Tiny deterministic CPU/GPU equivalence on the GPU rollout primitive
# ---------------------------------------------------------------------------


def _tiny_batch():
    """Tiny batch fixture: B=4 grids of shape (8, 8, 4, 4)."""
    rng = np.random.default_rng(0)
    B, Nx, Ny, Nz, Nw = 4, 8, 8, 4, 4
    states_o = (rng.random((B, Nx, Ny, Nz, Nw)) < 0.3).astype(np.uint8)
    # Build hidden / far perturbations that preserve mean_threshold:
    # toggle one pair of (ON, OFF) cells in a few fibres.
    states_h = states_o.copy()
    states_f = states_o.copy()
    # Flip first cell's first fibre's first ON+OFF for hidden;
    # last cell's last fibre's first ON+OFF for far. Trivial swap; the
    # exact pattern doesn't matter for equivalence.
    for b in range(B):
        # hidden: swap in candidate region (top-left quadrant)
        fib = states_h[b, 0, 0]
        ons = np.argwhere(fib == 1)
        offs = np.argwhere(fib == 0)
        if ons.size and offs.size:
            zi, wi = ons[0]
            zj, wj = offs[0]
            states_h[b, 0, 0, zi, wi] = 0
            states_h[b, 0, 0, zj, wj] = 1
        # far: swap in far region (bottom-right quadrant)
        fib2 = states_f[b, Nx - 1, Ny - 1]
        ons = np.argwhere(fib2 == 1)
        offs = np.argwhere(fib2 == 0)
        if ons.size and offs.size:
            zi, wi = ons[0]
            zj, wj = offs[0]
            states_f[b, Nx - 1, Ny - 1, zi, wi] = 0
            states_f[b, Nx - 1, Ny - 1, zj, wj] = 1

    # Conway-style rule LUTs.
    from observer_worlds.worlds.rules import BSRule
    bl, sl = BSRule(birth=(3,), survival=(2, 3)).to_lookup_tables(80)
    bl = np.broadcast_to(bl.astype(np.uint8), (B, 81)).copy()
    sl = np.broadcast_to(sl.astype(np.uint8), (B, 81)).copy()

    masks = np.zeros((B, Nx, Ny), dtype=np.uint8)
    masks[:, 0, 0] = 1
    masks[:, 0, 1] = 1
    masks[:, 1, 0] = 1
    avail = np.array([10, 10, 10, 10], dtype=np.int32)
    return dict(
        states_orig=states_o, states_hidden=states_h, states_far=states_f,
        birth_luts=bl, surv_luts=sl,
        candidate_local_masks=masks, avail_steps=avail,
    )


@needs_cupy
def test_measure_batch_on_gpu_binary_projection_equivalence():
    """For mean_threshold (binary), CPU vs GPU rollout HCE must be
    bit-identical."""
    fix = _tiny_batch()
    horizons = [1, 2, 5]
    cpu = get_backend("numpy")
    gpu = get_backend("cupy")
    cpu_h, cpu_f = measure_batch_on_gpu(
        backend=cpu, **fix, horizons=horizons, projection="mean_threshold",
    )
    gpu_h, gpu_f = measure_batch_on_gpu(
        backend=gpu, **fix, horizons=horizons, projection="mean_threshold",
    )
    np.testing.assert_array_equal(cpu_h, gpu_h)
    np.testing.assert_array_equal(cpu_f, gpu_f)


@needs_cupy
def test_measure_batch_on_gpu_continuous_projection_within_tolerance():
    fix = _tiny_batch()
    horizons = [1, 3]
    cpu = get_backend("numpy")
    gpu = get_backend("cupy")
    cpu_h, cpu_f = measure_batch_on_gpu(
        backend=cpu, **fix, horizons=horizons,
        projection="random_linear_projection",
        projection_params={"seed": 0},
    )
    gpu_h, gpu_f = measure_batch_on_gpu(
        backend=gpu, **fix, horizons=horizons,
        projection="random_linear_projection",
        projection_params={"seed": 0},
    )
    np.testing.assert_allclose(cpu_h, gpu_h, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(cpu_f, gpu_f, rtol=1e-4, atol=1e-3)


@needs_cupy
def test_measure_batch_on_gpu_avail_steps_masks_long_horizons():
    """Horizons that exceed avail_steps[b] must come back as NaN."""
    fix = _tiny_batch()
    fix["avail_steps"] = np.array([2, 3, 4, 5], dtype=np.int32)
    horizons = [1, 3, 5, 7]
    gpu = get_backend("cupy") if HAS_CUPY else get_backend("numpy")
    h, f = measure_batch_on_gpu(
        backend=gpu, **fix, horizons=horizons, projection="mean_threshold",
    )
    # row 0 (avail=2): only h=1 valid; h=3,5,7 NaN
    assert not np.isnan(h[0, 0])
    assert np.isnan(h[0, 1]) and np.isnan(h[0, 2]) and np.isnan(h[0, 3])
    # row 3 (avail=5): h=1,3,5 valid; h=7 NaN
    assert not np.isnan(h[3, 0]) and not np.isnan(h[3, 1])
    assert not np.isnan(h[3, 2])
    assert np.isnan(h[3, 3])


# ---------------------------------------------------------------------------
# 4-6. End-to-end smoke (only when cupy available)
# ---------------------------------------------------------------------------


REQUIRED_OUTPUTS = {
    "config.json", "frozen_manifest.json", "candidate_metrics.csv",
    "projection_summary.csv", "hce_by_projection.csv",
    "projection_artifact_audit.csv", "mechanism_by_projection.csv",
    "stats_summary.json", "summary.md", "perf_profile.json",
}


@needs_cupy
def test_gpu_runner_smoke_writes_full_artifact_bundle(tmp_path):
    """Run a tiny GPU pipeline and assert every required artifact
    exists and the GPU metadata block is present in stats_summary.json.
    """
    rc = gpu_runner.main([
        "--rules-from", str(REPO / "release" / "rules" / "m7_top_hce_rules.json"),
        "--n-rules-per-source", "1",
        "--seeds", "7000",
        "--timesteps", "60",
        "--grid", "16", "16", "4", "4",
        "--max-candidates", "3",
        "--hce-replicates", "1",
        "--horizons", "5", "10",
        "--projections", "mean_threshold", "parity_projection",
        "--backend", "cupy",
        "--gpu-batch-size", "16",
        "--gpu-memory-target-gb", "9.5",
        "--out-root", str(tmp_path),
        "--label", "g2_smoke",
        "--profile",
    ])
    assert rc == 0, "GPU runner failed"
    out_dirs = list(tmp_path.glob("g2_smoke_*"))
    assert len(out_dirs) == 1, f"expected one run dir, got {out_dirs!r}"
    out = out_dirs[0]
    have = {p.name for p in out.iterdir()
            if p.is_file() and p.name in REQUIRED_OUTPUTS}
    missing = REQUIRED_OUTPUTS - have
    assert not missing, (
        f"GPU runner missing required outputs: {sorted(missing)!r}; "
        f"present: {sorted(p.name for p in out.iterdir())!r}"
    )

    stats = json.loads((out / "stats_summary.json").read_text(encoding="utf-8"))
    assert "gpu_metadata" in stats
    meta = stats["gpu_metadata"]
    for key in ("gpu_backend_used", "gpu_batch_size",
                "gpu_memory_target_gb", "cpu_gpu_equivalence_mode",
                "state_transfer_policy"):
        assert key in meta, f"missing GPU metadata field {key!r}"
    assert meta["gpu_backend_used"] == "cupy"
    # No-copy contract recorded:
    policy = meta["state_transfer_policy"]
    assert "no copies inside timestep loop" in policy.lower()


@needs_cupy
def test_gpu_runner_csv_schema_matches_cpu(tmp_path):
    """The candidate_metrics.csv from the GPU runner must have the
    same column set as the CPU runner."""
    # Run a tiny GPU run.
    rc = gpu_runner.main([
        "--rules-from", str(REPO / "release" / "rules" / "m7_top_hce_rules.json"),
        "--n-rules-per-source", "1",
        "--seeds", "7000",
        "--timesteps", "60",
        "--grid", "16", "16", "4", "4",
        "--max-candidates", "3",
        "--hce-replicates", "1",
        "--horizons", "5",
        "--projections", "mean_threshold",
        "--backend", "cupy",
        "--gpu-batch-size", "16",
        "--out-root", str(tmp_path),
        "--label", "g2_schema",
    ])
    assert rc == 0
    out = next(tmp_path.glob("g2_schema_*"))
    gpu_csv = (out / "candidate_metrics.csv").read_text(encoding="utf-8")
    gpu_cols = gpu_csv.splitlines()[0].split(",")
    expected = {
        "rule_id", "rule_source", "seed", "projection", "candidate_id",
        "track_id", "peak_frame", "lifetime", "valid", "invalid_reason",
        "preservation_strategy", "HCE", "far_HCE", "sham_HCE",
        "hidden_vs_far_delta", "hidden_vs_sham_delta",
        "initial_projection_delta", "far_initial_projection_delta",
        "n_flipped_hidden", "n_flipped_far",
    }
    missing = expected - set(gpu_cols)
    assert not missing, f"GPU CSV missing CPU columns: {sorted(missing)}"
