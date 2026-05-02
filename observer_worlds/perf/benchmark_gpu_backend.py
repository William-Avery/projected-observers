"""Stage G1C — GPU backend benchmark harness.

Runs the new ``observer_worlds.backends`` primitives at varying batch
sizes on numpy and (if available) cupy. Measures wall time, throughput
(timesteps/sec, states/sec), the projection pass, and the full HCE
rollout. Writes a CSV + Markdown summary under
``outputs/gpu_benchmark_<timestamp>/``.

This benchmark is **not** a production run; it does not write any
scientific artifact and does not invoke any production runner. Its
sole job is to characterize the GPU primitive throughput and confirm
the no-copy contract on the timestep loop.

Example:

    python -m observer_worlds.perf.benchmark_gpu_backend \\
        --grid 64 64 8 8 --batch-sizes 16 32 64 128 \\
        --timesteps 100 --backend cupy --gpu-memory-target-gb 9.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from observer_worlds.backends import (
    estimate_max_safe_batch_size,
    get_backend,
    is_cupy_available,
)
from observer_worlds.worlds.rules import BSRule


# Stage-5/6 standard rule for the benchmark (Conway-like 4D variant).
# The exact rule does not affect the throughput measurement.
_BENCHMARK_RULE = BSRule(birth=(3,), survival=(2, 3))


def _make_initial_batch(
    *, B: int, grid: Sequence[int], density: float = 0.2, seed: int = 12345,
) -> np.ndarray:
    Nx, Ny, Nz, Nw = (int(g) for g in grid)
    rng = np.random.default_rng(int(seed))
    return (rng.random((B, Nx, Ny, Nz, Nw)) < float(density)).astype(np.uint8)


def _make_luts_for_rule(B: int, rule: BSRule) -> tuple[np.ndarray, np.ndarray]:
    bl, sl = rule.to_lookup_tables(80)
    bl = bl.astype(np.uint8)
    sl = sl.astype(np.uint8)
    return (
        np.broadcast_to(bl, (B, 81)).copy(),
        np.broadcast_to(sl, (B, 81)).copy(),
    )


def _make_candidate_masks(B: int, grid: Sequence[int], *, frac: float = 0.05) -> np.ndarray:
    Nx, Ny = int(grid[0]), int(grid[1])
    rng = np.random.default_rng(54321)
    return (rng.random((B, Nx, Ny)) < float(frac)).astype(np.uint8)


def _gpu_mem_used_mb() -> float | None:
    if not is_cupy_available():
        return None
    import cupy as cp
    free, total = cp.cuda.Device(0).mem_info
    return (total - free) / (1024 ** 2)


def _bench_one(
    *,
    backend_name: str,
    backend,
    B: int,
    grid: Sequence[int],
    timesteps: int,
    horizons: Sequence[int],
    projection: str,
) -> dict:
    """Benchmark one (backend, B) cell. Returns a result dict."""
    Nx, Ny, Nz, Nw = (int(g) for g in grid)
    states_h = _make_initial_batch(B=B, grid=grid)
    bl_h, sl_h = _make_luts_for_rule(B, _BENCHMARK_RULE)
    masks_h = _make_candidate_masks(B, grid)

    t0 = time.perf_counter()
    states_d = backend.asarray(states_h)
    bl_d = backend.asarray(bl_h)
    sl_d = backend.asarray(sl_h)
    if backend.is_gpu:
        backend.xp.cuda.Device().synchronize()
    t_xfer_in = time.perf_counter() - t0

    # ---- step-only loop (no projection) ----
    t0 = time.perf_counter()
    cur = states_d
    for _ in range(timesteps):
        cur = backend.step_4d_batch(cur, bl_d, sl_d)
    if backend.is_gpu:
        backend.xp.cuda.Device().synchronize()
    t_step_loop = time.perf_counter() - t0

    # ---- single projection pass ----
    t0 = time.perf_counter()
    _ = backend.project_batch(cur, method=projection)
    if backend.is_gpu:
        backend.xp.cuda.Device().synchronize()
    t_project = time.perf_counter() - t0

    # ---- hidden-perturbation pass ----
    t0 = time.perf_counter()
    pert = backend.apply_hidden_perturbations_batch(states_d, masks_h)
    if backend.is_gpu:
        backend.xp.cuda.Device().synchronize()
    t_perturb = time.perf_counter() - t0

    # ---- full HCE rollout (the rollout-heavy target) ----
    masks_for_metric = masks_h if not backend.is_gpu else backend.asarray(masks_h)
    t0 = time.perf_counter()
    metrics = backend.rollout_hce_batch(
        states_d, pert, bl_d, sl_d, masks_for_metric,
        horizons=horizons, projection=projection,
    )
    if backend.is_gpu:
        backend.xp.cuda.Device().synchronize()
    t_hce = time.perf_counter() - t0

    # ---- transfer compact metric back ----
    t0 = time.perf_counter()
    metrics_h = backend.asnumpy(metrics)
    if backend.is_gpu:
        backend.xp.cuda.Device().synchronize()
    t_xfer_out = time.perf_counter() - t0

    states_per_sec = (B * timesteps) / max(t_step_loop, 1e-9)
    timesteps_per_sec = timesteps / max(t_step_loop, 1e-9)

    return {
        "backend": backend_name,
        "batch_size": B,
        "grid": "x".join(str(g) for g in grid),
        "timesteps": timesteps,
        "projection": projection,
        "horizons": ",".join(str(h) for h in horizons),
        "t_xfer_in_sec": round(t_xfer_in, 6),
        "t_step_loop_sec": round(t_step_loop, 6),
        "t_project_sec": round(t_project, 6),
        "t_perturb_sec": round(t_perturb, 6),
        "t_rollout_hce_sec": round(t_hce, 6),
        "t_xfer_out_sec": round(t_xfer_out, 6),
        "timesteps_per_sec": round(timesteps_per_sec, 2),
        "states_per_sec": round(states_per_sec, 2),
        "metric_shape": "x".join(str(d) for d in metrics_h.shape),
        "gpu_mem_used_mb": (
            round(_gpu_mem_used_mb(), 1) if backend.is_gpu else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", nargs=4, type=int, default=[64, 64, 8, 8])
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default=[16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 5, 20, 80],
    )
    parser.add_argument(
        "--projection", default="mean_threshold",
        choices=[
            "mean_threshold", "sum_threshold", "max_projection",
            "parity_projection", "random_linear_projection",
            "multi_channel_projection",
        ],
    )
    parser.add_argument(
        "--backend", default="both",
        choices=["numpy", "cupy", "both"],
    )
    parser.add_argument(
        "--gpu-batch-size", type=int, default=None,
        help="If set, override --batch-sizes with this single value for cupy.",
    )
    parser.add_argument(
        "--gpu-memory-target-gb", type=float, default=9.5,
        help="Target VRAM cap for the planner (RTX 3080 Ti is 12 GB).",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--gpu-smoke-only", action="store_true",
        help="Run only the smallest batch size on the cupy path.",
    )
    parser.add_argument(
        "--label", default="gpu_benchmark",
        help="Prefix for the output dir (timestamp appended).",
    )
    args = parser.parse_args(argv)

    cupy_ok = is_cupy_available()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("outputs") / f"{args.label}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "grid": list(args.grid),
        "batch_sizes": list(args.batch_sizes),
        "timesteps": args.timesteps,
        "horizons": list(args.horizons),
        "projection": args.projection,
        "backend_arg": args.backend,
        "gpu_batch_size": args.gpu_batch_size,
        "gpu_memory_target_gb": args.gpu_memory_target_gb,
        "gpu_device": args.gpu_device,
        "gpu_smoke_only": args.gpu_smoke_only,
        "cupy_available": cupy_ok,
        "label": args.label,
        "timestamp_utc": timestamp,
    }

    if cupy_ok:
        import cupy as cp
        gpu_props = cp.cuda.runtime.getDeviceProperties(args.gpu_device)
        config["gpu_name"] = gpu_props["name"].decode("utf-8", errors="replace")
        free, total = cp.cuda.Device(args.gpu_device).mem_info
        config["gpu_total_mem_mb"] = round(total / (1024 ** 2), 1)
        config["gpu_free_mem_mb_at_start"] = round(free / (1024 ** 2), 1)
        config["planner_max_safe_batch"] = estimate_max_safe_batch_size(
            grid=args.grid,
            n_perturbation_conditions=2,
            n_projection_frames_per_state=len(args.horizons),
            target_gb=args.gpu_memory_target_gb,
        )

    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    backends_to_run: list[str] = []
    if args.backend in ("numpy", "both"):
        backends_to_run.append("numpy")
    if args.backend in ("cupy", "both"):
        if cupy_ok:
            backends_to_run.append("cupy")
        else:
            print("[warn] cupy not available; skipping cupy benchmark.")

    if not backends_to_run:
        print("[error] no benchmarkable backend selected.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for backend_name in backends_to_run:
        backend = get_backend(backend_name, device=args.gpu_device)
        sizes = list(args.batch_sizes)
        if backend_name == "cupy":
            if args.gpu_batch_size is not None:
                sizes = [int(args.gpu_batch_size)]
            if args.gpu_smoke_only:
                sizes = [min(sizes)]
        for B in sizes:
            print(f"[bench] backend={backend_name} B={B} ...", flush=True)
            try:
                row = _bench_one(
                    backend_name=backend_name,
                    backend=backend,
                    B=B,
                    grid=args.grid,
                    timesteps=args.timesteps,
                    horizons=args.horizons,
                    projection=args.projection,
                )
            except Exception as e:  # pragma: no cover
                print(f"  [error] B={B}: {e}", flush=True)
                row = {
                    "backend": backend_name, "batch_size": B,
                    "grid": "x".join(str(g) for g in args.grid),
                    "timesteps": args.timesteps,
                    "projection": args.projection,
                    "horizons": ",".join(str(h) for h in args.horizons),
                    "t_xfer_in_sec": None, "t_step_loop_sec": None,
                    "t_project_sec": None, "t_perturb_sec": None,
                    "t_rollout_hce_sec": None, "t_xfer_out_sec": None,
                    "timesteps_per_sec": None, "states_per_sec": None,
                    "metric_shape": None, "gpu_mem_used_mb": None,
                    "error": str(e),
                }
            rows.append(row)
            print(
                f"  step_loop={row.get('t_step_loop_sec')} s "
                f"hce={row.get('t_rollout_hce_sec')} s "
                f"states/sec={row.get('states_per_sec')}",
                flush=True,
            )

    # Write CSV.
    csv_path = out_dir / "benchmark_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # Build summary.md.
    by_size: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_size.setdefault(r["batch_size"], {})[r["backend"]] = r

    lines: list[str] = []
    lines.append("# GPU backend benchmark summary")
    lines.append("")
    lines.append(f"* timestamp: `{timestamp}`")
    lines.append(f"* grid: `{config['grid']}`")
    lines.append(f"* timesteps: `{config['timesteps']}`")
    lines.append(f"* horizons: `{config['horizons']}`")
    lines.append(f"* projection: `{config['projection']}`")
    lines.append(f"* cupy available: `{cupy_ok}`")
    if cupy_ok:
        lines.append(f"* GPU: `{config['gpu_name']}`")
        lines.append(
            f"* GPU total mem: `{config['gpu_total_mem_mb']:.0f} MB`; "
            f"target `{config['gpu_memory_target_gb']} GB`"
        )
        lines.append(
            f"* planner max safe batch: `{config['planner_max_safe_batch']}`"
        )
    lines.append("")

    lines.append("## Per-batch-size step-loop and HCE rollout")
    lines.append("")
    lines.append(
        "| batch | backend | step_loop (s) | states/sec | "
        "rollout_hce (s) | gpu_mem (MB) |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|")
    for B in sorted(by_size):
        for backend_name in ("numpy", "cupy"):
            if backend_name not in by_size[B]:
                continue
            r = by_size[B][backend_name]
            lines.append(
                f"| {B} | {backend_name} | "
                f"{r.get('t_step_loop_sec')} | {r.get('states_per_sec')} | "
                f"{r.get('t_rollout_hce_sec')} | {r.get('gpu_mem_used_mb')} |"
            )
    lines.append("")

    if "numpy" in backends_to_run and "cupy" in backends_to_run:
        lines.append("## Speedup vs numpy")
        lines.append("")
        lines.append(
            "| batch | step_loop x | rollout_hce x |"
        )
        lines.append("|---:|---:|---:|")
        for B in sorted(by_size):
            n = by_size[B].get("numpy")
            c = by_size[B].get("cupy")
            if not (n and c):
                continue
            try:
                step_x = round(
                    float(n["t_step_loop_sec"]) / float(c["t_step_loop_sec"]), 2,
                )
            except Exception:
                step_x = None
            try:
                hce_x = round(
                    float(n["t_rollout_hce_sec"]) / float(c["t_rollout_hce_sec"]), 2,
                )
            except Exception:
                hce_x = None
            lines.append(f"| {B} | {step_x} | {hce_x} |")
        lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone. Output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
