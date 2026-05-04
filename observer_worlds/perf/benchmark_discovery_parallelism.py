"""Stage G5A — Windows discovery parallelism benchmark.

Sweeps worker count x process backend over a representative
candidate-discovery workload and records:

* per-trial wall time
* completed cells / failed cells
* pool deaths (BrokenProcessPool / TerminatedWorkerError) detected
* per-worker crash log: which (rule_id, seed, projection) the worker
  was inside when it died (best-effort via per-worker JSONL files)
* candidate rows produced
* scratch disk written per trial

Backends compared (Windows-relevant):

* ``concurrent.futures.ProcessPoolExecutor`` — what G3 currently uses
* ``joblib.Parallel(backend="loky")`` — what Stage 6 production uses
* ``multiprocessing.Pool`` (``"spawn"`` context) — Python stdlib

Worker count grid: 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30. Adaptive
short-circuit: if a backend fails three trials in a row, the remaining
worker counts for that backend are skipped (they would also fail and
just waste minutes per trial).

This script is **read-only** with respect to scientific code paths.
It calls the existing
:func:`observer_worlds.experiments._parallel_discovery.discover_one_cell`
without modification; what we measure is just the framework around it.

Outputs:

    outputs/discovery_parallelism_benchmark_<timestamp>/
        config.json
        results.csv
        summary.md
        failure_log.jsonl
        per_trial/<backend>__n<workers>__<ts>/   (workers' crash logs)
"""
from __future__ import annotations

import os as _os

# Pin BLAS thread counts BEFORE importing numpy, joblib, etc. — workers
# inherit on Windows spawn. See _parallel_discovery for rationale.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, "1")

import argparse
import concurrent.futures as _cf
import csv
import faulthandler
import json
import multiprocessing as _mp
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from observer_worlds.experiments.run_followup_projection_robustness import (
    _load_rules,
    _parse_seeds_arg,
)


REPO = Path(__file__).resolve().parents[2]


# Backends and worker grid spec'd by the G5A task brief.
DEFAULT_BACKENDS = ("ProcessPoolExecutor", "joblib_loky", "mp_pool_spawn")
DEFAULT_WORKER_GRID = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Worker function (runs inside each subprocess)
# ---------------------------------------------------------------------------


def _worker_run_one_cell(
    *,
    rule_record: dict,
    seed: int,
    cfg: dict,
    scratch_dir: str,
    crash_log_dir: str,
    cell_id: int,
):
    """Run discover_one_cell with crash-log breadcrumbs.

    The breadcrumbs are written to a per-worker JSONL file as the cell
    proceeds; if the worker dies via Windows access violation, the file
    captures the last entered phase before death.
    """
    faulthandler.enable()
    pid = _os.getpid()
    crash_path = Path(crash_log_dir) / f"worker_{pid}.jsonl"
    crash_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(event: str, **kwargs):
        rec = {
            "ts": _now_iso(), "pid": pid, "cell_id": cell_id,
            "rule_id": rule_record["rule_id"],
            "rule_source": rule_record["rule_source"],
            "seed": int(seed),
            "event": event,
            **kwargs,
        }
        with crash_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            try:
                _os.fsync(f.fileno())
            except OSError:
                pass

    _log("cell_start", phase="discover_one_cell")
    try:
        # Import lazily inside the worker so module init cost is per-process.
        from observer_worlds.experiments._parallel_discovery import (
            discover_one_cell,
        )
        t0 = time.perf_counter()
        result = discover_one_cell(
            rule_record=rule_record, seed=int(seed),
            cfg=cfg, scratch_dir=scratch_dir,
        )
        dt = time.perf_counter() - t0
        _log(
            "cell_done", wall_s=round(dt, 3),
            n_scaffolds=len(result.scaffolds),
            payload_mb=result.payload_mb_estimate,
        )
        return {
            "cell_id": cell_id,
            "rule_id": result.rule_id,
            "rule_source": result.rule_source,
            "seed": result.seed,
            "wall_s": dt,
            "n_scaffolds": len(result.scaffolds),
            "payload_mb": result.payload_mb_estimate,
            "ok": True,
        }
    except Exception as e:  # noqa: BLE001
        _log("cell_exception", error_type=type(e).__name__, error=str(e),
             traceback=traceback.format_exc())
        return {
            "cell_id": cell_id,
            "rule_id": rule_record["rule_id"],
            "rule_source": rule_record["rule_source"],
            "seed": int(seed),
            "wall_s": None,
            "n_scaffolds": 0,
            "payload_mb": 0.0,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# Per-backend trial runners
# ---------------------------------------------------------------------------


def _trial_pe(workers: int, work, cfg, scratch_dir, crash_log_dir):
    pool_deaths = 0
    completed: list[dict] = []
    failed: list[tuple[int, str]] = []
    pending = list(range(len(work)))
    # In a benchmark we do NOT retry on pool death. We measure crashes.
    try:
        with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_idx = {
                ex.submit(
                    _worker_run_one_cell,
                    rule_record=work[i][0], seed=int(work[i][1]),
                    cfg=cfg, scratch_dir=scratch_dir,
                    crash_log_dir=crash_log_dir, cell_id=i,
                ): i
                for i in pending
            }
            try:
                for fut in _cf.as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    try:
                        res = fut.result()
                        if res.get("ok"):
                            completed.append(res)
                        else:
                            failed.append((i, res.get("error", "non-ok")))
                    except (_cf.process.BrokenProcessPool,
                            _cf.CancelledError) as e:
                        pool_deaths += 1
                        failed.append((i, f"pool_death:{type(e).__name__}"))
                        break  # the pool is dead; remaining futures will raise
                    except Exception as e:  # noqa: BLE001
                        failed.append((i, f"{type(e).__name__}:{e}"))
            except _cf.process.BrokenProcessPool as e:
                pool_deaths += 1
                failed.extend([(i, f"pool_death:{type(e).__name__}")
                               for i in pending if i not in
                               {x["cell_id"] for x in completed}])
    except _cf.process.BrokenProcessPool as e:
        pool_deaths += 1
        failed.extend([(i, f"pool_death:{type(e).__name__}")
                       for i in pending if i not in
                       {x["cell_id"] for x in completed}])
    return completed, failed, pool_deaths


def _trial_joblib(workers: int, work, cfg, scratch_dir, crash_log_dir):
    pool_deaths = 0
    completed: list[dict] = []
    failed: list[tuple[int, str]] = []
    try:
        from joblib import Parallel, delayed
        sub = Parallel(
            n_jobs=workers, backend="loky", verbose=0,
        )(
            delayed(_worker_run_one_cell)(
                rule_record=work[i][0], seed=int(work[i][1]),
                cfg=cfg, scratch_dir=scratch_dir,
                crash_log_dir=crash_log_dir, cell_id=i,
            )
            for i in range(len(work))
        )
        for res in sub:
            if res.get("ok"):
                completed.append(res)
            else:
                failed.append((res["cell_id"], res.get("error", "non-ok")))
    except Exception as e:  # noqa: BLE001  - includes TerminatedWorkerError
        pool_deaths += 1
        # Whichever cells didn't make it back are unknown failures; we
        # only record the count (joblib doesn't tell us which).
        unfinished = len(work) - len(completed)
        for k in range(unfinished):
            failed.append((-1, f"pool_death:{type(e).__name__}"))
    return completed, failed, pool_deaths


def _trial_mp_pool_spawn(workers: int, work, cfg, scratch_dir, crash_log_dir):
    pool_deaths = 0
    completed: list[dict] = []
    failed: list[tuple[int, str]] = []
    try:
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            args = [
                (work[i][0], int(work[i][1]), cfg,
                 scratch_dir, crash_log_dir, i)
                for i in range(len(work))
            ]
            try:
                results = pool.starmap(_mp_worker_starmap_adapter, args)
                for res in results:
                    if res.get("ok"):
                        completed.append(res)
                    else:
                        failed.append((res["cell_id"], res.get("error", "non-ok")))
            except Exception as e:  # noqa: BLE001
                pool_deaths += 1
                unfinished = len(work) - len(completed)
                for k in range(unfinished):
                    failed.append((-1, f"pool_death:{type(e).__name__}"))
    except Exception as e:  # noqa: BLE001
        pool_deaths += 1
        unfinished = len(work) - len(completed)
        for k in range(unfinished):
            failed.append((-1, f"pool_death:{type(e).__name__}"))
    return completed, failed, pool_deaths


def _mp_worker_starmap_adapter(rule_record, seed, cfg,
                                scratch_dir, crash_log_dir, cell_id):
    """Module-level adapter so multiprocessing.Pool.starmap can pickle it."""
    return _worker_run_one_cell(
        rule_record=rule_record, seed=seed, cfg=cfg,
        scratch_dir=scratch_dir, crash_log_dir=crash_log_dir, cell_id=cell_id,
    )


_BACKENDS = {
    "ProcessPoolExecutor": _trial_pe,
    "joblib_loky": _trial_joblib,
    "mp_pool_spawn": _trial_mp_pool_spawn,
}


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _build_workload(cfg: dict) -> list[tuple]:
    """Load the rule records and emit the (rule_record, seed) work list."""
    work: list[tuple[dict, int]] = []
    for path, source_label in [
        (Path(cfg["rules_json"]), "M7_HCE_optimized"),
        (Path(cfg["m4c_rules"]) if cfg.get("m4c_rules") else None,
         "M4C_observer_optimized"),
        (Path(cfg["m4a_rules"]) if cfg.get("m4a_rules") else None,
         "M4A_viability"),
    ]:
        if path is None:
            continue
        loaded = _load_rules(path, cfg["n_rules_per_source"])
        for i, r in enumerate(loaded):
            rid = f"{source_label}_rank{i+1:02d}"
            rec = {"rule": r, "rule_id": rid, "rule_source": source_label}
            for s in cfg["test_seeds"]:
                work.append((rec, int(s)))
    return work


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rules-from", "--rules-json", dest="rules_json",
                   type=Path,
                   default=REPO / "release" / "rules" / "m7_top_hce_rules.json")
    p.add_argument("--m4c-rules", type=Path, default=None)
    p.add_argument("--m4a-rules", type=Path, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=1)
    p.add_argument("--seeds", type=str, default="7000..7004")
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--projections", nargs="+",
                   default=["mean_threshold", "sum_threshold",
                            "max_projection", "parity_projection",
                            "random_linear_projection",
                            "multi_channel_projection"])
    p.add_argument("--cpu-discovery-backend", default="numpy",
                   choices=["numpy", "numba", "cuda"])
    p.add_argument(
        "--worker-counts", type=int, nargs="+",
        default=list(DEFAULT_WORKER_GRID),
    )
    p.add_argument("--backends", nargs="+",
                   choices=list(_BACKENDS),
                   default=list(DEFAULT_BACKENDS))
    p.add_argument("--max-consecutive-failures", type=int, default=3,
                   help="If a backend fails this many worker counts in a row, "
                        "skip the remaining counts for that backend.")
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument(
        "--label", type=str, default="discovery_parallelism_benchmark",
    )
    args = p.parse_args(argv)

    cfg = {
        "rules_json": str(args.rules_json),
        "m4c_rules": str(args.m4c_rules) if args.m4c_rules else None,
        "m4a_rules": str(args.m4a_rules) if args.m4a_rules else None,
        "n_rules_per_source": int(args.n_rules_per_source),
        "test_seeds": _parse_seeds_arg(args.seeds),
        "timesteps": int(args.timesteps),
        "grid": [int(g) for g in args.grid],
        "max_candidates": int(args.max_candidates),
        "hce_replicates": int(args.hce_replicates),
        "horizons": [int(h) for h in args.horizons],
        "projections": list(args.projections),
        "cpu_discovery_backend": args.cpu_discovery_backend,
    }

    ts = _ts()
    out = args.out_root / f"{args.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    failure_log = out / "failure_log.jsonl"
    results_csv = out / "results.csv"
    summary_md = out / "summary.md"

    work = _build_workload(cfg)

    # Resolve relative scratch root to a per-trial subdir under out.
    scratch_root = out / "_workitems"
    scratch_root.mkdir(parents=True, exist_ok=True)
    crash_root = out / "per_trial"
    crash_root.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(json.dumps({
        **cfg,
        "label": args.label,
        "timestamp_utc": ts,
        "n_cells": len(work),
        "backends": list(args.backends),
        "worker_counts": list(args.worker_counts),
        "platform": sys.platform,
    }, indent=2), encoding="utf-8")

    rows: list[dict] = []
    print(
        f"=== G5A discovery parallelism benchmark ==="
    )
    print(f"  out:           {out}")
    print(f"  workload:      {len(work)} cells (T={cfg['timesteps']}, "
          f"grid={cfg['grid']}, max_cands={cfg['max_candidates']})")
    print(f"  worker counts: {args.worker_counts}")
    print(f"  backends:      {args.backends}")
    print()

    for backend_name in args.backends:
        runner = _BACKENDS[backend_name]
        consecutive_fail = 0
        for nw in args.worker_counts:
            trial_label = f"{backend_name}__n{nw}__{_ts()}"
            scratch = scratch_root / trial_label
            scratch.mkdir(parents=True, exist_ok=True)
            crash_dir = crash_root / trial_label
            crash_dir.mkdir(parents=True, exist_ok=True)
            print(f"[trial] backend={backend_name} workers={nw} ...",
                  flush=True)
            t0 = time.perf_counter()
            completed, failed, pool_deaths = runner(
                workers=nw, work=work, cfg=cfg,
                scratch_dir=str(scratch), crash_log_dir=str(crash_dir),
            )
            wall = time.perf_counter() - t0

            n_done = len(completed)
            n_failed = len(failed)
            n_candidates = sum(c.get("n_scaffolds", 0) for c in completed)
            payload_mb = sum(c.get("payload_mb", 0.0) for c in completed)
            scratch_bytes = sum(
                p.stat().st_size for p in scratch.glob("*.npz") if p.is_file()
            )
            cells_per_sec = (n_done / wall) if wall > 0 else 0.0
            row = {
                "backend": backend_name,
                "n_workers": nw,
                "n_cells_attempted": len(work),
                "n_cells_completed": n_done,
                "n_cells_failed": n_failed,
                "pool_deaths": pool_deaths,
                "wall_s": round(wall, 2),
                "candidates_total": n_candidates,
                "payload_mb_completed": round(payload_mb, 1),
                "scratch_disk_mb": round(scratch_bytes / (1024 ** 2), 1),
                "cells_per_sec": round(cells_per_sec, 4),
                "trial_label": trial_label,
            }
            rows.append(row)
            print(
                f"  -> done={n_done}/{len(work)} fail={n_failed} "
                f"pool_deaths={pool_deaths} wall={wall:.1f}s "
                f"cands={n_candidates}",
                flush=True,
            )

            # Append failures to the global failure log.
            with failure_log.open("a", encoding="utf-8") as f:
                for cell_id, reason in failed:
                    rule_id = (
                        work[cell_id][0]["rule_id"] if 0 <= cell_id < len(work)
                        else "(unknown)"
                    )
                    seed = work[cell_id][1] if 0 <= cell_id < len(work) else -1
                    f.write(json.dumps({
                        "ts": _now_iso(),
                        "trial": trial_label,
                        "backend": backend_name,
                        "n_workers": nw,
                        "cell_id": cell_id,
                        "rule_id": rule_id,
                        "seed": seed,
                        "reason": reason,
                    }) + "\n")

            # Aggressively delete this trial's scratch to bound disk usage.
            try:
                shutil.rmtree(scratch)
            except OSError:
                pass

            # Adaptive short-circuit.
            trial_failed_overall = (n_done == 0) or (pool_deaths > 0)
            if trial_failed_overall:
                consecutive_fail += 1
                if consecutive_fail >= int(args.max_consecutive_failures):
                    print(
                        f"  [skip] {backend_name}: "
                        f"{consecutive_fail} consecutive failed trials; "
                        f"skipping remaining worker counts."
                    )
                    break
            else:
                consecutive_fail = 0

    # Write CSV.
    if rows:
        fieldnames = list(rows[0].keys())
        with results_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Write summary.md.
    lines: list[str] = []
    lines.append("# G5A discovery parallelism benchmark")
    lines.append("")
    lines.append(f"* timestamp: `{ts}`")
    lines.append(f"* workload: {len(work)} cells "
                 f"(T={cfg['timesteps']}, grid={cfg['grid']}, "
                 f"max_candidates={cfg['max_candidates']})")
    lines.append(f"* worker counts: {args.worker_counts}")
    lines.append(f"* backends: {args.backends}")
    lines.append("")
    lines.append("## Per-trial results")
    lines.append("")
    lines.append(
        "| backend | workers | done | fail | pool_deaths | wall (s) "
        "| candidates | scratch (MB) | cells/sec |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['backend']} | {r['n_workers']} | "
            f"{r['n_cells_completed']} | {r['n_cells_failed']} | "
            f"{r['pool_deaths']} | {r['wall_s']} | "
            f"{r['candidates_total']} | {r['scratch_disk_mb']} | "
            f"{r['cells_per_sec']} |"
        )

    # Best-stable-worker per backend.
    lines.append("")
    lines.append("## Stable worker ceilings (max workers with 100% success)")
    lines.append("")
    by_backend: dict[str, list[dict]] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)
    lines.append("| backend | max stable workers | best wall (s) at that count | best cells/sec |")
    lines.append("|---|---:|---:|---:|")
    for backend_name, brs in by_backend.items():
        full = [r for r in brs
                if r["n_cells_completed"] == r["n_cells_attempted"]
                and r["pool_deaths"] == 0]
        if not full:
            lines.append(f"| {backend_name} | (none) | — | — |")
            continue
        best = max(full, key=lambda r: r["n_workers"])
        lines.append(
            f"| {backend_name} | {best['n_workers']} | "
            f"{best['wall_s']} | {best['cells_per_sec']} |"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDone. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
