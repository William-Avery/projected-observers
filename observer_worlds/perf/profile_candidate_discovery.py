"""Stage G5B — candidate-discovery profiler.

Two complementary passes:

1. **Phase-timing pass** — runs the workload at the requested
   ``--n-workers`` and uses
   :mod:`observer_worlds.perf._phase_instrumented_discover` to
   attribute time across:

       substrate_rollout
       projection_stream    (per-projection)
       candidate_detection  (CC + tracker, per-projection)
       perturbation_construction (per-projection)
       npz_write            (per-projection)

   Output: ``profile_summary.csv`` (per-cell breakdown),
   ``phase_timing_by_projection.csv``, ``phase_timing_by_source.csv``,
   ``profile_summary.md`` (human-readable totals).

2. **cProfile pass** — runs **one cell** in a single process under
   :mod:`cProfile`, dumps pstats sorted by cumulative time to
   ``hot_functions.txt``. The intent is to expose per-function
   bottlenecks (``scipy.ndimage.convolve``, ``tracking._iou``,
   ``connected_components``, etc.) regardless of how they distribute
   across the named phases.

Outputs:

    outputs/g5b_candidate_discovery_profile_<timestamp>/
        config.json
        profile_summary.csv
        profile_summary.md
        hot_functions.txt
        phase_timing_by_projection.csv
        phase_timing_by_source.csv
"""
from __future__ import annotations

import os as _os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, "1")

import argparse
import cProfile
import csv
import io
import json
import pstats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import concurrent.futures as _cf

from observer_worlds.experiments.run_followup_projection_robustness import (
    _load_rules,
    _parse_seeds_arg,
)
from observer_worlds.perf._phase_instrumented_discover import (
    discover_one_cell_profiled,
)


REPO = Path(__file__).resolve().parents[2]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_workload(cfg: dict) -> list[tuple]:
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


def _profile_pass_parallel(work, cfg, scratch_dir, n_workers):
    """Phase-timing pass. Workers return CellProfile + scaffolds."""
    if n_workers <= 1 or len(work) <= 1:
        results = []
        for rec, seed in work:
            prof, _ = discover_one_cell_profiled(
                rule_record=rec, seed=int(seed),
                cfg=cfg, scratch_dir=scratch_dir,
            )
            results.append(prof)
        return results
    futures = []
    with _cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for rec, seed in work:
            futures.append(ex.submit(
                _run_one_for_pool,
                rec, int(seed), cfg, scratch_dir,
            ))
        out = []
        for fut in _cf.as_completed(futures):
            try:
                out.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[profile] worker exception: {type(e).__name__}: {e}",
                      file=sys.stderr)
        return out


def _run_one_for_pool(rec, seed, cfg, scratch_dir):
    """Top-level adapter so ProcessPoolExecutor can pickle it."""
    prof, _ = discover_one_cell_profiled(
        rule_record=rec, seed=int(seed),
        cfg=cfg, scratch_dir=scratch_dir,
    )
    return prof


def _cprofile_pass_one_cell(rec, seed, cfg, scratch_dir, top_n=30) -> str:
    """Run one cell under cProfile; return formatted top-N report."""
    pr = cProfile.Profile()
    pr.enable()
    discover_one_cell_profiled(
        rule_record=rec, seed=int(seed),
        cfg=cfg, scratch_dir=scratch_dir,
    )
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(int(top_n))
    return buf.getvalue()


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
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--cprofile-top-n", type=int, default=30)
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str, default="g5b_candidate_discovery_profile")
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
    scratch = out / "_workitems"
    scratch.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(
        json.dumps({**cfg, "label": args.label,
                    "timestamp_utc": ts, "n_workers": int(args.n_workers)},
                    indent=2),
        encoding="utf-8",
    )

    work = _build_workload(cfg)
    print(f"=== G5B candidate-discovery profiler ===")
    print(f"  out:        {out}")
    print(f"  workload:   {len(work)} cells "
          f"(T={cfg['timesteps']}, grid={cfg['grid']})")
    print(f"  workers:    {int(args.n_workers)}")
    print()

    # ---- Phase-timing pass ----
    print("[g5b] phase-timing pass ...")
    t0 = time.perf_counter()
    profiles = _profile_pass_parallel(
        work, cfg, str(scratch), int(args.n_workers),
    )
    phase_pass_wall = time.perf_counter() - t0
    print(f"[g5b] phase pass wall: {phase_pass_wall:.1f}s "
          f"({len(profiles)}/{len(work)} cells)")

    # ---- cProfile pass (1 cell, single-process) ----
    print("[g5b] cProfile pass (single cell, single-process) ...")
    if work:
        t0 = time.perf_counter()
        cprofile_dump = _cprofile_pass_one_cell(
            work[0][0], work[0][1], cfg, str(scratch),
            top_n=int(args.cprofile_top_n),
        )
        cprof_wall = time.perf_counter() - t0
        (out / "hot_functions.txt").write_text(
            f"# cProfile (cumulative) for one cell, "
            f"{cprof_wall:.1f}s wall\n\n" + cprofile_dump,
            encoding="utf-8",
        )
    else:
        (out / "hot_functions.txt").write_text(
            "(no cells in workload)", encoding="utf-8",
        )

    # ---- Aggregate phase timings -> CSVs / Markdown ----
    if not profiles:
        print("[g5b] no profiles produced; nothing to aggregate.",
              file=sys.stderr)
        return 1

    # profile_summary.csv: one row per cell.
    phase_names = list(profiles[0].per_phase_seconds.keys())
    fields_cell = ["rule_id", "rule_source", "seed",
                   "n_candidates_total", "work_files_written",
                   "payload_mb"] + phase_names
    with (out / "profile_summary.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields_cell)
        w.writeheader()
        for prof in sorted(profiles,
                           key=lambda p: (p.rule_source, p.rule_id, p.seed)):
            row = {
                "rule_id": prof.rule_id,
                "rule_source": prof.rule_source,
                "seed": prof.seed,
                "n_candidates_total": prof.n_candidates_total,
                "work_files_written": prof.work_files_written,
                "payload_mb": prof.payload_mb,
            }
            for ph in phase_names:
                row[ph] = round(prof.per_phase_seconds.get(ph, 0.0), 4)
            w.writerow(row)

    # phase_timing_by_projection.csv: one row per (phase, projection).
    proj_totals: dict[tuple[str, str], float] = {}
    for prof in profiles:
        for phase, perproj in prof.per_phase_per_projection.items():
            for proj, dt in perproj.items():
                proj_totals[(phase, proj)] = (
                    proj_totals.get((phase, proj), 0.0) + dt
                )
    with (out / "phase_timing_by_projection.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=["phase", "projection", "total_seconds"],
        )
        w.writeheader()
        for (phase, proj), total in sorted(proj_totals.items()):
            w.writerow({"phase": phase, "projection": proj,
                        "total_seconds": round(total, 3)})

    # phase_timing_by_source.csv: one row per (phase, source).
    source_totals: dict[tuple[str, str], float] = {}
    for prof in profiles:
        for phase, dt in prof.per_phase_seconds.items():
            source_totals[(phase, prof.rule_source)] = (
                source_totals.get((phase, prof.rule_source), 0.0) + dt
            )
    with (out / "phase_timing_by_source.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=["phase", "rule_source", "total_seconds"],
        )
        w.writeheader()
        for (phase, source), total in sorted(source_totals.items()):
            w.writerow({"phase": phase, "rule_source": source,
                        "total_seconds": round(total, 3)})

    # profile_summary.md: phase totals + per-projection breakdown.
    phase_totals = {ph: 0.0 for ph in phase_names}
    for prof in profiles:
        for ph in phase_names:
            phase_totals[ph] += prof.per_phase_seconds.get(ph, 0.0)
    grand_total = phase_totals.get("total", 0.0)

    lines: list[str] = []
    lines.append("# G5B candidate-discovery profile")
    lines.append("")
    lines.append(f"* timestamp: `{ts}`")
    lines.append(f"* cells profiled: {len(profiles)}")
    lines.append(f"* workers: {int(args.n_workers)}")
    lines.append(
        f"* phase-pass wall (parallel): {phase_pass_wall:.1f} s"
    )
    lines.append(
        f"* sum of per-cell `total` (worker-seconds): {grand_total:.1f} s"
    )
    lines.append("")
    lines.append("## Phase totals (worker-seconds, summed across cells)")
    lines.append("")
    lines.append("| phase | total (s) | % of total |")
    lines.append("|---|---:|---:|")
    grand = max(grand_total, 1e-9)
    for ph in phase_names:
        if ph == "total":
            continue
        v = phase_totals[ph]
        lines.append(f"| {ph} | {v:.2f} | {100.0 * v / grand:.1f}% |")
    lines.append(f"| **total** | **{grand_total:.2f}** | 100.0% |")
    lines.append("")
    lines.append("## Per-projection phase totals")
    lines.append("")
    proj_set = sorted({proj for (_p, proj) in proj_totals.keys()})
    phase_set = [p for p in phase_names if p != "total"]
    lines.append("| projection | " + " | ".join(phase_set) + " | sum |")
    lines.append("|---|" + "|".join(["---:"] * (len(phase_set) + 1)) + "|")
    for proj in proj_set:
        cells = [proj_totals.get((ph, proj), 0.0) for ph in phase_set]
        rowsum = sum(cells)
        lines.append(
            f"| {proj} | "
            + " | ".join(f"{v:.2f}" for v in cells)
            + f" | {rowsum:.2f} |"
        )
    lines.append("")
    lines.append("## Per-source phase totals")
    lines.append("")
    src_set = sorted({s for (_p, s) in source_totals.keys()})
    lines.append("| source | " + " | ".join(phase_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(phase_names)) + "|")
    for src in src_set:
        cells = [source_totals.get((ph, src), 0.0) for ph in phase_names]
        lines.append(
            f"| {src} | " + " | ".join(f"{v:.2f}" for v in cells) + " |"
        )
    lines.append("")
    lines.append("See `hot_functions.txt` for cProfile cumulative-time "
                 "ranking on a single cell.")
    (out / "profile_summary.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )

    print(f"[g5b] done. output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
