"""GPU-aware projection-robustness runner (Stage G2).

Routes the perturbed-rollout phase of Follow-up Topic 1 to a single
CuPy controller process while keeping candidate discovery, perturbation
construction, posthoc, and IO on CPU. Writes the same artifact bundle
as :mod:`observer_worlds.experiments.run_followup_projection_robustness`
plus a GPU metadata block in ``stats_summary.json``.

Contract recap (must hold; see ``docs/GPU_BACKEND_PLAN.md``):

* The CPU runner remains the source of truth. This module is additive.
* Candidate discovery and perturbation construction match the CPU
  runner's RNG path **exactly**, so binary-projection HCE numbers are
  bit-identical to the CPU output and continuous-projection numbers
  agree at fp32 tolerance.
* No CPU/GPU copy occurs inside the timestep loop; only initial state
  upload and compact ``(B, len(horizons))`` HCE matrices cross the
  boundary.
* No production GPU run is performed by this module unless the caller
  explicitly invokes it without ``--equivalence-audit``. This file's
  primary purpose at G2 is to *validate* the GPU path.

Equivalence-audit mode (``--equivalence-audit``) runs the full CPU
reference pipeline alongside the GPU pipeline on identical seeds /
config and writes a side-by-side report:

    outputs/gpu_equivalence_projection_<timestamp>/
        cpu/   (full CPU run-dir)
        gpu/   (full GPU run-dir)
        equivalence_report.json
        equivalence_summary.md
"""
from __future__ import annotations

import os as _os

# Pin BLAS thread counts BEFORE importing numpy / scipy / joblib so the
# joblib loky workers (Windows spawn) inherit the limits. See
# observer_worlds.experiments._parallel_discovery for the why.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_k, "1")

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from observer_worlds.analysis.projection_robustness_plots import write_all_plots
from observer_worlds.analysis.projection_robustness_stats import (
    aggregate_per_projection,
    aggregate_per_projection_and_source,
    compare_m7_vs_baselines_by_projection,
    write_summary_md,
)
from observer_worlds.backends import (
    estimate_max_safe_batch_size,
    get_backend,
    is_cupy_available,
)
from observer_worlds.experiments._followup_projection import (
    CandidateMetrics,
)
from observer_worlds.experiments._followup_projection_gpu import (
    measure_batch_on_gpu,
)
from observer_worlds.experiments._parallel_discovery import (
    CandidateScaffold as _CandidateScaffold,
    parallel_discover_all_cells,
)
from observer_worlds.experiments.run_followup_projection_robustness import (
    _build_frozen_manifest,
    _flatten_results,
    _load_rules,
    _parse_seeds_arg,
    _write_candidate_metrics_csv,
    _write_simple_csv,
)
from observer_worlds.perf import Profiler
from observer_worlds.projection import default_suite

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GPU-aware projection-robustness runner (Stage G2).",
    )
    p.add_argument("--rules-from", "--rules-json", dest="rules_json",
                   type=Path,
                   default=REPO / "release" / "rules" / "m7_top_hce_rules.json")
    p.add_argument("--m4c-rules", type=Path, default=None)
    p.add_argument("--m4a-rules", type=Path, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=5)
    p.add_argument("--seeds", type=str, default="7000..7019")
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8],
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--projections", nargs="+",
                   default=default_suite().names())
    p.add_argument("--backend", default="cupy", choices=["cupy", "numpy"])
    p.add_argument("--n-workers", type=int, default=None,
                   help="CPU candidate-discovery worker count "
                        "(joblib loky; default: cpu_count - 2). The GPU "
                        "controller is always single-process.")
    p.add_argument("--gpu-batch-size", type=int, default=64)
    p.add_argument("--gpu-memory-target-gb", type=float, default=9.5)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--cpu-discovery-backend", default="numpy",
                   choices=["numpy", "numba", "cuda"],
                   help="Backend used for the CPU-side substrate rollout "
                        "during candidate discovery.")
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str,
                   default="followup_projection_robustness_gpu")
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--equivalence-audit", action="store_true",
        help="Run CPU and GPU pipelines on identical config and emit "
             "a side-by-side equivalence report.",
    )
    return p


# ---------------------------------------------------------------------------
# Config + setup
# ---------------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace) -> dict:
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
        "backend": args.backend,
        "cpu_discovery_backend": args.cpu_discovery_backend,
        "gpu_batch_size": int(args.gpu_batch_size),
        "gpu_memory_target_gb": float(args.gpu_memory_target_gb),
        "gpu_device": int(args.gpu_device),
        "n_workers": (
            int(args.n_workers) if args.n_workers is not None
            else max(1, (os.cpu_count() or 2) - 2)
        ),
        "label": args.label,
        "equivalence_audit": bool(args.equivalence_audit),
    }
    suite = default_suite()
    for name in cfg["projections"]:
        if name not in suite.names():
            raise SystemExit(
                f"unknown projection {name!r}; available: {suite.names()}"
            )
    return cfg


def _make_out_dir(out_root: Path, label: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"{label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def _gather_rules(cfg: dict) -> list[dict]:
    rule_records: list[dict] = []
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
            rule_records.append({
                "rule": r, "rule_id": rid, "rule_source": source_label,
            })
    return rule_records


# ---------------------------------------------------------------------------
# CPU discovery (parallelized; see _parallel_discovery.py)
# ---------------------------------------------------------------------------
#
# ``_CandidateScaffold`` is imported from
# :mod:`observer_worlds.experiments._parallel_discovery` (alias above).
# G3 refactor: the previous in-file ``_WorkItem`` dataclass was retired
# because the parallel discovery path saves work items to per-(cell,
# projection) ``.npz`` files instead of returning them through joblib
# IPC. The GPU rollout phase loads those files lazily.


def _discover_and_build_work_list_parallel(
    *, cfg: dict, rule_records: list[dict],
    scratch_dir: str, profiler: Profiler,
) -> tuple[dict, dict, dict, dict]:
    """Run CPU-side discovery for every (rule, seed) cell in parallel
    via :func:`parallel_discover_all_cells`.

    Returns
    -------
    cell_meta
        ``(rule_id, seed, projection) -> meta_dict``
    scaffolds
        ``(rule_id, seed, projection, candidate_id) -> CandidateScaffold``
    work_files_per_proj
        ``projection -> [npz_path, ...]`` (sorted)
    discovery_stats
        ``{n_cells, n_workers, payload_mb_total}``
    """
    with profiler.phase("cpu_discovery"):
        return parallel_discover_all_cells(
            rule_records=rule_records,
            cfg=cfg,
            scratch_dir=scratch_dir,
            n_workers=int(cfg["n_workers"]),
        )


# ---------------------------------------------------------------------------
# GPU rollout phase
# ---------------------------------------------------------------------------


def _gpu_rollout_phase(
    *, work_files_per_proj: dict[str, list[str]],
    scaffolds: dict, cfg: dict,
    profiler: Profiler, backend_name: str,
) -> dict:
    """Drive the GPU rollouts: per projection, load each cell's work-item
    npz file, batch by ``gpu_batch_size``, and populate per-replicate
    HCE/far_HCE into scaffolds.

    The npz files are produced by the parallel CPU discovery phase
    (see :mod:`_parallel_discovery`); each file holds one (cell,
    projection)'s work items as ``(N, ...)`` numpy arrays. Loading
    one file at a time bounds peak host-side memory.

    Returns a dict of GPU stats:
    ``{gpu_batches, gpu_jobs, gpu_compute_s, gpu_memory_peak_mb,
       gpu_transfer_in_s, gpu_transfer_out_s}``.
    """
    horizons = sorted(set(int(h) for h in cfg["horizons"]))
    backend = get_backend(backend_name, device=cfg["gpu_device"])
    chunk = max(1, int(cfg["gpu_batch_size"]))

    stats = {
        "gpu_batches": 0,
        "gpu_jobs": 0,
        "gpu_compute_s": 0.0,
        "gpu_transfer_in_s": 0.0,
        "gpu_transfer_out_s": 0.0,
        "gpu_memory_peak_mb": 0.0,
    }

    for projection, file_paths in work_files_per_proj.items():
        proj_params = _projection_params(projection)
        for path in file_paths:
            data = np.load(path)
            n = int(data["states_orig"].shape[0])
            # Identify (rule_id, seed) from the path's parent npz name.
            # Filename pattern: cell_<rid>_<seed>_<projection>.npz
            # The (rid, seed) is needed to look up scaffolds.
            stem = Path(path).stem  # "cell_<rid>_<seed>_<projection>"
            # Strip the leading "cell_" and trailing "_<projection>".
            head = stem[len("cell_"):]
            head = head[:-(len(projection) + 1)]
            # Now `head` is "<rid>_<seed>". seed is the trailing int.
            last_us = head.rfind("_")
            rid = head[:last_us]
            seed = int(head[last_us + 1:])
            cand_ids = data["candidate_ids"]
            replicates = data["replicates"]

            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                states_o = data["states_orig"][start:end]
                states_h = data["states_hidden"][start:end]
                states_f = data["states_far"][start:end]
                bl = data["birth_luts"][start:end]
                sl = data["surv_luts"][start:end]
                masks = data["candidate_local_masks"][start:end]
                avail = data["avail_steps"][start:end]

                t_xfer_in = time.perf_counter()
                # measure_batch_on_gpu does the device upload internally;
                # we measure the call as a whole (transfer+compute).
                t0 = time.perf_counter()
                with profiler.phase("gpu_batch_rollout"):
                    hce, far = measure_batch_on_gpu(
                        backend=backend,
                        states_orig=states_o,
                        states_hidden=states_h,
                        states_far=states_f,
                        birth_luts=bl, surv_luts=sl,
                        candidate_local_masks=masks,
                        horizons=horizons,
                        avail_steps=avail,
                        projection=projection,
                        projection_params=proj_params,
                    )
                stats["gpu_compute_s"] += time.perf_counter() - t0
                stats["gpu_batches"] += 1
                stats["gpu_jobs"] += int(end - start)

                # Sample GPU mem after each batch; peak-track.
                if backend.is_gpu and is_cupy_available():
                    try:
                        import cupy as cp
                        free, total = cp.cuda.Device(int(cfg["gpu_device"])).mem_info
                        used_mb = (total - free) / (1024 ** 2)
                        if used_mb > stats["gpu_memory_peak_mb"]:
                            stats["gpu_memory_peak_mb"] = round(used_mb, 1)
                    except Exception:
                        pass

                # Distribute results back to scaffolds.
                for i in range(end - start):
                    cid = int(cand_ids[start + i])
                    rep = int(replicates[start + i])
                    key = (rid, seed, projection, cid)
                    sc = scaffolds[key]
                    while len(sc.hce_per_replicate_per_horizon) <= rep:
                        sc.hce_per_replicate_per_horizon.append(
                            [float("nan")] * len(horizons),
                        )
                        sc.far_per_replicate_per_horizon.append(
                            [float("nan")] * len(horizons),
                        )
                    sc.hce_per_replicate_per_horizon[rep] = hce[i].tolist()
                    sc.far_per_replicate_per_horizon[rep] = far[i].tolist()
            data.close() if hasattr(data, "close") else None
    return stats


def _projection_params(projection: str) -> dict:
    """Default kwargs the CPU runner uses for each projection.

    Matches the registration in
    :func:`observer_worlds.projection.projection_suite.default_suite`.
    """
    if projection == "mean_threshold":
        return {"theta": 0.5}
    if projection == "sum_threshold":
        return {"theta": 1}
    if projection == "random_linear_projection":
        return {"seed": 0}
    if projection == "multi_channel_projection":
        return {"n_channels": 4, "seed": 0}
    return {}


# ---------------------------------------------------------------------------
# Aggregate scaffolds -> CandidateMetrics + per-cell rollups
# ---------------------------------------------------------------------------


def _finalize_metrics(
    scaffolds: dict, cell_meta: dict, cfg: dict,
) -> tuple[list[dict], list[dict]]:
    """Build the same flattened ``(candidate_rows, cell_rows)`` shape as
    the CPU runner's :func:`_flatten_results`.

    Determinism note: candidate_rows / cell_rows order is fixed by
    ``(rule_id, seed, projection, candidate_id)`` so worker count does
    not change CSV row order.
    """
    horizons = sorted(set(int(h) for h in cfg["horizons"]))
    # Group scaffolds by cell, preserving deterministic order.
    by_cell: dict[tuple, dict[str, list[_CandidateScaffold]]] = {}
    sorted_keys = sorted(scaffolds.keys(), key=lambda k: (k[0], k[1], k[2], k[3]))
    for k in sorted_keys:
        sc = scaffolds[k]
        rid, seed, proj, _cid = k
        by_cell.setdefault((rid, seed), {}).setdefault(proj, []).append(sc)

    candidate_rows: list[dict] = []
    cell_rows: list[dict] = []
    for cell_key in sorted(by_cell.keys()):
        projections = by_cell[cell_key]
        for proj in sorted(projections.keys()):
            scs = projections[proj]
            meta = cell_meta[(cell_key[0], cell_key[1], proj)]
            cell_rows.append({
                "rule_id": meta["rule_id"],
                "rule_source": meta["rule_source"],
                "seed": meta["seed"],
                "projection": proj,
                "n_candidates": meta["n_candidates"],
                "projection_supports_threshold_margin":
                    meta["projection_supports_threshold_margin"],
                "projection_output_kind": meta["projection_output_kind"],
            })
            for sc in scs:
                cm = _scaffold_to_candidate_metrics(sc, horizons)
                candidate_rows.append({
                    "rule_id": sc.rule_id,
                    "rule_source": sc.rule_source,
                    "seed": sc.seed,
                    "projection": proj,
                    "n_candidates": meta["n_candidates"],
                    "projection_supports_threshold_margin":
                        meta["projection_supports_threshold_margin"],
                    "projection_output_kind": meta["projection_output_kind"],
                    "candidate_id": cm.candidate_id,
                    "track_id": cm.track_id,
                    "peak_frame": cm.peak_frame,
                    "lifetime": cm.lifetime,
                    "valid": bool(cm.valid),
                    "invalid_reason": cm.invalid_reason,
                    "preservation_strategy": cm.preservation_strategy,
                    "HCE": cm.HCE,
                    "far_HCE": cm.far_HCE,
                    "sham_HCE": cm.sham_HCE,
                    "hidden_vs_far_delta": cm.hidden_vs_far_delta,
                    "hidden_vs_sham_delta": cm.hidden_vs_sham_delta,
                    "initial_projection_delta": cm.initial_projection_delta,
                    "far_initial_projection_delta":
                        cm.far_initial_projection_delta,
                    "n_flipped_hidden": cm.n_flipped_hidden,
                    "n_flipped_far": cm.n_flipped_far,
                })
    return candidate_rows, cell_rows


def _scaffold_to_candidate_metrics(
    sc: _CandidateScaffold, horizons: Sequence[int],
) -> CandidateMetrics:
    if not sc.accepted_first_replicate:
        return CandidateMetrics(
            candidate_id=sc.candidate_id, track_id=sc.track_id,
            peak_frame=sc.peak_frame, lifetime=sc.lifetime,
            valid=False, invalid_reason=sc.invalid_reason,
            preservation_strategy=sc.preservation_strategy,
            HCE=None, far_HCE=None, sham_HCE=None,
            hidden_vs_far_delta=None, hidden_vs_sham_delta=None,
            initial_projection_delta=sc.initial_projection_delta,
            far_initial_projection_delta=sc.far_initial_projection_delta,
            n_flipped_hidden=sc.n_flipped_hidden_first,
            n_flipped_far=sc.n_flipped_far_first,
        )
    # Mirror CPU semantics: append per-(replicate, horizon) into one
    # flat list (skipping NaN entries from horizons that exceed
    # avail_steps), then take the mean. Empty -> 0.0.
    flat_h: list[float] = []
    flat_f: list[float] = []
    for r in range(len(sc.hce_per_replicate_per_horizon)):
        for j, h in enumerate(horizons):
            v_h = sc.hce_per_replicate_per_horizon[r][j]
            v_f = sc.far_per_replicate_per_horizon[r][j]
            if not math.isnan(v_h):
                flat_h.append(v_h)
            if not math.isnan(v_f):
                flat_f.append(v_f)
    hce = float(np.mean(flat_h)) if flat_h else 0.0
    far = float(np.mean(flat_f)) if flat_f else 0.0
    sham = 0.0
    return CandidateMetrics(
        candidate_id=sc.candidate_id, track_id=sc.track_id,
        peak_frame=sc.peak_frame, lifetime=sc.lifetime,
        valid=True, invalid_reason=None,
        preservation_strategy=sc.preservation_strategy,
        HCE=hce, far_HCE=far, sham_HCE=sham,
        hidden_vs_far_delta=hce - far,
        hidden_vs_sham_delta=hce - sham,
        initial_projection_delta=sc.initial_projection_delta,
        far_initial_projection_delta=sc.far_initial_projection_delta,
        n_flipped_hidden=sc.n_flipped_hidden_first,
        n_flipped_far=sc.n_flipped_far_first,
    )


# ---------------------------------------------------------------------------
# Output writing (mirrors CPU runner exactly) + GPU metadata block
# ---------------------------------------------------------------------------


def _write_outputs(
    *, out: Path, cfg: dict, candidate_rows: list[dict],
    cell_rows: list[dict], summary: dict, profiler: Profiler,
    args: argparse.Namespace, gpu_stats: dict | None,
) -> None:
    if candidate_rows:
        _write_candidate_metrics_csv(candidate_rows, out / "candidate_metrics.csv")
    else:
        (out / "candidate_metrics.csv").write_text(
            "rule_id,rule_source,seed,projection,candidate_id,HCE\n",
            encoding="utf-8",
        )
    _write_simple_csv(
        cell_rows, out / "projection_summary.csv",
        fields=["rule_id", "rule_source", "seed", "projection",
                "n_candidates",
                "projection_supports_threshold_margin",
                "projection_output_kind"],
    )

    hce_rows = []
    for proj, agg in summary["per_projection"].items():
        hce_rows.append({
            "projection": proj,
            "n_candidates_total": agg.get("n_candidates_total", 0),
            "n_valid_hidden_invisible": agg.get("n_valid_hidden_invisible", 0),
            "n_invalid_hidden_invisible":
                agg.get("n_invalid_hidden_invisible", 0),
            "mean_HCE": agg.get("mean_HCE"),
            "mean_far_HCE": agg.get("mean_far_HCE"),
            "mean_hidden_vs_far_delta": agg.get("mean_hidden_vs_far_delta"),
            "mean_hidden_vs_sham_delta": agg.get("mean_hidden_vs_sham_delta"),
            "mean_initial_projection_delta":
                agg.get("mean_initial_projection_delta"),
            "fraction_clean_initial_projection":
                agg.get("fraction_clean_initial_projection"),
        })
    _write_simple_csv(
        hce_rows, out / "hce_by_projection.csv",
        fields=["projection", "n_candidates_total",
                "n_valid_hidden_invisible", "n_invalid_hidden_invisible",
                "mean_HCE", "mean_far_HCE", "mean_hidden_vs_far_delta",
                "mean_hidden_vs_sham_delta",
                "mean_initial_projection_delta",
                "fraction_clean_initial_projection"],
    )

    _write_simple_csv(
        [], out / "mechanism_by_projection.csv",
        fields=["projection", "boundary_and_interior_co_mediated_fraction",
                "global_chaotic_fraction", "threshold_mediated_fraction",
                "_status"],
    )
    audit_rows = []
    for row in cell_rows:
        proj = row["projection"]
        agg = summary["per_projection"].get(proj, {})
        audit_rows.append({
            "rule_id": row["rule_id"],
            "rule_source": row["rule_source"],
            "seed": row["seed"],
            "projection": proj,
            "n_candidates": row["n_candidates"],
            "n_valid_hidden_invisible_in_projection":
                agg.get("n_valid_hidden_invisible", 0),
            "n_invalid_hidden_invisible_in_projection":
                agg.get("n_invalid_hidden_invisible", 0),
            "fraction_clean_initial_projection_in_projection":
                agg.get("fraction_clean_initial_projection"),
            "projection_supports_threshold_margin":
                row["projection_supports_threshold_margin"],
            "projection_output_kind": row["projection_output_kind"],
            "mean_initial_projection_delta_for_projection":
                agg.get("mean_initial_projection_delta"),
        })
    _write_simple_csv(
        audit_rows, out / "projection_artifact_audit.csv",
        fields=["rule_id", "rule_source", "seed", "projection",
                "n_candidates",
                "n_valid_hidden_invisible_in_projection",
                "n_invalid_hidden_invisible_in_projection",
                "fraction_clean_initial_projection_in_projection",
                "projection_supports_threshold_margin",
                "projection_output_kind",
                "mean_initial_projection_delta_for_projection"],
    )

    # Stats summary with GPU metadata block (the new G2 contract).
    summary["gpu_metadata"] = _gpu_metadata_block(cfg, gpu_stats)
    (out / "stats_summary.json").write_text(
        json.dumps(
            summary, indent=2,
            default=lambda o:
                float(o) if isinstance(o, np.floating) else
                (int(o) if isinstance(o, np.integer) else
                 (o.tolist() if isinstance(o, np.ndarray) else str(o))),
        ),
        encoding="utf-8",
    )

    try:
        write_all_plots(summary, candidate_rows, out / "plots")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plot generation failed: {e!r}")

    if args.profile:
        profiler.snapshot_memory("end_of_run")
        if cfg["backend"] == "cupy" and is_cupy_available():
            profiler.snapshot_gpu_memory("end_of_run")
        profiler.write_json(out / "perf_profile.json")

    write_summary_md(summary, out / "summary.md")


def _gpu_metadata_block(cfg: dict, gpu_stats: dict | None) -> dict:
    """The G2 stats_summary GPU block."""
    block = {
        "gpu_backend_used": cfg["backend"],
        "gpu_batch_size": cfg["gpu_batch_size"],
        "gpu_memory_target_gb": cfg["gpu_memory_target_gb"],
        "cpu_gpu_equivalence_mode": (
            "audit" if cfg.get("equivalence_audit") else "trust_g1"
        ),
        "state_transfer_policy":
            "states uploaded once per chunk; only (B,len(horizons)) "
            "metric matrices returned; no copies inside timestep loop",
        "gpu_model": None,
        "gpu_total_memory_mb": None,
        "gpu_stats": gpu_stats or {},
    }
    if cfg["backend"] == "cupy" and is_cupy_available():
        try:
            import cupy as cp
            props = cp.cuda.runtime.getDeviceProperties(int(cfg["gpu_device"]))
            block["gpu_model"] = props["name"].decode("utf-8", errors="replace")
            free, total = cp.cuda.Device(int(cfg["gpu_device"])).mem_info
            block["gpu_total_memory_mb"] = round(total / (1024 ** 2), 1)
            block["gpu_free_memory_mb_at_end"] = round(free / (1024 ** 2), 1)
        except Exception:  # noqa: BLE001
            pass
    return block


# ---------------------------------------------------------------------------
# Top-level runner (single GPU pipeline run)
# ---------------------------------------------------------------------------


def run_gpu_pipeline(
    args: argparse.Namespace, cfg: dict, out: Path,
) -> dict:
    """Execute the GPU pipeline end-to-end. Writes the standard artifact
    bundle into ``out`` and returns the summary dict."""
    profiler = Profiler(label="projection_robustness_gpu")

    (out / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8",
    )
    (out / "frozen_manifest.json").write_text(
        json.dumps(_build_frozen_manifest(cfg), indent=2), encoding="utf-8",
    )

    rule_records = _gather_rules(cfg)
    print("=" * 72)
    print("Follow-up Topic 1: projection robustness — Stage G2 GPU runner")
    print("=" * 72)
    sources_used = sorted({rec["rule_source"] for rec in rule_records})
    print(f"  out                   = {out}")
    print(f"  backend               = {cfg['backend']}")
    print(f"  cpu_discovery_backend = {cfg['cpu_discovery_backend']}")
    print(f"  sources               = {sources_used}")
    print(f"  rules                 = {len(rule_records)}")
    print(f"  seeds                 = {len(cfg['test_seeds'])} "
          f"({cfg['test_seeds'][0]}..{cfg['test_seeds'][-1]})")
    print(f"  timesteps             = {cfg['timesteps']}")
    print(f"  grid                  = {cfg['grid']}")
    print(f"  horizons              = {cfg['horizons']}")
    print(f"  projections           = {cfg['projections']}")
    print(f"  replicates            = {cfg['hce_replicates']}")
    print(f"  cpu_discovery_workers = {cfg['n_workers']}")
    print(f"  gpu_batch_size        = {cfg['gpu_batch_size']}")
    print(f"  gpu_memory_target_gb  = {cfg['gpu_memory_target_gb']}")
    if cfg["backend"] == "cupy" and is_cupy_available():
        import cupy as cp
        free, total = cp.cuda.Device(int(cfg["gpu_device"])).mem_info
        print(f"  gpu_free_mem_mb       = {free / (1024 ** 2):.0f} / "
              f"{total / (1024 ** 2):.0f}")
    print()

    t_total = time.perf_counter()
    # Phase 1: parallel CPU discovery + work-list construction.
    # Workers write per-(cell, projection) npz files under <out>/_workitems/
    # and return only compact metadata + file paths (no large arrays
    # cross joblib IPC).
    scratch_dir = out / "_workitems"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    t_disc = time.perf_counter()
    cell_meta, scaffolds, work_files_per_proj, discovery_stats = (
        _discover_and_build_work_list_parallel(
            cfg=cfg, rule_records=rule_records,
            scratch_dir=str(scratch_dir),
            profiler=profiler,
        )
    )
    discovery_wall = time.perf_counter() - t_disc
    n_jobs = sum(
        sum(int(np.load(p)["states_orig"].shape[0]) for p in paths)
        for paths in work_files_per_proj.values()
    )
    print(
        f"  CPU phase: {len(scaffolds)} candidates discovered, "
        f"{n_jobs} GPU rollout jobs staged "
        f"({discovery_stats['payload_mb_total']:.1f} MB on scratch), "
        f"{discovery_wall:.1f}s wall, "
        f"{cfg['n_workers']} workers."
    )

    # Phase 2: GPU rollout (single controller, npz-driven).
    t_gpu = time.perf_counter()
    if work_files_per_proj:
        gpu_stats = _gpu_rollout_phase(
            work_files_per_proj=work_files_per_proj,
            scaffolds=scaffolds, cfg=cfg,
            profiler=profiler, backend_name=cfg["backend"],
        )
    else:
        gpu_stats = {
            "gpu_batches": 0, "gpu_jobs": 0, "gpu_compute_s": 0.0,
            "gpu_transfer_in_s": 0.0, "gpu_transfer_out_s": 0.0,
            "gpu_memory_peak_mb": 0.0,
        }
    gpu_wall = time.perf_counter() - t_gpu
    print(
        f"  GPU phase: {gpu_stats['gpu_batches']} batches, "
        f"{gpu_stats['gpu_jobs']} jobs, "
        f"{gpu_stats['gpu_compute_s']:.2f}s compute, "
        f"{gpu_wall:.2f}s wall, "
        f"peak {gpu_stats['gpu_memory_peak_mb']:.0f} MB."
    )

    # Phase 3: aggregate, write CSVs.
    with profiler.phase("csv_write"):
        candidate_rows, cell_rows = _finalize_metrics(scaffolds, cell_meta, cfg)
    sources_present = sorted({r["rule_source"] for r in candidate_rows})

    with profiler.phase("stats"):
        summary = aggregate_per_projection(candidate_rows, cfg["projections"])
        summary["wall_time_seconds_sweep"] = time.perf_counter() - t_total
        summary["n_cells"] = len(cell_meta)
        summary["n_candidate_rows"] = len(candidate_rows)
        summary["projections_evaluated"] = list(cfg["projections"])
        summary["state_stream_returned"] = False
        summary["returned_payload_bytes_estimate"] = (
            sum(int(r.get("n_candidates", 0)) * 256 for r in cell_rows)
        )
        summary["sources_present"] = sources_present
        if len(sources_present) >= 2:
            summary.update(aggregate_per_projection_and_source(
                candidate_rows, cfg["projections"], sources=sources_present,
            ))
            summary.update(compare_m7_vs_baselines_by_projection(
                candidate_rows, cfg["projections"],
            ))

    # G3 perf counters into summary.
    total_wall = time.perf_counter() - t_total
    # Stage 6C CPU baseline (numpy, 30 workers): 23318 s = 6.48 h.
    stage6c_baseline_s = 23318.0
    summary["g3_perf"] = {
        "cpu_discovery_workers": int(cfg["n_workers"]),
        "cpu_discovery_wall_s": round(discovery_wall, 2),
        "cpu_discovery_cells_per_second": (
            round(discovery_stats["n_cells"] / discovery_wall, 3)
            if discovery_wall > 0 else 0
        ),
        "cpu_discovery_payload_mb_estimate":
            discovery_stats["payload_mb_total"],
        "gpu_rollout_wall_s": round(gpu_wall, 2),
        "gpu_batches": gpu_stats["gpu_batches"],
        "gpu_jobs": gpu_stats["gpu_jobs"],
        "gpu_rollouts_per_second": (
            round(gpu_stats["gpu_jobs"] / gpu_wall, 2)
            if gpu_wall > 0 else 0
        ),
        "gpu_memory_peak_mb": gpu_stats["gpu_memory_peak_mb"],
        "total_wall_s": round(total_wall, 2),
        "speedup_vs_stage6c_cpu_baseline": (
            round(stage6c_baseline_s / total_wall, 2)
            if total_wall > 0 else None
        ),
        "stage6c_cpu_baseline_s": stage6c_baseline_s,
        **gpu_stats,
    }

    _write_outputs(
        out=out, cfg=cfg, candidate_rows=candidate_rows,
        cell_rows=cell_rows, summary=summary, profiler=profiler,
        args=args, gpu_stats=gpu_stats,
    )

    print(f"\nGPU pipeline done in {total_wall:.1f}s. Output: {out}")
    return summary


# ---------------------------------------------------------------------------
# Equivalence audit driver
# ---------------------------------------------------------------------------


def _run_cpu_reference(args, cfg, out_cpu: Path) -> int:
    """Drive the CPU reference runner (existing module) on the same
    config and write into ``out_cpu``. Reuses the CPU runner's main()
    via argv construction for full code-path equivalence."""
    from observer_worlds.experiments import run_followup_projection_robustness as ref
    cpu_argv = [
        "--rules-from", cfg["rules_json"],
        "--n-rules-per-source", str(cfg["n_rules_per_source"]),
        "--seeds", ",".join(str(s) for s in cfg["test_seeds"]),
        "--timesteps", str(cfg["timesteps"]),
        "--grid", *(str(g) for g in cfg["grid"]),
        "--max-candidates", str(cfg["max_candidates"]),
        "--hce-replicates", str(cfg["hce_replicates"]),
        "--horizons", *(str(h) for h in cfg["horizons"]),
        "--projections", *cfg["projections"],
        "--backend", "numpy",
        "--n-workers", "1",
        "--label", out_cpu.name,
        "--out-root", str(out_cpu.parent),
    ]
    if cfg.get("m4c_rules"):
        cpu_argv += ["--m4c-rules", cfg["m4c_rules"]]
    if cfg.get("m4a_rules"):
        cpu_argv += ["--m4a-rules", cfg["m4a_rules"]]
    if args.profile:
        cpu_argv.append("--profile")
    return ref.main(cpu_argv)


def _compare_runs(cpu_dir: Path, gpu_dir: Path) -> dict:
    """Compare candidate_metrics.csv from a CPU and GPU run dir.

    Match rows by ``(rule_id, seed, projection, candidate_id)``. For
    each (projection, source) cell, compute mean / max abs delta of
    HCE, far_HCE, hidden_vs_far_delta, initial_projection_delta, and
    fraction matched.
    """
    cpu_rows = list(_read_csv(cpu_dir / "candidate_metrics.csv"))
    gpu_rows = list(_read_csv(gpu_dir / "candidate_metrics.csv"))

    def _index(rows):
        return {
            (r["rule_id"], int(r["seed"]), r["projection"],
             int(r["candidate_id"])): r
            for r in rows
        }

    cpu_idx = _index(cpu_rows)
    gpu_idx = _index(gpu_rows)
    keys_both = sorted(set(cpu_idx) & set(gpu_idx))
    keys_cpu_only = sorted(set(cpu_idx) - set(gpu_idx))
    keys_gpu_only = sorted(set(gpu_idx) - set(cpu_idx))

    # Tolerances per projection.
    binary_tol = 1e-5
    continuous_tol_abs = 5e-3
    continuous_tol_rel = 1e-3
    multi_tol = 1e-5

    def tol_for(projection: str) -> tuple[float, float]:
        if projection == "random_linear_projection":
            return continuous_tol_abs, continuous_tol_rel
        if projection == "multi_channel_projection":
            return multi_tol, 0.0
        return binary_tol, 0.0

    per_cell_stats: dict[tuple[str, str], dict] = {}
    overall = {
        "n_cpu_rows": len(cpu_rows),
        "n_gpu_rows": len(gpu_rows),
        "n_matched_keys": len(keys_both),
        "n_cpu_only": len(keys_cpu_only),
        "n_gpu_only": len(keys_gpu_only),
        "max_abs_HCE_delta": 0.0,
        "max_abs_far_HCE_delta": 0.0,
        "max_abs_init_delta_delta": 0.0,
        "n_HCE_within_tol": 0,
        "n_HCE_outside_tol": 0,
        "outside_tol_examples": [],
    }

    def _f(s):
        s = (s or "").strip()
        if s in ("", "None"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _within(a, b, tol_abs, tol_rel):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        d = abs(a - b)
        ref = max(abs(a), abs(b), 1.0)
        return d <= tol_abs + tol_rel * ref

    for k in keys_both:
        cpu_r = cpu_idx[k]
        gpu_r = gpu_idx[k]
        proj = k[2]
        rid = cpu_r["rule_source"]
        cell = (proj, rid)
        bucket = per_cell_stats.setdefault(cell, {
            "projection": proj,
            "rule_source": rid,
            "n_compared": 0,
            "max_abs_HCE_delta": 0.0,
            "max_abs_far_HCE_delta": 0.0,
            "max_abs_init_delta_delta": 0.0,
            "n_HCE_within_tol": 0,
            "n_HCE_outside_tol": 0,
        })
        bucket["n_compared"] += 1
        a, r = tol_for(proj)
        for label_name, key_name, max_field, count_within, count_outside in [
            ("HCE", "HCE", "max_abs_HCE_delta",
             "n_HCE_within_tol", "n_HCE_outside_tol"),
            ("far_HCE", "far_HCE", "max_abs_far_HCE_delta",
             None, None),
            ("init_delta", "initial_projection_delta",
             "max_abs_init_delta_delta", None, None),
        ]:
            v_c = _f(cpu_r.get(key_name))
            v_g = _f(gpu_r.get(key_name))
            if v_c is None or v_g is None:
                continue
            if math.isnan(v_c) or math.isnan(v_g):
                continue
            d = abs(v_c - v_g)
            if d > bucket[max_field]:
                bucket[max_field] = d
            if d > overall[max_field]:
                overall[max_field] = d
            if count_within is not None:
                if _within(v_c, v_g, a, r):
                    bucket[count_within] += 1
                    overall[count_within] += 1
                else:
                    bucket[count_outside] += 1
                    overall[count_outside] += 1
                    if len(overall["outside_tol_examples"]) < 10:
                        overall["outside_tol_examples"].append({
                            "key": list(k),
                            "field": label_name,
                            "cpu": v_c, "gpu": v_g, "abs_delta": d,
                            "tol_abs": a, "tol_rel": r,
                        })
    # JSON requires string-typed dict keys; flatten the tuple-keyed
    # per-cell dict into a list and a string-keyed view.
    overall["per_cell_list"] = list(per_cell_stats.values())
    overall["per_cell"] = {
        f"{cell[0]}|{cell[1]}": v
        for cell, v in per_cell_stats.items()
    }
    overall["pass"] = bool(overall["n_HCE_outside_tol"] == 0)
    return overall


def _read_csv(p: Path):
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield row


def _write_equivalence_report(audit_dir: Path, report: dict) -> None:
    (audit_dir / "equivalence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    lines = []
    lines.append("# Stage G2 — CPU vs GPU equivalence (projection robustness)")
    lines.append("")
    lines.append(f"* CPU rows compared: `{report['n_cpu_rows']}`")
    lines.append(f"* GPU rows compared: `{report['n_gpu_rows']}`")
    lines.append(f"* Matched keys: `{report['n_matched_keys']}`")
    lines.append(f"* CPU-only keys: `{report['n_cpu_only']}`")
    lines.append(f"* GPU-only keys: `{report['n_gpu_only']}`")
    lines.append(f"* HCE within tolerance: `{report['n_HCE_within_tol']}`")
    lines.append(f"* HCE outside tolerance: `{report['n_HCE_outside_tol']}`")
    lines.append(f"* max abs HCE delta: `{report['max_abs_HCE_delta']:.3e}`")
    lines.append(
        f"* max abs far_HCE delta: `{report['max_abs_far_HCE_delta']:.3e}`"
    )
    lines.append(
        f"* max abs init_proj_delta delta: "
        f"`{report['max_abs_init_delta_delta']:.3e}`"
    )
    lines.append(
        f"* Verdict: **{'PASS' if report['pass'] else 'FAIL'}**"
    )
    lines.append("")
    lines.append("## Per (projection × source) max abs deltas")
    lines.append("")
    lines.append(
        "| projection | source | n | max |HCE| | max |far_HCE| | "
        "max |init_proj_delta| | within tol | outside |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    cell_items = sorted(
        report["per_cell_list"],
        key=lambda b: (b["projection"], b["rule_source"]),
    )
    for bucket in cell_items:
        lines.append(
            f"| {bucket['projection']} | {bucket['rule_source']} | "
            f"{bucket['n_compared']} | "
            f"{bucket['max_abs_HCE_delta']:.3e} | "
            f"{bucket['max_abs_far_HCE_delta']:.3e} | "
            f"{bucket['max_abs_init_delta_delta']:.3e} | "
            f"{bucket['n_HCE_within_tol']} | {bucket['n_HCE_outside_tol']} |"
        )
    if report["outside_tol_examples"]:
        lines.append("")
        lines.append("## Examples outside tolerance (first 10)")
        lines.append("")
        for ex in report["outside_tol_examples"]:
            lines.append(
                f"* `{ex['key']}` field=`{ex['field']}` "
                f"CPU={ex['cpu']:.6g} GPU={ex['gpu']:.6g} "
                f"|Δ|={ex['abs_delta']:.3e}"
            )
    (audit_dir / "equivalence_summary.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


def run_equivalence_audit(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    cfg["equivalence_audit"] = True
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = args.out_root / f"gpu_equivalence_projection_{ts}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cpu_root = audit_dir / "_cpu_runs"
    cpu_root.mkdir(parents=True, exist_ok=True)
    gpu_dir = audit_dir / "gpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    (gpu_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"[audit] CPU reference run -> {cpu_root}/cpu_run_*")
    rc_cpu = _run_cpu_reference(args, cfg, cpu_root / "cpu_run")
    if rc_cpu != 0:
        print(f"[error] CPU reference returned {rc_cpu}", file=sys.stderr)
        return rc_cpu
    cpu_dirs = sorted(cpu_root.glob("cpu_run_*"))
    if not cpu_dirs:
        raise RuntimeError(f"CPU reference produced no run dir under {cpu_root}")
    cpu_actual = cpu_dirs[-1]
    # Move CPU output into audit_dir/cpu/ for the documented layout.
    target_cpu = audit_dir / "cpu"
    if target_cpu.exists():
        # Defensive cleanup of any prior empty dir.
        import shutil
        shutil.rmtree(target_cpu)
    cpu_actual.rename(target_cpu)
    if not any(cpu_root.iterdir()):
        cpu_root.rmdir()

    print(f"[audit] GPU run -> {gpu_dir}")
    run_gpu_pipeline(args, cfg, gpu_dir)

    print("[audit] comparing ...")
    report = _compare_runs(target_cpu, gpu_dir)
    _write_equivalence_report(audit_dir, report)
    print(f"[audit] verdict: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"[audit] report: {audit_dir / 'equivalence_summary.md'}")
    return 0 if report["pass"] else 2


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = _resolve_config(args)

    if args.backend == "cupy" and not is_cupy_available():
        print(
            "[error] --backend cupy requested but cupy is not available "
            "or no CUDA device is reachable. "
            "Install cupy-cuda12x or use --backend numpy.",
            file=sys.stderr,
        )
        return 3

    if args.equivalence_audit:
        return run_equivalence_audit(args)

    out = _make_out_dir(args.out_root, args.label)
    run_gpu_pipeline(args, cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
