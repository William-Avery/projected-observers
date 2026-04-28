"""Follow-up Topic 1 — projection-robustness experiment runner (Stage 2).

Runs the workhorse pipeline in
:mod:`observer_worlds.experiments._followup_projection` over
``(rule, seed)`` cells in parallel and writes the documented output
bundle:

    config.json
    frozen_manifest.json
    projection_summary.csv
    candidate_metrics.csv
    hce_by_projection.csv
    mechanism_by_projection.csv         (Stage-5+ stub for now)
    projection_artifact_audit.csv
    stats_summary.json
    summary.md
    plots/<seven plot files>

For each (rule, seed) the 4D substrate is run **once**; every requested
projection consumes the same in-memory state stream. Per-projection HCE
is measured by candidate-local hidden perturbations; ``far`` controls
locality and ``sham`` is identically zero by construction.

Stage 2 is intended for smoke runs (`--quick`). Production sweeps will
follow in Stage 5 once the pipeline is profiled.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from observer_worlds.analysis.projection_robustness_plots import write_all_plots
from observer_worlds.analysis.projection_robustness_stats import (
    PROJECTION_METRICS,
    aggregate_per_projection,
    write_summary_md,
)
from observer_worlds.experiments._followup_projection import (
    CandidateMetrics, run_one_cell,
)
from observer_worlds.perf import Profiler
from observer_worlds.projection import default_suite
from observer_worlds.search.rules import FractionalRule


REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """argparse parser. All overridable fields default to ``None`` so we
    can layer ``_smoke_defaults`` and explicit CLI args correctly in
    :func:`_resolve_config`."""
    p = argparse.ArgumentParser(
        description="Follow-up Topic 1: projection-robustness experiment.",
    )
    p.add_argument("--quick", action="store_true",
                   help="Smoke run with reduced defaults.")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--backend", type=str, default=None,
                   choices=["numpy", "numba", "cupy", "cuda-batched"])
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str, default="followup_projection_robustness")
    p.add_argument("--n-rules-per-source", type=int, default=None)
    p.add_argument("--test-seeds", type=int, nargs="+", default=None)
    p.add_argument("--projections", nargs="+", default=None,
                   help="Subset of projections to evaluate. "
                        "Default: all six in the default suite.")
    p.add_argument("--hce-replicates", type=int, default=None)
    p.add_argument("--grid", type=int, nargs=4, default=None,
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--rules-json", type=Path,
                   default=REPO / "release" / "rules" / "m7_top_hce_rules.json",
                   help="Path to the rules JSON to load.")
    p.add_argument("--profile", action="store_true",
                   help="Wrap the run with the M-perf profiler.")
    return p


def _full_defaults() -> dict:
    """Production defaults (used when neither --quick nor an explicit
    flag overrides)."""
    return {
        "n_workers": max(1, (os.cpu_count() or 2) - 2),
        "backend": "numpy",
        "max_candidates": 20,
        "timesteps": 500,
        "horizons": [1, 2, 3, 5, 10, 20, 40, 80],
        "n_rules_per_source": 5,
        "test_seeds": list(range(6000, 6020)),
        "projections": default_suite().names(),
        "hce_replicates": 3,
        "grid": [64, 64, 8, 8],
    }


def _smoke_defaults() -> dict:
    return {
        "n_rules_per_source": 1,
        "test_seeds": [6000, 6001],
        "timesteps": 100,
        "max_candidates": 5,
        "projections": ["mean_threshold", "max_projection",
                        "parity_projection"],
        "horizons": [5, 10],
        "hce_replicates": 1,
        "grid": [16, 16, 4, 4],
    }


def _resolve_config(args: argparse.Namespace) -> dict:
    """Layer order: production defaults -> --quick smoke defaults ->
    explicit CLI args. Explicit CLI overrides always win."""
    cfg = dict(_full_defaults())
    if args.quick:
        cfg.update(_smoke_defaults())
    # Layer explicit user-supplied args (None means "not supplied").
    for key in ("n_workers", "backend", "max_candidates", "timesteps",
                "horizons", "n_rules_per_source", "test_seeds",
                "projections", "hce_replicates", "grid"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = list(v) if isinstance(v, list) else v
    cfg["rules_json"] = str(args.rules_json)
    suite = default_suite()
    for name in cfg["projections"]:
        if name not in suite.names():
            raise SystemExit(
                f"unknown projection {name!r}; "
                f"available: {suite.names()}"
            )
    return cfg


def _make_out_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_root / f"{args.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Frozen manifest (lightweight; M8-style)
# ---------------------------------------------------------------------------


def _build_frozen_manifest(cfg: dict) -> dict:
    def _safe(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(REPO), timeout=10)
            return r.stdout.strip()
        except Exception:
            return ""
    return {
        "stage": 2,
        "experiment": "followup_projection_robustness",
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
        "git": {
            "commit": _safe(["git", "rev-parse", "HEAD"]),
            "branch": _safe(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_safe(["git", "status", "--porcelain"])),
            "tag": _safe(["git", "tag", "--points-at", "HEAD"]) or None,
        },
        "config": cfg,
        "platform": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
    }


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


def _load_rules(path: Path, n: int) -> list[FractionalRule]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FractionalRule.from_dict(r) for r in raw[:int(n)]]


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_candidate_metrics_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_simple_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _flatten_results(per_cell: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flatten ``per_cell`` (one dict per (rule, seed) returned by
    run_one_cell) into:
      * ``candidate_rows`` — one row per (rule, seed, projection, candidate)
      * ``cell_rows``      — one row per (rule, seed, projection)
    """
    candidate_rows: list[dict] = []
    cell_rows: list[dict] = []
    for cell in per_cell:
        # ``cell`` is dict[projection_name -> {"candidates": [...], ...}]
        for proj_name, payload in cell.items():
            base = {
                "rule_id": payload["rule_id"],
                "rule_source": payload["rule_source"],
                "seed": payload["seed"],
                "projection": proj_name,
                "n_candidates": payload["n_candidates"],
                "projection_supports_threshold_margin":
                    payload["projection_supports_threshold_margin"],
                "projection_output_kind": payload["projection_output_kind"],
            }
            cell_rows.append(base)
            for cm in payload["candidates"]:
                candidate_rows.append({
                    **base,
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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = _resolve_config(args)
    out = _make_out_dir(args)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "frozen_manifest.json").write_text(
        json.dumps(_build_frozen_manifest(cfg), indent=2), encoding="utf-8",
    )

    # Load rules.
    rules = _load_rules(args.rules_json, cfg["n_rules_per_source"])
    rule_records = []
    for i, r in enumerate(rules):
        rid = f"M7_HCE_optimized_rank{i+1:02d}"
        rule_records.append({
            "rule": r,
            "rule_id": rid,
            "rule_source": "M7_HCE_optimized",
        })

    print("=" * 72)
    print("Follow-up Topic 1: projection robustness — Stage 2")
    print("=" * 72)
    print(f"  out         = {out}")
    print(f"  backend     = {cfg['backend']}")
    print(f"  n_workers   = {cfg['n_workers']}")
    print(f"  rules       = {len(rule_records)}")
    print(f"  seeds       = {len(cfg['test_seeds'])} ({cfg['test_seeds'][0]}..{cfg['test_seeds'][-1]})")
    print(f"  timesteps   = {cfg['timesteps']}")
    print(f"  grid        = {cfg['grid']}")
    print(f"  horizons    = {cfg['horizons']}")
    print(f"  projections = {cfg['projections']}")
    print(f"  replicates  = {cfg['hce_replicates']}")
    print(f"  profile     = {bool(args.profile)}")

    profiler = Profiler(label="projection_robustness")

    # Build (rule, seed) tasks.
    tasks = []
    for rec in rule_records:
        for seed in cfg["test_seeds"]:
            tasks.append((rec, seed))

    def _task(rec, seed):
        return run_one_cell(
            rule_bs=rec["rule"].to_bsrule(),
            rule_id=rec["rule_id"],
            rule_source=rec["rule_source"],
            seed=int(seed),
            grid_shape=tuple(cfg["grid"]),
            timesteps=int(cfg["timesteps"]),
            backend=cfg["backend"],
            projections=cfg["projections"],
            suite=default_suite(),
            max_candidates=int(cfg["max_candidates"]),
            horizons=tuple(int(h) for h in cfg["horizons"]),
            hce_replicates=int(cfg["hce_replicates"]),
            initial_density=float(rec["rule"].initial_density),
        )

    t_sweep = time.time()
    with profiler.phase("sweep"):
        if int(cfg["n_workers"]) > 1 and len(tasks) > 1:
            per_cell = Parallel(
                n_jobs=int(cfg["n_workers"]),
                verbose=0,
                backend="loky",
            )(delayed(_task)(rec, seed) for rec, seed in tasks)
        else:
            per_cell = [_task(rec, seed) for rec, seed in tasks]
        for cell in per_cell:
            for proj_name, payload in cell.items():
                profiler.count("candidates", payload["n_candidates"])
                profiler.count("projections", 1)
            profiler.count("cells", 1)
    sweep_seconds = time.time() - t_sweep

    candidate_rows, cell_rows = _flatten_results(per_cell)

    # CSVs.
    print(f"\nWriting CSVs ({len(candidate_rows)} candidate rows, "
          f"{len(cell_rows)} cell rows)...")
    if candidate_rows:
        _write_candidate_metrics_csv(
            candidate_rows, out / "candidate_metrics.csv",
        )
    else:
        # Always write a header-only file for downstream stability.
        (out / "candidate_metrics.csv").write_text(
            "rule_id,rule_source,seed,projection,candidate_id,HCE\n",
            encoding="utf-8",
        )
    _write_simple_csv(cell_rows, out / "projection_summary.csv",
                      fields=["rule_id", "rule_source", "seed", "projection",
                              "n_candidates",
                              "projection_supports_threshold_margin",
                              "projection_output_kind"])

    # Aggregations
    summary = aggregate_per_projection(candidate_rows, cfg["projections"])
    summary["wall_time_seconds_sweep"] = sweep_seconds
    summary["n_cells"] = len(per_cell)
    summary["n_candidate_rows"] = len(candidate_rows)
    summary["projections_evaluated"] = list(cfg["projections"])

    # HCE by projection CSV (one row per projection).
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
    _write_simple_csv(hce_rows, out / "hce_by_projection.csv",
                      fields=["projection", "n_candidates_total",
                              "n_valid_hidden_invisible",
                              "n_invalid_hidden_invisible",
                              "mean_HCE", "mean_far_HCE",
                              "mean_hidden_vs_far_delta",
                              "mean_hidden_vs_sham_delta",
                              "mean_initial_projection_delta",
                              "fraction_clean_initial_projection"])

    # Mechanism by projection — Stage-5+ placeholder; emit an empty CSV
    # with documented columns so downstream readers don't crash.
    _write_simple_csv(
        [], out / "mechanism_by_projection.csv",
        fields=["projection", "boundary_and_interior_co_mediated_fraction",
                "global_chaotic_fraction", "threshold_mediated_fraction",
                "_status"]
    )
    # Note rows for the artifact audit.
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
    _write_simple_csv(audit_rows, out / "projection_artifact_audit.csv",
                      fields=["rule_id", "rule_source", "seed", "projection",
                              "n_candidates",
                              "n_valid_hidden_invisible_in_projection",
                              "n_invalid_hidden_invisible_in_projection",
                              "fraction_clean_initial_projection_in_projection",
                              "projection_supports_threshold_margin",
                              "projection_output_kind",
                              "mean_initial_projection_delta_for_projection"])

    # Stats summary JSON.
    (out / "stats_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o:
                   float(o) if isinstance(o, np.floating) else
                   (int(o) if isinstance(o, np.integer) else
                    (o.tolist() if isinstance(o, np.ndarray) else str(o)))),
        encoding="utf-8",
    )

    # Plots.
    try:
        write_all_plots(summary, candidate_rows, out / "plots")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plot generation failed: {e!r}")

    # Profiler.
    if args.profile:
        profiler.snapshot_memory("end_of_run")
        profiler.write_json(out / "perf_profile.json")

    # Markdown summary.
    write_summary_md(summary, out / "summary.md")

    print(f"\nDone in {sweep_seconds:.1f}s. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
