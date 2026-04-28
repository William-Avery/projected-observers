"""Follow-up Topic 2 — hidden identity swap experiment runner (Stage 3).

Discovers candidates per (rule, seed), pairs them across cells, runs
the hybrid-state swap intervention, and writes the documented output
bundle:

    config.json
    frozen_manifest.json
    candidate_pairs.csv
    swap_interventions.csv
    identity_scores.csv
    stats_summary.json
    summary.md
    plots/

Stage 3 is a smoke-only build. Production sweeps are Stage 5+.
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

from observer_worlds.analysis.identity_swap_plots import write_all_plots
from observer_worlds.analysis.identity_swap_stats import (
    aggregate_identity_results, write_summary_md,
)
from observer_worlds.experiments._followup_identity_swap import (
    SUPPORTED_MATCH_MODES, IdentityPairResult,
    discover_candidates_for_cell, find_candidate_pairs, measure_pair,
)
from observer_worlds.search.rules import FractionalRule


REPO = Path(__file__).resolve().parents[2]

MATCHING_MODES = SUPPORTED_MATCH_MODES + (
    "morphology_nearest", "observer_score_bin", "mechanism_class",
    "strict_projection_equal",
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Follow-up Topic 2: hidden identity swap experiment.",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--backend", type=str, default=None,
                   choices=["numpy", "numba", "cupy", "cuda-batched"])
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap on number of swap pairs per source.")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str, default="followup_hidden_identity_swap")
    p.add_argument("--n-rules-per-source", type=int, default=None)
    p.add_argument("--test-seeds", type=int, nargs="+", default=None)
    p.add_argument("--matching-mode", type=str, default=None,
                   choices=MATCHING_MODES,
                   help="How to pair candidates A, B.")
    p.add_argument("--projection", "--projection-name", dest="projection",
                   type=str, default=None,
                   help="Single projection used for the swap. Stage 3 "
                        "supports one projection per run.")
    p.add_argument("--projection-tolerance", type=float, default=None,
                   help="Max allowed projection-preservation error at t=0.")
    p.add_argument("--min-visible-similarity", type=float, default=None,
                   help="Reject pairs whose translation-aligned visible "
                        "similarity is below this threshold (combined "
                        "score over IoU + area_ratio + bbox_aspect).")
    p.add_argument("--hce-replicates", type=int, default=None)
    p.add_argument("--grid", type=int, nargs=4, default=None,
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--rules-json", type=Path,
                   default=REPO / "release" / "rules" / "m7_top_hce_rules.json")
    p.add_argument("--profile", action="store_true")
    return p


def _full_defaults() -> dict:
    return {
        "n_workers": max(1, (os.cpu_count() or 2) - 2),
        "backend": "numpy",
        "max_candidates": 20,
        "max_pairs": 100,
        "timesteps": 500,
        "horizons": [1, 2, 3, 5, 10, 20, 40, 80],
        "n_rules_per_source": 5,
        "test_seeds": list(range(6000, 6020)),
        "matching_mode": "feature_nearest",
        "projection": "mean_threshold",
        "projection_tolerance": 1e-6,
        "hce_replicates": 3,
        "grid": [64, 64, 8, 8],
        "min_visible_similarity": 0.30,
    }


def _smoke_defaults() -> dict:
    # Stage 3 swap needs candidates from at least two different cells
    # (different_cell=True). 5 seeds provides headroom for seed-specific
    # candidate-collapse without leaving no cross-cell pairs.
    return {
        "n_rules_per_source": 1,
        "test_seeds": [6000, 6001, 6002, 6003, 6004],
        "timesteps": 100,
        "max_candidates": 10,
        "max_pairs": 10,
        "horizons": [5, 10],
        "hce_replicates": 1,
        "grid": [16, 16, 4, 4],
    }


def _resolve_config(args: argparse.Namespace) -> dict:
    cfg = dict(_full_defaults())
    if args.quick:
        cfg.update(_smoke_defaults())
    for key in ("n_workers", "backend", "max_candidates", "max_pairs",
                "timesteps", "horizons", "n_rules_per_source", "test_seeds",
                "matching_mode", "projection", "projection_tolerance",
                "hce_replicates", "grid", "min_visible_similarity"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = list(v) if isinstance(v, list) else v
    cfg["rules_json"] = str(args.rules_json)
    if cfg["matching_mode"] not in SUPPORTED_MATCH_MODES:
        raise SystemExit(
            f"matching_mode {cfg['matching_mode']!r} not yet implemented in "
            f"Stage 3; supported: {SUPPORTED_MATCH_MODES}"
        )
    return cfg


def _make_out_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_root / f"{args.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def _build_frozen_manifest(cfg: dict) -> dict:
    def _safe(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(REPO), timeout=10)
            return r.stdout.strip()
        except Exception:
            return ""
    return {
        "stage": 3,
        "experiment": "followup_hidden_identity_swap",
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
        "git": {
            "commit": _safe(["git", "rev-parse", "HEAD"]),
            "branch": _safe(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_safe(["git", "status", "--porcelain"])),
        },
        "config": cfg,
        "platform": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
    }


def _load_rules(path: Path, n: int) -> list[FractionalRule]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FractionalRule.from_dict(r) for r in raw[:int(n)]]


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_simple_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _pair_summary_row(p: IdentityPairResult, direction: str) -> dict:
    """Per (pair, direction) row for swap_interventions.csv."""
    if direction == "A_with_B_hidden":
        return {
            "pair_id": p.pair_id,
            "direction": direction,
            "rule_id": p.rule_id,
            "rule_source": p.rule_source,
            "seed_host": p.seed_a,
            "seed_donor": p.seed_b,
            "projection": p.projection_name,
            "candidate_host_id": p.candidate_a_id,
            "candidate_donor_id": p.candidate_b_id,
            "match_mode": p.match_mode,
            "match_distance": p.match_distance,
            "visible_similarity": p.visible_similarity,
            "area_host": p.area_a,
            "area_donor": p.area_b,
            "hidden_distance": p.hidden_distance,
            "n_cells_in_mask": p.n_cells_in_mask_a,
            "n_cells_swapped": p.n_cells_swapped_a,
            "projection_preservation_error": p.projection_preservation_error_a,
            "valid_swap": p.valid_swap_a,
            "invalid_reason": p.invalid_reason_a,
        }
    return {
        "pair_id": p.pair_id,
        "direction": direction,
        "rule_id": p.rule_id,
        "rule_source": p.rule_source,
        "seed_host": p.seed_b,
        "seed_donor": p.seed_a,
        "projection": p.projection_name,
        "candidate_host_id": p.candidate_b_id,
        "candidate_donor_id": p.candidate_a_id,
        "match_mode": p.match_mode,
        "match_distance": p.match_distance,
        "visible_similarity": p.visible_similarity,
        "area_host": p.area_b,
        "area_donor": p.area_a,
        "hidden_distance": p.hidden_distance,
        "n_cells_in_mask": p.n_cells_in_mask_b,
        "n_cells_swapped": p.n_cells_swapped_b,
        "projection_preservation_error": p.projection_preservation_error_b,
        "valid_swap": p.valid_swap_b,
        "invalid_reason": p.invalid_reason_b,
    }


def _identity_score_rows(results: list[IdentityPairResult]) -> list[dict]:
    rows = []
    for p in results:
        for direction, host_per_h, donor_per_h, pull_per_h, valid in (
            ("A_with_B_hidden",
             p.host_similarity_a_per_h, p.donor_similarity_a_per_h,
             p.hidden_identity_pull_a_per_h, p.valid_swap_a),
            ("B_with_A_hidden",
             p.host_similarity_b_per_h, p.donor_similarity_b_per_h,
             p.hidden_identity_pull_b_per_h, p.valid_swap_b),
        ):
            for h, hs, ds, pl in zip(p.horizons, host_per_h, donor_per_h,
                                     pull_per_h):
                rows.append({
                    "pair_id": p.pair_id,
                    "direction": direction,
                    "rule_id": p.rule_id,
                    "rule_source": p.rule_source,
                    "projection": p.projection_name,
                    "match_mode": p.match_mode,
                    "horizon": int(h),
                    "valid_swap": valid,
                    "host_similarity": hs,
                    "donor_similarity": ds,
                    "hidden_identity_pull": pl,
                })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = _resolve_config(args)
    out = _make_out_dir(args)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "frozen_manifest.json").write_text(
        json.dumps(_build_frozen_manifest(cfg), indent=2), encoding="utf-8",
    )
    rules = _load_rules(args.rules_json, cfg["n_rules_per_source"])
    rule_records = []
    for i, r in enumerate(rules):
        rid = f"M7_HCE_optimized_rank{i+1:02d}"
        rule_records.append({"rule": r, "rule_id": rid,
                              "rule_source": "M7_HCE_optimized"})

    print("=" * 72)
    print("Follow-up Topic 2: hidden identity swap — Stage 3")
    print("=" * 72)
    print(f"  out             = {out}")
    print(f"  backend         = {cfg['backend']}")
    print(f"  n_workers       = {cfg['n_workers']}")
    print(f"  rules           = {len(rule_records)}")
    print(f"  seeds           = {len(cfg['test_seeds'])} ({cfg['test_seeds'][0]}..{cfg['test_seeds'][-1]})")
    print(f"  timesteps       = {cfg['timesteps']}")
    print(f"  grid            = {cfg['grid']}")
    print(f"  projection      = {cfg['projection']}")
    print(f"  matching_mode   = {cfg['matching_mode']}")
    print(f"  max_candidates  = {cfg['max_candidates']}")
    print(f"  max_pairs       = {cfg['max_pairs']}")
    print(f"  horizons        = {cfg['horizons']}")
    print(f"  proj_tolerance  = {cfg['projection_tolerance']}")
    print(f"  min_visible_sim = {cfg['min_visible_similarity']}")

    t_total = time.time()

    # Discovery phase: candidates per (rule, seed).
    discovery_tasks = [
        (rec, seed)
        for rec in rule_records
        for seed in cfg["test_seeds"]
    ]
    def _discover(rec, seed):
        return discover_candidates_for_cell(
            rule_bs=rec["rule"].to_bsrule(),
            rule_id=rec["rule_id"], rule_source=rec["rule_source"],
            seed=int(seed),
            grid_shape=tuple(cfg["grid"]),
            timesteps=int(cfg["timesteps"]),
            backend=cfg["backend"],
            projection_name=cfg["projection"],
            max_candidates=int(cfg["max_candidates"]),
            initial_density=float(rec["rule"].initial_density),
        )
    if int(cfg["n_workers"]) > 1 and len(discovery_tasks) > 1:
        per_cell_candidates = Parallel(
            n_jobs=int(cfg["n_workers"]), verbose=0, backend="loky",
        )(delayed(_discover)(rec, seed) for rec, seed in discovery_tasks)
    else:
        per_cell_candidates = [_discover(rec, seed)
                               for rec, seed in discovery_tasks]
    all_candidates = [c for cell in per_cell_candidates for c in cell]
    print(f"\nDiscovered {len(all_candidates)} candidates across "
          f"{len(per_cell_candidates)} cells.")

    # Pairing per rule (different seeds within same rule).
    by_rule: dict[str, list] = {}
    for c in all_candidates:
        by_rule.setdefault(c.rule_id, []).append(c)
    all_pairs = []
    for rid, cands in by_rule.items():
        pairs = find_candidate_pairs(
            cands, match_mode=cfg["matching_mode"],
            max_pairs=int(cfg["max_pairs"]),
        )
        all_pairs.extend(pairs)
    print(f"Formed {len(all_pairs)} candidate pairs.")

    # Measurement phase.
    rule_bs_by_id = {rec["rule_id"]: rec["rule"].to_bsrule()
                      for rec in rule_records}
    def _measure(args_tuple):
        pair_id, (A, B, meta) = args_tuple
        return measure_pair(
            pair_id=pair_id, A=A, B=B, match_meta=meta,
            match_mode=cfg["matching_mode"],
            projection_name=cfg["projection"],
            horizons=cfg["horizons"],
            rule_bs=rule_bs_by_id[A.rule_id],
            backend=cfg["backend"],
            min_visible_similarity=float(cfg["min_visible_similarity"]),
        )
    results: list[IdentityPairResult] = []
    if int(cfg["n_workers"]) > 1 and len(all_pairs) > 1:
        results = Parallel(
            n_jobs=int(cfg["n_workers"]), verbose=0, backend="loky",
        )(delayed(_measure)((i, p)) for i, p in enumerate(all_pairs))
    else:
        results = [_measure((i, p)) for i, p in enumerate(all_pairs)]

    # CSVs.
    pair_rows = []
    swap_rows = []
    for p in results:
        pair_rows.append({
            "pair_id": p.pair_id, "rule_id": p.rule_id,
            "rule_source": p.rule_source,
            "seed_a": p.seed_a, "seed_b": p.seed_b,
            "projection": p.projection_name,
            "candidate_a_id": p.candidate_a_id,
            "candidate_b_id": p.candidate_b_id,
            "match_mode": p.match_mode,
            "match_distance": p.match_distance,
            "visible_similarity": p.visible_similarity,
            "area_a": p.area_a, "area_b": p.area_b,
            "hidden_distance": p.hidden_distance,
            "n_cells_in_mask_a": p.n_cells_in_mask_a,
            "n_cells_swapped_a": p.n_cells_swapped_a,
            "projection_preservation_error_a": p.projection_preservation_error_a,
            "valid_swap_a": p.valid_swap_a,
            "invalid_reason_a": p.invalid_reason_a,
            "n_cells_in_mask_b": p.n_cells_in_mask_b,
            "n_cells_swapped_b": p.n_cells_swapped_b,
            "projection_preservation_error_b": p.projection_preservation_error_b,
            "valid_swap_b": p.valid_swap_b,
            "invalid_reason_b": p.invalid_reason_b,
        })
        swap_rows.append(_pair_summary_row(p, "A_with_B_hidden"))
        swap_rows.append(_pair_summary_row(p, "B_with_A_hidden"))

    _write_simple_csv(pair_rows, out / "candidate_pairs.csv", fields=[
        "pair_id", "rule_id", "rule_source", "seed_a", "seed_b",
        "projection", "candidate_a_id", "candidate_b_id",
        "match_mode", "match_distance", "visible_similarity",
        "area_a", "area_b", "hidden_distance",
        "n_cells_in_mask_a", "n_cells_swapped_a",
        "projection_preservation_error_a", "valid_swap_a",
        "invalid_reason_a",
        "n_cells_in_mask_b", "n_cells_swapped_b",
        "projection_preservation_error_b", "valid_swap_b",
        "invalid_reason_b",
    ])
    _write_simple_csv(swap_rows, out / "swap_interventions.csv", fields=[
        "pair_id", "direction", "rule_id", "rule_source",
        "seed_host", "seed_donor", "projection",
        "candidate_host_id", "candidate_donor_id",
        "match_mode", "match_distance", "visible_similarity",
        "area_host", "area_donor", "hidden_distance",
        "n_cells_in_mask", "n_cells_swapped",
        "projection_preservation_error", "valid_swap", "invalid_reason",
    ])
    score_rows = _identity_score_rows(results)
    _write_simple_csv(score_rows, out / "identity_scores.csv", fields=[
        "pair_id", "direction", "rule_id", "rule_source",
        "projection", "match_mode", "horizon", "valid_swap",
        "host_similarity", "donor_similarity", "hidden_identity_pull",
    ])

    summary = aggregate_identity_results(results, score_rows)
    summary["wall_time_seconds"] = float(time.time() - t_total)
    summary["n_pairs_attempted"] = len(results)
    summary["n_candidates_total"] = len(all_candidates)
    summary["projection"] = cfg["projection"]
    summary["matching_mode"] = cfg["matching_mode"]

    (out / "stats_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o:
                   float(o) if isinstance(o, np.floating) else
                   (int(o) if isinstance(o, np.integer) else
                    (o.tolist() if isinstance(o, np.ndarray) else str(o)))),
        encoding="utf-8",
    )
    write_summary_md(summary, out / "summary.md")
    try:
        write_all_plots(summary, score_rows, out / "plots")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plot generation failed: {e!r}")

    print(f"\nDone in {summary['wall_time_seconds']:.1f}s. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
