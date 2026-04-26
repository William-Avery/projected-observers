"""M6C — Hidden Organization Taxonomy CLI."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis.hidden_features import HIDDEN_FEATURE_NAMES
from observer_worlds.analysis.m6c_plots import write_all_m6c_plots
from observer_worlds.analysis.m6c_stats import (
    m6c_full_summary,
    render_m6c_summary_md,
)
from observer_worlds.experiments._m6c_taxonomy import (
    ABLATION_TYPES,
    M6CRow,
    run_m6c_taxonomy,
)
from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.search import FractionalRule


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M6C hidden-organization taxonomy.")
    p.add_argument("--rules-from", type=str, required=True)
    p.add_argument("--also-rules-from", type=str, default=None)
    p.add_argument("--n-rules", type=int, default=10)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=2000)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--max-candidates", type=int, default=30,
                   help="Top-K candidates by observer_score per (rule, seed).")
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    p.add_argument("--backend", choices=["numba", "numpy"], default="numba")
    p.add_argument("--n-cv-splits", type=int, default=5)
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m6c")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Process-parallelism: number of worker processes "
                        "for the (rule, seed) sweep. Default: cpu_count-2.")
    p.add_argument("--quick", action="store_true")
    return p


def _quick(args):
    if args.quick:
        args.n_rules = min(args.n_rules, 1)
        args.seeds = min(args.seeds, 1)
        args.timesteps = min(args.timesteps, 80)
        args.grid = [16, 16, 4, 4]
        args.max_candidates = min(args.max_candidates, 5)
        args.replicates = min(args.replicates, 2)
        args.horizons = [5, 10]
        args.n_cv_splits = 2
    return args


def _load_rules(path, n, tag):
    rules = load_top_rules(Path(path), n)
    return [(r, f"{tag}_rank{i:02d}", tag) for i, r in enumerate(rules, 1)]


def _infer_tag(path):
    s = str(path).lower()
    if "m4c" in s or "observer" in s or "evolve" in s:
        return "M4C_observer_optimized"
    if "m4a" in s or "viability" in s:
        return "M4A_viability"
    return "unknown"


def _write_features_csv(rows, path: Path) -> None:
    cols = (
        "rule_id", "rule_source", "seed", "candidate_id", "snapshot_t",
        *HIDDEN_FEATURE_NAMES,
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        # Take the median-horizon row per candidate so features are unique per candidate.
        seen = set()
        for r in rows:
            key = (r.rule_id, r.seed, r.candidate_id, r.snapshot_t)
            if key in seen: continue
            seen.add(key)
            row = list(key)
            for fn in HIDDEN_FEATURE_NAMES:
                row.append(f"{r.features.get(fn, 0.0):.6f}")
            w.writerow(row)


def _write_joined_csv(rows, path: Path) -> None:
    cols = (
        "rule_id", "rule_source", "seed", "candidate_id", "snapshot_t",
        "horizon", "candidate_area", "candidate_lifetime", "observer_score",
        "future_div_hidden_invisible", "local_div_hidden_invisible",
        "future_div_sham", "local_div_far_hidden",
        "hidden_vs_sham_delta", "hidden_vs_far_delta",
        "future_div_visible", "hidden_vs_visible_ratio", "survival_delta", "HCE",
        *HIDDEN_FEATURE_NAMES,
        *(f"ablation_{t}_future_div" for t in ABLATION_TYPES),
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            row = [
                r.rule_id, r.rule_source, r.seed, r.candidate_id, r.snapshot_t,
                r.horizon, f"{r.candidate_area:.2f}", r.candidate_lifetime,
                "" if r.observer_score is None else f"{r.observer_score:.4f}",
                f"{r.future_div_hidden_invisible:.6f}",
                f"{r.local_div_hidden_invisible:.6f}",
                f"{r.future_div_sham:.6f}",
                f"{r.local_div_far_hidden:.6f}",
                f"{r.hidden_vs_sham_delta:.6f}",
                f"{r.hidden_vs_far_delta:.6f}",
                f"{r.future_div_visible:.6f}",
                f"{r.hidden_vs_visible_ratio:.6f}",
                f"{r.survival_delta:.6f}",
                f"{r.HCE:.6f}",
            ]
            for fn in HIDDEN_FEATURE_NAMES:
                row.append(f"{r.features.get(fn, 0.0):.6f}")
            for t in ABLATION_TYPES:
                v = r.ablation_future_div.get(t)
                row.append("" if v is None else f"{v:.6f}")
            w.writerow(row)


def _write_correlation_csv(summary: dict, path: Path) -> None:
    cors = summary.get("correlations", [])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "outcome", "pearson_r", "spearman_r", "n"])
        for c in cors:
            w.writerow([c["feature"], c["outcome"],
                       f"{c['pearson_r']:.4f}", f"{c['spearman_r']:.4f}", c["n"]])


def _write_threshold_audit_csv(summary: dict, path: Path) -> None:
    audit = summary.get("threshold_audit", [])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filter", "n_candidates", "mean_future_div",
                    "mean_vs_sham", "mean_vs_far", "fraction_future_div_gt_zero"])
        for a in audit:
            w.writerow([a["filter"], a["n_candidates"],
                       f"{a['mean_future_div']:.6f}",
                       f"{a['mean_vs_sham']:.6f}",
                       f"{a['mean_vs_far']:.6f}",
                       f"{a['fraction_future_div_gt_zero']:.4f}"])


def _write_feature_importances_csv(summary: dict, path: Path) -> None:
    ms = summary.get("model_scores", [])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "outcome", "feature", "importance"])
        for m in ms:
            for feat, imp in m.get("feature_importances", {}).items():
                w.writerow([m["model"], m["outcome"], feat, f"{imp:.6f}"])


def _write_ablation_csv(rows, path: Path, *, horizon: int) -> None:
    cols = ["rule_id", "rule_source", "seed", "candidate_id", "snapshot_t",
            *(f"ablation_{t}_future_div" for t in ABLATION_TYPES)]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            if r.horizon != horizon: continue
            if not r.ablation_future_div: continue
            row = [r.rule_id, r.rule_source, r.seed, r.candidate_id, r.snapshot_t]
            for t in ABLATION_TYPES:
                v = r.ablation_future_div.get(t)
                row.append("" if v is None else f"{v:.6f}")
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_sims"
    workdir.mkdir(parents=True, exist_ok=True)

    rules = _load_rules(args.rules_from, args.n_rules, _infer_tag(args.rules_from))
    if args.also_rules_from:
        rules.extend(_load_rules(args.also_rules_from, args.n_rules,
                                _infer_tag(args.also_rules_from)))
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    grid = tuple(args.grid)

    cfg_dump = {
        "rules_from": args.rules_from,
        "also_rules_from": args.also_rules_from,
        "n_rules": len(rules),
        "seeds": seeds,
        "timesteps": args.timesteps,
        "grid": list(grid),
        "max_candidates": args.max_candidates,
        "replicates": args.replicates,
        "horizons": args.horizons,
        "backend": args.backend,
        "n_cv_splits": args.n_cv_splits,
        "rules": [{"rule_id": rid, "rule_source": rs, "rule": r.to_dict()}
                 for r, rid, rs in rules],
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"M6C -> {out_dir}")
    print(f"  rules={len(rules)} seeds={len(seeds)} T={args.timesteps} "
          f"grid={grid} backend={args.backend} horizons={args.horizons}")

    t0 = time.time()
    rows = run_m6c_taxonomy(
        rules=rules, seeds=seeds, grid_shape=grid, timesteps=args.timesteps,
        max_candidates=args.max_candidates, horizons=args.horizons,
        n_replicates=args.replicates, backend=args.backend,
        workdir=workdir, progress=print,
        n_workers=args.n_workers,
    )
    elapsed = time.time() - t0
    print(f"\nGenerated {len(rows)} rows in {elapsed:.0f}s")

    headline_h = args.horizons[len(args.horizons) // 2]
    print("writing CSVs...")
    _write_features_csv(rows, out_dir / "hidden_features.csv")
    _write_joined_csv(rows, out_dir / "hce_joined_features.csv")
    _write_ablation_csv(rows, out_dir / "ablation_results.csv",
                       horizon=headline_h)

    print("computing stats...")
    summary = m6c_full_summary(rows, horizons=args.horizons,
                              n_splits=args.n_cv_splits, seed=args.base_seed)
    (out_dir / "stats_summary.json").write_text(
        json.dumps(summary, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                   else (int(o) if isinstance(o, np.integer)
                         else (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    )
    _write_correlation_csv(summary, out_dir / "correlation_table.csv")
    _write_threshold_audit_csv(summary, out_dir / "threshold_audit.csv")
    _write_feature_importances_csv(summary, out_dir / "feature_importances.csv")
    (out_dir / "model_scores.json").write_text(
        json.dumps(summary.get("model_scores", []), indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                   else (int(o) if isinstance(o, np.integer)
                         else str(o)))
    )

    print("writing plots...")
    write_all_m6c_plots(rows, summary.get("model_scores", []),
                       summary.get("threshold_audit", []),
                       plots_dir, horizon=headline_h)

    print("writing summary.md...")
    md = render_m6c_summary_md(summary)
    (out_dir / "summary.md").write_text(md)

    print(f"\nDone. Run dir: {out_dir}")
    print(f"Headline horizon: {headline_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
