"""M6B — Hidden Causal Dependence replication driver."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis.m6b_plots import write_all_m6b_plots
from observer_worlds.analysis.m6b_stats import (
    m6b_full_summary,
    render_m6b_summary_md,
)
from observer_worlds.experiments._m6b_replication import (
    M6BRow,
    run_m6b_replication,
)
from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.search import FractionalRule


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M6B Hidden-Causal-Dependence replication.")
    p.add_argument("--rules-from", type=str, required=True,
                   help="Primary rule source (M4A or M4C leaderboard.json).")
    p.add_argument("--also-rules-from", type=str, default=None,
                   help="Optional second rule source merged with the first.")
    p.add_argument("--n-rules", type=int, default=10,
                   help="Top-N rules from each source.")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=2000,
                   help="Held-out base seed (avoid 1000 used by M4C).")
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--max-candidates", type=int, default=20,
                   help="Max candidates PER selection mode (3 modes total).")
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    p.add_argument("--backend", choices=["numba", "numpy"], default="numba")
    p.add_argument("--no-per-step-shuffled", action="store_true",
                   help="Skip the per-step-shuffled condition (faster smoke).")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m6b")
    p.add_argument("--quick", action="store_true",
                   help="Reduce defaults: n_rules=1, seeds=1, T=80, "
                        "grid 16x16x4x4, max_candidates=3, replicates=2, "
                        "horizons=[5,10], no per_step_shuffled.")
    return p


def _quick(args):
    if args.quick:
        args.n_rules = min(args.n_rules, 1)
        args.seeds = min(args.seeds, 1)
        args.timesteps = min(args.timesteps, 80)
        args.grid = [16, 16, 4, 4]
        args.max_candidates = min(args.max_candidates, 3)
        args.replicates = min(args.replicates, 2)
        args.horizons = [5, 10]
        args.no_per_step_shuffled = True
        args.n_bootstrap = min(args.n_bootstrap, 500)
    return args


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


def _load_rules_with_source(path: str | Path, n_rules: int, source_tag: str
                            ) -> list[tuple[FractionalRule, str, str]]:
    rules = load_top_rules(Path(path), n_rules)
    out = []
    for i, r in enumerate(rules, start=1):
        out.append((r, f"{source_tag}_rank{i:02d}", source_tag))
    return out


def _infer_source_tag(path: str) -> str:
    s = str(path).lower()
    if "m4c" in s or "observer" in s or "evolve" in s:
        return "M4C_observer_optimized"
    if "m4a" in s or "viability" in s:
        return "M4A_viability"
    return "unknown"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


CSV_COLS = (
    "rule_id", "rule_source", "seed", "candidate_id",
    "candidate_selection_mode", "condition", "intervention_type",
    "replicate", "horizon",
    "initial_projection_delta", "future_projection_divergence",
    "local_future_divergence", "global_future_divergence",
    "hidden_causal_dependence", "hidden_vs_visible_ratio",
    "hidden_vs_sham_delta", "hidden_vs_far_delta",
    "survival_original", "survival_intervened", "survival_delta",
    "trajectory_divergence", "recovery_delta",
    "candidate_area", "candidate_lifetime", "observer_score",
    "morphology_class", "n_flips_applied", "flip_fraction_for_visible",
    "snapshot_t",
)


def _write_rows_csv(rows: list[M6BRow], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in rows:
            d = asdict(r)
            row = []
            for c in CSV_COLS:
                v = d.get(c)
                if v is None:
                    row.append("")
                elif isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(v)
            w.writerow(row)


def _write_candidate_summary_csv(rows: list[M6BRow], path: Path,
                                 *, headline_horizon: int) -> None:
    """One row per (rule, seed, candidate, intervention) with mean across
    replicates at the headline horizon. Lighter than the full raw CSV."""
    by_key: dict = {}
    for r in rows:
        if r.horizon != headline_horizon:
            continue
        key = (r.rule_id, r.seed, r.candidate_id, r.condition,
               r.intervention_type)
        by_key.setdefault(key, []).append(r)
    cols = (
        "rule_id", "rule_source", "seed", "candidate_id",
        "candidate_selection_mode", "condition", "intervention_type",
        "n_replicates", "horizon",
        "mean_initial_projection_delta", "mean_future_projection_divergence",
        "mean_local_future_divergence", "mean_hidden_vs_sham_delta",
        "mean_hidden_vs_far_delta", "mean_hidden_vs_visible_ratio",
        "mean_HCE", "mean_survival_delta",
        "candidate_area", "candidate_lifetime", "observer_score",
        "morphology_class",
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, rs in sorted(by_key.items()):
            r0 = rs[0]
            w.writerow([
                r0.rule_id, r0.rule_source, r0.seed, r0.candidate_id,
                r0.candidate_selection_mode, r0.condition,
                r0.intervention_type, len(rs), headline_horizon,
                f"{np.mean([r.initial_projection_delta for r in rs]):.6f}",
                f"{np.mean([r.future_projection_divergence for r in rs]):.6f}",
                f"{np.mean([r.local_future_divergence for r in rs]):.6f}",
                f"{np.mean([r.hidden_vs_sham_delta for r in rs]):.6f}",
                f"{np.mean([r.hidden_vs_far_delta for r in rs]):.6f}",
                f"{np.mean([r.hidden_vs_visible_ratio for r in rs]):.6f}",
                f"{np.mean([r.hidden_causal_dependence for r in rs]):.6f}",
                f"{np.mean([r.survival_delta for r in rs]):.6f}",
                f"{r0.candidate_area:.2f}",
                r0.candidate_lifetime,
                "" if r0.observer_score is None else f"{r0.observer_score:.4f}",
                r0.morphology_class,
            ])


def _write_condition_summary_csv(rows, path: Path, *, horizon: int) -> None:
    """One row per (condition, intervention) with summary stats at horizon."""
    by_key: dict = {}
    for r in rows:
        if r.horizon != horizon: continue
        by_key.setdefault((r.condition, r.intervention_type), []).append(r)
    cols = (
        "condition", "intervention_type", "horizon", "n_rows",
        "mean_initial_projection_delta", "mean_future_div",
        "mean_local_div", "mean_vs_sham", "mean_vs_far",
        "mean_HCE", "mean_survival_delta",
        "mean_n_flips_applied",
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (cond, intv), rs in sorted(by_key.items()):
            w.writerow([
                cond, intv, horizon, len(rs),
                f"{np.mean([r.initial_projection_delta for r in rs]):.6f}",
                f"{np.mean([r.future_projection_divergence for r in rs]):.6f}",
                f"{np.mean([r.local_future_divergence for r in rs]):.6f}",
                f"{np.mean([r.hidden_vs_sham_delta for r in rs]):.6f}",
                f"{np.mean([r.hidden_vs_far_delta for r in rs]):.6f}",
                f"{np.mean([r.hidden_causal_dependence for r in rs]):.6f}",
                f"{np.mean([r.survival_delta for r in rs]):.6f}",
                f"{np.mean([r.n_flips_applied for r in rs]):.2f}",
            ])


def _write_paired_diffs_csv(rows, path: Path, *, horizon: int) -> None:
    """Per (rule, seed, candidate) paired differences across key intervention pairs."""
    pairs = [
        ("hidden_invisible_local", "sham"),
        ("hidden_invisible_local", "hidden_invisible_far"),
        ("hidden_invisible_local", "visible_match_count"),
        ("one_time_scramble_local", "hidden_invisible_local"),
        ("fiber_replacement_local", "hidden_invisible_local"),
    ]
    by_key: dict = {}
    for r in rows:
        if r.condition != "coherent_4d" or r.horizon != horizon: continue
        key = (r.rule_id, r.seed, r.candidate_id)
        by_key.setdefault(key, {}).setdefault(r.intervention_type, []).append(r)
    cols = ["rule_id", "seed", "candidate_id"]
    for a, b in pairs:
        cols.extend([f"{a}_minus_{b}_future", f"{a}_minus_{b}_local"])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, by_intv in sorted(by_key.items()):
            row = list(key)
            for a, b in pairs:
                if a not in by_intv or b not in by_intv:
                    row.extend(["", ""]); continue
                fa = np.mean([r.future_projection_divergence for r in by_intv[a]])
                fb = np.mean([r.future_projection_divergence for r in by_intv[b]])
                la = np.mean([r.local_future_divergence for r in by_intv[a]])
                lb = np.mean([r.local_future_divergence for r in by_intv[b]])
                row.extend([f"{fa-fb:.6f}", f"{la-lb:.6f}"])
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Rule loading.
    rules = _load_rules_with_source(
        args.rules_from, args.n_rules, _infer_source_tag(args.rules_from)
    )
    if args.also_rules_from:
        rules.extend(_load_rules_with_source(
            args.also_rules_from, args.n_rules, _infer_source_tag(args.also_rules_from)
        ))

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    grid = tuple(args.grid)

    cfg_dump = {
        "rules_from": args.rules_from,
        "also_rules_from": args.also_rules_from,
        "n_rules_per_source": args.n_rules,
        "n_total_rules": len(rules),
        "seeds": seeds,
        "timesteps": args.timesteps,
        "grid": list(grid),
        "max_candidates": args.max_candidates,
        "replicates": args.replicates,
        "horizons": args.horizons,
        "backend": args.backend,
        "include_per_step_shuffled": not args.no_per_step_shuffled,
        "rules": [{"rule_id": rid, "rule_source": rs, "rule": r.to_dict()}
                  for r, rid, rs in rules],
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"M6B -> {out_dir}")
    print(f"  rules={len(rules)} seeds={len(seeds)} T={args.timesteps} "
          f"grid={grid} backend={args.backend} horizons={args.horizons}")
    workdir = out_dir / "_sims"
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = run_m6b_replication(
        rules=rules, seeds=seeds, grid_shape=grid,
        timesteps=args.timesteps,
        max_candidates_per_mode=args.max_candidates,
        horizons=args.horizons, n_replicates=args.replicates,
        backend=args.backend,
        include_per_step_shuffled=not args.no_per_step_shuffled,
        workdir_for_zarr=workdir,
        progress=print,
    )
    elapsed = time.time() - t0
    print(f"\nGenerated {len(rows)} rows in {elapsed:.0f}s")

    # Outputs.
    headline_h = args.horizons[len(args.horizons) // 2]
    print("writing CSVs...")
    _write_rows_csv(rows, out_dir / "hidden_interventions_raw.csv")
    _write_candidate_summary_csv(rows, out_dir / "candidate_hidden_dependence.csv",
                                 headline_horizon=headline_h)
    _write_condition_summary_csv(rows, out_dir / "condition_summary.csv",
                                 horizon=headline_h)
    _write_paired_diffs_csv(rows, out_dir / "paired_differences.csv",
                            horizon=headline_h)

    print("computing stats...")
    summary = m6b_full_summary(rows, horizons=args.horizons,
                               n_boot=args.n_bootstrap, seed=args.base_seed)
    (out_dir / "stats_summary.json").write_text(
        json.dumps(summary, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                   else (int(o) if isinstance(o, np.integer)
                         else (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    )

    print("writing plots...")
    write_all_m6b_plots(rows, plots_dir, horizon=headline_h)

    print("writing summary.md...")
    md = render_m6b_summary_md(summary)
    (out_dir / "summary.md").write_text(md)

    print(f"\nDone. Run dir: {out_dir}")
    print(f"Headline horizon: {headline_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
