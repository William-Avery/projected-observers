"""M8 — mechanism-discovery CLI."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis.m8_plots import write_all_m8_plots
from observer_worlds.analysis.m8_stats import (
    m8_full_summary,
    render_m8_summary_md,
)
from observer_worlds.experiments._m8_mechanism import (
    M8CandidateResult,
    run_m8_mechanism_discovery,
)
from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.experiments.run_m7b_production_holdout import (
    build_frozen_manifest, _autodetect_m7_seed_splits, check_seed_disjointness,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M8 mechanism discovery.")
    p.add_argument("--m7-rules", type=str, required=True)
    p.add_argument("--m4c-rules", type=str, default=None)
    p.add_argument("--m4a-rules", type=str, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=5)
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=list(range(6000, 6020)))
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--backend", choices=["numba", "numpy"], default="numpy")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m8")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Process-parallelism: number of worker processes "
                        "for the (rule, seed) sweep. Default: cpu_count-2.")
    return p


def _quick(args):
    if args.quick:
        args.n_rules_per_source = min(args.n_rules_per_source, 1)
        args.test_seeds = args.test_seeds[:2]
        args.timesteps = min(args.timesteps, 80)
        args.grid = [16, 16, 4, 4]
        args.max_candidates = min(args.max_candidates, 4)
        args.hce_replicates = min(args.hce_replicates, 1)
        args.horizons = [1, 2, 5, 10]
    return args


def _load_with_source(path, n, tag):
    rules = load_top_rules(Path(path), n)
    return [(r, f"{tag}_rank{i:02d}", tag) for i, r in enumerate(rules, 1)]


def _infer_tag(path):
    s = str(path).lower()
    if "m7" in s: return "M7_HCE_optimized"
    if "m4c" in s or "observer" in s: return "M4C_observer_optimized"
    if "m4a" in s or "viability" in s: return "M4A_viability"
    return "unknown"


def _write_mechanism_labels_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "mechanism", "confidence", "near_threshold_fraction",
            "boundary_response_fraction", "interior_response_fraction",
            "environment_response_fraction", "first_visible_effect_time",
            "hidden_to_visible_conversion_time",
            "boundary_mediation_index", "candidate_locality_index",
            "far_hidden_effect", "candidate_lifetime", "observer_score")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.mechanism.label, f"{r.mechanism.confidence:.3f}",
                f"{r.near_threshold_fraction:.4f}",
                f"{r.response_map.boundary_response_fraction:.4f}",
                f"{r.response_map.interior_response_fraction:.4f}",
                f"{r.response_map.environment_response_fraction:.4f}",
                r.timing.first_visible_effect_time,
                r.pathway.hidden_to_visible_conversion_time,
                f"{r.mediation.boundary_mediation_index:.4f}",
                f"{r.mediation.candidate_locality_index:.4f}",
                f"{r.mediation.far_hidden_effect:.4f}",
                r.candidate_lifetime,
                "" if r.observer_score is None else f"{r.observer_score:.4f}",
            ])


def _write_mediation_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "interior_hidden_effect", "boundary_hidden_effect",
            "environment_hidden_effect", "far_hidden_effect",
            "visible_boundary_effect", "visible_environment_effect",
            "boundary_mediation_index", "candidate_locality_index")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            m = r.mediation
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                f"{m.interior_hidden_effect:.6f}",
                f"{m.boundary_hidden_effect:.6f}",
                f"{m.environment_hidden_effect:.6f}",
                f"{m.far_hidden_effect:.6f}",
                f"{m.visible_boundary_effect:.6f}",
                f"{m.visible_environment_effect:.6f}",
                f"{m.boundary_mediation_index:.4f}",
                f"{m.candidate_locality_index:.6f}",
            ])


def _write_pathway_traces_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id", "step",
            "hidden_mass", "visible_mass", "spread_radius_4d", "spread_radius_2d")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            for t in range(r.pathway.n_steps):
                w.writerow([
                    r.rule_id, r.rule_source, r.seed, r.candidate_id, t + 1,
                    r.pathway.hidden_mass_per_step[t],
                    r.pathway.visible_mass_per_step[t],
                    f"{r.pathway.spread_radius_4d[t]:.4f}"
                    if t < len(r.pathway.spread_radius_4d) else "",
                    f"{r.pathway.spread_radius_2d[t]:.4f}"
                    if t < len(r.pathway.spread_radius_2d) else "",
                ])


def _write_lifetime_tradeoff_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "candidate_lifetime", "HCE_at_headline_horizon",
            "boundary_response_fraction", "near_threshold_fraction",
            "candidate_locality_index", "first_visible_effect_time")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            h = r.timing.horizons
            hce = r.timing.full_grid_l1_per_horizon[len(h) // 2] if h else 0.0
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.candidate_lifetime,
                f"{hce:.6f}",
                f"{r.response_map.boundary_response_fraction:.4f}",
                f"{r.near_threshold_fraction:.4f}",
                f"{r.mediation.candidate_locality_index:.6f}",
                r.timing.first_visible_effect_time,
            ])


def _write_response_maps_csv(results, path):
    """One row per (candidate, x, y) with the response value."""
    cols = ("rule_id", "rule_source", "seed", "candidate_id", "horizon",
            "x", "y", "response_value", "is_interior")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            rg = r.response_map.response_grid
            mask = r.response_map.interior_mask
            Nx, Ny = rg.shape
            for x in range(Nx):
                for y in range(Ny):
                    if rg[x, y] != 0 or mask[x, y]:
                        w.writerow([
                            r.rule_id, r.rule_source, r.seed, r.candidate_id,
                            r.response_map.horizon, x, y,
                            f"{rg[x, y]:.6f}", int(mask[x, y]),
                        ])


def _write_feature_dynamics_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "leading_feature", "lag", "lead_lag_correlation")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            for (fn, lag, corr) in r.feature_dynamics.leading_features:
                w.writerow([
                    r.rule_id, r.rule_source, r.seed, r.candidate_id,
                    fn, lag, f"{corr:.4f}",
                ])


def _write_response_arrays(results, arrays_dir):
    """Save each response_map.response_grid as .npy under arrays/."""
    arrays_dir = Path(arrays_dir); arrays_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fname = f"response_{r.rule_source}_{r.rule_id}_seed{r.seed}_track{r.candidate_id}.npy"
        np.save(arrays_dir / fname, r.response_map.response_grid)


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = out_dir / "arrays"; arrays_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_sims"; workdir.mkdir(parents=True, exist_ok=True)

    # Frozen manifest + seed disjointness.
    autodetected = _autodetect_m7_seed_splits(args.m7_rules)

    class _ManifestArgs:
        pass
    margs = _ManifestArgs()
    margs.m7_rules = args.m7_rules
    margs.m4c_rules = args.m4c_rules
    margs.m4a_rules = args.m4a_rules
    margs.optimized_2d_rules = None
    margs.n_rules_per_source = args.n_rules_per_source
    margs.timesteps = args.timesteps
    margs.grid = args.grid
    margs.max_candidates = args.max_candidates
    margs.hce_replicates = args.hce_replicates
    margs.horizons = args.horizons
    margs.backend = args.backend
    margs.n_bootstrap = 1000
    margs.n_permutations = 1000
    margs.test_seeds = args.test_seeds
    manifest = build_frozen_manifest(margs, out_dir, autodetected)
    (out_dir / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    err = check_seed_disjointness(args.test_seeds, autodetected)
    if err:
        print(f"ABORT: {err}")
        return 2

    # Load rules from each source.
    sources = []
    sources.append(("M7_HCE_optimized",
                   _load_with_source(args.m7_rules, args.n_rules_per_source,
                                     "M7_HCE_optimized")))
    if args.m4c_rules:
        sources.append(("M4C_observer_optimized",
                       _load_with_source(args.m4c_rules, args.n_rules_per_source,
                                         "M4C_observer_optimized")))
    if args.m4a_rules:
        sources.append(("M4A_viability",
                       _load_with_source(args.m4a_rules, args.n_rules_per_source,
                                         "M4A_viability")))

    # Concatenate all rules into one list — the runner is source-agnostic.
    all_rules = []
    for _, rule_list in sources:
        all_rules.extend(rule_list)

    grid = tuple(args.grid)
    cfg_dump = {
        "test_seeds": args.test_seeds, "timesteps": args.timesteps,
        "grid": list(grid), "max_candidates": args.max_candidates,
        "hce_replicates": args.hce_replicates, "horizons": args.horizons,
        "backend": args.backend,
        "n_rules_per_source": args.n_rules_per_source,
        "n_total_rules": len(all_rules),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"M8 -> {out_dir}")
    print(f"  rules={len(all_rules)} test_seeds={len(args.test_seeds)} "
          f"T={args.timesteps} grid={grid} backend={args.backend}")
    print(f"  horizons={args.horizons}")
    print(f"  manifest commit={manifest['git']['commit']}  "
          f"dirty={manifest['git']['dirty']}")

    t0 = time.time()
    results = run_m8_mechanism_discovery(
        rules=all_rules, seeds=args.test_seeds, grid_shape=grid,
        timesteps=args.timesteps, max_candidates=args.max_candidates,
        horizons=args.horizons, n_replicates=args.hce_replicates,
        backend=args.backend, workdir=workdir, progress=print,
        n_workers=args.n_workers,
    )
    elapsed = time.time() - t0
    print(f"\nMeasured {len(results)} candidates in {elapsed:.0f}s")

    # CSVs.
    print("writing CSVs + arrays...")
    _write_mechanism_labels_csv(results, out_dir / "mechanism_labels.csv")
    _write_mediation_csv(results, out_dir / "mediation_summary.csv")
    _write_pathway_traces_csv(results, out_dir / "pathway_traces.csv")
    _write_lifetime_tradeoff_csv(results, out_dir / "lifetime_tradeoff.csv")
    _write_response_maps_csv(results, out_dir / "response_maps.csv")
    _write_feature_dynamics_csv(results, out_dir / "feature_dynamics.csv")
    _write_response_arrays(results, arrays_dir)

    # Mechanism-candidates: same as labels but indexed for downstream use.
    _write_mechanism_labels_csv(results, out_dir / "mechanism_candidates.csv")

    # Stats.
    print("computing stats...")
    summary = m8_full_summary(results)
    (out_dir / "stats_summary.json").write_text(
        json.dumps(summary, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                   else (int(o) if isinstance(o, np.integer)
                         else (o.tolist() if isinstance(o, np.ndarray)
                               else str(o))))
    )

    # Condition summary (mirrors aggregates).
    with (out_dir / "condition_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ("source", "n", "mean_observer", "mean_lifetime", "mean_HCE",
                "boundary_response", "interior_response",
                "first_visible_effect_time",
                "hidden_to_visible_conversion_time",
                "near_threshold_fraction")
        w.writerow(cols)
        for src, a in summary.get("aggregates", {}).items():
            w.writerow([src, a["n_candidates"],
                       f"{a['mean_observer']:.4f}",
                       f"{a['mean_lifetime']:.1f}",
                       f"{a['mean_HCE']:.4f}",
                       f"{a['mean_boundary_response_fraction']:.4f}",
                       f"{a['mean_interior_response_fraction']:.4f}",
                       f"{a['mean_first_visible_effect_time']:.2f}",
                       f"{a['mean_hidden_to_visible_conversion_time']:.2f}",
                       f"{a['mean_near_threshold_fraction']:.4f}"])

    # Plots.
    print("writing plots...")
    write_all_m8_plots(results, plots_dir)

    # Summary.md.
    print("writing summary.md...")
    md = render_m8_summary_md(summary)
    (out_dir / "summary.md").write_text(md)

    print(f"\nDone. Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
