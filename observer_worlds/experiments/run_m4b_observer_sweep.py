"""M4B — multi-rule, multi-seed observer-metric sweep.

Loads the top-K viable rules from an M4A leaderboard, runs paired
(rule, seed) evaluations across the three conditions, computes paired
statistics, writes CSVs/JSON/plots/videos, and produces a `summary.md`
that interprets the result using the rules in the M4B spec.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis import (
    render_stats_summary_md,
    write_all_m4b_plots,
    write_stats_summary_json,
)
from observer_worlds.experiments._m4b_sweep import (
    CONDITION_NAMES,
    SUMMARY_METRICS,
    run_sweep,
)
from observer_worlds.experiments._m4b_writers import (
    write_candidate_metrics_csv,
    write_condition_summary_csv,
    write_paired_differences_csv,
    write_paired_runs_csv,
)
from observer_worlds.search import FractionalRule
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import BSRule


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


def _infer_rule_source(rules_from: str) -> str:
    """Heuristic: infer a rule_source tag from the path.

    Looks for substrings like 'm4a', 'm4c', 'observer', 'viability' in the
    path. If none match, returns 'unknown'.
    """
    s = str(rules_from).lower()
    if "m4c" in s or "observer" in s or "evolve" in s:
        return "M4C_observer_optimized"
    if "m4a" in s or "viability" in s:
        return "M4A_viability"
    return "unknown"


def load_top_rules(rules_from: Path, n_rules: int) -> list[FractionalRule]:
    """Load the top-N rules from a leaderboard.

    Three formats supported:
      1. M4A leaderboard.json: entries sorted by `viability_score`.
      2. M4C leaderboard.json: entries sorted by `fitness` (observer score).
      3. top_rules.json: bare list of FractionalRule.to_dict() dicts.
    """
    data = json.loads(Path(rules_from).read_text())
    if isinstance(data, list) and data and "rule" in data[0]:
        # Pick the available ranking key (viability_score or fitness);
        # fall back to insertion order if neither is present.
        sort_key = "viability_score" if "viability_score" in data[0] else (
            "fitness" if "fitness" in data[0] else None
        )
        if sort_key is not None:
            entries = sorted(data, key=lambda e: -float(e.get(sort_key, 0.0)))[:n_rules]
        else:
            entries = data[:n_rules]
        return [FractionalRule.from_dict(e["rule"]) for e in entries]
    if isinstance(data, list) and data and "birth_min" in data[0]:
        return [FractionalRule.from_dict(d) for d in data[:n_rules]]
    raise ValueError(f"Unrecognized rules JSON format at {rules_from}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M4B observer-metric sweep.")
    p.add_argument("--rules-from", type=str, required=True,
                   help="Path to leaderboard.json (M4A) or top_rules.json.")
    p.add_argument("--n-rules", type=int, default=10)
    p.add_argument("--seeds", type=int, default=10,
                   help="Number of seeds per rule (uses base_seed+0..N-1).")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8],
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--top-k-videos", type=int, default=3,
                   help="Top-K candidates per condition to save as GIFs.")
    p.add_argument("--rollout-steps", type=int, default=6,
                   help="Steps per causality / resilience rollout.")
    p.add_argument("--initial-density-2d", type=float, default=0.3)
    p.add_argument("--backend", choices=["numba", "numpy"], default="numba")
    p.add_argument("--video-frames-kept", type=int, default=120,
                   help="Number of projected frames retained per condition "
                        "for GIF rendering. 0 disables video output.")
    p.add_argument("--snapshots-per-run", type=int, default=2,
                   help="4D snapshots per coherent/shuffled run for "
                        "causality + resilience.")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-permutations", type=int, default=2000)
    p.add_argument("--out-root", type=str, default="outputs",
                   help="Directory under which a timestamped m4b_<UTC>/ "
                        "is created.")
    p.add_argument("--label", type=str, default="m4b")
    # Provenance flags (Part B): record where the rules came from + whether
    # the comparison is fair.
    p.add_argument("--rule-source", type=str, default=None,
                   help="Tag describing where rules came from "
                        "(e.g. 'M4A_viability', 'M4C_observer_optimized', 'manual'). "
                        "If omitted, inferred from the leaderboard path.")
    p.add_argument("--optimization-objective", type=str, default=None,
                   help="If rules were optimized: the objective used "
                        "(e.g. 'lifetime_weighted_mean_score').")
    p.add_argument("--training-seeds", type=int, nargs="+", default=None,
                   help="Seeds used during rule optimization. Used to detect "
                        "whether evaluation seeds overlap the training set.")
    p.add_argument("--baseline-optimized", action="store_true",
                   help="Set this if the matched_2d baseline was also "
                        "observer-fitness-optimized (suppresses the "
                        "'optimized rules' caveat).")
    p.add_argument("--quick", action="store_true",
                   help="Reduce defaults for a fast smoke test: "
                        "n_rules=2, seeds=2, T=100, grid 32x32x4x4, "
                        "n_bootstrap=500, n_permutations=500.")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Process-parallelism: number of worker processes "
                        "for the sweep. Default: cpu_count-2. Use 1 for "
                        "serial (debugging).")
    return p


def apply_quick_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.n_rules = min(args.n_rules, 2)
        args.seeds = min(args.seeds, 2)
        args.timesteps = min(args.timesteps, 100)
        args.grid = [32, 32, 4, 4]
        args.n_bootstrap = min(args.n_bootstrap, 500)
        args.n_permutations = min(args.n_permutations, 500)
    return args


def main(argv: list[str] | None = None) -> int:
    args = apply_quick_overrides(build_arg_parser().parse_args(argv))

    # ---------------- output dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    plots_dir = out_dir / "plots"
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- rules
    rules = load_top_rules(Path(args.rules_from), args.n_rules)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    grid_4d = tuple(args.grid)
    grid_2d = (grid_4d[0], grid_4d[1])

    # ---------------- provenance (Part B)
    rule_source = args.rule_source or _infer_rule_source(args.rules_from)
    training_seeds = (
        sorted(set(args.training_seeds)) if args.training_seeds else None
    )
    overlap = (
        bool(set(seeds).intersection(training_seeds))
        if training_seeds is not None else False
    )
    provenance = {
        "rule_source": rule_source,
        "optimization_objective": args.optimization_objective,
        "training_seeds": training_seeds,
        "evaluation_seeds": list(seeds),
        "evaluation_overlaps_training": overlap,
        "baseline_optimized": bool(args.baseline_optimized),
        "rules_from": str(args.rules_from),
    }

    # ---------------- config
    cfg_dump = {
        "rules_from": args.rules_from,
        "n_rules": len(rules),
        "n_seeds": len(seeds),
        "base_seed": args.base_seed,
        "timesteps": args.timesteps,
        "grid_4d": list(grid_4d),
        "grid_2d": list(grid_2d),
        "rollout_steps": args.rollout_steps,
        "initial_density_2d": args.initial_density_2d,
        "backend": args.backend,
        "video_frames_kept": args.video_frames_kept,
        "snapshots_per_run": args.snapshots_per_run,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "rules": [r.to_dict() for r in rules],
        "provenance": provenance,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"M4B sweep -> {out_dir}")
    print(f"  rules={len(rules)} seeds={len(seeds)} T={args.timesteps} "
          f"grid={grid_4d} backend={args.backend}")
    total_runs = len(rules) * len(seeds) * 3
    print(f"  total per-condition runs: {total_runs}")

    # ---------------- run sweep
    t0 = time.time()
    records = run_sweep(
        rules=rules,
        seeds=seeds,
        grid_shape_4d=grid_4d,
        grid_shape_2d=grid_2d,
        timesteps=args.timesteps,
        initial_density_2d=args.initial_density_2d,
        detection_config=DetectionConfig(),
        backend=args.backend,
        rollout_steps=args.rollout_steps,
        rule_2d=BSRule.life(),
        video_frames_kept=args.video_frames_kept,
        snapshots_per_run=args.snapshots_per_run,
        progress=print,
        n_workers=args.n_workers,
    )
    sweep_seconds = time.time() - t0
    print(f"sweep done in {sweep_seconds:.0f}s")

    # ---------------- baseline checks (regression guards)
    print("baseline checks:")
    n_byte_identical = 0
    n_2d_dead = 0
    for rec in records:
        if rec.coherent_4d.projected_hash == rec.shuffled_4d.projected_hash:
            n_byte_identical += 1
        if rec.matched_2d.mean_active < 1e-3:
            n_2d_dead += 1
    print(f"  coherent vs shuffled byte-identical: {n_byte_identical} / {len(records)}")
    print(f"  2D baseline (active < 0.001): {n_2d_dead} / {len(records)}")

    # ---------------- writers
    print("writing CSVs...")
    write_paired_runs_csv(records, out_dir / "paired_runs.csv")
    write_condition_summary_csv(records, out_dir / "condition_summary.csv")
    write_candidate_metrics_csv(records, out_dir / "candidate_metrics.csv")
    write_paired_differences_csv(records, out_dir / "paired_differences.csv")

    # ---------------- stats (with provenance plumbed through)
    print("computing stats...")
    stats = write_stats_summary_json(
        records, out_dir / "stats_summary.json",
        provenance=provenance,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.base_seed,
    )

    # ---------------- plots + videos
    print("writing plots...")
    write_all_m4b_plots(records, stats, plots_dir)
    if args.video_frames_kept > 0:
        # write_all_m4b_plots also writes videos under plots_dir/videos but
        # the spec puts them under out_dir/videos/top_candidates/. The plots
        # writer accepts either; we prefer the top-level location.
        from observer_worlds.analysis import write_top_candidate_videos
        write_top_candidate_videos(records, videos_dir / "top_candidates",
                                   top_per_condition=args.top_k_videos)

    # ---------------- summary.md
    print("writing summary.md...")
    md_lines = [
        f"# M4B observer-metric sweep — {args.label}",
        "",
        f"- Run dir: `{out_dir}`",
        f"- Rules: {len(rules)} (loaded from `{args.rules_from}`)",
        f"- Seeds per rule: {len(seeds)}  (range: {seeds[0]}..{seeds[-1]})",
        f"- Timesteps: {args.timesteps}",
        f"- Grid: {grid_4d}  (2D baseline: {grid_2d})",
        f"- Sweep wall time: {sweep_seconds:.0f}s",
        "",
        f"- Rule source: **{rule_source or 'unknown'}**",
        f"- Optimization objective: {args.optimization_objective or '—'}",
        f"- Baseline optimized: {bool(args.baseline_optimized)}",
        f"- Eval seeds overlap training seeds: {overlap}",
        "",
        "## Baseline checks",
        "",
        f"- coherent vs shuffled projected-hash collisions: **{n_byte_identical} / {len(records)}** "
        "(should be 0; otherwise the shuffle no-op bug has regressed)",
        f"- 2D baseline near-dead pairs (mean_active < 0.001): {n_2d_dead} / {len(records)}",
        "",
        "## Statistical summary",
        "",
        render_stats_summary_md(stats),
        "",
        "## Artefacts",
        "",
        "- `paired_runs.csv` — wide format, one row per (rule, seed)",
        "- `condition_summary.csv` — long format, one row per (rule, seed, condition)",
        "- `candidate_metrics.csv` — one row per (rule, seed, condition, candidate)",
        "- `paired_differences.csv` — paired diffs for each summary metric",
        "- `stats_summary.json` — bootstrap CIs / permutation p-values / effect sizes",
        "- `plots/*.png` — boxplots, paired-line plots, scatter, forest plot",
        "- `videos/top_candidates/<condition>/rule<R>_seed<S>.gif`",
    ]
    (out_dir / "summary.md").write_text("\n".join(md_lines))
    print(f"\nDone. Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
