"""M4D — held-out validation of M4C-optimized 4D rules.

Two-pass design:
    Pass A: matched_2d = Conway's Life (fixed baseline) -> vs_fixed_2d/
    Pass B: matched_2d = top optimized 2D rule         -> vs_optimized_2d/
            (only if --optimized-2d-rules is provided)

Combined summary.md applies the M4D interpretation rules from the spec:
  - coh beats fixed but not optimized -> "beat standard 2D Life, not equally optimized"
  - coh beats both                    -> "advantage beyond optimization alone"
  - optimized 2D beats coh            -> "advantage was due to optimization"
  - only Pass A available             -> echo Pass A finding + caveat
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from observer_worlds.analysis.m4b_stats import (
    _significant_negative,
    _significant_positive,
    render_stats_summary_md,
    write_stats_summary_json,
)
from observer_worlds.experiments._m4b_sweep import run_sweep
from observer_worlds.experiments._m4b_writers import (
    write_candidate_metrics_csv,
    write_condition_summary_csv,
    write_paired_differences_csv,
    write_paired_runs_csv,
)
from observer_worlds.experiments.run_m4b_observer_sweep import (
    _infer_rule_source,
    load_top_rules,
)
from observer_worlds.search import FractionalRule
from observer_worlds.utils.config import DetectionConfig
from observer_worlds.worlds import BSRule


# Canonical interpretation sentences for combined summary.
COMBINED_BEAT_FIXED_NOT_OPTIMIZED = (
    "4D optimized rules beat standard 2D Life, but not an equally "
    "optimized 2D rule class."
)
COMBINED_BEAT_BOTH = (
    "This is stronger evidence that 4D projected dynamics offer an "
    "advantage beyond optimization alone."
)
COMBINED_OPTIMIZED_2D_WINS = (
    "The apparent advantage was due to optimization, not dimensional "
    "projection."
)
COMBINED_ONLY_PASS_A_POSITIVE = (
    "Coherent 4D significantly beats fixed 2D Life on normalized metrics. "
    "Optimized 2D baseline not provided — comparison limited to fixed 2D Life."
)
COMBINED_ONLY_PASS_A_NEGATIVE = (
    "Coherent 4D does not significantly beat fixed 2D Life on normalized "
    "metrics. Optimized 2D baseline not provided — comparison limited to "
    "fixed 2D Life."
)
COMBINED_MIXED = (
    "Mixed result across both 2D baselines; no strong dimensional "
    "advantage established."
)


_PRIMARY_METRICS = ("score_per_track", "lifetime_weighted_mean_score")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M4D held-out validation.")
    p.add_argument("--rules-from", type=str, required=True,
                   help="Path to leaderboard.json (M4A or M4C).")
    p.add_argument("--n-rules", type=int, default=10)
    p.add_argument("--seeds", type=int, default=10,
                   help="Number of held-out evaluation seeds.")
    p.add_argument("--base-eval-seed", type=int, default=2000,
                   help="Held-out seeds start here (default 2000 to avoid "
                        "the M4B/M4C training default of 1000).")
    p.add_argument("--training-seeds", type=int, nargs="+", default=None,
                   help="Seeds used during optimization. Aborts if eval "
                        "seeds overlap unless --allow-overlap is set.")
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--rollout-steps", type=int, default=6)
    p.add_argument("--initial-density-2d-life", type=float, default=0.30)
    p.add_argument("--backend", choices=["numba", "numpy", "cuda"], default="numba")
    p.add_argument("--video-frames-kept", type=int, default=120)
    p.add_argument("--snapshots-per-run", type=int, default=2)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-permutations", type=int, default=2000)
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m4d")
    p.add_argument("--optimized-2d-rules", type=str, default=None,
                   help="Path to top_2d_rules.json from evolve_2d_observer_rules. "
                        "If given, runs an additional comparison vs the top "
                        "optimized 2D rule.")
    p.add_argument("--quick", action="store_true",
                   help="Reduce defaults: n_rules=2, seeds=2, T=100, grid 32 32 4 4, "
                        "n_bootstrap=500, n_permutations=500.")
    return p


def _apply_quick(args):
    if args.quick:
        args.n_rules = min(args.n_rules, 2)
        args.seeds = min(args.seeds, 2)
        args.timesteps = min(args.timesteps, 100)
        args.grid = [32, 32, 4, 4]
        args.n_bootstrap = min(args.n_bootstrap, 500)
        args.n_permutations = min(args.n_permutations, 500)
    return args


# ---------------------------------------------------------------------------
# Pass runner
# ---------------------------------------------------------------------------


def _run_one_pass(
    *,
    pass_label: str,
    out_dir: Path,
    rules: list,
    seeds: list[int],
    grid_4d: tuple[int, int, int, int],
    grid_2d: tuple[int, int],
    args,
    rule_2d: BSRule,
    initial_density_2d: float,
    provenance: dict,
) -> dict:
    """Run a single M4B-style sweep with a specific 2D baseline. Returns the
    stats dict for the combined-summary interpretation."""
    from observer_worlds.analysis import write_all_m4b_plots, write_top_candidate_videos

    sub = out_dir / pass_label
    sub.mkdir(parents=True, exist_ok=True)
    plots_dir = sub / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = sub / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Pass: {pass_label} ===")
    print(f"  matched_2d_rule: B={rule_2d.birth} S={rule_2d.survival} "
          f"initial_density={initial_density_2d}")

    records = run_sweep(
        rules=rules,
        seeds=seeds,
        grid_shape_4d=grid_4d,
        grid_shape_2d=grid_2d,
        timesteps=args.timesteps,
        initial_density_2d=initial_density_2d,
        detection_config=DetectionConfig(),
        backend=args.backend,
        rollout_steps=args.rollout_steps,
        rule_2d=rule_2d,
        video_frames_kept=args.video_frames_kept,
        snapshots_per_run=args.snapshots_per_run,
        progress=lambda s: print("  " + s),
    )

    # Standard M4B writers.
    write_paired_runs_csv(records, sub / "paired_runs.csv")
    write_condition_summary_csv(records, sub / "condition_summary.csv")
    write_candidate_metrics_csv(records, sub / "candidate_metrics.csv")
    write_paired_differences_csv(records, sub / "paired_differences.csv")
    stats = write_stats_summary_json(
        records, sub / "stats_summary.json",
        provenance=provenance,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.base_eval_seed,
    )
    write_all_m4b_plots(records, stats, plots_dir)
    if args.video_frames_kept > 0:
        write_top_candidate_videos(
            records, videos_dir / "top_candidates", top_per_condition=3
        )

    # Per-pass summary.
    sub_md = [
        f"# M4D pass — {pass_label}",
        "",
        f"- N pairs: {len(records)}",
        f"- 2D baseline: B={list(rule_2d.birth)} S={list(rule_2d.survival)} "
        f"initial_density={initial_density_2d}",
        f"- baseline_optimized: {provenance.get('baseline_optimized', False)}",
        "",
        render_stats_summary_md(stats),
    ]
    (sub / "summary.md").write_text("\n".join(sub_md))
    return stats


# ---------------------------------------------------------------------------
# Combined interpretation
# ---------------------------------------------------------------------------


def _coh_beats_2d_normalized(stats: dict) -> bool:
    c = stats.get("comparisons", {}).get("coherent_4d_vs_matched_2d", {})
    return any(_significant_positive(c.get(m, {})) for m in _PRIMARY_METRICS)


def _2d_beats_coh_normalized(stats: dict) -> bool:
    c = stats.get("comparisons", {}).get("coherent_4d_vs_matched_2d", {})
    return any(_significant_negative(c.get(m, {})) for m in _PRIMARY_METRICS)


def combined_interpretation(stats_a: dict, stats_b: dict | None) -> str:
    """Apply the M4D interpretation rules. ``stats_a`` is vs_fixed_2d;
    ``stats_b`` is vs_optimized_2d (or None)."""
    a_coh_wins = _coh_beats_2d_normalized(stats_a)
    if stats_b is None:
        return COMBINED_ONLY_PASS_A_POSITIVE if a_coh_wins \
            else COMBINED_ONLY_PASS_A_NEGATIVE

    b_coh_wins = _coh_beats_2d_normalized(stats_b)
    b_2d_wins = _2d_beats_coh_normalized(stats_b)

    if b_2d_wins:
        return COMBINED_OPTIMIZED_2D_WINS
    if a_coh_wins and b_coh_wins:
        return COMBINED_BEAT_BOTH
    if a_coh_wins and not b_coh_wins:
        return COMBINED_BEAT_FIXED_NOT_OPTIMIZED
    return COMBINED_MIXED


def _headline_table(stats: dict, pass_name: str) -> list[str]:
    """One line per primary metric showing mean_diff, CI, p-value."""
    lines = [f"### {pass_name}", "",
             "| metric | mean_diff | 95% CI | perm p | win_rate_a |",
             "|---|---|---|---|---|"]
    c = stats.get("comparisons", {}).get("coherent_4d_vs_matched_2d", {})
    for m in _PRIMARY_METRICS:
        d = c.get(m, {})
        ci = (d.get("bootstrap_ci_low", 0.0), d.get("bootstrap_ci_high", 0.0))
        lines.append(
            f"| {m} | {d.get('mean_difference', 0.0):+.5f} | "
            f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | "
            f"{d.get('permutation_p_value', 1.0):.4f} | "
            f"{d.get('win_rate_a', 0.0):.2f} |"
        )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _apply_quick(build_arg_parser().parse_args(argv))

    # Output dir.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rules and seeds.
    rules = load_top_rules(Path(args.rules_from), args.n_rules)
    seeds = list(range(args.base_eval_seed, args.base_eval_seed + args.seeds))

    # Seed-overlap guard (Part E).
    training_seeds = sorted(set(args.training_seeds)) if args.training_seeds else None
    overlap = (
        bool(set(seeds).intersection(training_seeds))
        if training_seeds is not None else False
    )
    if overlap and not args.allow_overlap:
        print(f"ERROR: held-out seeds overlap training seeds "
              f"{sorted(set(seeds) & set(training_seeds))}; "
              f"pass --allow-overlap to proceed.", flush=True)
        return 2

    rule_source = _infer_rule_source(args.rules_from)
    grid_4d = tuple(args.grid)
    grid_2d = (grid_4d[0], grid_4d[1])

    # Two passes; both use the same rules + seeds.
    base_provenance = {
        "rule_source": rule_source,
        "rules_from": str(args.rules_from),
        "training_seeds": training_seeds,
        "evaluation_seeds": list(seeds),
        "evaluation_overlaps_training": overlap,
        "optimization_objective": "lifetime_weighted_mean_score",  # M4C default
    }

    config_dump = {
        "rules_from": args.rules_from,
        "n_rules": len(rules),
        "n_seeds": len(seeds),
        "base_eval_seed": args.base_eval_seed,
        "timesteps": args.timesteps,
        "grid_4d": list(grid_4d),
        "grid_2d": list(grid_2d),
        "rollout_steps": args.rollout_steps,
        "initial_density_2d_life": args.initial_density_2d_life,
        "backend": args.backend,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "optimized_2d_rules": args.optimized_2d_rules,
        "training_seeds": training_seeds,
        "rule_source": rule_source,
        "rules": [r.to_dict() for r in rules],
    }
    (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    print(f"M4D held-out validation -> {out_dir}")
    print(f"  rule_source={rule_source}")
    print(f"  n_rules={len(rules)} eval_seeds={seeds[0]}..{seeds[-1]} "
          f"T={args.timesteps} grid={grid_4d}")
    if training_seeds:
        print(f"  training_seeds={training_seeds}  overlap={overlap}")

    # Pass A: vs fixed 2D Life.
    t0 = time.time()
    prov_a = {**base_provenance, "baseline_optimized": False, "pass": "vs_fixed_2d"}
    stats_a = _run_one_pass(
        pass_label="vs_fixed_2d",
        out_dir=out_dir, rules=rules, seeds=seeds,
        grid_4d=grid_4d, grid_2d=grid_2d, args=args,
        rule_2d=BSRule.life(),
        initial_density_2d=args.initial_density_2d_life,
        provenance=prov_a,
    )

    # Pass B: vs optimized 2D rule (optional).
    stats_b = None
    optimized_2d_rule = None
    if args.optimized_2d_rules:
        opt_data = json.loads(Path(args.optimized_2d_rules).read_text())
        if not isinstance(opt_data, list) or not opt_data:
            print(f"WARNING: optimized-2d-rules file is empty: {args.optimized_2d_rules}")
        else:
            top_2d = opt_data[0]
            optimized_2d_rule = FractionalRule.from_dict(top_2d)
            opt_density = optimized_2d_rule.initial_density
            opt_bsrule = optimized_2d_rule.to_bsrule(max_count=8)
            prov_b = {**base_provenance, "baseline_optimized": True, "pass": "vs_optimized_2d",
                      "optimized_2d_rule": optimized_2d_rule.to_dict()}
            stats_b = _run_one_pass(
                pass_label="vs_optimized_2d",
                out_dir=out_dir, rules=rules, seeds=seeds,
                grid_4d=grid_4d, grid_2d=grid_2d, args=args,
                rule_2d=opt_bsrule, initial_density_2d=opt_density,
                provenance=prov_b,
            )
    elapsed = time.time() - t0

    # Combined summary.
    combined_md: list[str] = [
        f"# M4D held-out validation — {args.label}",
        "",
        f"- Run dir: `{out_dir}`",
        f"- Rule source: **{rule_source}**",
        f"- Rules from: `{args.rules_from}`",
        f"- N rules: {len(rules)}",
        f"- Held-out evaluation seeds: {seeds}",
        f"- Training seeds: {training_seeds}",
        f"- Eval/training overlap: **{overlap}**",
        f"- Timesteps: {args.timesteps}, grid: {grid_4d}",
        f"- Total wall time: {elapsed:.0f}s",
        "",
        "## Headline numbers (coherent_4d - matched_2d, normalized metrics)",
        "",
    ]
    combined_md.extend(_headline_table(stats_a, "Pass A — vs fixed 2D Life"))
    combined_md.append("")
    if stats_b is not None:
        combined_md.extend(_headline_table(stats_b, "Pass B — vs optimized 2D"))
        combined_md.append("")
        combined_md.append(
            f"- Optimized 2D rule: {optimized_2d_rule.to_dict() if optimized_2d_rule else '—'}"
        )
    else:
        combined_md.append("_Pass B not run — `--optimized-2d-rules` not provided._")
    combined_md.extend([
        "",
        "## Combined interpretation",
        "",
        combined_interpretation(stats_a, stats_b),
        "",
        "## Per-pass details",
        "",
        "- `vs_fixed_2d/summary.md` — full M4B-style summary for Pass A",
    ])
    if stats_b is not None:
        combined_md.append("- `vs_optimized_2d/summary.md` — full M4B-style summary for Pass B")
    (out_dir / "summary.md").write_text("\n".join(combined_md))

    print(f"\nDone. Run dir: {out_dir}")
    print(f"Combined interpretation: {combined_interpretation(stats_a, stats_b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
