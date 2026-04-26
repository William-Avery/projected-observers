"""M7 — evolve 4D rules using HCE-aware composite fitness.

Train/validation/test seed split protocol:
  --train-seeds N    seeds used during evolution (default 5)
  --validation-seeds N seeds used to re-rank top rules after evolution (default 5)
  --test-seeds STR   reserved for run_m7_hce_holdout_validation; never used here

The CLI **never touches test seeds** — the seed splits are constructed
so train/validation/test are disjoint by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.search.hce_search_4d import (
    DEFAULT_M7_SCALES,
    DEFAULT_M7_WEIGHTS,
    M7Fitness,
    evaluate_rule_m7,
    evolutionary_search_hce,
    random_search_hce,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M7 HCE-guided rule search.")
    p.add_argument("--strategy", choices=["random", "evolve"], default="evolve")
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--lam", type=int, default=None,
                   help="Offspring per generation; defaults to population.")
    p.add_argument("--train-seeds", type=int, default=5,
                   help="Number of seeds used during evolution (default 5).")
    p.add_argument("--validation-seeds", type=int, default=5,
                   help="Number of seeds used to re-rank top rules after evolution.")
    p.add_argument("--train-base-seed", type=int, default=1000)
    p.add_argument("--validation-base-seed", type=int, default=4000,
                   help="Validation seeds disjoint from train + test.")
    p.add_argument("--test-base-seed", type=int, default=3000,
                   help="Reserved for holdout; this CLI never uses these.")
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--grid", type=int, nargs=4, default=[32, 32, 4, 4])
    p.add_argument("--max-candidates", type=int, default=15)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--backend", choices=["numba", "numpy", "cuda"], default="numpy")
    p.add_argument("--seed-population", type=str, default=None,
                   help="Path to an M4A or M4C leaderboard.json to seed initial pop.")
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m7_hce_evolve")
    p.add_argument("--quick", action="store_true")
    # Weight overrides.
    for k in DEFAULT_M7_WEIGHTS:
        p.add_argument(f"--w-{k}", type=float, default=None,
                       help=f"Override default weight for '{k}' "
                            f"({DEFAULT_M7_WEIGHTS[k]}).")
    return p


def _quick(args):
    if args.quick:
        args.population = min(args.population, 8)
        args.generations = min(args.generations, 3)
        args.train_seeds = min(args.train_seeds, 2)
        args.validation_seeds = min(args.validation_seeds, 2)
        args.timesteps = min(args.timesteps, 80)
        args.grid = [16, 16, 4, 4]
        args.max_candidates = min(args.max_candidates, 5)
        args.hce_replicates = min(args.hce_replicates, 1)
        args.horizons = [10]
        args.top_k = min(args.top_k, 3)
    if args.lam is None:
        args.lam = args.population
    return args


def _load_seed_pop(path: str | None, mu: int):
    if not path: return None
    return load_top_rules(Path(path), mu)


def _check_seed_disjointness(args) -> tuple[list[int], list[int], list[int]]:
    """Build train, validation, and (theoretical) test seed lists; ensure
    the three are disjoint."""
    tr = list(range(args.train_base_seed, args.train_base_seed + args.train_seeds))
    val = list(range(args.validation_base_seed,
                     args.validation_base_seed + args.validation_seeds))
    test_n_check = max(args.validation_seeds, 5)
    test = list(range(args.test_base_seed, args.test_base_seed + test_n_check))
    s_tr, s_val, s_test = set(tr), set(val), set(test)
    overlap = (s_tr & s_val) | (s_tr & s_test) | (s_val & s_test)
    if overlap:
        raise SystemExit(
            f"ERROR: train / validation / test seeds overlap at {sorted(overlap)}. "
            f"Adjust --train-base-seed, --validation-base-seed, --test-base-seed."
        )
    return tr, val, test


def _gather_weights(args):
    out = dict(DEFAULT_M7_WEIGHTS)
    for k in DEFAULT_M7_WEIGHTS:
        v = getattr(args, f"w_{k}")
        if v is not None: out[k] = v
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _write_top_hce_rules_json(reports: list[M7Fitness], path: Path, top_k: int):
    out = []
    for rank, r in enumerate(reports[:top_k], start=1):
        d = r.rule.to_dict()
        d["_metadata"] = {
            "rank": rank,
            "m7_fitness": float(r.fitness),
            "n_seeds": r.n_seeds,
            "mean_observer_score": float(r.mean_observer_score),
            "mean_hidden_vs_sham_delta": float(r.mean_hidden_vs_sham_delta),
            "mean_hidden_vs_far_delta": float(r.mean_hidden_vs_far_delta),
            "mean_candidate_lifetime": float(r.mean_candidate_lifetime),
            "mean_near_threshold_fraction": float(r.mean_near_threshold_fraction),
            "n_candidates_total": int(r.n_candidates_total),
            "phase": "evolution",
        }
        out.append(d)
    path.write_text(json.dumps(out, indent=2, default=_json_default))


def _write_scores_csv(reports: list[M7Fitness], path: Path, *, phase: str):
    cols = (
        "rank", "phase", "m7_fitness", "n_seeds",
        "birth_min", "birth_max", "survive_min", "survive_max", "initial_density",
        "mean_observer_score", "mean_hidden_vs_sham_delta",
        "mean_hidden_vs_far_delta", "mean_candidate_lifetime",
        "mean_recovery", "mean_near_threshold_fraction",
        "mean_excess_global_divergence", "mean_fragility_penalty",
        "mean_degenerate_candidate_penalty", "mean_initial_projection_delta",
        "n_candidates_total", "aborted_seeds", "sim_time_seconds",
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rank, r in enumerate(reports, start=1):
            w.writerow([
                rank, phase, f"{r.fitness:.6f}", r.n_seeds,
                f"{r.rule.birth_min:.6f}", f"{r.rule.birth_max:.6f}",
                f"{r.rule.survive_min:.6f}", f"{r.rule.survive_max:.6f}",
                f"{r.rule.initial_density:.6f}",
                f"{r.mean_observer_score:.6f}",
                f"{r.mean_hidden_vs_sham_delta:.6f}",
                f"{r.mean_hidden_vs_far_delta:.6f}",
                f"{r.mean_candidate_lifetime:.3f}",
                f"{r.mean_recovery:.4f}",
                f"{r.mean_near_threshold_fraction:.4f}",
                f"{r.mean_excess_global_divergence:.6f}",
                f"{r.mean_fragility_penalty:.4f}",
                f"{r.mean_degenerate_candidate_penalty:.4f}",
                f"{r.mean_initial_projection_delta:.6f}",
                r.n_candidates_total, r.aborted_seeds,
                f"{r.sim_time_seconds:.3f}",
            ])


def _write_history_csv(history: list[dict], path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if history:
            w.writerow(list(history[0].keys()))
            for h in history:
                w.writerow(list(h.values()))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_history_curves(history, plots_dir: Path):
    if not history: return
    gens = [h["generation"] for h in history]

    # 1. fitness
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, [h["best_fitness"] for h in history], "-o", label="best")
    ax.plot(gens, [h["mean_fitness"] for h in history], "--s", label="mean")
    ax.plot(gens, [h["median_fitness"] for h in history], ":^", label="median")
    ax.set_xlabel("generation"); ax.set_ylabel("M7 fitness")
    ax.set_title("Fitness over generations")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(plots_dir / "fitness_over_generations.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. observer_score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, [h["best_observer_score"] for h in history], "-o", color="#1f77b4")
    ax.set_xlabel("generation"); ax.set_ylabel("best observer_score")
    ax.set_title("Best observer_score over generations")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "observer_score_over_generations.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. HCE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, [h["best_hidden_vs_sham"] for h in history], "-o",
            label="hidden_vs_sham")
    ax.plot(gens, [h["best_hidden_vs_far"] for h in history], "--s",
            label="hidden_vs_far")
    ax.set_xlabel("generation"); ax.set_ylabel("best HCE measure")
    ax.set_title("Best HCE measures over generations")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(plots_dir / "hce_over_generations.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 4. threshold penalty
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, [h["best_near_threshold_fraction"] for h in history], "-o",
            color="#d62728")
    ax.set_xlabel("generation")
    ax.set_ylabel("min near_threshold_fraction (lower=less artifact)")
    ax.set_title("Threshold penalty (lower is better)")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "threshold_penalty_over_generations.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_train_vs_validation(train: list[M7Fitness], val: list[M7Fitness],
                              plots_dir: Path):
    """Per-rule scatter: x = train fitness, y = validation fitness."""
    val_by_rule = {(r.rule.birth_min, r.rule.birth_max,
                   r.rule.survive_min, r.rule.survive_max,
                   r.rule.initial_density): r for r in val}
    pairs = []
    for r in train:
        key = (r.rule.birth_min, r.rule.birth_max, r.rule.survive_min,
               r.rule.survive_max, r.rule.initial_density)
        if key in val_by_rule:
            pairs.append((r.fitness, val_by_rule[key].fitness))
    if not pairs: return
    fig, ax = plt.subplots(figsize=(7, 7))
    xs, ys = zip(*pairs)
    ax.scatter(xs, ys, alpha=0.7)
    lim = max(max(xs), max(ys), 0.001) * 1.1
    lim_lo = min(min(xs), min(ys), 0)
    ax.plot([lim_lo, lim], [lim_lo, lim], "--", color="gray", label="y=x")
    ax.set_xlabel("train fitness"); ax.set_ylabel("validation fitness")
    n_reproduce = sum(1 for x, y in pairs if y > 0)
    ax.set_title(f"Train vs validation fitness (N={len(pairs)}, "
                 f"validation positive {n_reproduce}/{len(pairs)})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(plots_dir / "train_vs_validation_fitness.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_rule_parameter_distributions(reports: list[M7Fitness], plots_dir: Path):
    if not reports: return
    rules = [r.rule for r in reports]
    fields = ["birth_min", "birth_max", "survive_min", "survive_max", "initial_density"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, fld in zip(axes, fields):
        vals = [getattr(r, fld) for r in rules]
        ax.hist(vals, bins=15, color="#1f77b4", alpha=0.7)
        ax.set_xlabel(fld); ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Top-rule parameter distributions")
    fig.tight_layout()
    fig.savefig(plots_dir / "rule_parameter_distributions.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))
    train_seeds, val_seeds, _test = _check_seed_disjointness(args)
    weights = _gather_weights(args)
    grid = tuple(args.grid)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_sims"; workdir.mkdir(parents=True, exist_ok=True)

    cfg_dump = {
        "strategy": args.strategy, "population": args.population,
        "generations": args.generations, "lam": args.lam,
        "train_seeds": train_seeds, "validation_seeds": val_seeds,
        "timesteps": args.timesteps, "grid": list(grid),
        "max_candidates": args.max_candidates,
        "hce_replicates": args.hce_replicates, "horizons": args.horizons,
        "backend": args.backend, "weights": weights,
        "scales": dict(DEFAULT_M7_SCALES),
        "seed_population": args.seed_population,
        "top_k": args.top_k,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2,
                                                    default=_json_default))

    print(f"M7 HCE evolution -> {out_dir}")
    print(f"  strategy={args.strategy} pop={args.population} "
          f"generations={args.generations}")
    print(f"  train_seeds={train_seeds}  validation_seeds={val_seeds}")
    print(f"  T={args.timesteps} grid={grid} max_cand={args.max_candidates} "
          f"replicates={args.hce_replicates} horizons={args.horizons}")
    print(f"  weights={weights}")

    initial = _load_seed_pop(args.seed_population, args.population)

    history = []
    t0 = time.time()
    if args.strategy == "random":
        train_reports = random_search_hce(
            n_rules=args.population, train_seeds=train_seeds,
            grid_shape=grid, timesteps=args.timesteps,
            max_candidates=args.max_candidates, horizons=args.horizons,
            n_replicates=args.hce_replicates, backend=args.backend,
            sampler_seed=args.sampler_seed, weights=weights,
            workdir=workdir, progress=print,
        )
    else:
        train_reports, history = evolutionary_search_hce(
            n_generations=args.generations, mu=args.population,
            lam=args.lam, train_seeds=train_seeds,
            grid_shape=grid, timesteps=args.timesteps,
            max_candidates=args.max_candidates, horizons=args.horizons,
            n_replicates=args.hce_replicates, backend=args.backend,
            sampler_seed=args.sampler_seed, weights=weights,
            initial_population=initial, workdir=workdir, progress=print,
        )
    train_seconds = time.time() - t0
    print(f"\nTraining done in {train_seconds:.0f}s")

    _write_history_csv(history, out_dir / "evolution_history.csv")
    _write_scores_csv(train_reports, out_dir / "train_scores.csv", phase="train")

    # Validation: re-evaluate the top-K on disjoint seeds.
    print(f"\nValidating top-{min(args.top_k, len(train_reports))} rules on "
          f"validation seeds {val_seeds}...")
    val_reports = []
    for r in train_reports[:args.top_k]:
        vfit = evaluate_rule_m7(
            r.rule, seeds=val_seeds, grid_shape=grid,
            timesteps=args.timesteps, max_candidates=args.max_candidates,
            horizons=args.horizons, n_replicates=args.hce_replicates,
            backend=args.backend, weights=weights,
            workdir=workdir,
        )
        val_reports.append(vfit)
        print(f"  rule {r.rule.short_repr()}  train={r.fitness:+.3f}  "
              f"val={vfit.fitness:+.3f}")
    _write_scores_csv(val_reports, out_dir / "validation_scores.csv", phase="validation")
    val_reports.sort(key=lambda r: -r.fitness)

    # Save top-K rules JSON ordered by validation fitness.
    _write_top_hce_rules_json(val_reports, out_dir / "top_hce_rules.json", args.top_k)

    # Plots.
    _plot_history_curves(history, plots_dir)
    _plot_train_vs_validation(train_reports[:args.top_k], val_reports, plots_dir)
    _plot_rule_parameter_distributions(val_reports, plots_dir)

    # Summary.md.
    md = [f"# M7 HCE-guided evolution — {args.label}", ""]
    md.append(f"- Strategy: {args.strategy}")
    md.append(f"- Population: {args.population}, generations: {args.generations}")
    md.append(f"- Train seeds: {train_seeds}")
    md.append(f"- Validation seeds: {val_seeds}")
    md.append(f"- Total wall time: {train_seconds:.0f}s")
    md.append("")
    md.append("## Top-K validation results (sorted by validation fitness)")
    md.append("")
    md.append("| rank | val_fitness | obs | vs_sham | vs_far | lifetime | "
              "near_thresh | rule |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(val_reports, start=1):
        md.append(
            f"| {i} | {r.fitness:+.3f} | {r.mean_observer_score:+.3f} | "
            f"{r.mean_hidden_vs_sham_delta:+.4f} | "
            f"{r.mean_hidden_vs_far_delta:+.4f} | "
            f"{r.mean_candidate_lifetime:.0f} | "
            f"{r.mean_near_threshold_fraction:.2f} | "
            f"{r.rule.short_repr()} |"
        )
    md.append("")
    md.append("## Train→validation generalization")
    md.append("")
    train_top = {(r.rule.birth_min, r.rule.birth_max, r.rule.survive_min,
                 r.rule.survive_max, r.rule.initial_density): r
                for r in train_reports[:args.top_k]}
    matched = []
    for v in val_reports:
        key = (v.rule.birth_min, v.rule.birth_max, v.rule.survive_min,
               v.rule.survive_max, v.rule.initial_density)
        if key in train_top:
            matched.append((train_top[key], v))
    if matched:
        diffs = np.array([v.fitness - t.fitness for t, v in matched])
        md.append(f"- N rules with both train + validation: {len(matched)}")
        md.append(f"- Mean train fitness: {np.mean([t.fitness for t,_ in matched]):+.3f}")
        md.append(f"- Mean val fitness:   {np.mean([v.fitness for _,v in matched]):+.3f}")
        md.append(f"- Mean drop (val - train): {diffs.mean():+.3f}")
        md.append(f"- N validation > 0: {int((np.array([v.fitness for _,v in matched]) > 0).sum())}")
    md.append("")
    md.append("## Artefacts")
    md.append("- `evolution_history.csv` — per-generation best/mean/median + obs/HCE/threshold")
    md.append("- `train_scores.csv` — full training-population scores")
    md.append("- `validation_scores.csv` — top-K re-evaluated on validation seeds")
    md.append("- `top_hce_rules.json` — feed to `run_m7_hce_holdout_validation --m7-rules`")
    md.append("- `plots/*.png`")
    (out_dir / "summary.md").write_text("\n".join(md))

    print(f"\nDone. Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
