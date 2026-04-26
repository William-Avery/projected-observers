"""Part D — evolve 2D fractional rules using observer_score.

Produces an optimized 2D baseline that the M4D held-out validation can use
as a fair comparison against optimized 4D rules. Without this, the only
2D baseline is Conway's Life, which is itself an unoptimized hand-picked
rule.

Output:
    outputs/m4d_2d_evolve_<UTC>/
      config.json
      top_2d_rules.json       — list of FractionalRule dicts (top-K),
                                with a `_metadata` key per entry.
      leaderboard.csv         — full ranked CSV
      leaderboard.json        — full ranked JSON
      evolution_history.csv   — per-generation best/mean/median (evolve only)
      summary.md
      plots/
        fitness_vs_generation.png  (evolve only)
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

from observer_worlds.search import (
    FITNESS_MODES,
    FractionalRule,
    ObserverFitnessReport,
    evaluate_observer_fitness_2d,
    evolutionary_search_observer_2d,
    random_search_observer_2d,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evolve 2D fractional rules on observer fitness.")
    p.add_argument("--strategy", choices=["random", "evolve"], default="evolve")
    p.add_argument("--population", type=int, default=50,
                   help="mu for evolve, n_rules for random.")
    p.add_argument("--generations", type=int, default=25,
                   help="evolve only.")
    p.add_argument("--lam", type=int, default=None,
                   help="evolve only; defaults to --population.")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-eval-seed", type=int, default=1000)
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--grid", type=int, nargs=2, default=[32, 32])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--fitness-mode", choices=list(FITNESS_MODES),
                   default="lifetime_weighted")
    p.add_argument("--out-dir", type=str, default=None)
    return p


def _json_default(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _write_top_2d_rules(reports: list[ObserverFitnessReport], path: Path, top_k: int) -> None:
    """Write the file consumed by run_m4d_holdout_validation.py --optimized-2d-rules.

    Each entry has all 5 FractionalRule fields at top level (so
    FractionalRule.from_dict works), plus a `_metadata` key with rank/fitness.
    """
    out = []
    for rank, r in enumerate(reports[:top_k], start=1):
        d = r.rule.to_dict()
        d["_metadata"] = {
            "rank": rank,
            "fitness": float(r.fitness),
            "fitness_mode": r.fitness_mode,
            "n_seeds": r.n_seeds,
            "mean_n_candidates": float(r.mean_n_candidates),
            "mean_lifetime_weighted_mean_score": float(r.mean_lifetime_weighted_mean_score),
            "mean_top5_mean_score": float(r.mean_top5_mean_score),
        }
        out.append(d)
    path.write_text(json.dumps(out, indent=2, default=_json_default))


def _write_leaderboard_csv(reports: list[ObserverFitnessReport], path: Path) -> None:
    cols = (
        "rank", "fitness", "fitness_mode", "n_seeds",
        "birth_min", "birth_max", "survive_min", "survive_max", "initial_density",
        "mean_n_tracks", "mean_n_candidates", "mean_max_score", "mean_top5_mean_score",
        "mean_p95_score", "mean_lifetime_weighted_mean_score", "mean_score_per_track",
        "mean_late_active", "mean_max_component_lifetime",
        "per_seed_fitness", "aborted_seeds", "sim_time_seconds",
    )
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rank, r in enumerate(reports, start=1):
            w.writerow([
                rank, f"{r.fitness:.6f}", r.fitness_mode, r.n_seeds,
                f"{r.rule.birth_min:.6f}", f"{r.rule.birth_max:.6f}",
                f"{r.rule.survive_min:.6f}", f"{r.rule.survive_max:.6f}",
                f"{r.rule.initial_density:.6f}",
                f"{r.mean_n_tracks:.3f}", f"{r.mean_n_candidates:.3f}",
                f"{r.mean_max_score:.6f}", f"{r.mean_top5_mean_score:.6f}",
                f"{r.mean_p95_score:.6f}", f"{r.mean_lifetime_weighted_mean_score:.6f}",
                f"{r.mean_score_per_track:.6f}", f"{r.mean_late_active:.6f}",
                f"{r.mean_max_component_lifetime:.3f}",
                ";".join(f"{x:.6f}" for x in r.per_seed_fitness),
                r.aborted_seeds, f"{r.sim_time_seconds:.3f}",
            ])


def _write_leaderboard_json(reports: list[ObserverFitnessReport], path: Path) -> None:
    entries = []
    for rank, r in enumerate(reports, start=1):
        entries.append({
            "rank": rank, "fitness": float(r.fitness),
            "fitness_mode": r.fitness_mode, "n_seeds": r.n_seeds,
            "rule": r.rule.to_dict(),
            "mean_n_tracks": r.mean_n_tracks,
            "mean_n_candidates": r.mean_n_candidates,
            "mean_top5_mean_score": r.mean_top5_mean_score,
            "mean_lifetime_weighted_mean_score": r.mean_lifetime_weighted_mean_score,
            "mean_score_per_track": r.mean_score_per_track,
            "per_seed_fitness": r.per_seed_fitness,
            "aborted_seeds": r.aborted_seeds,
            "sim_time_seconds": r.sim_time_seconds,
        })
    path.write_text(json.dumps(entries, indent=2, default=_json_default))


def _write_history_csv(history: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness", "mean_fitness",
                    "median_fitness", "population_size"])
        for h in history:
            w.writerow([h["generation"], f"{h['best_fitness']:.6f}",
                        f"{h['mean_fitness']:.6f}", f"{h['median_fitness']:.6f}",
                        h["population_size"]])


def _plot_history(history: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    gens = [h["generation"] for h in history]
    ax.plot(gens, [h["best_fitness"] for h in history], "-o", label="best", color="#1f77b4")
    ax.plot(gens, [h["mean_fitness"] for h in history], "--s", label="mean", color="#ff7f0e")
    ax.plot(gens, [h["median_fitness"] for h in history], ":^", label="median", color="#2ca02c")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("Evolution of 2D rule fitness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _evaluate_life_baseline(args) -> ObserverFitnessReport | None:
    """Evaluate Conway's Life as a comparison anchor."""
    life = FractionalRule(
        birth_min=3 / 8, birth_max=3 / 8,
        survive_min=2 / 8, survive_max=3 / 8,
        initial_density=0.30,
    )
    try:
        return evaluate_observer_fitness_2d(
            life, n_seeds=args.n_seeds, base_seed=args.base_eval_seed,
            grid_shape=tuple(args.grid), timesteps=args.timesteps,
            fitness_mode=args.fitness_mode,
        )
    except Exception as e:
        print(f"  warning: Life baseline eval failed: {e}")
        return None


def _build_summary_md(
    reports: list[ObserverFitnessReport], history: list[dict],
    args, life_report: ObserverFitnessReport | None, out_dir: Path,
) -> str:
    lines = [f"# 2D observer-fitness evolution — {args.strategy}", ""]
    lines.append(f"- Out dir: `{out_dir}`")
    lines.append(f"- Strategy: {args.strategy}")
    lines.append(f"- Population: {args.population}")
    if args.strategy == "evolve":
        lines.append(f"- Generations: {args.generations}, lambda: {args.lam or args.population}")
    lines.append(f"- Seeds per rule: {args.n_seeds}")
    lines.append(f"- Grid: {tuple(args.grid)}, T: {args.timesteps}")
    lines.append(f"- Fitness mode: {args.fitness_mode}")
    lines.append("")
    lines.append("## Top rules")
    lines.append("")
    lines.append("| rank | fitness | top5 | lwm | sptr | n_cand | rule |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(reports[: args.top_k], start=1):
        lines.append(
            f"| {i} | {r.fitness:+.4f} | {r.mean_top5_mean_score:+.3f} | "
            f"{r.mean_lifetime_weighted_mean_score:+.3f} | "
            f"{r.mean_score_per_track:+.5f} | "
            f"{r.mean_n_candidates:.0f} | "
            f"B[{r.rule.birth_min:.2f},{r.rule.birth_max:.2f}]"
            f"_S[{r.rule.survive_min:.2f},{r.rule.survive_max:.2f}]"
            f"_d{r.rule.initial_density:.2f} |"
        )
    if life_report is not None:
        lines.append("")
        lines.append("## Conway's Life anchor (for comparison)")
        lines.append("")
        lines.append(f"- fitness ({args.fitness_mode}): **{life_report.fitness:+.4f}**")
        lines.append(f"- top5: {life_report.mean_top5_mean_score:+.3f}, "
                     f"lwm: {life_report.mean_lifetime_weighted_mean_score:+.3f}, "
                     f"score_per_track: {life_report.mean_score_per_track:+.5f}, "
                     f"n_cand: {life_report.mean_n_candidates:.0f}")
        if reports:
            best = reports[0]
            delta = best.fitness - life_report.fitness
            verdict = "above" if delta > 0 else "below"
            lines.append(f"- Best evolved rule is **{abs(delta):.4f} {verdict}** Life.")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `top_2d_rules.json` — feed to `run_m4d_holdout_validation --optimized-2d-rules`")
    lines.append("- `leaderboard.csv`, `leaderboard.json`")
    if args.strategy == "evolve":
        lines.append("- `evolution_history.csv`, `plots/fitness_vs_generation.png`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.lam is None:
        args.lam = args.population
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = f"outputs/m4d_2d_evolve_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"2D observer search ({args.strategy}) -> {out_dir}")
    print(f"  pop={args.population} seeds={args.n_seeds} T={args.timesteps} "
          f"grid={tuple(args.grid)} fitness={args.fitness_mode}")

    history: list[dict] = []
    t0 = time.time()
    if args.strategy == "random":
        reports = random_search_observer_2d(
            n_rules=args.population, n_seeds=args.n_seeds,
            base_seed=args.base_eval_seed, sampler_seed=args.sampler_seed,
            grid_shape=tuple(args.grid), timesteps=args.timesteps,
            fitness_mode=args.fitness_mode, progress=print,
        )
    else:
        reports, history = evolutionary_search_observer_2d(
            n_generations=args.generations,
            mu=args.population, lam=args.lam,
            n_seeds=args.n_seeds, base_seed=args.base_eval_seed,
            sampler_seed=args.sampler_seed,
            grid_shape=tuple(args.grid), timesteps=args.timesteps,
            fitness_mode=args.fitness_mode, progress=print,
        )
    elapsed = time.time() - t0
    print(f"\nsearch done in {elapsed:.0f}s")

    # Optional Life anchor.
    print("evaluating Conway's Life as anchor...")
    life_report = _evaluate_life_baseline(args)

    # Outputs.
    _write_leaderboard_csv(reports, out_dir / "leaderboard.csv")
    _write_leaderboard_json(reports, out_dir / "leaderboard.json")
    _write_top_2d_rules(reports, out_dir / "top_2d_rules.json", args.top_k)
    if history:
        _write_history_csv(history, out_dir / "evolution_history.csv")
        _plot_history(history, plots_dir / "fitness_vs_generation.png")
    summary = _build_summary_md(reports, history, args, life_report, out_dir)
    (out_dir / "summary.md").write_text(summary)

    cfg = {
        "strategy": args.strategy,
        "population": args.population,
        "generations": args.generations if args.strategy == "evolve" else None,
        "lam": args.lam if args.strategy == "evolve" else None,
        "n_seeds": args.n_seeds,
        "base_eval_seed": args.base_eval_seed,
        "sampler_seed": args.sampler_seed,
        "timesteps": args.timesteps,
        "grid": list(args.grid),
        "top_k": args.top_k,
        "fitness_mode": args.fitness_mode,
        "elapsed_seconds": elapsed,
        "life_anchor_fitness": life_report.fitness if life_report else None,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=_json_default))

    print(f"\nDone. Out dir: {out_dir}")
    if reports:
        print(f"Top rule: {reports[0].rule.to_dict()}")
        print(f"  fitness ({reports[0].fitness_mode}): {reports[0].fitness:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
