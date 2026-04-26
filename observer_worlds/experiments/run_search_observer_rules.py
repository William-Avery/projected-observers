"""M4C — observer-metric-guided rule search (random or evolutionary).

For random search: samples N rules and evaluates each on the full M2
metric suite, ranking by the chosen fitness mode (default
``lifetime_weighted``).

For evolutionary search: runs a textbook ``(μ+λ)`` loop on top of the
random search base.

Outputs a leaderboard CSV/JSON, a top-K artifact directory (with
re-runnable RunConfigs), and (for evolve) a per-generation history file.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.search import (
    DEFAULT_MUTATION_SIGMAS,
    FITNESS_MODES,
    FractionalRule,
    ObserverFitnessReport,
    evolutionary_search_observer,
    random_search_observer,
)
from observer_worlds.utils import RunConfig
from observer_worlds.utils.config import DetectionConfig, OutputConfig, ProjectionConfig, WorldConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M4C observer-metric-guided rule search.")
    p.add_argument("--strategy", choices=["random", "evolve"], required=True)
    # Random:
    p.add_argument("--n-rules", type=int, default=50)
    # Evolve:
    p.add_argument("--n-generations", type=int, default=5)
    p.add_argument("--mu", type=int, default=8)
    p.add_argument("--lam", type=int, default=8)
    # Common:
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--base-eval-seed", type=int, default=1000)
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--grid", type=int, nargs=4, default=[32, 32, 4, 4])
    p.add_argument("--backend", choices=["numba", "numpy"], default="numba")
    p.add_argument("--fitness-mode", choices=list(FITNESS_MODES),
                   default="lifetime_weighted")
    p.add_argument("--rollout-steps", type=int, default=6)
    p.add_argument("--snapshots-per-run", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Defaults to outputs/observer_search/observer_<UTC>/")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed-population", type=str, default=None,
                   help="Path to an M4A leaderboard.json; loads top-mu rules "
                        "as the initial evolutionary population.")
    return p


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "rank", "fitness", "fitness_mode", "n_seeds",
    "birth_min", "birth_max", "survive_min", "survive_max", "initial_density",
    "mean_n_tracks", "mean_n_candidates", "mean_max_score", "mean_top5_mean_score",
    "mean_p95_score", "mean_lifetime_weighted_mean_score", "mean_score_per_track",
    "mean_late_active", "mean_max_component_lifetime",
    "per_seed_fitness", "per_seed_n_candidates", "per_seed_n_tracks",
    "aborted_seeds", "sim_time_seconds",
)


def _write_leaderboard_csv(reports: list[ObserverFitnessReport], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LEADERBOARD_COLUMNS)
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
                ";".join(str(x) for x in r.per_seed_n_candidates),
                ";".join(str(x) for x in r.per_seed_n_tracks),
                r.aborted_seeds, f"{r.sim_time_seconds:.3f}",
            ])


def _json_default(obj):
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _write_leaderboard_json(reports: list[ObserverFitnessReport], path: Path) -> None:
    entries = []
    for rank, r in enumerate(reports, start=1):
        entries.append({
            "rank": rank,
            "fitness": float(r.fitness),
            "fitness_mode": r.fitness_mode,
            "n_seeds": r.n_seeds,
            "rule": r.rule.to_dict(),
            "mean_n_tracks": r.mean_n_tracks,
            "mean_n_candidates": r.mean_n_candidates,
            "mean_max_score": r.mean_max_score,
            "mean_top5_mean_score": r.mean_top5_mean_score,
            "mean_p95_score": r.mean_p95_score,
            "mean_lifetime_weighted_mean_score": r.mean_lifetime_weighted_mean_score,
            "mean_score_per_track": r.mean_score_per_track,
            "mean_late_active": r.mean_late_active,
            "mean_max_component_lifetime": r.mean_max_component_lifetime,
            "per_seed_fitness": r.per_seed_fitness,
            "per_seed_n_candidates": r.per_seed_n_candidates,
            "per_seed_n_tracks": r.per_seed_n_tracks,
            "aborted_seeds": r.aborted_seeds,
            "sim_time_seconds": r.sim_time_seconds,
            "seeds_used": r.seeds_used,
        })
    path.write_text(json.dumps(entries, indent=2, default=_json_default))


def _build_run_config(rule: FractionalRule, *, grid_shape, timesteps, backend, base_seed, label) -> RunConfig:
    bs = rule.to_bsrule()
    cfg = RunConfig(
        world=WorldConfig(
            nx=grid_shape[0], ny=grid_shape[1], nz=grid_shape[2], nw=grid_shape[3],
            timesteps=timesteps,
            initial_density=rule.initial_density,
            rule_birth=tuple(int(x) for x in bs.birth),
            rule_survival=tuple(int(x) for x in bs.survival),
            backend=backend,
        ),
        projection=ProjectionConfig(method="mean_threshold", theta=0.5),
        detection=DetectionConfig(),
        output=OutputConfig(),
        seed=base_seed,
        label=label,
    )
    return cfg


def _write_top_k_artifacts(
    reports: list[ObserverFitnessReport],
    out_dir: Path,
    *,
    top_k: int,
    grid_shape, timesteps, backend, base_seed,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, r in enumerate(reports[:top_k], start=1):
        sub = out_dir / f"rule_{rank:03d}"
        sub.mkdir(parents=True, exist_ok=True)
        cfg = _build_run_config(
            r.rule, grid_shape=grid_shape, timesteps=timesteps,
            backend=backend, base_seed=base_seed,
            label=f"observer_rule_{rank}_{r.rule.short_repr()}",
        )
        cfg.save(sub / "config.json")
        rule_payload = {
            "rank": rank,
            "fitness": float(r.fitness),
            "fitness_mode": r.fitness_mode,
            "rule": r.rule.to_dict(),
            "mean_n_candidates": r.mean_n_candidates,
            "mean_top5_mean_score": r.mean_top5_mean_score,
            "mean_lifetime_weighted_mean_score": r.mean_lifetime_weighted_mean_score,
            "mean_score_per_track": r.mean_score_per_track,
            "per_seed_fitness": r.per_seed_fitness,
        }
        (sub / "rule.json").write_text(
            json.dumps(rule_payload, indent=2, default=_json_default)
        )


def _print_top_table(reports: list[ObserverFitnessReport], top_k: int) -> None:
    print()
    print(f"Top {min(top_k, len(reports))} rules by {reports[0].fitness_mode if reports else 'fitness'}:")
    print(f"{'rank':>4} {'fitness':>10} {'top5':>9} {'lwm':>9} {'sptr':>9} "
          f"{'n_cand':>7} {'rule'}")
    for i, r in enumerate(reports[:top_k], start=1):
        print(f"  {i:>2} {r.fitness:>+10.4f} {r.mean_top5_mean_score:>+9.3f} "
              f"{r.mean_lifetime_weighted_mean_score:>+9.3f} "
              f"{r.mean_score_per_track:>+9.4f} "
              f"{r.mean_n_candidates:>7.0f} {r.rule.short_repr()}")


# ---------------------------------------------------------------------------
# Optional: load initial evolutionary population from M4A leaderboard
# ---------------------------------------------------------------------------


def _load_initial_population(path: Path, mu: int) -> list[FractionalRule]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected format at {path}")
    if "rule" in data[0]:
        # M4A leaderboard.json format.
        entries = sorted(data, key=lambda e: -float(e.get("viability_score", 0.0)))
        return [FractionalRule.from_dict(e["rule"]) for e in entries[:mu]]
    if "birth_min" in data[0]:
        return [FractionalRule.from_dict(d) for d in data[:mu]]
    raise ValueError(f"could not parse initial population from {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # ---------------- output dir
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = f"outputs/observer_search/observer_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_shape = tuple(args.grid)

    print(f"M4C observer search ({args.strategy}) -> {out_dir}")
    print(f"  fitness_mode={args.fitness_mode}  n_seeds={args.n_seeds}  "
          f"T={args.timesteps}  grid={grid_shape}  backend={args.backend}")

    history: list[dict] = []

    t0 = time.time()
    if args.strategy == "random":
        reports = random_search_observer(
            n_rules=args.n_rules,
            n_seeds=args.n_seeds,
            base_seed=args.base_eval_seed,
            sampler_seed=args.sampler_seed,
            grid_shape=grid_shape,
            timesteps=args.timesteps,
            backend=args.backend,
            fitness_mode=args.fitness_mode,
            rollout_steps=args.rollout_steps,
            snapshots_per_run=args.snapshots_per_run,
            progress=print,
        )
    else:
        initial_population = None
        if args.seed_population is not None:
            initial_population = _load_initial_population(
                Path(args.seed_population), args.mu
            )
            print(f"  seeded initial population from {args.seed_population} "
                  f"({len(initial_population)} rules)")
        reports, history = evolutionary_search_observer(
            n_generations=args.n_generations,
            mu=args.mu, lam=args.lam,
            n_seeds=args.n_seeds,
            base_seed=args.base_eval_seed,
            sampler_seed=args.sampler_seed,
            grid_shape=grid_shape,
            timesteps=args.timesteps,
            backend=args.backend,
            fitness_mode=args.fitness_mode,
            rollout_steps=args.rollout_steps,
            snapshots_per_run=args.snapshots_per_run,
            initial_population=initial_population,
            progress=print,
        )
    elapsed = time.time() - t0
    print(f"\nsearch done in {elapsed:.0f}s")

    # ---------------- write outputs
    _write_leaderboard_csv(reports, out_dir / "leaderboard.csv")
    _write_leaderboard_json(reports, out_dir / "leaderboard.json")
    _write_top_k_artifacts(
        reports, out_dir / "top_k",
        top_k=args.top_k, grid_shape=grid_shape, timesteps=args.timesteps,
        backend=args.backend, base_seed=args.base_eval_seed,
    )
    if history:
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=_json_default)
        )

    config_dump = {
        "strategy": args.strategy,
        "n_seeds": args.n_seeds,
        "fitness_mode": args.fitness_mode,
        "grid": list(grid_shape),
        "timesteps": args.timesteps,
        "backend": args.backend,
        "base_eval_seed": args.base_eval_seed,
        "sampler_seed": args.sampler_seed,
        "rollout_steps": args.rollout_steps,
        "snapshots_per_run": args.snapshots_per_run,
        "elapsed_seconds": elapsed,
    }
    if args.strategy == "random":
        config_dump["n_rules"] = args.n_rules
    else:
        config_dump["n_generations"] = args.n_generations
        config_dump["mu"] = args.mu
        config_dump["lam"] = args.lam
        config_dump["seed_population"] = args.seed_population
    (out_dir / "config.json").write_text(
        json.dumps(config_dump, indent=2, default=_json_default)
    )

    _print_top_table(reports, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
