"""Random search over 4D fractional totalistic rules for viability (M4A)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from observer_worlds.search import (
    ViabilityWeights,
    evaluate_viability_multi_seed,
    sample_random_fractional_rule,
)
from observer_worlds.search.leaderboard import (
    write_leaderboard_csv,
    write_leaderboard_json,
    write_top_k_artifacts,
)
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import DetectionConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search 4D fractional rules for viability.")
    p.add_argument("--n-rules", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8],
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="outputs/rule_search/viability")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the rule-sampling RNG.")
    p.add_argument("--base-eval-seed", type=int, default=10000,
                   help="Base seed for per-rule scout simulations; each rule "
                        "uses base_eval_seed..base_eval_seed+seeds-1.")
    p.add_argument("--backend", choices=["numba", "numpy", "cuda"], default="numba")
    p.add_argument("--min-component-age", type=int, default=20)
    p.add_argument("--early-abort", action="store_true", default=True)
    p.add_argument("--no-early-abort", dest="early_abort", action="store_false")
    p.add_argument("--no-top-k-artifacts", action="store_true",
                   help="Skip per-rule artifact directory generation (CSV/JSON only).")
    p.add_argument("--progress-every", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = seeded_rng(args.seed)
    detection_config = DetectionConfig()
    weights = ViabilityWeights()  # defaults

    print(f"Searching {args.n_rules} rules x {args.seeds} seeds x {args.timesteps} steps "
          f"on grid {tuple(args.grid)} (backend={args.backend})")
    print(f"  out_dir: {out_dir}")
    print()

    reports = []
    t0 = time.time()
    for i in range(args.n_rules):
        rule = sample_random_fractional_rule(rng)
        rep = evaluate_viability_multi_seed(
            rule,
            n_seeds=args.seeds,
            base_seed=args.base_eval_seed,
            grid_shape=tuple(args.grid),
            timesteps=args.timesteps,
            detection_config=detection_config,
            backend=args.backend,
            weights=weights,
            min_component_age=args.min_component_age,
            early_abort=args.early_abort,
        )
        reports.append(rep)
        if (i + 1) % args.progress_every == 0 or (i + 1) == args.n_rules:
            elapsed = time.time() - t0
            best = max(reports, key=lambda r: r.viability_score)
            print(f"  [{i+1}/{args.n_rules}] elapsed={elapsed:.0f}s  "
                  f"best score so far: {best.viability_score:+.3f}  "
                  f"({best.rule.short_repr()})")

    # Sort and write artifacts.
    reports.sort(key=lambda r: -r.viability_score)
    write_leaderboard_csv(reports, out_dir / "leaderboard.csv")
    write_leaderboard_json(reports, out_dir / "leaderboard.json")
    if not args.no_top_k_artifacts:
        write_top_k_artifacts(
            reports, out_dir / "top_k",
            top_k=args.top_k,
            grid_shape=tuple(args.grid),
            timesteps=args.timesteps,
            base_seed=args.base_eval_seed,
            detection_config=detection_config,
            backend=args.backend,
        )

    # Print top-K table.
    print()
    print(f"Top {min(args.top_k, len(reports))} rules by viability_score:")
    print(f"{'rank':>4}  {'score':>8}  {'n_persist':>9}  {'max_age':>7}  rule")
    for i, r in enumerate(reports[:args.top_k]):
        print(f"  {i+1:>2}  {r.viability_score:>+8.3f}  "
              f"{r.n_persistent_components:>9.1f}  {r.max_component_lifetime:>7d}  "
              f"{r.rule.short_repr()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
