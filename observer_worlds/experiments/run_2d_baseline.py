"""Experiment 2 — 2D baseline (Conway's Life by default) with M2 metrics.

This is the M3 control experiment: run a flat 2D CA on a 2D grid (no 4D
fibers, no projection) and score the resulting structures with the same
observer-likeness metrics used by the 4D experiment.  Causality and
resilience are skipped because there is no 4D bulk to perturb — those
metric semantics differ across dimensionalities, so it would not be a
fair comparison to compute them here.

The hypothesis under test (M3): 4D-projected structures score higher
than matched 2D-native structures.  This script produces the control
side of that comparison.

Pipeline:
    1. Run a 2D CA (default Conway's Life, B3/S23) and write each frame.
    2. Detect connected components per frame, link into tracks.
    3. Filter to observer-candidates by persistence.
    4. For every candidate: compute time, memory, selfhood (no causality
       or resilience for 2D worlds).
    5. Combine into observer_score (z-normalized across the run population).
    6. Write CSVs, GIF, plots, summary.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observer_worlds.experiments._pipeline import (
    build_summary,
    compute_full_metrics,
    detect_and_track,
    simulate_2d_to_zarr,
    write_observer_scores_csv,
    write_plots_and_gif,
)
from observer_worlds.metrics import score_persistence
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import RunConfig, seeded_rng
from observer_worlds.worlds import BSRule


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the 2D baseline observer_worlds experiment (default: "
            "Conway's Life) with the M2 observer-likeness metrics."
        )
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--grid", type=int, nargs=2, default=None,
                   metavar=("NX", "NY"),
                   help="2D grid size (Nx Ny). Hidden dims nz, nw are forced to 1.")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--initial-density", type=float, default=None,
                   help="Initial random density of live cells in [0,1]. "
                        "Overrides cfg.world.initial_density. Conway's Life "
                        "needs ~0.3 to produce sustained activity; the "
                        "default config value (0.15) often dies out.")
    p.add_argument("--rule-birth", type=int, nargs="+", default=None,
                   metavar="K",
                   help="Birth neighbour counts (e.g. --rule-birth 3 for "
                        "Life). Default: Conway's Life B3.")
    p.add_argument("--rule-survival", type=int, nargs="+", default=None,
                   metavar="K",
                   help="Survival neighbour counts (e.g. --rule-survival 2 "
                        "3 for Life). Default: Conway's Life S23.")
    p.add_argument("--rollout-steps", type=int, default=10,
                   help="Unused for 2D worlds (no causality/resilience), "
                        "but accepted for CLI symmetry with the 4D script.")
    return p


def build_run_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.load(args.config) if args.config else RunConfig()

    # 2D world: hidden dims are trivial.  Force nz = nw = 1 so the dataclass
    # stays consistent (some code paths inspect cfg.world.shape).
    if args.grid is not None:
        cfg.world.nx, cfg.world.ny = args.grid
    cfg.world.nz = 1
    cfg.world.nw = 1

    if args.timesteps is not None:
        cfg.world.timesteps = args.timesteps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.label is not None:
        cfg.label = args.label
    if args.output_root is not None:
        cfg.output.output_root = args.output_root
    if args.no_gif:
        cfg.output.save_gif = False
    if args.initial_density is not None:
        cfg.world.initial_density = args.initial_density

    # No 4D state in this experiment.
    cfg.output.save_4d_snapshots = False

    # Round-trip through dict so __post_init__ validators run on overrides.
    return RunConfig.from_dict(cfg.to_dict())


def build_rule(args: argparse.Namespace) -> BSRule:
    """Build the 2D rule from CLI flags. Defaults to Conway's Life (B3/S23)."""
    if args.rule_birth is None and args.rule_survival is None:
        return BSRule.life()
    birth = tuple(args.rule_birth) if args.rule_birth is not None else (3,)
    survival = tuple(args.rule_survival) if args.rule_survival is not None else (2, 3)
    return BSRule(birth=birth, survival=survival)


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    cfg = build_run_config(args)
    rule_2d = build_rule(args)
    rng = seeded_rng(cfg.seed)

    print(f"observer_worlds — 2D baseline label={cfg.label!r} seed={cfg.seed}")
    print(f"  grid=({cfg.world.nx}, {cfg.world.ny}) "
          f"timesteps={cfg.world.timesteps}")
    print(f"  rule: B={rule_2d.birth}, S={rule_2d.survival} "
          f"initial_density={cfg.world.initial_density}")

    run_dir = ZarrRunStore.make_run_dir(cfg.output.output_root, cfg.label)
    store = ZarrRunStore(
        run_dir,
        timesteps=cfg.world.timesteps,
        shape_2d=(cfg.world.nx, cfg.world.ny),
        save_4d_snapshots=False,
    )
    store.write_config_json(cfg)

    print("[1/5] simulating 2D CA...")
    simulate_2d_to_zarr(cfg, rule_2d, store, rng)

    print("[2/5] detecting + tracking...")
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)

    print("[3/5] scoring persistence...")
    candidates = score_persistence(
        tracks, grid_shape=(cfg.world.nx, cfg.world.ny), config=cfg.detection
    )
    store.write_tracks_csv(tracks)
    store.write_candidates_csv(candidates)

    print("[4/5] computing M2 metrics for candidates "
          "(causality + resilience skipped for 2D)...")
    observer_scores, per_candidate = compute_full_metrics(
        cfg, tracks, candidates, store,
        rollout_steps=args.rollout_steps,
        world_kind="2d",
    )
    write_observer_scores_csv(
        store.data_dir / "observer_scores.csv", observer_scores, per_candidate
    )

    print("[5/5] writing plots + GIF + summary...")
    write_plots_and_gif(cfg, store, frames, tracks, candidates)
    extra_lines = [
        f"- 2D rule: B={list(rule_2d.birth)}, S={list(rule_2d.survival)}",
        f"- Initial density: {cfg.world.initial_density}",
    ]
    summary = build_summary(
        cfg, tracks, candidates, observer_scores, run_dir,
        world_kind="2d", extra_lines=extra_lines,
    )
    store.write_summary_md(summary)

    print(f"\nDone. Run dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
