"""Experiment 1 — 4D-to-2D simulation with full M2 observer-likeness metrics.

Pipeline:
    1. Run the 4D CA, project to 2D each step.  Optionally save 4D snapshots
       at fixed intervals (needed for causality + resilience scores).
    2. Detect connected components per frame, link into tracks.
    3. Filter to observer-candidates by persistence.
    4. For every candidate: compute time, memory, selfhood, causality
       (when 4D snapshots are available), resilience.
    5. Combine into observer_score (z-normalized across the run population).
    6. Write CSVs, GIF, plots, summary.md.

The default heuristic 4D rule will likely produce trivial dynamics; rule
search (M4) is the principled way to find rules that produce persistent
bounded structures with internal variation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from observer_worlds.experiments._pipeline import (
    build_summary,
    compute_full_metrics,
    detect_and_track,
    simulate_4d_to_zarr,
    write_observer_scores_csv,
    write_plots_and_gif,
)
from observer_worlds.metrics import score_persistence
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import RunConfig, seeded_rng


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the 4D-to-2D observer_worlds experiment with M2 metrics."
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--grid", type=int, nargs=4, default=None,
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", choices=["numba", "numpy", "cuda"], default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--save-4d-snapshots", action="store_true",
                   help="Save 4D state at each snapshot_interval (needed for "
                        "causality + resilience scores).")
    p.add_argument("--snapshot-interval", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=10)
    return p


def build_run_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.load(args.config) if args.config else RunConfig()
    if args.grid is not None:
        cfg.world.nx, cfg.world.ny, cfg.world.nz, cfg.world.nw = args.grid
    if args.timesteps is not None:
        cfg.world.timesteps = args.timesteps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.backend is not None:
        cfg.world.backend = args.backend
    if args.label is not None:
        cfg.label = args.label
    if args.output_root is not None:
        cfg.output.output_root = args.output_root
    if args.no_gif:
        cfg.output.save_gif = False
    if args.save_4d_snapshots:
        cfg.output.save_4d_snapshots = True
    if args.snapshot_interval is not None:
        cfg.output.snapshot_interval = args.snapshot_interval
    return RunConfig.from_dict(cfg.to_dict())


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    cfg = build_run_config(args)
    rng = seeded_rng(cfg.seed)

    print(f"observer_worlds — 4D run label={cfg.label!r} seed={cfg.seed}")
    print(f"  grid={cfg.world.shape} timesteps={cfg.world.timesteps} "
          f"backend={cfg.world.backend}")

    run_dir = ZarrRunStore.make_run_dir(cfg.output.output_root, cfg.label)
    store = ZarrRunStore(
        run_dir,
        timesteps=cfg.world.timesteps,
        shape_2d=(cfg.world.nx, cfg.world.ny),
        save_4d_snapshots=cfg.output.save_4d_snapshots,
        shape_4d=cfg.world.shape if cfg.output.save_4d_snapshots else None,
    )
    store.write_config_json(cfg)

    print("[1/5] simulating + projecting...")
    simulate_4d_to_zarr(cfg, store, rng)

    print("[2/5] detecting + tracking...")
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)

    print("[3/5] scoring persistence...")
    candidates = score_persistence(
        tracks, grid_shape=(cfg.world.nx, cfg.world.ny), config=cfg.detection
    )
    store.write_tracks_csv(tracks)
    store.write_candidates_csv(candidates)

    print("[4/5] computing M2 metrics for candidates...")
    observer_scores, per_candidate = compute_full_metrics(
        cfg, tracks, candidates, store,
        rollout_steps=args.rollout_steps,
        world_kind="4d",
    )
    write_observer_scores_csv(
        store.data_dir / "observer_scores.csv", observer_scores, per_candidate
    )

    print("[5/5] writing plots + GIF + summary...")
    write_plots_and_gif(cfg, store, frames, tracks, candidates)
    summary = build_summary(
        cfg, tracks, candidates, observer_scores, run_dir, world_kind="4d"
    )
    store.write_summary_md(summary)

    print(f"\nDone. Run dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
