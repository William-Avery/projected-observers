"""Hidden-shuffled baseline (M3).

Runs the same 4D CA as :mod:`run_4d_projection`, but at each timestep
(subject to ``--shuffle-every``), the (z, w) values within every (x, y)
column are randomly permuted *before projection*.  This preserves the
per-column active-cell count (so the projection's mean-threshold output
statistics are unchanged in distribution) while destroying any coherent
organisation in the hidden dimensions.

Hypothesis under test (M3): 4D-projected structures with coherent hidden
dynamics score higher on observer-likeness metrics than the same
simulation with hidden coherence destroyed.

The state mutator hook in
:func:`observer_worlds.experiments._pipeline.simulate_4d_to_zarr` is the
splice-point: ``(state, t, rng) -> mutated_state`` is called every step,
after the CA update but before projection.  Because the helper passes us
``state.copy()``, it is safe (and faster) to mutate in place.

Implementation note: every (x, y) column is shuffled, including columns
with all-zero or all-one fibers.  Shuffling those is a no-op, so this is
correct and avoids a per-column branch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
        description="Hidden-shuffled baseline: 4D CA with z,w fibers permuted "
                    "per (x, y) column before each projection."
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--grid", type=int, nargs=4, default=None,
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", choices=["numba", "numpy"], default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--save-4d-snapshots", action="store_true",
                   help="Save 4D state at each snapshot_interval (needed for "
                        "causality + resilience scores). For this baseline, "
                        "snapshots store the *post-shuffle* state.")
    p.add_argument("--snapshot-interval", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=10)
    p.add_argument("--shuffle-every", type=int, default=1,
                   help="Apply the column-wise z,w shuffle every N steps "
                        "(default 1 = every step). Steps where t %% N != 0 "
                        "leave the state unchanged.")
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


def make_hidden_shuffle_mutator(shuffle_every: int):
    """Build a state-mutator that permutes (z, w) values per (x, y) column.

    The mutator preserves the count of active cells in each column (the
    invariant that keeps mean-threshold projection statistics comparable
    to the un-shuffled run) while destroying any coherent ordering of
    activity along the hidden axes.

    Vectorisation: we reshape the 4D state to ``(Nx*Ny, Nz*Nw)`` and
    permute each row using an ``argsort`` of uniform random keys.  This
    runs in a single numpy pass — no Python loop over columns.

    The returned closure captures ``shuffle_every`` only; the RNG used to
    draw the permutation keys is the one plumbed through
    :func:`simulate_4d_to_zarr`, so the run remains reproducible from
    ``cfg.seed``.
    """
    every = max(1, int(shuffle_every))

    def mutator(state: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        if t % every != 0:
            return state
        nx, ny, nz, nw = state.shape
        flat = state.reshape(nx * ny, nz * nw)
        # Vectorised per-row permutation via argsort on random keys.
        keys = rng.random(flat.shape)
        order = np.argsort(keys, axis=1)
        shuffled = np.take_along_axis(flat, order, axis=1)
        # Preserve dtype (uint8) and the original 4D shape.  Writing back
        # through `flat` would alias the input; build the final array
        # explicitly to be safe.
        return shuffled.reshape(nx, ny, nz, nw).astype(state.dtype, copy=False)

    return mutator


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    cfg = build_run_config(args)
    rng = seeded_rng(cfg.seed)

    print(
        f"observer_worlds — shuffled-hidden baseline label={cfg.label!r} "
        f"seed={cfg.seed} shuffle_every={args.shuffle_every}"
    )
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

    mutator = make_hidden_shuffle_mutator(args.shuffle_every)

    print("[1/5] simulating + projecting (with hidden-axis shuffle)...")
    simulate_4d_to_zarr(cfg, store, rng, state_mutator=mutator)

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
    extra_lines = [
        f"- Shuffle: every {args.shuffle_every} step(s); preserves per-column "
        f"active-cell counts, destroys coherent (z, w) structure",
        "- All (x, y) columns are shuffled (zero-active columns are no-ops)",
    ]
    summary = build_summary(
        cfg, tracks, candidates, observer_scores, run_dir,
        world_kind="shuffled_4d",
        extra_lines=extra_lines,
    )
    store.write_summary_md(summary)

    print(f"\nDone. Run dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
