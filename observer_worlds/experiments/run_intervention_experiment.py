"""M5 — intervention experiment driver.

Two modes:
    --from-run PATH   : open an existing 4D run dir (with 4D snapshots),
                        re-detect tracks, re-score persistence + observer,
                        and run interventions on the top-K candidates.
    --config PATH     : start fresh from a RunConfig JSON; runs the 4D
                        experiment with snapshotting forced on, then
                        proceeds as in --from-run.

For each top-K observer-candidate, applies all four intervention types
at the latest snapshot inside its lifetime, runs paired rollouts, and
records per-step divergence trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis import write_all_m5_plots
from observer_worlds.experiments._m5_interventions import (
    INTERVENTION_TYPES,
    aggregate_intervention_summaries,
    run_candidate_interventions,
)
from observer_worlds.experiments._pipeline import (
    compute_full_metrics,
    detect_and_track,
    simulate_4d_to_zarr,
)
from observer_worlds.metrics import score_persistence
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import RunConfig, seeded_rng
from observer_worlds.worlds import BSRule


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M5 intervention experiment.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run", type=str,
                     help="Existing run dir with snapshots (data/states.zarr).")
    src.add_argument("--config", type=str,
                     help="RunConfig JSON to run a fresh 4D experiment with snapshots.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--flip-fraction", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--label", type=str, default="m5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["numba", "numpy", "cuda"], default="numpy")
    return p


def _load_or_run_4d(args: argparse.Namespace, out_dir: Path):
    """Returns (cfg, store, frames, tracks, candidates, observer_scores)."""
    if args.from_run:
        run_dir = Path(args.from_run)
        cfg = RunConfig.load(run_dir / "config.json")
        # Wrap the existing zarr store. ZarrRunStore re-creates 'frames_2d'
        # if mode='w'; we need a read path. Use the underlying zarr API.
        import zarr
        store = ZarrRunStore.__new__(ZarrRunStore)
        store._run_dir = run_dir
        store._data_dir = run_dir / "data"
        store._frames_dir = run_dir / "frames"
        store._plots_dir = run_dir / "plots"
        store._zarr_path = store._data_dir / "states.zarr"
        store._root = zarr.open(str(store._zarr_path), mode="r")
        store._frames_2d = store._root["frames_2d"]
        store._save_4d_snapshots = "snapshots_4d" in store._root
        store._snapshots_group = (
            store._root["snapshots_4d"] if store._save_4d_snapshots else None
        )
        store._shape_2d = store._frames_2d.shape[1:]
        store._shape_4d = None
        if store._snapshots_group is not None:
            for name in store._snapshots_group:
                arr = store._snapshots_group[name]
                store._shape_4d = arr.shape
                break
    else:
        cfg = RunConfig.load(args.config)
        cfg.output.save_4d_snapshots = True
        if cfg.output.snapshot_interval <= 0:
            cfg.output.snapshot_interval = max(1, cfg.world.timesteps // 8)
        run_root = out_dir / "run"
        run_root.mkdir(parents=True, exist_ok=True)
        store = ZarrRunStore(
            run_root,
            timesteps=cfg.world.timesteps,
            shape_2d=(cfg.world.nx, cfg.world.ny),
            save_4d_snapshots=True,
            shape_4d=cfg.world.shape,
        )
        store.write_config_json(cfg)
        rng = seeded_rng(cfg.seed)
        print("[fresh] simulating 4D...")
        simulate_4d_to_zarr(cfg, store, rng)

    print("[1/3] detecting + tracking...")
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)
    print("[2/3] scoring persistence + observer metrics...")
    candidates = score_persistence(
        tracks, grid_shape=(cfg.world.nx, cfg.world.ny), config=cfg.detection
    )
    observer_scores, per_candidate = compute_full_metrics(
        cfg, tracks, candidates, store, rollout_steps=6, world_kind="4d",
    )
    return cfg, store, frames, tracks, candidates, observer_scores, per_candidate


def _pick_snapshot(track, snap_times: list[int]) -> int | None:
    for t in reversed(snap_times):
        if track.birth_frame <= t <= track.last_frame:
            return t
    return None


def _mask_for_track(track, frame_idx: int, kind: str) -> np.ndarray | None:
    if frame_idx in track.frames:
        i = track.frames.index(frame_idx)
    else:
        nearest = min(track.frames, key=lambda f: abs(f - frame_idx))
        i = track.frames.index(nearest)
    # For thin candidates (1-2 cells wide), the eroded interior is empty
    # and the boundary spans the whole mask.  Fall back to the candidate's
    # own mask as "interior" in that case so interventions still target the
    # candidate.  Boundary similarly falls back to mask.  Env stays as the
    # shell -- env is computed by dilation, not erosion, so it's well-defined
    # even for thin structures.
    if kind == "interior":
        m = track.interior_history[i]
        return m if m.any() else track.mask_history[i]
    if kind == "boundary":
        m = track.boundary_history[i]
        return m if m.any() else track.mask_history[i]
    if kind == "env":
        return track.env_history[i]
    raise ValueError(kind)


def _write_intervention_summary_csv(reports, path: Path) -> None:
    cols = [
        "track_id", "track_age", "snapshot_t", "observer_score",
        "intervention_type", "n_steps", "flip_fraction",
        "interior_size", "boundary_size", "env_size",
        "mean_full_grid_l1", "mean_candidate_footprint_l1", "auc_full_grid_l1",
        "final_survival", "final_area_ratio",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in reports:
            for kind, traj in r.trajectories.items():
                w.writerow([
                    r.track_id, r.track_age, r.snapshot_t,
                    "" if r.observer_score is None else f"{r.observer_score:.4f}",
                    kind, r.n_steps, r.flip_fraction,
                    r.interior_size, r.boundary_size, r.env_size,
                    f"{traj.mean_full_grid_l1:.6f}",
                    f"{traj.mean_candidate_footprint_l1:.6f}",
                    f"{traj.auc_full_grid_l1:.6f}",
                    int(traj.final_survival),
                    f"{traj.final_area_ratio:.6f}",
                ])


def _write_trajectories_json(reports, path: Path) -> None:
    out = []
    for r in reports:
        rep = {
            "track_id": r.track_id,
            "track_age": r.track_age,
            "snapshot_t": r.snapshot_t,
            "observer_score": r.observer_score,
            "n_steps": r.n_steps,
            "flip_fraction": r.flip_fraction,
            "interior_size": r.interior_size,
            "boundary_size": r.boundary_size,
            "env_size": r.env_size,
            "trajectories": {},
        }
        for kind, t in r.trajectories.items():
            rep["trajectories"][kind] = {
                "n_steps": t.n_steps,
                "flip_fraction": t.flip_fraction,
                "full_grid_l1": t.full_grid_l1,
                "candidate_footprint_l1": t.candidate_footprint_l1,
                "candidate_active_orig": t.candidate_active_orig,
                "candidate_active_intervened": t.candidate_active_intervened,
                "mean_full_grid_l1": t.mean_full_grid_l1,
                "mean_candidate_footprint_l1": t.mean_candidate_footprint_l1,
                "auc_full_grid_l1": t.auc_full_grid_l1,
                "final_survival": t.final_survival,
                "final_area_ratio": t.final_area_ratio,
            }
        out.append(rep)
    path.write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else int(o) if isinstance(o, np.integer) else str(o)))


def _build_summary_md(reports, agg, top_k: int, out_dir: Path) -> str:
    lines = [f"# M5 intervention experiment — {out_dir.name}", ""]
    lines.append(f"- Top-K candidates analyzed: {len(reports)}")
    lines.append(f"- Intervention types: {', '.join(INTERVENTION_TYPES)}")
    if reports:
        lines.append(f"- Rollout steps: {reports[0].n_steps}")
        lines.append(f"- Flip fraction: {reports[0].flip_fraction}")
    lines.append("")
    lines.append("## Aggregate intervention summary")
    lines.append("")
    if not agg:
        lines.append("No intervention reports.")
    else:
        lines.append("| intervention | mean_full_l1 | mean_cand_footprint_l1 | final_area_ratio | survival_rate |")
        lines.append("|---|---|---|---|---|")
        for kind in INTERVENTION_TYPES:
            if kind not in agg:
                continue
            s = agg[kind]
            lines.append(
                f"| {kind} | {s.get('mean_full_grid_l1', 0.0):.4f} | "
                f"{s.get('mean_candidate_footprint_l1', 0.0):.4f} | "
                f"{s.get('final_area_ratio', 0.0):.3f} | "
                f"{s.get('final_survival', 0.0):.2f} |"
            )

        # Highlights.
        worst_div = max(
            agg.items(),
            key=lambda kv: kv[1].get("mean_full_grid_l1", 0.0),
        )
        most_killing = min(
            agg.items(),
            key=lambda kv: kv[1].get("final_survival", 1.0),
        )
        lines.append("")
        lines.append(
            f"- **Largest mean divergence**: `{worst_div[0]}` "
            f"(mean_full_l1={worst_div[1]['mean_full_grid_l1']:.4f})"
        )
        lines.append(
            f"- **Most candidate-killing**: `{most_killing[0]}` "
            f"(survival_rate={most_killing[1].get('final_survival', 0.0):.2f})"
        )

    lines.append("")
    lines.append("## Per-candidate breakdown")
    lines.append("")
    if not reports:
        lines.append("No candidates.")
    else:
        lines.append("| track_id | snapshot_t | obs_score | interior_size | most_disruptive |")
        lines.append("|---|---|---|---|---|")
        for r in reports:
            best_kind = max(
                r.intervention_summary.items(),
                key=lambda kv: kv[1]["mean_full_grid_l1"],
                default=(None, None),
            )[0]
            lines.append(
                f"| {r.track_id} | {r.snapshot_t} | "
                f"{r.observer_score:+.3f} | {r.interior_size} | "
                f"{best_kind or '—'} |"
            )
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `intervention_summary.csv` — one row per (track_id, intervention_type)")
    lines.append("- `intervention_trajectories.json` — full per-step lists")
    lines.append("- `plots/aggregate_divergence_*.png`")
    lines.append("- `plots/intervention_heatmap_*.png`, `plots/intervention_summary_bars.png`")
    lines.append("- `plots/per_candidate/track_<id>_{divergence,resilience}.png`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = f"outputs/{args.label}_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg, store, frames, tracks, candidates, observer_scores, per_candidate = \
        _load_or_run_4d(args, out_dir)

    snap_times = store.list_snapshots()
    print(f"  available snapshots: {len(snap_times)}")
    if not snap_times:
        print("ERROR: no 4D snapshots in run; cannot run interventions.")
        return 1

    # Sort observer-candidates by combined score; walk down the list and
    # keep the first top_k that have a valid snapshot inside their
    # lifetime AND non-degenerate interior/boundary/env masks at that
    # snapshot.  Top observer-scored candidates are often tiny structures
    # whose interior collapses to empty under erosion, so skipping is
    # common; widen the search past top_k as needed.
    sorted_obs = sorted(observer_scores, key=lambda o: -o.combined)
    track_by_id = {t.track_id: t for t in tracks}
    rule = BSRule(birth=cfg.world.rule_birth, survival=cfg.world.rule_survival)

    print(f"[3/3] running interventions: searching for {args.top_k} eligible candidates...")
    reports = []
    n_skipped = 0
    t0 = time.time()
    for i, obs in enumerate(sorted_obs):
        if len(reports) >= args.top_k:
            break
        tr = track_by_id.get(obs.track_id)
        if tr is None:
            n_skipped += 1
            continue
        snap_t = _pick_snapshot(tr, snap_times)
        if snap_t is None:
            n_skipped += 1
            continue
        interior = _mask_for_track(tr, snap_t, "interior")
        boundary = _mask_for_track(tr, snap_t, "boundary")
        env = _mask_for_track(tr, snap_t, "env")
        if not (interior.any() and boundary.any() and env.any()):
            n_skipped += 1
            continue
        snapshot_4d = store.read_snapshot_4d(snap_t)
        rep = run_candidate_interventions(
            snapshot_4d, rule, interior, boundary, env,
            track_id=tr.track_id, track_age=tr.age,
            snapshot_t=snap_t, observer_score=float(obs.combined),
            n_steps=args.n_steps, flip_fraction=args.flip_fraction,
            backend=args.backend, seed=args.seed + len(reports),
        )
        reports.append(rep)
        print(f"  [{len(reports)}/{args.top_k}] (rank {i+1}) "
              f"track {tr.track_id} age={tr.age} snap_t={snap_t} "
              f"interior={int(interior.sum())} obs_score={obs.combined:+.3f}")
    print(f"  found {len(reports)} eligible candidates after walking "
          f"{len(reports) + n_skipped} ranks ({n_skipped} skipped)")
    elapsed = time.time() - t0
    print(f"interventions done in {elapsed:.1f}s")

    agg = aggregate_intervention_summaries(reports)

    # ---------------- write outputs
    _write_intervention_summary_csv(reports, out_dir / "intervention_summary.csv")
    _write_trajectories_json(reports, out_dir / "intervention_trajectories.json")
    write_all_m5_plots(reports, plots_dir, per_candidate_max=min(args.top_k, len(reports)))

    summary = _build_summary_md(reports, agg, args.top_k, out_dir)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    config_dump = {
        "from_run": args.from_run,
        "config": args.config,
        "top_k": args.top_k,
        "n_steps": args.n_steps,
        "flip_fraction": args.flip_fraction,
        "label": args.label,
        "seed": args.seed,
        "backend": args.backend,
        "elapsed_seconds": elapsed,
        "n_reports": len(reports),
    }
    (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    print(f"\nDone. Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
