"""Large-candidate search.

M8 found that most candidates produced by current rules are <10 cells.
That collapses the boundary/interior decomposition. M8B requires
candidates with `area >= 25` (and ideally `area >= 50`) so that
erosion leaves a non-empty interior.

This CLI iterates over rules from M7 / M4C / M4A leaderboards and a
seed sweep, runs each (rule, seed) simulation at production grid size,
and saves *every* observer-candidate that passes the area + lifetime
gate. The downstream M8B CLI reads the resulting CSV and reuses the
saved 4D snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.detection.morphology import classify_morphology
from observer_worlds.experiments._pipeline import (
    detect_and_track, simulate_4d_to_zarr,
)
from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.metrics import score_persistence
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import (
    DetectionConfig, OutputConfig, ProjectionConfig, RunConfig, WorldConfig,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Large-candidate search.")
    p.add_argument("--m7-rules", type=str, default=None)
    p.add_argument("--m4c-rules", type=str, default=None)
    p.add_argument("--m4a-rules", type=str, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(range(7000, 7020)))
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--min-area", type=int, default=25)
    p.add_argument("--min-lifetime", type=int, default=50)
    p.add_argument("--max-candidates-per-run", type=int, default=100)
    p.add_argument("--snapshot-interval", type=int, default=0,
                   help="If 0, snapshots taken every timesteps//6 frames.")
    p.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="large_candidate_search")
    p.add_argument("--quick", action="store_true")
    return p


def _quick(args):
    if args.quick:
        args.n_rules_per_source = min(args.n_rules_per_source, 1)
        args.seeds = args.seeds[:2]
        args.timesteps = min(args.timesteps, 80)
        args.grid = [16, 16, 4, 4]
        args.min_area = 4
        args.min_lifetime = 5
        args.max_candidates_per_run = min(args.max_candidates_per_run, 20)
    return args


def _load_with_source(path, n, tag):
    if path is None: return []
    rules = load_top_rules(Path(path), n)
    return [(r, f"{tag}_rank{i:02d}", tag) for i, r in enumerate(rules, 1)]


def search_one(
    *,
    rule, rule_id, rule_source, seed, grid_shape, timesteps, backend,
    workdir, min_area, min_lifetime, max_candidates_per_run,
    snapshot_interval,
):
    bs = rule.to_bsrule()
    interval = snapshot_interval or max(1, timesteps // 6)
    cfg = RunConfig(
        world=WorldConfig(
            nx=grid_shape[0], ny=grid_shape[1], nz=grid_shape[2], nw=grid_shape[3],
            timesteps=timesteps, initial_density=rule.initial_density,
            rule_birth=tuple(int(x) for x in bs.birth),
            rule_survival=tuple(int(x) for x in bs.survival),
            backend=backend,
        ),
        projection=ProjectionConfig(method="mean_threshold", theta=0.5),
        detection=DetectionConfig(),
        output=OutputConfig(save_4d_snapshots=True, snapshot_interval=interval),
        seed=seed, label=f"lcs_{rule_id}_seed{seed}",
    )
    rundir = workdir / f"{rule_id}_seed{seed}"
    rundir.mkdir(parents=True, exist_ok=True)
    store = ZarrRunStore(
        rundir, timesteps=timesteps,
        shape_2d=(grid_shape[0], grid_shape[1]),
        save_4d_snapshots=True, shape_4d=tuple(grid_shape),
    )
    simulate_4d_to_zarr(cfg, store, seeded_rng(seed))
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)
    candidates = score_persistence(
        tracks, grid_shape=(grid_shape[0], grid_shape[1]),
        config=cfg.detection,
    )
    snap_times = store.list_snapshots()

    out = []
    track_by_id = {t.track_id: t for t in tracks}
    for c in candidates:
        if len(out) >= max_candidates_per_run: break
        if not c.is_candidate: continue
        tr = track_by_id.get(c.track_id)
        if tr is None: continue
        if tr.age < min_lifetime: continue
        # Find the largest mask within the track that passes min_area, near
        # an available snapshot.
        best_idx = None
        best_area = 0
        for i, m in enumerate(tr.mask_history):
            a = int(m.sum())
            if a >= min_area and a > best_area:
                best_idx = i
                best_area = a
        if best_idx is None: continue
        target_frame = tr.frames[best_idx]
        # Pick the snapshot closest to that frame within the track's lifetime.
        snap_t = None
        for st in sorted(snap_times, key=lambda x: abs(x - target_frame)):
            if tr.birth_frame <= st <= tr.last_frame:
                snap_t = st; break
        if snap_t is None: continue
        # Find the mask at snap_t.
        if snap_t in tr.frames:
            i_at = tr.frames.index(snap_t)
        else:
            nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
            i_at = tr.frames.index(nearest)
        mask_at_snap = tr.mask_history[i_at]
        area_at_snap = int(mask_at_snap.sum())
        if area_at_snap < min_area: continue
        morph = classify_morphology(mask_at_snap)
        out.append({
            "rule_id": rule_id, "rule_source": rule_source, "seed": seed,
            "candidate_id": tr.track_id, "snapshot_t": snap_t,
            "best_frame": target_frame, "area": area_at_snap,
            "lifetime": int(tr.age),
            "morphology_class": morph.morphology_class,
            "erosion1_size": morph.erosion1_interior_size,
            "erosion2_size": morph.erosion2_interior_size,
            "boundary_size": morph.boundary_size,
            "environment_size": morph.environment_size,
            "rule_dict": rule.to_dict(),
            "_mask": mask_at_snap.astype(np.uint8),
            "_rundir": str(rundir),
            "_snap_t": snap_t,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir = out_dir / "candidate_snapshots"; snaps_dir.mkdir()
    workdir = out_dir / "_sims"; workdir.mkdir()

    cfg_dump = {
        "n_rules_per_source": args.n_rules_per_source,
        "seeds": args.seeds, "timesteps": args.timesteps,
        "grid": args.grid, "min_area": args.min_area,
        "min_lifetime": args.min_lifetime,
        "max_candidates_per_run": args.max_candidates_per_run,
        "backend": args.backend,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    sources = []
    for path, tag in (
        (args.m7_rules, "M7_HCE_optimized"),
        (args.m4c_rules, "M4C_observer_optimized"),
        (args.m4a_rules, "M4A_viability"),
    ):
        if path is None: continue
        sources.extend(_load_with_source(path, args.n_rules_per_source, tag))

    if not sources:
        print("ABORT: no rules provided")
        return 2

    print(f"Large-candidate search -> {out_dir}")
    print(f"  rules={len(sources)} seeds={len(args.seeds)} "
          f"T={args.timesteps} grid={args.grid} "
          f"min_area={args.min_area} min_life={args.min_lifetime}")

    all_out: list[dict] = []
    rule_seed_index: dict = {}
    n_total = len(sources) * len(args.seeds)
    n_done = 0
    t0 = time.time()
    for rule, rule_id, rule_source in sources:
        for seed in args.seeds:
            n_done += 1
            elapsed = time.time() - t0
            eta = (elapsed / n_done) * (n_total - n_done) if n_done > 0 else 0.0
            print(f"  [{n_done}/{n_total}] rule={rule_id} seed={seed} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")
            try:
                rs = search_one(
                    rule=rule, rule_id=rule_id, rule_source=rule_source,
                    seed=seed, grid_shape=tuple(args.grid),
                    timesteps=args.timesteps, backend=args.backend,
                    workdir=workdir, min_area=args.min_area,
                    min_lifetime=args.min_lifetime,
                    max_candidates_per_run=args.max_candidates_per_run,
                    snapshot_interval=args.snapshot_interval,
                )
                for r in rs:
                    fname = (f"{r['rule_source']}_{r['rule_id']}_"
                             f"seed{r['seed']}_track{r['candidate_id']}_"
                             f"t{r['snapshot_t']}.npy")
                    np.save(snaps_dir / fname, r["_mask"])
                    r["mask_npy"] = str(snaps_dir / fname)
                    rule_seed_index.setdefault(
                        f"{rule_id}|{seed}", {"rule_dict": r["rule_dict"],
                                              "rule_source": rule_source,
                                              "rundir": r["_rundir"],
                                              "candidates": []}
                    )["candidates"].append({
                        "candidate_id": r["candidate_id"],
                        "snapshot_t": r["snapshot_t"],
                        "area": r["area"],
                        "lifetime": r["lifetime"],
                        "morphology_class": r["morphology_class"],
                        "mask_npy": r["mask_npy"],
                    })
                    # Drop _mask from CSV row.
                    r.pop("_mask"); r.pop("_rundir"); r.pop("_snap_t")
                    r.pop("rule_dict")
                all_out.extend(rs)
            except Exception as e:
                print(f"    error: {e}")

    cols = ("rule_id", "rule_source", "seed", "candidate_id", "snapshot_t",
            "best_frame", "area", "lifetime", "morphology_class",
            "erosion1_size", "erosion2_size", "boundary_size",
            "environment_size", "mask_npy")
    with (out_dir / "large_candidates.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in all_out:
            w.writerow([r.get(c, "") for c in cols])
    (out_dir / "rule_seed_index.json").write_text(
        json.dumps(rule_seed_index, indent=2)
    )

    # Per-morphology summary.
    from collections import Counter
    morph_counts = Counter(r["morphology_class"] for r in all_out)
    src_counts = Counter(r["rule_source"] for r in all_out)
    md = ["# Large-candidate search", ""]
    md.append(f"- N candidates: {len(all_out)}")
    md.append(f"- N rule×seed combos run: {n_total}")
    md.append("")
    md.append("## Morphology breakdown")
    md.append("")
    md.append("| morphology | count |")
    md.append("|---|---|")
    for k in ("very_thick_candidate", "thick_candidate",
              "thin_candidate", "degenerate"):
        md.append(f"| {k} | {morph_counts.get(k, 0)} |")
    md.append("")
    md.append("## Per-source breakdown")
    md.append("")
    md.append("| source | count |")
    md.append("|---|---|")
    for src, n in src_counts.items():
        md.append(f"| {src} | {n} |")
    (out_dir / "summary.md").write_text("\n".join(md))

    print(f"\nFound {len(all_out)} candidates.")
    print(f"  morphology: {dict(morph_counts)}")
    print(f"  by source : {dict(src_counts)}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
