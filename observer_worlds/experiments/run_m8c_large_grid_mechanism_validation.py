"""M8C — large-grid mechanism validation CLI.

Simulates rules×seeds at large grid (default 96×96×8×8) so the far-mask
antipode is actually distant relative to candidate radius. Reuses M8B's
region-aware response measurement and morphology gates; adds adaptive
far-control selection with explicit distance + activity matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis.hidden_features import candidate_hidden_features
from observer_worlds.analysis.m8c_plots import write_all_m8c_plots
from observer_worlds.analysis.m8c_stats import (
    m8c_full_summary, render_m8c_summary_md,
)
from observer_worlds.detection.morphology import classify_morphology
from observer_worlds.experiments._m8c_validation import (
    M8CCandidateResult, measure_candidate_m8c,
)
from observer_worlds.experiments._pipeline import (
    compute_full_metrics, detect_and_track, simulate_4d_to_zarr,
)
from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.experiments.run_m7b_production_holdout import (
    _autodetect_m7_seed_splits, build_frozen_manifest, check_seed_disjointness,
)
from observer_worlds.metrics import score_persistence
from observer_worlds.storage import ZarrRunStore
from observer_worlds.utils import seeded_rng
from observer_worlds.utils.config import (
    DetectionConfig, OutputConfig, ProjectionConfig, RunConfig, WorldConfig,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M8C large-grid mechanism validation."
    )
    p.add_argument("--m7-rules", type=str, required=True)
    p.add_argument("--m4c-rules", type=str, default=None)
    p.add_argument("--m4a-rules", type=str, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=3)
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=list(range(8000, 8040)))
    p.add_argument("--timesteps", type=int, default=600)
    p.add_argument("--grid", type=int, nargs=4, default=[96, 96, 8, 8])
    p.add_argument("--max-candidates-per-run", type=int, default=50)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--region-shell-widths", type=int, nargs="+",
                   default=[1, 2, 3])
    p.add_argument("--min-far-distance-floor", type=int, default=32)
    p.add_argument("--min-far-distance-radius-mult", type=float, default=5.0)
    p.add_argument("--backend", choices=["numpy", "numba", "cuda"], default="numpy")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m8c")
    p.add_argument("--quick", action="store_true")
    return p


def _quick(args):
    if args.quick:
        args.n_rules_per_source = min(args.n_rules_per_source, 1)
        args.test_seeds = args.test_seeds[:2]
        args.timesteps = min(args.timesteps, 100)
        args.grid = [32, 32, 4, 4]
        args.max_candidates_per_run = min(args.max_candidates_per_run, 5)
        args.hce_replicates = 1
        args.horizons = [1, 2, 5, 10]
        args.region_shell_widths = [1, 2]
        args.min_far_distance_floor = 8
        args.min_far_distance_radius_mult = 3.0
    return args


def _load_with_source(path, n, tag):
    if path is None: return []
    rules = load_top_rules(Path(path), n)
    return [(r, f"{tag}_rank{i:02d}", tag) for i, r in enumerate(rules, 1)]


def _measure_one_rule_seed(
    *, rule, rule_id, rule_source, seed, grid_shape, timesteps,
    backend, workdir, max_candidates, horizons, hce_replicates,
    region_shell_widths, min_far_distance_floor, min_far_distance_radius_mult,
    progress=None,
) -> list[M8CCandidateResult]:
    bs = rule.to_bsrule()
    cfg = RunConfig(
        world=WorldConfig(
            nx=grid_shape[0], ny=grid_shape[1],
            nz=grid_shape[2], nw=grid_shape[3],
            timesteps=timesteps, initial_density=rule.initial_density,
            rule_birth=tuple(int(x) for x in bs.birth),
            rule_survival=tuple(int(x) for x in bs.survival),
            backend=backend,
        ),
        projection=ProjectionConfig(method="mean_threshold", theta=0.5),
        detection=DetectionConfig(),
        output=OutputConfig(save_4d_snapshots=True,
                            snapshot_interval=max(1, timesteps // 6)),
        seed=seed, label=f"m8c_{rule_id}_seed{seed}",
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

    track_by_id = {t.track_id: t for t in tracks}
    results: list[M8CCandidateResult] = []
    # Prefer large-area candidates first (we explicitly want thick).
    for c in sorted(candidates, key=lambda c: -c.mean_area):
        if len(results) >= max_candidates: break
        if not c.is_candidate: continue
        tr = track_by_id.get(c.track_id)
        if tr is None: continue
        # Pick the largest-area frame within the track that is near a snapshot.
        best_idx = None; best_area = 0
        for i, m in enumerate(tr.mask_history):
            a = int(m.sum())
            if a > best_area:
                best_idx = i; best_area = a
        if best_idx is None or best_area < 4: continue
        target_frame = tr.frames[best_idx]
        snap_t = None
        for st in sorted(snap_times, key=lambda x: abs(x - target_frame)):
            if tr.birth_frame <= st <= tr.last_frame:
                snap_t = st; break
        if snap_t is None: continue
        if snap_t in tr.frames: i_at = tr.frames.index(snap_t)
        else:
            nearest = min(tr.frames, key=lambda f: abs(f - snap_t))
            i_at = tr.frames.index(nearest)
        mask = tr.mask_history[i_at]
        area = int(mask.sum())
        if area < 4: continue
        try:
            snapshot_4d = store.read_snapshot_4d(snap_t)
        except Exception:
            continue
        feats = candidate_hidden_features(snapshot_4d, mask)
        near_thresh = float(feats.get("near_threshold_fraction", 0.0))
        if progress:
            progress(f"    cand {len(results)+1}/{max_candidates} "
                     f"track={tr.track_id} area={area} "
                     f"morph={classify_morphology(mask).morphology_class}")
        try:
            res = measure_candidate_m8c(
                snapshot_4d=snapshot_4d, candidate_mask_2d=mask,
                rule=bs, rule_id=rule_id, rule_source=rule_source, seed=seed,
                candidate_id=tr.track_id, snapshot_t=snap_t,
                candidate_area=area, candidate_lifetime=int(tr.age),
                observer_score=None,
                near_threshold_fraction=near_thresh,
                horizons=horizons, n_replicates=hce_replicates,
                backend=backend,
                rng_seed=seed * 13 + tr.track_id * 17 + len(results) + 1,
                region_shell_widths=tuple(region_shell_widths),
                min_far_distance_floor=min_far_distance_floor,
                min_far_distance_radius_mult=min_far_distance_radius_mult,
            )
            results.append(res)
        except Exception as e:
            if progress: progress(f"    error: {e}")
    return results


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"; plots_dir.mkdir()
    arrays_dir = out_dir / "arrays"; arrays_dir.mkdir()
    workdir = out_dir / "_sims"; workdir.mkdir()

    autodetected = _autodetect_m7_seed_splits(args.m7_rules)

    class _MArgs: pass
    margs = _MArgs()
    margs.m7_rules = args.m7_rules
    margs.m4c_rules = args.m4c_rules
    margs.m4a_rules = args.m4a_rules
    margs.optimized_2d_rules = None
    margs.n_rules_per_source = args.n_rules_per_source
    margs.timesteps = args.timesteps
    margs.grid = args.grid
    margs.max_candidates = args.max_candidates_per_run
    margs.hce_replicates = args.hce_replicates
    margs.horizons = args.horizons; margs.backend = args.backend
    margs.n_bootstrap = 1000; margs.n_permutations = 1000
    margs.test_seeds = args.test_seeds
    manifest = build_frozen_manifest(margs, out_dir, autodetected)
    (out_dir / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    err = check_seed_disjointness(args.test_seeds, autodetected)
    if err:
        print(f"ABORT: {err}"); return 2

    cfg_dump = {
        "test_seeds": args.test_seeds, "timesteps": args.timesteps,
        "grid": args.grid, "max_candidates_per_run": args.max_candidates_per_run,
        "hce_replicates": args.hce_replicates, "horizons": args.horizons,
        "region_shell_widths": args.region_shell_widths,
        "min_far_distance_floor": args.min_far_distance_floor,
        "min_far_distance_radius_mult": args.min_far_distance_radius_mult,
        "backend": args.backend,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    sources = []
    sources.extend(_load_with_source(args.m7_rules, args.n_rules_per_source,
                                     "M7_HCE_optimized"))
    if args.m4c_rules:
        sources.extend(_load_with_source(args.m4c_rules, args.n_rules_per_source,
                                         "M4C_observer_optimized"))
    if args.m4a_rules:
        sources.extend(_load_with_source(args.m4a_rules, args.n_rules_per_source,
                                         "M4A_viability"))

    print(f"M8C -> {out_dir}")
    print(f"  rules={len(sources)} test_seeds={len(args.test_seeds)} "
          f"T={args.timesteps} grid={args.grid} backend={args.backend}")
    print(f"  manifest commit={manifest['git']['commit']} "
          f"dirty={manifest['git']['dirty']}")

    all_results: list[M8CCandidateResult] = []
    n_total = len(sources) * len(args.test_seeds)
    n_done = 0
    t0 = time.time()
    for rule, rule_id, rule_source in sources:
        for seed in args.test_seeds:
            n_done += 1
            elapsed = time.time() - t0
            eta = (elapsed / n_done) * (n_total - n_done) if n_done else 0.0
            print(f"  [{n_done}/{n_total}] rule={rule_id} src={rule_source} "
                  f"seed={seed} elapsed={elapsed:.0f}s eta={eta:.0f}s")
            try:
                rs = _measure_one_rule_seed(
                    rule=rule, rule_id=rule_id, rule_source=rule_source,
                    seed=seed, grid_shape=tuple(args.grid),
                    timesteps=args.timesteps, backend=args.backend,
                    workdir=workdir, max_candidates=args.max_candidates_per_run,
                    horizons=args.horizons,
                    hce_replicates=args.hce_replicates,
                    region_shell_widths=args.region_shell_widths,
                    min_far_distance_floor=args.min_far_distance_floor,
                    min_far_distance_radius_mult=args.min_far_distance_radius_mult,
                    progress=print,
                )
                all_results.extend(rs)
            except Exception as e:
                print(f"    error: {e}")

    elapsed = time.time() - t0
    print(f"\nMeasured {len(all_results)} candidates in {elapsed:.0f}s")

    # CSVs.
    print("writing CSVs...")
    _write_morphology_gates_csv(all_results, out_dir / "morphology_gates.csv")
    _write_far_control_csv(all_results, out_dir / "far_control_quality.csv")
    _write_region_response_csv(all_results, out_dir / "region_response_metrics.csv")
    _write_mechanism_labels_csv(all_results, out_dir / "mechanism_labels.csv")
    _write_candidate_summary_csv(all_results, out_dir / "candidate_summary.csv")
    _write_condition_summary_csv(all_results, out_dir / "condition_summary.csv")

    # Stats + summary.
    print("computing stats...")
    summary = m8c_full_summary(all_results)
    (out_dir / "stats_summary.json").write_text(
        json.dumps(
            summary, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating)
            else (int(o) if isinstance(o, np.integer)
                  else (o.tolist() if isinstance(o, np.ndarray) else str(o))),
        )
    )
    print("writing plots...")
    write_all_m8c_plots(all_results, plots_dir)
    print("writing summary.md...")
    (out_dir / "summary.md").write_text(render_m8c_summary_md(summary), encoding="utf-8")

    print(f"\nDone. Run dir: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CSV writers (mirror M8B's pattern with M8C-specific extra columns)
# ---------------------------------------------------------------------------


def _write_morphology_gates_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "area", "candidate_radius",
            "candidate_diameter", "erosion1_size", "erosion2_size",
            "boundary_size", "environment_size",
            "can_separate_boundary_from_interior",
            "can_classify_environment_coupled")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            m = r.morphology
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                m.morphology_class, m.area,
                f"{r.far_control.candidate_radius:.2f}",
                f"{r.far_control.candidate_diameter:.2f}",
                m.erosion1_interior_size, m.erosion2_interior_size,
                m.boundary_size, m.environment_size,
                int(m.can_separate_boundary_from_interior),
                int(m.can_classify_environment_coupled),
            ])


def _write_far_control_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "candidate_radius", "candidate_diameter",
            "far_control_valid", "far_control_distance",
            "far_control_distance_over_radius",
            "far_control_min_distance_required",
            "far_control_translation_dy", "far_control_translation_dx",
            "far_control_projected_activity_diff",
            "far_control_hidden_activity_diff",
            "rejection_reason")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            fc = r.far_control
            tr = fc.far_control_translation or ("", "")
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                f"{fc.candidate_radius:.2f}",
                f"{fc.candidate_diameter:.2f}",
                int(fc.far_control_valid),
                f"{fc.far_control_distance:.2f}",
                f"{fc.far_control_distance_over_radius:.2f}",
                f"{fc.far_control_min_distance_required:.2f}",
                tr[0], tr[1],
                f"{fc.far_control_projected_activity_diff:.4f}",
                f"{fc.far_control_hidden_activity_diff:.4f}",
                fc.rejection_reason,
            ])


def _write_region_response_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id", "region",
            "n_perturbed_cells_2d", "n_flipped_cells_4d",
            "region_hidden_effect", "region_local_divergence",
            "region_response_fraction", "region_effect_per_cell",
            "region_effect_per_flipped_cell")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            for region_name, eff in {**r.region_effects,
                                     "far": r.far_effect}.items():
                w.writerow([
                    r.rule_id, r.rule_source, r.seed, r.candidate_id,
                    region_name, eff.n_perturbed_cells_2d,
                    eff.n_flipped_cells_4d,
                    f"{eff.region_hidden_effect:.6f}",
                    f"{eff.region_local_divergence:.6f}",
                    f"{eff.region_response_fraction:.4f}",
                    f"{eff.region_effect_per_cell:.8f}",
                    f"{eff.region_effect_per_flipped_cell:.6f}",
                ])


def _write_mechanism_labels_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "mechanism_label", "confidence",
            "candidate_area", "candidate_radius", "candidate_lifetime",
            "near_threshold_fraction", "interior_per_cell",
            "boundary_per_cell", "environment_per_cell", "far_effect",
            "far_control_valid", "far_distance_over_radius",
            "first_visible_effect_time")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class, r.mechanism_label,
                f"{r.mechanism_confidence:.3f}",
                r.candidate_area,
                f"{r.far_control.candidate_radius:.2f}",
                r.candidate_lifetime,
                f"{r.near_threshold_fraction:.4f}",
                f"{r.region_effects['interior'].region_effect_per_cell:.8f}",
                f"{r.region_effects['boundary'].region_effect_per_cell:.8f}",
                f"{r.region_effects['environment'].region_effect_per_cell:.8f}",
                f"{r.far_effect.region_hidden_effect:.6f}",
                int(r.far_control.far_control_valid),
                f"{r.far_control.far_control_distance_over_radius:.2f}",
                r.first_visible_effect_time,
            ])


def _write_candidate_summary_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "mechanism_label", "candidate_area",
            "candidate_radius", "candidate_lifetime",
            "near_threshold_fraction",
            "HCE_whole", "interior_per_cell", "boundary_per_cell",
            "environment_per_cell", "far_effect", "far_control_valid",
            "far_distance_over_radius",
            "first_visible_effect_time",
            "hidden_to_visible_conversion_time",
            "fraction_hidden_at_end", "fraction_visible_at_end")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class, r.mechanism_label,
                r.candidate_area, f"{r.far_control.candidate_radius:.2f}",
                r.candidate_lifetime,
                f"{r.near_threshold_fraction:.4f}",
                f"{r.region_effects['whole'].region_hidden_effect:.6f}",
                f"{r.region_effects['interior'].region_effect_per_cell:.8f}",
                f"{r.region_effects['boundary'].region_effect_per_cell:.8f}",
                f"{r.region_effects['environment'].region_effect_per_cell:.8f}",
                f"{r.far_effect.region_hidden_effect:.6f}",
                int(r.far_control.far_control_valid),
                f"{r.far_control.far_control_distance_over_radius:.2f}",
                r.first_visible_effect_time,
                r.hidden_to_visible_conversion_time,
                f"{r.fraction_hidden_at_end:.4f}",
                f"{r.fraction_visible_at_end:.4f}",
            ])


def _write_condition_summary_csv(results, path):
    by_src: dict = {}
    for r in results: by_src.setdefault(r.rule_source, []).append(r)
    cols = ("source", "n_total", "n_thick", "thick_fraction",
            "mean_HCE", "mean_locality", "global_chaotic_thick_frac",
            "interior_thick_frac", "whole_body_thick_frac",
            "boundary_thick_frac", "environment_thick_frac",
            "far_valid_frac", "mean_far_dist_over_radius")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for src, rs in by_src.items():
            thick = [r for r in rs
                     if r.morphology.morphology_class
                     in ("thick_candidate", "very_thick_candidate")]
            valid_far = [r for r in rs if r.far_control.far_control_valid]
            n = max(len(rs), 1); nt = max(len(thick), 1)
            w.writerow([
                src, len(rs), len(thick), f"{len(thick)/n:.2f}",
                f"{np.mean([r.region_effects['whole'].region_hidden_effect for r in rs]):.4f}" if rs else "0",
                f"{np.mean([r.region_effects['whole'].region_hidden_effect - r.far_effect.region_hidden_effect for r in valid_far]):.4f}" if valid_far else "0",
                f"{sum(1 for r in thick if r.mechanism_label == 'global_chaotic') / nt:.2f}",
                f"{sum(1 for r in thick if r.mechanism_label == 'interior_reservoir') / nt:.2f}",
                f"{sum(1 for r in thick if r.mechanism_label == 'whole_body_hidden_support') / nt:.2f}",
                f"{sum(1 for r in thick if r.mechanism_label == 'boundary_mediated') / nt:.2f}",
                f"{sum(1 for r in thick if r.mechanism_label == 'environment_coupled') / nt:.2f}",
                f"{len(valid_far) / n:.2f}",
                f"{np.mean([r.far_control.far_control_distance_over_radius for r in valid_far]):.1f}" if valid_far else "0",
            ])


if __name__ == "__main__":
    raise SystemExit(main())
