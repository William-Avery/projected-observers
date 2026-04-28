"""M8D — global-chaotic decomposition CLI.

Simulates rule×seed pairs at production grid, runs the full M8D
decomposition (multi-distance probes, background sampling, feature
audit, stabilization variants, 6-subclass relabel) on every observer-
candidate. The decomposition is most informative for candidates that
the M8B/M8C v2 classifier originally labeled `global_chaotic`; non-
global candidates still get distance-effect traces so the comparison
plots (interior_reservoir vs global) work.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis.hidden_features import candidate_hidden_features
from observer_worlds.analysis.m8d_plots import write_all_m8d_plots
from observer_worlds.analysis.m8d_stats import (
    m8d_full_summary, render_m8d_summary_md,
)
from observer_worlds.detection.morphology import classify_morphology
from observer_worlds.experiments._m8d_decomposition import (
    M8DCandidateResult, measure_candidate_m8d,
)
from observer_worlds.experiments._pipeline import (
    detect_and_track, simulate_4d_to_zarr,
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
    p = argparse.ArgumentParser(description="M8D global-chaotic decomposition.")
    p.add_argument("--m7-rules", type=str, required=True)
    p.add_argument("--m4c-rules", type=str, default=None)
    p.add_argument("--m4a-rules", type=str, default=None)
    p.add_argument("--n-rules-per-source", type=int, default=3)
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=list(range(9000, 9040)))
    p.add_argument("--timesteps", type=int, default=600)
    p.add_argument("--grid", type=int, nargs=4, default=[96, 96, 8, 8])
    p.add_argument("--max-candidates-per-run", type=int, default=50)
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--background-n-samples", type=int, default=16)
    p.add_argument("--background-sample-size", type=int, default=8)
    p.add_argument("--stabilization-window-dilation", type=int, default=5)
    p.add_argument("--min-far-distance-floor", type=int, default=24)
    p.add_argument("--min-far-distance-radius-mult", type=float, default=5.0)
    p.add_argument("--backend", choices=["numpy", "numba", "cuda"], default="numpy")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m8d")
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
        args.background_n_samples = 4
        args.background_sample_size = 4
        args.stabilization_window_dilation = 3
        args.min_far_distance_floor = 8
        args.min_far_distance_radius_mult = 3.0
    return args


def _load_with_source(path, n, tag):
    if path is None: return []
    rules = load_top_rules(Path(path), n)
    return [(r, f"{tag}_rank{i:02d}", tag) for i, r in enumerate(rules, 1)]


def _measure_one(*, rule, rule_id, rule_source, seed, grid_shape, timesteps,
                 backend, workdir, max_candidates, horizons, hce_replicates,
                 background_n, background_size, stab_window, min_far_floor,
                 min_far_mult, progress=None):
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
        seed=seed, label=f"m8d_{rule_id}_seed{seed}",
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

    out: list[M8DCandidateResult] = []
    for c in sorted(candidates, key=lambda c: -c.mean_area):
        if len(out) >= max_candidates: break
        if not c.is_candidate: continue
        tr = track_by_id.get(c.track_id)
        if tr is None: continue
        best_idx = None; best_area = 0
        for i, m in enumerate(tr.mask_history):
            a = int(m.sum())
            if a > best_area: best_idx = i; best_area = a
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
            progress(f"    cand {len(out)+1}/{max_candidates} "
                     f"track={tr.track_id} area={area} "
                     f"morph={classify_morphology(mask).morphology_class}")
        try:
            res = measure_candidate_m8d(
                snapshot_4d=snapshot_4d, candidate_mask_2d=mask, rule=bs,
                rule_id=rule_id, rule_source=rule_source, seed=seed,
                candidate_id=tr.track_id, snapshot_t=snap_t,
                candidate_area=area, candidate_lifetime=int(tr.age),
                near_threshold_fraction=near_thresh,
                horizons=horizons, n_replicates=hce_replicates, backend=backend,
                rng_seed=seed * 13 + tr.track_id * 17 + len(out) + 1,
                background_n_samples=background_n,
                background_sample_size=background_size,
                stabilization_window_dilation=stab_window,
                min_far_distance_floor=min_far_floor,
                min_far_distance_radius_mult=min_far_mult,
            )
            out.append(res)
        except Exception as e:
            if progress: progress(f"    error: {e}")
    return out


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
    margs.timesteps = args.timesteps; margs.grid = args.grid
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
    if err: print(f"ABORT: {err}"); return 2

    cfg_dump = {
        "test_seeds": args.test_seeds, "timesteps": args.timesteps,
        "grid": args.grid,
        "max_candidates_per_run": args.max_candidates_per_run,
        "hce_replicates": args.hce_replicates, "horizons": args.horizons,
        "background_n_samples": args.background_n_samples,
        "background_sample_size": args.background_sample_size,
        "stabilization_window_dilation": args.stabilization_window_dilation,
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

    print(f"M8D -> {out_dir}")
    print(f"  rules={len(sources)} test_seeds={len(args.test_seeds)} "
          f"T={args.timesteps} grid={args.grid} backend={args.backend}")
    print(f"  manifest commit={manifest['git']['commit']} "
          f"dirty={manifest['git']['dirty']}")

    all_results: list[M8DCandidateResult] = []
    n_total = len(sources) * len(args.test_seeds); n_done = 0
    t0 = time.time()
    for rule, rule_id, rule_source in sources:
        for seed in args.test_seeds:
            n_done += 1
            elapsed = time.time() - t0
            eta = (elapsed / n_done) * (n_total - n_done) if n_done else 0
            print(f"  [{n_done}/{n_total}] rule={rule_id} src={rule_source} "
                  f"seed={seed} elapsed={elapsed:.0f}s eta={eta:.0f}s")
            try:
                rs = _measure_one(
                    rule=rule, rule_id=rule_id, rule_source=rule_source,
                    seed=seed, grid_shape=tuple(args.grid),
                    timesteps=args.timesteps, backend=args.backend,
                    workdir=workdir, max_candidates=args.max_candidates_per_run,
                    horizons=args.horizons, hce_replicates=args.hce_replicates,
                    background_n=args.background_n_samples,
                    background_size=args.background_sample_size,
                    stab_window=args.stabilization_window_dilation,
                    min_far_floor=args.min_far_distance_floor,
                    min_far_mult=args.min_far_distance_radius_mult,
                    progress=print,
                )
                all_results.extend(rs)
            except Exception as e:
                print(f"    error: {e}")

    elapsed = time.time() - t0
    print(f"\nMeasured {len(all_results)} candidates in {elapsed:.0f}s")

    print("writing CSVs...")
    _write_multi_distance_csv(all_results, out_dir / "multi_distance_effects.csv")
    _write_background_csv(all_results, out_dir / "background_sensitivity.csv")
    _write_global_features_csv(all_results, out_dir / "global_candidate_features.csv")
    _write_stabilization_csv(all_results, out_dir / "stabilization_results.csv")
    _write_relabeled_mechanisms_csv(all_results, out_dir / "relabeled_mechanisms.csv")
    _write_condition_summary_csv(all_results, out_dir / "condition_summary.csv")

    print("computing stats...")
    summary = m8d_full_summary(all_results)
    (out_dir / "stats_summary.json").write_text(
        json.dumps(
            summary, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating)
            else (int(o) if isinstance(o, np.integer)
                  else (o.tolist() if isinstance(o, np.ndarray) else str(o))),
        )
    )
    print("writing plots...")
    write_all_m8d_plots(all_results, plots_dir)
    print("writing summary.md...")
    (out_dir / "summary.md").write_text(render_m8d_summary_md(summary), encoding="utf-8")
    print(f"\nDone. Run dir: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_multi_distance_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "base_mechanism", "probe_name",
            "distance", "distance_over_radius", "n_perturbed_2d",
            "raw_effect", "effect_per_cell")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            for e in r.distance_effects:
                w.writerow([
                    r.rule_id, r.rule_source, r.seed, r.candidate_id,
                    r.morphology.morphology_class, r.base_mechanism_label,
                    e.name, f"{e.distance:.2f}",
                    f"{e.distance_over_radius:.2f}",
                    e.n_perturbed_2d, f"{e.raw_effect:.6f}",
                    f"{e.effect_per_cell:.8f}",
                ])


def _write_background_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "base_mechanism", "background_mean",
            "background_p95", "background_p99",
            "body_over_background", "far_over_background")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class, r.base_mechanism_label,
                f"{r.background_mean:.6f}",
                f"{r.background_p95:.6f}", f"{r.background_p99:.6f}",
                f"{r.body_over_background:.4f}",
                f"{r.far_over_background:.4f}",
            ])


def _write_global_features_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "base_mechanism", "final_mechanism",
            "near_threshold_fraction", "mean_threshold_margin",
            "hidden_temporal_persistence", "hidden_volatility",
            "mean_hidden_entropy", "hidden_spatial_autocorrelation",
            "mean_active_fraction", "hidden_heterogeneity")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            f_ = r.feature_audit
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class, r.base_mechanism_label,
                r.final_mechanism_label,
                f"{f_.get('near_threshold_fraction', 0.0):.4f}",
                f"{f_.get('mean_threshold_margin', 0.0):.4f}",
                f"{f_.get('hidden_temporal_persistence', 0.0):.4f}",
                f"{f_.get('hidden_volatility', 0.0):.4f}",
                f"{f_.get('mean_hidden_entropy', 0.0):.4f}",
                f"{f_.get('hidden_spatial_autocorrelation', 0.0):.4f}",
                f"{f_.get('mean_active_fraction', 0.0):.4f}",
                f"{f_.get('hidden_heterogeneity', 0.0):.4f}",
            ])


def _write_stabilization_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "base_mechanism", "variant", "body", "far",
            "label_would_fire")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            for variant, d in (r.stabilization or {}).items():
                w.writerow([
                    r.rule_id, r.rule_source, r.seed, r.candidate_id,
                    r.base_mechanism_label, variant,
                    f"{d.get('body', 0.0):.6f}" if isinstance(d.get("body"), (int, float)) else "",
                    f"{d.get('far', 0.0):.6f}" if isinstance(d.get("far"), (int, float)) else "",
                    d.get("global_chaotic_label_would_fire", ""),
                ])


def _write_relabeled_mechanisms_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "candidate_area", "candidate_lifetime",
            "near_threshold_fraction", "base_mechanism", "base_confidence",
            "final_mechanism", "final_confidence",
            "body_over_background", "far_over_background",
            "decay_slope", "decay_floor")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            rm = r.relabel_metrics or {}
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class, r.candidate_area,
                r.candidate_lifetime,
                f"{r.near_threshold_fraction:.4f}",
                r.base_mechanism_label, f"{r.base_mechanism_confidence:.3f}",
                r.final_mechanism_label, f"{r.final_mechanism_confidence:.3f}",
                f"{r.body_over_background:.4f}",
                f"{r.far_over_background:.4f}",
                f"{rm.get('decay_slope', 0.0):.6f}",
                f"{rm.get('decay_floor', 0.0):.6f}",
            ])


def _write_condition_summary_csv(results, path):
    by_src: dict = {}
    for r in results: by_src.setdefault(r.rule_source, []).append(r)
    cols = ("source", "n_total", "n_thick", "n_global_base",
            "global_base_thick_frac", "mean_body_over_bg",
            "mean_far_over_bg", "global_subclass_top1")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for src, rs in by_src.items():
            thick = [r for r in rs if r.morphology.morphology_class
                     in ("thick_candidate", "very_thick_candidate")]
            global_base = [r for r in thick
                          if r.base_mechanism_label == "global_chaotic"]
            from collections import Counter
            top = Counter(r.final_mechanism_label
                         for r in global_base).most_common(1)
            w.writerow([
                src, len(rs), len(thick), len(global_base),
                f"{len(global_base)/max(len(thick),1):.2f}",
                f"{np.mean([r.body_over_background for r in thick]):.2f}" if thick else "0",
                f"{np.mean([r.far_over_background for r in thick]):.2f}" if thick else "0",
                top[0][0] if top else "n/a",
            ])


if __name__ == "__main__":
    raise SystemExit(main())
