"""M8B — spatial mechanism disambiguation CLI.

Consumes the CSV produced by `search_large_candidates.py`, filters to
candidates whose morphology supports a clean boundary/interior/
environment decomposition, runs region-aware perturbations, and writes
the M8B output bundle (CSVs + plots + summary.md).
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
from observer_worlds.analysis.m8b_plots import write_all_m8b_plots
from observer_worlds.analysis.m8b_stats import (
    m8b_full_summary, render_m8b_summary_md,
)
from observer_worlds.detection.morphology import (
    MORPHOLOGY_CLASSES, classify_morphology,
)
from observer_worlds.experiments._m8b_spatial import (
    M8BCandidateResult, measure_candidate_m8b,
)
from observer_worlds.experiments.run_m7b_production_holdout import (
    _file_hash, build_frozen_manifest,
)
from observer_worlds.search.rules import FractionalRule

import zarr


def _read_snapshot_from_rundir(rundir: Path, t: int) -> np.ndarray:
    """Read a 4D snapshot from a saved zarr store without going through
    ZarrRunStore (which requires shape/timesteps args we don't have)."""
    zarr_path = rundir / "data" / "states.zarr"
    root = zarr.open(str(zarr_path), mode="r")
    name = f"t{int(t):06d}"
    snapshots = root["snapshots_4d"]
    if name not in snapshots:
        raise KeyError(f"no snapshot at t={t} in {zarr_path}")
    return np.asarray(snapshots[name][:])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M8B spatial mechanism disambiguation."
    )
    p.add_argument("--large-candidates", type=str, required=True,
                   help="Path to large_candidates.csv from "
                        "search_large_candidates.py")
    p.add_argument("--max-candidates", type=int, default=100)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 20, 40, 80])
    p.add_argument("--hce-replicates", type=int, default=3)
    p.add_argument("--region-shell-widths", type=int, nargs="+",
                   default=[1, 2, 3])
    p.add_argument("--backend", choices=["numpy", "numba", "cuda"], default="numpy")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m8b")
    p.add_argument("--quick", action="store_true")
    return p


def _quick(args):
    if args.quick:
        args.max_candidates = min(args.max_candidates, 10)
        args.horizons = [1, 2, 5, 10]
        args.hce_replicates = min(args.hce_replicates, 1)
        args.region_shell_widths = [1, 2]
    return args


def _load_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader: rows.append(r)
    return rows


def _load_rule_seed_index(lc_path: Path) -> dict:
    idx_path = lc_path.parent / "rule_seed_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"rule_seed_index.json not found at {idx_path}"
        )
    return json.loads(idx_path.read_text())


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"; plots_dir.mkdir()
    arrays_dir = out_dir / "arrays"; arrays_dir.mkdir()

    lc_path = Path(args.large_candidates)
    rows = _load_candidates(lc_path)
    rs_index = _load_rule_seed_index(lc_path)

    # Filter to thick candidates first; fall back to including thin if no thick.
    thick_rows = [r for r in rows
                  if r["morphology_class"] in
                  ("thick_candidate", "very_thick_candidate")]
    if len(thick_rows) >= args.max_candidates:
        rows = thick_rows[:args.max_candidates]
        print(f"  using {len(rows)} thick candidates "
              f"(of {len(thick_rows)} total thick)")
    else:
        rows = rows[:args.max_candidates]
        thin_count = sum(1 for r in rows
                        if r["morphology_class"] == "thin_candidate")
        print(f"  using {len(rows)} candidates "
              f"({len(thick_rows)} thick, {thin_count} thin)")

    # Frozen manifest. We don't have rule files directly; hash the input CSV
    # + index instead, plus reuse the M7B manifest helper for the rest.
    class _MArgs: pass
    margs = _MArgs()
    margs.m7_rules = str(lc_path)  # input provenance
    margs.m4c_rules = None; margs.m4a_rules = None
    margs.optimized_2d_rules = None
    margs.n_rules_per_source = 0
    margs.timesteps = 0; margs.grid = [0, 0, 0, 0]
    margs.max_candidates = args.max_candidates
    margs.hce_replicates = args.hce_replicates
    margs.horizons = args.horizons; margs.backend = args.backend
    margs.n_bootstrap = 1000; margs.n_permutations = 1000
    margs.test_seeds = sorted({int(r["seed"]) for r in rows})
    manifest = build_frozen_manifest(margs, out_dir, autodetected_seeds={})
    manifest["large_candidates_csv"] = _file_hash(str(lc_path))
    (out_dir / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    cfg_dump = {
        "max_candidates": args.max_candidates,
        "horizons": args.horizons,
        "hce_replicates": args.hce_replicates,
        "region_shell_widths": args.region_shell_widths,
        "backend": args.backend,
        "n_input_rows": len(rows),
        "test_seeds": margs.test_seeds,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"M8B -> {out_dir}")
    print(f"  rows={len(rows)} horizons={args.horizons} "
          f"replicates={args.hce_replicates} backend={args.backend}")
    print(f"  manifest commit={manifest['git']['commit']} "
          f"dirty={manifest['git']['dirty']}")

    results: list[M8BCandidateResult] = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        elapsed = time.time() - t0
        eta = (elapsed / i) * (len(rows) - i) if i > 0 else 0.0
        print(f"  [{i}/{len(rows)}] {row['rule_source']} {row['rule_id']} "
              f"seed={row['seed']} track={row['candidate_id']} "
              f"area={row['area']} morph={row['morphology_class']} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")
        try:
            mask = np.load(row["mask_npy"]).astype(bool)
            key = f"{row['rule_id']}|{row['seed']}"
            entry = rs_index.get(key)
            if entry is None:
                print(f"    skip: rule_seed entry missing for {key}")
                continue
            rule = FractionalRule.from_dict(entry["rule_dict"])
            bs = rule.to_bsrule()
            rundir = Path(entry["rundir"])
            snap_t = int(row["snapshot_t"])
            snapshot_4d = _read_snapshot_from_rundir(rundir, snap_t)
            feats = candidate_hidden_features(snapshot_4d, mask)
            near_thresh = float(feats.get("near_threshold_fraction", 0.0))
            res = measure_candidate_m8b(
                snapshot_4d=snapshot_4d, candidate_mask_2d=mask,
                rule=bs, rule_id=row["rule_id"],
                rule_source=row["rule_source"], seed=int(row["seed"]),
                candidate_id=int(row["candidate_id"]), snapshot_t=snap_t,
                candidate_area=int(row["area"]),
                candidate_lifetime=int(row["lifetime"]),
                observer_score=None, near_threshold_fraction=near_thresh,
                horizons=args.horizons, n_replicates=args.hce_replicates,
                backend=args.backend,
                rng_seed=int(row["seed"]) * 13 + int(row["candidate_id"]) * 17 + i,
                region_shell_widths=tuple(args.region_shell_widths),
            )
            results.append(res)
        except Exception as e:
            print(f"    error: {e}")

    elapsed = time.time() - t0
    print(f"\nMeasured {len(results)} candidates in {elapsed:.0f}s")

    # CSVs.
    print("writing CSVs...")
    _write_morphology_gates_csv(results, out_dir / "morphology_gates.csv")
    _write_region_response_csv(results, out_dir / "region_response_metrics.csv")
    _write_mechanism_labels_csv(results, out_dir / "mechanism_labels.csv")
    _write_candidate_summary_csv(results, out_dir / "candidate_summary.csv")
    _write_condition_summary_csv(results, out_dir / "condition_summary.csv")

    # Stats.
    print("computing stats...")
    summary = m8b_full_summary(results)
    (out_dir / "stats_summary.json").write_text(
        json.dumps(
            summary, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating)
            else (int(o) if isinstance(o, np.integer)
                  else (o.tolist() if isinstance(o, np.ndarray) else str(o))),
        )
    )

    # Plots.
    print("writing plots...")
    write_all_m8b_plots(results, plots_dir)

    # Summary.
    print("writing summary.md...")
    (out_dir / "summary.md").write_text(render_m8b_summary_md(summary))

    print(f"\nDone. Run dir: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_morphology_gates_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "area", "erosion1_size", "erosion2_size",
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
                m.erosion1_interior_size, m.erosion2_interior_size,
                m.boundary_size, m.environment_size,
                int(m.can_separate_boundary_from_interior),
                int(m.can_classify_environment_coupled),
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
            "candidate_area", "candidate_lifetime",
            "near_threshold_fraction", "interior_per_cell",
            "boundary_per_cell", "environment_per_cell", "far_effect",
            "first_visible_effect_time")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class,
                r.mechanism_label, f"{r.mechanism_confidence:.3f}",
                r.candidate_area, r.candidate_lifetime,
                f"{r.near_threshold_fraction:.4f}",
                f"{r.region_effects['interior'].region_effect_per_cell:.8f}",
                f"{r.region_effects['boundary'].region_effect_per_cell:.8f}",
                f"{r.region_effects['environment'].region_effect_per_cell:.8f}",
                f"{r.far_effect.region_hidden_effect:.6f}",
                r.first_visible_effect_time,
            ])


def _write_candidate_summary_csv(results, path):
    cols = ("rule_id", "rule_source", "seed", "candidate_id",
            "morphology_class", "mechanism_label", "candidate_area",
            "candidate_lifetime", "near_threshold_fraction",
            "HCE_whole", "interior_per_cell", "boundary_per_cell",
            "environment_per_cell", "far_effect",
            "first_visible_effect_time",
            "hidden_to_visible_conversion_time",
            "fraction_hidden_at_end", "fraction_visible_at_end")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([
                r.rule_id, r.rule_source, r.seed, r.candidate_id,
                r.morphology.morphology_class, r.mechanism_label,
                r.candidate_area, r.candidate_lifetime,
                f"{r.near_threshold_fraction:.4f}",
                f"{r.region_effects['whole'].region_hidden_effect:.6f}",
                f"{r.region_effects['interior'].region_effect_per_cell:.8f}",
                f"{r.region_effects['boundary'].region_effect_per_cell:.8f}",
                f"{r.region_effects['environment'].region_effect_per_cell:.8f}",
                f"{r.far_effect.region_hidden_effect:.6f}",
                r.first_visible_effect_time,
                r.hidden_to_visible_conversion_time,
                f"{r.fraction_hidden_at_end:.4f}",
                f"{r.fraction_visible_at_end:.4f}",
            ])


def _write_condition_summary_csv(results, path):
    by_src: dict = {}
    for r in results: by_src.setdefault(r.rule_source, []).append(r)
    cols = ("source", "n_total", "n_thick", "n_thin", "mean_area",
            "mean_lifetime", "mean_HCE_whole", "mean_interior_pc",
            "mean_boundary_pc", "mean_environment_pc", "mean_far",
            "dominant_mechanism_thick")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for src, rs in by_src.items():
            thick = [r for r in rs if r.morphology.morphology_class
                     in ("thick_candidate", "very_thick_candidate")]
            thin = [r for r in rs
                    if r.morphology.morphology_class == "thin_candidate"]
            mech_counts = Counter(r.mechanism_label for r in thick)
            dominant = mech_counts.most_common(1)[0][0] if mech_counts else "n/a"
            w.writerow([
                src, len(rs), len(thick), len(thin),
                f"{np.mean([r.candidate_area for r in rs]):.1f}" if rs else "0",
                f"{np.mean([r.candidate_lifetime for r in rs]):.1f}" if rs else "0",
                f"{np.mean([r.region_effects['whole'].region_hidden_effect for r in rs]):.4f}" if rs else "0",
                f"{np.mean([r.region_effects['interior'].region_effect_per_cell for r in rs]):.6f}" if rs else "0",
                f"{np.mean([r.region_effects['boundary'].region_effect_per_cell for r in rs]):.6f}" if rs else "0",
                f"{np.mean([r.region_effects['environment'].region_effect_per_cell for r in rs]):.6f}" if rs else "0",
                f"{np.mean([r.far_effect.region_hidden_effect for r in rs]):.4f}" if rs else "0",
                dominant,
            ])


if __name__ == "__main__":
    raise SystemExit(main())
