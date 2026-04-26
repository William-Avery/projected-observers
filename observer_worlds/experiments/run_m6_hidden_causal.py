"""M6 — Hidden Causal Dependence experiment driver.

Two operating modes:

  --from-run PATH                 — open an existing 4D run dir (with
                                    snapshots) and run M6 on its top-K
                                    candidates.
  --config PATH                   — start fresh from a RunConfig JSON;
                                    runs the 4D experiment with snapshots
                                    forced on, then proceeds.

Optional second arm (the headline comparison):

  --shuffled-config PATH          — run a parallel hidden-causal
                                    experiment using a hidden-shuffled-4D
                                    baseline (same rule, but the simulation
                                    has its z,w fibers permuted into
                                    ca.state every step). The combined
                                    summary then reports
                                    HCE(coherent) - HCE(shuffled).

The headline question: do coherent 4D candidates have stronger hidden
causal effect than candidates produced by hidden-shuffled-4D dynamics?
A statistically significant positive paired difference would be the
first signature unique to *coherent* 4D dynamics (not just to "having
4D state at all").
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer_worlds.analysis import write_all_m4b_plots  # noqa: F401  (kept for symmetry)
from observer_worlds.experiments._m4b_sweep import hidden_shuffle_mutator
from observer_worlds.experiments._m6_hidden_causal import (
    HiddenCausalReport,
    aggregate_hce_stats,
    compare_hce_paired,
    run_hidden_causal_experiment,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M6 hidden-causal-dependence experiment.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run", type=str,
                     help="Existing 4D run dir with snapshots (data/states.zarr).")
    src.add_argument("--config", type=str,
                     help="RunConfig JSON to run a fresh 4D experiment with snapshots.")
    p.add_argument("--shuffled-config", type=str, default=None,
                   help="Optional: a RunConfig JSON to run a hidden-shuffled-4D "
                        "control. M6 will be evaluated on candidates from BOTH the "
                        "coherent run and this shuffled run, and the paired "
                        "HCE difference reported.")
    p.add_argument("--top-k", type=int, default=8,
                   help="Top-K observer candidates per run to analyze.")
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-replicates", type=int, default=5)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--label", type=str, default="m6")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["numba", "numpy", "cuda"], default="numpy")
    return p


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------


def _open_or_create_run(
    *, run_dir_arg: str | None, config_arg: str | None,
    out_dir: Path, suffix: str, shuffled: bool = False,
):
    """Returns (cfg, store, frames, tracks, candidates, observer_scores).

    If run_dir_arg is provided, opens that store read-only.
    If config_arg is provided, runs a fresh 4D experiment under
    ``out_dir / 'run_{suffix}'`` with snapshots forced on. If shuffled=True,
    threads the hidden-shuffle mutator into the simulator.
    """
    import zarr

    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        cfg = RunConfig.load(run_dir / "config.json")
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
                store._shape_4d = store._snapshots_group[name].shape
                break
    else:
        cfg = RunConfig.load(config_arg)
        cfg.output.save_4d_snapshots = True
        if cfg.output.snapshot_interval <= 0:
            cfg.output.snapshot_interval = max(1, cfg.world.timesteps // 8)
        run_root = out_dir / f"run_{suffix}"
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
        if shuffled:
            mutator_rng = np.random.default_rng(cfg.seed * 7919 + 1)

            def _mutator(state, t, _rng):
                return hidden_shuffle_mutator(state, t, mutator_rng)

            print(f"[fresh shuffled-4D] simulating with hidden-shuffle mutator...")
            simulate_4d_to_zarr(cfg, store, rng, state_mutator=_mutator)
        else:
            print(f"[fresh coherent-4D] simulating...")
            simulate_4d_to_zarr(cfg, store, rng)

    print("  detecting + tracking...")
    frames = store.read_frames_2d()
    tracks = detect_and_track(cfg, frames)
    candidates = score_persistence(
        tracks, grid_shape=(cfg.world.nx, cfg.world.ny), config=cfg.detection
    )
    print("  scoring observer metrics for candidate ranking...")
    observer_scores, _ = compute_full_metrics(
        cfg, tracks, candidates, store, rollout_steps=6, world_kind="4d",
    )
    return cfg, store, frames, tracks, candidates, observer_scores


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
    if kind == "interior":
        m = track.interior_history[i]
        return m if m.any() else track.mask_history[i]
    if kind == "mask":
        return track.mask_history[i]
    raise ValueError(kind)


def _run_m6_on_run(
    *, label: str, cfg, store, tracks, observer_scores, args,
) -> list[HiddenCausalReport]:
    """Walk down candidates by observer_score; for each that has a snapshot
    in its lifetime AND non-empty interior, run the M6 experiment."""
    snap_times = store.list_snapshots()
    if not snap_times:
        print(f"  [{label}] no 4D snapshots; cannot run M6")
        return []
    rule = BSRule(birth=cfg.world.rule_birth, survival=cfg.world.rule_survival)
    sorted_obs = sorted(observer_scores, key=lambda o: -o.combined)
    track_by_id = {t.track_id: t for t in tracks}

    reports: list[HiddenCausalReport] = []
    skipped = 0
    for obs in sorted_obs:
        if len(reports) >= args.top_k:
            break
        tr = track_by_id.get(obs.track_id)
        if tr is None:
            skipped += 1
            continue
        snap_t = _pick_snapshot(tr, snap_times)
        if snap_t is None:
            skipped += 1
            continue
        interior = _mask_for_track(tr, snap_t, "interior")
        if not interior.any():
            skipped += 1
            continue
        snapshot_4d = store.read_snapshot_4d(snap_t)
        rep = run_hidden_causal_experiment(
            snapshot_4d, rule, interior,
            track_id=tr.track_id, track_age=tr.age, snapshot_t=snap_t,
            observer_score=float(obs.combined),
            n_steps=args.n_steps, n_replicates=args.n_replicates,
            backend=args.backend, seed=args.seed + len(reports),
        )
        reports.append(rep)
        print(
            f"  [{label}] track {tr.track_id:>4} snap_t={snap_t:>3} "
            f"interior={int(interior.sum()):>3} HCE={rep.HCE:.4f} "
            f"vis={rep.visible_final_l1:.4f} ratio={rep.hce_to_visible_ratio:.2f}"
        )
    if skipped:
        print(f"  [{label}] skipped {skipped} candidates (no snapshot or empty mask)")
    return reports


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_summary_csv(reports: dict[str, list[HiddenCausalReport]], out_path: Path) -> None:
    cols = [
        "condition", "track_id", "snapshot_t", "track_age",
        "observer_score", "interior_size",
        "n_steps", "n_replicates", "n_flips_mean",
        "HCE", "visible_final_l1", "hce_to_visible_ratio",
        "hce_immediate_check",
        "hce_full_grid_l1_mean_per_step", "vis_full_grid_l1_mean_per_step",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cond, recs in reports.items():
            for r in recs:
                w.writerow([
                    cond, r.track_id, r.snapshot_t, r.track_age,
                    "" if r.observer_score is None else f"{r.observer_score:.4f}",
                    r.interior_size, r.n_steps, r.n_replicates,
                    f"{r.hidden_invisible.mean_n_flips:.2f}",
                    f"{r.HCE:.6f}", f"{r.visible_final_l1:.6f}",
                    f"{r.hce_to_visible_ratio:.4f}",
                    f"{r.hce_immediate_check:.6f}",
                    ";".join(f"{x:.5f}" for x in r.hidden_invisible.full_grid_l1_mean),
                    ";".join(f"{x:.5f}" for x in r.visible_match_count.full_grid_l1_mean),
                ])


def _write_trajectories_json(reports: dict[str, list[HiddenCausalReport]], path: Path) -> None:
    out = {}
    for cond, recs in reports.items():
        out[cond] = []
        for r in recs:
            out[cond].append({
                "track_id": r.track_id,
                "snapshot_t": r.snapshot_t,
                "track_age": r.track_age,
                "observer_score": r.observer_score,
                "interior_size": r.interior_size,
                "n_steps": r.n_steps,
                "n_replicates": r.n_replicates,
                "HCE": r.HCE,
                "visible_final_l1": r.visible_final_l1,
                "hce_to_visible_ratio": r.hce_to_visible_ratio,
                "hce_immediate_check": r.hce_immediate_check,
                "hidden_invisible": {
                    "full_grid_l1_mean": r.hidden_invisible.full_grid_l1_mean,
                    "full_grid_l1_std": r.hidden_invisible.full_grid_l1_std,
                    "candidate_footprint_l1_mean": r.hidden_invisible.candidate_footprint_l1_mean,
                    "candidate_footprint_l1_std": r.hidden_invisible.candidate_footprint_l1_std,
                    "mean_n_flips": r.hidden_invisible.mean_n_flips,
                    "per_replicate_final": r.hidden_invisible.per_replicate_final,
                },
                "visible_match_count": {
                    "full_grid_l1_mean": r.visible_match_count.full_grid_l1_mean,
                    "full_grid_l1_std": r.visible_match_count.full_grid_l1_std,
                    "candidate_footprint_l1_mean": r.visible_match_count.candidate_footprint_l1_mean,
                    "candidate_footprint_l1_std": r.visible_match_count.candidate_footprint_l1_std,
                    "mean_n_flips": r.visible_match_count.mean_n_flips,
                    "per_replicate_final": r.visible_match_count.per_replicate_final,
                },
            })
    path.write_text(json.dumps(out, indent=2,
                               default=lambda o: float(o) if isinstance(o, np.floating)
                               else (int(o) if isinstance(o, np.integer)
                                     else (o.tolist() if isinstance(o, np.ndarray) else str(o)))))


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------


def _interpret_solo(stats: dict) -> str:
    if stats["n_candidates"] == 0:
        return "No eligible candidates found."
    n = stats["n_candidates"]
    mean_hce = stats["mean_HCE"]
    p_pos = stats["one_sample_p_hce_gt_zero"]
    if mean_hce > 0 and p_pos < 0.05:
        return (f"Mean HCE = {mean_hce:.4f} is significantly positive "
                f"across {n} candidates (sign-test p={p_pos:.4f}). "
                f"Hidden-dimensional structure has measurable causal weight "
                f"on the projected future. This is an effect that 2D "
                f"systems cannot exhibit by construction.")
    if mean_hce > 0:
        return (f"Mean HCE = {mean_hce:.4f} is directionally positive but "
                f"not statistically established at sign-test p={p_pos:.4f} "
                f"(N={n}). Hidden state plausibly matters, but more "
                f"candidates needed for confidence.")
    return (f"Mean HCE = {mean_hce:.4f}; no evidence that hidden "
            f"structure has causal weight beyond noise.")


def _interpret_paired(paired_stats: dict) -> str:
    if paired_stats["n_paired"] == 0:
        return ("No comparable candidates between coherent and shuffled "
                "runs.")
    strat = paired_stats.get("comparison_strategy", "unknown")
    strat_note = ""
    if strat == "rank_pairing":
        strat_note = (
            " *(Comparison uses rank-pairing — coherent's k-th-best HCE "
            "vs shuffled's k-th-best — because track IDs are not shared "
            "across the two simulations. Treat as pseudo-paired.)*"
        )
    elif strat == "id_pairing":
        strat_note = " *(Track IDs happen to overlap; true paired comparison.)*"
    diff = paired_stats["mean_diff_coh_minus_shuf"]
    p = paired_stats["sign_test_p"]
    n = paired_stats["n_paired"]
    ci = (paired_stats["bootstrap_ci_low"], paired_stats["bootstrap_ci_high"])
    if diff > 0 and p < 0.05 and ci[0] > 0:
        msg = (f"**Coherent-4D candidates have significantly stronger "
                f"hidden causal effect than shuffled-4D candidates** "
                f"(mean diff = {diff:+.4f}, sign-test p={p:.4f}, "
                f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], N={n}). "
                f"This is the first signature unique to *coherent* hidden "
                f"dynamics, not just to having 4D state.")
    elif diff > 0:
        msg = (f"Coherent-4D HCE is directionally larger than "
                f"shuffled-4D (diff = {diff:+.4f}, p={p:.4f}, N={n}) "
                f"but not statistically established at this N.")
    elif diff < 0 and p < 0.05:
        msg = (f"**Shuffled-4D candidates have stronger HCE than coherent** "
                f"(diff = {diff:+.4f}, p={p:.4f}). Surprising — random "
                f"hidden state matters more than coherent hidden state for "
                f"the projected future of these candidates.")
    else:
        msg = (f"No significant coherent-vs-shuffled HCE difference "
                f"(diff = {diff:+.4f}, p={p:.4f}, N={n}).")
    return msg + strat_note


def _build_combined_summary(
    coherent_reports: list[HiddenCausalReport],
    shuffled_reports: list[HiddenCausalReport] | None,
    args, out_dir: Path,
) -> str:
    lines = [f"# M6 — Hidden Causal Dependence — {args.label}", ""]
    lines.append(f"- Run dir: `{out_dir}`")
    lines.append(f"- Top-K candidates per run: {args.top_k}")
    lines.append(f"- Rollout steps: {args.n_steps}")
    lines.append(f"- Replicates per candidate: {args.n_replicates}")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("**HCE** = mean projected divergence at the *final* rollout step "
                 "under a `hidden_invisible` perturbation (z,w permuted inside the "
                 "candidate's interior, projection at t=0 byte-identical). "
                 "In a pure 2D system this is identically zero. In 4D it can be "
                 "positive iff hidden state has causal weight on the projected future.")
    lines.append("")
    lines.append("**visible_match_count** = control: same number of bit-flips "
                 "applied uniformly to interior 4D fibers (does change projection at t=0).")
    lines.append("")

    coh_stats = aggregate_hce_stats(coherent_reports)
    lines.append("## Coherent 4D")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| n_candidates | {coh_stats['n_candidates']} |")
    lines.append(f"| mean HCE | {coh_stats['mean_HCE']:.5f} |")
    lines.append(f"| median HCE | {coh_stats['median_HCE']:.5f} |")
    lines.append(f"| std HCE | {coh_stats['std_HCE']:.5f} |")
    lines.append(f"| min / max HCE | {coh_stats.get('min_HCE', 0):.4f} / "
                 f"{coh_stats.get('max_HCE', 0):.4f} |")
    lines.append(f"| mean visible_final_l1 | {coh_stats['mean_visible_final_l1']:.5f} |")
    lines.append(f"| mean HCE / visible ratio | {coh_stats['mean_ratio']:.3f} |")
    lines.append(f"| fraction HCE > 0 | {coh_stats['fraction_hce_positive']:.2f} |")
    lines.append(f"| sign-test p (HCE > 0) | {coh_stats['one_sample_p_hce_gt_zero']:.4f} |")
    lines.append(f"| mean immediate (sanity, ≈0) | {coh_stats['mean_immediate_check']:.5f} |")
    lines.append("")
    lines.append("**Interpretation**: " + _interpret_solo(coh_stats))
    lines.append("")

    if shuffled_reports is not None:
        sh_stats = aggregate_hce_stats(shuffled_reports)
        lines.append("## Hidden-shuffled 4D")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| n_candidates | {sh_stats['n_candidates']} |")
        lines.append(f"| mean HCE | {sh_stats['mean_HCE']:.5f} |")
        lines.append(f"| median HCE | {sh_stats['median_HCE']:.5f} |")
        lines.append(f"| mean visible_final_l1 | {sh_stats['mean_visible_final_l1']:.5f} |")
        lines.append(f"| mean HCE / visible ratio | {sh_stats['mean_ratio']:.3f} |")
        lines.append(f"| fraction HCE > 0 | {sh_stats['fraction_hce_positive']:.2f} |")
        lines.append("")
        lines.append("## Headline: coherent vs shuffled (paired by track_id)")
        lines.append("")
        paired = compare_hce_paired(coherent_reports, shuffled_reports)
        lines.append(f"- N paired = **{paired['n_paired']}**")
        lines.append(f"- Mean HCE diff (coh - shuf) = "
                     f"**{paired['mean_diff_coh_minus_shuf']:+.5f}**")
        lines.append(f"- 95% bootstrap CI = "
                     f"[{paired['bootstrap_ci_low']:+.5f}, "
                     f"{paired['bootstrap_ci_high']:+.5f}]")
        lines.append(f"- Sign-test p = {paired['sign_test_p']:.4f}")
        lines.append(f"- Coherent wins / shuffled wins = "
                     f"{paired['n_coherent_wins']} / {paired['n_shuffled_wins']}")
        lines.append("")
        lines.append("**Interpretation**: " + _interpret_paired(paired))
        lines.append("")

    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `hidden_causal_summary.csv` — one row per (condition, candidate)")
    lines.append("- `hidden_causal_trajectories.json` — full per-step lists per replicate")
    lines.append("- `stats_summary.json` — aggregate stats per condition + paired comparison")
    lines.append("- `plots/*.png`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = f"outputs/{args.label}_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"M6 -> {out_dir}")
    print(f"  top_k={args.top_k} n_steps={args.n_steps} "
          f"n_replicates={args.n_replicates} backend={args.backend}")

    # ---------------- coherent run
    print("\n=== Coherent 4D ===")
    cfg_c, store_c, _, tracks_c, _, obs_c = _open_or_create_run(
        run_dir_arg=args.from_run, config_arg=args.config,
        out_dir=out_dir, suffix="coherent", shuffled=False,
    )
    t0 = time.time()
    coherent_reports = _run_m6_on_run(
        label="coh", cfg=cfg_c, store=store_c,
        tracks=tracks_c, observer_scores=obs_c, args=args,
    )
    coh_seconds = time.time() - t0

    # ---------------- shuffled run (optional)
    shuffled_reports: list[HiddenCausalReport] | None = None
    sh_seconds = 0.0
    if args.shuffled_config:
        print("\n=== Hidden-shuffled 4D ===")
        cfg_s, store_s, _, tracks_s, _, obs_s = _open_or_create_run(
            run_dir_arg=None, config_arg=args.shuffled_config,
            out_dir=out_dir, suffix="shuffled", shuffled=True,
        )
        t0 = time.time()
        shuffled_reports = _run_m6_on_run(
            label="shuf", cfg=cfg_s, store=store_s,
            tracks=tracks_s, observer_scores=obs_s, args=args,
        )
        sh_seconds = time.time() - t0

    # ---------------- aggregate stats
    coh_stats = aggregate_hce_stats(coherent_reports)
    full_stats = {"coherent": coh_stats}
    if shuffled_reports is not None:
        full_stats["shuffled"] = aggregate_hce_stats(shuffled_reports)
        full_stats["paired_coh_minus_shuf"] = compare_hce_paired(
            coherent_reports, shuffled_reports
        )
    (out_dir / "stats_summary.json").write_text(
        json.dumps(full_stats, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                   else (int(o) if isinstance(o, np.integer)
                         else (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    )

    # ---------------- writers
    reports_dict: dict[str, list[HiddenCausalReport]] = {"coherent": coherent_reports}
    if shuffled_reports is not None:
        reports_dict["shuffled"] = shuffled_reports
    _write_summary_csv(reports_dict, out_dir / "hidden_causal_summary.csv")
    _write_trajectories_json(reports_dict, out_dir / "hidden_causal_trajectories.json")

    # ---------------- plots
    print("writing plots...")
    from observer_worlds.analysis.m6_plots import write_all_m6_plots
    write_all_m6_plots(reports_dict, plots_dir)

    # ---------------- summary
    summary = _build_combined_summary(coherent_reports, shuffled_reports, args, out_dir)
    (out_dir / "summary.md").write_text(summary)

    cfg_dump = {
        "from_run": args.from_run, "config": args.config,
        "shuffled_config": args.shuffled_config,
        "top_k": args.top_k, "n_steps": args.n_steps,
        "n_replicates": args.n_replicates,
        "label": args.label, "seed": args.seed, "backend": args.backend,
        "coh_seconds": coh_seconds, "sh_seconds": sh_seconds,
        "n_coherent_reports": len(coherent_reports),
        "n_shuffled_reports": (len(shuffled_reports) if shuffled_reports else 0),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"\nDone. Run dir: {out_dir}")
    print(f"Coherent: mean HCE = {coh_stats['mean_HCE']:.5f}  "
          f"(N={coh_stats['n_candidates']})")
    if shuffled_reports is not None:
        sh = full_stats["shuffled"]
        paired = full_stats["paired_coh_minus_shuf"]
        print(f"Shuffled: mean HCE = {sh['mean_HCE']:.5f}  "
              f"(N={sh['n_candidates']})")
        print(f"Paired  : diff = {paired['mean_diff_coh_minus_shuf']:+.5f}  "
              f"sign-test p = {paired['sign_test_p']:.4f}  "
              f"N_paired = {paired['n_paired']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
