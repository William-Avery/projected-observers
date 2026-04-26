"""M7 — HCE holdout validation.

Compares four conditions on **test seeds** (disjoint from train + val):
    A. M4A viable rules
    B. M4C observer-optimized rules
    C. M7 HCE-optimized rules
    D. optimized 2D baseline (HCE undefined; observer_score only)

Primary success criterion (per spec):
    M7 rules produce candidates with **stronger local hidden causal
    dependence** than M4A/M4C rules while **retaining nontrivial
    observer_score and persistence**.

This CLI emits canonical interpretation paragraphs based on which
comparisons survive paired-difference significance, and runs a
threshold-filtered HCE re-analysis to check that any M7 advantage
isn't just exploiting projection-threshold sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.analysis.m6c_stats import threshold_artifact_audit
from observer_worlds.experiments._m6c_taxonomy import M6CRow, run_m6c_taxonomy
from observer_worlds.experiments.run_m4b_observer_sweep import load_top_rules
from observer_worlds.search import FractionalRule


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M7 HCE holdout validation.")
    p.add_argument("--m7-rules", type=str, required=True,
                   help="Path to top_hce_rules.json from evolve_4d_hce_rules.")
    p.add_argument("--m4c-rules", type=str, default=None,
                   help="Optional M4C leaderboard for comparison.")
    p.add_argument("--m4a-rules", type=str, default=None,
                   help="Optional M4A leaderboard for comparison.")
    p.add_argument("--optimized-2d-rules", type=str, default=None,
                   help="Optional optimized 2D top_2d_rules.json for "
                        "observer_score-only comparison.")
    p.add_argument("--n-rules", type=int, default=5,
                   help="Top-N from each source.")
    p.add_argument("--test-seeds", type=int, nargs="+",
                   default=[3000, 3001, 3002, 3003, 3004,
                            3005, 3006, 3007, 3008, 3009])
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--grid", type=int, nargs=4, default=[64, 64, 8, 8])
    p.add_argument("--max-candidates", type=int, default=30)
    p.add_argument("--hce-replicates", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    p.add_argument("--backend", choices=["numba", "numpy"], default="numpy")
    p.add_argument("--out-root", type=str, default="outputs")
    p.add_argument("--label", type=str, default="m7_holdout")
    p.add_argument("--quick", action="store_true",
                   help="Reduce defaults: 1 rule per source, 2 seeds, T=80, "
                        "grid 16x16x4x4, max_cand=4, replicates=1, horizons=[5,10].")
    return p


def _quick(args):
    if args.quick:
        args.n_rules = min(args.n_rules, 1)
        args.test_seeds = args.test_seeds[:2]
        args.timesteps = min(args.timesteps, 80)
        args.grid = [16, 16, 4, 4]
        args.max_candidates = min(args.max_candidates, 4)
        args.hce_replicates = min(args.hce_replicates, 1)
        args.horizons = [5, 10]
    return args


def _load_with_source(path, n, tag):
    rules = load_top_rules(Path(path), n)
    return [(r, f"{tag}_rank{i:02d}", tag) for i, r in enumerate(rules, 1)]


def _load_optimized_2d(path, n_rules):
    """Load 2D rules from top_2d_rules.json. Returns list of (rule, rule_id, tag)."""
    data = json.loads(Path(path).read_text())
    out = []
    for i, d in enumerate(data[:n_rules], start=1):
        rule = FractionalRule.from_dict(d)
        out.append((rule, f"M4D_2D_optimized_rank{i:02d}", "M4D_2D_optimized"))
    return out


# ---------------------------------------------------------------------------
# Per-source aggregates
# ---------------------------------------------------------------------------


def _aggregate_source(rows: list[M6CRow], horizon: int) -> dict:
    """Aggregate M6C rows for one rule source at a given horizon."""
    sub = [r for r in rows if r.horizon == horizon]
    if not sub:
        return {"n": 0, "mean_observer": 0.0, "mean_lifetime": 0.0,
                "mean_future_div": 0.0, "mean_local_div": 0.0,
                "mean_vs_sham": 0.0, "mean_vs_far": 0.0,
                "mean_HCE": 0.0, "mean_near_threshold": 0.0}
    obs = [r.observer_score for r in sub if r.observer_score is not None]
    return {
        "n": len(sub),
        "n_unique_candidates": len({(r.rule_id, r.seed, r.candidate_id) for r in sub}),
        "n_unique_rules": len({r.rule_id for r in sub}),
        "mean_observer": float(np.mean(obs)) if obs else 0.0,
        "mean_lifetime": float(np.mean([r.candidate_lifetime for r in sub])),
        "mean_future_div": float(np.mean([r.future_div_hidden_invisible for r in sub])),
        "mean_local_div": float(np.mean([r.local_div_hidden_invisible for r in sub])),
        "mean_vs_sham": float(np.mean([r.hidden_vs_sham_delta for r in sub])),
        "mean_vs_far": float(np.mean([r.hidden_vs_far_delta for r in sub])),
        "mean_HCE": float(np.mean([r.HCE for r in sub])),
        "mean_near_threshold": float(np.mean(
            [r.features.get("near_threshold_fraction", 0.0) for r in sub]
        )),
    }


def _paired_compare_sources(rows_a: list, rows_b: list, *, horizon: int,
                            field: str) -> dict:
    """Per (test_seed, candidate_idx_within_seed) compare M7 vs M4C/M4A.

    Since rule ids differ across sources, we pair rules by *rank* within
    source and use grouped-bootstrap resampling by (rule_rank, seed).
    """
    by_key_a = {}
    by_key_b = {}
    for r in rows_a:
        if r.horizon != horizon: continue
        # Use rule_id and seed as the grouping key.
        key = (r.rule_id, r.seed)
        by_key_a.setdefault(key, []).append(getattr(r, field))
    for r in rows_b:
        if r.horizon != horizon: continue
        key = (r.rule_id, r.seed)
        by_key_b.setdefault(key, []).append(getattr(r, field))
    # We compare distributions, not paired (different rule ids).
    a_means = [float(np.mean(v)) for v in by_key_a.values()]
    b_means = [float(np.mean(v)) for v in by_key_b.values()]
    if not a_means or not b_means:
        return {"n_a": len(a_means), "n_b": len(b_means),
                "mean_diff_a_minus_b": 0.0,
                "bootstrap_ci_low": 0.0, "bootstrap_ci_high": 0.0,
                "p_a_gt_b": 0.5}
    a = np.array(a_means); b = np.array(b_means)
    diff = float(a.mean() - b.mean())
    # Bootstrap on the difference of independent means.
    rng = np.random.default_rng(0)
    n_boot = 2000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        boot_diffs[i] = sa.mean() - sb.mean()
    p_a_gt_b = float((a[:, None] > b[None, :]).mean())
    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff_a_minus_b": diff,
        "bootstrap_ci_low": float(np.quantile(boot_diffs, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot_diffs, 0.975)),
        "p_a_gt_b": p_a_gt_b,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_candidate_metrics_csv(rows_by_source: dict, path: Path, horizon: int):
    cols = ("source", "rule_id", "seed", "candidate_id", "snapshot_t",
            "horizon", "observer_score", "candidate_lifetime",
            "future_div_hidden_invisible", "local_div_hidden_invisible",
            "future_div_sham", "local_div_far_hidden",
            "hidden_vs_sham_delta", "hidden_vs_far_delta",
            "HCE", "near_threshold_fraction", "mean_threshold_margin")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for src, rows in rows_by_source.items():
            for r in rows:
                if r.horizon != horizon: continue
                w.writerow([
                    src, r.rule_id, r.seed, r.candidate_id, r.snapshot_t,
                    r.horizon,
                    "" if r.observer_score is None else f"{r.observer_score:.4f}",
                    r.candidate_lifetime,
                    f"{r.future_div_hidden_invisible:.6f}",
                    f"{r.local_div_hidden_invisible:.6f}",
                    f"{r.future_div_sham:.6f}",
                    f"{r.local_div_far_hidden:.6f}",
                    f"{r.hidden_vs_sham_delta:.6f}",
                    f"{r.hidden_vs_far_delta:.6f}",
                    f"{r.HCE:.6f}",
                    f"{r.features.get('near_threshold_fraction', 0):.4f}",
                    f"{r.features.get('mean_threshold_margin', 0):.4f}",
                ])


def _write_threshold_audit_csv(rows_by_source: dict, path: Path, horizon: int):
    cols = ("source", "filter", "n_candidates", "mean_future_div",
            "mean_vs_sham", "mean_vs_far", "fraction_future_div_gt_zero")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for src, rows in rows_by_source.items():
            audit = threshold_artifact_audit(rows, horizon=horizon)
            for a in audit:
                w.writerow([src, a["filter"], a["n_candidates"],
                           f"{a['mean_future_div']:.6f}",
                           f"{a['mean_vs_sham']:.6f}",
                           f"{a['mean_vs_far']:.6f}",
                           f"{a['fraction_future_div_gt_zero']:.4f}"])


def _write_condition_summary_csv(rows_by_source: dict, path: Path,
                                 horizon: int):
    cols = ("source", "n_candidates", "mean_observer", "mean_lifetime",
            "mean_future_div", "mean_local_div", "mean_vs_sham",
            "mean_vs_far", "mean_HCE", "mean_near_threshold")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for src, rows in rows_by_source.items():
            agg = _aggregate_source(rows, horizon)
            w.writerow([
                src, agg["n_unique_candidates"],
                f"{agg['mean_observer']:.4f}",
                f"{agg['mean_lifetime']:.1f}",
                f"{agg['mean_future_div']:.6f}",
                f"{agg['mean_local_div']:.6f}",
                f"{agg['mean_vs_sham']:.6f}",
                f"{agg['mean_vs_far']:.6f}",
                f"{agg['mean_HCE']:.6f}",
                f"{agg['mean_near_threshold']:.4f}",
            ])


def _write_paired_diffs_csv(rows_m7, rows_m4c, rows_m4a, path: Path,
                            horizon: int):
    cols = ("comparison", "metric", "n_a", "n_b", "mean_a", "mean_b",
            "mean_diff_a_minus_b", "ci_low", "ci_high", "p_a_gt_b")
    metrics = ("hidden_vs_sham_delta", "hidden_vs_far_delta",
               "future_div_hidden_invisible", "local_div_hidden_invisible",
               "HCE", "observer_score")
    pairs = []
    if rows_m4c is not None:
        pairs.append(("M7_vs_M4C", rows_m7, rows_m4c))
    if rows_m4a is not None:
        pairs.append(("M7_vs_M4A", rows_m7, rows_m4a))

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cmp_name, a, b in pairs:
            for metric in metrics:
                if metric == "observer_score":
                    # Special handling: filter None.
                    aa = [r for r in a if r.horizon == horizon and r.observer_score is not None]
                    bb = [r for r in b if r.horizon == horizon and r.observer_score is not None]
                    avals = [r.observer_score for r in aa]
                    bvals = [r.observer_score for r in bb]
                    if not avals or not bvals: continue
                    arr_a = np.array(avals); arr_b = np.array(bvals)
                    diff = float(arr_a.mean() - arr_b.mean())
                    rng = np.random.default_rng(0)
                    boot = np.empty(2000)
                    for i in range(2000):
                        sa = rng.choice(arr_a, size=arr_a.size, replace=True)
                        sb = rng.choice(arr_b, size=arr_b.size, replace=True)
                        boot[i] = sa.mean() - sb.mean()
                    p_a_gt_b = float((arr_a[:, None] > arr_b[None, :]).mean())
                    w.writerow([cmp_name, metric, arr_a.size, arr_b.size,
                               f"{arr_a.mean():.6f}", f"{arr_b.mean():.6f}",
                               f"{diff:.6f}",
                               f"{np.quantile(boot, 0.025):.6f}",
                               f"{np.quantile(boot, 0.975):.6f}",
                               f"{p_a_gt_b:.4f}"])
                    continue
                stats = _paired_compare_sources(a, b, horizon=horizon, field=metric)
                w.writerow([cmp_name, metric, stats["n_a"], stats["n_b"],
                           f"{stats.get('mean_a', 0):.6f}",
                           f"{stats.get('mean_b', 0):.6f}",
                           f"{stats['mean_diff_a_minus_b']:.6f}",
                           f"{stats['bootstrap_ci_low']:.6f}",
                           f"{stats['bootstrap_ci_high']:.6f}",
                           f"{stats['p_a_gt_b']:.4f}"])


# ---------------------------------------------------------------------------
# Plots (compact set)
# ---------------------------------------------------------------------------


def _plot_hce_by_source(rows_by_source: dict, path: Path, horizon: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for src, rows in rows_by_source.items():
        sub = [r.future_div_hidden_invisible for r in rows if r.horizon == horizon]
        if sub:
            data.append(sub); labels.append(f"{src}\n(N={len(sub)})")
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("future_projection_divergence (HCE)")
    ax.set_title(f"HCE distribution by rule source (horizon={horizon})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _plot_observer_vs_hce(rows_by_source: dict, path: Path, horizon: int):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"M7_HCE_optimized": "#1f77b4",
               "M4C_observer_optimized": "#ff7f0e",
               "M4A_viability": "#2ca02c",
               "M4D_2D_optimized": "#7f7f7f"}
    for src, rows in rows_by_source.items():
        sub = [r for r in rows if r.horizon == horizon
               and r.observer_score is not None]
        if not sub: continue
        x = [r.observer_score for r in sub]
        y = [r.future_div_hidden_invisible for r in sub]
        ax.scatter(x, y, s=15, alpha=0.5, label=src,
                   color=palette.get(src, "#999"))
    ax.set_xlabel("observer_score"); ax.set_ylabel("HCE (future divergence)")
    ax.set_title("Observer-likeness vs HCE tradeoff (per candidate)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _plot_threshold_audit_by_source(audit_by_source: dict, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    sources = list(audit_by_source.keys())
    filters = ["all_candidates", "near_threshold_fraction<0.25",
               "near_threshold_fraction<0.1", "mean_threshold_margin>0.10"]
    x = np.arange(len(filters)); width = 0.20
    for i, src in enumerate(sources):
        vals = []
        for fname in filters:
            row = next((a for a in audit_by_source[src]
                       if a["filter"] == fname), None)
            vals.append(row["mean_future_div"] if row else 0.0)
        ax.bar(x + (i - len(sources) / 2 + 0.5) * width, vals, width=width,
               label=src, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(filters, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("mean future_div"); ax.set_title("Threshold audit by source")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _interpret(summary: dict, horizon: int) -> list[str]:
    """Apply the M7 holdout interpretation rules from the spec."""
    out = []
    by_src = summary["aggregates"]
    audit = summary["threshold_audit"]
    if "M7_HCE_optimized" not in by_src:
        return ["M7 rules absent; cannot interpret."]
    m7 = by_src["M7_HCE_optimized"]
    m4c = by_src.get("M4C_observer_optimized")
    m4a = by_src.get("M4A_viability")
    opt2d = by_src.get("M4D_2D_optimized")

    # Rule: M7 HCE > 0 away from threshold.
    m7_audit = audit.get("M7_HCE_optimized", [])
    far_row = next((a for a in m7_audit
                   if a["filter"] == "mean_threshold_margin>0.10"), None)
    if far_row and far_row["mean_future_div"] > 0 and \
            far_row["fraction_future_div_gt_zero"] > 0.5:
        out.append("M7 found non-threshold-mediated hidden causal dependence.")

    # M7 vs M4C HCE.
    if m4c is not None:
        # HCE improvement vs observer collapse.
        hce_improved = m7["mean_vs_sham"] > m4c["mean_vs_sham"] + 0.005
        obs_collapsed = m7["mean_observer"] < m4c["mean_observer"] - 0.05
        obs_preserved = m7["mean_observer"] > m4c["mean_observer"] - 0.05
        if hce_improved and obs_collapsed:
            out.append(
                "HCE can be optimized, but the search found hidden "
                "sensitivity rather than observer-like projected candidates."
            )
        elif hce_improved and obs_preserved:
            out.append(
                "HCE-guided search found candidates with both observer-like "
                "projected structure and hidden causal dependence."
            )

    # Threshold-mediated check on M7.
    m7_all = next((a for a in m7_audit if a["filter"] == "all_candidates"), None)
    m7_far = next((a for a in m7_audit
                  if a["filter"] == "mean_threshold_margin>0.10"), None)
    if m7_all and m7_far and m7_all["mean_future_div"] > 0:
        ratio = m7_far["mean_future_div"] / m7_all["mean_future_div"]
        if ratio < 0.3:
            out.append("HCE optimization exploited projection-threshold sensitivity.")

    # Global vs local check on M7.
    if m7["mean_local_div"] > 0:
        local_far_ratio = m7["mean_vs_far"] / max(m7["mean_local_div"], 1e-9)
        if m7["mean_future_div"] > 2 * m7["mean_local_div"] and local_far_ratio < 0.5:
            out.append(
                "The search found globally chaotic hidden sensitivity, not "
                "candidate-local hidden support."
            )

    if not out:
        out.append("Mixed result; no strong directional conclusion.")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _quick(build_arg_parser().parse_args(argv))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_sims"; workdir.mkdir(parents=True, exist_ok=True)

    sources = []
    sources.append(("M7_HCE_optimized",
                   _load_with_source(args.m7_rules, args.n_rules, "M7_HCE_optimized")))
    if args.m4c_rules:
        sources.append(("M4C_observer_optimized",
                       _load_with_source(args.m4c_rules, args.n_rules,
                                         "M4C_observer_optimized")))
    if args.m4a_rules:
        sources.append(("M4A_viability",
                       _load_with_source(args.m4a_rules, args.n_rules, "M4A_viability")))
    optim_2d_rules = []
    if args.optimized_2d_rules:
        optim_2d_rules = _load_optimized_2d(args.optimized_2d_rules, args.n_rules)

    grid = tuple(args.grid)
    cfg_dump = {
        "m7_rules": args.m7_rules, "m4c_rules": args.m4c_rules,
        "m4a_rules": args.m4a_rules, "optimized_2d_rules": args.optimized_2d_rules,
        "n_rules": args.n_rules, "test_seeds": args.test_seeds,
        "timesteps": args.timesteps, "grid": list(grid),
        "max_candidates": args.max_candidates,
        "hce_replicates": args.hce_replicates, "horizons": args.horizons,
        "backend": args.backend,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    print(f"M7 holdout -> {out_dir}")
    print(f"  test_seeds={args.test_seeds}  T={args.timesteps}  grid={grid}")
    print(f"  sources: {[s[0] for s in sources]}")
    if optim_2d_rules:
        print(f"  + optimized_2d ({len(optim_2d_rules)} rules; observer-only)")

    # Run M6C-style measurement on each 4D source.
    rows_by_source = {}
    t0 = time.time()
    for src_name, rule_list in sources:
        if not rule_list: continue
        print(f"\n=== {src_name} ===")
        rows = run_m6c_taxonomy(
            rules=rule_list, seeds=args.test_seeds, grid_shape=grid,
            timesteps=args.timesteps, max_candidates=args.max_candidates,
            horizons=args.horizons, n_replicates=args.hce_replicates,
            backend=args.backend, workdir=workdir / src_name, progress=print,
        )
        rows_by_source[src_name] = rows
        print(f"  {src_name}: {len(rows)} rows")

    elapsed = time.time() - t0
    print(f"\n4D measurement done in {elapsed:.0f}s")

    # 2D optimized: just observer score (no HCE/snapshots needed).
    if optim_2d_rules:
        from observer_worlds.search.observer_search_2d import (
            evaluate_observer_fitness_2d,
        )
        obs2d_results = []
        for rule_2d, rid, _ in optim_2d_rules:
            for seed in args.test_seeds:
                r = evaluate_observer_fitness_2d(
                    rule_2d, n_seeds=1, base_seed=seed,
                    grid_shape=(grid[0], grid[1]), timesteps=args.timesteps,
                )
                obs2d_results.append({
                    "rule_id": rid, "seed": seed,
                    "observer_score": float(r.mean_lifetime_weighted_mean_score),
                    "n_candidates": int(r.mean_n_candidates),
                })
        with (out_dir / "optimized_2d_observer_scores.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rule_id", "seed", "observer_score", "n_candidates"])
            for r in obs2d_results:
                w.writerow([r["rule_id"], r["seed"],
                           f"{r['observer_score']:.4f}", r["n_candidates"]])

    # Aggregates + audits.
    headline_h = args.horizons[len(args.horizons) // 2]
    aggregates = {src: _aggregate_source(rows, headline_h)
                  for src, rows in rows_by_source.items()}
    audit_by_source = {src: threshold_artifact_audit(rows, horizon=headline_h)
                       for src, rows in rows_by_source.items()}

    # CSV writers.
    _write_candidate_metrics_csv(rows_by_source,
                                out_dir / "candidate_metrics.csv", headline_h)
    _write_threshold_audit_csv(rows_by_source, out_dir / "threshold_audit.csv",
                              headline_h)
    _write_condition_summary_csv(rows_by_source,
                                out_dir / "condition_summary.csv", headline_h)
    _write_paired_diffs_csv(
        rows_by_source.get("M7_HCE_optimized", []),
        rows_by_source.get("M4C_observer_optimized"),
        rows_by_source.get("M4A_viability"),
        out_dir / "paired_differences.csv", headline_h,
    )

    summary = {
        "headline_horizon": headline_h,
        "aggregates": aggregates,
        "threshold_audit": audit_by_source,
        "test_seeds": args.test_seeds,
    }
    (out_dir / "stats_summary.json").write_text(
        json.dumps(summary, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                   else (int(o) if isinstance(o, np.integer)
                         else (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    )

    # Plots.
    _plot_hce_by_source(rows_by_source, plots_dir / "hce_by_source.png",
                       headline_h)
    _plot_observer_vs_hce(rows_by_source, plots_dir / "observer_vs_hce.png",
                         headline_h)
    _plot_threshold_audit_by_source(audit_by_source,
                                   plots_dir / "threshold_audit_by_source.png")

    # Summary.md.
    md = [f"# M7 HCE holdout validation — {args.label}", ""]
    md.append(f"- Test seeds: {args.test_seeds}")
    md.append(f"- Timesteps: {args.timesteps}, grid: {grid}")
    md.append(f"- Headline horizon: {headline_h}")
    md.append(f"- Wall time: {elapsed:.0f}s")
    md.append("")
    md.append("## Per-source summary at headline horizon")
    md.append("")
    md.append("| source | n_cand | mean_obs | mean_life | mean_future_div | "
              "mean_vs_sham | mean_vs_far | mean_HCE | mean_near_thresh |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for src, agg in aggregates.items():
        md.append(
            f"| {src} | {agg['n_unique_candidates']} | "
            f"{agg['mean_observer']:+.3f} | {agg['mean_lifetime']:.0f} | "
            f"{agg['mean_future_div']:+.4f} | {agg['mean_vs_sham']:+.4f} | "
            f"{agg['mean_vs_far']:+.4f} | {agg['mean_HCE']:+.4f} | "
            f"{agg['mean_near_threshold']:.2f} |"
        )
    md.append("")
    md.append("## Threshold audit per source")
    md.append("")
    for src, audit in audit_by_source.items():
        md.append(f"### {src}")
        md.append("")
        md.append("| filter | n | mean_future_div | mean_vs_sham | mean_vs_far | fraction>0 |")
        md.append("|---|---|---|---|---|---|")
        for a in audit:
            md.append(
                f"| {a['filter']} | {a['n_candidates']} | "
                f"{a['mean_future_div']:+.4f} | {a['mean_vs_sham']:+.4f} | "
                f"{a['mean_vs_far']:+.4f} | "
                f"{a['fraction_future_div_gt_zero']:.2f} |"
            )
        md.append("")
    md.append("## Interpretation")
    md.append("")
    for line in _interpret(summary, headline_h):
        md.append(f"- {line}")
    (out_dir / "summary.md").write_text("\n".join(md))

    print(f"\nDone. Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
