"""Stage 5C2 post-hoc analysis on a Stage-5C run directory.

Operates on the CSV artifacts only — no simulation. Promotes
``normalized_HCE = HCE / (HCE + far_HCE + eps)`` to first-class output
and adds grouped-bootstrap CIs for per (projection × source) means.

Usage::

    python -m observer_worlds.analysis.projection_robustness_posthoc \\
        --run-dir outputs/stage5c_projection_robustness_production_<ts>/ \\
        --n-boot 2000

Outputs (under the run dir):

    projection_robustness_posthoc.csv
    projection_robustness_posthoc.json
    projection_robustness_posthoc_summary.md
    plots/normalized_hce_ci_by_projection.png
    plots/m7_advantage_by_projection.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SOURCES_DEFAULT: tuple[str, ...] = (
    "M7_HCE_optimized",
    "M4C_observer_optimized",
    "M4A_viability",
)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truthy_valid(r):
    v = r.get("valid")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return False


def _safe_mean(xs):
    xs = [float(x) for x in xs if x not in (None, "", "None")]
    return float(np.mean(xs)) if xs else None


def _normalized_hce(hce: float, far: float) -> float:
    denom = abs(hce) + abs(far) + EPS
    return float(hce / denom)


def _grouped_bootstrap_mean(
    values: list[float], groups: list, *, n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float, float] | None:
    """Cluster bootstrap of the mean by ``(rule_id, seed)`` group.

    Returns ``(point, ci_low, ci_high)`` or ``None`` if too few groups."""
    if not values:
        return None
    vals = np.asarray(values, dtype=np.float64)
    grp = np.asarray(groups)
    rng = np.random.default_rng(int(seed))
    unique = np.unique(grp)
    if unique.size < 2:
        return float(vals.mean()), None, None
    idx_by_g = {g: np.where(grp == g)[0] for g in unique}
    boots = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idxs = np.concatenate([idx_by_g[g] for g in sampled])
        boots[i] = float(vals[idxs].mean())
    return (
        float(vals.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def _grouped_bootstrap_diff(
    a_vals, a_groups, b_vals, b_groups, *, n_boot: int = 2000, seed: int = 0,
) -> dict | None:
    if not a_vals or not b_vals:
        return None
    a = np.asarray(a_vals, dtype=np.float64); b = np.asarray(b_vals, dtype=np.float64)
    ag = np.asarray(a_groups); bg = np.asarray(b_groups)
    rng = np.random.default_rng(int(seed))
    ua = np.unique(ag); ub = np.unique(bg)
    if ua.size < 2 or ub.size < 2:
        return {
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(a.mean() - b.mean()),
            "ci_low": None, "ci_high": None,
            "n_a": int(a.size), "n_b": int(b.size),
            "ci_excludes_zero": False,
        }
    idx_a = {g: np.where(ag == g)[0] for g in ua}
    idx_b = {g: np.where(bg == g)[0] for g in ub}
    diffs = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        sa = rng.choice(ua, size=ua.size, replace=True)
        sb = rng.choice(ub, size=ub.size, replace=True)
        ia = np.concatenate([idx_a[g] for g in sa])
        ib = np.concatenate([idx_b[g] for g in sb])
        diffs[i] = float(a[ia].mean() - b[ib].mean())
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": lo, "ci_high": hi,
        "ci_excludes_zero": (lo > 0.0 or hi < 0.0),
        "n_a": int(a.size), "n_b": int(b.size),
    }


def _cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0: return 0.0
    return float(((a[:, None] > b[None, :]).sum()
                   - (a[:, None] < b[None, :]).sum())
                  / (a.size * b.size))


# ---------------------------------------------------------------------------
# Per (projection × source) aggregation with CIs
# ---------------------------------------------------------------------------


def aggregate_with_cis(
    candidate_rows: list[dict], projections: list[str], sources: list[str],
    *, n_boot: int = 2000, seed: int = 0,
) -> dict:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in candidate_rows:
        if not _truthy_valid(r):
            continue
        by_key[(r["projection"], r["rule_source"])].append(r)
    out: dict[str, dict[str, dict]] = {}
    for proj in projections:
        out[proj] = {}
        for src in sources:
            rs = by_key.get((proj, src), [])
            if not rs:
                out[proj][src] = {"n": 0}
                continue
            def col(k):
                return [float(r[k]) for r in rs
                        if r.get(k) not in (None, "", "None")]
            hces = col("HCE"); fars = col("far_HCE")
            d_far = col("hidden_vs_far_delta")
            init = col("initial_projection_delta")
            groups = [f"{r['rule_id']}|{r['seed']}" for r in rs]
            # Normalized HCE per candidate.
            norm = []
            for h, f in zip(hces, fars):
                norm.append(_normalized_hce(h, f))
            # Grouped bootstraps.
            hce_ci = _grouped_bootstrap_mean(hces, groups, n_boot=n_boot, seed=seed)
            far_ci = _grouped_bootstrap_mean(fars, groups, n_boot=n_boot, seed=seed + 1)
            d_far_ci = _grouped_bootstrap_mean(d_far, groups, n_boot=n_boot, seed=seed + 2)
            norm_ci = _grouped_bootstrap_mean(norm, groups, n_boot=n_boot, seed=seed + 3)
            frac_local = float(np.mean([h > f for h, f in zip(hces, fars)]))
            out[proj][src] = {
                "n": len(rs),
                "n_groups": int(len(set(groups))),
                "mean_HCE": _safe_mean(hces),
                "HCE_ci": (None, None) if hce_ci is None else hce_ci[1:],
                "mean_far_HCE": _safe_mean(fars),
                "mean_hidden_vs_far_delta": _safe_mean(d_far),
                "hidden_vs_far_delta_ci":
                    (None, None) if d_far_ci is None else d_far_ci[1:],
                "mean_normalized_HCE": _safe_mean(norm),
                "normalized_HCE_ci":
                    (None, None) if norm_ci is None else norm_ci[1:],
                "fraction_HCE_gt_far": frac_local,
                "mean_initial_projection_delta": _safe_mean(init),
            }
    return out


# ---------------------------------------------------------------------------
# Robustness summary
# ---------------------------------------------------------------------------


def robustness_summary(per_ps: dict) -> dict:
    """Cross-projection summary of robustness."""
    cells = []
    for proj, by_src in per_ps.items():
        for src, agg in by_src.items():
            if agg.get("n", 0) == 0: continue
            cells.append({
                "projection": proj, "source": src,
                "mean_HCE": agg.get("mean_HCE"),
                "mean_normalized_HCE": agg.get("mean_normalized_HCE"),
                "normalized_HCE_ci_low":
                    (agg.get("normalized_HCE_ci") or [None, None])[0],
                "normalized_HCE_ci_high":
                    (agg.get("normalized_HCE_ci") or [None, None])[1],
                "fraction_HCE_gt_far": agg.get("fraction_HCE_gt_far"),
            })
    n = len(cells)
    n_above = sum(1 for c in cells
                  if c["mean_normalized_HCE"] is not None
                  and c["mean_normalized_HCE"] > 0.5)
    n_ci_above = sum(1 for c in cells
                     if c["normalized_HCE_ci_low"] is not None
                     and c["normalized_HCE_ci_low"] > 0.5)
    proj_means: dict[str, list[float]] = defaultdict(list)
    for c in cells:
        if c["mean_normalized_HCE"] is not None:
            proj_means[c["projection"]].append(c["mean_normalized_HCE"])
    proj_ranking = sorted(
        ((p, float(np.mean(v))) for p, v in proj_means.items()),
        key=lambda x: -x[1],
    )
    return {
        "n_cells": n,
        "n_cells_normalized_hce_gt_0_5": n_above,
        "n_cells_ci_lower_bound_gt_0_5": n_ci_above,
        "fraction_cells_normalized_hce_gt_0_5":
            (n_above / n) if n else None,
        "fraction_cells_ci_lower_bound_gt_0_5":
            (n_ci_above / n) if n else None,
        "projection_ranking_by_mean_normalized_hce": proj_ranking,
    }


# ---------------------------------------------------------------------------
# M7-vs-baseline comparison summary
# ---------------------------------------------------------------------------


def m7_advantage_summary(
    candidate_rows: list[dict], projections: list[str],
    *, n_boot: int = 2000, seed: int = 0,
) -> dict:
    by_psrc: dict[tuple, list[dict]] = defaultdict(list)
    for r in candidate_rows:
        if not _truthy_valid(r):
            continue
        by_psrc[(r["projection"], r["rule_source"])].append(r)

    per_proj: dict[str, dict] = {}
    summary_buckets = {
        "M7_beats_M4C_clean": [],
        "M7_beats_M4A_clean": [],
        "M7_loses_M4C_clean": [],
        "M7_loses_M4A_clean": [],
        "M7_indistinguishable_M4C": [],
        "M7_indistinguishable_M4A": [],
    }
    for proj in projections:
        m7 = by_psrc.get((proj, "M7_HCE_optimized"), [])
        if not m7:
            per_proj[proj] = {"_status": "no M7 candidates"}
            continue
        per_proj[proj] = {}
        for baseline_full, label in (
            ("M4C_observer_optimized", "M4C"),
            ("M4A_viability", "M4A"),
        ):
            base = by_psrc.get((proj, baseline_full), [])
            if not base:
                per_proj[proj][label] = {"_status": "no baseline candidates"}
                continue
            entry: dict[str, dict] = {}
            for metric in ("HCE", "hidden_vs_far_delta"):
                a = [float(r[metric]) for r in m7
                     if r.get(metric) not in (None, "", "None")]
                b = [float(r[metric]) for r in base
                     if r.get(metric) not in (None, "", "None")]
                ag = [f"{r['rule_id']}|{r['seed']}" for r in m7
                      if r.get(metric) not in (None, "", "None")]
                bg = [f"{r['rule_id']}|{r['seed']}" for r in base
                      if r.get(metric) not in (None, "", "None")]
                cmp = _grouped_bootstrap_diff(
                    a, ag, b, bg, n_boot=n_boot, seed=seed,
                )
                if cmp is not None:
                    cmp["cliffs_delta"] = _cliffs_delta(a, b)
                entry[metric] = cmp
            per_proj[proj][label] = entry

            # Bucket for HCE only.
            hce = entry.get("HCE")
            if hce and hce.get("ci_low") is not None:
                if hce["ci_excludes_zero"] and hce["mean_diff"] > 0:
                    summary_buckets[f"M7_beats_{label}_clean"].append(proj)
                elif hce["ci_excludes_zero"] and hce["mean_diff"] < 0:
                    summary_buckets[f"M7_loses_{label}_clean"].append(proj)
                else:
                    summary_buckets[f"M7_indistinguishable_{label}"].append(proj)
    return {"per_projection": per_proj, "summary_buckets": summary_buckets}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_csv(per_ps: dict, path: Path) -> None:
    fields = [
        "projection", "source", "n", "n_groups",
        "mean_HCE", "HCE_ci_low", "HCE_ci_high",
        "mean_far_HCE",
        "mean_hidden_vs_far_delta",
        "hidden_vs_far_delta_ci_low", "hidden_vs_far_delta_ci_high",
        "mean_normalized_HCE",
        "normalized_HCE_ci_low", "normalized_HCE_ci_high",
        "fraction_HCE_gt_far",
        "mean_initial_projection_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for proj, by_src in per_ps.items():
            for src, agg in by_src.items():
                row = {"projection": proj, "source": src,
                       "n": agg.get("n", 0),
                       "n_groups": agg.get("n_groups", 0)}
                if agg.get("n", 0) == 0:
                    w.writerow({k: row.get(k, "") for k in fields}); continue
                hci = agg.get("HCE_ci") or (None, None)
                dci = agg.get("hidden_vs_far_delta_ci") or (None, None)
                nci = agg.get("normalized_HCE_ci") or (None, None)
                row.update({
                    "mean_HCE": agg.get("mean_HCE"),
                    "HCE_ci_low": hci[0], "HCE_ci_high": hci[1],
                    "mean_far_HCE": agg.get("mean_far_HCE"),
                    "mean_hidden_vs_far_delta": agg.get("mean_hidden_vs_far_delta"),
                    "hidden_vs_far_delta_ci_low": dci[0],
                    "hidden_vs_far_delta_ci_high": dci[1],
                    "mean_normalized_HCE": agg.get("mean_normalized_HCE"),
                    "normalized_HCE_ci_low": nci[0],
                    "normalized_HCE_ci_high": nci[1],
                    "fraction_HCE_gt_far": agg.get("fraction_HCE_gt_far"),
                    "mean_initial_projection_delta":
                        agg.get("mean_initial_projection_delta"),
                })
                w.writerow({k: row.get(k, "") for k in fields})


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def _fmt(v, fmt: str = "+.4f"):
    return f"{v:{fmt}}" if v is not None else "—"


def write_summary_md(payload: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage 5C2 — projection-robustness post-hoc analysis")
    lines.append("")
    lines.append("Computed from the existing Stage 5C run; no simulation rerun.")
    lines.append("")
    lines.append("**Cross-projection comparable metric: `normalized_HCE = "
                 "HCE / (HCE + far_HCE + eps)`**, in `[0, 1]`. Values "
                 "above 0.5 indicate the hidden perturbation produced more "
                 "candidate-local divergence than the far perturbation. "
                 "Raw HCE is **not comparable across binary vs continuous "
                 "projections** (random_linear is real-valued; its raw HCE "
                 "is on a different scale).")
    lines.append("")
    lines.append("`random_linear_projection` is reported as **near-invisible "
                 "under tolerance 5e-3**, not exactly invariant.")
    lines.append("")

    # Per (projection × source) main table.
    lines.append("## Per (projection × source) means with grouped-bootstrap CIs")
    lines.append("")
    sources = payload["sources_present"]
    lines.append(_md_row([
        "projection", "source", "n", "n_groups",
        "mean HCE", "HCE 95% CI",
        "mean normalized HCE", "norm. HCE 95% CI",
        "frac HCE > far",
    ]))
    lines.append(_md_row(["---"] * 9))
    for proj in payload["projections"]:
        for src in sources:
            agg = payload["per_projection_source"][proj][src]
            n = agg.get("n", 0)
            if n == 0:
                lines.append(_md_row([proj, src, "0", "—",
                                       "—", "—", "—", "—", "—"]))
                continue
            hci = agg.get("HCE_ci") or (None, None)
            nci = agg.get("normalized_HCE_ci") or (None, None)
            hce_ci_str = (
                f"[{hci[0]:+.3f}, {hci[1]:+.3f}]"
                if hci[0] is not None else "—"
            )
            n_ci_str = (
                f"[{nci[0]:+.3f}, {nci[1]:+.3f}]"
                if nci[0] is not None else "—"
            )
            lines.append(_md_row([
                proj, src, n, agg.get("n_groups", 0),
                _fmt(agg.get("mean_HCE")),
                hce_ci_str,
                _fmt(agg.get("mean_normalized_HCE"), "+.3f"),
                n_ci_str,
                _fmt(agg.get("fraction_HCE_gt_far"), ".2f"),
            ]))
    lines.append("")

    # Robustness summary.
    rs = payload["robustness_summary"]
    lines.append("## Projection-robustness summary")
    lines.append("")
    lines.append(f"* Cells (projection × source) measured: {rs['n_cells']}")
    lines.append(f"* Cells with mean normalized_HCE > 0.5: "
                 f"{rs['n_cells_normalized_hce_gt_0_5']} / "
                 f"{rs['n_cells']} "
                 f"({(rs['fraction_cells_normalized_hce_gt_0_5'] or 0)*100:.0f}%)")
    lines.append(f"* Cells with bootstrap CI lower bound > 0.5: "
                 f"{rs['n_cells_ci_lower_bound_gt_0_5']} / "
                 f"{rs['n_cells']} "
                 f"({(rs['fraction_cells_ci_lower_bound_gt_0_5'] or 0)*100:.0f}%)")
    lines.append("")
    lines.append("### Projection ranking by mean normalized_HCE")
    lines.append("")
    lines.append(_md_row(["projection", "mean normalized HCE"]))
    lines.append(_md_row(["---", "---:"]))
    for p, v in rs["projection_ranking_by_mean_normalized_hce"]:
        lines.append(_md_row([p, _fmt(v, "+.3f")]))
    lines.append("")

    # M7 advantage.
    adv = payload["m7_advantage"]
    lines.append("## M7 advantage by projection")
    lines.append("")
    buckets = adv["summary_buckets"]
    lines.append("Projections where M7 vs baseline HCE difference is "
                 "CI-clean (95% bootstrap CI excludes zero):")
    lines.append("")
    lines.append(_md_row([
        "comparison", "M7 wins", "M7 loses", "indistinguishable",
    ]))
    lines.append(_md_row(["---", "---", "---", "---"]))
    for label in ("M4C", "M4A"):
        lines.append(_md_row([
            f"M7 vs {label}",
            ", ".join(buckets[f"M7_beats_{label}_clean"]) or "—",
            ", ".join(buckets[f"M7_loses_{label}_clean"]) or "—",
            ", ".join(buckets[f"M7_indistinguishable_{label}"]) or "—",
        ]))
    lines.append("")
    lines.append("### Per-projection M7 vs baseline (HCE, grouped bootstrap)")
    lines.append("")
    lines.append(_md_row([
        "projection", "vs", "diff", "95% CI", "Cliff's δ", "CI excl 0",
    ]))
    lines.append(_md_row(["---"] * 6))
    for proj, by_b in adv["per_projection"].items():
        if isinstance(by_b, dict) and "_status" in by_b:
            continue
        for label, by_metric in by_b.items():
            cmp = (by_metric or {}).get("HCE")
            if not isinstance(cmp, dict) or cmp.get("ci_low") is None:
                continue
            ci = f"[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]"
            lines.append(_md_row([
                proj, f"M7_vs_{label}",
                _fmt(cmp.get("mean_diff")),
                ci, _fmt(cmp.get("cliffs_delta"), "+.3f"),
                "yes" if cmp.get("ci_excludes_zero") else "no",
            ]))
    lines.append("")

    # Activated language.
    lines.append("## Activated interpretation")
    lines.append("")
    if rs["fraction_cells_normalized_hce_gt_0_5"] == 1.0:
        lines.append(
            "* **HCE itself is projection-robust**: local hidden "
            "perturbations exceeded far perturbations across all "
            "projection/source cells."
        )
    elif (rs["fraction_cells_normalized_hce_gt_0_5"] or 0) > 0.8:
        lines.append(
            "* HCE robustness across projections is well-supported: "
            f"{rs['n_cells_normalized_hce_gt_0_5']} / {rs['n_cells']} "
            "cells exceed normalized_HCE > 0.5."
        )
    n_clean_M4C = len(buckets["M7_beats_M4C_clean"])
    n_clean_M4A = len(buckets["M7_beats_M4A_clean"])
    n_total_proj = len(adv["per_projection"])
    if n_clean_M4C < n_total_proj or n_clean_M4A < n_total_proj:
        lines.append(
            "* **M7's advantage is projection-conditional, not "
            "universal.** Of "
            f"{n_total_proj} projections measured, M7 wins CI-clean over "
            f"M4C on {n_clean_M4C} and over M4A on {n_clean_M4A}."
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_plots(payload: dict, plots_dir: Path) -> None:
    try:
        plt = _import_plt()
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    sources = payload["sources_present"]
    projections = payload["projections"]

    # Plot 1: normalized_HCE per (projection × source) with CI bars.
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    x = np.arange(len(projections))
    src_colors = {"M7_HCE_optimized": "#3a7",
                  "M4C_observer_optimized": "#357",
                  "M4A_viability": "#a73"}
    for i, src in enumerate(sources):
        means = []; lows = []; highs = []
        for proj in projections:
            agg = payload["per_projection_source"][proj][src]
            m = agg.get("mean_normalized_HCE")
            ci = agg.get("normalized_HCE_ci") or (None, None)
            if m is None:
                means.append(0.0); lows.append(0.0); highs.append(0.0)
                continue
            means.append(float(m))
            lows.append(float(m - ci[0]) if ci[0] is not None else 0.0)
            highs.append(float(ci[1] - m) if ci[1] is not None else 0.0)
        ax.bar(x + (i - 1) * width, means, width=width,
                yerr=[lows, highs], capsize=4,
                color=src_colors.get(src, "#666"),
                label=src.split("_")[0])
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--",
                label="0.5 (HCE = far)")
    ax.set_xticks(x); ax.set_xticklabels(
        [p.replace("_projection", "").replace("_threshold", "_thr")
         for p in projections], rotation=15)
    ax.set_ylabel("mean normalized_HCE = HCE / (HCE + far_HCE)")
    ax.set_title("normalized_HCE by projection × source (95% bootstrap CI)")
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(plots_dir / "normalized_hce_ci_by_projection.png", dpi=120)
    plt.close(fig)

    # Plot 2: M7 advantage (mean diff with CI) per projection vs M4C/M4A.
    adv = payload["m7_advantage"]["per_projection"]
    rows = []
    for proj in projections:
        by_b = adv.get(proj, {})
        if not isinstance(by_b, dict) or "_status" in by_b: continue
        for label in ("M4C", "M4A"):
            cmp = (by_b.get(label) or {}).get("HCE")
            if not isinstance(cmp, dict) or cmp.get("ci_low") is None:
                continue
            rows.append((proj, label, cmp["mean_diff"],
                         cmp["ci_low"], cmp["ci_high"],
                         cmp.get("ci_excludes_zero", False)))
    fig, ax = plt.subplots(figsize=(11, 5))
    if rows:
        labels = [f"{p}\nvs {b}" for p, b, *_ in rows]
        diffs = [r[2] for r in rows]
        lows = [r[2] - r[3] for r in rows]
        highs = [r[4] - r[2] for r in rows]
        colors = ["#3a7" if r[5] and r[2] > 0
                  else "#a37" if r[5] else "#999" for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, diffs, yerr=[lows, highs], capsize=4, color=colors)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_ylabel("M7 − baseline mean HCE  (95% bootstrap CI)")
        ax.set_title(
            "M7 advantage by projection (green = M7 wins CI-clean; "
            "rose = M7 loses CI-clean; grey = indistinguishable)"
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "m7_advantage_by_projection.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run(run_dir: Path, *, n_boot: int = 2000, seed: int = 0) -> dict:
    candidates_path = run_dir / "candidate_metrics.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"no candidate_metrics.csv in {run_dir}")
    rows = list(csv.DictReader(candidates_path.open(encoding="utf-8")))
    projections = sorted({r["projection"] for r in rows})
    sources = sorted({r["rule_source"] for r in rows})

    per_ps = aggregate_with_cis(
        rows, projections, sources, n_boot=n_boot, seed=seed,
    )
    rs = robustness_summary(per_ps)
    adv = m7_advantage_summary(rows, projections, n_boot=n_boot, seed=seed)
    payload = {
        "stage": "5C2",
        "run_dir": str(run_dir).replace("\\", "/"),
        "n_candidate_rows": len(rows),
        "projections": projections,
        "sources_present": sources,
        "n_boot": int(n_boot),
        "per_projection_source": per_ps,
        "robustness_summary": rs,
        "m7_advantage": adv,
    }
    out_csv = run_dir / "projection_robustness_posthoc.csv"
    out_json = run_dir / "projection_robustness_posthoc.json"
    out_md = run_dir / "projection_robustness_posthoc_summary.md"
    plots_dir = run_dir / "plots"
    _write_csv(per_ps, out_csv)
    out_json.write_text(json.dumps(payload, indent=2, default=str),
                         encoding="utf-8")
    write_summary_md(payload, out_md)
    try:
        write_plots(payload, plots_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plot generation failed: {e!r}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 5C2 post-hoc projection-robustness analysis.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}")
        return 2
    payload = run(run_dir, n_boot=args.n_boot, seed=args.seed)
    rs = payload["robustness_summary"]
    print(f"Stage 5C2 post-hoc -> {run_dir}")
    print(f"  cells with normalized_HCE > 0.5: "
          f"{rs['n_cells_normalized_hce_gt_0_5']}/{rs['n_cells']} "
          f"({(rs['fraction_cells_normalized_hce_gt_0_5'] or 0)*100:.0f}%)")
    print(f"  cells with CI lower bound > 0.5: "
          f"{rs['n_cells_ci_lower_bound_gt_0_5']}/{rs['n_cells']} "
          f"({(rs['fraction_cells_ci_lower_bound_gt_0_5'] or 0)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
