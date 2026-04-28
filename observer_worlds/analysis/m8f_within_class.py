"""M8F — within-class HCE bootstrap analysis.

Operates on the CSV artifacts of an M8E run (or any M8-style run that
produces ``mechanism_labels.csv`` and ``lifetime_tradeoff.csv``).
Does NOT re-run any simulation.

For each focus mechanism class and each baseline source (M4C, M4A),
compares M7's per-candidate metrics against the baseline using:

    - grouped bootstrap by (rule_id, seed) on the mean difference
    - permutation test on the mean difference (within-class)
    - Cliff's delta and win rate

Outputs (under ``<run_dir>``):

    m8f_within_class_stats.json
    m8f_within_class_summary.csv
    m8f_within_class_summary.md
    plots/m8f_within_class_hce_ci.png
    plots/m8f_within_class_effect_sizes.png
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOCUS_CLASSES = ("boundary_mediated", "global_chaotic")
DESCRIPTIVE_CLASSES = (
    "interior_reservoir",
    "threshold_mediated",
    "delayed_hidden_channel",
    "unclear",
    "environment_coupled",
)

SOURCES = ("M7_HCE_optimized", "M4C_observer_optimized", "M4A_viability")
SOURCE_SHORT = {
    "M7_HCE_optimized": "M7",
    "M4C_observer_optimized": "M4C",
    "M4A_viability": "M4A",
}

METRICS = (
    "HCE",
    "candidate_locality_index",
    "far_hidden_effect",
    "first_visible_effect_time",
    "candidate_lifetime",
    "near_threshold_fraction",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_candidates(run_dir: Path) -> list[dict]:
    """Load mechanism_labels.csv and join HCE from lifetime_tradeoff.csv.

    Returns one dict per candidate with all metrics needed by M8F.
    """
    labels_path = run_dir / "mechanism_labels.csv"
    trade_path = run_dir / "lifetime_tradeoff.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"missing {labels_path}")
    if not trade_path.exists():
        raise FileNotFoundError(f"missing {trade_path}")

    def _key(r): return (r["rule_id"], r["seed"], r["candidate_id"])

    with labels_path.open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    with trade_path.open(encoding="utf-8") as f:
        trade = list(csv.DictReader(f))
    hce_by_key = {_key(r): float(r["HCE_at_headline_horizon"]) for r in trade}

    out = []
    for r in labels:
        hce = hce_by_key.get(_key(r))
        if hce is None:
            continue
        out.append({
            "rule_id": r["rule_id"],
            "rule_source": r["rule_source"],
            "seed": r["seed"],
            "candidate_id": r["candidate_id"],
            "mechanism": r["mechanism"],
            "group": f"{r['rule_id']}|{r['seed']}",
            "HCE": hce,
            "candidate_locality_index": float(r["candidate_locality_index"]),
            "far_hidden_effect": float(r["far_hidden_effect"]),
            "first_visible_effect_time": float(r["first_visible_effect_time"]),
            "candidate_lifetime": float(r["candidate_lifetime"]),
            "near_threshold_fraction": float(r["near_threshold_fraction"]),
            "boundary_response_fraction": float(r["boundary_response_fraction"]),
            "interior_response_fraction": float(r["interior_response_fraction"]),
            "boundary_mediation_index": float(r["boundary_mediation_index"]),
        })
    return out


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def grouped_bootstrap_diff(
    a_vals: np.ndarray, a_groups: np.ndarray,
    b_vals: np.ndarray, b_groups: np.ndarray,
    *, n_boot: int = 5000, seed: int = 0,
) -> dict:
    """Cluster-bootstrap of mean(a) - mean(b), resampling whole groups.

    Each group is treated as the unit of resampling; within each
    bootstrap iter, all candidates from each sampled group are taken.
    """
    if a_vals.size == 0 or b_vals.size == 0:
        return {"mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "n_boot": int(n_boot)}
    rng = np.random.default_rng(seed)
    ua = np.unique(a_groups); ub = np.unique(b_groups)
    idx_a = {g: np.where(a_groups == g)[0] for g in ua}
    idx_b = {g: np.where(b_groups == g)[0] for g in ub}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(ua, size=ua.size, replace=True)
        sb = rng.choice(ub, size=ub.size, replace=True)
        ia = np.concatenate([idx_a[g] for g in sa])
        ib = np.concatenate([idx_b[g] for g in sb])
        diffs[i] = float(a_vals[ia].mean() - b_vals[ib].mean())
    return {
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "mean_diff": float(a_vals.mean() - b_vals.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_boot": int(n_boot),
    }


def permutation_test_mean_diff(
    a: np.ndarray, b: np.ndarray,
    *, n_permutations: int = 5000, seed: int = 0,
) -> float:
    """Two-sided permutation test on |mean(a) - mean(b)|.

    Permutes candidate-level source labels within the pooled within-class
    population. Group structure is broken by design: this complements
    the grouped-bootstrap CI rather than replacing it.
    """
    if a.size == 0 or b.size == 0:
        return 1.0
    observed = abs(float(a.mean()) - float(b.mean()))
    pooled = np.concatenate([a, b]).copy()
    rng = np.random.default_rng(seed)
    n_a = a.size
    n_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= observed - 1e-12:
            n_extreme += 1
    return float((n_extreme + 1) / (n_permutations + 1))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b). Range [-1, +1]."""
    if a.size == 0 or b.size == 0:
        return 0.0
    A = a[:, None]; B = b[None, :]
    return float(((A > B).sum() - (A < B).sum()) / (a.size * b.size))


def win_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of (a_i, b_j) pairs where a_i > b_j (strict)."""
    if a.size == 0 or b.size == 0:
        return 0.5
    return float((a[:, None] > b[None, :]).mean())


# ---------------------------------------------------------------------------
# Cross-source comparison driver
# ---------------------------------------------------------------------------


def compare_within_class(
    cands: list[dict], *,
    source_a: str, source_b: str, mechanism: str, metric: str,
    n_boot: int = 5000, n_perm: int = 5000, seed: int = 0,
) -> dict:
    """Build the within-class M7-vs-baseline comparison record."""
    a_rows = [c for c in cands
              if c["rule_source"] == source_a and c["mechanism"] == mechanism]
    b_rows = [c for c in cands
              if c["rule_source"] == source_b and c["mechanism"] == mechanism]
    a_vals = np.array([r[metric] for r in a_rows], dtype=np.float64)
    b_vals = np.array([r[metric] for r in b_rows], dtype=np.float64)
    a_groups = np.array([r["group"] for r in a_rows])
    b_groups = np.array([r["group"] for r in b_rows])

    boot = grouped_bootstrap_diff(
        a_vals, a_groups, b_vals, b_groups,
        n_boot=n_boot, seed=seed,
    )
    p = permutation_test_mean_diff(
        a_vals, b_vals, n_permutations=n_perm, seed=seed + 1,
    )
    delta = cliffs_delta(a_vals, b_vals)
    wr = win_rate(a_vals, b_vals)
    return {
        "comparison": f"{SOURCE_SHORT[source_a]}_vs_{SOURCE_SHORT[source_b]}",
        "source_a": source_a,
        "source_b": source_b,
        "mechanism": mechanism,
        "metric": metric,
        "n_a": int(a_vals.size),
        "n_b": int(b_vals.size),
        "n_groups_a": int(np.unique(a_groups).size) if a_groups.size else 0,
        "n_groups_b": int(np.unique(b_groups).size) if b_groups.size else 0,
        "mean_a": boot["mean_a"],
        "mean_b": boot["mean_b"],
        "mean_diff": boot["mean_diff"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "ci_excludes_zero": (boot["ci_low"] > 0.0 or boot["ci_high"] < 0.0),
        "perm_p": float(p),
        "cliffs_delta": float(delta),
        "win_rate_a": float(wr),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Per-class descriptive (no comparison; tiny N safe-out)
# ---------------------------------------------------------------------------


def class_descriptives(cands: list[dict], mechanism: str) -> dict:
    out = {}
    for src in SOURCES:
        rs = [c for c in cands
              if c["rule_source"] == src and c["mechanism"] == mechanism]
        rec: dict = {"n": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs]
            rec[f"mean_{m}"] = float(np.mean(vals)) if vals else None
        # Bootstrap CI for mean HCE within class (per source)
        if rs:
            vals = np.array([r["HCE"] for r in rs])
            grps = np.array([r["group"] for r in rs])
            mean_pt, lo, hi = _grouped_bootstrap_mean(vals, grps, n_boot=5000, seed=42)
            rec["mean_HCE_ci"] = [lo, hi]
            rec["mean_HCE_ci_width"] = float(hi - lo)
        out[src] = rec
    return out


def _grouped_bootstrap_mean(values, groups, *, n_boot=5000, seed=0):
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(unique, size=unique.size, replace=True)
        idxs = np.concatenate([idx_by_group[g] for g in s])
        boot[b] = float(values[idxs].mean())
    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


# ---------------------------------------------------------------------------
# HCE-vs-lifetime per source (descriptive)
# ---------------------------------------------------------------------------


def hce_lifetime_corr(cands: list[dict]) -> dict:
    out = {}
    for src in SOURCES:
        rs = [c for c in cands if c["rule_source"] == src]
        if len(rs) < 3:
            out[src] = {"n": len(rs), "pearson_HCE_vs_lifetime": None}
            continue
        h = np.array([r["HCE"] for r in rs])
        l = np.array([r["candidate_lifetime"] for r in rs])
        if h.std() < 1e-12 or l.std() < 1e-12:
            out[src] = {"n": len(rs), "pearson_HCE_vs_lifetime": None}
            continue
        out[src] = {
            "n": len(rs),
            "pearson_HCE_vs_lifetime": float(np.corrcoef(h, l)[0, 1]),
            "mean_HCE": float(h.mean()),
            "mean_lifetime": float(l.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def write_summary_md(stats: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# M8F — within-class HCE bootstrap analysis")
    lines.append("")
    lines.append(f"Run dir: `{stats['run_dir']}`")
    lines.append(f"Generated: {stats['generated_at_utc']}")
    lines.append(f"Bootstrap: {stats['n_boot']} iters, grouped by `(rule_id, seed)`. "
                 f"Permutation: {stats['n_perm']} iters, candidate-level.")
    lines.append("")

    lines.append("## n by source × class")
    lines.append("")
    headers = ["source"] + list(FOCUS_CLASSES + DESCRIPTIVE_CLASSES)
    lines.append(_md_row(headers))
    lines.append(_md_row(["---"] + ["---:"] * (len(headers) - 1)))
    for src in SOURCES:
        cells = [SOURCE_SHORT[src]]
        for cls in FOCUS_CLASSES + DESCRIPTIVE_CLASSES:
            cells.append(str(stats["descriptives"][cls][src]["n"]))
        lines.append(_md_row(cells))
    lines.append("")

    for cls in FOCUS_CLASSES:
        lines.append(f"## Focus class: `{cls}`")
        lines.append("")
        lines.append("### Per-source mean HCE with grouped-bootstrap CI")
        lines.append("")
        lines.append(_md_row(["source", "n", "mean HCE", "95% CI"]))
        lines.append(_md_row(["---", "---:", "---:", "---:"]))
        for src in SOURCES:
            d = stats["descriptives"][cls][src]
            n = d["n"]
            if n == 0:
                lines.append(_md_row([SOURCE_SHORT[src], "0", "—", "—"]))
            else:
                ci = d["mean_HCE_ci"]
                lines.append(_md_row([
                    SOURCE_SHORT[src], str(n),
                    f"{d['mean_HCE']:+.4f}",
                    f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                ]))
        lines.append("")

        lines.append("### M7 vs baseline within `" + cls + "`")
        lines.append("")
        lines.append(_md_row([
            "comparison", "metric", "mean_M7", "mean_base",
            "diff", "95% CI", "CI excludes 0", "perm p",
            "Cliff's δ", "win % M7",
        ]))
        lines.append(_md_row(["---"] * 10))
        for cmp_rec in stats["comparisons"][cls]:
            ci = f"[{cmp_rec['ci_low']:+.4f}, {cmp_rec['ci_high']:+.4f}]"
            lines.append(_md_row([
                cmp_rec["comparison"], cmp_rec["metric"],
                f"{cmp_rec['mean_a']:+.4f}",
                f"{cmp_rec['mean_b']:+.4f}",
                f"{cmp_rec['mean_diff']:+.4f}",
                ci,
                "yes" if cmp_rec["ci_excludes_zero"] else "no",
                f"{cmp_rec['perm_p']:.4f}",
                f"{cmp_rec['cliffs_delta']:+.3f}",
                f"{100 * cmp_rec['win_rate_a']:.1f}%",
            ]))
        lines.append("")

    lines.append("## Descriptive classes (small N — no within-class hypothesis test)")
    lines.append("")
    for cls in DESCRIPTIVE_CLASSES:
        lines.append(f"### `{cls}`")
        lines.append("")
        lines.append(_md_row(["source", "n", "mean HCE", "mean far_hid", "mean lifetime", "mean near_thr"]))
        lines.append(_md_row(["---", "---:", "---:", "---:", "---:", "---:"]))
        for src in SOURCES:
            d = stats["descriptives"][cls][src]
            if d["n"] == 0:
                lines.append(_md_row([SOURCE_SHORT[src], "0", "—", "—", "—", "—"]))
            else:
                lines.append(_md_row([
                    SOURCE_SHORT[src], str(d["n"]),
                    f"{d['mean_HCE']:+.4f}",
                    f"{d['mean_far_hidden_effect']:+.4f}",
                    f"{d['mean_candidate_lifetime']:.1f}",
                    f"{d['mean_near_threshold_fraction']:.3f}",
                ]))
        lines.append("")

    lines.append("## HCE / lifetime descriptives by source")
    lines.append("")
    lines.append(_md_row(["source", "n", "mean HCE", "mean life", "Pearson(HCE, life)"]))
    lines.append(_md_row(["---", "---:", "---:", "---:", "---:"]))
    for src in SOURCES:
        c = stats["hce_lifetime"][src]
        if c.get("pearson_HCE_vs_lifetime") is None:
            lines.append(_md_row([SOURCE_SHORT[src], str(c["n"]), "—", "—", "—"]))
        else:
            lines.append(_md_row([
                SOURCE_SHORT[src], str(c["n"]),
                f"{c['mean_HCE']:+.4f}",
                f"{c['mean_lifetime']:.1f}",
                f"{c['pearson_HCE_vs_lifetime']:+.3f}",
            ]))
    lines.append("")
    lines.append("Reported as descriptive only. No formal test of the across-source "
                 "correlation difference is run here; the sign-flip should be "
                 "interpreted as a qualitative cross-source observation, not a "
                 "hypothesis-tested claim.")
    lines.append("")

    lines.append("## Caveat — classifier-conditional language")
    lines.append("")
    lines.append("Mechanism labels are produced by the priority-ordered rule-based "
                 "classifier in `_m8_mechanism.classify_mechanism`. "
                 "The `boundary_mediated` label captures *boundary-organized "
                 "response geometry*, not boundary-localized mediation: per the "
                 "M8 classifier audit, BMI ≈ 0.5 inside that class across all "
                 "three rule sources. Read every `boundary_mediated`-class "
                 "comparison below as a within-classifier comparison, not as a "
                 "claim about boundary-vs-interior physical localization.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(stats: dict, out_path: Path) -> None:
    fields = [
        "comparison", "mechanism", "metric",
        "n_a", "n_b", "n_groups_a", "n_groups_b",
        "mean_a", "mean_b", "mean_diff",
        "ci_low", "ci_high", "ci_excludes_zero",
        "perm_p", "cliffs_delta", "win_rate_a",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cls in FOCUS_CLASSES:
            for rec in stats["comparisons"][cls]:
                w.writerow({k: rec[k] for k in fields})


def write_plots(stats: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    # 1. Per-class mean HCE with CI bars per source.
    fig, axes = plt.subplots(1, len(FOCUS_CLASSES), figsize=(10, 4), sharey=False)
    if len(FOCUS_CLASSES) == 1:
        axes = [axes]
    for ax, cls in zip(axes, FOCUS_CLASSES):
        means, lows, highs, labels, ns = [], [], [], [], []
        for src in SOURCES:
            d = stats["descriptives"][cls][src]
            if d["n"] == 0:
                continue
            ci = d["mean_HCE_ci"]
            means.append(d["mean_HCE"])
            lows.append(d["mean_HCE"] - ci[0])
            highs.append(ci[1] - d["mean_HCE"])
            labels.append(f"{SOURCE_SHORT[src]}\n(n={d['n']})")
            ns.append(d["n"])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[lows, highs], capsize=6,
               color=["#3a7", "#357", "#a73"][:len(labels)])
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(f"mean HCE within `{cls}`")
        ax.set_ylabel("mean HCE (95% CI, grouped boot)")
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("M8F — within-class HCE per source with grouped-bootstrap CI")
    fig.tight_layout()
    fig.savefig(plots_dir / "m8f_within_class_hce_ci.png", dpi=120)
    plt.close(fig)

    # 2. Effect sizes (Cliff's delta) for HCE per comparison × class.
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_data: list[tuple[str, str, float, float, float, str]] = []
    for cls in FOCUS_CLASSES:
        for rec in stats["comparisons"][cls]:
            if rec["metric"] != "HCE":
                continue
            label = f"{rec['comparison']}\n{cls}"
            bar_data.append((label, rec["comparison"],
                             rec["mean_diff"],
                             rec["ci_low"], rec["ci_high"],
                             rec["metric"]))
    if bar_data:
        labels = [b[0] for b in bar_data]
        diffs = [b[2] for b in bar_data]
        lows = [b[2] - b[3] for b in bar_data]
        highs = [b[4] - b[2] for b in bar_data]
        x = np.arange(len(labels))
        ax.bar(x, diffs, yerr=[lows, highs], capsize=6,
               color=["#3a7", "#357"] * (len(labels) // 2 + 1))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("HCE mean diff (M7 - baseline) with 95% CI")
        ax.set_title("M8F — within-class HCE: M7 vs baselines (grouped-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(plots_dir / "m8f_within_class_effect_sizes.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run(run_dir: Path, *, n_boot: int = 5000, n_perm: int = 5000) -> dict:
    cands = load_candidates(run_dir)

    descriptives: dict = {}
    for cls in FOCUS_CLASSES + DESCRIPTIVE_CLASSES:
        descriptives[cls] = class_descriptives(cands, cls)

    comparisons: dict = {}
    for cls in FOCUS_CLASSES:
        comparisons[cls] = []
        for src_b in ("M4C_observer_optimized", "M4A_viability"):
            for metric in METRICS:
                rec = compare_within_class(
                    cands, source_a="M7_HCE_optimized", source_b=src_b,
                    mechanism=cls, metric=metric,
                    n_boot=n_boot, n_perm=n_perm,
                )
                comparisons[cls].append(rec)

    hce_lt = hce_lifetime_corr(cands)

    stats = {
        "run_dir": str(run_dir).replace("\\", "/"),
        "generated_at_utc": (
            _dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "n_candidates_total": len(cands),
        "descriptives": descriptives,
        "comparisons": comparisons,
        "hce_lifetime": hce_lt,
    }

    (run_dir / "m8f_within_class_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8",
    )
    write_summary_csv(stats, run_dir / "m8f_within_class_summary.csv")
    write_summary_md(stats, run_dir / "m8f_within_class_summary.md")
    write_plots(stats, run_dir / "plots")

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M8F within-class HCE bootstrap analysis on existing M8E artifacts."
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="M8E output directory containing mechanism_labels.csv "
                        "and lifetime_tradeoff.csv.")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-perm", type=int, default=5000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run dir does not exist: {run_dir}")
        return 2
    stats = run(run_dir, n_boot=args.n_boot, n_perm=args.n_perm)
    print(f"M8F written to {run_dir}")
    print(f"  candidates joined: {stats['n_candidates_total']}")
    for cls in FOCUS_CLASSES:
        for rec in stats["comparisons"][cls]:
            if rec["metric"] != "HCE":
                continue
            tag = "***" if rec["ci_excludes_zero"] else "   "
            print(f"  {tag} {cls:20s} {rec['comparison']:18s} "
                  f"diff={rec['mean_diff']:+.4f} "
                  f"CI=[{rec['ci_low']:+.4f},{rec['ci_high']:+.4f}] "
                  f"p={rec['perm_p']:.4f} cliffs={rec['cliffs_delta']:+.2f} "
                  f"win={rec['win_rate_a']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
