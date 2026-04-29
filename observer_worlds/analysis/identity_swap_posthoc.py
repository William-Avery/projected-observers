"""Stage 5D post-hoc analysis for identity-swap production runs.

Operates on the CSV artifacts of a Topic-2 run — no simulation. Adds
the cross-source decomposition the production spec asks for:

* per-source pair / valid-swap / invalid-reason counts
* per-source mean ``hidden_identity_pull`` with grouped-bootstrap CIs
  (groups = ``(rule_id, seed_host, seed_donor)``)
* per-horizon-bucket (short / medium / long) mean pull
* per visible-similarity quartile mean pull
* M7 vs M4C and M7 vs M4A grouped-bootstrap diffs on pull

Usage::

    python -m observer_worlds.analysis.identity_swap_posthoc \\
        --run-dir outputs/stage5d_identity_swap_production_<ts>/ \\
        --n-boot 2000

Outputs (under the run dir):

    identity_swap_posthoc.csv
    identity_swap_posthoc.json
    identity_swap_posthoc_summary.md
    plots/identity_pull_by_source_ci.png
    plots/identity_pull_by_horizon_bucket.png
    plots/m7_vs_baselines_identity_pull.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SHORT_HORIZONS = (1, 2, 3, 5)
MEDIUM_HORIZONS = (10, 20)
LONG_HORIZONS = (40, 80)
HORIZON_BUCKETS = {
    "short": SHORT_HORIZONS,
    "medium": MEDIUM_HORIZONS,
    "long": LONG_HORIZONS,
}

SOURCES_DEFAULT: tuple[str, ...] = (
    "M7_HCE_optimized",
    "M4C_observer_optimized",
    "M4A_viability",
)


def _truthy(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.lower() == "true"
    return bool(v)


def _safe_mean(xs):
    xs = [float(x) for x in xs if x not in (None, "", "None")]
    return float(np.mean(xs)) if xs else None


def _safe_std(xs):
    xs = [float(x) for x in xs if x not in (None, "", "None")]
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else None


def _grouped_bootstrap_mean(values, groups, *, n_boot=2000, seed=0):
    if not values: return None
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


def _grouped_bootstrap_diff(a_vals, a_groups, b_vals, b_groups,
                             *, n_boot=2000, seed=0):
    if not a_vals or not b_vals: return None
    a = np.asarray(a_vals, dtype=np.float64); b = np.asarray(b_vals, dtype=np.float64)
    ag = np.asarray(a_groups); bg = np.asarray(b_groups)
    rng = np.random.default_rng(int(seed))
    ua = np.unique(ag); ub = np.unique(bg)
    if ua.size < 2 or ub.size < 2:
        return None
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
        "n_groups_a": int(ua.size), "n_groups_b": int(ub.size),
    }


def _cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0: return 0.0
    return float(((a[:, None] > b[None, :]).sum()
                   - (a[:, None] < b[None, :]).sum())
                  / (a.size * b.size))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_identity_scores(run_dir: Path) -> list[dict]:
    p = run_dir / "identity_scores.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    return rows


def _load_candidate_pairs(run_dir: Path) -> list[dict]:
    p = run_dir / "candidate_pairs.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def _valid_score_rows(score_rows: list[dict]) -> list[dict]:
    return [r for r in score_rows
            if _truthy(r.get("valid_swap"))
            and r.get("hidden_identity_pull") not in (None, "", "None")]


# ---------------------------------------------------------------------------
# Per-source aggregation (pair-level + score-level)
# ---------------------------------------------------------------------------


def per_source_pair_counts(pair_rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in pair_rows:
        by_src[r["rule_source"]].append(r)
    for src, rs in by_src.items():
        out[src] = {
            "n_pairs": len(rs),
            "n_valid_a": sum(1 for r in rs if _truthy(r.get("valid_swap_a"))),
            "n_valid_b": sum(1 for r in rs if _truthy(r.get("valid_swap_b"))),
            "n_invalid_a": sum(1 for r in rs
                                if not _truthy(r.get("valid_swap_a"))),
            "n_invalid_b": sum(1 for r in rs
                                if not _truthy(r.get("valid_swap_b"))),
            "mean_visible_similarity":
                _safe_mean(r.get("visible_similarity") for r in rs),
            "mean_hidden_distance":
                _safe_mean(r.get("hidden_distance") for r in rs),
            "mean_projection_preservation_error_a":
                _safe_mean(r.get("projection_preservation_error_a")
                           for r in rs),
            "mean_projection_preservation_error_b":
                _safe_mean(r.get("projection_preservation_error_b")
                           for r in rs),
        }
    return out


def per_source_pull_with_ci(score_rows: list[dict],
                              *, n_boot: int = 2000) -> dict:
    """Per-source mean hidden_identity_pull with grouped-bootstrap CI.

    Group key: ``(rule_id, seed_host, seed_donor)`` so pairs from the
    same rule × seed-pair count as one cluster.
    """
    valid = _valid_score_rows(score_rows)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_src[r["rule_source"]].append(r)
    out: dict[str, dict] = {}
    seed_offset = 0
    for src in sorted(by_src):
        rs = by_src[src]
        pulls = [float(r["hidden_identity_pull"]) for r in rs]
        host = [float(r["host_similarity"]) for r in rs]
        donor = [float(r["donor_similarity"]) for r in rs]
        groups = [
            f"{r['rule_id']}|{r.get('direction', '')}|{r.get('horizon', '')}"
            for r in rs
        ]
        # Bootstrap on per-rule_id grouping to be conservative.
        rule_groups = [r["rule_id"] for r in rs]
        ci_pull = _grouped_bootstrap_mean(
            pulls, rule_groups, n_boot=n_boot, seed=seed_offset,
        )
        seed_offset += 1
        out[src] = {
            "n": len(rs),
            "mean_pull": _safe_mean(pulls),
            "std_pull": _safe_std(pulls),
            "pull_ci": (None, None) if ci_pull is None else ci_pull[1:],
            "fraction_pull_positive":
                float(np.mean([p > 0 for p in pulls])) if pulls else None,
            "mean_host_similarity": _safe_mean(host),
            "mean_donor_similarity": _safe_mean(donor),
        }
    return out


# ---------------------------------------------------------------------------
# Per-horizon-bucket pull
# ---------------------------------------------------------------------------


def per_horizon_bucket_pull(score_rows: list[dict],
                              *, n_boot: int = 2000) -> dict:
    valid = _valid_score_rows(score_rows)
    out: dict[str, dict] = {}
    for bucket, hs in HORIZON_BUCKETS.items():
        bucket_rows = [r for r in valid if int(r["horizon"]) in hs]
        if not bucket_rows:
            out[bucket] = {"n": 0}
            continue
        pulls = [float(r["hidden_identity_pull"]) for r in bucket_rows]
        rule_groups = [r["rule_id"] for r in bucket_rows]
        ci = _grouped_bootstrap_mean(pulls, rule_groups,
                                       n_boot=n_boot, seed=hash(bucket) & 0xFFFF)
        # Per-source breakdown within the bucket.
        per_src: dict[str, dict] = {}
        for src in sorted({r["rule_source"] for r in bucket_rows}):
            rs = [r for r in bucket_rows if r["rule_source"] == src]
            ps = [float(r["hidden_identity_pull"]) for r in rs]
            rgs = [r["rule_id"] for r in rs]
            sci = _grouped_bootstrap_mean(ps, rgs, n_boot=n_boot,
                                            seed=(hash(bucket + src) & 0xFFFF))
            per_src[src] = {
                "n": len(rs),
                "mean_pull": _safe_mean(ps),
                "pull_ci": (None, None) if sci is None else sci[1:],
            }
        out[bucket] = {
            "horizons": list(hs),
            "n": len(bucket_rows),
            "mean_pull": _safe_mean(pulls),
            "pull_ci": (None, None) if ci is None else ci[1:],
            "fraction_pull_positive":
                float(np.mean([p > 0 for p in pulls])) if pulls else None,
            "per_source": per_src,
        }
    return out


# ---------------------------------------------------------------------------
# Pull by visible-similarity quartile
# ---------------------------------------------------------------------------


def pull_by_visible_similarity_quartile(
    score_rows: list[dict], pair_rows: list[dict],
) -> dict:
    """For each quartile of visible_similarity (computed across all
    valid pairs), compute the mean hidden_identity_pull. Tests whether
    higher-quality matches show stronger/weaker pull."""
    # visible_similarity is on candidate_pairs.csv per pair_id.
    sim_by_pair = {r["pair_id"]: float(r["visible_similarity"])
                    for r in pair_rows
                    if r.get("visible_similarity")
                       not in (None, "", "None")}
    valid = _valid_score_rows(score_rows)
    rows = []
    for r in valid:
        sim = sim_by_pair.get(r["pair_id"])
        if sim is None:
            continue
        rows.append((sim, float(r["hidden_identity_pull"])))
    if not rows:
        return {}
    sims = np.array([r[0] for r in rows])
    pulls = np.array([r[1] for r in rows])
    qs = np.quantile(sims, [0.25, 0.5, 0.75])
    out: dict[str, dict] = {}
    for q_lo, q_hi, label in [
        (-np.inf, qs[0], "Q1"),
        (qs[0], qs[1], "Q2"),
        (qs[1], qs[2], "Q3"),
        (qs[2], np.inf, "Q4"),
    ]:
        sel = (sims > q_lo) & (sims <= q_hi)
        out[label] = {
            "visible_similarity_range": [float(q_lo), float(q_hi)],
            "n": int(sel.sum()),
            "mean_pull": float(pulls[sel].mean()) if sel.any() else None,
            "fraction_positive":
                float((pulls[sel] > 0).mean()) if sel.any() else None,
        }
    return out


# ---------------------------------------------------------------------------
# M7 vs baseline by pull
# ---------------------------------------------------------------------------


def m7_vs_baselines_pull(
    score_rows: list[dict], *,
    baselines=("M4C_observer_optimized", "M4A_viability"),
    n_boot: int = 2000,
) -> dict:
    valid = _valid_score_rows(score_rows)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_src[r["rule_source"]].append(r)
    m7 = by_src.get("M7_HCE_optimized", [])
    if not m7:
        return {b.split("_")[0]: {"_status": "no M7 valid rows"}
                 for b in baselines}
    out: dict[str, dict] = {}
    for b in baselines:
        base = by_src.get(b, [])
        if not base:
            out[b.split("_")[0]] = {"_status": f"no {b} valid rows"}
            continue
        a_vals = [float(r["hidden_identity_pull"]) for r in m7]
        b_vals = [float(r["hidden_identity_pull"]) for r in base]
        a_groups = [r["rule_id"] for r in m7]
        b_groups = [r["rule_id"] for r in base]
        cmp = _grouped_bootstrap_diff(
            a_vals, a_groups, b_vals, b_groups,
            n_boot=n_boot, seed=hash(b) & 0xFFFF,
        )
        if cmp:
            cmp["cliffs_delta"] = _cliffs_delta(a_vals, b_vals)
        out[b.split("_")[0]] = cmp or {"_status": "bootstrap unavailable"}
    return out


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
    src_colors = {"M7_HCE_optimized": "#3a7",
                  "M4C_observer_optimized": "#357",
                  "M4A_viability": "#a73"}

    # Plot 1: pull by source with CI.
    try:
        per_src = payload["per_source_pull"]
        sources = list(per_src.keys())
        means = []; lows = []; highs = []
        for s in sources:
            d = per_src[s]
            m = d.get("mean_pull"); ci = d.get("pull_ci") or (None, None)
            means.append(0.0 if m is None else float(m))
            lows.append(0.0 if (ci[0] is None or m is None)
                         else float(m - ci[0]))
            highs.append(0.0 if (ci[1] is None or m is None)
                          else float(ci[1] - m))
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sources))
        ax.bar(x, means, yerr=[lows, highs], capsize=6,
                color=[src_colors.get(s, "#666") for s in sources])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("_")[0] for s in sources])
        ax.set_ylabel("mean hidden_identity_pull")
        ax.set_title("hidden_identity_pull by source (95% bootstrap CI)")
        fig.tight_layout()
        fig.savefig(plots_dir / "identity_pull_by_source_ci.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] identity_pull_by_source_ci.png: {e!r}")

    # Plot 2: pull by horizon bucket.
    try:
        buckets = payload["per_horizon_bucket"]
        order = ["short", "medium", "long"]
        means = []; lows = []; highs = []; labels = []
        for b in order:
            d = buckets.get(b, {})
            if d.get("n", 0) == 0: continue
            m = d.get("mean_pull"); ci = d.get("pull_ci") or (None, None)
            means.append(float(m) if m is not None else 0.0)
            lows.append(0.0 if ci[0] is None or m is None else float(m - ci[0]))
            highs.append(0.0 if ci[1] is None or m is None else float(ci[1] - m))
            labels.append(f"{b}\n(h={d['horizons']})")
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[lows, highs], capsize=6,
                color="#357")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("mean hidden_identity_pull")
        ax.set_title("hidden_identity_pull by horizon bucket")
        fig.tight_layout()
        fig.savefig(plots_dir / "identity_pull_by_horizon_bucket.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] identity_pull_by_horizon_bucket.png: {e!r}")

    # Plot 3: M7 vs baselines.
    try:
        cmps = payload["m7_vs_baselines_pull"]
        rows = []
        for b, cmp in cmps.items():
            if not isinstance(cmp, dict) or cmp.get("ci_low") is None:
                continue
            rows.append((b, cmp))
        if rows:
            labels = [f"M7 vs {b}" for b, _ in rows]
            diffs = [c["mean_diff"] for _, c in rows]
            lows = [c["mean_diff"] - c["ci_low"] for _, c in rows]
            highs = [c["ci_high"] - c["mean_diff"] for _, c in rows]
            colors = ["#3a7" if c["ci_excludes_zero"] and c["mean_diff"] > 0
                       else "#a37" if c["ci_excludes_zero"] else "#999"
                       for _, c in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(labels))
            ax.bar(x, diffs, yerr=[lows, highs], capsize=6, color=colors)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel("M7 − baseline mean hidden_identity_pull")
            ax.set_title("M7 vs baseline identity-pull (grouped-bootstrap CI)")
            fig.tight_layout()
            fig.savefig(plots_dir / "m7_vs_baselines_identity_pull.png",
                         dpi=120)
            plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] m7_vs_baselines_identity_pull.png: {e!r}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(payload: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage 5D — identity-swap production post-hoc")
    lines.append("")
    lines.append(
        "Computed from the existing Stage 5D run; no simulation rerun. "
        "Adds per-source / per-horizon-bucket / M7-vs-baselines "
        "decompositions of ``hidden_identity_pull`` with grouped-"
        "bootstrap CIs."
    )
    lines.append("")
    lines.append(
        "**This is a functional causal-identity test.** It is not a "
        "personal-identity or consciousness claim."
    )
    lines.append("")

    # Pair counts.
    psrc = payload["per_source_pair_counts"]
    if psrc:
        lines.append("## Pair counts and validity by source")
        lines.append("")
        lines.append(_md_row([
            "source", "n_pairs", "valid_a", "invalid_a",
            "valid_b", "invalid_b",
            "mean visible sim", "mean projection preservation Δ",
        ]))
        lines.append(_md_row(["---"] + ["---:"] * 7))
        for src, d in psrc.items():
            lines.append(_md_row([
                src, d["n_pairs"], d["n_valid_a"], d["n_invalid_a"],
                d["n_valid_b"], d["n_invalid_b"],
                f"{d.get('mean_visible_similarity'):+.3f}"
                    if d.get("mean_visible_similarity") is not None else "—",
                f"{d.get('mean_projection_preservation_error_a'):.4g}"
                    if d.get("mean_projection_preservation_error_a") is not None
                    else "—",
            ]))
        lines.append("")

    # Per-source pull.
    pull = payload["per_source_pull"]
    if pull:
        lines.append("## hidden_identity_pull by source")
        lines.append("")
        lines.append(_md_row([
            "source", "n", "mean pull", "95% CI",
            "frac pull > 0", "host_sim", "donor_sim",
        ]))
        lines.append(_md_row(["---"] + ["---:"] * 6))
        for src, d in pull.items():
            ci = d.get("pull_ci") or (None, None)
            ci_str = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                      if ci[0] is not None else "—")
            lines.append(_md_row([
                src, d["n"],
                f"{d['mean_pull']:+.4f}" if d.get("mean_pull") is not None
                    else "—",
                ci_str,
                f"{d.get('fraction_pull_positive'):.2f}"
                    if d.get("fraction_pull_positive") is not None else "—",
                f"{d.get('mean_host_similarity'):+.3f}"
                    if d.get("mean_host_similarity") is not None else "—",
                f"{d.get('mean_donor_similarity'):+.3f}"
                    if d.get("mean_donor_similarity") is not None else "—",
            ]))
        lines.append("")

    # Per-horizon-bucket.
    hb = payload["per_horizon_bucket"]
    if hb:
        lines.append("## hidden_identity_pull by horizon bucket")
        lines.append("")
        lines.append(_md_row([
            "bucket", "horizons", "n", "mean pull", "95% CI",
            "frac > 0",
        ]))
        lines.append(_md_row(["---"] + ["---:"] * 5))
        for bucket in ("short", "medium", "long"):
            d = hb.get(bucket, {})
            if d.get("n", 0) == 0:
                lines.append(_md_row([bucket, "—", "0", "—", "—", "—"]))
                continue
            ci = d.get("pull_ci") or (None, None)
            ci_str = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                      if ci[0] is not None else "—")
            lines.append(_md_row([
                bucket, str(d["horizons"]), d["n"],
                f"{d.get('mean_pull'):+.4f}"
                    if d.get("mean_pull") is not None else "—",
                ci_str,
                f"{d.get('fraction_pull_positive'):.2f}"
                    if d.get("fraction_pull_positive") is not None else "—",
            ]))
        lines.append("")

    # Per visible-sim quartile.
    qs = payload.get("pull_by_visible_quartile", {})
    if qs:
        lines.append("## hidden_identity_pull by visible-similarity quartile")
        lines.append("")
        lines.append(_md_row(["quartile", "vis sim range", "n",
                               "mean pull", "frac > 0"]))
        lines.append(_md_row(["---"] + ["---:"] * 4))
        for label in ("Q1", "Q2", "Q3", "Q4"):
            d = qs.get(label, {})
            if d.get("n", 0) == 0: continue
            r = d["visible_similarity_range"]
            lines.append(_md_row([
                label,
                f"({r[0]:.3f}, {r[1]:.3f}]",
                d["n"],
                f"{d.get('mean_pull'):+.4f}"
                    if d.get("mean_pull") is not None else "—",
                f"{d.get('fraction_positive'):.2f}"
                    if d.get("fraction_positive") is not None else "—",
            ]))
        lines.append("")

    # M7 vs baselines.
    cmps = payload["m7_vs_baselines_pull"]
    lines.append("## M7 vs baseline mean pull (grouped bootstrap)")
    lines.append("")
    lines.append(_md_row([
        "comparison", "mean_M7", "mean_base", "diff", "95% CI",
        "Cliff's δ", "CI excl 0",
    ]))
    lines.append(_md_row(["---"] * 7))
    for b, cmp in cmps.items():
        if not isinstance(cmp, dict) or cmp.get("ci_low") is None:
            lines.append(_md_row([f"M7 vs {b}", "—", "—", "—", "—", "—", "—"]))
            continue
        ci = f"[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]"
        lines.append(_md_row([
            f"M7 vs {b}",
            f"{cmp['mean_a']:+.4f}",
            f"{cmp['mean_b']:+.4f}",
            f"{cmp['mean_diff']:+.4f}",
            ci,
            f"{cmp.get('cliffs_delta', 0):+.3f}",
            "yes" if cmp.get("ci_excludes_zero") else "no",
        ]))
    lines.append("")

    # Activated interpretation.
    lines.append("## Activated interpretation")
    lines.append("")
    overall_pulls: list[float] = []
    for d in pull.values():
        if d.get("mean_pull") is not None:
            overall_pulls.append(float(d["mean_pull"]))
    if overall_pulls:
        m = float(np.mean(overall_pulls))
        if abs(m) < 0.01:
            lines.append(
                f"* Overall mean pull across sources is {m:+.4f} "
                "(near zero). **No clear hidden identity transfer "
                "under this matching/projection setup.**"
            )
        elif m > 0:
            lines.append(
                f"* Overall mean pull across sources is {m:+.4f} > 0. "
                "**Hidden substrate identity partially carries future "
                "trajectory.**"
            )
        else:
            lines.append(
                f"* Overall mean pull across sources is {m:+.4f} < 0. "
                "**Visible host structure dominates future trajectory "
                "after hidden swap.**"
            )
    # Sign change over horizon?
    if hb:
        means = [hb[b].get("mean_pull") for b in ("short", "medium", "long")
                 if b in hb and hb[b].get("mean_pull") is not None]
        if len(means) == 3 and means[0] * means[2] < 0:
            lines.append(
                "* Pull changes sign across horizons "
                f"(short: {means[0]:+.4f}, medium: {means[1]:+.4f}, "
                f"long: {means[2]:+.4f}). **Visible structure "
                "dominates early, but hidden substrate influence "
                "emerges later** (or vice versa)."
            )
    # M7 advantage.
    for b, cmp in cmps.items():
        if not isinstance(cmp, dict) or cmp.get("ci_low") is None: continue
        if cmp["ci_excludes_zero"] and cmp["mean_diff"] > 0:
            lines.append(
                f"* M7 vs {b} pull diff = {cmp['mean_diff']:+.4f} "
                f"(CI {cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}; "
                "CI-clean). **M7 candidates may be more hidden-substrate "
                "dependent in identity-swap terms.**"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run(run_dir: Path, *, n_boot: int = 2000) -> dict:
    score_rows = _load_identity_scores(run_dir)
    pair_rows = _load_candidate_pairs(run_dir)
    payload = {
        "stage": "5D",
        "run_dir": str(run_dir).replace("\\", "/"),
        "n_score_rows": len(score_rows),
        "n_pair_rows": len(pair_rows),
        "n_boot": int(n_boot),
        "per_source_pair_counts": per_source_pair_counts(pair_rows),
        "per_source_pull": per_source_pull_with_ci(score_rows, n_boot=n_boot),
        "per_horizon_bucket": per_horizon_bucket_pull(score_rows, n_boot=n_boot),
        "pull_by_visible_quartile":
            pull_by_visible_similarity_quartile(score_rows, pair_rows),
        "m7_vs_baselines_pull":
            m7_vs_baselines_pull(score_rows, n_boot=n_boot),
    }
    out_csv = run_dir / "identity_swap_posthoc.csv"
    out_json = run_dir / "identity_swap_posthoc.json"
    out_md = run_dir / "identity_swap_posthoc_summary.md"
    plots_dir = run_dir / "plots"

    # Flat CSV: per-source rows.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "source", "n_pairs", "n_valid_a", "n_valid_b",
            "n_score_rows", "mean_pull", "ci_low", "ci_high",
            "fraction_pull_positive",
            "mean_host_similarity", "mean_donor_similarity",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        psrc = payload["per_source_pair_counts"]
        for src, d in payload["per_source_pull"].items():
            ci = d.get("pull_ci") or (None, None)
            row = {
                "source": src,
                "n_pairs": (psrc.get(src) or {}).get("n_pairs", 0),
                "n_valid_a": (psrc.get(src) or {}).get("n_valid_a", 0),
                "n_valid_b": (psrc.get(src) or {}).get("n_valid_b", 0),
                "n_score_rows": d.get("n", 0),
                "mean_pull": d.get("mean_pull"),
                "ci_low": ci[0], "ci_high": ci[1],
                "fraction_pull_positive": d.get("fraction_pull_positive"),
                "mean_host_similarity": d.get("mean_host_similarity"),
                "mean_donor_similarity": d.get("mean_donor_similarity"),
            }
            w.writerow(row)
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
        description="Stage 5D post-hoc identity-swap analysis.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}")
        return 2
    payload = run(run_dir, n_boot=args.n_boot)
    print(f"Stage 5D post-hoc -> {run_dir}")
    for src, d in payload["per_source_pull"].items():
        ci = d.get("pull_ci") or (None, None)
        ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci[0] is not None else "—"
        print(f"  {src:30s} n={d['n']:5d} "
              f"mean pull = {d.get('mean_pull'):+.4f} CI={ci_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
