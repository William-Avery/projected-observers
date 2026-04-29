"""Stage 5E post-hoc analysis for the Topic-3 agent-task production run.

Operates on the CSV artifacts only — no simulation. Adds the cross-
source decomposition the production spec calls for:

* per (source × task) means + grouped-bootstrap CIs on task_score
* per (source × task) Pearson(HCE, task_score) and
  Pearson(observer_score, task_score) with 3-variable OLS
* per (source × task) hidden / visible intervention delta means
* horizon-bucket (short / medium / long) per (source × task)
* high-HCE vs low-HCE candidate split per (source × task)

Usage::

    python -m observer_worlds.analysis.agent_task_posthoc \\
        --run-dir outputs/stage5e_agent_tasks_production_<ts>/ \\
        --n-boot 2000

Outputs (under the run dir):

    agent_task_posthoc.csv
    agent_task_posthoc.json
    agent_task_posthoc_summary.md
    plots/task_score_by_source_task.png
    plots/hce_task_correlation_by_source_task.png
    plots/hidden_vs_visible_delta_by_task.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


def _pearson(xs, ys):
    pairs = [(float(x), float(y))
             for x, y in zip(xs, ys)
             if x not in (None, "", "None") and y not in (None, "", "None")]
    if len(pairs) < 3: return None
    xs2 = np.array([p[0] for p in pairs])
    ys2 = np.array([p[1] for p in pairs])
    if xs2.std() < 1e-12 or ys2.std() < 1e-12: return None
    return float(np.corrcoef(xs2, ys2)[0, 1])


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


def _ols_three_var(target, var1, var2, source_dummies):
    """OLS: target = a + b1*var1 + b2*var2 + sum(c_s * dummy_s).

    ``source_dummies`` is a list of vectors (one per source minus
    reference). Returns dict with coefficients and R²."""
    pairs = []
    for t, v1, v2, *dums in zip(target, var1, var2, *source_dummies):
        if (t in (None, "", "None") or v1 in (None, "", "None")
                or v2 in (None, "", "None")):
            continue
        pairs.append((float(t), float(v1), float(v2),
                      *[float(d) for d in dums]))
    n = len(pairs)
    if n < 5: return None
    Y = np.array([p[0] for p in pairs])
    cols = [np.ones(n),
            [p[1] for p in pairs],
            [p[2] for p in pairs]]
    for s in range(len(source_dummies)):
        cols.append([p[3 + s] for p in pairs])
    X = np.column_stack(cols)
    try:
        coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    pred = X @ coef
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    ss_res = float(((Y - pred) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "n": n,
        "intercept": float(coef[0]),
        "beta_HCE": float(coef[1]),
        "beta_observer": float(coef[2]),
        "beta_source_dummies": [float(c) for c in coef[3:]],
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_trials(run_dir: Path) -> list[dict]:
    p = run_dir / "task_trials.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return list(csv.DictReader(p.open(encoding="utf-8")))


def _by_candidate_task(trials: list[dict]) -> list[dict]:
    """Collapse per-horizon trials to one row per (rule, seed,
    candidate, task) — the unit used by the regression layer."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for t in trials:
        key = (t["rule_id"], t["seed"], t["candidate_id"], t["task_name"])
        by_key[key].append(t)
    out = []
    for (rid, seed, cid, task), ts in by_key.items():
        scores = [float(t["task_score"]) for t in ts
                  if t.get("task_score") not in (None, "", "None")]
        if not scores: continue
        hce = next((float(t["hce"]) for t in ts
                    if t.get("hce") not in (None, "", "None")), None)
        obs = next((float(t["observer_score"]) for t in ts
                    if t.get("observer_score") not in (None, "", "None")),
                   None)
        h_deltas = [float(t["hidden_intervention_task_delta"]) for t in ts
                    if t.get("hidden_intervention_task_delta") not in (None, "", "None")]
        v_deltas = [float(t["visible_intervention_task_delta"]) for t in ts
                    if t.get("visible_intervention_task_delta") not in (None, "", "None")]
        out.append({
            "rule_id": rid, "rule_source": ts[0].get("rule_source"),
            "seed": seed, "candidate_id": cid, "task_name": task,
            "n_horizons": len(scores),
            "mean_task_score": float(np.mean(scores)),
            "max_task_score": float(np.max(scores)),
            "hce": hce,
            "observer_score": obs,
            "mean_hidden_delta": _safe_mean(h_deltas),
            "mean_visible_delta": _safe_mean(v_deltas),
        })
    return out


# ---------------------------------------------------------------------------
# Per-source × task aggregation
# ---------------------------------------------------------------------------


def per_source_task_summary(
    by_cand: list[dict], *, n_boot: int = 2000,
) -> dict:
    by_st: dict[tuple, list[dict]] = defaultdict(list)
    for r in by_cand:
        by_st[(r["rule_source"], r["task_name"])].append(r)
    out: dict[str, dict[str, dict]] = {}
    for (src, task), rs in by_st.items():
        scores = [r["mean_task_score"] for r in rs]
        hce = [r["hce"] for r in rs if r["hce"] is not None]
        obs = [r["observer_score"] for r in rs if r["observer_score"] is not None]
        groups = [r["rule_id"] for r in rs]
        ci = _grouped_bootstrap_mean(scores, groups, n_boot=n_boot,
                                       seed=hash(f"{src}{task}") & 0xFFFF)
        # Correlations on per-candidate aggregates.
        cand_hce = [r["hce"] for r in rs if r["hce"] is not None
                    and r["mean_task_score"] is not None]
        cand_score_h = [r["mean_task_score"] for r in rs
                        if r["hce"] is not None
                        and r["mean_task_score"] is not None]
        cand_obs = [r["observer_score"] for r in rs
                    if r["observer_score"] is not None
                    and r["mean_task_score"] is not None]
        cand_score_o = [r["mean_task_score"] for r in rs
                        if r["observer_score"] is not None
                        and r["mean_task_score"] is not None]
        # Hidden / visible deltas.
        h_deltas = [r["mean_hidden_delta"] for r in rs
                    if r["mean_hidden_delta"] is not None]
        v_deltas = [r["mean_visible_delta"] for r in rs
                    if r["mean_visible_delta"] is not None]
        # High-HCE vs low-HCE split.
        cand_pairs = [(r["hce"], r["mean_task_score"]) for r in rs
                      if r["hce"] is not None
                      and r["mean_task_score"] is not None]
        cand_pairs.sort(key=lambda x: x[0])
        n = len(cand_pairs)
        low_split = cand_pairs[: n // 2]
        high_split = cand_pairs[(n + 1) // 2:]
        out.setdefault(src, {})[task] = {
            "n_candidates": len(rs),
            "mean_task_score": _safe_mean(scores),
            "task_score_ci": (None, None) if ci is None else ci[1:],
            "std_task_score": _safe_std(scores),
            "n_with_hce": len(hce),
            "mean_hce": _safe_mean(hce),
            "mean_observer_score": _safe_mean(obs),
            "pearson_hce_task":
                _pearson(cand_hce, cand_score_h),
            "pearson_observer_task":
                _pearson(cand_obs, cand_score_o),
            "n_with_hidden_delta": len(h_deltas),
            "mean_hidden_delta": _safe_mean(h_deltas),
            "fraction_hidden_perturbation_hurt":
                (sum(1 for d in h_deltas if d > 0) / len(h_deltas))
                if h_deltas else None,
            "n_with_visible_delta": len(v_deltas),
            "mean_visible_delta": _safe_mean(v_deltas),
            "fraction_visible_perturbation_hurt":
                (sum(1 for d in v_deltas if d > 0) / len(v_deltas))
                if v_deltas else None,
            "mean_hidden_minus_visible":
                _safe_mean([h - v for h, v in zip(h_deltas, v_deltas)
                            if h is not None and v is not None]),
            "low_hce_n": len(low_split),
            "high_hce_n": len(high_split),
            "low_hce_mean_task": _safe_mean([p[1] for p in low_split]),
            "high_hce_mean_task": _safe_mean([p[1] for p in high_split]),
            "high_minus_low_task_diff":
                ((_safe_mean([p[1] for p in high_split]) or 0.0)
                 - (_safe_mean([p[1] for p in low_split]) or 0.0)),
        }
    return out


# ---------------------------------------------------------------------------
# Horizon-bucket analysis (uses per-trial rows, NOT collapsed)
# ---------------------------------------------------------------------------


def per_horizon_bucket_by_source_task(
    trials: list[dict], *, n_boot: int = 2000,
) -> dict:
    out: dict[str, dict] = {}
    for bucket, hs in HORIZON_BUCKETS.items():
        bucket_rows = [t for t in trials
                       if t.get("horizon") not in (None, "", "None")
                       and int(t["horizon"]) in hs
                       and t.get("task_score") not in (None, "", "None")]
        if not bucket_rows:
            out[bucket] = {"n": 0}; continue
        per_st: dict[tuple, list[dict]] = defaultdict(list)
        for t in bucket_rows:
            per_st[(t["rule_source"], t["task_name"])].append(t)
        per_st_out: dict[tuple, dict] = {}
        for (src, task), rs in per_st.items():
            scores = [float(r["task_score"]) for r in rs]
            groups = [r["rule_id"] for r in rs]
            ci = _grouped_bootstrap_mean(
                scores, groups, n_boot=n_boot,
                seed=hash(f"{bucket}{src}{task}") & 0xFFFF,
            )
            per_st_out[f"{src}|{task}"] = {
                "n": len(rs),
                "mean_task_score": _safe_mean(scores),
                "task_score_ci":
                    (None, None) if ci is None else ci[1:],
            }
        out[bucket] = {
            "horizons": list(hs),
            "n_total": len(bucket_rows),
            "per_source_task": per_st_out,
        }
    return out


# ---------------------------------------------------------------------------
# Pooled regressions
# ---------------------------------------------------------------------------


def pooled_regressions(
    by_cand: list[dict], *, n_boot: int = 2000,
) -> dict:
    target = [r["mean_task_score"] for r in by_cand]
    hce = [r["hce"] for r in by_cand]
    obs = [r["observer_score"] for r in by_cand]
    sources = sorted({r["rule_source"] for r in by_cand if r.get("rule_source")})
    # Use M7 as reference; dummies for the others.
    other_sources = [s for s in sources if s != "M7_HCE_optimized"]
    dummies = [
        [1 if r["rule_source"] == s else 0 for r in by_cand]
        for s in other_sources
    ]
    return {
        "task_score ~ HCE": {
            "n": sum(1 for r in by_cand if r["hce"] is not None),
            "pearson_r": _pearson(hce, target),
        },
        "task_score ~ observer_score": {
            "n": sum(1 for r in by_cand if r["observer_score"] is not None),
            "pearson_r": _pearson(obs, target),
        },
        "task_score ~ HCE + observer_score": _ols_three_var(
            target, hce, obs, [],
        ),
        "task_score ~ HCE + observer_score + source": {
            "reference_source": "M7_HCE_optimized",
            "other_sources": other_sources,
            **(_ols_three_var(target, hce, obs, dummies) or {
                "_status": "unable to fit"
            }),
        },
    }


# ---------------------------------------------------------------------------
# Hidden vs visible delta comparison
# ---------------------------------------------------------------------------


def hidden_vs_visible_delta_summary(by_cand: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in by_cand:
        by_task[r["task_name"]].append(r)
    out: dict[str, dict] = {}
    for task, rs in by_task.items():
        h = [r["mean_hidden_delta"] for r in rs
             if r["mean_hidden_delta"] is not None]
        v = [r["mean_visible_delta"] for r in rs
             if r["mean_visible_delta"] is not None]
        # Paired diffs per candidate.
        paired = [(r["mean_hidden_delta"], r["mean_visible_delta"]) for r in rs
                  if r["mean_hidden_delta"] is not None
                  and r["mean_visible_delta"] is not None]
        diffs = [h - v for h, v in paired]
        out[task] = {
            "n_paired": len(paired),
            "mean_hidden_delta": _safe_mean(h),
            "mean_visible_delta": _safe_mean(v),
            "mean_paired_diff_h_minus_v": _safe_mean(diffs),
            "fraction_visible_more_harmful_than_hidden":
                (sum(1 for d in diffs if d < 0) / len(diffs))
                if diffs else None,
        }
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

    # 1. task_score by (source × task) with CI.
    try:
        per_st = payload["per_source_task"]
        sources = list(per_st.keys())
        tasks = sorted({t for s in per_st for t in per_st[s]})
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.25
        x = np.arange(len(tasks))
        for i, src in enumerate(sources):
            means = []; lows = []; highs = []
            for task in tasks:
                d = (per_st.get(src) or {}).get(task, {})
                m = d.get("mean_task_score")
                ci = d.get("task_score_ci") or (None, None)
                means.append(0.0 if m is None else float(m))
                lows.append(0.0 if (ci[0] is None or m is None)
                             else float(m - ci[0]))
                highs.append(0.0 if (ci[1] is None or m is None)
                              else float(ci[1] - m))
            ax.bar(x + (i - 1) * width, means, width=width,
                    yerr=[lows, highs], capsize=4,
                    color=src_colors.get(src, "#666"),
                    label=src.split("_")[0])
        ax.set_xticks(x); ax.set_xticklabels(tasks)
        ax.set_ylabel("mean task_score (95% bootstrap CI)")
        ax.set_title("task_score by source × task")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "task_score_by_source_task.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] task_score_by_source_task.png: {e!r}")

    # 2. HCE-task correlations by (source × task).
    try:
        per_st = payload["per_source_task"]
        sources = list(per_st.keys())
        tasks = sorted({t for s in per_st for t in per_st[s]})
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.25
        x = np.arange(len(tasks))
        for i, src in enumerate(sources):
            rs = []
            for task in tasks:
                d = (per_st.get(src) or {}).get(task, {})
                r = d.get("pearson_hce_task")
                rs.append(0.0 if r is None else float(r))
            ax.bar(x + (i - 1) * width, rs, width=width,
                    color=src_colors.get(src, "#666"),
                    label=src.split("_")[0])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(tasks)
        ax.set_ylabel("Pearson(HCE, task_score)")
        ax.set_title("HCE-task correlation by source × task")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "hce_task_correlation_by_source_task.png",
                     dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hce_task_correlation_by_source_task.png: {e!r}")

    # 3. Hidden vs visible delta by task.
    try:
        hv = payload["hidden_vs_visible"]
        tasks = list(hv.keys())
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.35
        x = np.arange(len(tasks))
        h_means = [hv[t].get("mean_hidden_delta") or 0.0 for t in tasks]
        v_means = [hv[t].get("mean_visible_delta") or 0.0 for t in tasks]
        ax.bar(x - width / 2, h_means, width=width, label="hidden Δ",
                color="#a37")
        ax.bar(x + width / 2, v_means, width=width, label="visible Δ",
                color="#37a")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(tasks)
        ax.set_ylabel("mean intervention task_score Δ "
                       "(positive = perturbation hurt)")
        ax.set_title("hidden vs visible intervention task delta by task")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "hidden_vs_visible_delta_by_task.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hidden_vs_visible_delta_by_task.png: {e!r}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(payload: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage 5E - agent-task production post-hoc")
    lines.append("")
    lines.append("Computed from existing Stage 5E artifacts; no simulation.")
    lines.append("")
    lines.append("**Functional task probes only.** Not a consciousness or "
                 "agency claim.")
    lines.append("")

    # Per (source × task) main table.
    lines.append("## Per (source x task) summary")
    lines.append("")
    lines.append(_md_row([
        "source", "task", "n_cands", "mean task", "task CI",
        "mean HCE", "mean obs",
        "Pearson(HCE, task)", "Pearson(obs, task)",
        "high-low HCE diff",
    ]))
    lines.append(_md_row(["---"] * 10))
    per_st = payload["per_source_task"]
    for src in sorted(per_st):
        for task in sorted(per_st[src]):
            d = per_st[src][task]
            ci = d.get("task_score_ci") or (None, None)
            ci_str = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                      if ci[0] is not None else "—")
            lines.append(_md_row([
                src.split("_")[0], task, d["n_candidates"],
                f"{d.get('mean_task_score'):+.4f}"
                    if d.get("mean_task_score") is not None else "—",
                ci_str,
                f"{d.get('mean_hce'):+.4f}"
                    if d.get("mean_hce") is not None else "—",
                f"{d.get('mean_observer_score'):.2f}"
                    if d.get("mean_observer_score") is not None else "—",
                f"{d.get('pearson_hce_task'):+.3f}"
                    if d.get("pearson_hce_task") is not None else "—",
                f"{d.get('pearson_observer_task'):+.3f}"
                    if d.get("pearson_observer_task") is not None else "—",
                f"{d.get('high_minus_low_task_diff'):+.4f}",
            ]))
    lines.append("")

    # Hidden vs visible delta summary.
    hv = payload["hidden_vs_visible"]
    if hv:
        lines.append("## Hidden vs visible intervention deltas (per task)")
        lines.append("")
        lines.append(
            "Positive Δ = perturbation hurt task_score; "
            "negative Δ = perturbation helped. "
            "``mean_paired_diff_h_minus_v < 0`` ⇒ visible perturbations "
            "hurt more than hidden ones (paired by candidate)."
        )
        lines.append("")
        lines.append(_md_row([
            "task", "n paired", "mean hidden Δ", "mean visible Δ",
            "mean (h − v)", "frac visible more harmful",
        ]))
        lines.append(_md_row(["---"] + ["---:"] * 5))
        for task, d in hv.items():
            lines.append(_md_row([
                task, d.get("n_paired", 0),
                f"{d.get('mean_hidden_delta'):+.4f}"
                    if d.get("mean_hidden_delta") is not None else "—",
                f"{d.get('mean_visible_delta'):+.4f}"
                    if d.get("mean_visible_delta") is not None else "—",
                f"{d.get('mean_paired_diff_h_minus_v'):+.4f}"
                    if d.get("mean_paired_diff_h_minus_v") is not None else "—",
                f"{d.get('fraction_visible_more_harmful_than_hidden'):.2f}"
                    if d.get("fraction_visible_more_harmful_than_hidden")
                    is not None else "—",
            ]))
        lines.append("")

    # Pooled regression.
    reg = payload["regressions"]
    lines.append("## Pooled regressions (per (candidate, task))")
    lines.append("")
    lines.append(_md_row(["model", "n", "fit"]))
    lines.append(_md_row(["---", "---:", "---:"]))
    for k in ("task_score ~ HCE", "task_score ~ observer_score"):
        v = reg.get(k, {})
        r = v.get("pearson_r")
        lines.append(_md_row([
            k, v.get("n", 0),
            f"r = {r:+.3f}" if r is not None else "—",
        ]))
    multi = reg.get("task_score ~ HCE + observer_score")
    if multi and multi.get("r2") is not None:
        lines.append(_md_row([
            "task_score ~ HCE + observer_score", multi["n"],
            f"R² = {multi['r2']:+.3f}, β_HCE = {multi['beta_HCE']:+.3f}, "
            f"β_obs = {multi['beta_observer']:+.4f}",
        ]))
    src_reg = reg.get("task_score ~ HCE + observer_score + source", {})
    if src_reg.get("r2") is not None:
        lines.append(_md_row([
            "task_score ~ HCE + obs + source",
            src_reg.get("n", 0),
            f"R² = {src_reg['r2']:+.3f}, β_HCE = {src_reg['beta_HCE']:+.3f}, "
            f"β_obs = {src_reg['beta_observer']:+.4f}, "
            f"β_src_dummies = {[round(x, 4) for x in src_reg['beta_source_dummies']]} "
            f"(others vs M7)",
        ]))
    lines.append("")

    # Horizon-bucket analysis.
    hb = payload["per_horizon_bucket"]
    lines.append("## Per-horizon bucket task_score (all sources × tasks)")
    lines.append("")
    for bucket in ("short", "medium", "long"):
        d = hb.get(bucket, {})
        if d.get("n_total", 0) == 0: continue
        lines.append(f"### {bucket} (h={d.get('horizons')})")
        lines.append("")
        lines.append(_md_row(["source × task", "n", "mean task", "95% CI"]))
        lines.append(_md_row(["---"] + ["---:"] * 3))
        for key, v in sorted(d.get("per_source_task", {}).items()):
            ci = v.get("task_score_ci") or (None, None)
            ci_str = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                      if ci[0] is not None else "—")
            src, task = key.split("|", 1)
            lines.append(_md_row([
                f"{src.split('_')[0]} × {task}",
                v.get("n", 0),
                f"{v.get('mean_task_score'):+.4f}"
                    if v.get("mean_task_score") is not None else "—",
                ci_str,
            ]))
        lines.append("")

    # Activated interpretation.
    lines.append("## Activated interpretation")
    lines.append("")
    # HCE-memory association?
    mem_reg = []
    for src in per_st:
        d = per_st[src].get("memory") or {}
        r = d.get("pearson_hce_task")
        if r is not None:
            mem_reg.append(r)
    if mem_reg:
        avg_r = float(np.mean(mem_reg))
        if avg_r > 0.10:
            lines.append(
                f"* Mean Pearson(HCE, memory_score) across sources = "
                f"{avg_r:+.3f}. **HCE is associated with memory-like "
                "task behavior in this task definition.**"
            )
    rep_reg = []
    for src in per_st:
        d = per_st[src].get("repair") or {}
        r = d.get("pearson_hce_task")
        if r is not None:
            rep_reg.append(r)
    if rep_reg and abs(float(np.mean(rep_reg))) < 0.10:
        lines.append(
            f"* Mean Pearson(HCE, repair_score) across sources = "
            f"{float(np.mean(rep_reg)):+.3f} (small). **Repair appears "
            "more tied to visible structure or task mechanics than "
            "hidden causal dependence.**"
        )
    # Hidden vs visible.
    if hv.get("repair", {}).get("mean_paired_diff_h_minus_v") is not None:
        d = hv["repair"]["mean_paired_diff_h_minus_v"]
        if d < 0:
            lines.append(
                f"* repair task: hidden Δ < visible Δ "
                f"(mean h − v = {d:+.4f}). **Repair is more sensitive "
                "to visible damage than hidden-invisible modulation.**"
            )
    if hv.get("memory", {}).get("mean_paired_diff_h_minus_v") is not None:
        d = hv["memory"]["mean_paired_diff_h_minus_v"]
        if d > 0:
            lines.append(
                f"* memory task: hidden Δ > visible Δ "
                f"(mean h − v = {d:+.4f}). **Memory-like behavior may "
                "be more hidden-state dependent than visible-state "
                "dependent.**"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run(run_dir: Path, *, n_boot: int = 2000) -> dict:
    trials = _load_trials(run_dir)
    by_cand = _by_candidate_task(trials)
    payload = {
        "stage": "5E",
        "run_dir": str(run_dir).replace("\\", "/"),
        "n_trials": len(trials),
        "n_candidate_task_rows": len(by_cand),
        "n_boot": int(n_boot),
        "per_source_task": per_source_task_summary(by_cand, n_boot=n_boot),
        "per_horizon_bucket":
            per_horizon_bucket_by_source_task(trials, n_boot=n_boot),
        "regressions": pooled_regressions(by_cand, n_boot=n_boot),
        "hidden_vs_visible": hidden_vs_visible_delta_summary(by_cand),
    }
    out_csv = run_dir / "agent_task_posthoc.csv"
    out_json = run_dir / "agent_task_posthoc.json"
    out_md = run_dir / "agent_task_posthoc_summary.md"
    plots_dir = run_dir / "plots"

    # Flat CSV: per (source × task) row.
    fields = [
        "source", "task", "n_candidates", "mean_task_score",
        "task_score_ci_low", "task_score_ci_high",
        "mean_hce", "mean_observer_score",
        "pearson_hce_task", "pearson_observer_task",
        "n_with_hidden_delta", "mean_hidden_delta",
        "n_with_visible_delta", "mean_visible_delta",
        "mean_hidden_minus_visible",
        "high_minus_low_task_diff",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for src, by_task in payload["per_source_task"].items():
            for task, d in by_task.items():
                ci = d.get("task_score_ci") or (None, None)
                w.writerow({
                    "source": src, "task": task,
                    "n_candidates": d["n_candidates"],
                    "mean_task_score": d.get("mean_task_score"),
                    "task_score_ci_low": ci[0], "task_score_ci_high": ci[1],
                    "mean_hce": d.get("mean_hce"),
                    "mean_observer_score": d.get("mean_observer_score"),
                    "pearson_hce_task": d.get("pearson_hce_task"),
                    "pearson_observer_task": d.get("pearson_observer_task"),
                    "n_with_hidden_delta": d.get("n_with_hidden_delta"),
                    "mean_hidden_delta": d.get("mean_hidden_delta"),
                    "n_with_visible_delta": d.get("n_with_visible_delta"),
                    "mean_visible_delta": d.get("mean_visible_delta"),
                    "mean_hidden_minus_visible":
                        d.get("mean_hidden_minus_visible"),
                    "high_minus_low_task_diff":
                        d.get("high_minus_low_task_diff"),
                })
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
        description="Stage 5E post-hoc agent-task analysis.",
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
    print(f"Stage 5E post-hoc -> {run_dir}")
    for src, by_task in payload["per_source_task"].items():
        for task, d in by_task.items():
            ci = d.get("task_score_ci") or (None, None)
            ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci[0] is not None else "-"
            print(f"  {src:30s} {task:8s} n={d['n_candidates']:4d} "
                  f"mean={d.get('mean_task_score'):+.4f} CI={ci_str} "
                  f"r_hce={d.get('pearson_hce_task') or 0:+.3f} "
                  f"r_obs={d.get('pearson_observer_task') or 0:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
