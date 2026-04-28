"""Aggregator + summary writer for Follow-up Topic 3 (Stage 4)."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev


TASK_METRICS: tuple[str, ...] = (
    "task_score",
    "survival_time",
    "recovery_score",
    "perturbation_resilience",
    "cue_memory_score",
    "resource_contact_time",
    "movement_toward_resource",
    "HCE",
    "observer_score",
    "hidden_intervention_effect_on_task_score",
)

REGRESSION_MODELS: tuple[str, ...] = (
    "task_score ~ HCE",
    "task_score ~ observer_score",
    "task_score ~ HCE + observer_score",
    "task_score ~ HCE + observer_score + mechanism_class",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(_mean(xs)) if xs else None


def _safe_std(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(_stdev(xs)) if len(xs) >= 2 else None


def _pearson(xs, ys):
    """Pearson r, returning None when degenerate (constant x or y, or
    fewer than 3 paired points)."""
    pairs = [(float(x), float(y))
             for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx = _mean(xs2); my = _mean(ys2)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    if sx < 1e-12 or sy < 1e-12:
        return None
    cov = sum((xs2[i] - mx) * (ys2[i] - my) for i in range(len(xs2)))
    return float(cov / (sx * sy))


def _ols_two_vars(target, var1, var2):
    """Tiny two-variable OLS: target = a + b*var1 + c*var2.

    Returns dict with coefficients and R². Returns None if degenerate.
    Used for the ``task_score ~ HCE + observer_score`` model.
    """
    pairs = [
        (float(t), float(v1), float(v2))
        for t, v1, v2 in zip(target, var1, var2)
        if t is not None and v1 is not None and v2 is not None
    ]
    n = len(pairs)
    if n < 4:
        return None
    import numpy as np
    Y = np.array([p[0] for p in pairs])
    X = np.column_stack([
        np.ones(n),
        [p[1] for p in pairs],
        [p[2] for p in pairs],
    ])
    try:
        coef, residuals, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    pred = X @ coef
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    ss_res = float(((Y - pred) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "n": n,
        "intercept": float(coef[0]),
        "beta_var1": float(coef[1]),
        "beta_var2": float(coef[2]),
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_agent_task_results(
    trials, join_rows: list[dict],
) -> dict:
    """Build the stats_summary.json payload."""
    by_task: dict[str, list] = defaultdict(list)
    for t in trials:
        by_task[t.task_name].append(t)

    per_task: dict[str, dict] = {}
    for task, ts in by_task.items():
        h_deltas = [t.hidden_intervention_task_delta for t in ts
                     if t.hidden_intervention_task_delta is not None]
        v_deltas = [t.visible_intervention_task_delta for t in ts
                     if t.visible_intervention_task_delta is not None]
        # Per-trial paired hidden − visible delta when both exist.
        hv_pairs = [
            (t.hidden_intervention_task_delta - t.visible_intervention_task_delta)
            for t in ts
            if (t.hidden_intervention_task_delta is not None
                and t.visible_intervention_task_delta is not None)
        ]
        per_task[task] = {
            "n_trials": len(ts),
            "n_survived": sum(1 for t in ts if t.survived),
            "mean_task_score":
                _safe_mean(t.task_score for t in ts),
            "std_task_score":
                _safe_std(t.task_score for t in ts),
            "mean_survival_time":
                _safe_mean(t.survival_time for t in ts),
            "mean_hce": _safe_mean(t.hce for t in ts),
            "mean_observer_score":
                _safe_mean(t.observer_score for t in ts),
            # Hidden-intervention delta (Stage 5A).
            "n_with_hidden_intervention_delta": len(h_deltas),
            "mean_hidden_intervention_task_delta":
                _safe_mean(h_deltas),
            "fraction_perturbation_hurt_task":
                (sum(1 for d in h_deltas if d > 0) / len(h_deltas))
                if h_deltas else None,
            # Visible-intervention delta (Stage 5B).
            "n_with_visible_intervention_delta": len(v_deltas),
            "mean_visible_intervention_task_delta":
                _safe_mean(v_deltas),
            "fraction_visible_perturbation_hurt_task":
                (sum(1 for d in v_deltas if d > 0) / len(v_deltas))
                if v_deltas else None,
            # Hidden vs visible — paired per trial.
            "n_with_hidden_vs_visible_delta": len(hv_pairs),
            "mean_hidden_vs_visible_task_delta":
                _safe_mean(hv_pairs),
        }

    # Correlations: per-candidate (one row per candidate × task in
    # join_rows; ``mean_task_score`` is the per-candidate aggregate).
    correlations: dict[str, dict] = {}
    for task in by_task:
        rows = [r for r in join_rows if r["task_name"] == task]
        if len(rows) < 3:
            correlations[task] = {
                "n": len(rows),
                "pearson_HCE_vs_task_score": None,
                "pearson_observer_score_vs_task_score": None,
                "_status": "insufficient candidates for correlation",
            }
            continue
        correlations[task] = {
            "n": len(rows),
            "pearson_HCE_vs_task_score": _pearson(
                [r["hce"] for r in rows],
                [r["mean_task_score"] for r in rows],
            ),
            "pearson_observer_score_vs_task_score": _pearson(
                [r["observer_score"] for r in rows],
                [r["mean_task_score"] for r in rows],
            ),
        }

    # Regression: pooled across tasks.
    pooled_target = [r["mean_task_score"] for r in join_rows]
    pooled_hce = [r["hce"] for r in join_rows]
    pooled_obs = [r["observer_score"] for r in join_rows]
    regressions = {
        "task_score ~ HCE": {
            "n": sum(1 for r in join_rows if r["hce"] is not None),
            "pearson_r": _pearson(pooled_hce, pooled_target),
        },
        "task_score ~ observer_score": {
            "n": sum(1 for r in join_rows if r["observer_score"] is not None),
            "pearson_r": _pearson(pooled_obs, pooled_target),
        },
        "task_score ~ HCE + observer_score":
            _ols_two_vars(pooled_target, pooled_hce, pooled_obs),
    }

    # High-HCE vs low-HCE split (per task).
    split_results = {}
    for task in by_task:
        rows = [r for r in join_rows if r["task_name"] == task
                and r["hce"] is not None
                and r["mean_task_score"] is not None]
        if len(rows) < 4:
            split_results[task] = {
                "n": len(rows),
                "_status": "insufficient candidates for high/low split",
            }
            continue
        hce_vals = sorted(rows, key=lambda r: r["hce"])
        n = len(hce_vals)
        low = hce_vals[: n // 2]
        high = hce_vals[(n + 1) // 2:]
        split_results[task] = {
            "n_low": len(low),
            "n_high": len(high),
            "mean_task_score_low_hce":
                _safe_mean(r["mean_task_score"] for r in low),
            "mean_task_score_high_hce":
                _safe_mean(r["mean_task_score"] for r in high),
            "diff_high_minus_low":
                ((_safe_mean(r["mean_task_score"] for r in high) or 0.0)
                 - (_safe_mean(r["mean_task_score"] for r in low) or 0.0)),
        }

    return {
        "stage": 4,
        "metrics_recorded": list(TASK_METRICS),
        "regression_models_planned": list(REGRESSION_MODELS),
        "per_task": per_task,
        "correlations": correlations,
        "regressions": regressions,
        "high_low_hce_split": split_results,
    }


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(summary: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Follow-up Topic 3 — agent-task environments (Stage 4)")
    lines.append("")
    lines.append(
        "Three minimal **functional** task probes per candidate:\n\n"
        "* **repair** — knock out the candidate's hidden support; measure "
        "projected re-activity in the candidate region.\n"
        "* **foraging** — passive resource disc at fixed offset from "
        "centroid; measure projected activity inside it and centroid drift "
        "toward it. **Resource is non-coupling** — this is a smoke-level "
        "drift / proximity signal, not a foraging claim.\n"
        "* **memory** — apply two distinct deterministic hidden cues; "
        "measure projected divergence between cue-A and cue-B futures "
        "inside the candidate region. High divergence = cue identity "
        "persisted (memory-like).\n"
    )
    lines.append("")
    lines.append(
        "**These are functional probes.** No claim of agency, intent, "
        "or consciousness."
    )
    lines.append("")

    # Per-task summary
    lines.append("## Per-task summary")
    lines.append("")
    lines.append(_md_row(["task", "n_trials", "n_survived",
                           "mean task_score", "std task_score",
                           "mean HCE", "mean observer_proxy",
                           "mean survival_time"]))
    lines.append(_md_row(["---"] + ["---:"] * 7))
    for task, agg in summary["per_task"].items():
        lines.append(_md_row([
            task, agg["n_trials"], agg["n_survived"],
            f"{agg['mean_task_score']:+.4f}"
                if agg["mean_task_score"] is not None else "—",
            f"{agg['std_task_score']:+.4f}"
                if agg["std_task_score"] is not None else "—",
            f"{agg['mean_hce']:+.4f}"
                if agg["mean_hce"] is not None else "—",
            f"{agg['mean_observer_score']:+.2f}"
                if agg["mean_observer_score"] is not None else "—",
            f"{agg['mean_survival_time']:.2f}"
                if agg["mean_survival_time"] is not None else "—",
        ]))
    lines.append("")

    # Hidden + visible intervention task deltas (Stage 5A / 5B).
    has_h_delta = any(
        agg.get("n_with_hidden_intervention_delta", 0) > 0
        for agg in summary["per_task"].values()
    )
    has_v_delta = any(
        agg.get("n_with_visible_intervention_delta", 0) > 0
        for agg in summary["per_task"].values()
    )
    if has_h_delta or has_v_delta:
        lines.append("## Hidden vs visible intervention task deltas")
        lines.append("")
        lines.append(
            "``X_intervention_task_delta = task_score_original − "
            "task_score_X_perturbed``. Positive ⇒ the perturbation hurt "
            "task performance; negative ⇒ helped. ``hidden`` is a "
            "projection-preserving (invisible) perturbation; ``visible`` "
            "deliberately changes the projection."
        )
        lines.append("")
        lines.append(_md_row([
            "task", "n_h", "mean hidden Δ", "frac h-hurt",
            "n_v", "mean visible Δ", "frac v-hurt",
            "mean (h − v)",
        ]))
        lines.append(_md_row(["---"] + ["---:"] * 7))
        for task, agg in summary["per_task"].items():
            n_h = agg.get("n_with_hidden_intervention_delta", 0)
            mean_h = agg.get("mean_hidden_intervention_task_delta")
            frac_h = agg.get("fraction_perturbation_hurt_task")
            n_v = agg.get("n_with_visible_intervention_delta", 0)
            mean_v = agg.get("mean_visible_intervention_task_delta")
            frac_v = agg.get("fraction_visible_perturbation_hurt_task")
            mean_hv = agg.get("mean_hidden_vs_visible_task_delta")
            lines.append(_md_row([
                task, n_h,
                f"{mean_h:+.4f}" if mean_h is not None else "—",
                f"{frac_h:.2f}" if frac_h is not None else "—",
                n_v,
                f"{mean_v:+.4f}" if mean_v is not None else "—",
                f"{frac_v:.2f}" if frac_v is not None else "—",
                f"{mean_hv:+.4f}" if mean_hv is not None else "—",
            ]))
        lines.append("")

    # Correlations.
    lines.append("## HCE / observer_score vs task_score (per-candidate)")
    lines.append("")
    lines.append(_md_row(["task", "n", "Pearson(HCE, task)",
                           "Pearson(observer, task)"]))
    lines.append(_md_row(["---", "---:", "---:", "---:"]))
    for task, cor in summary["correlations"].items():
        lines.append(_md_row([
            task, cor.get("n", 0),
            f"{cor['pearson_HCE_vs_task_score']:+.3f}"
                if cor.get("pearson_HCE_vs_task_score") is not None else "—",
            f"{cor['pearson_observer_score_vs_task_score']:+.3f}"
                if cor.get("pearson_observer_score_vs_task_score") is not None
                else "—",
        ]))
    lines.append("")

    # Regression on pooled candidate × task.
    reg = summary.get("regressions", {})
    lines.append("## Pooled regression (per (candidate, task))")
    lines.append("")
    lines.append(_md_row(["model", "n", "Pearson r / R²"]))
    lines.append(_md_row(["---", "---:", "---:"]))
    for k in ("task_score ~ HCE", "task_score ~ observer_score"):
        v = reg.get(k, {})
        r = v.get("pearson_r")
        lines.append(_md_row([
            k, v.get("n", 0),
            f"r = {r:+.3f}" if r is not None else "—",
        ]))
    multi = reg.get("task_score ~ HCE + observer_score")
    if multi:
        lines.append(_md_row([
            "task_score ~ HCE + observer_score",
            multi["n"],
            f"R² = {multi['r2']:+.3f}, β_HCE = {multi['beta_var1']:+.3f}, "
            f"β_obs = {multi['beta_var2']:+.4f}"
            if multi.get("r2") is not None else "—",
        ]))
    lines.append("")

    # High vs low HCE split.
    if summary.get("high_low_hce_split"):
        lines.append("## High-HCE vs low-HCE candidates per task")
        lines.append("")
        lines.append(_md_row(["task", "n_low", "n_high",
                               "mean task low-HCE", "mean task high-HCE",
                               "diff (high − low)"]))
        lines.append(_md_row(["---"] + ["---:"] * 5))
        for task, split in summary["high_low_hce_split"].items():
            if "_status" in split:
                lines.append(_md_row([task, split.get("n", 0), "—",
                                       "—", "—", "—"]))
                continue
            lines.append(_md_row([
                task, split["n_low"], split["n_high"],
                f"{split['mean_task_score_low_hce']:+.4f}"
                    if split["mean_task_score_low_hce"] is not None else "—",
                f"{split['mean_task_score_high_hce']:+.4f}"
                    if split["mean_task_score_high_hce"] is not None else "—",
                f"{split['diff_high_minus_low']:+.4f}",
            ]))
        lines.append("")

    # Activated interpretation rule.
    lines.append("## Activated interpretation")
    lines.append("")
    if summary["n_trials"] == 0:
        lines.append(
            "* **No trials were run.** The smoke established no behavioural "
            "evidence."
        )
    else:
        # Interpretation logic based on pooled correlations.
        r_hce = (reg.get("task_score ~ HCE") or {}).get("pearson_r")
        r_obs = (reg.get("task_score ~ observer_score") or {}).get("pearson_r")
        thr = 0.10
        if (r_hce is None) and (r_obs is None):
            lines.append(
                "* **Insufficient candidates for a correlation read.** The "
                "smoke established infrastructure but not enough behavioural "
                "evidence."
            )
        elif r_hce is not None and r_obs is not None:
            if abs(r_hce) > thr and abs(r_hce) > abs(r_obs):
                lines.append(
                    f"* **Pearson(HCE, task_score) = {r_hce:+.3f}** is larger "
                    f"in magnitude than Pearson(observer, task_score) = "
                    f"{r_obs:+.3f}. Smoke suggests hidden causal dependence "
                    "may carry functional task-relevant information beyond "
                    "generic observer-likeness."
                )
            elif abs(r_obs) > thr and abs(r_obs) > abs(r_hce):
                lines.append(
                    f"* **Pearson(observer, task_score) = {r_obs:+.3f}** "
                    f"is larger than Pearson(HCE, task_score) = {r_hce:+.3f}. "
                    "Projected organization matters more than hidden causal "
                    "dependence for these tasks."
                )
            else:
                lines.append(
                    f"* **Both correlations are small** (HCE r = {r_hce:+.3f}, "
                    f"observer r = {r_obs:+.3f}). Current task definitions may "
                    "not align with existing candidate metrics."
                )

    path.write_text("\n".join(lines), encoding="utf-8")
