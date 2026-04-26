"""M8 — mechanism-discovery statistics.

Computes per-source and paired comparisons on the M8 measurements,
mechanism-class proportions, and HCE/lifetime trade-off correlations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from observer_worlds.analysis.m7b_stats import (
    cliffs_delta,
    cluster_bootstrap_by_groups,
    permutation_test_mean_diff,
    rank_biserial,
)
from observer_worlds.experiments._m8_mechanism import (
    MECHANISM_CLASSES,
    M8CandidateResult,
)


# ---------------------------------------------------------------------------
# Per-source aggregates
# ---------------------------------------------------------------------------


def _safe_mean(arr): return float(np.mean(arr)) if len(arr) else 0.0


def aggregate_per_source(results: list[M8CandidateResult]) -> dict:
    """Group results by rule_source and compute headline aggregates."""
    by_src: dict[str, list[M8CandidateResult]] = {}
    for r in results:
        by_src.setdefault(r.rule_source, []).append(r)
    out = {}
    for src, rs in by_src.items():
        out[src] = {
            "n_candidates": len(rs),
            "n_unique_rules": len({r.rule_id for r in rs}),
            "n_unique_seeds": len({r.seed for r in rs}),
            "mean_observer": _safe_mean([r.observer_score for r in rs
                                        if r.observer_score is not None]),
            "mean_lifetime": _safe_mean([r.candidate_lifetime for r in rs]),
            "mean_HCE": _safe_mean([r.timing.full_grid_l1_per_horizon[
                                    len(r.timing.horizons) // 2] for r in rs]),
            "mean_local_div": _safe_mean([r.timing.local_l1_per_horizon[
                                         len(r.timing.horizons) // 2] for r in rs]),
            "mean_first_visible_effect_time": _safe_mean([
                r.timing.first_visible_effect_time for r in rs
                if r.timing.first_visible_effect_time > 0
            ]),
            "mean_hidden_to_visible_conversion_time": _safe_mean([
                r.pathway.hidden_to_visible_conversion_time for r in rs
                if r.pathway.hidden_to_visible_conversion_time > 0
            ]),
            "mean_boundary_response_fraction": _safe_mean(
                [r.response_map.boundary_response_fraction for r in rs]
            ),
            "mean_interior_response_fraction": _safe_mean(
                [r.response_map.interior_response_fraction for r in rs]
            ),
            "mean_environment_response_fraction": _safe_mean(
                [r.response_map.environment_response_fraction for r in rs]
            ),
            "mean_boundary_mediation_index": _safe_mean(
                [r.mediation.boundary_mediation_index for r in rs]
            ),
            "mean_candidate_locality_index": _safe_mean(
                [r.mediation.candidate_locality_index for r in rs]
            ),
            "mean_far_hidden_effect": _safe_mean(
                [r.mediation.far_hidden_effect for r in rs]
            ),
            "mean_near_threshold_fraction": _safe_mean(
                [r.near_threshold_fraction for r in rs]
            ),
            "mean_fraction_hidden_at_end": _safe_mean(
                [r.pathway.fraction_hidden_at_end for r in rs]
            ),
            "mean_fraction_visible_at_end": _safe_mean(
                [r.pathway.fraction_visible_at_end for r in rs]
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Mechanism-class proportions
# ---------------------------------------------------------------------------


def mechanism_class_distribution(results: list[M8CandidateResult]) -> dict:
    """Per-source breakdown of mechanism labels with bootstrap CIs."""
    by_src: dict[str, list[M8CandidateResult]] = {}
    for r in results:
        by_src.setdefault(r.rule_source, []).append(r)
    out = {}
    rng = np.random.default_rng(0)
    n_boot = 500
    for src, rs in by_src.items():
        labels = [r.mechanism.label for r in rs]
        counts = Counter(labels)
        total = len(rs)
        per_class = {}
        for cls in MECHANISM_CLASSES:
            n = counts.get(cls, 0)
            frac = n / total if total > 0 else 0.0
            # Bootstrap CI over candidate-level resampling.
            if total > 0:
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    sample_labels = [labels[i] for i in
                                    rng.integers(0, total, size=total)]
                    boot[b] = sum(1 for l in sample_labels if l == cls) / total
                ci = (float(np.quantile(boot, 0.025)),
                      float(np.quantile(boot, 0.975)))
            else:
                ci = (0.0, 0.0)
            per_class[cls] = {"count": n, "fraction": float(frac),
                              "ci_low": ci[0], "ci_high": ci[1]}
        out[src] = {"n_total": total, "per_class": per_class}
    return out


# ---------------------------------------------------------------------------
# HCE / lifetime tradeoff and correlations
# ---------------------------------------------------------------------------


def _per_candidate_hce(r: M8CandidateResult) -> float:
    h = r.timing.horizons
    return r.timing.full_grid_l1_per_horizon[len(h) // 2] if h else 0.0


def hce_lifetime_correlations(results: list[M8CandidateResult]) -> dict:
    """Pearson correlations across candidates."""
    if len(results) < 3: return {}
    hce = np.array([_per_candidate_hce(r) for r in results])
    life = np.array([r.candidate_lifetime for r in results], dtype=np.float64)
    bnd = np.array([r.response_map.boundary_response_fraction for r in results])
    near_th = np.array([r.near_threshold_fraction for r in results])
    first_vis = np.array([
        max(r.timing.first_visible_effect_time, 0) for r in results
    ], dtype=np.float64)
    out = {}
    if hce.std() > 1e-12 and life.std() > 1e-12:
        out["pearson_HCE_vs_lifetime"] = float(np.corrcoef(hce, life)[0, 1])
    if hce.std() > 1e-12 and bnd.std() > 1e-12:
        out["pearson_HCE_vs_boundary_response"] = float(np.corrcoef(hce, bnd)[0, 1])
    if hce.std() > 1e-12 and near_th.std() > 1e-12:
        out["pearson_HCE_vs_near_threshold"] = float(np.corrcoef(hce, near_th)[0, 1])
    if hce.std() > 1e-12 and first_vis.std() > 1e-12:
        out["pearson_HCE_vs_first_visible_effect_time"] = float(
            np.corrcoef(hce, first_vis)[0, 1]
        )
    return out


# ---------------------------------------------------------------------------
# Paired source comparisons
# ---------------------------------------------------------------------------


def compare_sources_on_metric(
    results_a: list[M8CandidateResult], results_b: list[M8CandidateResult],
    *, metric_fn, source_a_name: str, source_b_name: str,
    n_boot: int = 1000, n_permutations: int = 1000, seed: int = 0,
) -> dict:
    """Compare two sources on a per-candidate metric extracted by metric_fn."""
    a_vals = np.array([metric_fn(r) for r in results_a])
    b_vals = np.array([metric_fn(r) for r in results_b])
    if a_vals.size == 0 or b_vals.size == 0:
        return {"n_a": int(a_vals.size), "n_b": int(b_vals.size),
                "mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "perm_p": 1.0,
                "cliffs_delta": 0.0, "rank_biserial": 0.0,
                "win_rate_a": 0.5}
    # Cluster bootstrap on the difference using rule+seed grouping.
    a_groups = np.array([f"{r.rule_id}|{r.seed}" for r in results_a])
    b_groups = np.array([f"{r.rule_id}|{r.seed}" for r in results_b])
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
        "n_a": int(a_vals.size), "n_b": int(b_vals.size),
        "mean_a": float(a_vals.mean()), "mean_b": float(b_vals.mean()),
        "mean_diff": float(a_vals.mean() - b_vals.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "perm_p": permutation_test_mean_diff(
            a_vals, b_vals, n_permutations=n_permutations, seed=seed + 1,
        ),
        "cliffs_delta": cliffs_delta(a_vals, b_vals),
        "rank_biserial": rank_biserial(a_vals, b_vals),
        "win_rate_a": float((a_vals[:, None] > b_vals[None, :]).mean()),
    }


def m7_vs_baseline_grid(results: list[M8CandidateResult]) -> dict:
    """Build a grid of M7 vs each baseline on each primary M8 metric."""
    by_src: dict[str, list[M8CandidateResult]] = {}
    for r in results:
        by_src.setdefault(r.rule_source, []).append(r)
    if "M7_HCE_optimized" not in by_src:
        return {}
    rows_m7 = by_src["M7_HCE_optimized"]
    metrics = {
        "HCE": _per_candidate_hce,
        "boundary_response_fraction":
            lambda r: r.response_map.boundary_response_fraction,
        "interior_response_fraction":
            lambda r: r.response_map.interior_response_fraction,
        "first_visible_effect_time":
            lambda r: max(r.timing.first_visible_effect_time, 0),
        "hidden_to_visible_conversion_time":
            lambda r: max(r.pathway.hidden_to_visible_conversion_time, 0),
        "boundary_mediation_index":
            lambda r: r.mediation.boundary_mediation_index,
        "candidate_locality_index":
            lambda r: r.mediation.candidate_locality_index,
        "fraction_hidden_at_end":
            lambda r: r.pathway.fraction_hidden_at_end,
        "candidate_lifetime":
            lambda r: float(r.candidate_lifetime),
    }
    out: dict[str, dict[str, dict]] = {}
    for src in by_src:
        if src == "M7_HCE_optimized": continue
        cmp_key = f"M7_vs_{src}"
        out[cmp_key] = {}
        for metric_name, fn in metrics.items():
            out[cmp_key][metric_name] = compare_sources_on_metric(
                rows_m7, by_src[src],
                metric_fn=fn, source_a_name="M7_HCE_optimized",
                source_b_name=src,
            )
    return out


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------


def m8_full_summary(results: list[M8CandidateResult]) -> dict:
    return {
        "n_candidates": len(results),
        "aggregates": aggregate_per_source(results),
        "mechanism_distribution": mechanism_class_distribution(results),
        "correlations_per_source": {
            src: hce_lifetime_correlations([r for r in results
                                           if r.rule_source == src])
            for src in {r.rule_source for r in results}
        },
        "comparison_grid": m7_vs_baseline_grid(results),
    }


# ---------------------------------------------------------------------------
# Markdown rendering + interpretation
# ---------------------------------------------------------------------------


_INTERP_BOUNDARY_MEDIATED = (
    "M7's hidden causal dependence is primarily mediated by hidden state "
    "under projected candidate boundaries."
)
_INTERP_INTERIOR_RESERVOIR = (
    "M7 candidates appear to use hidden fibers as latent internal state "
    "reservoirs."
)
_INTERP_DELAYED_CHANNEL = (
    "Hidden perturbations propagate invisibly before becoming visible, "
    "supporting a hidden-channel mechanism."
)
_INTERP_GLOBAL_CHAOTIC = (
    "M7's HCE may reflect broad instability rather than candidate-local "
    "hidden support."
)
_INTERP_NOT_THRESHOLD = (
    "M8 confirms M7's HCE is not primarily threshold-mediated."
)
_INTERP_TRADEOFF_STRONG = (
    "Hidden causal sensitivity increases observer-like dependence but "
    "reduces persistence."
)
_INTERP_STABLE_SUBPOPULATION = (
    "M8 identifies a promising subpopulation of stable hidden-supported "
    "projected observers."
)


def _select_interpretations(summary: dict) -> list[str]:
    out: list[str] = []
    mech_dist = summary.get("mechanism_distribution", {})
    m7_dist = mech_dist.get("M7_HCE_optimized", {}).get("per_class", {})
    if not m7_dist:
        return ["No M7 candidates found; cannot interpret."]

    # Find dominant class.
    fractions = {cls: d["fraction"] for cls, d in m7_dist.items()}
    if fractions:
        dominant = max(fractions, key=fractions.get)
        if fractions[dominant] > 0.4:
            if dominant == "boundary_mediated":
                out.append(_INTERP_BOUNDARY_MEDIATED)
            elif dominant == "interior_reservoir":
                out.append(_INTERP_INTERIOR_RESERVOIR)
            elif dominant == "delayed_hidden_channel":
                out.append(_INTERP_DELAYED_CHANNEL)
            elif dominant == "global_chaotic":
                out.append(_INTERP_GLOBAL_CHAOTIC)

    # Threshold-mediated check.
    thresh_frac = fractions.get("threshold_mediated", 0.0)
    if thresh_frac < 0.15:
        out.append(_INTERP_NOT_THRESHOLD)

    # Trade-off check.
    cors = summary.get("correlations_per_source", {})
    m7_cors = cors.get("M7_HCE_optimized", {})
    hce_life = m7_cors.get("pearson_HCE_vs_lifetime", 0.0)
    if hce_life < -0.3:
        out.append(_INTERP_TRADEOFF_STRONG)

    # Stable subpopulation check.
    m7_aggs = summary.get("aggregates", {}).get("M7_HCE_optimized", {})
    if m7_aggs.get("mean_lifetime", 0) > 30 and m7_aggs.get("mean_HCE", 0) > 0.10:
        out.append(_INTERP_STABLE_SUBPOPULATION)

    if not out:
        out.append("Mixed result; no single mechanism dominates.")
    return out


def render_m8_summary_md(summary: dict) -> str:
    lines = ["# M8 — Mechanism Discovery", ""]
    lines.append(f"- N candidates measured: {summary['n_candidates']}")
    lines.append("")
    lines.append("## Per-source aggregates")
    lines.append("")
    aggs = summary.get("aggregates", {})
    if aggs:
        lines.append("| source | n | mean_obs | mean_life | mean_HCE | "
                    "boundary_resp | interior_resp | env_resp | "
                    "first_visible_t | hidden→visible_t | near_thresh |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for src, a in aggs.items():
            lines.append(
                f"| {src} | {a['n_candidates']} | "
                f"{a['mean_observer']:+.3f} | {a['mean_lifetime']:.0f} | "
                f"{a['mean_HCE']:+.4f} | "
                f"{a['mean_boundary_response_fraction']:.2f} | "
                f"{a['mean_interior_response_fraction']:.2f} | "
                f"{a['mean_environment_response_fraction']:.2f} | "
                f"{a['mean_first_visible_effect_time']:.1f} | "
                f"{a['mean_hidden_to_visible_conversion_time']:.1f} | "
                f"{a['mean_near_threshold_fraction']:.2f} |"
            )
    lines.append("")
    lines.append("## Mechanism class distribution per source")
    lines.append("")
    mech = summary.get("mechanism_distribution", {})
    for src, d in mech.items():
        lines.append(f"### {src} (N={d['n_total']})")
        lines.append("")
        lines.append("| mechanism | count | fraction | 95% CI |")
        lines.append("|---|---|---|---|")
        for cls in MECHANISM_CLASSES:
            r = d["per_class"].get(cls, {})
            lines.append(
                f"| {cls} | {r.get('count', 0)} | "
                f"{r.get('fraction', 0):.2f} | "
                f"[{r.get('ci_low', 0):.2f}, {r.get('ci_high', 0):.2f}] |"
            )
        lines.append("")
    lines.append("## HCE / property correlations per source")
    lines.append("")
    cors = summary.get("correlations_per_source", {})
    for src, c in cors.items():
        lines.append(f"### {src}")
        for k, v in c.items():
            lines.append(f"- {k} = {v:+.3f}")
        lines.append("")
    lines.append("## Paired comparisons (M7 vs each baseline) on M8 metrics")
    lines.append("")
    grid = summary.get("comparison_grid", {})
    if grid:
        lines.append("| comparison | metric | mean_a | mean_b | mean_diff | "
                    "95% CI | perm p | Cliff's δ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cmp_key, by_metric in grid.items():
            for metric, c in by_metric.items():
                lines.append(
                    f"| {cmp_key} | {metric} | "
                    f"{c['mean_a']:+.4f} | {c['mean_b']:+.4f} | "
                    f"{c['mean_diff']:+.4f} | "
                    f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                    f"{c['perm_p']:.4f} | {c['cliffs_delta']:+.3f} |"
                )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for p in _select_interpretations(summary):
        lines.append(f"- {p}")
    return "\n".join(lines)
