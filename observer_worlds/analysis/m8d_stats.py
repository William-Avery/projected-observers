"""M8D — global-chaotic decomposition statistics."""

from __future__ import annotations

from collections import Counter

import numpy as np

from observer_worlds.experiments._m8d_decomposition import (
    M8D_GLOBAL_SUBCLASSES, M8D_MECHANISM_CLASSES, M8DCandidateResult,
)


def _by_source(rs):
    out = {}
    for r in rs: out.setdefault(r.rule_source, []).append(r)
    return out


def _is_thick(r):
    return r.morphology.morphology_class in (
        "thick_candidate", "very_thick_candidate"
    )


def _safe_mean(arr): return float(np.mean(arr)) if len(arr) else 0.0


def aggregate_per_source(results: list[M8DCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        global_base = [r for r in thick
                       if r.base_mechanism_label == "global_chaotic"]
        n = max(len(rs), 1)
        out[src] = {
            "n_total": len(rs),
            "n_thick": len(thick),
            "n_global_base": len(global_base),
            "global_chaotic_base_thick_fraction":
                len(global_base) / max(len(thick), 1),
            "mean_body_over_background": _safe_mean([
                r.body_over_background for r in thick
            ]),
            "mean_far_over_background": _safe_mean([
                r.far_over_background for r in thick
            ]),
            "global_subclass_counts": Counter(
                r.final_mechanism_label for r in global_base
            ),
        }
    return out


def global_subclass_distribution(results: list[M8DCandidateResult]) -> dict:
    """Distribution of M8D global subclasses among the originally
    global_chaotic candidates."""
    rng = np.random.default_rng(0); n_boot = 500
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        global_base = [r for r in thick
                       if r.base_mechanism_label == "global_chaotic"]
        labels = [r.final_mechanism_label for r in global_base]
        n = len(labels)
        per_class = {}
        for cls in M8D_GLOBAL_SUBCLASSES:
            k = sum(1 for l in labels if l == cls)
            frac = k / n if n > 0 else 0.0
            if n > 0:
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    sample = [labels[i] for i in
                             rng.integers(0, n, size=n)]
                    boot[b] = sum(1 for l in sample if l == cls) / n
                ci = (float(np.quantile(boot, 0.025)),
                      float(np.quantile(boot, 0.975)))
            else:
                ci = (0.0, 0.0)
            per_class[cls] = {"count": k, "fraction": float(frac),
                              "ci_low": ci[0], "ci_high": ci[1]}
        out[src] = {"n_global_base": n, "per_class": per_class}
    return out


def feature_comparison_global_vs_local(
    results: list[M8DCandidateResult],
) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        global_thick = [r for r in thick
                        if r.base_mechanism_label == "global_chaotic"]
        local_thick = [r for r in thick if r.base_mechanism_label
                       in ("interior_reservoir", "whole_body_hidden_support")]
        if not (global_thick and local_thick):
            out[src] = {"n_global": len(global_thick), "n_local": len(local_thick)}
            continue
        feat_keys = list(global_thick[0].feature_audit.keys())
        diffs = {}
        for fk in feat_keys:
            g_vals = np.array([r.feature_audit.get(fk, 0.0)
                               for r in global_thick])
            l_vals = np.array([r.feature_audit.get(fk, 0.0)
                               for r in local_thick])
            diffs[fk] = {
                "n_global": len(g_vals), "n_local": len(l_vals),
                "mean_global": float(g_vals.mean()),
                "mean_local": float(l_vals.mean()),
                "mean_diff": float(g_vals.mean() - l_vals.mean()),
            }
        out[src] = diffs
    return out


def stabilization_reclassification_rates(
    results: list[M8DCandidateResult],
) -> dict:
    """Per-stabilization-variant: fraction of originally-global
    candidates that no longer fire global_chaotic under that variant."""
    out = {}
    for src, rs in _by_source(results).items():
        global_base = [r for r in rs if _is_thick(r)
                       and r.base_mechanism_label == "global_chaotic"]
        if not global_base:
            out[src] = {"n": 0}; continue
        variants = {}
        for variant in ("baseline", "short_horizon", "local_window"):
            fires = []
            for r in global_base:
                if variant in r.stabilization:
                    v = r.stabilization[variant]
                    if "global_chaotic_label_would_fire" in v:
                        fires.append(v["global_chaotic_label_would_fire"])
            if fires:
                variants[variant] = {
                    "n": len(fires),
                    "fires_fraction": float(np.mean(fires)),
                    "no_longer_fires_fraction": 1.0 - float(np.mean(fires)),
                }
        out[src] = {"n": len(global_base), "by_variant": variants}
    return out


def m8d_full_summary(results: list[M8DCandidateResult]) -> dict:
    return {
        "n_candidates": len(results),
        "aggregates": aggregate_per_source(results),
        "global_subclass_distribution": global_subclass_distribution(results),
        "feature_comparison_global_vs_local":
            feature_comparison_global_vs_local(results),
        "stabilization_reclassification_rates":
            stabilization_reclassification_rates(results),
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


_INTERP_TRUE_GLOBAL = (
    "The global-chaotic subset reflects true system-wide hidden instability."
)
_INTERP_BROAD_COUPLING = (
    "The global-chaotic subset is better described as broad hidden "
    "coupling, not pure artifact."
)
_INTERP_BACKGROUND_SENSITIVE = (
    "Global-chaotic labels arise from high background hidden sensitivity "
    "in those worlds."
)
_INTERP_THRESHOLD_VOL = (
    "Global-chaotic candidates are largely volatility or threshold-mediated."
)
_INTERP_LOCAL_WINDOW_REMOVES = (
    "Global-chaotic labels reflect propagation through the broader world "
    "rather than candidate-intrinsic support."
)
_INTERP_RESIDUAL_BODY = (
    "Even globally sensitive candidates retain candidate-specific "
    "hidden support."
)


def select_interpretations(summary: dict) -> list[str]:
    out: list[str] = []
    sub = summary.get("global_subclass_distribution", {})
    m7 = sub.get("M7_HCE_optimized", {})
    if m7.get("n_global_base", 0) == 0:
        out.append("No M7 global_chaotic candidates available for "
                   "decomposition.")
        return out
    f = {cls: d["fraction"] for cls, d in m7.get("per_class", {}).items()}

    if f.get("global_instability", 0) >= 0.40:
        out.append(_INTERP_TRUE_GLOBAL)
    if f.get("broad_hidden_coupling", 0) >= 0.40:
        out.append(_INTERP_BROAD_COUPLING)
    if f.get("background_sensitive_world", 0) >= 0.30:
        out.append(_INTERP_BACKGROUND_SENSITIVE)
    if f.get("threshold_volatility_artifact", 0) >= 0.30:
        out.append(_INTERP_THRESHOLD_VOL)

    stab = summary.get("stabilization_reclassification_rates", {}) \
                  .get("M7_HCE_optimized", {}).get("by_variant", {}).get("local_window")
    if stab and stab.get("no_longer_fires_fraction", 0) >= 0.50:
        out.append(_INTERP_LOCAL_WINDOW_REMOVES)

    aggs = summary.get("aggregates", {}).get("M7_HCE_optimized", {})
    if aggs.get("mean_body_over_background", 0) > 1.5:
        out.append(_INTERP_RESIDUAL_BODY)

    if not out:
        out.append("Mixed result; no single global-decomposition pattern dominates.")
    return out


def render_m8d_summary_md(summary: dict) -> str:
    lines = ["# M8D — Global-chaotic Decomposition", ""]
    lines.append(f"- N candidates measured: {summary['n_candidates']}")
    lines.append("")
    lines.append("## Per-source aggregates")
    lines.append("")
    aggs = summary.get("aggregates", {})
    if aggs:
        lines.append("| source | n | n_thick | n_global_base | "
                    "%global_thick | body_over_bg | far_over_bg |")
        lines.append("|---|---|---|---|---|---|---|")
        for src, a in aggs.items():
            lines.append(
                f"| {src} | {a['n_total']} | {a['n_thick']} | "
                f"{a['n_global_base']} | "
                f"{a['global_chaotic_base_thick_fraction']:.2f} | "
                f"{a['mean_body_over_background']:.2f} | "
                f"{a['mean_far_over_background']:.2f} |"
            )
        lines.append("")
    lines.append("## Global subclass distribution among global_chaotic candidates")
    lines.append("")
    sub = summary.get("global_subclass_distribution", {})
    for src, d in sub.items():
        lines.append(f"### {src} (N_global_base={d['n_global_base']})")
        lines.append("| subclass | count | fraction | 95% CI |")
        lines.append("|---|---|---|---|")
        for cls in M8D_GLOBAL_SUBCLASSES:
            r = d["per_class"].get(cls, {})
            lines.append(
                f"| {cls} | {r.get('count', 0)} | "
                f"{r.get('fraction', 0):.2f} | "
                f"[{r.get('ci_low', 0):.2f}, {r.get('ci_high', 0):.2f}] |"
            )
        lines.append("")
    lines.append("## Stabilization reclassification rates")
    lines.append("")
    stab = summary.get("stabilization_reclassification_rates", {})
    for src, d in stab.items():
        if not d.get("by_variant"): continue
        lines.append(f"### {src} (N_global_base={d.get('n', 0)})")
        lines.append("| variant | %still_global | %no_longer_global |")
        lines.append("|---|---|---|")
        for variant, vd in d["by_variant"].items():
            lines.append(
                f"| {variant} | {vd['fires_fraction']:.2f} | "
                f"{vd['no_longer_fires_fraction']:.2f} |"
            )
        lines.append("")
    lines.append("## Feature comparison: global_chaotic vs interior/whole-body")
    lines.append("")
    fc = summary.get("feature_comparison_global_vs_local", {})
    for src, fd in fc.items():
        if isinstance(fd, dict) and fd and "n_global" in fd:
            continue  # not enough data
        lines.append(f"### {src}")
        if not fd: lines.append("(insufficient data)"); continue
        lines.append("| feature | mean_global | mean_local | mean_diff |")
        lines.append("|---|---|---|---|")
        for fk, vals in fd.items():
            if isinstance(vals, dict) and "mean_global" in vals:
                lines.append(
                    f"| {fk} | {vals['mean_global']:+.4f} | "
                    f"{vals['mean_local']:+.4f} | "
                    f"{vals['mean_diff']:+.4f} |"
                )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for p in select_interpretations(summary):
        lines.append(f"- {p}")
    return "\n".join(lines)
