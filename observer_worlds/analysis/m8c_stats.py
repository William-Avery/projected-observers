"""M8C — large-grid mechanism validation statistics.

Reuses M8B's per-source / thick-only logic and adds far-control-quality
diagnostics.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from observer_worlds.analysis.m7b_stats import (
    cliffs_delta,
    permutation_test_mean_diff,
    rank_biserial,
)
from observer_worlds.detection.morphology import MORPHOLOGY_CLASSES
from observer_worlds.experiments._m8b_spatial import M8B_MECHANISM_CLASSES
from observer_worlds.experiments._m8c_validation import M8CCandidateResult


def _by_source(results):
    out = {}
    for r in results: out.setdefault(r.rule_source, []).append(r)
    return out


def _is_thick(r):
    return r.morphology.morphology_class in (
        "thick_candidate", "very_thick_candidate"
    )


def _hce(r): return float(r.region_effects["whole"].region_hidden_effect)


def _safe_mean(arr): return float(np.mean(arr)) if len(arr) else 0.0


# ---------------------------------------------------------------------------
# Per-source aggregates with M8C-specific columns
# ---------------------------------------------------------------------------


def aggregate_per_source(results: list[M8CCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        very_thick = [r for r in rs if r.morphology.morphology_class
                      == "very_thick_candidate"]
        thin = [r for r in rs if r.morphology.morphology_class == "thin_candidate"]
        valid_far = [r for r in rs if r.far_control.far_control_valid]
        thick_valid_far = [r for r in thick if r.far_control.far_control_valid]
        n = max(len(rs), 1)
        gc_thick = sum(1 for r in thick if r.mechanism_label == "global_chaotic")
        ir_thick = sum(1 for r in thick
                       if r.mechanism_label == "interior_reservoir")
        wb_thick = sum(1 for r in thick
                       if r.mechanism_label == "whole_body_hidden_support")
        bm_thick = sum(1 for r in thick
                       if r.mechanism_label == "boundary_mediated")
        ec_thick = sum(1 for r in thick
                       if r.mechanism_label == "environment_coupled")
        out[src] = {
            "n_total": len(rs),
            "n_thick": len(thick),
            "n_very_thick": len(very_thick),
            "thick_fraction": len(thick) / n,
            "very_thick_fraction": len(very_thick) / n,
            "n_unique_rules": len({r.rule_id for r in rs}),
            "n_unique_seeds": len({r.seed for r in rs}),
            "mean_lifetime": _safe_mean([r.candidate_lifetime for r in rs]),
            "mean_area": _safe_mean([r.candidate_area for r in rs]),
            "mean_HCE": _safe_mean([_hce(r) for r in rs]),
            "threshold_filtered_HCE": _safe_mean([
                _hce(r) for r in rs if r.near_threshold_fraction < 0.10
            ]),
            "candidate_locality_index": _safe_mean([
                _hce(r) - r.far_effect.region_hidden_effect for r in valid_far
            ]),
            "global_chaotic_thick_fraction":
                gc_thick / max(len(thick), 1),
            "interior_reservoir_thick_fraction":
                ir_thick / max(len(thick), 1),
            "whole_body_thick_fraction":
                wb_thick / max(len(thick), 1),
            "boundary_mediated_thick_fraction":
                bm_thick / max(len(thick), 1),
            "environment_coupled_thick_fraction":
                ec_thick / max(len(thick), 1),
            "far_control_valid_fraction":
                len(valid_far) / n,
            "mean_far_distance_over_radius": _safe_mean([
                r.far_control.far_control_distance_over_radius
                for r in valid_far
            ]),
        }
    return out


def morphology_class_distribution(results: list[M8CCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        cnts = Counter(r.morphology.morphology_class for r in rs)
        n = max(len(rs), 1)
        out[src] = {
            "n_total": len(rs),
            "per_class": {
                cls: {"count": cnts.get(cls, 0),
                      "fraction": cnts.get(cls, 0) / n}
                for cls in MORPHOLOGY_CLASSES
            },
        }
    return out


def mechanism_class_distribution_thick_only(
    results: list[M8CCandidateResult],
) -> dict:
    rng = np.random.default_rng(0)
    n_boot = 500
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        labels = [r.mechanism_label for r in thick]
        n = len(thick)
        per_class = {}
        for cls in M8B_MECHANISM_CLASSES:
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
        out[src] = {"n_thick": n, "per_class": per_class}
    return out


def far_control_quality_summary(results: list[M8CCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        valid = [r for r in rs if r.far_control.far_control_valid]
        n = max(len(rs), 1)
        out[src] = {
            "n_total": len(rs),
            "n_valid_far": len(valid),
            "valid_far_fraction": len(valid) / n,
            "mean_far_distance": _safe_mean([
                r.far_control.far_control_distance for r in valid
            ]),
            "mean_far_distance_over_radius": _safe_mean([
                r.far_control.far_control_distance_over_radius for r in valid
            ]),
            "mean_far_proj_activity_diff": _safe_mean([
                r.far_control.far_control_projected_activity_diff for r in valid
            ]),
            "mean_far_hidden_activity_diff": _safe_mean([
                r.far_control.far_control_hidden_activity_diff for r in valid
            ]),
        }
    return out


def m7_vs_baselines_thick_grid(results: list[M8CCandidateResult]) -> dict:
    by_src = _by_source(results)
    thick_by = {s: [r for r in rs if _is_thick(r)] for s, rs in by_src.items()}
    if "M7_HCE_optimized" not in thick_by or not thick_by["M7_HCE_optimized"]:
        return {}
    rows_m7 = thick_by["M7_HCE_optimized"]
    metrics = {
        "HCE_whole": _hce,
        "candidate_locality": lambda r: _hce(r) - r.far_effect.region_hidden_effect,
        "interior_per_cell":
            lambda r: r.region_effects["interior"].region_effect_per_cell,
        "boundary_per_cell":
            lambda r: r.region_effects["boundary"].region_effect_per_cell,
        "environment_per_cell":
            lambda r: r.region_effects["environment"].region_effect_per_cell,
        "candidate_area": lambda r: float(r.candidate_area),
        "candidate_lifetime": lambda r: float(r.candidate_lifetime),
        "far_distance_over_radius":
            lambda r: r.far_control.far_control_distance_over_radius,
    }
    out = {}
    for src in thick_by:
        if src == "M7_HCE_optimized" or not thick_by[src]: continue
        out[f"M7_thick_vs_{src}_thick"] = {}
        for m, fn in metrics.items():
            a = np.array([fn(r) for r in rows_m7])
            b = np.array([fn(r) for r in thick_by[src]])
            out[f"M7_thick_vs_{src}_thick"][m] = {
                "n_a": int(a.size), "n_b": int(b.size),
                "mean_a": float(a.mean()) if a.size else 0.0,
                "mean_b": float(b.mean()) if b.size else 0.0,
                "mean_diff": float((a.mean() if a.size else 0.0)
                                   - (b.mean() if b.size else 0.0)),
                "perm_p": (permutation_test_mean_diff(a, b, n_permutations=500,
                                                     seed=0)
                           if a.size and b.size else 1.0),
                "cliffs_delta": (cliffs_delta(a, b)
                                if a.size and b.size else 0.0),
            }
    return out


def m8c_full_summary(results: list[M8CCandidateResult]) -> dict:
    return {
        "n_candidates": len(results),
        "aggregates": aggregate_per_source(results),
        "morphology_distribution": morphology_class_distribution(results),
        "mechanism_distribution_thick": mechanism_class_distribution_thick_only(results),
        "far_control_quality": far_control_quality_summary(results),
        "comparison_grid_thick": m7_vs_baselines_thick_grid(results),
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


_INTERP_LARGE_GRID_INTERIOR = (
    "Large-grid M8C confirms M7's hidden causal support is primarily "
    "interior/whole-body, not boundary-mediated."
)
_INTERP_BOUNDARY_REVEALED = (
    "M8C revises M8B: boundary mediation becomes visible once "
    "morphology is sufficiently resolved."
)
_INTERP_ENVIRONMENT_DOMINANT = (
    "Hidden support is local-environment mediated rather than "
    "internal-body mediated."
)
_INTERP_GLOBAL_CHAOS_PERSISTS = (
    "M7 HCE may reflect broad dynamical instability rather than "
    "localized hidden support."
)
_INTERP_NO_THICK_BASELINES = (
    "M7 uniquely produces thick hidden-supported candidates under this "
    "search regime; matched thick-baseline comparison remains unavailable."
)


def select_interpretations(summary: dict) -> list[str]:
    out: list[str] = []
    mech = summary.get("mechanism_distribution_thick", {})
    m7 = mech.get("M7_HCE_optimized", {})
    if m7.get("n_thick", 0) == 0:
        out.append("M7 produced no thick candidates at this scale; M8C "
                   "mechanism evidence is inconclusive.")
        return out
    f = {cls: d["fraction"] for cls, d in m7.get("per_class", {}).items()}
    interior_fam = f.get("interior_reservoir", 0) + f.get("whole_body_hidden_support", 0)
    if interior_fam >= 0.50 \
            and f.get("boundary_mediated", 0) < 0.20 \
            and f.get("environment_coupled", 0) < 0.20:
        out.append(_INTERP_LARGE_GRID_INTERIOR)
    elif f.get("boundary_mediated", 0) >= 0.40:
        out.append(_INTERP_BOUNDARY_REVEALED)
    elif f.get("environment_coupled", 0) >= 0.40:
        out.append(_INTERP_ENVIRONMENT_DOMINANT)
    if f.get("global_chaotic", 0) >= 0.40:
        out.append(_INTERP_GLOBAL_CHAOS_PERSISTS)

    aggs = summary.get("aggregates", {})
    n_thick_m4c = aggs.get("M4C_observer_optimized", {}).get("n_thick", 0)
    n_thick_m4a = aggs.get("M4A_viability", {}).get("n_thick", 0)
    if n_thick_m4c == 0 and n_thick_m4a < 5:
        out.append(_INTERP_NO_THICK_BASELINES)

    if not out:
        out.append("Mixed result; no single mechanism dominates among thick M7.")
    return out


def render_m8c_summary_md(summary: dict) -> str:
    lines = ["# M8C — Large-grid Mechanism Validation", ""]
    lines.append(f"- N candidates measured: {summary['n_candidates']}")
    lines.append("")
    lines.append("## Per-source headline aggregates")
    lines.append("")
    aggs = summary.get("aggregates", {})
    if aggs:
        lines.append("| source | n | n_thick | thick_frac | mean_HCE | "
                    "thresh_filt_HCE | locality | %global_chaotic | "
                    "%interior | %whole_body | %boundary | %env | "
                    "%far_valid | mean_far/r |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for src, a in aggs.items():
            lines.append(
                f"| {src} | {a['n_total']} | {a['n_thick']} | "
                f"{a['thick_fraction']:.2f} | "
                f"{a['mean_HCE']:+.4f} | {a['threshold_filtered_HCE']:+.4f} | "
                f"{a['candidate_locality_index']:+.4f} | "
                f"{a['global_chaotic_thick_fraction']:.2f} | "
                f"{a['interior_reservoir_thick_fraction']:.2f} | "
                f"{a['whole_body_thick_fraction']:.2f} | "
                f"{a['boundary_mediated_thick_fraction']:.2f} | "
                f"{a['environment_coupled_thick_fraction']:.2f} | "
                f"{a['far_control_valid_fraction']:.2f} | "
                f"{a['mean_far_distance_over_radius']:.1f} |"
            )
        lines.append("")
    lines.append("## Morphology distribution per source")
    lines.append("")
    morph = summary.get("morphology_distribution", {})
    for src, d in morph.items():
        lines.append(f"### {src} (N={d['n_total']})")
        lines.append("| morphology | count | fraction |")
        lines.append("|---|---|---|")
        for cls in MORPHOLOGY_CLASSES:
            r = d["per_class"].get(cls, {})
            lines.append(f"| {cls} | {r.get('count', 0)} | "
                        f"{r.get('fraction', 0):.2f} |")
        lines.append("")
    lines.append("## Mechanism distribution among thick candidates")
    lines.append("")
    mech = summary.get("mechanism_distribution_thick", {})
    for src, d in mech.items():
        lines.append(f"### {src} (N_thick={d['n_thick']})")
        lines.append("| mechanism | count | fraction | 95% CI |")
        lines.append("|---|---|---|---|")
        for cls in M8B_MECHANISM_CLASSES:
            r = d["per_class"].get(cls, {})
            lines.append(
                f"| {cls} | {r.get('count', 0)} | "
                f"{r.get('fraction', 0):.2f} | "
                f"[{r.get('ci_low', 0):.2f}, {r.get('ci_high', 0):.2f}] |"
            )
        lines.append("")
    lines.append("## Far-control quality")
    lines.append("")
    fcq = summary.get("far_control_quality", {})
    lines.append("| source | n | n_valid_far | valid_frac | mean_dist | "
                "mean_dist/radius | proj_act_diff | hidden_act_diff |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for src, d in fcq.items():
        lines.append(
            f"| {src} | {d['n_total']} | {d['n_valid_far']} | "
            f"{d['valid_far_fraction']:.2f} | "
            f"{d['mean_far_distance']:.1f} | "
            f"{d['mean_far_distance_over_radius']:.1f} | "
            f"{d['mean_far_proj_activity_diff']:.3f} | "
            f"{d['mean_far_hidden_activity_diff']:.3f} |"
        )
    lines.append("")
    lines.append("## Paired M7 (thick) vs baselines (thick)")
    lines.append("")
    grid = summary.get("comparison_grid_thick", {})
    if grid:
        lines.append("| comparison | metric | mean_a | mean_b | mean_diff | "
                    "perm p | Cliff's δ |")
        lines.append("|---|---|---|---|---|---|---|")
        for cmp_key, by_metric in grid.items():
            for m, c in by_metric.items():
                lines.append(
                    f"| {cmp_key} | {m} | {c['mean_a']:+.5f} | "
                    f"{c['mean_b']:+.5f} | {c['mean_diff']:+.5f} | "
                    f"{c['perm_p']:.3f} | {c['cliffs_delta']:+.3f} |"
                )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for p in select_interpretations(summary):
        lines.append(f"- {p}")
    return "\n".join(lines)
