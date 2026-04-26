"""M8B statistics — thick-only mechanism proportions, paired
comparisons, area/lifetime dependence, thin-vs-thick HCE.

The interpretation engine answers the M8B question directly:
"Is M7's hidden causal support boundary-mediated, interior-reservoir,
environment-coupled, whole-body, thin-only, or unresolved?"
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from observer_worlds.analysis.m7b_stats import (
    cliffs_delta,
    cluster_bootstrap_by_groups,
    permutation_test_mean_diff,
    rank_biserial,
)
from observer_worlds.detection.morphology import MORPHOLOGY_CLASSES
from observer_worlds.experiments._m8b_spatial import (
    M8B_MECHANISM_CLASSES,
    M8BCandidateResult,
)


def _by_source(results: list[M8BCandidateResult]) -> dict:
    out: dict[str, list] = {}
    for r in results: out.setdefault(r.rule_source, []).append(r)
    return out


def _by_morphology(results: list[M8BCandidateResult]) -> dict:
    out: dict[str, list] = {}
    for r in results: out.setdefault(r.morphology.morphology_class, []).append(r)
    return out


def _safe_mean(arr): return float(np.mean(arr)) if len(arr) else 0.0


def _hce(r: M8BCandidateResult) -> float:
    return float(r.region_effects["whole"].region_hidden_effect)


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def aggregate_per_source(results: list[M8BCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if r.morphology.morphology_class
                 in ("thick_candidate", "very_thick_candidate")]
        thin = [r for r in rs if r.morphology.morphology_class == "thin_candidate"]
        out[src] = {
            "n_candidates": len(rs),
            "n_thick": len(thick), "n_thin": len(thin),
            "n_unique_rules": len({r.rule_id for r in rs}),
            "n_unique_seeds": len({r.seed for r in rs}),
            "mean_observer": _safe_mean([r.observer_score for r in rs
                                        if r.observer_score is not None]),
            "mean_lifetime": _safe_mean([r.candidate_lifetime for r in rs]),
            "mean_HCE_whole": _safe_mean([_hce(r) for r in rs]),
            "mean_interior_per_cell": _safe_mean(
                [r.region_effects["interior"].region_effect_per_cell for r in rs]
            ),
            "mean_boundary_per_cell": _safe_mean(
                [r.region_effects["boundary"].region_effect_per_cell for r in rs]
            ),
            "mean_environment_per_cell": _safe_mean(
                [r.region_effects["environment"].region_effect_per_cell for r in rs]
            ),
            "mean_far_effect": _safe_mean(
                [r.far_effect.region_hidden_effect for r in rs]
            ),
            "mean_first_visible_t": _safe_mean(
                [r.first_visible_effect_time for r in rs
                 if r.first_visible_effect_time > 0]
            ),
            "mean_HCE_thick": _safe_mean([_hce(r) for r in thick]),
            "mean_HCE_thin": _safe_mean([_hce(r) for r in thin]),
            "mean_area": _safe_mean([r.candidate_area for r in rs]),
        }
    return out


def morphology_class_distribution(results: list[M8BCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        cnts = Counter(r.morphology.morphology_class for r in rs)
        total = max(len(rs), 1)
        out[src] = {
            "n_total": len(rs),
            "per_class": {
                cls: {"count": cnts.get(cls, 0),
                      "fraction": cnts.get(cls, 0) / total}
                for cls in MORPHOLOGY_CLASSES
            },
        }
    return out


def mechanism_class_distribution_thick_only(
    results: list[M8BCandidateResult],
) -> dict:
    """Per-source breakdown of mechanism labels, restricted to thick
    morphologies (where boundary/interior/env can be separated)."""
    rng = np.random.default_rng(0)
    n_boot = 500
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if r.morphology.morphology_class
                 in ("thick_candidate", "very_thick_candidate")]
        labels = [r.mechanism_label for r in thick]
        total = len(thick)
        per_class = {}
        for cls in M8B_MECHANISM_CLASSES:
            n = sum(1 for l in labels if l == cls)
            frac = n / total if total > 0 else 0.0
            if total > 0:
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    sample = [labels[i] for i in
                             rng.integers(0, total, size=total)]
                    boot[b] = sum(1 for l in sample if l == cls) / total
                ci = (float(np.quantile(boot, 0.025)),
                      float(np.quantile(boot, 0.975)))
            else:
                ci = (0.0, 0.0)
            per_class[cls] = {"count": n, "fraction": float(frac),
                              "ci_low": ci[0], "ci_high": ci[1]}
        out[src] = {"n_thick": total, "per_class": per_class}
    return out


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def compare_metric(
    results_a: list[M8BCandidateResult],
    results_b: list[M8BCandidateResult],
    *, metric_fn, n_boot: int = 1000, n_perm: int = 1000, seed: int = 0,
) -> dict:
    a = np.array([metric_fn(r) for r in results_a])
    b = np.array([metric_fn(r) for r in results_b])
    if a.size == 0 or b.size == 0:
        return {"n_a": int(a.size), "n_b": int(b.size),
                "mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "perm_p": 1.0,
                "cliffs_delta": 0.0, "rank_biserial": 0.0}
    a_grp = np.array([f"{r.rule_id}|{r.seed}" for r in results_a])
    b_grp = np.array([f"{r.rule_id}|{r.seed}" for r in results_b])
    rng = np.random.default_rng(seed)
    ua = np.unique(a_grp); ub = np.unique(b_grp)
    ia = {g: np.where(a_grp == g)[0] for g in ua}
    ib = {g: np.where(b_grp == g)[0] for g in ub}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(ua, size=ua.size, replace=True)
        sb = rng.choice(ub, size=ub.size, replace=True)
        ja = np.concatenate([ia[g] for g in sa])
        jb = np.concatenate([ib[g] for g in sb])
        diffs[i] = float(a[ja].mean() - b[jb].mean())
    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "perm_p": permutation_test_mean_diff(a, b, n_permutations=n_perm,
                                            seed=seed + 1),
        "cliffs_delta": cliffs_delta(a, b),
        "rank_biserial": rank_biserial(a, b),
    }


def m7_vs_baseline_grid_thick(results: list[M8BCandidateResult]) -> dict:
    """Build a comparison grid restricted to thick candidates."""
    by = _by_source(results)
    thick_by = {
        s: [r for r in rs if r.morphology.morphology_class
            in ("thick_candidate", "very_thick_candidate")]
        for s, rs in by.items()
    }
    if "M7_HCE_optimized" not in thick_by or not thick_by["M7_HCE_optimized"]:
        return {}
    rows_m7 = thick_by["M7_HCE_optimized"]
    metrics = {
        "HCE_whole": _hce,
        "interior_effect_per_cell":
            lambda r: r.region_effects["interior"].region_effect_per_cell,
        "boundary_effect_per_cell":
            lambda r: r.region_effects["boundary"].region_effect_per_cell,
        "environment_effect_per_cell":
            lambda r: r.region_effects["environment"].region_effect_per_cell,
        "far_effect": lambda r: r.far_effect.region_hidden_effect,
        "candidate_area": lambda r: float(r.candidate_area),
        "candidate_lifetime": lambda r: float(r.candidate_lifetime),
        "first_visible_effect_time":
            lambda r: max(r.first_visible_effect_time, 0),
    }
    out = {}
    for src in thick_by:
        if src == "M7_HCE_optimized" or not thick_by[src]: continue
        out[f"M7_thick_vs_{src}_thick"] = {}
        for m, fn in metrics.items():
            out[f"M7_thick_vs_{src}_thick"][m] = compare_metric(
                rows_m7, thick_by[src], metric_fn=fn,
            )
    return out


def thin_vs_thick_hce(results: list[M8BCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if r.morphology.morphology_class
                 in ("thick_candidate", "very_thick_candidate")]
        thin = [r for r in rs if r.morphology.morphology_class == "thin_candidate"]
        if thick and thin:
            out[src] = compare_metric(thick, thin, metric_fn=_hce, n_boot=500,
                                      n_perm=500)
        else:
            out[src] = {"n_a": len(thick), "n_b": len(thin),
                        "mean_a": _safe_mean([_hce(r) for r in thick]),
                        "mean_b": _safe_mean([_hce(r) for r in thin]),
                        "mean_diff": 0.0, "perm_p": 1.0}
    return out


def boundary_vs_interior_paired(results: list[M8BCandidateResult]) -> dict:
    """For thick candidates only, paired comparison of boundary vs
    interior per-cell effects."""
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if r.morphology.morphology_class
                 in ("thick_candidate", "very_thick_candidate")]
        if not thick:
            out[src] = {"n": 0, "mean_boundary_per_cell": 0.0,
                        "mean_interior_per_cell": 0.0, "mean_diff": 0.0,
                        "frac_boundary_dominant": 0.0,
                        "frac_interior_dominant": 0.0}
            continue
        b = np.array([r.region_effects["boundary"].region_effect_per_cell
                      for r in thick])
        i = np.array([r.region_effects["interior"].region_effect_per_cell
                      for r in thick])
        diffs = b - i
        n = len(thick)
        out[src] = {
            "n": n,
            "mean_boundary_per_cell": float(b.mean()),
            "mean_interior_per_cell": float(i.mean()),
            "mean_diff": float(diffs.mean()),
            "frac_boundary_dominant": float((b > 1.5 * i).mean()),
            "frac_interior_dominant": float((i > 1.5 * b).mean()),
            "frac_similar": float(((b <= 1.5 * i) & (i <= 1.5 * b)).mean()),
        }
    return out


def environment_vs_candidate_paired(results: list[M8BCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if r.morphology.morphology_class
                 in ("thick_candidate", "very_thick_candidate")]
        if not thick:
            out[src] = {"n": 0}
            continue
        env = np.array([r.region_effects["environment"].region_effect_per_cell
                        for r in thick])
        cand = np.array([
            max(r.region_effects["interior"].region_effect_per_cell,
                r.region_effects["boundary"].region_effect_per_cell)
            for r in thick
        ])
        out[src] = {
            "n": len(thick),
            "mean_env_per_cell": float(env.mean()),
            "mean_candidate_per_cell": float(cand.mean()),
            "frac_env_dominant": float((env > 1.5 * cand).mean()),
            "frac_candidate_dominant": float((cand > 1.5 * env).mean()),
        }
    return out


def area_lifetime_dependence(results: list[M8BCandidateResult]) -> dict:
    out = {}
    for src, rs in _by_source(results).items():
        if len(rs) < 3: out[src] = {}; continue
        hce = np.array([_hce(r) for r in rs])
        area = np.array([r.candidate_area for r in rs], dtype=np.float64)
        life = np.array([r.candidate_lifetime for r in rs], dtype=np.float64)
        cors = {}
        if hce.std() > 1e-12 and area.std() > 1e-12:
            cors["pearson_HCE_vs_area"] = float(np.corrcoef(hce, area)[0, 1])
        if hce.std() > 1e-12 and life.std() > 1e-12:
            cors["pearson_HCE_vs_lifetime"] = float(np.corrcoef(hce, life)[0, 1])
        out[src] = cors
    return out


# ---------------------------------------------------------------------------
# Top-level summary + interpretation
# ---------------------------------------------------------------------------


def m8b_full_summary(results: list[M8BCandidateResult]) -> dict:
    return {
        "n_candidates": len(results),
        "aggregates": aggregate_per_source(results),
        "morphology_distribution": morphology_class_distribution(results),
        "mechanism_distribution_thick": mechanism_class_distribution_thick_only(results),
        "comparison_grid_thick": m7_vs_baseline_grid_thick(results),
        "thin_vs_thick_hce": thin_vs_thick_hce(results),
        "boundary_vs_interior_paired": boundary_vs_interior_paired(results),
        "environment_vs_candidate_paired": environment_vs_candidate_paired(results),
        "area_lifetime_dependence": area_lifetime_dependence(results),
    }


_INTERP_BOUNDARY_GENUINE = (
    "M7 hidden support is genuinely boundary-mediated: among thick "
    "M7 candidates, per-cell boundary effect dominates per-cell interior "
    "effect, supporting a boundary-specific mechanism."
)
_INTERP_INTERIOR_RESERVOIR = (
    "M7 hidden support acts like latent internal state: thick M7 "
    "candidates show stronger per-cell interior than per-cell boundary "
    "effects."
)
_INTERP_ENVIRONMENT_COUPLED = (
    "M7 candidates are strongly coupled to hidden state in their local "
    "environment shell, not the candidate body alone."
)
_INTERP_WHOLE_BODY = (
    "M7 hidden support is whole-body rather than region-specific: "
    "thick candidates show comparable per-cell boundary and interior "
    "effects, both well above the far-region control."
)
_INTERP_THIN_ONLY = (
    "Current HCE is mainly a thin-body phenomenon; thick-candidate "
    "HCE is much weaker than thin-candidate HCE, leaving the region-"
    "specific mechanism question unresolved."
)
_INTERP_THICK_HCE_PERSISTS = (
    "HCE persists in thick morphologies where boundary, interior, and "
    "environment can be separated — the dimension-specific advantage "
    "is not an artifact of small-candidate shell-mask collapse."
)
_INTERP_INSUFFICIENT_THICK = (
    "Insufficient thick candidates to disambiguate boundary vs interior "
    "vs environment mechanisms at this run scale; consider widening "
    "the rule sweep or relaxing the area gate."
)


def select_interpretations(summary: dict) -> list[str]:
    out: list[str] = []
    mech_thick = summary.get("mechanism_distribution_thick", {})
    m7_thick = mech_thick.get("M7_HCE_optimized", {})
    n_thick = m7_thick.get("n_thick", 0)
    if n_thick == 0:
        out.append(_INTERP_INSUFFICIENT_THICK)
        return out

    fractions = {cls: d["fraction"] for cls, d
                 in m7_thick.get("per_class", {}).items()}
    dominant = max(fractions, key=fractions.get) if fractions else None
    if dominant and fractions[dominant] >= 0.4:
        if dominant == "boundary_mediated":
            out.append(_INTERP_BOUNDARY_GENUINE)
        elif dominant == "interior_reservoir":
            out.append(_INTERP_INTERIOR_RESERVOIR)
        elif dominant == "environment_coupled":
            out.append(_INTERP_ENVIRONMENT_COUPLED)
        elif dominant == "whole_body_hidden_support":
            out.append(_INTERP_WHOLE_BODY)

    # Thin-vs-thick comparison.
    tvt = summary.get("thin_vs_thick_hce", {}).get("M7_HCE_optimized", {})
    mean_thick = tvt.get("mean_a", 0.0)
    mean_thin = tvt.get("mean_b", 0.0)
    if mean_thin > 2 * max(mean_thick, 1e-9) and mean_thin > 0.02:
        out.append(_INTERP_THIN_ONLY)
    elif mean_thick > 0.02:
        out.append(_INTERP_THICK_HCE_PERSISTS)

    if not out:
        out.append("Mixed result; no single mechanism dominates among thick M7 candidates.")
    return out


def render_m8b_summary_md(summary: dict) -> str:
    lines = ["# M8B — Spatial Mechanism Disambiguation", ""]
    lines.append(f"- N candidates measured: {summary['n_candidates']}")
    lines.append("")
    lines.append("## Per-source aggregates")
    lines.append("")
    aggs = summary.get("aggregates", {})
    if aggs:
        lines.append("| source | n | n_thick | mean_obs | mean_life | mean_area | "
                    "mean_HCE_whole | int_pc | bnd_pc | env_pc | far |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for src, a in aggs.items():
            lines.append(
                f"| {src} | {a['n_candidates']} | {a['n_thick']} | "
                f"{a['mean_observer']:+.3f} | {a['mean_lifetime']:.0f} | "
                f"{a['mean_area']:.0f} | {a['mean_HCE_whole']:+.4f} | "
                f"{a['mean_interior_per_cell']:+.5f} | "
                f"{a['mean_boundary_per_cell']:+.5f} | "
                f"{a['mean_environment_per_cell']:+.5f} | "
                f"{a['mean_far_effect']:+.4f} |"
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
    lines.append("## Boundary vs interior paired (thick only)")
    lines.append("")
    bvi = summary.get("boundary_vs_interior_paired", {})
    if bvi:
        lines.append("| source | n_thick | bnd_pc | int_pc | diff | "
                    "%bnd-dom | %int-dom | %similar |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for src, d in bvi.items():
            lines.append(
                f"| {src} | {d.get('n', 0)} | "
                f"{d.get('mean_boundary_per_cell', 0):+.5f} | "
                f"{d.get('mean_interior_per_cell', 0):+.5f} | "
                f"{d.get('mean_diff', 0):+.5f} | "
                f"{d.get('frac_boundary_dominant', 0):.2f} | "
                f"{d.get('frac_interior_dominant', 0):.2f} | "
                f"{d.get('frac_similar', 0):.2f} |"
            )
        lines.append("")
    lines.append("## M7 (thick) vs baselines (thick) on M8B metrics")
    lines.append("")
    grid = summary.get("comparison_grid_thick", {})
    if grid:
        lines.append("| comparison | metric | mean_a | mean_b | mean_diff | "
                    "95% CI | perm p | Cliff's δ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cmp_key, by_metric in grid.items():
            for m, c in by_metric.items():
                lines.append(
                    f"| {cmp_key} | {m} | "
                    f"{c['mean_a']:+.5f} | {c['mean_b']:+.5f} | "
                    f"{c['mean_diff']:+.5f} | "
                    f"[{c['ci_low']:+.5f}, {c['ci_high']:+.5f}] | "
                    f"{c['perm_p']:.4f} | {c['cliffs_delta']:+.3f} |"
                )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for p in select_interpretations(summary):
        lines.append(f"- {p}")
    return "\n".join(lines)
