"""M6B — paired statistics with grouped bootstrap and primary-quantity reporting.

Distinct from M4B's stats module:
  * Records *raw* future divergence + sham-subtracted + far-control-subtracted
    quantities so HCE-as-ratio can't dominate the headline.
  * Bootstrap resamples by **rule** (and optionally by seed) so the unit of
    inference matches the unit of generalization. Per-row resampling would
    give artificially tight CIs because rows within a rule are correlated.
  * Tracks per-horizon trends, per-candidate-property correlations.

Primary quantities for the headline:
  - mean coherent local future divergence at horizon H
  - paired delta vs sham (numerical floor)
  - paired delta vs far_hidden control (localization)
  - paired delta vs per_step_hidden_shuffled (rule-track contrast)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Iterable

import numpy as np


PRIMARY_INTERVENTIONS_LOCAL = (
    "hidden_invisible_local",
    "one_time_scramble_local",
    "fiber_replacement_local",
)


# ---------------------------------------------------------------------------
# Row → tidy ndarray helpers
# ---------------------------------------------------------------------------


def rows_to_arrays(rows, *, fields: Iterable[str]) -> dict[str, np.ndarray]:
    """Convert a list of M6BRow into a dict of named numpy arrays.

    Useful for slicing by condition / intervention / horizon downstream.
    """
    out: dict[str, list] = {f: [] for f in fields}
    for r in rows:
        for f in fields:
            out[f].append(getattr(r, f))
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Grouped bootstrap (resample by rule, then optionally by seed within rule)
# ---------------------------------------------------------------------------


def grouped_bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean + bootstrap CI when observations are grouped (e.g. by rule).

    Resamples whole groups with replacement, then computes the mean across
    all values in the resampled groups.  This is the cluster-bootstrap
    that respects within-group correlation.

    Returns ``(mean, ci_low, ci_high)``.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = unique_groups.size
    if n_groups == 0:
        return 0.0, 0.0, 0.0
    # Index into values by group.
    idx_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        idxs = np.concatenate([idx_by_group[g] for g in sampled])
        boot_means[b] = float(values[idxs].mean())
    alpha = 1.0 - confidence
    return (
        float(values.mean()),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1.0 - alpha / 2)),
    )


def sign_test_p(values: np.ndarray) -> float:
    """Two-sided sign test against zero. ``values`` is a vector of paired
    differences (typically already mean-per-rule)."""
    n = int(values.size)
    if n == 0:
        return 1.0
    n_pos = int((values > 0).sum())
    upper = sum(comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * min(upper, 1 - upper + (1 / 2 ** n))))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = P(a > b) - P(a < b).  Independent samples."""
    if a.size == 0 or b.size == 0:
        return 0.0
    A = a[:, None]
    B = b[None, :]
    return float(((A > B).sum() - (A < B).sum()) / (a.size * b.size))


# ---------------------------------------------------------------------------
# Headline aggregates per (condition, intervention)
# ---------------------------------------------------------------------------


@dataclass
class ConditionInterventionAggregate:
    condition: str
    intervention_type: str
    horizon: int
    n_rows: int
    n_rules: int
    mean_initial_projection_delta: float
    mean_future_projection_divergence: float
    mean_local_future_divergence: float
    mean_hidden_vs_visible_ratio: float
    mean_hidden_vs_sham_delta: float
    mean_hidden_vs_far_delta: float
    mean_hidden_causal_dependence: float
    mean_survival_delta: float
    mean_trajectory_divergence: float
    fraction_future_div_gt_zero: float
    # Cluster-bootstrap CI on mean future divergence (resampled by rule).
    bootstrap_ci_low_future: float
    bootstrap_ci_high_future: float
    # Cluster-bootstrap CI on mean hidden_vs_sham_delta.
    bootstrap_ci_low_vs_sham: float
    bootstrap_ci_high_vs_sham: float


def aggregate_by_condition_intervention_horizon(
    rows, *, n_boot: int = 2000, seed: int = 0,
) -> list[ConditionInterventionAggregate]:
    """For each (condition, intervention_type, horizon), compute the
    headline aggregates with cluster-bootstrap CIs."""
    keys = set()
    for r in rows:
        keys.add((r.condition, r.intervention_type, r.horizon))
    out: list[ConditionInterventionAggregate] = []
    for cond, intv, h in sorted(keys):
        sub = [r for r in rows if r.condition == cond
               and r.intervention_type == intv and r.horizon == h]
        if not sub:
            continue
        future = np.array([r.future_projection_divergence for r in sub])
        local = np.array([r.local_future_divergence for r in sub])
        init = np.array([r.initial_projection_delta for r in sub])
        ratio = np.array([r.hidden_vs_visible_ratio for r in sub])
        vs_sham = np.array([r.hidden_vs_sham_delta for r in sub])
        vs_far = np.array([r.hidden_vs_far_delta for r in sub])
        hce = np.array([r.hidden_causal_dependence for r in sub])
        surv_delta = np.array([r.survival_delta for r in sub])
        traj = np.array([r.trajectory_divergence for r in sub])
        groups = np.array([r.rule_id for r in sub])
        n_rules = int(np.unique(groups).size)
        m_fut, lo_fut, hi_fut = grouped_bootstrap_mean_ci(
            future, groups, n_boot=n_boot, seed=seed
        )
        _, lo_sham, hi_sham = grouped_bootstrap_mean_ci(
            vs_sham, groups, n_boot=n_boot, seed=seed + 1
        )
        out.append(ConditionInterventionAggregate(
            condition=cond, intervention_type=intv, horizon=h,
            n_rows=len(sub), n_rules=n_rules,
            mean_initial_projection_delta=float(init.mean()),
            mean_future_projection_divergence=float(future.mean()),
            mean_local_future_divergence=float(local.mean()),
            mean_hidden_vs_visible_ratio=float(ratio.mean()),
            mean_hidden_vs_sham_delta=float(vs_sham.mean()),
            mean_hidden_vs_far_delta=float(vs_far.mean()),
            mean_hidden_causal_dependence=float(hce.mean()),
            mean_survival_delta=float(surv_delta.mean()),
            mean_trajectory_divergence=float(traj.mean()),
            fraction_future_div_gt_zero=float((future > 0).mean()),
            bootstrap_ci_low_future=lo_fut, bootstrap_ci_high_future=hi_fut,
            bootstrap_ci_low_vs_sham=lo_sham, bootstrap_ci_high_vs_sham=hi_sham,
        ))
    return out


# ---------------------------------------------------------------------------
# Paired comparisons (collapse rows to per-(rule, candidate) means before
# pairing across interventions/conditions)
# ---------------------------------------------------------------------------


@dataclass
class PairedComparison:
    name: str
    intervention_a: str
    intervention_b: str
    condition: str
    horizon: int
    n_paired: int
    mean_diff_a_minus_b: float
    median_diff: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    sign_test_p: float
    n_a_wins: int
    n_b_wins: int


def _per_candidate_mean(
    rows, *, condition: str, intervention: str, horizon: int, field: str,
) -> dict[tuple[str, int, int], float]:
    """Collapse rows to one value per (rule_id, seed, candidate_id) by
    averaging across replicates."""
    by_key: dict[tuple[str, int, int], list[float]] = {}
    for r in rows:
        if (r.condition == condition and r.intervention_type == intervention
                and r.horizon == horizon):
            key = (r.rule_id, r.seed, r.candidate_id)
            by_key.setdefault(key, []).append(getattr(r, field))
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def compare_paired(
    rows,
    *,
    name: str,
    intervention_a: str,
    intervention_b: str,
    condition: str = "coherent_4d",
    horizon: int,
    field: str = "future_projection_divergence",
    n_boot: int = 2000,
    seed: int = 0,
) -> PairedComparison:
    """Per (rule, seed, candidate) compare two interventions on a chosen
    quantity.  Pairs by triple-key.  Cluster-bootstrap CI grouped by rule.
    """
    a_means = _per_candidate_mean(
        rows, condition=condition, intervention=intervention_a,
        horizon=horizon, field=field,
    )
    b_means = _per_candidate_mean(
        rows, condition=condition, intervention=intervention_b,
        horizon=horizon, field=field,
    )
    common = sorted(set(a_means).intersection(b_means))
    if not common:
        return PairedComparison(
            name=name, intervention_a=intervention_a,
            intervention_b=intervention_b, condition=condition,
            horizon=horizon, n_paired=0,
            mean_diff_a_minus_b=0.0, median_diff=0.0,
            bootstrap_ci_low=0.0, bootstrap_ci_high=0.0,
            sign_test_p=1.0, n_a_wins=0, n_b_wins=0,
        )
    diffs = np.array([a_means[k] - b_means[k] for k in common])
    rule_ids = np.array([k[0] for k in common])
    _, lo, hi = grouped_bootstrap_mean_ci(diffs, rule_ids, n_boot=n_boot, seed=seed)
    return PairedComparison(
        name=name, intervention_a=intervention_a,
        intervention_b=intervention_b, condition=condition,
        horizon=horizon, n_paired=len(common),
        mean_diff_a_minus_b=float(diffs.mean()),
        median_diff=float(np.median(diffs)),
        bootstrap_ci_low=lo, bootstrap_ci_high=hi,
        sign_test_p=sign_test_p(diffs),
        n_a_wins=int((diffs > 0).sum()),
        n_b_wins=int((diffs < 0).sum()),
    )


def standard_paired_comparisons(
    rows, *, horizon: int, n_boot: int = 2000, seed: int = 0,
) -> dict[str, PairedComparison]:
    """The set of paired comparisons the M6B spec asks for."""
    out: dict[str, PairedComparison] = {}
    out["coh_local_vs_sham"] = compare_paired(
        rows, name="coh local hidden vs sham",
        intervention_a="hidden_invisible_local",
        intervention_b="sham", horizon=horizon,
        n_boot=n_boot, seed=seed,
    )
    out["coh_local_vs_far"] = compare_paired(
        rows, name="coh local hidden vs far hidden",
        intervention_a="hidden_invisible_local",
        intervention_b="hidden_invisible_far", horizon=horizon,
        field="local_future_divergence",  # localization is about candidate-region
        n_boot=n_boot, seed=seed,
    )
    out["coh_local_vs_visible"] = compare_paired(
        rows, name="coh local hidden vs visible_match_count",
        intervention_a="hidden_invisible_local",
        intervention_b="visible_match_count", horizon=horizon,
        n_boot=n_boot, seed=seed,
    )
    out["one_time_vs_hidden_invisible"] = compare_paired(
        rows, name="one_time_scramble vs hidden_invisible (both local)",
        intervention_a="one_time_scramble_local",
        intervention_b="hidden_invisible_local", horizon=horizon,
        n_boot=n_boot, seed=seed,
    )
    out["fiber_repl_vs_hidden_invisible"] = compare_paired(
        rows, name="fiber_replacement vs hidden_invisible (both local)",
        intervention_a="fiber_replacement_local",
        intervention_b="hidden_invisible_local", horizon=horizon,
        n_boot=n_boot, seed=seed,
    )
    return out


# ---------------------------------------------------------------------------
# Cross-condition comparison (coherent vs per-step shuffled, on the same
# intervention).  Cluster-bootstrap by rule with rank-pairing fallback.
# ---------------------------------------------------------------------------


def compare_conditions_on_intervention(
    rows, *, intervention: str, horizon: int,
    n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Per rule, compute the mean of `future_projection_divergence` under
    `intervention` for coherent vs per_step_shuffled. Compare via
    paired-by-rule sign test + cluster bootstrap."""
    by_rule_coh: dict[str, list[float]] = {}
    by_rule_shuf: dict[str, list[float]] = {}
    for r in rows:
        if r.intervention_type != intervention or r.horizon != horizon:
            continue
        if r.condition == "coherent_4d":
            by_rule_coh.setdefault(r.rule_id, []).append(r.future_projection_divergence)
        elif r.condition == "per_step_hidden_shuffled_4d":
            by_rule_shuf.setdefault(r.rule_id, []).append(r.future_projection_divergence)
    common_rules = sorted(set(by_rule_coh).intersection(by_rule_shuf))
    if not common_rules:
        return {
            "intervention": intervention, "horizon": horizon,
            "n_rules": 0, "mean_diff_coh_minus_shuf": 0.0,
            "bootstrap_ci_low": 0.0, "bootstrap_ci_high": 0.0,
            "sign_test_p": 1.0, "n_coherent_wins": 0, "n_shuffled_wins": 0,
        }
    diffs = np.array([
        np.mean(by_rule_coh[r]) - np.mean(by_rule_shuf[r]) for r in common_rules
    ])
    rule_groups = np.array(common_rules)
    _, lo, hi = grouped_bootstrap_mean_ci(diffs, rule_groups, n_boot=n_boot, seed=seed)
    return {
        "intervention": intervention, "horizon": horizon,
        "n_rules": len(common_rules),
        "mean_diff_coh_minus_shuf": float(diffs.mean()),
        "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
        "sign_test_p": sign_test_p(diffs),
        "n_coherent_wins": int((diffs > 0).sum()),
        "n_shuffled_wins": int((diffs < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Win rates (per-row probabilities over candidates × replicates × rules)
# ---------------------------------------------------------------------------


def win_rates(rows, horizon: int) -> dict[str, dict[str, float]]:
    """Compute the win-rate panel from the spec:
      P(coherent local hidden > far hidden)
      P(coherent local hidden > sham)
      P(coherent HCE > per-step shuffled HCE)
    """
    def _coh_per_cand(intv: str, field: str = "future_projection_divergence"):
        return _per_candidate_mean(
            rows, condition="coherent_4d", intervention=intv,
            horizon=horizon, field=field,
        )

    out: dict[str, dict[str, float]] = {}
    a = _coh_per_cand("hidden_invisible_local", field="local_future_divergence")
    b = _coh_per_cand("hidden_invisible_far", field="local_future_divergence")
    common = set(a) & set(b)
    if common:
        diffs = np.array([a[k] - b[k] for k in common])
        out["coh_local_vs_far"] = {
            "p_a_gt_b": float((diffs > 0).mean()),
            "p_a_eq_b": float((diffs == 0).mean()),
            "n": int(len(common)),
        }
    a = _coh_per_cand("hidden_invisible_local")
    b = _coh_per_cand("sham")
    common = set(a) & set(b)
    if common:
        diffs = np.array([a[k] - b[k] for k in common])
        out["coh_local_vs_sham"] = {
            "p_a_gt_b": float((diffs > 0).mean()),
            "p_a_eq_b": float((diffs == 0).mean()),
            "n": int(len(common)),
        }
    cond_cmp = compare_conditions_on_intervention(
        rows, intervention="hidden_invisible_local", horizon=horizon,
    )
    n = cond_cmp["n_rules"]
    if n > 0:
        out["coh_HCE_vs_shuf_HCE"] = {
            "p_a_gt_b": cond_cmp["n_coherent_wins"] / n,
            "p_a_eq_b": (n - cond_cmp["n_coherent_wins"]
                        - cond_cmp["n_shuffled_wins"]) / n,
            "n": n,
        }
    return out


# ---------------------------------------------------------------------------
# Horizon trends
# ---------------------------------------------------------------------------


def horizon_trends(
    rows, *, condition: str = "coherent_4d",
    intervention: str = "hidden_invisible_local",
) -> dict[int, dict[str, float]]:
    """Mean future divergence (and other primary quantities) per horizon."""
    horizons = sorted({r.horizon for r in rows})
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        sub = [r for r in rows if r.condition == condition
               and r.intervention_type == intervention and r.horizon == h]
        if not sub:
            continue
        out[h] = {
            "n_rows": len(sub),
            "mean_future_div": float(np.mean([r.future_projection_divergence for r in sub])),
            "mean_local_div": float(np.mean([r.local_future_divergence for r in sub])),
            "mean_vs_sham": float(np.mean([r.hidden_vs_sham_delta for r in sub])),
            "mean_vs_far": float(np.mean([r.hidden_vs_far_delta for r in sub])),
            "mean_HCE": float(np.mean([r.hidden_causal_dependence for r in sub])),
        }
    return out


# ---------------------------------------------------------------------------
# Candidate-property correlations
# ---------------------------------------------------------------------------


def candidate_property_correlations(
    rows, *, intervention: str = "hidden_invisible_local",
    condition: str = "coherent_4d", horizon: int,
) -> dict[str, float]:
    """Pearson correlations of future_projection_divergence with
    candidate properties (observer_score, lifetime, area).  Drops rows
    with missing observer_score for that correlation."""
    sub = [r for r in rows if r.condition == condition
           and r.intervention_type == intervention and r.horizon == horizon]
    if not sub:
        return {}
    out = {}
    fut = np.array([r.future_projection_divergence for r in sub])
    age = np.array([r.candidate_lifetime for r in sub], dtype=np.float64)
    area = np.array([r.candidate_area for r in sub], dtype=np.float64)
    if age.std() > 1e-12:
        out["pearson_future_vs_lifetime"] = float(np.corrcoef(fut, age)[0, 1])
    if area.std() > 1e-12:
        out["pearson_future_vs_area"] = float(np.corrcoef(fut, area)[0, 1])
    obs_pairs = [(r.observer_score, r.future_projection_divergence)
                 for r in sub if r.observer_score is not None]
    if len(obs_pairs) >= 3:
        obs = np.array([p[0] for p in obs_pairs])
        fut2 = np.array([p[1] for p in obs_pairs])
        if obs.std() > 1e-12:
            out["pearson_future_vs_observer_score"] = float(np.corrcoef(obs, fut2)[0, 1])
    # Per rule_source if mixed.
    sources = {r.rule_source for r in sub}
    if len(sources) > 1:
        for src in sources:
            sub2 = [r for r in sub if r.rule_source == src]
            if sub2:
                out[f"mean_future_div_{src}"] = float(
                    np.mean([r.future_projection_divergence for r in sub2])
                )
    return out


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------


def m6b_full_summary(
    rows, *, horizons: list[int],
    n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Return a JSON-serializable nested dict combining all the M6B stats."""
    summary = {
        "n_rows": len(rows),
        "horizons": list(horizons),
        "headline_horizon": horizons[len(horizons) // 2],   # middle horizon
    }
    H = summary["headline_horizon"]
    aggs = aggregate_by_condition_intervention_horizon(rows, n_boot=n_boot, seed=seed)
    summary["aggregates"] = [a.__dict__ for a in aggs]
    paired = standard_paired_comparisons(rows, horizon=H, n_boot=n_boot, seed=seed)
    summary["paired_comparisons"] = {k: v.__dict__ for k, v in paired.items()}
    summary["condition_compare"] = {
        intv: compare_conditions_on_intervention(
            rows, intervention=intv, horizon=H, n_boot=n_boot, seed=seed,
        )
        for intv in PRIMARY_INTERVENTIONS_LOCAL
    }
    summary["win_rates"] = win_rates(rows, horizon=H)
    summary["horizon_trends"] = horizon_trends(rows)
    summary["candidate_property_correlations"] = candidate_property_correlations(
        rows, horizon=H,
    )
    return summary


# ---------------------------------------------------------------------------
# Markdown rendering + interpretation
# ---------------------------------------------------------------------------


_INTERP_HCE = (
    "Coherent projected candidates show hidden-causal dependence: invisible "
    "hidden-state changes alter future projected dynamics."
)
_INTERP_TEMPORAL = (
    "Hidden temporal coherence matters, but the specific instantaneous "
    "hidden arrangement may be less important than ongoing coherent dynamics."
)
_INTERP_MICROSTATE = (
    "The specific hidden microstate under the candidate carries causal "
    "information not present in the 2D projection."
)
_INTERP_NOT_LOCAL = (
    "The effect is not candidate-local; hidden perturbations may be globally "
    "destabilizing rather than self-specific."
)
_INTERP_OBS_ALIGNED = (
    "Generic observer-likeness and hidden-causal dependence are aligned in "
    "this rule family."
)
_INTERP_OBS_DISTINCT = (
    "Hidden-causal dependence is a distinct dimension-specific property not "
    "captured by the current observer_score."
)
_INTERP_RULE_REGIME = (
    "HCE depends on rule-selection regime."
)


def _significant_positive(comp: PairedComparison | dict) -> bool:
    if isinstance(comp, PairedComparison):
        d = comp
        ci_low = d.bootstrap_ci_low
        p = d.sign_test_p
        diff = d.mean_diff_a_minus_b
    else:
        ci_low = float(comp.get("bootstrap_ci_low", 0.0))
        p = float(comp.get("sign_test_p", 1.0))
        diff = float(comp.get("mean_diff_a_minus_b",
                              comp.get("mean_diff_coh_minus_shuf", 0.0)))
    return diff > 0 and ci_low > 0


def render_m6b_summary_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# M6B — Hidden-Causal Dependence Replication")
    lines.append("")
    lines.append(f"- N rows: {summary['n_rows']}")
    lines.append(f"- Horizons: {summary['horizons']}")
    lines.append(f"- Headline horizon: {summary['headline_horizon']}")
    H = summary["headline_horizon"]

    lines.append("")
    lines.append("## Per-condition × intervention × horizon aggregates")
    lines.append("")
    lines.append("| condition | intervention | h | n_rows | n_rules | "
                 "init_d | future_d | ci_low | ci_high | "
                 "vs_sham | vs_far | survival_d |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in summary["aggregates"]:
        lines.append(
            f"| {a['condition']} | {a['intervention_type']} | "
            f"{a['horizon']} | {a['n_rows']} | {a['n_rules']} | "
            f"{a['mean_initial_projection_delta']:+.4f} | "
            f"{a['mean_future_projection_divergence']:+.4f} | "
            f"{a['bootstrap_ci_low_future']:+.4f} | "
            f"{a['bootstrap_ci_high_future']:+.4f} | "
            f"{a['mean_hidden_vs_sham_delta']:+.4f} | "
            f"{a['mean_hidden_vs_far_delta']:+.4f} | "
            f"{a['mean_survival_delta']:+.2f} |"
        )

    lines.append("")
    lines.append(f"## Paired comparisons at headline horizon = {H}")
    lines.append("")
    lines.append("| name | n | mean_diff | 95% CI | sign-p | a wins | b wins |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, p in summary["paired_comparisons"].items():
        lines.append(
            f"| {p['name']} | {p['n_paired']} | "
            f"{p['mean_diff_a_minus_b']:+.4f} | "
            f"[{p['bootstrap_ci_low']:+.4f}, {p['bootstrap_ci_high']:+.4f}] | "
            f"{p['sign_test_p']:.4f} | {p['n_a_wins']} | {p['n_b_wins']} |"
        )

    lines.append("")
    lines.append(f"## Coherent vs per-step shuffled, per intervention "
                 f"(horizon={H})")
    lines.append("")
    lines.append("| intervention | n_rules | diff_coh-shuf | 95% CI | sign-p | coh wins | shuf wins |")
    lines.append("|---|---|---|---|---|---|---|")
    for intv, c in summary["condition_compare"].items():
        lines.append(
            f"| {intv} | {c['n_rules']} | "
            f"{c['mean_diff_coh_minus_shuf']:+.4f} | "
            f"[{c['bootstrap_ci_low']:+.4f}, {c['bootstrap_ci_high']:+.4f}] | "
            f"{c['sign_test_p']:.4f} | {c['n_coherent_wins']} | "
            f"{c['n_shuffled_wins']} |"
        )

    lines.append("")
    lines.append("## Win rates")
    lines.append("")
    for k, v in summary["win_rates"].items():
        lines.append(f"- {k}: P(a > b) = {v['p_a_gt_b']:.3f} (N={v['n']})")

    lines.append("")
    lines.append("## Horizon trends (coherent_4d, hidden_invisible_local)")
    lines.append("")
    lines.append("| horizon | n_rows | future_div | local_div | vs_sham | vs_far | HCE |")
    lines.append("|---|---|---|---|---|---|---|")
    for h, v in sorted(summary["horizon_trends"].items()):
        lines.append(
            f"| {h} | {v['n_rows']} | "
            f"{v['mean_future_div']:+.4f} | {v['mean_local_div']:+.4f} | "
            f"{v['mean_vs_sham']:+.4f} | {v['mean_vs_far']:+.4f} | "
            f"{v['mean_HCE']:+.3f} |"
        )

    lines.append("")
    lines.append("## Candidate-property correlations "
                 f"(horizon={H}, hidden_invisible_local)")
    lines.append("")
    for k, v in summary["candidate_property_correlations"].items():
        lines.append(f"- {k} = {v:+.4f}")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    paragraphs = _select_interpretations(summary)
    for p in paragraphs:
        lines.append(f"- {p}")
    return "\n".join(lines)


def _select_interpretations(summary: dict) -> list[str]:
    """Decide which canonical sentences to include."""
    out: list[str] = []
    paired = summary["paired_comparisons"]
    cond_cmp = summary["condition_compare"]
    cors = summary["candidate_property_correlations"]

    coh_vs_sham = paired.get("coh_local_vs_sham", {})
    coh_vs_far = paired.get("coh_local_vs_far", {})
    one_time_vs_hi = paired.get("one_time_vs_hidden_invisible", {})

    # Rule 1: HCE if vs_sham is significantly positive.
    if isinstance(coh_vs_sham, dict) and _significant_positive(coh_vs_sham):
        out.append(_INTERP_HCE)

    # Rule 2: temporal coherence specific.
    coh_vs_shuf_hi = cond_cmp.get("hidden_invisible_local", {})
    if (_significant_positive(coh_vs_shuf_hi)
            and isinstance(one_time_vs_hi, dict)
            and not _significant_positive(one_time_vs_hi)):
        out.append(_INTERP_TEMPORAL)

    # Rule 3: microstate matters (one_time_scramble does much).
    if isinstance(one_time_vs_hi, dict) and _significant_positive(one_time_vs_hi):
        out.append(_INTERP_MICROSTATE)

    # Rule 4: not candidate-local (far ~ local).
    if (isinstance(coh_vs_far, dict) and coh_vs_far.get("n_paired", 0) > 0
            and not _significant_positive(coh_vs_far)):
        out.append(_INTERP_NOT_LOCAL)

    # Rule 5/6: observer_score correlation.
    obs_corr = cors.get("pearson_future_vs_observer_score")
    if obs_corr is not None:
        if abs(obs_corr) > 0.3:
            out.append(_INTERP_OBS_ALIGNED + f" (Pearson r = {obs_corr:+.2f})")
        else:
            out.append(_INTERP_OBS_DISTINCT + f" (Pearson r = {obs_corr:+.2f})")

    # Rule 7: rule_source dependence.
    src_keys = [k for k in cors if k.startswith("mean_future_div_")]
    if len(src_keys) >= 2:
        vals = [cors[k] for k in src_keys]
        if max(vals) > 0 and min(vals) <= 0:
            out.append(_INTERP_RULE_REGIME)

    if not out:
        out.append("Mixed result; no strong directional conclusion.")
    return out
