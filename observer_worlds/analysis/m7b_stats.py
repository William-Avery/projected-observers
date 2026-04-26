"""M7B — production-scale statistical analysis with stronger guarantees.

What this module adds beyond M6B / M6C / M7 stats:

  * **Three cluster-bootstrap levels**: by rule_id, by seed, by
    (rule_id, seed). The first respects between-rule heterogeneity;
    the second between-seed heterogeneity; the third treats the joint
    factor as the unit of resampling.
  * **Multiple effect-size measures** per primary comparison: Cliff's
    delta, rank-biserial (signed-rank-style), and Cohen's d as a
    secondary check.
  * **Generalization gap** between train / validation / production-test
    fitness for matched M7 rules.
  * **Success-criterion classifier** that maps a comparison summary to
    one of the four canonical interpretation paragraphs in the M7B
    spec.

All bootstrap routines accept a numpy RNG so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


# ---------------------------------------------------------------------------
# Cluster bootstraps
# ---------------------------------------------------------------------------


def cluster_bootstrap_by_groups(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Cluster-bootstrap (mean, ci_low, ci_high).

    Resamples whole groups with replacement; computes the requested
    statistic over all values in the resampled groups.

    ``statistic``: "mean" or "median".
    """
    if values.size == 0 or groups.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique}
    n_groups = unique.size
    if n_groups == 0:
        return 0.0, 0.0, 0.0

    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique, size=n_groups, replace=True)
        idxs = np.concatenate([idx_by_group[g] for g in sampled])
        if statistic == "median":
            boot[b] = float(np.median(values[idxs]))
        else:
            boot[b] = float(values[idxs].mean())
    point = float(values.mean()) if statistic == "mean" else float(np.median(values))
    alpha = 1 - confidence
    return (
        point,
        float(np.quantile(boot, alpha / 2)),
        float(np.quantile(boot, 1 - alpha / 2)),
    )


def multi_level_bootstrap(
    values: np.ndarray,
    rule_ids: np.ndarray,
    seeds: np.ndarray,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> dict:
    """Returns CIs at three cluster levels: by rule, by seed, by both.

    "By both" treats each (rule_id, seed) as the unit of resampling.
    """
    if values.size == 0:
        return {"by_rule": (0.0, 0.0, 0.0), "by_seed": (0.0, 0.0, 0.0),
                "by_rule_and_seed": (0.0, 0.0, 0.0)}
    pair_groups = np.array([f"{r}|{s}" for r, s in zip(rule_ids, seeds)])
    return {
        "by_rule": cluster_bootstrap_by_groups(
            values, rule_ids, n_boot=n_boot, confidence=confidence, seed=rng_seed,
        ),
        "by_seed": cluster_bootstrap_by_groups(
            values, seeds, n_boot=n_boot, confidence=confidence, seed=rng_seed + 1,
        ),
        "by_rule_and_seed": cluster_bootstrap_by_groups(
            values, pair_groups, n_boot=n_boot, confidence=confidence,
            seed=rng_seed + 2,
        ),
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) − P(a < b). Range [-1, +1]; +1 means a always beats b."""
    if a.size == 0 or b.size == 0:
        return 0.0
    A = a[:, None]; B = b[None, :]
    return float(((A > B).sum() - (A < B).sum()) / (a.size * b.size))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation = 2 * (mean rank of a / (n_a + n_b + 1)) - 1.

    For two independent samples, equivalent to Cliff's delta in [-1, +1].
    Implemented via Mann-Whitney U formula:
        rb = 1 - 2 * U_b / (n_a * n_b)
    where U_b is U for b winning ties counted as 0.5.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    A = a[:, None]; B = b[None, :]
    U_a = float((A > B).sum() + 0.5 * (A == B).sum())
    rb = 2 * U_a / (a.size * b.size) - 1
    return float(rb)


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples (pooled sd).

    Reported as secondary; ranks-based measures are primary because
    M7B distributions can be heavy-tailed.
    """
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = float(a.std(ddof=1))
    sb = float(b.std(ddof=1))
    pooled = float(
        np.sqrt(((a.size - 1) * sa ** 2 + (b.size - 1) * sb ** 2) /
                (a.size + b.size - 2))
    )
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def permutation_test_mean_diff(
    a: np.ndarray, b: np.ndarray,
    *, n_permutations: int = 2000, seed: int = 0,
) -> float:
    """Two-sided permutation test on the difference of means."""
    if a.size == 0 or b.size == 0:
        return 1.0
    observed = abs(float(a.mean()) - float(b.mean()))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    n_a = a.size
    n_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        if abs(perm_a.mean() - perm_b.mean()) >= observed - 1e-12:
            n_extreme += 1
    return float((n_extreme + 1) / (n_permutations + 1))


def sign_test_p(diffs: np.ndarray) -> float:
    n = int(diffs.size)
    if n == 0:
        return 1.0
    n_pos = int((diffs > 0).sum())
    upper = sum(comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * min(upper, 1 - upper + (1 / 2 ** n))))


# ---------------------------------------------------------------------------
# Two-source comparison summary
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Full statistical comparison of two sources on one metric."""

    metric: str
    source_a: str
    source_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    # Three-level cluster bootstrap of mean(a) - mean(b).
    bootstrap_by_rule: tuple[float, float, float]      # (mean_diff, lo, hi)
    bootstrap_by_seed: tuple[float, float, float]
    bootstrap_by_rule_and_seed: tuple[float, float, float]
    # Effect sizes.
    cliffs_delta: float
    rank_biserial: float
    cohens_d: float
    # Permutation p (mean diff).
    permutation_p: float
    # Win rate of A vs B at the candidate level.
    win_rate_a: float


def _per_pair_means(rows, *, source: str, metric: str, horizon: int):
    """For one source, collapse rows to one value per (rule_id, seed,
    candidate_id) by averaging across replicates / horizons (filtered
    to ``horizon``)."""
    by_key: dict = {}
    for r in rows:
        if r.horizon != horizon: continue
        key = (r.rule_id, r.seed, r.candidate_id)
        v = getattr(r, metric, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        by_key.setdefault(key, []).append(float(v))
    return {k: float(np.mean(vs)) for k, vs in by_key.items() if vs}


def compare_sources(
    rows_a: list, rows_b: list,
    *,
    metric: str,
    source_a_name: str = "A",
    source_b_name: str = "B",
    horizon: int,
    n_boot: int = 2000,
    n_permutations: int = 2000,
    seed: int = 0,
) -> ComparisonResult:
    """Compare distributions of `metric` between two sources at one horizon.

    Source A and B contain *different* rule_ids (separate evolutions),
    so we don't pair candidate-by-candidate. We compute per-(rule, seed,
    candidate) means within each source, then run independent-samples
    bootstrap CIs at three cluster levels and effect sizes.
    """
    a_means = _per_pair_means(rows_a, source=source_a_name, metric=metric,
                              horizon=horizon)
    b_means = _per_pair_means(rows_b, source=source_b_name, metric=metric,
                              horizon=horizon)
    if not a_means or not b_means:
        return ComparisonResult(
            metric=metric, source_a=source_a_name, source_b=source_b_name,
            n_a=len(a_means), n_b=len(b_means),
            mean_a=0.0, mean_b=0.0, mean_diff=0.0,
            bootstrap_by_rule=(0.0, 0.0, 0.0),
            bootstrap_by_seed=(0.0, 0.0, 0.0),
            bootstrap_by_rule_and_seed=(0.0, 0.0, 0.0),
            cliffs_delta=0.0, rank_biserial=0.0, cohens_d=0.0,
            permutation_p=1.0, win_rate_a=0.5,
        )
    a_keys = list(a_means.keys()); b_keys = list(b_means.keys())
    a_vals = np.array([a_means[k] for k in a_keys])
    b_vals = np.array([b_means[k] for k in b_keys])
    a_rule = np.array([k[0] for k in a_keys])
    a_seed = np.array([k[1] for k in a_keys])
    b_rule = np.array([k[0] for k in b_keys])
    b_seed = np.array([k[1] for k in b_keys])

    # Cluster bootstraps on the *difference of cluster-means*: for each
    # bootstrap sample, resample groups in both sources and compute the
    # difference of group-mean of source A minus source B.
    def _diff_bootstrap(group_a, group_b, *, sub_seed):
        rng = np.random.default_rng(sub_seed)
        ua = np.unique(group_a); ub = np.unique(group_b)
        idx_a = {g: np.where(group_a == g)[0] for g in ua}
        idx_b = {g: np.where(group_b == g)[0] for g in ub}
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            sa = rng.choice(ua, size=ua.size, replace=True)
            sb = rng.choice(ub, size=ub.size, replace=True)
            ia = np.concatenate([idx_a[g] for g in sa])
            ib = np.concatenate([idx_b[g] for g in sb])
            diffs[i] = float(a_vals[ia].mean() - b_vals[ib].mean())
        return (float(diffs.mean()),
                float(np.quantile(diffs, 0.025)),
                float(np.quantile(diffs, 0.975)))

    by_rule = _diff_bootstrap(a_rule, b_rule, sub_seed=seed)
    by_seed = _diff_bootstrap(a_seed, b_seed, sub_seed=seed + 1)
    pair_a = np.array([f"{r}|{s}" for r, s in zip(a_rule, a_seed)])
    pair_b = np.array([f"{r}|{s}" for r, s in zip(b_rule, b_seed)])
    by_pair = _diff_bootstrap(pair_a, pair_b, sub_seed=seed + 2)

    return ComparisonResult(
        metric=metric, source_a=source_a_name, source_b=source_b_name,
        n_a=int(a_vals.size), n_b=int(b_vals.size),
        mean_a=float(a_vals.mean()), mean_b=float(b_vals.mean()),
        mean_diff=float(a_vals.mean() - b_vals.mean()),
        bootstrap_by_rule=by_rule,
        bootstrap_by_seed=by_seed,
        bootstrap_by_rule_and_seed=by_pair,
        cliffs_delta=cliffs_delta(a_vals, b_vals),
        rank_biserial=rank_biserial(a_vals, b_vals),
        cohens_d=cohens_d_independent(a_vals, b_vals),
        permutation_p=permutation_test_mean_diff(
            a_vals, b_vals, n_permutations=n_permutations, seed=seed + 3,
        ),
        win_rate_a=float((a_vals[:, None] > b_vals[None, :]).mean()),
    )


# ---------------------------------------------------------------------------
# Generalization gap
# ---------------------------------------------------------------------------


def compute_generalization_gap(
    train_scores: dict, validation_scores: dict, test_scores: dict,
) -> dict:
    """For each rule_id present in all three score dicts, report
    (train, validation, test) fitness and the per-rule drops.
    """
    common = sorted(set(train_scores) & set(validation_scores) & set(test_scores))
    if not common:
        return {"n": 0, "rules": []}
    rows = []
    for rid in common:
        rows.append({
            "rule_id": rid,
            "train": float(train_scores[rid]),
            "validation": float(validation_scores[rid]),
            "test": float(test_scores[rid]),
            "train_to_validation": float(validation_scores[rid] - train_scores[rid]),
            "train_to_test": float(test_scores[rid] - train_scores[rid]),
            "validation_to_test": float(test_scores[rid] - validation_scores[rid]),
        })
    return {
        "n": len(rows),
        "rules": rows,
        "mean_train": float(np.mean([r["train"] for r in rows])),
        "mean_validation": float(np.mean([r["validation"] for r in rows])),
        "mean_test": float(np.mean([r["test"] for r in rows])),
        "mean_train_to_test_drop": float(np.mean([r["train_to_test"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# Success-criterion classifier (M7B spec interpretations)
# ---------------------------------------------------------------------------


_INTERP_STRONG_SUCCESS = (
    "M7 production validation supports the core claim: HCE-guided "
    "evolution produces projected candidates that are both observer-like "
    "and locally dependent on hidden state."
)
_INTERP_PARTIAL_SUCCESS = (
    "M7 optimized hidden dependence but partially traded off generic "
    "observer-likeness."
)
_INTERP_DISTINCT_OBJECTIVES = (
    "HCE and observer_score remain distinct objectives."
)
_INTERP_FAILURE_NOT_REPLICATED = (
    "The M7 result did not replicate at production scale."
)
_INTERP_FAILURE_THRESHOLD = (
    "M7 exploited projection-threshold sensitivity."
)
_INTERP_LOCAL_NOT_GLOBAL = (
    "The effect is candidate-local, not merely global hidden chaos."
)
_INTERP_2D_BEATS_OBS = (
    "This does not invalidate the HCE result; it confirms that HCE is "
    "the dimension-specific contribution, while generic observer_score "
    "may remain easier in 2D."
)


def _significant_positive(comp: ComparisonResult) -> bool:
    """Mean diff > 0 AND CI by rule_and_seed excludes 0 AND perm p < 0.05."""
    return (comp.mean_diff > 0
            and comp.bootstrap_by_rule_and_seed[1] > 0
            and comp.permutation_p < 0.05)


def _significant_negative(comp: ComparisonResult) -> bool:
    return (comp.mean_diff < 0
            and comp.bootstrap_by_rule_and_seed[2] < 0
            and comp.permutation_p < 0.05)


def select_interpretations(
    *,
    m7_vs_m4c: dict,            # {metric_name: ComparisonResult}
    m7_vs_m4a: dict,
    m7_vs_2d_observer: ComparisonResult | None,
    m7_threshold_audit: list,   # list of audit dicts from m6c_stats
) -> list[str]:
    """Apply the M7B canonical interpretation rules. Returns ordered
    paragraphs (most important first)."""
    out: list[str] = []

    hce_vs_m4c = m7_vs_m4c.get("hidden_vs_sham_delta")
    obs_vs_m4c = m7_vs_m4c.get("observer_score")
    hce_vs_m4a = m7_vs_m4a.get("hidden_vs_sham_delta")

    hce_wins = (hce_vs_m4c is not None and _significant_positive(hce_vs_m4c)
                and hce_vs_m4a is not None and _significant_positive(hce_vs_m4a))
    hce_directional = (hce_vs_m4c is not None and hce_vs_m4c.mean_diff > 0
                       and hce_vs_m4a is not None and hce_vs_m4a.mean_diff > 0)
    obs_preserved = (obs_vs_m4c is not None and obs_vs_m4c.mean_diff >= -0.05)
    obs_lost = (obs_vs_m4c is not None and _significant_negative(obs_vs_m4c))

    # Threshold check: did M7 retain HCE under strict filtering?
    threshold_collapsed = False
    if m7_threshold_audit:
        all_row = next((a for a in m7_threshold_audit
                       if a["filter"] == "all_candidates"), None)
        far_row = next((a for a in m7_threshold_audit
                       if a["filter"] == "mean_threshold_margin>0.10"), None)
        if all_row and far_row and all_row["mean_future_div"] > 0:
            ratio = far_row["mean_future_div"] / all_row["mean_future_div"]
            threshold_collapsed = (ratio < 0.3)

    if threshold_collapsed:
        out.append(_INTERP_FAILURE_THRESHOLD)
    elif hce_vs_m4c is not None and hce_vs_m4c.mean_diff <= 0:
        out.append(_INTERP_FAILURE_NOT_REPLICATED)
    elif hce_wins and obs_preserved:
        out.append(_INTERP_STRONG_SUCCESS)
    elif hce_directional and obs_lost:
        out.append(_INTERP_PARTIAL_SUCCESS)
    elif hce_directional:
        out.append(_INTERP_DISTINCT_OBJECTIVES)
    else:
        out.append("Mixed result; no single canonical interpretation fired.")

    # Always check far-vs-local localization.
    far_vs_m4c = m7_vs_m4c.get("hidden_vs_far_delta")
    if far_vs_m4c is not None and far_vs_m4c.mean_diff > 0:
        out.append(_INTERP_LOCAL_NOT_GLOBAL)

    # 2D observer-only check.
    if m7_vs_2d_observer is not None and _significant_negative(m7_vs_2d_observer):
        out.append(_INTERP_2D_BEATS_OBS)

    return out


# ---------------------------------------------------------------------------
# Top-level summary builder
# ---------------------------------------------------------------------------


PRIMARY_METRICS_M7B = (
    "hidden_vs_sham_delta",
    "hidden_vs_far_delta",
    "future_div_hidden_invisible",
    "local_div_hidden_invisible",
    "HCE",
    "observer_score",
    "candidate_lifetime",
)


def m7b_full_comparison_grid(
    rows_by_source: dict, *, horizon: int,
    n_boot: int = 2000, n_permutations: int = 2000, seed: int = 0,
) -> dict:
    """For every (source_a, source_b, metric) combination of interest,
    compute the full ComparisonResult."""
    out: dict[str, dict[str, dict]] = {}
    sources = list(rows_by_source.keys())
    if "M7_HCE_optimized" not in sources:
        return out
    rows_m7 = rows_by_source["M7_HCE_optimized"]
    for other in sources:
        if other == "M7_HCE_optimized": continue
        if other == "M4D_2D_optimized":
            # Only observer_score is meaningful for 2D.
            metrics = ("observer_score",)
        else:
            metrics = PRIMARY_METRICS_M7B
        cmp_key = f"M7_vs_{other}"
        out[cmp_key] = {}
        for m in metrics:
            out[cmp_key][m] = compare_sources(
                rows_m7, rows_by_source[other],
                metric=m, source_a_name="M7_HCE_optimized",
                source_b_name=other, horizon=horizon,
                n_boot=n_boot, n_permutations=n_permutations, seed=seed,
            )
    return out
