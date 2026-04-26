"""Paired-statistics helpers for the M4B observer-metric sweep.

The M4B sweep produces a list of :class:`PairedRecord` objects.  Each
record contains three :class:`ConditionResult` instances (one per
condition: ``coherent_4d``, ``shuffled_4d``, ``matched_2d``) for the
same ``(rule_idx, seed)``.

This module provides paired comparisons across those triples:

* :func:`compute_paired_difference` -- bootstrap CI, permutation p-value,
  Cohen's d (paired), Cliff's delta (paired/rank-biserial), per-pair win
  rates for two parallel arrays.
* :func:`compute_all_paired_differences` -- the full grid of
  ``(pair, metric)`` comparisons across a list of records.
* :func:`win_rate_random_candidate` -- candidate-level win rate using
  the per-condition ``all_combined_scores`` arrays.
* :func:`stats_summary_dict` / :func:`write_stats_summary_json` --
  JSON-serializable nested summary suitable for ``stats_summary.json``.
* :func:`render_stats_summary_md` -- human-readable markdown report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from observer_worlds.experiments._m4b_sweep import (
    CONDITION_NAMES,
    PairedRecord,
    SUMMARY_METRICS,
    metrics_dict,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# The three pairwise comparisons we always report.
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("coherent_4d", "shuffled_4d"),
    ("coherent_4d", "matched_2d"),
    ("shuffled_4d", "matched_2d"),
)


# Default subset of SUMMARY_METRICS that we headline in stats reports.
HEADLINE_METRICS: tuple[str, ...] = (
    "max_score",
    "top5_mean_score",
    "p95_score",
    "lifetime_weighted_mean_score",
    "score_per_track",
)


# ---------------------------------------------------------------------------
# PairedDifference dataclass
# ---------------------------------------------------------------------------


@dataclass
class PairedDifference:
    """Statistical summary of one (condition_a vs condition_b, metric) comparison."""

    condition_a: str
    condition_b: str
    metric: str
    n_pairs: int
    mean_difference: float           # mean(a - b)
    median_difference: float
    bootstrap_ci_low: float          # 95% by default
    bootstrap_ci_high: float
    permutation_p_value: float       # two-sided
    cohens_d_paired: float           # mean(d) / std(d, ddof=1)
    cliffs_delta: float              # rank-biserial form for paired data
    win_rate_a: float                # P(a > b) over the n_pairs
    win_rate_b: float                # P(a < b)
    tie_rate: float


# ---------------------------------------------------------------------------
# Core paired-difference computation
# ---------------------------------------------------------------------------


def compute_paired_difference(
    a_values: list[float],
    b_values: list[float],
    *,
    condition_a: str = "a",
    condition_b: str = "b",
    metric: str = "metric",
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> PairedDifference:
    """Compute the paired-difference statistics for two parallel arrays.

    ``a_values`` and ``b_values`` must be the same length; pair ``i`` is
    ``(a_values[i], b_values[i])``.
    """
    a = np.asarray(a_values, dtype=np.float64)
    b = np.asarray(b_values, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"a_values and b_values must have the same length; got "
            f"{a.shape} and {b.shape}"
        )
    n = a.size
    diffs = a - b

    if n == 0:
        return PairedDifference(
            condition_a=condition_a, condition_b=condition_b, metric=metric,
            n_pairs=0,
            mean_difference=0.0, median_difference=0.0,
            bootstrap_ci_low=0.0, bootstrap_ci_high=0.0,
            permutation_p_value=1.0,
            cohens_d_paired=0.0, cliffs_delta=0.0,
            win_rate_a=0.0, win_rate_b=0.0, tie_rate=0.0,
        )

    rng = np.random.default_rng(seed)

    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))

    # Bootstrap CI (paired): resample pair indices with replacement.
    if n_bootstrap > 0:
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_means = diffs[idx].mean(axis=1)
        alpha = 1.0 - confidence
        ci_low = float(np.quantile(boot_means, alpha / 2.0))
        ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    else:
        ci_low = ci_high = mean_diff

    # Permutation test (paired, two-sided): randomly flip per-pair signs.
    observed = abs(mean_diff)
    if n_permutations > 0:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
        perm_means = (signs * diffs).mean(axis=1)
        n_extreme = int(np.sum(np.abs(perm_means) >= observed))
        p_value = (n_extreme + 1) / (n_permutations + 1)
    else:
        p_value = 1.0

    # Cohen's d (paired).
    if n > 1:
        std = float(np.std(diffs, ddof=1))
    else:
        std = 0.0
    cohens_d = (mean_diff / std) if std > 0.0 else 0.0
    if not np.isfinite(cohens_d):
        cohens_d = 0.0

    # Cliff's delta (paired form / rank-biserial).
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_tie = int(n - n_pos - n_neg)
    cliffs_delta = (n_pos - n_neg) / n if n > 0 else 0.0

    win_rate_a = n_pos / n if n > 0 else 0.0
    win_rate_b = n_neg / n if n > 0 else 0.0
    tie_rate = n_tie / n if n > 0 else 0.0

    return PairedDifference(
        condition_a=condition_a,
        condition_b=condition_b,
        metric=metric,
        n_pairs=int(n),
        mean_difference=mean_diff,
        median_difference=median_diff,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        permutation_p_value=float(p_value),
        cohens_d_paired=float(cohens_d),
        cliffs_delta=float(cliffs_delta),
        win_rate_a=float(win_rate_a),
        win_rate_b=float(win_rate_b),
        tie_rate=float(tie_rate),
    )


def compute_all_paired_differences(
    records: list[PairedRecord],
    *,
    metrics: Iterable[str] = SUMMARY_METRICS,
    pairs: Iterable[tuple[str, str]] = COMPARISON_PAIRS,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> list[PairedDifference]:
    """Compute paired stats for every ``(pair, metric)`` combination."""
    metrics = tuple(metrics)
    pairs = tuple(pairs)

    # Pre-extract metrics-per-condition from each record so we don't redo
    # the dict construction for every (pair, metric) combo.
    per_condition: dict[str, list[dict[str, float]]] = {
        cond: [metrics_dict(getattr(rec, cond)) for rec in records]
        for cond in CONDITION_NAMES
    }

    out: list[PairedDifference] = []
    # Use a derived sub-seed per (pair, metric) so each comparison has its
    # own deterministic RNG stream but the overall result is reproducible.
    base_rng = np.random.default_rng(seed)
    for cond_a, cond_b in pairs:
        for metric in metrics:
            a_vals = [d[metric] for d in per_condition[cond_a]]
            b_vals = [d[metric] for d in per_condition[cond_b]]
            sub_seed = int(base_rng.integers(0, 2**31 - 1))
            out.append(
                compute_paired_difference(
                    a_vals, b_vals,
                    condition_a=cond_a, condition_b=cond_b,
                    metric=metric,
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    confidence=confidence,
                    seed=sub_seed,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Candidate-level random-pick win rate
# ---------------------------------------------------------------------------


def win_rate_random_candidate(
    records: list[PairedRecord],
    condition_a: str,
    condition_b: str,
    *,
    n_samples: int = 5000,
    seed: int | None = 0,
) -> dict[str, float]:
    """Per-pair random-candidate win rate.

    For each ``(rule_idx, seed)`` pair, draw a random candidate's combined
    score from ``condition_a`` and one from ``condition_b`` (using the
    ``all_combined_scores`` lists on each ConditionResult).  Counts are
    accumulated across all pairs and across ``n_samples`` draws per pair,
    then aggregated.

    Pairs where either condition has zero candidates are skipped (and
    counted in ``n_skipped``).
    """
    rng = np.random.default_rng(seed)

    total_a_wins = 0
    total_b_wins = 0
    total_ties = 0
    total_draws = 0
    n_skipped = 0
    n_used = 0

    for rec in records:
        a_scores = list(getattr(rec, condition_a).all_combined_scores)
        b_scores = list(getattr(rec, condition_b).all_combined_scores)
        if not a_scores or not b_scores:
            n_skipped += 1
            continue
        a_arr = np.asarray(a_scores, dtype=np.float64)
        b_arr = np.asarray(b_scores, dtype=np.float64)
        ai = rng.integers(0, a_arr.size, size=n_samples)
        bi = rng.integers(0, b_arr.size, size=n_samples)
        a_pick = a_arr[ai]
        b_pick = b_arr[bi]
        total_a_wins += int(np.sum(a_pick > b_pick))
        total_b_wins += int(np.sum(a_pick < b_pick))
        total_ties += int(np.sum(a_pick == b_pick))
        total_draws += int(n_samples)
        n_used += 1

    if total_draws == 0:
        return {
            "win_rate_a": 0.0,
            "win_rate_b": 0.0,
            "tie_rate": 0.0,
            "n_skipped": float(n_skipped),
            "n_used": 0.0,
            "n_samples_per_pair": float(n_samples),
            "total_draws": 0.0,
        }

    return {
        "win_rate_a": total_a_wins / total_draws,
        "win_rate_b": total_b_wins / total_draws,
        "tie_rate": total_ties / total_draws,
        "n_skipped": float(n_skipped),
        "n_used": float(n_used),
        "n_samples_per_pair": float(n_samples),
        "total_draws": float(total_draws),
    }


# ---------------------------------------------------------------------------
# JSON-serializable summary dict
# ---------------------------------------------------------------------------


def _json_default(obj):
    """Convert numpy scalar types to native Python types for json.dump."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _comparison_key(cond_a: str, cond_b: str) -> str:
    return f"{cond_a}_vs_{cond_b}"


def stats_summary_dict(
    records: list[PairedRecord],
    *,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
    provenance: dict | None = None,
) -> dict:
    """Build the JSON-serializable nested summary written to stats_summary.json."""
    n_pairs = len(records)

    # Count rules and per-rule seeds.
    rules_to_seeds: dict[int, set[int]] = {}
    for rec in records:
        rules_to_seeds.setdefault(rec.rule_idx, set()).add(rec.seed)
    n_rules = len(rules_to_seeds)
    if n_rules > 0:
        n_seeds_per_rule = float(
            sum(len(s) for s in rules_to_seeds.values()) / n_rules
        )
    else:
        n_seeds_per_rule = 0.0

    diffs = compute_all_paired_differences(
        records,
        metrics=SUMMARY_METRICS,
        pairs=COMPARISON_PAIRS,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed,
    )

    # Group by comparison key, then by metric.
    comparisons: dict[str, dict[str, dict]] = {}
    for d in diffs:
        key = _comparison_key(d.condition_a, d.condition_b)
        comparisons.setdefault(key, {})[d.metric] = asdict(d)

    # Candidate-level random-pick win rates per comparison.
    cand_win_rates: dict[str, dict[str, float]] = {}
    base_rng = np.random.default_rng(seed)
    for cond_a, cond_b in COMPARISON_PAIRS:
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        cand_win_rates[_comparison_key(cond_a, cond_b)] = win_rate_random_candidate(
            records, cond_a, cond_b, n_samples=5000, seed=sub_seed,
        )

    out = {
        "n_pairs": int(n_pairs),
        "n_rules": int(n_rules),
        "n_seeds_per_rule": float(n_seeds_per_rule),
        "comparisons": comparisons,
        "candidate_level_win_rates": cand_win_rates,
        "headline_metrics": list(HEADLINE_METRICS),
    }
    if provenance is not None:
        out["provenance"] = provenance
    return out


def write_stats_summary_json(
    records: list[PairedRecord],
    out_path: str,
    *,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
    provenance: dict | None = None,
) -> dict:
    """Compute :func:`stats_summary_dict` and write it to ``out_path``.

    Returns the dict so callers can inspect / re-use it without re-reading.
    """
    summary = stats_summary_dict(
        records,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        confidence=confidence,
        seed=seed,
        provenance=provenance,
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return summary


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


# Canonical interpretation sentences -- exposed so tests can substring-match.
# Headline finding sentences (primary normalized-metric outcomes).
_INTERP_COH_BEATS_2D_NORMALIZED = (
    "Coherent 4D significantly beats matched 2D on track-count-resistant "
    "metrics. Coherent vs shuffled is not yet statistically established."
)
_INTERP_COH_BEATS_BOTH_NORMALIZED = (
    "Coherent 4D shows evidence of stronger observer-like structure on "
    "normalized metrics against both shuffled-4D and matched-2D baselines."
)
_INTERP_COH_BEATS_2D_NOT_SHUF = (
    "The current evidence supports 4D dynamics over the standard 2D "
    "baseline, but does not yet establish that coherent hidden-dimensional "
    "structure beats shuffled 4D."
)
_INTERP_2D_WINS_NORMALIZED = (
    "The current M3 hypothesis is not supported for these rules; 4D "
    "projection is not outperforming matched 2D on normalized metrics."
)
_INTERP_MIXED_NORMALIZED = (
    "Mixed result on normalized metrics; no strong directional conclusion."
)

# Confound caveats appended below the headline.
_CAVEAT_TRACK_COUNT_DIFFERS = (
    "Extreme-score metrics are confounded by different candidate counts. "
    "Normalized metrics are treated as primary for this comparison."
)
_CAVEAT_OPTIMIZED_RULES = (
    "Because these rules were selected using observer-score, this is an "
    "optimized-regime result. A fairer comparison requires held-out "
    "validation and an optimized 2D baseline."
)
_CAVEAT_SEED_OVERLAP = (
    "WARNING: training seeds overlap evaluation seeds. This is not a "
    "held-out test; the result may overstate performance."
)

# Backwards-compat aliases (kept so existing tests still substring-match).
_INTERP_COH_STRONG = _INTERP_COH_BEATS_BOTH_NORMALIZED
_INTERP_COH_MAX_ONLY = _INTERP_MIXED_NORMALIZED
_INTERP_SHUF_MAX_TRACK_CONFOUND = _CAVEAT_TRACK_COUNT_DIFFERS
_INTERP_2D_WINS_TOP5 = _INTERP_2D_WINS_NORMALIZED
_INTERP_MIXED = _INTERP_MIXED_NORMALIZED


# Primary metrics that headline a finding (track-count-resistant).
_PRIMARY_METRICS: tuple[str, ...] = (
    "score_per_track",
    "lifetime_weighted_mean_score",
)
# Threshold for "track count differs substantially": |coh - other| / max(coh, 1) > 0.5.
_TRACK_COUNT_RATIO_DIFFERS = 0.5
# Win-rate threshold for "positive" finding.
_WIN_RATE_THRESHOLD = 0.65


def _fmt(x: float | int) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return str(v)
    return f"{v:.4f}"


def _significant_positive(d: dict) -> bool:
    """A paired-difference dict is a 'significant positive finding' iff:

      * permutation_p_value < 0.05
      * bootstrap_ci_low > 0  (CI excludes 0 on the positive side)
      * win_rate_a > _WIN_RATE_THRESHOLD
    """
    if not d:
        return False
    try:
        p = float(d.get("permutation_p_value", 1.0))
        ci_low = float(d.get("bootstrap_ci_low", 0.0))
        wr = float(d.get("win_rate_a", 0.0))
    except (TypeError, ValueError):
        return False
    return p < 0.05 and ci_low > 0.0 and wr > _WIN_RATE_THRESHOLD


def _significant_negative(d: dict) -> bool:
    """Significant evidence that condition_b beats condition_a."""
    if not d:
        return False
    try:
        p = float(d.get("permutation_p_value", 1.0))
        ci_high = float(d.get("bootstrap_ci_high", 0.0))
        wr = float(d.get("win_rate_b", 0.0))
    except (TypeError, ValueError):
        return False
    return p < 0.05 and ci_high < 0.0 and wr > _WIN_RATE_THRESHOLD


def _track_counts_differ_substantially(comparison: dict) -> bool:
    """True if mean candidate or track counts differ by > _TRACK_COUNT_RATIO_DIFFERS."""
    nc = comparison.get("n_candidates", {})
    nt = comparison.get("n_tracks", {})
    for d in (nc, nt):
        try:
            diff = float(d.get("mean_difference", 0.0))
        except (TypeError, ValueError):
            continue
        # Estimate the larger of the two means from the diff direction; we
        # don't have means directly here, so use abs(diff) > 1 as a coarse
        # threshold (any condition that produces 1+ extra candidate per
        # pair is enough to bias extreme-score metrics).
        if abs(diff) > 1.0:
            return True
    return False


def _interpretation(stats: dict) -> tuple[str, list[str]]:
    """Return (headline_sentence, list_of_caveat_sentences).

    Decision rules:

      1. If `score_per_track` OR `lifetime_weighted_mean_score` is a
         significant positive for coh-vs-2D AND coh-vs-shuf:
           => COH_BEATS_BOTH_NORMALIZED
      2. Else if it's significant positive for coh-vs-2D only:
           => COH_BEATS_2D_NOT_SHUF
      3. Else if it's significant negative for coh-vs-2D (2D wins):
           => 2D_WINS_NORMALIZED
      4. Else: MIXED_NORMALIZED

    Caveats appended (any combination):

      * If track counts differ substantially in any comparison:
          CAVEAT_TRACK_COUNT_DIFFERS
      * If stats.provenance.rule_source indicates observer optimization
        and stats.provenance.baseline_optimized is False:
          CAVEAT_OPTIMIZED_RULES
      * If stats.provenance.evaluation_overlaps_training is True:
          CAVEAT_SEED_OVERLAP
    """
    comparisons = stats.get("comparisons", {})
    coh_vs_shuf = comparisons.get(_comparison_key("coherent_4d", "shuffled_4d"), {})
    coh_vs_2d = comparisons.get(_comparison_key("coherent_4d", "matched_2d"), {})

    # Headline rule selection: check each primary metric.
    coh_beats_2d = any(
        _significant_positive(coh_vs_2d.get(m, {})) for m in _PRIMARY_METRICS
    )
    coh_beats_shuf = any(
        _significant_positive(coh_vs_shuf.get(m, {})) for m in _PRIMARY_METRICS
    )
    twod_beats_coh = any(
        _significant_negative(coh_vs_2d.get(m, {})) for m in _PRIMARY_METRICS
    )

    if coh_beats_2d and coh_beats_shuf:
        headline = _INTERP_COH_BEATS_BOTH_NORMALIZED
    elif coh_beats_2d and not coh_beats_shuf:
        headline = _INTERP_COH_BEATS_2D_NOT_SHUF
    elif twod_beats_coh:
        headline = _INTERP_2D_WINS_NORMALIZED
    else:
        headline = _INTERP_MIXED_NORMALIZED

    caveats: list[str] = []
    if _track_counts_differ_substantially(coh_vs_shuf) or \
            _track_counts_differ_substantially(coh_vs_2d):
        caveats.append(_CAVEAT_TRACK_COUNT_DIFFERS)

    prov = stats.get("provenance", {})
    if prov.get("rule_source") and "observer" in str(prov.get("rule_source", "")).lower():
        if not prov.get("baseline_optimized", False):
            caveats.append(_CAVEAT_OPTIMIZED_RULES)
    if prov.get("evaluation_overlaps_training"):
        caveats.append(_CAVEAT_SEED_OVERLAP)

    return headline, caveats


def _metric_table(stats: dict, metric: str) -> str:
    """Render a single markdown table for one HEADLINE_METRIC."""
    header = (
        "| comparison | mean_diff | 95% CI | perm p | cohen_d | cliff_delta "
        "| win_rate_a | win_rate_b |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows: list[str] = []
    comparisons = stats.get("comparisons", {})
    for cond_a, cond_b in COMPARISON_PAIRS:
        key = _comparison_key(cond_a, cond_b)
        d = comparisons.get(key, {}).get(metric)
        if d is None:
            continue
        ci = f"[{_fmt(d.get('bootstrap_ci_low', 0.0))}, {_fmt(d.get('bootstrap_ci_high', 0.0))}]"
        rows.append(
            f"| {cond_a} vs {cond_b} "
            f"| {_fmt(d.get('mean_difference', 0.0))} "
            f"| {ci} "
            f"| {_fmt(d.get('permutation_p_value', 1.0))} "
            f"| {_fmt(d.get('cohens_d_paired', 0.0))} "
            f"| {_fmt(d.get('cliffs_delta', 0.0))} "
            f"| {_fmt(d.get('win_rate_a', 0.0))} "
            f"| {_fmt(d.get('win_rate_b', 0.0))} |"
        )
    return header + "\n".join(rows) + "\n"


def render_stats_summary_md(stats: dict) -> str:
    """Render a human-readable markdown report of paired differences for
    HEADLINE_METRICS across all three pair comparisons.
    """
    lines: list[str] = []
    lines.append("# M4B Paired-Statistics Summary\n")
    lines.append(
        f"- **n_pairs**: {stats.get('n_pairs', 0)}\n"
        f"- **n_rules**: {stats.get('n_rules', 0)}\n"
        f"- **n_seeds_per_rule**: {stats.get('n_seeds_per_rule', 0.0):.2f}\n"
    )

    for metric in HEADLINE_METRICS:
        lines.append(f"\n## {metric}\n")
        lines.append(_metric_table(stats, metric))

    # Candidate-level random-pick win rates.
    lines.append("\n## Candidate-level random-pick win rates\n")
    cand = stats.get("candidate_level_win_rates", {})
    lines.append(
        "| comparison | win_rate_a | win_rate_b | tie_rate | n_used | n_skipped |\n"
        "|---|---|---|---|---|---|\n"
    )
    cand_rows: list[str] = []
    for cond_a, cond_b in COMPARISON_PAIRS:
        key = _comparison_key(cond_a, cond_b)
        c = cand.get(key, {})
        cand_rows.append(
            f"| {cond_a} vs {cond_b} "
            f"| {_fmt(c.get('win_rate_a', 0.0))} "
            f"| {_fmt(c.get('win_rate_b', 0.0))} "
            f"| {_fmt(c.get('tie_rate', 0.0))} "
            f"| {_fmt(c.get('n_used', 0.0))} "
            f"| {_fmt(c.get('n_skipped', 0.0))} |"
        )
    lines.append("\n".join(cand_rows) + "\n")

    # Provenance (Part B): if recorded, show it before interpretation.
    prov = stats.get("provenance", {})
    if prov:
        lines.append("\n## Rule provenance\n")
        for key in (
            "rule_source", "optimization_objective", "baseline_optimized",
            "evaluation_overlaps_training",
        ):
            if key in prov:
                lines.append(f"- **{key}**: {prov[key]}\n")
        if prov.get("training_seeds") is not None:
            lines.append(f"- **training_seeds**: {prov['training_seeds']}\n")
        if prov.get("evaluation_seeds") is not None:
            lines.append(f"- **evaluation_seeds**: {prov['evaluation_seeds']}\n")

    lines.append("\n## Interpretation\n")
    headline, caveats = _interpretation(stats)
    lines.append(headline + "\n")
    for c in caveats:
        lines.append("\n_" + c + "_\n")

    return "".join(lines)
