"""M6C — correlation, regression (grouped CV), threshold-artifact audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observer_worlds.analysis.hidden_features import HIDDEN_FEATURE_NAMES


# Headline outcome variables we test against features.
OUTCOMES: tuple[str, ...] = (
    "future_div_hidden_invisible",
    "local_div_hidden_invisible",
    "hidden_vs_sham_delta",
    "hidden_vs_far_delta",
    "HCE",
    "survival_delta",
)


# ---------------------------------------------------------------------------
# Convert M6CRow list to feature matrix + outcome vectors
# ---------------------------------------------------------------------------


def rows_to_matrix(rows, *, horizon: int):
    """Filter rows by horizon and stack features into X, outcomes into y_dict.

    Returns:
        X: (n_rows, n_features) float matrix
        y: dict[outcome_name → (n_rows,) float vector]
        groups: (n_rows,) array of rule_id strings for grouped CV
        feature_names: tuple of column names
    """
    sub = [r for r in rows if r.horizon == horizon]
    if not sub:
        return None, None, None, ()
    feature_names = tuple(HIDDEN_FEATURE_NAMES)
    X = np.zeros((len(sub), len(feature_names)), dtype=np.float64)
    for i, r in enumerate(sub):
        for j, fn in enumerate(feature_names):
            X[i, j] = float(r.features.get(fn, 0.0))
    y = {o: np.array([float(getattr(r, o)) for r in sub]) for o in OUTCOMES}
    groups = np.array([r.rule_id for r in sub])
    return X, y, groups, feature_names


# ---------------------------------------------------------------------------
# Correlation table
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, NaN-safe (returns 0 on constant input)."""
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x, y):
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def feature_outcome_correlations(rows, *, horizon: int) -> list[dict]:
    """Returns one row per (feature, outcome) with Pearson and Spearman r."""
    X, y, _, names = rows_to_matrix(rows, horizon=horizon)
    if X is None:
        return []
    out = []
    for j, fn in enumerate(names):
        for outcome in OUTCOMES:
            out.append({
                "feature": fn,
                "outcome": outcome,
                "pearson_r": _pearson(X[:, j], y[outcome]),
                "spearman_r": _spearman(X[:, j], y[outcome]),
                "n": int(X.shape[0]),
            })
    return out


# ---------------------------------------------------------------------------
# Grouped-CV regression (Linear + RandomForest)
# ---------------------------------------------------------------------------


@dataclass
class ModelScore:
    model: str
    outcome: str
    cv_mean_r2: float
    cv_std_r2: float
    cv_mean_mae: float
    cv_std_mae: float
    n_folds: int
    n_samples: int
    feature_importances: dict[str, float]


def grouped_cv_regression(
    rows, *, horizon: int, n_splits: int = 5, seed: int = 0,
) -> list[ModelScore]:
    """Fit Linear + RandomForest with GroupKFold by rule_id; report
    cross-validated R^2, MAE, and feature importances."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, r2_score

    X, y, groups, names = rows_to_matrix(rows, horizon=horizon)
    if X is None or X.shape[0] < n_splits + 1:
        return []
    n_groups = int(np.unique(groups).size)
    n_splits = min(n_splits, max(2, n_groups))

    out = []
    for outcome in OUTCOMES:
        yv = y[outcome]
        if yv.std() < 1e-12:  # constant outcome
            continue
        for model_name, ctor in (
            ("Ridge", lambda: Ridge(alpha=1.0)),
            ("RandomForest", lambda: RandomForestRegressor(
                n_estimators=80, max_depth=8, random_state=seed, n_jobs=1)),
        ):
            r2s, maes, importances = [], [], np.zeros(len(names))
            try:
                gkf = GroupKFold(n_splits=n_splits)
                for tr, te in gkf.split(X, yv, groups=groups):
                    m = ctor()
                    m.fit(X[tr], yv[tr])
                    pred = m.predict(X[te])
                    r2s.append(float(r2_score(yv[te], pred)))
                    maes.append(float(mean_absolute_error(yv[te], pred)))
                    if hasattr(m, "feature_importances_"):
                        importances += m.feature_importances_
                    elif hasattr(m, "coef_"):
                        importances += np.abs(m.coef_)
                if r2s:
                    importances /= len(r2s)
                    out.append(ModelScore(
                        model=model_name, outcome=outcome,
                        cv_mean_r2=float(np.mean(r2s)),
                        cv_std_r2=float(np.std(r2s)),
                        cv_mean_mae=float(np.mean(maes)),
                        cv_std_mae=float(np.std(maes)),
                        n_folds=len(r2s), n_samples=int(X.shape[0]),
                        feature_importances={
                            n: float(v) for n, v in zip(names, importances)
                        },
                    ))
            except Exception as e:
                continue
    return out


# ---------------------------------------------------------------------------
# Threshold-artifact audit
# ---------------------------------------------------------------------------


def threshold_artifact_audit(rows, *, horizon: int) -> list[dict]:
    """Compute mean future_div / vs_sham_delta / vs_far_delta on subsets
    of candidates filtered by their threshold-margin features."""
    sub = [r for r in rows if r.horizon == horizon]
    if not sub:
        return []
    out = []

    def _stats(filt_name, filtered):
        if not filtered:
            return None
        future = np.array([r.future_div_hidden_invisible for r in filtered])
        vs_sham = np.array([r.hidden_vs_sham_delta for r in filtered])
        vs_far = np.array([r.hidden_vs_far_delta for r in filtered])
        return {
            "filter": filt_name,
            "n_candidates": len(filtered),
            "mean_future_div": float(future.mean()),
            "mean_vs_sham": float(vs_sham.mean()),
            "mean_vs_far": float(vs_far.mean()),
            "fraction_future_div_gt_zero": float((future > 0).mean()),
        }

    out.append(_stats("all_candidates", sub))
    for thresh in (0.25, 0.10):
        flt = [r for r in sub if r.features.get("near_threshold_fraction", 1.0) < thresh]
        out.append(_stats(f"near_threshold_fraction<{thresh}", flt))
    flt = [r for r in sub if r.features.get("mean_threshold_margin", 0.0) > 0.10]
    out.append(_stats("mean_threshold_margin>0.10", flt))
    return [x for x in out if x is not None]


# ---------------------------------------------------------------------------
# Top-level summary builder
# ---------------------------------------------------------------------------


def m6c_full_summary(rows, *, horizons: list[int],
                    n_splits: int = 5, seed: int = 0) -> dict:
    headline_h = horizons[len(horizons) // 2]
    return {
        "n_rows": len(rows),
        "horizons": list(horizons),
        "headline_horizon": headline_h,
        "correlations": feature_outcome_correlations(rows, horizon=headline_h),
        "model_scores": [
            {**ms.__dict__,
             "feature_importances": ms.feature_importances}
            for ms in grouped_cv_regression(
                rows, horizon=headline_h, n_splits=n_splits, seed=seed
            )
        ],
        "threshold_audit": threshold_artifact_audit(rows, horizon=headline_h),
    }


# ---------------------------------------------------------------------------
# Markdown rendering + interpretation
# ---------------------------------------------------------------------------


_INTERP_THRESHOLD_DOMINATES = (
    "HCE is largely mediated by projection-threshold sensitivity. This is "
    "real hidden causal dependence, but it may be projection-specific."
)
_INTERP_HCE_PERSISTS_AWAY_FROM_THRESHOLD = (
    "HCE persists away from projection thresholds, supporting a stronger "
    "hidden-causal-substrate interpretation."
)
_INTERP_TEMPORAL_PERSISTENCE = (
    "Hidden causal dependence is associated with temporally coherent "
    "hidden structure."
)
_INTERP_ENTROPY_HETEROGENEITY = (
    "HCE is associated with hidden microstate complexity under the "
    "projected candidate."
)
_INTERP_NO_FEATURES = (
    "Current feature set does not explain HCE; hidden dependence may be "
    "rule-specific or require richer descriptors."
)
_INTERP_OBSERVER_INDEPENDENT = (
    "Generic observer-likeness and hidden-causal dependence remain "
    "partially independent axes."
)


def _select_interpretations(summary: dict) -> list[str]:
    out: list[str] = []
    audit = summary.get("threshold_audit", [])
    if audit:
        all_row = next((a for a in audit if a["filter"] == "all_candidates"), None)
        far_row = next((a for a in audit
                       if a["filter"] == "mean_threshold_margin>0.10"), None)
        if all_row and far_row and far_row["n_candidates"] >= 3:
            ratio = (far_row["mean_future_div"] /
                     max(all_row["mean_future_div"], 1e-9))
            if ratio < 0.3:
                out.append(_INTERP_THRESHOLD_DOMINATES)
            elif ratio > 0.5:
                out.append(_INTERP_HCE_PERSISTS_AWAY_FROM_THRESHOLD)

    cors = summary.get("correlations", [])
    # Find features predictive of HCE (Spearman |r| > 0.25).
    hce_cors = [c for c in cors if c["outcome"] == "HCE"]
    strong = [c for c in hce_cors if abs(c["spearman_r"]) > 0.25]

    temporal_strong = any(c["feature"] == "hidden_temporal_persistence"
                          for c in strong)
    if temporal_strong:
        out.append(_INTERP_TEMPORAL_PERSISTENCE)

    entropy_features = ("mean_hidden_entropy", "hidden_heterogeneity",
                        "std_active_fraction")
    if any(c["feature"] in entropy_features for c in strong):
        out.append(_INTERP_ENTROPY_HETEROGENEITY)

    # If model_scores all have R^2 < 0.1, no features predict.
    ms = summary.get("model_scores", [])
    if ms:
        max_r2 = max(m["cv_mean_r2"] for m in ms if m["outcome"] == "HCE") \
            if any(m["outcome"] == "HCE" for m in ms) else float("-inf")
        if max_r2 < 0.1:
            out.append(_INTERP_NO_FEATURES)

    # Observer-score correlation.
    obs_cors = [c for c in cors if c["feature"] == "observer_score"]
    # observer_score isn't in HIDDEN_FEATURE_NAMES; we don't have it as a feature.
    # Skip this check.

    if not out:
        out.append("Mixed result; no strong directional conclusion from "
                   "features → HCE.")
    return out


def render_m6c_summary_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# M6C — Hidden Organization Taxonomy")
    lines.append("")
    lines.append(f"- N rows: {summary['n_rows']}")
    lines.append(f"- Horizons: {summary['horizons']}")
    lines.append(f"- Headline horizon: {summary['headline_horizon']}")

    # Threshold audit.
    lines.append("\n## Threshold-artifact audit")
    lines.append("")
    audit = summary.get("threshold_audit", [])
    if audit:
        lines.append("| filter | n | mean_future_div | mean_vs_sham | mean_vs_far | fraction_future>0 |")
        lines.append("|---|---|---|---|---|---|")
        for a in audit:
            lines.append(
                f"| {a['filter']} | {a['n_candidates']} | "
                f"{a['mean_future_div']:+.4f} | {a['mean_vs_sham']:+.4f} | "
                f"{a['mean_vs_far']:+.4f} | "
                f"{a['fraction_future_div_gt_zero']:.2f} |"
            )
    else:
        lines.append("(no rows)")

    # Top correlations with HCE.
    lines.append("\n## Top features predicting HCE (by |Spearman r|)")
    lines.append("")
    cors = [c for c in summary.get("correlations", []) if c["outcome"] == "HCE"]
    cors.sort(key=lambda c: -abs(c["spearman_r"]))
    lines.append("| feature | Spearman r | Pearson r | n |")
    lines.append("|---|---|---|---|")
    for c in cors[:10]:
        lines.append(
            f"| {c['feature']} | {c['spearman_r']:+.3f} | "
            f"{c['pearson_r']:+.3f} | {c['n']} |"
        )

    # Model scores.
    lines.append("\n## Grouped-CV regression scores (predicting outcomes from features)")
    lines.append("")
    ms = summary.get("model_scores", [])
    if ms:
        lines.append("| model | outcome | cv R² | ± | cv MAE | ± | n_folds | n_samples |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in ms:
            lines.append(
                f"| {m['model']} | {m['outcome']} | "
                f"{m['cv_mean_r2']:+.3f} | {m['cv_std_r2']:.3f} | "
                f"{m['cv_mean_mae']:.4f} | {m['cv_std_mae']:.4f} | "
                f"{m['n_folds']} | {m['n_samples']} |"
            )
    else:
        lines.append("(insufficient data for grouped CV)")

    # Top RF feature importances for HCE.
    lines.append("\n## Top RandomForest feature importances for HCE")
    lines.append("")
    rf_hce = next((m for m in ms if m["model"] == "RandomForest"
                  and m["outcome"] == "HCE"), None)
    if rf_hce:
        sorted_imp = sorted(rf_hce["feature_importances"].items(),
                            key=lambda kv: -kv[1])[:10]
        lines.append("| feature | importance |")
        lines.append("|---|---|")
        for feat, imp in sorted_imp:
            lines.append(f"| {feat} | {imp:.4f} |")

    # Interpretation.
    lines.append("\n## Interpretation")
    lines.append("")
    for p in _select_interpretations(summary):
        lines.append(f"- {p}")
    return "\n".join(lines)
