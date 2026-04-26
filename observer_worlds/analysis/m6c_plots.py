"""M6C — plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.analysis.hidden_features import HIDDEN_FEATURE_NAMES


def _empty(out_path: Path, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save(fig, out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _filter(rows, horizon: int):
    return [r for r in rows if r.horizon == horizon]


def _scatter_feature_vs_hce(rows, feature_name: str, out_path: Path,
                            *, horizon: int, title: str = None) -> None:
    sub = _filter(rows, horizon)
    if not sub:
        _empty(out_path, "no rows"); return
    x = np.array([r.features.get(feature_name, 0.0) for r in sub])
    y = np.array([r.HCE for r in sub])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=20, alpha=0.6, color="#1f77b4")
    if x.std() > 1e-12 and y.std() > 1e-12:
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(title or f"HCE vs {feature_name} (Pearson r={r:+.2f}, n={x.size})")
    else:
        ax.set_title(title or f"HCE vs {feature_name} (n={x.size})")
    ax.set_xlabel(feature_name); ax.set_ylabel("HCE")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_hce_vs_threshold_margin(rows, out_path, *, horizon: int):
    _scatter_feature_vs_hce(rows, "mean_threshold_margin", out_path,
                            horizon=horizon)


def plot_hce_vs_near_threshold_fraction(rows, out_path, *, horizon: int):
    _scatter_feature_vs_hce(rows, "near_threshold_fraction", out_path,
                            horizon=horizon)


def plot_hce_vs_hidden_entropy(rows, out_path, *, horizon: int):
    _scatter_feature_vs_hce(rows, "mean_hidden_entropy", out_path,
                            horizon=horizon)


def plot_hce_vs_hidden_autocorrelation(rows, out_path, *, horizon: int):
    _scatter_feature_vs_hce(rows, "mean_hidden_spatial_autocorrelation",
                            out_path, horizon=horizon)


def plot_hce_vs_hidden_temporal_persistence(rows, out_path, *, horizon: int):
    _scatter_feature_vs_hce(rows, "hidden_temporal_persistence", out_path,
                            horizon=horizon)


def plot_hce_vs_hidden_heterogeneity(rows, out_path, *, horizon: int):
    _scatter_feature_vs_hce(rows, "hidden_heterogeneity", out_path,
                            horizon=horizon)


def plot_feature_importance_bar(model_scores: list[dict], out_path: Path) -> None:
    """Top-15 RF feature importances for HCE."""
    out_path = Path(out_path)
    rf_hce = next((m for m in model_scores
                  if m.get("model") == "RandomForest" and m.get("outcome") == "HCE"), None)
    if rf_hce is None:
        _empty(out_path, "no RF model for HCE"); return
    imps = rf_hce.get("feature_importances", {})
    items = sorted(imps.items(), key=lambda kv: -kv[1])[:15]
    if not items:
        _empty(out_path, "no importances"); return
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(items))))
    ax.barh(range(len(items)), vals, color="#1f77b4", alpha=0.7)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RF feature importance")
    ax.set_title("Top features predicting HCE (RandomForest, grouped CV)")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, out_path)


def plot_threshold_audit_hce_boxplot(rows, audit: list[dict], out_path: Path,
                                     *, horizon: int) -> None:
    """Box plot of HCE on each filter subset."""
    out_path = Path(out_path)
    sub = _filter(rows, horizon)
    if not sub or not audit:
        _empty(out_path, "no audit data"); return
    data, labels = [], []
    # Filter subsets
    data.append([r.HCE for r in sub])
    labels.append(f"all\n(N={len(sub)})")
    for thresh in (0.25, 0.10):
        flt = [r for r in sub if r.features.get("near_threshold_fraction", 1.0) < thresh]
        if flt:
            data.append([r.HCE for r in flt])
            labels.append(f"near<{thresh}\n(N={len(flt)})")
    far = [r for r in sub if r.features.get("mean_threshold_margin", 0.0) > 0.10]
    if far:
        data.append([r.HCE for r in far])
        labels.append(f"margin>0.10\n(N={len(far)})")
    if not data or all(len(d) == 0 for d in data):
        _empty(out_path, "no data after filtering"); return
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(data)), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("HCE")
    ax.set_title("Threshold-artifact audit: HCE by candidate subset")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_predicted_vs_actual_hce(rows, out_path: Path, *, horizon: int,
                                seed: int = 0) -> None:
    """Fit RandomForest on all data, plot leave-one-rule-out predictions vs actual."""
    out_path = Path(out_path)
    sub = _filter(rows, horizon)
    if len(sub) < 6:
        _empty(out_path, "too few rows"); return
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GroupKFold
    except ImportError:
        _empty(out_path, "sklearn not available"); return
    X = np.array([
        [r.features.get(fn, 0.0) for fn in HIDDEN_FEATURE_NAMES] for r in sub
    ])
    y = np.array([r.HCE for r in sub])
    groups = np.array([r.rule_id for r in sub])
    n_groups = int(np.unique(groups).size)
    if n_groups < 2:
        _empty(out_path, "need >=2 rule groups"); return
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.empty_like(y)
    preds.fill(np.nan)
    for tr, te in gkf.split(X, y, groups=groups):
        m = RandomForestRegressor(n_estimators=80, max_depth=8,
                                  random_state=seed, n_jobs=1)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    valid = ~np.isnan(preds)
    if valid.sum() < 3:
        _empty(out_path, "no valid CV predictions"); return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y[valid], preds[valid], s=20, alpha=0.6, color="#1f77b4")
    lim = max(float(y[valid].max()), float(preds[valid].max()), 0.001) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.6, label="y=x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    if y[valid].std() > 1e-12:
        from sklearn.metrics import r2_score
        r2 = r2_score(y[valid], preds[valid])
        ax.set_title(f"Predicted vs actual HCE (grouped CV, R²={r2:+.3f}, n={int(valid.sum())})")
    ax.set_xlabel("actual HCE"); ax.set_ylabel("predicted HCE")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out_path)


def plot_hce_by_rule_source(rows, out_path: Path, *, horizon: int) -> None:
    out_path = Path(out_path)
    sub = _filter(rows, horizon)
    by_src: dict[str, list[float]] = defaultdict(list)
    for r in sub:
        by_src[r.rule_source].append(r.HCE)
    if not by_src:
        _empty(out_path, "no rows"); return
    sources = sorted(by_src)
    data = [by_src[s] for s in sources]
    labels = [f"{s}\n(N={len(by_src[s])})" for s in sources]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(sources)), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("HCE")
    ax.set_title("HCE distribution by rule source")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_ablation_effects_by_type(rows, out_path: Path, *, horizon: int) -> None:
    """Box plot of ablation_future_div per ablation type. Only median-horizon
    rows have ablations populated."""
    out_path = Path(out_path)
    sub = [r for r in rows if r.horizon == horizon and r.ablation_future_div]
    if not sub:
        _empty(out_path, "no ablation data"); return
    types = sorted(sub[0].ablation_future_div.keys())
    data = [[r.ablation_future_div[t] for r in sub
             if t in r.ablation_future_div] for t in types]
    labels = [f"{t}\n(N={len(d)})" for t, d in zip(types, data)]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(types)), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("future_div under ablation")
    ax.set_title("Ablation effects by intervention type")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    _save(fig, out_path)


def plot_hidden_feature_correlation_heatmap(rows, out_path: Path,
                                            *, horizon: int) -> None:
    """Heatmap of pairwise Pearson correlation among hidden features."""
    out_path = Path(out_path)
    sub = _filter(rows, horizon)
    if len(sub) < 5:
        _empty(out_path, "too few rows"); return
    X = np.array([
        [r.features.get(fn, 0.0) for fn in HIDDEN_FEATURE_NAMES] for r in sub
    ])
    # Drop columns with zero variance.
    var = X.std(axis=0)
    keep = var > 1e-12
    if keep.sum() < 2:
        _empty(out_path, "all features constant"); return
    Xs = X[:, keep]
    names = [n for n, k in zip(HIDDEN_FEATURE_NAMES, keep) if k]
    C = np.corrcoef(Xs.T)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(names)),
                                    max(8, 0.4 * len(names))))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=80, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Hidden-feature pairwise correlation")
    _save(fig, out_path)


def write_all_m6c_plots(rows, model_scores, audit, out_dir, *, horizon: int) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_hce_vs_threshold_margin(rows, out_dir / "hce_vs_threshold_margin.png",
                                horizon=horizon)
    plot_hce_vs_near_threshold_fraction(
        rows, out_dir / "hce_vs_near_threshold_fraction.png", horizon=horizon
    )
    plot_hce_vs_hidden_entropy(rows, out_dir / "hce_vs_hidden_entropy.png",
                              horizon=horizon)
    plot_hce_vs_hidden_autocorrelation(
        rows, out_dir / "hce_vs_hidden_autocorrelation.png", horizon=horizon
    )
    plot_hce_vs_hidden_temporal_persistence(
        rows, out_dir / "hce_vs_hidden_temporal_persistence.png", horizon=horizon
    )
    plot_hce_vs_hidden_heterogeneity(
        rows, out_dir / "hce_vs_hidden_heterogeneity.png", horizon=horizon
    )
    plot_feature_importance_bar(model_scores, out_dir / "feature_importance_bar.png")
    plot_threshold_audit_hce_boxplot(rows, audit,
                                    out_dir / "threshold_audit_hce_boxplot.png",
                                    horizon=horizon)
    plot_predicted_vs_actual_hce(rows, out_dir / "predicted_vs_actual_hce.png",
                                horizon=horizon)
    plot_hce_by_rule_source(rows, out_dir / "hce_by_rule_source.png",
                           horizon=horizon)
    plot_ablation_effects_by_type(rows, out_dir / "ablation_effects_by_type.png",
                                 horizon=horizon)
    plot_hidden_feature_correlation_heatmap(
        rows, out_dir / "hidden_feature_correlation_heatmap.png", horizon=horizon
    )
