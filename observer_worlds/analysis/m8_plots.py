"""M8 — plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.experiments._m8_mechanism import (
    MECHANISM_CLASSES,
    M8CandidateResult,
)


SOURCE_PALETTE: dict[str, str] = {
    "M7_HCE_optimized": "#1f77b4",
    "M4C_observer_optimized": "#ff7f0e",
    "M4A_viability": "#2ca02c",
}
MECHANISM_PALETTE: dict[str, str] = {
    "boundary_mediated": "#1f77b4",
    "interior_reservoir": "#ff7f0e",
    "environment_coupled": "#2ca02c",
    "global_chaotic": "#d62728",
    "threshold_mediated": "#9467bd",
    "delayed_hidden_channel": "#17becf",
    "unclear": "#7f7f7f",
}


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


def _by_source(results: list[M8CandidateResult]) -> dict:
    out: dict = {}
    for r in results:
        out.setdefault(r.rule_source, []).append(r)
    return out


def _per_candidate_hce(r): return r.timing.full_grid_l1_per_horizon[
    len(r.timing.horizons) // 2] if r.timing.horizons else 0.0


# ---------------------------------------------------------------------------
# Plot implementations
# ---------------------------------------------------------------------------


def plot_response_map_examples_top_hce(results, out_path, *, k: int = 4):
    """Side-by-side response maps for the top-k highest-HCE candidates."""
    out_path = Path(out_path)
    if not results:
        _empty(out_path, "no candidates"); return
    ranked = sorted(results, key=lambda r: -_per_candidate_hce(r))[:k]
    fig, axes = plt.subplots(1, len(ranked), figsize=(4 * len(ranked), 4))
    if len(ranked) == 1: axes = [axes]
    for ax, r in zip(axes, ranked):
        rg = r.response_map.response_grid
        im = ax.imshow(rg, cmap="viridis", aspect="equal")
        # Outline interior.
        rows, cols = np.where(r.response_map.interior_mask)
        if rows.size > 0:
            ax.scatter(cols, rows, s=2, c="red", marker=".", alpha=0.4)
        ax.set_title(f"{r.rule_source[:6]} | t={r.candidate_id}\n"
                     f"HCE={_per_candidate_hce(r):.3f} mech={r.mechanism.label[:10]}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Top-HCE response maps (red dots = interior cells)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_path)


def plot_boundary_vs_interior_response_by_source(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    sources = list(by_src.keys())
    bnd = [[r.response_map.boundary_response_fraction for r in by_src[s]]
           for s in sources]
    inter = [[r.response_map.interior_response_fraction for r in by_src[s]]
             for s in sources]
    x = np.arange(len(sources)); width = 0.4
    ax.bar(x - width/2, [np.mean(b) if b else 0.0 for b in bnd], width=width,
           label="boundary", alpha=0.7, color="#1f77b4")
    ax.bar(x + width/2, [np.mean(i) if i else 0.0 for i in inter], width=width,
           label="interior", alpha=0.7, color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("mean response fraction")
    ax.set_title("Boundary vs interior response fraction by source")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out_path)


def plot_first_visible_effect_time_by_source(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for src, rs in by_src.items():
        vals = [r.timing.first_visible_effect_time for r in rs
                if r.timing.first_visible_effect_time > 0]
        if vals:
            data.append(vals); labels.append(f"{src[:8]}\n(N={len(vals)})")
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("first horizon with local_div > epsilon")
    ax.set_title("First-visible-effect time by source")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    _save(fig, out_path)


def plot_hidden_mass_vs_visible_mass_over_time(results, out_path, *, k: int = 6):
    out_path = Path(out_path)
    if not results: _empty(out_path, "no candidates"); return
    ranked = sorted(results, key=lambda r: -_per_candidate_hce(r))[:k]
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in ranked:
        ts = np.arange(1, r.pathway.n_steps + 1)
        ax.plot(ts, r.pathway.hidden_mass_per_step,
               linestyle="--", alpha=0.7, label=f"hidden t={r.candidate_id}")
        ax.plot(ts, r.pathway.visible_mass_per_step,
               linestyle="-", alpha=0.7, label=f"visible t={r.candidate_id}")
    ax.set_xlabel("rollout step"); ax.set_ylabel("XOR mass")
    ax.set_title(f"Hidden vs visible mass over time (top-{len(ranked)} HCE candidates)")
    ax.grid(True, alpha=0.3)
    if len(ranked) <= 6: ax.legend(fontsize=7, loc="upper left")
    _save(fig, out_path)


def plot_hidden_to_visible_conversion_time_by_source(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for src, rs in by_src.items():
        vals = [r.pathway.hidden_to_visible_conversion_time for r in rs
                if r.pathway.hidden_to_visible_conversion_time > 0]
        if vals:
            data.append(vals); labels.append(f"{src[:8]}\n(N={len(vals)})")
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("hidden→visible conversion time (steps)")
    ax.set_title("Hidden-to-visible conversion time by source")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    _save(fig, out_path)


def plot_hce_vs_lifetime_tradeoff(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, rs in _by_source(results).items():
        x = [r.candidate_lifetime for r in rs]
        y = [_per_candidate_hce(r) for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6,
                   label=src, color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("candidate lifetime"); ax.set_ylabel("HCE")
    ax.set_title("HCE vs candidate lifetime tradeoff")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_hidden_volatility_vs_lifetime(results, out_path):
    """Use boundary_response_fraction as a volatility-leaning proxy."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, rs in _by_source(results).items():
        x = [r.candidate_lifetime for r in rs]
        y = [r.response_map.boundary_response_fraction for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6,
                   label=src, color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("candidate lifetime")
    ax.set_ylabel("boundary response fraction (volatility proxy)")
    ax.set_title("Hidden boundary-volatility vs lifetime")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_hce_vs_boundary_response_fraction(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, rs in _by_source(results).items():
        x = [r.response_map.boundary_response_fraction for r in rs]
        y = [_per_candidate_hce(r) for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6,
                   label=src, color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("boundary response fraction"); ax.set_ylabel("HCE")
    ax.set_title("HCE vs boundary response fraction")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_mechanism_class_distribution(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    fig, ax = plt.subplots(figsize=(11, 6))
    sources = list(by_src.keys())
    width = 0.8 / len(sources)
    x = np.arange(len(MECHANISM_CLASSES))
    for i, src in enumerate(sources):
        labels = [r.mechanism.label for r in by_src[src]]
        n_total = max(len(labels), 1)
        fracs = [sum(1 for l in labels if l == cls) / n_total
                 for cls in MECHANISM_CLASSES]
        ax.bar(x + (i - len(sources) / 2 + 0.5) * width, fracs, width=width,
               label=f"{src} (N={n_total})",
               color=SOURCE_PALETTE.get(src, "#999"), alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(MECHANISM_CLASSES, rotation=20,
                                         ha="right", fontsize=8)
    ax.set_ylabel("fraction of candidates")
    ax.set_title("Mechanism class distribution by source")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_local_vs_far_effect_by_mechanism(results, out_path):
    out_path = Path(out_path)
    by_mech: dict[str, list] = defaultdict(list)
    for r in results:
        by_mech[r.mechanism.label].append(r)
    if not by_mech: _empty(out_path, "no candidates"); return
    fig, ax = plt.subplots(figsize=(11, 5))
    classes = sorted(by_mech.keys())
    local_means = [np.mean([r.mediation.candidate_locality_index
                           for r in by_mech[c]]) for c in classes]
    far_means = [np.mean([r.mediation.far_hidden_effect for r in by_mech[c]])
                for c in classes]
    x = np.arange(len(classes)); width = 0.4
    ax.bar(x - width/2, local_means, width=width, label="locality_index",
           color="#1f77b4", alpha=0.7)
    ax.bar(x + width/2, far_means, width=width, label="far_hidden_effect",
           color="#d62728", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
    ax.set_title("Local vs far hidden effect by mechanism class")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out_path)


def plot_feature_lead_lag_heatmap(results, out_path):
    """Heatmap: rows = feature names, cols = lag, cell = mean correlation
    across candidates."""
    out_path = Path(out_path)
    if not results: _empty(out_path, "no candidates"); return
    feature_names = ["mean_active_fraction", "mean_threshold_margin",
                     "near_threshold_fraction", "mean_hidden_entropy",
                     "hidden_heterogeneity"]
    lags = [1, 2, 3, 5]
    M = np.full((len(feature_names), len(lags)), np.nan)
    for i, fn in enumerate(feature_names):
        for j, lag in enumerate(lags):
            corrs = []
            for r in results:
                for (name, l, c) in r.feature_dynamics.leading_features:
                    if name == fn and l == lag:
                        corrs.append(c)
            if corrs:
                M[i, j] = float(np.mean(corrs))
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"lag={l}" for l in lags])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("Hidden-feature → visible-divergence lead-lag correlation")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, out_path)


def plot_example_candidate_pathway_trace(results, out_path):
    out_path = Path(out_path)
    if not results: _empty(out_path, "no candidates"); return
    r = max(results, key=_per_candidate_hce)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ts = np.arange(1, r.pathway.n_steps + 1)
    axes[0].plot(ts, r.pathway.hidden_mass_per_step, "-o", label="hidden mass",
                 color="#1f77b4")
    axes[0].plot(ts, r.pathway.visible_mass_per_step, "-s", label="visible mass",
                 color="#d62728")
    axes[0].set_ylabel("XOR mass"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].set_title(f"Pathway trace — track {r.candidate_id}, "
                      f"{r.rule_source}, mech={r.mechanism.label}")
    axes[1].plot(ts, r.pathway.spread_radius_4d, "-o", label="4D spread radius",
                 color="#1f77b4")
    axes[1].plot(ts, r.pathway.spread_radius_2d, "-s", label="2D spread radius",
                 color="#d62728")
    axes[1].set_xlabel("rollout step"); axes[1].set_ylabel("spread radius (cells)")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    _save(fig, out_path)


def write_all_m8_plots(results: list, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_response_map_examples_top_hce(
        results, out_dir / "response_map_examples_top_hce.png"
    )
    plot_boundary_vs_interior_response_by_source(
        results, out_dir / "boundary_vs_interior_response_by_source.png"
    )
    plot_first_visible_effect_time_by_source(
        results, out_dir / "first_visible_effect_time_by_source.png"
    )
    plot_hidden_mass_vs_visible_mass_over_time(
        results, out_dir / "hidden_mass_vs_visible_mass_over_time.png"
    )
    plot_hidden_to_visible_conversion_time_by_source(
        results, out_dir / "hidden_to_visible_conversion_time_by_source.png"
    )
    plot_hce_vs_lifetime_tradeoff(results, out_dir / "hce_vs_lifetime_tradeoff.png")
    plot_hidden_volatility_vs_lifetime(
        results, out_dir / "hidden_volatility_vs_lifetime.png"
    )
    plot_hce_vs_boundary_response_fraction(
        results, out_dir / "hce_vs_boundary_response_fraction.png"
    )
    plot_mechanism_class_distribution(
        results, out_dir / "mechanism_class_distribution.png"
    )
    plot_local_vs_far_effect_by_mechanism(
        results, out_dir / "local_vs_far_effect_by_mechanism.png"
    )
    plot_feature_lead_lag_heatmap(results, out_dir / "feature_lead_lag_heatmap.png")
    plot_example_candidate_pathway_trace(
        results, out_dir / "example_candidate_pathway_trace.png"
    )
