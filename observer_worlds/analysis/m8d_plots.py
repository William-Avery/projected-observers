"""M8D plots — 12 spec'd outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.experiments._m8d_decomposition import (
    M8D_GLOBAL_SUBCLASSES, M8DCandidateResult,
)


SOURCE_PALETTE = {
    "M7_HCE_optimized": "#1f77b4",
    "M4C_observer_optimized": "#ff7f0e",
    "M4A_viability": "#2ca02c",
}
SUBCLASS_PALETTE = {
    "global_instability": "#d62728",
    "broad_hidden_coupling": "#9467bd",
    "background_sensitive_world": "#17becf",
    "far_control_artifact": "#7f7f7f",
    "threshold_volatility_artifact": "#bcbd22",
    "unresolved_global": "#8c564b",
}


def _save(fig, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _empty(p, msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off"); _save(fig, p)


def _by_source(rs):
    out = {}
    for r in rs: out.setdefault(r.rule_source, []).append(r)
    return out


def _is_thick(r):
    return r.morphology.morphology_class in (
        "thick_candidate", "very_thick_candidate"
    )


def _global_thick(rs):
    return [r for r in rs if _is_thick(r)
            and r.base_mechanism_label == "global_chaotic"]


# 1
def plot_effect_decay_by_distance(results, out_path):
    out_path = Path(out_path)
    g = [r for r in results
         if _is_thick(r) and r.base_mechanism_label == "global_chaotic"]
    if not g: _empty(out_path, "no global_chaotic thick"); return
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in g:
        xs = [e.distance for e in r.distance_effects if e.n_perturbed_2d > 0]
        ys = [e.effect_per_cell for e in r.distance_effects
              if e.n_perturbed_2d > 0]
        ax.plot(xs, ys, alpha=0.4, marker="o",
                color=SOURCE_PALETTE.get(r.rule_source, "#999"))
    ax.set_xlabel("perturbation distance (cells)")
    ax.set_ylabel("per-cell effect")
    ax.set_title(f"Effect decay by distance (global_chaotic thick, N={len(g)})")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


# 2
def plot_global_vs_interior_decay_curves(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in (("global_chaotic", "#d62728"),
                         ("interior_reservoir", "#1f77b4"),
                         ("whole_body_hidden_support", "#9467bd")):
        rs = [r for r in results if _is_thick(r)
              and r.base_mechanism_label == label]
        if not rs: continue
        # Aggregate per-distance mean.
        bins: dict = defaultdict(list)
        for r in rs:
            for e in r.distance_effects:
                if e.n_perturbed_2d > 0:
                    key = round(e.distance / 4) * 4
                    bins[key].append(e.effect_per_cell)
        if not bins: continue
        xs = sorted(bins.keys())
        ys = [np.mean(bins[k]) for k in xs]
        ax.plot(xs, ys, marker="o", linewidth=2,
                label=f"{label} (N={len(rs)})", color=color)
    ax.set_xlabel("perturbation distance (cells, binned)")
    ax.set_ylabel("mean per-cell effect")
    ax.set_title("Effect decay: global_chaotic vs interior vs whole-body")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out_path)


# 3
def plot_candidate_body_vs_background_effect(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        if not thick: continue
        body = [next((e.raw_effect for e in r.distance_effects
                     if e.name == "body"), 0.0) for r in thick]
        bg = [r.background_mean for r in thick]
        ax.scatter(bg, body, s=22, alpha=0.6,
                   label=f"{src} (N={len(thick)})",
                   color=SOURCE_PALETTE.get(src, "#999"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("background mean effect"); ax.set_ylabel("body raw effect")
    ax.set_title("Candidate body vs background mean (thick)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 4
def plot_far_effect_vs_background_distribution(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        if not thick: continue
        far = [r.far_effect.region_hidden_effect for r in thick]
        bg_p95 = [r.background_p95 for r in thick]
        ax.scatter(bg_p95, far, s=22, alpha=0.6,
                   label=f"{src} (N={len(thick)})",
                   color=SOURCE_PALETTE.get(src, "#999"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("background p95 effect")
    ax.set_ylabel("far effect (validated antipode)")
    ax.set_title("Far effect vs background p95")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 5
def plot_global_class_feature_comparison(results, out_path):
    out_path = Path(out_path)
    g = [r for r in results if _is_thick(r)
         and r.base_mechanism_label == "global_chaotic"]
    l = [r for r in results if _is_thick(r) and r.base_mechanism_label
         in ("interior_reservoir", "whole_body_hidden_support")]
    if not (g and l):
        _empty(out_path, "insufficient comparison data"); return
    feat_keys = list(g[0].feature_audit.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35; x = np.arange(len(feat_keys))
    g_means = [np.mean([r.feature_audit.get(k, 0) for r in g])
               for k in feat_keys]
    l_means = [np.mean([r.feature_audit.get(k, 0) for r in l])
               for k in feat_keys]
    ax.bar(x - width/2, g_means, width=width, label=f"global (N={len(g)})",
           color="#d62728", alpha=0.7)
    ax.bar(x + width/2, l_means, width=width, label=f"interior/whole (N={len(l)})",
           color="#1f77b4", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(feat_keys, rotation=20, ha="right",
                                        fontsize=8)
    ax.set_ylabel("mean feature value")
    ax.set_title("Hidden-feature comparison: global vs interior/whole")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out_path)


# 6
def plot_threshold_margin_global_vs_local(results, out_path):
    _violin_compare(
        results, out_path,
        feature_key="mean_threshold_margin",
        title="mean_threshold_margin: global vs interior/whole",
    )


# 7
def plot_hidden_volatility_global_vs_local(results, out_path):
    _violin_compare(
        results, out_path,
        feature_key="hidden_volatility",
        title="hidden_volatility: global vs interior/whole",
    )


def _violin_compare(results, out_path, *, feature_key, title):
    out_path = Path(out_path)
    g = [r.feature_audit.get(feature_key, 0.0) for r in results
         if _is_thick(r) and r.base_mechanism_label == "global_chaotic"]
    l = [r.feature_audit.get(feature_key, 0.0) for r in results
         if _is_thick(r) and r.base_mechanism_label
         in ("interior_reservoir", "whole_body_hidden_support")]
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [d for d in (g, l) if d]
    labels = [f"{n} (N={len(d)})" for n, d in
              zip(("global", "interior/whole"), (g, l)) if d]
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel(feature_key)
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


# 8
def plot_stabilization_reclassification_rates(results, out_path):
    out_path = Path(out_path)
    g = [r for r in results if _is_thick(r)
         and r.base_mechanism_label == "global_chaotic"]
    if not g: _empty(out_path, "no global_chaotic thick"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = ("baseline", "short_horizon", "local_window")
    fires = []
    for v in variants:
        vals = []
        for r in g:
            d = r.stabilization.get(v)
            if d and "global_chaotic_label_would_fire" in d:
                vals.append(d["global_chaotic_label_would_fire"])
        fires.append(np.mean(vals) if vals else 0.0)
    ax.bar(variants, [1.0 - f for f in fires], color="#1f77b4", alpha=0.7)
    ax.set_ylabel("fraction no longer global_chaotic")
    ax.set_title(f"Stabilization reclassification (N={len(g)})")
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, 1.0)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    _save(fig, out_path)


# 9
def plot_global_subclass_distribution(results, out_path):
    out_path = Path(out_path)
    g = [r for r in results if _is_thick(r)
         and r.base_mechanism_label == "global_chaotic"]
    if not g: _empty(out_path, "no global_chaotic thick"); return
    fig, ax = plt.subplots(figsize=(11, 5))
    by_src = _by_source(g); sources = list(by_src.keys())
    if not sources: _empty(out_path, "no sources"); return
    width = 0.8 / max(len(sources), 1); x = np.arange(len(M8D_GLOBAL_SUBCLASSES))
    for i, src in enumerate(sources):
        n = max(len(by_src[src]), 1)
        fracs = [sum(1 for r in by_src[src] if r.final_mechanism_label == c) / n
                 for c in M8D_GLOBAL_SUBCLASSES]
        ax.bar(x + (i - len(sources) / 2 + 0.5) * width, fracs, width=width,
               label=f"{src} (N={n})",
               color=SOURCE_PALETTE.get(src, "#999"), alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(M8D_GLOBAL_SUBCLASSES, rotation=20,
                                        ha="right", fontsize=8)
    ax.set_ylabel("fraction of global_chaotic candidates")
    ax.set_title("Global subclass distribution (M8D relabel)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 10
def plot_body_over_background_by_subclass(results, out_path):
    _box_by_subclass(
        results, out_path, key="body_over_background",
        title="body_over_background by global subclass",
    )


# 11
def plot_far_over_background_by_subclass(results, out_path):
    _box_by_subclass(
        results, out_path, key="far_over_background",
        title="far_over_background by global subclass",
    )


def _box_by_subclass(results, out_path, *, key, title):
    out_path = Path(out_path)
    g = [r for r in results if _is_thick(r)
         and r.base_mechanism_label == "global_chaotic"]
    if not g: _empty(out_path, "no global_chaotic thick"); return
    fig, ax = plt.subplots(figsize=(11, 5))
    data, labels = [], []
    for cls in M8D_GLOBAL_SUBCLASSES:
        vals = [getattr(r, key) for r in g if r.final_mechanism_label == cls]
        if vals:
            data.append(vals); labels.append(f"{cls}\n(N={len(vals)})")
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel(key); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7)
    _save(fig, out_path)


# 12
def plot_example_global_candidate_traces(results, out_path):
    out_path = Path(out_path)
    g = [r for r in results if _is_thick(r)
         and r.base_mechanism_label == "global_chaotic"]
    if not g: _empty(out_path, "no global_chaotic thick"); return
    examples = sorted(g, key=lambda r: -r.body_over_background)[:6]
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, examples):
        xs = [e.distance for e in r.distance_effects if e.n_perturbed_2d > 0]
        ys = [e.effect_per_cell for e in r.distance_effects
              if e.n_perturbed_2d > 0]
        ax.plot(xs, ys, marker="o", color="#d62728")
        ax.axhline(r.background_mean / max(int(r.candidate_area), 1) / 64,
                   color="k", linestyle="--", alpha=0.4,
                   label="≈ background scale")
        ax.set_title(f"{r.rule_source[:6]}\nseed={r.seed}\ntrack={r.candidate_id}\n"
                     f"final={r.final_mechanism_label}",
                     fontsize=7)
        ax.set_xlabel("distance"); ax.set_ylabel("per-cell")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Example global_chaotic candidates: distance traces", fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


def write_all_m8d_plots(results: list, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_effect_decay_by_distance(results, out_dir / "effect_decay_by_distance.png")
    plot_global_vs_interior_decay_curves(
        results, out_dir / "global_vs_interior_decay_curves.png")
    plot_candidate_body_vs_background_effect(
        results, out_dir / "candidate_body_vs_background_effect.png")
    plot_far_effect_vs_background_distribution(
        results, out_dir / "far_effect_vs_background_distribution.png")
    plot_global_class_feature_comparison(
        results, out_dir / "global_class_feature_comparison.png")
    plot_threshold_margin_global_vs_local(
        results, out_dir / "threshold_margin_global_vs_local.png")
    plot_hidden_volatility_global_vs_local(
        results, out_dir / "hidden_volatility_global_vs_local.png")
    plot_stabilization_reclassification_rates(
        results, out_dir / "stabilization_reclassification_rates.png")
    plot_global_subclass_distribution(
        results, out_dir / "global_subclass_distribution.png")
    plot_body_over_background_by_subclass(
        results, out_dir / "body_over_background_by_subclass.png")
    plot_far_over_background_by_subclass(
        results, out_dir / "far_over_background_by_subclass.png")
    plot_example_global_candidate_traces(
        results, out_dir / "example_global_candidate_traces.png")
