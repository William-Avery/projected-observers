"""M8C plots — 12 spec'd outputs.

Mostly distinct from M8B's set: M8C focuses on far-control distance
geometry, the global_chaotic vs distance relationship, and a
cross-source headline figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.detection.morphology import MORPHOLOGY_CLASSES
from observer_worlds.experiments._m8b_spatial import M8B_MECHANISM_CLASSES
from observer_worlds.experiments._m8c_validation import M8CCandidateResult


SOURCE_PALETTE = {
    "M7_HCE_optimized": "#1f77b4",
    "M4C_observer_optimized": "#ff7f0e",
    "M4A_viability": "#2ca02c",
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


def _hce(r): return float(r.region_effects["whole"].region_hidden_effect)


# 1
def plot_thick_fraction_by_source(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    sources = list(by_src.keys())
    fracs = [
        sum(1 for r in by_src[s] if _is_thick(r)) / max(len(by_src[s]), 1)
        for s in sources
    ]
    ns = [len(by_src[s]) for s in sources]
    ax.bar(sources, fracs, color=[SOURCE_PALETTE.get(s, "#999") for s in sources],
           alpha=0.7)
    for i, (n, f) in enumerate(zip(ns, fracs)):
        ax.text(i, f + 0.01, f"N={n}", ha="center", fontsize=9)
    ax.set_ylabel("thick-candidate fraction"); ax.set_ylim(0, 1.0)
    ax.set_title("Thick-candidate fraction by source (M8C)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    _save(fig, out_path)


# 2
def plot_mechanism_distribution_thick_only(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    fig, ax = plt.subplots(figsize=(13, 6))
    sources = list(by_src.keys()); width = 0.8 / max(len(sources), 1)
    x = np.arange(len(M8B_MECHANISM_CLASSES))
    for i, src in enumerate(sources):
        thick = [r for r in by_src[src] if _is_thick(r)]
        n = max(len(thick), 1)
        fracs = [sum(1 for r in thick if r.mechanism_label == c) / n
                 for c in M8B_MECHANISM_CLASSES]
        ax.bar(x + (i - len(sources) / 2 + 0.5) * width, fracs, width=width,
               label=f"{src} (N_thick={len(thick)})",
               color=SOURCE_PALETTE.get(src, "#999"), alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(M8B_MECHANISM_CLASSES, rotation=25,
                                        ha="right", fontsize=7)
    ax.set_ylabel("fraction of thick candidates")
    ax.set_title("Mechanism class distribution among THICK candidates (M8C)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 3
def plot_interior_vs_boundary_effect_scatter(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        if not thick: continue
        x = [r.region_effects["interior"].region_effect_per_cell for r in thick]
        y = [r.region_effects["boundary"].region_effect_per_cell for r in thick]
        ax.scatter(x, y, s=22, alpha=0.6,
                   label=f"{src} (N={len(thick)})",
                   color=SOURCE_PALETTE.get(src, "#999"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("interior effect per cell")
    ax.set_ylabel("boundary effect per cell")
    ax.set_title("Interior vs boundary per-cell effect (thick only, M8C)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 4
def plot_candidate_vs_far_effect_scatter(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r) and r.far_control.far_control_valid]
        if not thick: continue
        x = [_hce(r) for r in thick]
        y = [r.far_effect.region_hidden_effect for r in thick]
        ax.scatter(x, y, s=22, alpha=0.6,
                   label=f"{src} (N={len(thick)})",
                   color=SOURCE_PALETTE.get(src, "#999"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("candidate HCE_whole")
    ax.set_ylabel("far-control hidden effect")
    ax.set_title("Candidate vs far hidden effect (thick + far-valid, M8C)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 5
def plot_far_distance_over_radius_distribution(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, rs in _by_source(results).items():
        valid = [r.far_control.far_control_distance_over_radius
                 for r in rs if r.far_control.far_control_valid]
        if not valid: continue
        ax.hist(valid, bins=20, alpha=0.5, label=f"{src} (N={len(valid)})",
                color=SOURCE_PALETTE.get(src, "#999"))
    ax.axvline(5.0, color="k", linestyle="--", alpha=0.5,
               label="min required (5x radius)")
    ax.set_xlabel("far_control_distance / candidate_radius")
    ax.set_ylabel("count")
    ax.set_title("Far-control geometry quality (M8C)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 6
def plot_global_chaotic_fraction_vs_far_distance(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, rs in _by_source(results).items():
        thick_valid = [r for r in rs if _is_thick(r)
                       and r.far_control.far_control_valid]
        if not thick_valid: continue
        # Bin by distance/radius.
        dists = np.array([r.far_control.far_control_distance_over_radius
                          for r in thick_valid])
        gc = np.array([r.mechanism_label == "global_chaotic"
                       for r in thick_valid])
        bins = [0, 5, 10, 20, 40, 100]
        for i in range(len(bins) - 1):
            sel = (dists >= bins[i]) & (dists < bins[i + 1])
            if sel.any():
                ax.scatter(dists[sel].mean(), gc[sel].mean(),
                           s=80, alpha=0.7,
                           color=SOURCE_PALETTE.get(src, "#999"))
        ax.plot([], [], "o", color=SOURCE_PALETTE.get(src, "#999"),
                label=src)
    ax.set_xlabel("mean far_distance/radius (binned)")
    ax.set_ylabel("global_chaotic fraction in bin")
    ax.set_title("Global-chaotic rate vs far-control geometry quality")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 7
def plot_hce_by_candidate_area(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for src, rs in _by_source(results).items():
        x = [r.candidate_area for r in rs]; y = [_hce(r) for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6, label=src,
                   color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("candidate area"); ax.set_ylabel("HCE_whole")
    ax.set_title("HCE vs candidate area (M8C)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 8
def plot_hce_by_candidate_lifetime(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for src, rs in _by_source(results).items():
        x = [r.candidate_lifetime for r in rs]; y = [_hce(r) for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6, label=src,
                   color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("candidate lifetime"); ax.set_ylabel("HCE_whole")
    ax.set_title("HCE vs candidate lifetime (M8C)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 9
def plot_mechanism_by_candidate_area(results, out_path):
    out_path = Path(out_path)
    thick = [r for r in results if _is_thick(r)]
    if not thick: _empty(out_path, "no thick candidates"); return
    fig, ax = plt.subplots(figsize=(11, 6))
    classes = sorted({r.mechanism_label for r in thick})
    for cls in classes:
        sel = [r for r in thick if r.mechanism_label == cls]
        if not sel: continue
        x = [r.candidate_area for r in sel]
        y = [_hce(r) for r in sel]
        ax.scatter(x, y, s=22, alpha=0.6, label=cls)
    ax.set_xlabel("candidate area"); ax.set_ylabel("HCE_whole")
    ax.set_title("Mechanism vs candidate area (thick only, M8C)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    _save(fig, out_path)


# 10, 11
def _example_grid(results, out_path, label_pred, title):
    out_path = Path(out_path)
    examples = [r for r in results if label_pred(r)]
    if not examples: _empty(out_path, f"no examples for: {title}"); return
    examples = sorted(examples, key=lambda r: -_hce(r))[:6]
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, examples):
        ax.text(0.5, 0.5,
                f"{r.rule_source[:6]}\nrule={r.rule_id}\n"
                f"seed={r.seed}\narea={r.candidate_area} "
                f"life={r.candidate_lifetime}\n"
                f"morph={r.morphology.morphology_class}\n"
                f"mech={r.mechanism_label}\n"
                f"HCE={_hce(r):.4f}\n"
                f"d/r={r.far_control.far_control_distance_over_radius:.1f}",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    _save(fig, out_path)


def plot_whole_body_support_examples(results, out_path):
    _example_grid(
        results, out_path,
        label_pred=lambda r: r.mechanism_label == "whole_body_hidden_support",
        title="Examples: whole-body hidden support (M8C)",
    )


def plot_interior_reservoir_examples(results, out_path):
    _example_grid(
        results, out_path,
        label_pred=lambda r: r.mechanism_label == "interior_reservoir",
        title="Examples: interior reservoir (M8C)",
    )


# 12
def plot_source_comparison_summary(results, out_path):
    """Headline figure: per-source bars showing mean HCE, locality
    index, and thick-fraction side by side."""
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sources = list(by_src.keys())
    colors = [SOURCE_PALETTE.get(s, "#999") for s in sources]

    # HCE.
    hces = [np.mean([_hce(r) for r in by_src[s]]) if by_src[s] else 0.0
            for s in sources]
    axes[0].bar(sources, hces, color=colors, alpha=0.7)
    axes[0].set_title("Mean HCE_whole"); axes[0].set_ylabel("HCE")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Candidate locality (thick + valid far).
    locs = []
    for s in sources:
        thick_valid = [r for r in by_src[s]
                       if _is_thick(r) and r.far_control.far_control_valid]
        if thick_valid:
            locs.append(np.mean([_hce(r) - r.far_effect.region_hidden_effect
                                 for r in thick_valid]))
        else:
            locs.append(0.0)
    axes[1].bar(sources, locs, color=colors, alpha=0.7)
    axes[1].set_title("Candidate-locality (thick + valid far)")
    axes[1].set_ylabel("HCE − far_effect")
    axes[1].grid(True, axis="y", alpha=0.3)

    # Thick fraction.
    fracs = [sum(1 for r in by_src[s] if _is_thick(r)) / max(len(by_src[s]), 1)
             for s in sources]
    axes[2].bar(sources, fracs, color=colors, alpha=0.7)
    axes[2].set_title("Thick-candidate fraction")
    axes[2].set_ylabel("fraction")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, axis="y", alpha=0.3)

    for a in axes:
        plt.setp(a.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.suptitle("M8C source comparison summary", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


def write_all_m8c_plots(results: list, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_thick_fraction_by_source(results, out_dir / "thick_fraction_by_source.png")
    plot_mechanism_distribution_thick_only(
        results, out_dir / "mechanism_distribution_thick_only.png")
    plot_interior_vs_boundary_effect_scatter(
        results, out_dir / "interior_vs_boundary_effect_scatter.png")
    plot_candidate_vs_far_effect_scatter(
        results, out_dir / "candidate_vs_far_effect_scatter.png")
    plot_far_distance_over_radius_distribution(
        results, out_dir / "far_distance_over_radius_distribution.png")
    plot_global_chaotic_fraction_vs_far_distance(
        results, out_dir / "global_chaotic_fraction_vs_far_distance.png")
    plot_hce_by_candidate_area(results, out_dir / "hce_by_candidate_area.png")
    plot_hce_by_candidate_lifetime(results, out_dir / "hce_by_candidate_lifetime.png")
    plot_mechanism_by_candidate_area(
        results, out_dir / "mechanism_by_candidate_area.png")
    plot_whole_body_support_examples(
        results, out_dir / "whole_body_support_examples.png")
    plot_interior_reservoir_examples(
        results, out_dir / "interior_reservoir_examples.png")
    plot_source_comparison_summary(
        results, out_dir / "source_comparison_summary.png")
