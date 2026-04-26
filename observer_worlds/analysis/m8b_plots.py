"""M8B plots — 12 spec'd outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.detection.morphology import MORPHOLOGY_CLASSES
from observer_worlds.experiments._m8b_spatial import (
    M8B_MECHANISM_CLASSES,
    M8BCandidateResult,
)


SOURCE_PALETTE = {
    "M7_HCE_optimized": "#1f77b4",
    "M4C_observer_optimized": "#ff7f0e",
    "M4A_viability": "#2ca02c",
}


def _save(fig, out_path: Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _empty(out_path: Path, msg: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    _save(fig, out_path)


def _by_source(rs):
    out: dict = {}
    for r in rs: out.setdefault(r.rule_source, []).append(r)
    return out


def _hce(r): return float(r.region_effects["whole"].region_hidden_effect)


def _is_thick(r):
    return r.morphology.morphology_class in (
        "thick_candidate", "very_thick_candidate"
    )


# 1
def plot_morphology_class_distribution(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    fig, ax = plt.subplots(figsize=(11, 5))
    sources = list(by_src.keys()); width = 0.8 / max(len(sources), 1)
    x = np.arange(len(MORPHOLOGY_CLASSES))
    for i, src in enumerate(sources):
        counts = [sum(1 for r in by_src[src]
                     if r.morphology.morphology_class == c)
                 for c in MORPHOLOGY_CLASSES]
        total = max(len(by_src[src]), 1)
        fracs = [c / total for c in counts]
        ax.bar(x + (i - len(sources) / 2 + 0.5) * width, fracs, width=width,
               label=f"{src} (N={total})",
               color=SOURCE_PALETTE.get(src, "#999"), alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(MORPHOLOGY_CLASSES, rotation=20,
                                        ha="right", fontsize=8)
    ax.set_ylabel("fraction of candidates")
    ax.set_title("Morphology class distribution by source")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 2
def plot_region_effect_per_cell_by_source(results, out_path):
    out_path = Path(out_path)
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    regions = ["interior", "boundary", "environment"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sources = list(by_src.keys()); width = 0.25
    x = np.arange(len(sources))
    for j, region in enumerate(regions):
        means = [np.mean([r.region_effects[region].region_effect_per_cell
                          for r in by_src[s]]) if by_src[s] else 0.0
                 for s in sources]
        ax.bar(x + (j - 1) * width, means, width=width, label=region, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=15, ha="right",
                                        fontsize=8)
    ax.set_ylabel("effect per perturbed 4D cell")
    ax.set_title("Region effect per cell by source (all candidates)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out_path)


# 3
def plot_boundary_vs_interior_effect_scatter(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        if not thick: continue
        x = [r.region_effects["interior"].region_effect_per_cell for r in thick]
        y = [r.region_effects["boundary"].region_effect_per_cell for r in thick]
        ax.scatter(x, y, s=25, alpha=0.6, label=f"{src} (N={len(thick)})",
                   color=SOURCE_PALETTE.get(src, "#999"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("interior effect per cell")
    ax.set_ylabel("boundary effect per cell")
    ax.set_title("Boundary vs interior per-cell effect (thick only)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 4
def plot_environment_vs_candidate_effect_scatter(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    for src, rs in _by_source(results).items():
        thick = [r for r in rs if _is_thick(r)]
        if not thick: continue
        x = [max(r.region_effects["interior"].region_effect_per_cell,
                 r.region_effects["boundary"].region_effect_per_cell)
             for r in thick]
        y = [r.region_effects["environment"].region_effect_per_cell
             for r in thick]
        ax.scatter(x, y, s=25, alpha=0.6, label=f"{src} (N={len(thick)})",
                   color=SOURCE_PALETTE.get(src, "#999"))
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1e-6)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("max(interior, boundary) per-cell")
    ax.set_ylabel("environment per-cell")
    ax.set_title("Environment vs candidate per-cell effect (thick only)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 5
def plot_mechanism_class_distribution_thick_only(results, out_path):
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
    ax.set_title("Mechanism class distribution among THICK candidates")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 6
def plot_thin_vs_thick_hce(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    sources = list(_by_source(results).keys())
    width = 0.35; x = np.arange(len(sources))
    thick_means = []; thin_means = []
    for src in sources:
        rs = _by_source(results)[src]
        thick = [_hce(r) for r in rs if _is_thick(r)]
        thin = [_hce(r) for r in rs
                if r.morphology.morphology_class == "thin_candidate"]
        thick_means.append(np.mean(thick) if thick else 0.0)
        thin_means.append(np.mean(thin) if thin else 0.0)
    ax.bar(x - width/2, thick_means, width=width, label="thick", color="#1f77b4")
    ax.bar(x + width/2, thin_means, width=width, label="thin", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=15, ha="right",
                                        fontsize=8)
    ax.set_ylabel("mean HCE (whole-candidate)")
    ax.set_title("Thin vs thick HCE per source")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out_path)


# 7
def plot_region_effect_by_candidate_area(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for src, rs in _by_source(results).items():
        x = [r.candidate_area for r in rs]
        y = [_hce(r) for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6, label=src,
                   color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("candidate area"); ax.set_ylabel("HCE (whole)")
    ax.set_title("HCE vs candidate area")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 8
def plot_region_effect_by_lifetime(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for src, rs in _by_source(results).items():
        x = [r.candidate_lifetime for r in rs]
        y = [_hce(r) for r in rs]
        ax.scatter(x, y, s=20, alpha=0.6, label=src,
                   color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("candidate lifetime"); ax.set_ylabel("HCE (whole)")
    ax.set_title("HCE vs candidate lifetime")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 9
def plot_environment_coupling_by_shell_width(results, out_path):
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    by_src = _by_source(results)
    if not by_src: _empty(out_path, "no candidates"); return
    widths = sorted({int(k.split("_w")[1])
                     for r in results
                     for k in r.region_effects.keys()
                     if k.startswith("environment_w")})
    if not widths:
        widths = []
    widths = [1] + widths
    for src, rs in by_src.items():
        thick = [r for r in rs if _is_thick(r)]
        if not thick: continue
        ys = []
        for w in widths:
            key = "environment" if w == 1 else f"environment_w{w}"
            vals = [r.region_effects[key].region_effect_per_cell
                    for r in thick if key in r.region_effects]
            ys.append(np.mean(vals) if vals else 0.0)
        ax.plot(widths, ys, marker="o", label=src,
                color=SOURCE_PALETTE.get(src, "#999"))
    ax.set_xlabel("environment shell width")
    ax.set_ylabel("env effect per cell")
    ax.set_title("Environment coupling by shell width (thick only)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_path)


# 10, 11, 12: example candidate plots
def _plot_example_grid(results, out_path, *, label_pred, title):
    out_path = Path(out_path)
    examples = [r for r in results if label_pred(r)]
    if not examples:
        _empty(out_path, f"no examples for: {title}"); return
    examples = sorted(examples, key=lambda r: -_hce(r))[:6]
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, examples):
        # Show interior (red) + boundary (orange) + env (blue) using
        # the morphology shells.
        from observer_worlds.detection.morphology import shell_masks_strict
        mask = np.zeros_like(r.region_effects["whole"].n_perturbed_cells_2d
                             if False else None, dtype=np.uint8)
        # Reconstruct rendering from candidate_area only — we don't store
        # the full mask in M8BCandidateResult. So just render a summary.
        ax.text(0.5, 0.5,
                f"{r.rule_source[:6]}\nrule={r.rule_id}\nseed={r.seed}\n"
                f"area={r.candidate_area} life={r.candidate_lifetime}\n"
                f"morph={r.morphology.morphology_class}\n"
                f"mech={r.mechanism_label}\n"
                f"HCE={_hce(r):.3f}\nconf={r.mechanism_confidence:.2f}",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


def plot_whole_body_support_examples(results, out_path):
    _plot_example_grid(
        results, out_path,
        label_pred=lambda r: r.mechanism_label == "whole_body_hidden_support",
        title="Examples: whole-body hidden support",
    )


def plot_boundary_mediated_examples(results, out_path):
    _plot_example_grid(
        results, out_path,
        label_pred=lambda r: r.mechanism_label == "boundary_mediated",
        title="Examples: boundary-mediated",
    )


def plot_interior_reservoir_examples(results, out_path):
    _plot_example_grid(
        results, out_path,
        label_pred=lambda r: r.mechanism_label == "interior_reservoir",
        title="Examples: interior reservoir",
    )


def write_all_m8b_plots(results: list, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_morphology_class_distribution(
        results, out_dir / "morphology_class_distribution.png")
    plot_region_effect_per_cell_by_source(
        results, out_dir / "region_effect_per_cell_by_source.png")
    plot_boundary_vs_interior_effect_scatter(
        results, out_dir / "boundary_vs_interior_effect_scatter.png")
    plot_environment_vs_candidate_effect_scatter(
        results, out_dir / "environment_vs_candidate_effect_scatter.png")
    plot_mechanism_class_distribution_thick_only(
        results, out_dir / "mechanism_class_distribution_thick_only.png")
    plot_thin_vs_thick_hce(results, out_dir / "thin_vs_thick_hce.png")
    plot_region_effect_by_candidate_area(
        results, out_dir / "region_effect_by_candidate_area.png")
    plot_region_effect_by_lifetime(
        results, out_dir / "region_effect_by_lifetime.png")
    plot_environment_coupling_by_shell_width(
        results, out_dir / "environment_coupling_by_shell_width.png")
    plot_whole_body_support_examples(
        results, out_dir / "whole_body_support_examples.png")
    plot_boundary_mediated_examples(
        results, out_dir / "boundary_mediated_examples.png")
    plot_interior_reservoir_examples(
        results, out_dir / "interior_reservoir_examples.png")
