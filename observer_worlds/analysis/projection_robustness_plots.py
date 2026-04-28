"""Plot generators for Follow-up Topic 1.

Stage 2: produce the seven plots specified in the roadmap. The two
plots that require revised mechanism labels are written as placeholders
(empty bars + a "Stage 5+" note) until M8G integration across
projections is added.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Stage-1 surface kept for tests:
PLOT_FILENAMES: tuple[str, ...] = (
    "hce_by_projection.png",
    "hidden_vs_far_by_projection.png",
    "observer_score_by_projection.png",
    "candidate_count_by_projection.png",
    "mechanism_distribution_by_projection.png",
    "hce_within_revised_class_by_projection.png",
    "initial_projection_delta_by_projection.png",
)


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _projections_in_order(summary: dict) -> list[str]:
    """Preserve the order in which projections were evaluated. Falls back
    to the keys' default order."""
    explicit = summary.get("projections_evaluated")
    if explicit:
        return list(explicit)
    return list(summary.get("per_projection", {}))


def _bar_chart(
    plt, projections: list[str], values: list[float | None],
    *, title: str, ylabel: str,
    color: str = "#357",
):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(projections))
    plotted = [0.0 if v is None else float(v) for v in values]
    bars = ax.bar(x, plotted, color=color, edgecolor="white")
    for i, v in enumerate(values):
        if v is None:
            bars[i].set_alpha(0.25)
            ax.text(x[i], 0, "n/a", ha="center", va="bottom",
                    fontsize=8, color="#888")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in projections],
                       fontsize=8, rotation=0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def write_all_plots(
    summary: dict,
    candidate_rows: list[dict],
    plots_dir: Path,
) -> None:
    """Emit the seven roadmap plots into ``plots_dir``.

    No exception escapes — failures per-plot are caught and reported as
    print warnings so a missing matplotlib does not kill the run.
    """
    try:
        plt = _import_plt()
    except ImportError:
        print("  [warn] matplotlib unavailable; skipping plots")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    projs = _projections_in_order(summary)
    per = summary.get("per_projection", {})

    # 1. HCE by projection.
    try:
        vals = [per.get(p, {}).get("mean_HCE") for p in projs]
        fig = _bar_chart(plt, projs, vals,
                         title="mean HCE by projection (Stage 2 smoke)",
                         ylabel="mean per-candidate HCE",
                         color="#3a7")
        fig.savefig(plots_dir / "hce_by_projection.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hce_by_projection.png: {e!r}")

    # 2. Hidden vs far by projection.
    try:
        vals = [per.get(p, {}).get("mean_hidden_vs_far_delta") for p in projs]
        fig = _bar_chart(plt, projs, vals,
                         title="mean (HCE − far_HCE) by projection",
                         ylabel="mean hidden_vs_far_delta")
        fig.savefig(plots_dir / "hidden_vs_far_by_projection.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hidden_vs_far_by_projection.png: {e!r}")

    # 3. Observer score by projection (proxy: mean lifetime — full
    #    observer_score battery is deferred).
    try:
        vals = [per.get(p, {}).get("mean_lifetime") for p in projs]
        fig = _bar_chart(plt, projs, vals,
                         title="mean candidate lifetime by projection "
                               "(proxy for observer_score, Stage 2)",
                         ylabel="mean candidate lifetime (frames)",
                         color="#a73")
        fig.savefig(plots_dir / "observer_score_by_projection.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] observer_score_by_projection.png: {e!r}")

    # 4. Candidate count by projection.
    try:
        vals = [per.get(p, {}).get("n_candidates", 0) for p in projs]
        fig = _bar_chart(plt, projs, [float(v) for v in vals],
                         title="candidate count by projection",
                         ylabel="number of candidates measured",
                         color="#357")
        fig.savefig(plots_dir / "candidate_count_by_projection.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] candidate_count_by_projection.png: {e!r}")

    # 5. Mechanism distribution placeholder.
    try:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5,
                "Stage 5+: mechanism distribution per projection requires\n"
                "full M8G classifier integration across projections.\n"
                "Not implemented in Stage 2; see "
                "docs/FOLLOWUP_RESEARCH_ROADMAP.md",
                ha="center", va="center", fontsize=11, color="#666",
                transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(plots_dir / "mechanism_distribution_by_projection.png",
                    dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] mechanism_distribution_by_projection.png: {e!r}")

    # 6. HCE within revised class — same placeholder.
    try:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5,
                "Stage 5+: HCE within revised mechanism class per "
                "projection.\nNot implemented in Stage 2.",
                ha="center", va="center", fontsize=11, color="#666",
                transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(plots_dir / "hce_within_revised_class_by_projection.png",
                    dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hce_within_revised_class_by_projection.png: {e!r}")

    # 7. Initial projection delta by projection.
    try:
        vals = [per.get(p, {}).get("mean_initial_projection_delta")
                for p in projs]
        fig = _bar_chart(
            plt, projs, vals,
            title="mean initial_projection_delta by projection "
                  "(0 = clean hidden-invisible perturbation)",
            ylabel="mean |P(perturbed) - P(unperturbed)| at peak frame",
            color="#a37",
        )
        fig.savefig(plots_dir / "initial_projection_delta_by_projection.png",
                    dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] initial_projection_delta_by_projection.png: {e!r}")


def write_all(*_, **__) -> None:
    """Stage-1 surface — kept for backward compat with tests."""
    raise NotImplementedError(
        "use write_all_plots(summary, candidate_rows, plots_dir) instead."
    )
