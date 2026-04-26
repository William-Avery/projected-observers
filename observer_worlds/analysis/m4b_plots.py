"""Plot helpers for the M4B observer-metric sweep.

Consumes :class:`PairedRecord` objects produced by
``observer_worlds.experiments._m4b_sweep.run_sweep`` and emits the nine
M4B-spec plots plus the top-candidate videos.

All plot functions are headless (Agg backend) and write a PNG to a path
the caller chooses.  ``write_all_m4b_plots`` is the convenience entry
point used by the experiment driver.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from observer_worlds.experiments._m4b_sweep import (  # noqa: E402
    CONDITION_NAMES,
    PairedRecord,
    metrics_dict,
)
from observer_worlds.analysis.videos import write_projected_gif  # noqa: E402


# Stable color palette (used everywhere a condition needs a color).
CONDITION_COLORS: dict[str, str] = {
    "coherent_4d": "#1f77b4",
    "shuffled_4d": "#ff7f0e",
    "matched_2d": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_figure(message: str, out_path: str | Path) -> None:
    """Render a centered "no data" placeholder figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        transform=ax.transAxes, fontsize=14,
    )
    ax.set_axis_off()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _ensure_parent(out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _values_for_metric(records: list[PairedRecord], cond: str, metric: str) -> list[float]:
    out: list[float] = []
    for rec in records:
        cr = getattr(rec, cond)
        try:
            out.append(float(getattr(cr, metric)))
        except AttributeError:
            # Fall back through metrics_dict if it isn't a direct attribute.
            md = metrics_dict(cr)
            out.append(float(md.get(metric, 0.0)))
    return out


# ---------------------------------------------------------------------------
# 1. Boxplot by condition
# ---------------------------------------------------------------------------


def plot_condition_boxplot(
    records: list[PairedRecord],
    metric: str,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Boxplot of `metric` grouped by condition."""
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    data = [_values_for_metric(records, cond, metric) for cond in CONDITION_NAMES]
    if all(len(d) == 0 for d in data):
        _empty_figure("no data", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        tick_labels=list(CONDITION_NAMES),
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 7,
        },
        patch_artist=True,
    )
    for patch, cond in zip(bp["boxes"], CONDITION_NAMES):
        patch.set_facecolor(CONDITION_COLORS[cond])
        patch.set_alpha(0.5)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by condition")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Violin by condition
# ---------------------------------------------------------------------------


def plot_condition_violin(
    records: list[PairedRecord],
    metric: str,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Violin plot of `metric` grouped by condition."""
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    data = [_values_for_metric(records, cond, metric) for cond in CONDITION_NAMES]
    # A violinplot needs each group to be non-empty AND have variance.
    # Violinplot raises on empty arrays; substitute a singleton zero.
    safe_data = [d if len(d) > 0 else [0.0] for d in data]
    if all(len(d) == 0 for d in data):
        _empty_figure("no data", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(
        safe_data,
        positions=list(range(1, len(safe_data) + 1)),
        showmeans=True,
        showmedians=True,
    )
    bodies = parts.get("bodies", []) if isinstance(parts, dict) else parts["bodies"]
    for body, cond in zip(bodies, CONDITION_NAMES):
        body.set_facecolor(CONDITION_COLORS[cond])
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    ax.set_xticks(list(range(1, len(CONDITION_NAMES) + 1)))
    ax.set_xticklabels(list(CONDITION_NAMES))
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by condition (violin)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Paired lines (A vs B)
# ---------------------------------------------------------------------------


def plot_paired_lines(
    records: list[PairedRecord],
    metric: str,
    condition_a: str,
    condition_b: str,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Two-column scatter (paired) connecting per-pair A vs B values."""
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    a_vals = _values_for_metric(records, condition_a, metric)
    b_vals = _values_for_metric(records, condition_b, metric)
    if not a_vals and not b_vals:
        _empty_figure("no data", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x_a, x_b = 0.0, 1.0
    for a, b in zip(a_vals, b_vals):
        if a > b:
            color = "#2ca02c"  # green: A wins
        elif b > a:
            color = "#d62728"  # red: B wins
        else:
            color = "#888888"  # gray: tie
        ax.plot([x_a, x_b], [a, b], color=color, alpha=0.5, linewidth=1.0)
        ax.plot(x_a, a, marker="o", markersize=5,
                markerfacecolor=CONDITION_COLORS.get(condition_a, "black"),
                markeredgecolor="black", linestyle="None")
        ax.plot(x_b, b, marker="o", markersize=5,
                markerfacecolor=CONDITION_COLORS.get(condition_b, "black"),
                markeredgecolor="black", linestyle="None")

    ax.set_xticks([x_a, x_b])
    ax.set_xticklabels([condition_a, condition_b])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Paired {metric}: {condition_a} vs {condition_b}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Scatter score vs n_tracks
# ---------------------------------------------------------------------------


def plot_scatter_score_vs_tracks(
    records: list[PairedRecord],
    score_metric: str,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Scatter (n_tracks, score_metric) per (rule, seed, condition)."""
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    any_drawn = False
    for cond in CONDITION_NAMES:
        x_vals: list[float] = []
        y_vals: list[float] = []
        for rec in records:
            cr = getattr(rec, cond)
            x_vals.append(float(cr.n_tracks))
            try:
                y_vals.append(float(getattr(cr, score_metric)))
            except AttributeError:
                y_vals.append(float(metrics_dict(cr).get(score_metric, 0.0)))
        if not x_vals:
            continue
        any_drawn = True
        ax.scatter(
            x_vals, y_vals,
            color=CONDITION_COLORS[cond],
            label=cond,
            alpha=0.65,
            edgecolors="black",
            linewidths=0.5,
            s=40,
        )
        # Faint regression line (deg=1) when we have at least 2 points and
        # variation in x.
        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        if x_arr.size >= 2 and float(np.ptp(x_arr)) > 0:
            try:
                coeffs = np.polyfit(x_arr, y_arr, deg=1)
                xs = np.linspace(x_arr.min(), x_arr.max(), 50)
                ax.plot(
                    xs, np.polyval(coeffs, xs),
                    color=CONDITION_COLORS[cond], alpha=0.3, linewidth=1.5,
                )
            except (np.linalg.LinAlgError, ValueError):
                pass

    if not any_drawn:
        plt.close(fig)
        _empty_figure("no data", out_path)
        return

    ax.set_xlabel("n_tracks")
    ax.set_ylabel(score_metric)
    ax.set_title(title or f"{score_metric} vs n_tracks by condition")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Track count by condition (diagnostic)
# ---------------------------------------------------------------------------


def plot_track_count_by_condition(
    records: list[PairedRecord],
    out_path: str | Path,
) -> None:
    """Boxplot of n_tracks per condition (diagnostic baseline plot)."""
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    data = [_values_for_metric(records, cond, "n_tracks") for cond in CONDITION_NAMES]
    if all(len(d) == 0 for d in data):
        _empty_figure("no data", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        tick_labels=list(CONDITION_NAMES),
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 7,
        },
        patch_artist=True,
    )
    for patch, cond in zip(bp["boxes"], CONDITION_NAMES):
        patch.set_facecolor(CONDITION_COLORS[cond])
        patch.set_alpha(0.5)
    ax.set_ylabel("n_tracks")
    ax.set_title("Track count by condition (diagnostic baseline)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Best-candidate metric breakdown
# ---------------------------------------------------------------------------


_BEST_METRICS: tuple[str, ...] = (
    "best_time_score",
    "best_memory_score",
    "best_selfhood_score",
    "best_causality_score",
    "best_resilience_score",
    "best_persistence",
)


def plot_metric_breakdown_best_candidates(
    records: list[PairedRecord],
    out_path: str | Path,
) -> None:
    """Grouped bar chart of mean best-candidate metric per condition."""
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    # means[metric][cond] = mean across pairs (ignoring None).
    means: dict[str, dict[str, float]] = {}
    for metric in _BEST_METRICS:
        means[metric] = {}
        for cond in CONDITION_NAMES:
            vals: list[float] = []
            for rec in records:
                v = getattr(getattr(rec, cond), metric)
                if v is None:
                    continue
                try:
                    vf = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(vf):
                    continue
                vals.append(vf)
            means[metric][cond] = float(np.mean(vals)) if vals else 0.0

    n_metrics = len(_BEST_METRICS)
    n_conds = len(CONDITION_NAMES)
    bar_width = 0.8 / n_conds
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for ci, cond in enumerate(CONDITION_NAMES):
        heights = [means[m][cond] for m in _BEST_METRICS]
        offsets = (ci - (n_conds - 1) / 2.0) * bar_width
        ax.bar(
            x + offsets,
            heights,
            width=bar_width,
            color=CONDITION_COLORS[cond],
            label=cond,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(_BEST_METRICS, rotation=20, ha="right")
    ax.set_ylabel("mean (best candidate per pair)")
    ax.set_title("Best-candidate metric breakdown by condition")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Bootstrap CI forest plot
# ---------------------------------------------------------------------------


_DEFAULT_FOREST_METRICS: list[str] = [
    "max_score",
    "top5_mean_score",
    "p95_score",
    "lifetime_weighted_mean_score",
    "score_per_track",
]


def plot_bootstrap_ci_paired_differences(
    stats_summary: dict,
    out_path: str | Path,
    *,
    metrics: list[str] | None = None,
) -> None:
    """Forest plot of bootstrap CIs for paired-difference means."""
    out_path = _ensure_parent(out_path)
    if metrics is None:
        metrics = list(_DEFAULT_FOREST_METRICS)

    comparisons = (stats_summary or {}).get("comparisons", {})
    if not comparisons:
        _empty_figure("no stats summary available", out_path)
        return

    # Build (label, mean, ci_low, ci_high) list, in stable order:
    # outer loop = comparison, inner = metric.
    rows: list[tuple[str, float, float, float]] = []
    for comp_key, per_metric in comparisons.items():
        for metric in metrics:
            d = per_metric.get(metric)
            if d is None:
                continue
            rows.append((
                f"{comp_key} :: {metric}",
                float(d.get("mean_difference", 0.0)),
                float(d.get("bootstrap_ci_low", 0.0)),
                float(d.get("bootstrap_ci_high", 0.0)),
            ))

    if not rows:
        _empty_figure("no comparisons available", out_path)
        return

    n = len(rows)
    fig_h = max(4.0, 0.32 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_positions = np.arange(n)
    for i, (label, mean, lo, hi) in enumerate(rows):
        excludes_zero = (lo > 0.0) or (hi < 0.0)
        color = "#d62728" if excludes_zero else "#888888"
        # Horizontal error bar with mean at center.
        err_low = mean - lo
        err_high = hi - mean
        ax.errorbar(
            mean, i,
            xerr=[[err_low], [err_high]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean(a - b) with 95% bootstrap CI")
    ax.set_title("Paired-difference bootstrap CIs (red: CI excludes 0)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Activity traces (currently 3-panel bar diagnostic)
# ---------------------------------------------------------------------------


def plot_activity_traces(
    records: list[PairedRecord],
    out_path: str | Path,
    *,
    n_examples: int = 3,
) -> None:
    """Bar chart of (mean_active, late_active, activity_variance) for the
    first ``n_examples`` paired records.

    The full per-step active-fraction trace isn't stored on
    :class:`ConditionResult`; the title makes that explicit.
    """
    out_path = _ensure_parent(out_path)
    if not records:
        _empty_figure("no data", out_path)
        return

    take = records[: max(int(n_examples), 0)]
    if not take:
        _empty_figure("no data", out_path)
        return

    fields = ("mean_active", "late_active", "activity_variance")
    n_rows = len(take)
    fig, axes = plt.subplots(
        n_rows, len(fields),
        figsize=(4.0 * len(fields), 2.8 * n_rows),
        squeeze=False,
    )

    for ri, rec in enumerate(take):
        for fi, field in enumerate(fields):
            ax = axes[ri][fi]
            heights = [float(getattr(getattr(rec, cond), field)) for cond in CONDITION_NAMES]
            colors = [CONDITION_COLORS[cond] for cond in CONDITION_NAMES]
            ax.bar(
                list(CONDITION_NAMES), heights,
                color=colors, edgecolor="black", linewidth=0.5,
            )
            ax.set_title(f"rule {rec.rule_idx} seed {rec.seed} -- {field}", fontsize=9)
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Per-pair activity diagnostics "
        "(full traces require re-running the sweep with --keep-traces)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Top-candidate videos
# ---------------------------------------------------------------------------


def write_top_candidate_videos(
    records: list[PairedRecord],
    out_dir: str | Path,
    *,
    top_per_condition: int = 3,
    fps: int = 10,
) -> None:
    """Emit a GIF for the top-`top_per_condition` records per condition.

    Records without ``frames_for_video`` are skipped silently.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    for cond in CONDITION_NAMES:
        cond_dir = out_dir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        ranked = sorted(
            records,
            key=lambda rec, c=cond: float(getattr(getattr(rec, c), "max_score")),
            reverse=True,
        )
        n_emitted = 0
        for rec in ranked:
            if n_emitted >= top_per_condition:
                break
            cr = getattr(rec, cond)
            frames = cr.frames_for_video
            if frames is None:
                continue
            try:
                fr = np.asarray(frames)
            except Exception:
                continue
            if fr.size == 0 or fr.ndim != 3:
                continue
            out_path = cond_dir / f"rule{rec.rule_idx}_seed{rec.seed}.gif"
            try:
                write_projected_gif(fr, tracks=None, out_path=out_path, fps=fps)
            except Exception:
                continue
            n_emitted += 1


# ---------------------------------------------------------------------------
# Convenience: all plots in one call
# ---------------------------------------------------------------------------


def write_all_m4b_plots(
    records: list[PairedRecord],
    stats_summary: dict | None,
    out_dir: str | Path,
) -> None:
    """Run every M4B plot writer plus the top-candidate video extractor.

    File names match the M4B spec exactly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_condition_boxplot(
        records, "max_score",
        out_dir / "condition_boxplot_max_score.png",
        title="max_score by condition",
    )
    plot_condition_boxplot(
        records, "top5_mean_score",
        out_dir / "condition_boxplot_top5_score.png",
        title="top5_mean_score by condition",
    )
    plot_condition_violin(
        records, "p95_score",
        out_dir / "condition_violin_95th_percentile.png",
        title="p95_score by condition",
    )
    plot_paired_lines(
        records, "max_score",
        condition_a="coherent_4d", condition_b="shuffled_4d",
        out_path=out_dir / "paired_lines_coherent_vs_shuffled.png",
        title="max_score: coherent_4d vs shuffled_4d (paired)",
    )
    plot_paired_lines(
        records, "max_score",
        condition_a="coherent_4d", condition_b="matched_2d",
        out_path=out_dir / "paired_lines_coherent_vs_2d.png",
        title="max_score: coherent_4d vs matched_2d (paired)",
    )
    plot_scatter_score_vs_tracks(
        records, "max_score",
        out_dir / "observer_score_vs_num_tracks.png",
        title="max_score vs n_tracks",
    )
    plot_track_count_by_condition(
        records,
        out_dir / "track_count_by_condition.png",
    )
    plot_metric_breakdown_best_candidates(
        records,
        out_dir / "metric_breakdown_best_candidates.png",
    )
    if stats_summary is not None:
        plot_bootstrap_ci_paired_differences(
            stats_summary,
            out_dir / "bootstrap_ci_paired_differences.png",
        )
    plot_activity_traces(
        records,
        out_dir / "activity_traces.png",
    )
    write_top_candidate_videos(
        records,
        out_dir / "videos" / "top_candidates",
    )
