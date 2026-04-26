"""M5 — per-candidate intervention plots.

Consumes :class:`CandidateInterventionReport` instances produced by
``experiments._m5_interventions.run_candidate_interventions``.

Plot files:
  * ``per_candidate/track_<id>_divergence.png`` — 4 lines per candidate,
    one per intervention type, showing per-step divergence.
  * ``per_candidate/track_<id>_resilience.png`` — 2x2 panels, candidate
    active-cell counts in unperturbed vs intervened rollouts.
  * ``aggregate_divergence_<metric>.png`` — mean ± stdev across candidates.
  * ``intervention_heatmap_<metric>.png`` — candidates x intervention types.
  * ``intervention_summary_bars.png`` — grouped bars of aggregate metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.experiments._m5_interventions import (
    INTERVENTION_TYPES,
    CandidateInterventionReport,
    aggregate_intervention_summaries,
)


INTERVENTION_COLORS: dict[str, str] = {
    "internal_flip":    "#1f77b4",
    "boundary_flip":    "#ff7f0e",
    "environment_flip": "#2ca02c",
    "hidden_shuffle":   "#d62728",
}


def _empty_figure(msg: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save(fig, out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-candidate plots
# ---------------------------------------------------------------------------


def plot_per_candidate_divergence(
    report: CandidateInterventionReport,
    out_path: str | Path,
    *,
    metric: str = "full_grid_l1",
) -> None:
    """4 lines (one per intervention type) showing divergence over time."""
    out_path = Path(out_path)
    if not report.trajectories:
        _empty_figure(f"track {report.track_id}: no trajectories", out_path)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, traj in report.trajectories.items():
        y = traj.full_grid_l1 if metric == "full_grid_l1" else traj.candidate_footprint_l1
        ax.plot(range(1, len(y) + 1), y,
                color=INTERVENTION_COLORS.get(kind, "gray"),
                label=kind, linewidth=1.5)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Track {report.track_id}  (snap_t={report.snapshot_t}, "
        f"age={report.track_age}, obs_score="
        f"{report.observer_score:+.2f})" if report.observer_score is not None
        else f"Track {report.track_id}  (snap_t={report.snapshot_t})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_path)


def plot_resilience_curves(
    report: CandidateInterventionReport,
    out_path: str | Path,
) -> None:
    """2x2 panel: per intervention, dashed=orig active, solid=intervened."""
    out_path = Path(out_path)
    if not report.trajectories:
        _empty_figure(f"track {report.track_id}: no trajectories", out_path)
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, INTERVENTION_TYPES):
        traj = report.trajectories.get(kind)
        if traj is None:
            ax.text(0.5, 0.5, f"{kind}: no data", ha="center", va="center")
            ax.axis("off")
            continue
        steps = range(1, traj.n_steps + 1)
        c = INTERVENTION_COLORS.get(kind, "gray")
        ax.plot(steps, traj.candidate_active_orig, "--", color=c, alpha=0.6,
                label="orig")
        ax.plot(steps, traj.candidate_active_intervened, "-", color=c,
                label="intervened")
        ax.set_title(kind, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Resilience curves — track {report.track_id} "
        f"(snap_t={report.snapshot_t}, age={report.track_age})",
        fontsize=12,
    )
    fig.supxlabel("rollout step")
    fig.supylabel("active cells inside candidate footprint")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------


def plot_aggregate_divergence(
    reports: list[CandidateInterventionReport],
    out_path: str | Path,
    *,
    metric: str = "full_grid_l1",
) -> None:
    """For each intervention type, mean ± stdev across candidates over time."""
    out_path = Path(out_path)
    valid_reports = [r for r in reports if r.trajectories]
    if not valid_reports:
        _empty_figure("no reports with trajectories", out_path)
        return
    n_steps = next(iter(valid_reports[0].trajectories.values())).n_steps
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = np.arange(1, n_steps + 1)
    for kind in INTERVENTION_TYPES:
        traj_list = [r.trajectories.get(kind) for r in valid_reports]
        traj_list = [t for t in traj_list if t is not None]
        if not traj_list:
            continue
        # Collect per-step values, padded if lengths differ.
        arr = []
        for t in traj_list:
            y = t.full_grid_l1 if metric == "full_grid_l1" else t.candidate_footprint_l1
            if len(y) >= n_steps:
                arr.append(y[:n_steps])
        if not arr:
            continue
        a = np.asarray(arr)
        mean = a.mean(axis=0)
        std = a.std(axis=0)
        c = INTERVENTION_COLORS.get(kind, "gray")
        ax.plot(steps, mean, color=c, linewidth=1.8, label=kind)
        ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(f"mean ± stdev {metric}")
    ax.set_title(f"Aggregate divergence over {len(valid_reports)} candidates")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_path)


def plot_intervention_heatmap(
    reports: list[CandidateInterventionReport],
    out_path: str | Path,
    *,
    metric: str = "mean_full_grid_l1",
) -> None:
    """Heatmap: rows=track_id, cols=intervention_type; cell = metric."""
    out_path = Path(out_path)
    valid = [r for r in reports if r.intervention_summary]
    if not valid:
        _empty_figure("no reports with intervention summaries", out_path)
        return
    rows = [str(r.track_id) for r in valid]
    cols = list(INTERVENTION_TYPES)
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(valid):
        for j, kind in enumerate(cols):
            s = r.intervention_summary.get(kind)
            if s is not None and metric in s:
                M[i, j] = s[metric]
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2, 0.4 * len(rows) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("intervention type")
    ax.set_ylabel("track_id")
    ax.set_title(f"Intervention heatmap: {metric}")
    fig.colorbar(im, ax=ax, fraction=0.04)
    # Annotate cells.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < np.nanmedian(M) else "black",
                        fontsize=8)
    _save(fig, out_path)


def plot_intervention_summary_bars(
    reports: list[CandidateInterventionReport],
    out_path: str | Path,
) -> None:
    """Grouped bars of aggregate metrics per intervention type."""
    out_path = Path(out_path)
    agg = aggregate_intervention_summaries(reports)
    if not agg:
        _empty_figure("no intervention reports", out_path)
        return
    metrics = ["mean_full_grid_l1", "mean_candidate_footprint_l1",
               "final_area_ratio", "final_survival"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.20
    for i, kind in enumerate(INTERVENTION_TYPES):
        if kind not in agg:
            continue
        vals = [agg[kind].get(m, 0.0) for m in metrics]
        ax.bar(x + (i - 1.5) * width, vals, width=width,
               color=INTERVENTION_COLORS.get(kind, "gray"), label=kind)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("value (mean across candidates)")
    ax.set_title("Aggregate intervention summary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def write_all_m5_plots(
    reports: list[CandidateInterventionReport],
    out_dir: str | Path,
    *,
    per_candidate_max: int = 5,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_aggregate_divergence(reports, out_dir / "aggregate_divergence_full_grid.png",
                              metric="full_grid_l1")
    plot_aggregate_divergence(reports, out_dir / "aggregate_divergence_candidate_footprint.png",
                              metric="candidate_footprint_l1")
    plot_intervention_heatmap(reports, out_dir / "intervention_heatmap_full_grid.png",
                              metric="mean_full_grid_l1")
    plot_intervention_heatmap(reports, out_dir / "intervention_heatmap_candidate_footprint.png",
                              metric="mean_candidate_footprint_l1")
    plot_intervention_summary_bars(reports, out_dir / "intervention_summary_bars.png")

    per_dir = out_dir / "per_candidate"
    per_dir.mkdir(parents=True, exist_ok=True)
    # Pick top-K by observer_score (treating None as -inf).
    sorted_reports = sorted(
        reports,
        key=lambda r: (r.observer_score if r.observer_score is not None else -np.inf),
        reverse=True,
    )
    for r in sorted_reports[:per_candidate_max]:
        plot_per_candidate_divergence(
            r, per_dir / f"track_{r.track_id:04d}_divergence.png",
        )
        plot_resilience_curves(
            r, per_dir / f"track_{r.track_id:04d}_resilience.png",
        )
