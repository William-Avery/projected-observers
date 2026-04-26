"""Leaderboard artifacts for the M4A viability search.

Given a list of :class:`ViabilityReport` objects (one per scouted rule),
this module writes:

- ``leaderboard.csv``: one row per rule, sorted by descending viability,
  with a flat column set suitable for spreadsheet inspection.
- ``leaderboard.json``: richer dump that retains per-seed activity and
  component-count traces for downstream plotting.
- ``top_k/rule_NNN/`` per-rule artifact directories with a runnable
  ``config.json``, the rule, the single-rule report, plots, and a GIF
  of one scout seed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from observer_worlds.search.fitness import simulate_4d_in_memory
from observer_worlds.search.rules import FractionalRule
from observer_worlds.search.viability import ViabilityReport


LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "rank", "viability_score", "n_seeds",
    "birth_min", "birth_max", "survive_min", "survive_max", "initial_density",
    "persistent_component_score", "target_activity_score", "temporal_change_score",
    "boundedness_score", "diversity_score",
    "extinction_penalty", "saturation_penalty", "frozen_world_penalty",
    "final_active_fraction", "mean_late_active_fraction",
    "n_components_over_time_mean", "max_component_lifetime",
    "mean_component_lifetime", "n_persistent_components",
    "per_seed_scores", "per_seed_aborted",
    "sim_time_seconds",
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serializable: {type(obj)}")


def _report_to_json_entry(rank: int, report: ViabilityReport) -> dict:
    """Convert a :class:`ViabilityReport` to a JSON-serializable dict."""
    return {
        "rank": int(rank),
        "viability_score": float(report.viability_score),
        "rule": report.rule.to_dict(),
        "n_seeds": int(report.n_seeds),
        "persistent_component_score": float(report.persistent_component_score),
        "target_activity_score": float(report.target_activity_score),
        "temporal_change_score": float(report.temporal_change_score),
        "boundedness_score": float(report.boundedness_score),
        "diversity_score": float(report.diversity_score),
        "extinction_penalty": float(report.extinction_penalty),
        "saturation_penalty": float(report.saturation_penalty),
        "frozen_world_penalty": float(report.frozen_world_penalty),
        "final_active_fraction": float(report.final_active_fraction),
        "mean_late_active_fraction": float(report.mean_late_active_fraction),
        "n_components_over_time_mean": float(report.n_components_over_time_mean),
        "max_component_lifetime": int(report.max_component_lifetime),
        "mean_component_lifetime": float(report.mean_component_lifetime),
        "n_persistent_components": float(report.n_persistent_components),
        "per_seed_scores": [float(s) for s in report.per_seed_scores],
        "per_seed_aborted": [bool(a) for a in report.per_seed_aborted],
        "activity_traces": [[float(x) for x in trace] for trace in report.activity_traces],
        "component_count_traces": [
            [int(x) for x in trace] for trace in report.component_count_traces
        ],
        "sim_time_seconds": float(report.sim_time_seconds),
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def write_leaderboard_csv(reports: list[ViabilityReport], out_path: str | Path) -> None:
    """Write a sorted (by descending ``viability_score``) leaderboard CSV.

    ``rank`` starts at 1.  ``per_seed_scores`` and ``per_seed_aborted``
    are joined with ``;`` so each row stays a single CSV cell per column.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_reports = sorted(reports, key=lambda r: -r.viability_score)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEADERBOARD_COLUMNS))
        writer.writeheader()
        for i, r in enumerate(sorted_reports):
            row = {
                "rank": i + 1,
                "viability_score": f"{r.viability_score:.6f}",
                "n_seeds": r.n_seeds,
                "birth_min": f"{r.rule.birth_min:.6f}",
                "birth_max": f"{r.rule.birth_max:.6f}",
                "survive_min": f"{r.rule.survive_min:.6f}",
                "survive_max": f"{r.rule.survive_max:.6f}",
                "initial_density": f"{r.rule.initial_density:.6f}",
                "persistent_component_score": f"{r.persistent_component_score:.6f}",
                "target_activity_score": f"{r.target_activity_score:.6f}",
                "temporal_change_score": f"{r.temporal_change_score:.6f}",
                "boundedness_score": f"{r.boundedness_score:.6f}",
                "diversity_score": f"{r.diversity_score:.6f}",
                "extinction_penalty": f"{r.extinction_penalty:.6f}",
                "saturation_penalty": f"{r.saturation_penalty:.6f}",
                "frozen_world_penalty": f"{r.frozen_world_penalty:.6f}",
                "final_active_fraction": f"{r.final_active_fraction:.6f}",
                "mean_late_active_fraction": f"{r.mean_late_active_fraction:.6f}",
                "n_components_over_time_mean": f"{r.n_components_over_time_mean:.6f}",
                "max_component_lifetime": int(r.max_component_lifetime),
                "mean_component_lifetime": f"{r.mean_component_lifetime:.6f}",
                "n_persistent_components": f"{r.n_persistent_components:.6f}",
                "per_seed_scores": ";".join(f"{float(s):.6f}" for s in r.per_seed_scores),
                "per_seed_aborted": ";".join(str(bool(a)) for a in r.per_seed_aborted),
                "sim_time_seconds": f"{r.sim_time_seconds:.6f}",
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def write_leaderboard_json(reports: list[ViabilityReport], out_path: str | Path) -> None:
    """Write a richer JSON dump including activity_traces and component_count_traces."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_reports = sorted(reports, key=lambda r: -r.viability_score)
    entries = [_report_to_json_entry(i + 1, r) for i, r in enumerate(sorted_reports)]
    out_path.write_text(json.dumps(entries, indent=2, default=_json_default))


# ---------------------------------------------------------------------------
# Top-K artifacts
# ---------------------------------------------------------------------------


def _plot_per_seed_lines(
    traces: list[list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Write a line plot with one line per seed.  Headless matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if traces:
        for i, trace in enumerate(traces):
            if len(trace) == 0:
                continue
            ax.plot(range(len(trace)), trace, linewidth=1.0, label=f"seed {i}")
        if any(len(t) > 0 for t in traces):
            ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _build_run_config(
    rule: FractionalRule,
    *,
    label: str,
    grid_shape: tuple[int, int, int, int],
    timesteps: int,
    base_seed: int,
    detection_config,
    backend: str,
):
    """Build a RunConfig that reproduces this rule's scout simulation."""
    from observer_worlds.utils.config import (
        DetectionConfig,
        OutputConfig,
        ProjectionConfig,
        RunConfig,
        WorldConfig,
    )

    bsrule = rule.to_bsrule()
    nx, ny, nz, nw = grid_shape
    world = WorldConfig(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nw=int(nw),
        timesteps=int(timesteps),
        initial_density=float(rule.initial_density),
        rule_birth=tuple(int(b) for b in bsrule.birth),
        rule_survival=tuple(int(s) for s in bsrule.survival),
        backend=backend,
    )
    detection = detection_config if detection_config is not None else DetectionConfig()
    cfg = RunConfig(
        world=world,
        projection=ProjectionConfig(),
        detection=detection,
        output=OutputConfig(),
        seed=int(base_seed),
        label=label,
    )
    return cfg


def write_top_k_artifacts(
    reports: list[ViabilityReport],
    out_dir: str | Path,
    *,
    top_k: int = 10,
    grid_shape: tuple[int, int, int, int] = (64, 64, 8, 8),
    timesteps: int = 300,
    base_seed: int = 0,
    detection_config=None,
    backend: str = "numba",
) -> None:
    """For the top-K rules by ``viability_score``, write per-rule artifacts.

    Each rule gets a ``rule_NNN/`` sub-directory with a runnable
    ``config.json``, ``rule.json``, ``viability_report.json``, plots of
    activity / component count, and a projected GIF of one scout seed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_reports = sorted(reports, key=lambda r: -r.viability_score)
    k = min(top_k, len(sorted_reports))

    # Local imports to avoid hard deps at module import time.
    from observer_worlds.analysis.videos import write_projected_gif

    for i in range(k):
        report = sorted_reports[i]
        rule = report.rule
        rank = i + 1
        rule_dir = out_dir / f"rule_{rank:03d}"
        rule_dir.mkdir(parents=True, exist_ok=True)

        # --- rule.json -----------------------------------------------------
        (rule_dir / "rule.json").write_text(
            json.dumps(rule.to_dict(), indent=2, default=_json_default)
        )

        # --- viability_report.json ----------------------------------------
        entry = _report_to_json_entry(rank, report)
        (rule_dir / "viability_report.json").write_text(
            json.dumps(entry, indent=2, default=_json_default)
        )

        # --- config.json (runnable via experiments.run_4d_projection) -----
        label = f"rule_{rank:03d}_{rule.short_repr()}"
        cfg = _build_run_config(
            rule,
            label=label,
            grid_shape=grid_shape,
            timesteps=timesteps,
            base_seed=base_seed,
            detection_config=detection_config,
            backend=backend,
        )
        cfg.save(rule_dir / "config.json")

        # --- activity_trace.png -------------------------------------------
        title_score = f"score={report.viability_score:+.3f}"
        _plot_per_seed_lines(
            report.activity_traces,
            title=f"{rule.short_repr()}  {title_score}",
            ylabel="mean projected activity",
            out_path=rule_dir / "activity_trace.png",
        )

        # --- components_over_time.png -------------------------------------
        # component_count_traces are list[list[int]]; cast to float for plotting.
        comp_traces_float: list[list[float]] = [
            [float(c) for c in trace] for trace in report.component_count_traces
        ]
        _plot_per_seed_lines(
            comp_traces_float,
            title=f"{rule.short_repr()}  {title_score}",
            ylabel="# components per frame",
            out_path=rule_dir / "components_over_time.png",
        )

        # --- video.gif (re-run a single scout seed) ------------------------
        try:
            frames, _, _ = simulate_4d_in_memory(
                rule.to_bsrule(),
                grid_shape=grid_shape,
                timesteps=timesteps,
                initial_density=rule.initial_density,
                seed=base_seed,
                backend=backend,
                early_abort=False,
            )
            write_projected_gif(
                frames,
                tracks=None,
                out_path=rule_dir / "video.gif",
                fps=10,
                max_frames=min(timesteps, 200),
                upsample=4,
            )
        except Exception as exc:  # pragma: no cover - GIF failures shouldn't kill search
            (rule_dir / "video_error.txt").write_text(
                f"Failed to render video: {type(exc).__name__}: {exc}\n"
            )
