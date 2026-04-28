"""Plot generators for Follow-up Topic 3 (Stage 4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


PLOT_FILENAMES: tuple[str, ...] = (
    "task_score_vs_hce.png",
    "task_score_vs_observer_score.png",
    "hce_task_correlation_by_task.png",
    "high_hce_vs_low_hce_task_score.png",
    "hidden_intervention_effect_on_task_score.png",
    "repair_recovery_examples.png",
    "cue_memory_examples.png",
)


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _placeholder(plt, path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=11, color="#666", transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _trials_to_xy(trials, x_attr: str):
    xs, ys, tasks = [], [], []
    for t in trials:
        x = getattr(t, x_attr, None)
        y = t.task_score
        if x is None or y is None:
            continue
        xs.append(float(x)); ys.append(float(y)); tasks.append(t.task_name)
    return xs, ys, tasks


def write_all_plots(summary: dict, trials, plots_dir: Path) -> None:
    try:
        plt = _import_plt()
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    task_colors = {"repair": "#3a7", "foraging": "#357",
                   "memory": "#a37"}

    # 1. task_score vs hce
    try:
        path = plots_dir / "task_score_vs_hce.png"
        xs, ys, tasks = _trials_to_xy(trials, "hce")
        if not xs:
            _placeholder(plt, path, "no trials with HCE recorded")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            for task in sorted(set(tasks)):
                xi = [x for x, tk in zip(xs, tasks) if tk == task]
                yi = [y for y, tk in zip(ys, tasks) if tk == task]
                ax.scatter(xi, yi, alpha=0.55,
                            color=task_colors.get(task, "#666"), label=task)
            ax.set_xlabel("HCE (smoke estimate)")
            ax.set_ylabel("task_score")
            ax.set_title("task_score vs HCE")
            ax.legend()
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] task_score_vs_hce.png: {e!r}")

    # 2. task_score vs observer_score (proxy)
    try:
        path = plots_dir / "task_score_vs_observer_score.png"
        xs, ys, tasks = _trials_to_xy(trials, "observer_score")
        if not xs:
            _placeholder(plt, path, "no trials with observer_score recorded")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            for task in sorted(set(tasks)):
                xi = [x for x, tk in zip(xs, tasks) if tk == task]
                yi = [y for y, tk in zip(ys, tasks) if tk == task]
                ax.scatter(xi, yi, alpha=0.55,
                            color=task_colors.get(task, "#666"), label=task)
            ax.set_xlabel("observer_score (smoke proxy = candidate lifetime)")
            ax.set_ylabel("task_score")
            ax.set_title("task_score vs observer_score (smoke proxy)")
            ax.legend()
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] task_score_vs_observer_score.png: {e!r}")

    # 3. hce_task_correlation_by_task
    try:
        path = plots_dir / "hce_task_correlation_by_task.png"
        cor = summary.get("correlations", {})
        tasks = list(cor)
        rs_hce = [cor[t].get("pearson_HCE_vs_task_score") or 0.0
                  for t in tasks]
        rs_obs = [cor[t].get("pearson_observer_score_vs_task_score") or 0.0
                  for t in tasks]
        if not tasks:
            _placeholder(plt, path, "no per-task correlations available")
        else:
            x = np.arange(len(tasks))
            w = 0.4
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - w / 2, rs_hce, width=w, color="#3a7",
                    label="Pearson(HCE, task)")
            ax.bar(x + w / 2, rs_obs, width=w, color="#357",
                    label="Pearson(observer, task)")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels(tasks)
            ax.set_ylabel("Pearson r")
            ax.set_title("HCE / observer-score correlations with task_score, by task")
            ax.legend()
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hce_task_correlation_by_task.png: {e!r}")

    # 4. high_hce_vs_low_hce_task_score
    try:
        path = plots_dir / "high_hce_vs_low_hce_task_score.png"
        split = summary.get("high_low_hce_split", {})
        usable = [(t, s) for t, s in split.items() if "_status" not in s]
        if not usable:
            _placeholder(plt, path,
                         "insufficient candidates for HCE high/low split")
        else:
            tasks = [t for t, _ in usable]
            lows = [s["mean_task_score_low_hce"] or 0.0 for _, s in usable]
            highs = [s["mean_task_score_high_hce"] or 0.0 for _, s in usable]
            x = np.arange(len(tasks)); w = 0.4
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - w / 2, lows, width=w, color="#bbb", label="low HCE")
            ax.bar(x + w / 2, highs, width=w, color="#3a7", label="high HCE")
            ax.set_xticks(x); ax.set_xticklabels(tasks)
            ax.set_ylabel("mean task_score")
            ax.set_title("Mean task_score: high-HCE vs low-HCE candidates")
            ax.legend()
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] high_hce_vs_low_hce_task_score.png: {e!r}")

    # 5. hidden_intervention_effect_on_task_score — Stage-5+ placeholder.
    try:
        path = plots_dir / "hidden_intervention_effect_on_task_score.png"
        _placeholder(
            plt, path,
            "Stage 5+: hidden_intervention_task_delta plot.\n"
            "Stage 4 smoke does not yet record this delta;\n"
            "see hidden_intervention_task_delta column in task_trials.csv "
            "(currently None for Stage 4).",
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hidden_intervention_effect_on_task_score.png: {e!r}")

    # 6. repair_recovery_examples — Stage-5+ placeholder (per-trial visual).
    try:
        path = plots_dir / "repair_recovery_examples.png"
        _placeholder(
            plt, path,
            "Stage 5+: repair recovery example trajectories.\n"
            "Stage 4 smoke writes per-horizon repair_score to "
            "task_trials.csv;\nthe per-trial trajectory image is a "
            "Stage-5+ deliverable.",
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] repair_recovery_examples.png: {e!r}")

    # 7. cue_memory_examples — Stage-5+ placeholder.
    try:
        path = plots_dir / "cue_memory_examples.png"
        _placeholder(
            plt, path,
            "Stage 5+: cue memory example trajectories.\n"
            "Stage 4 smoke writes per-horizon cue_memory_score to "
            "task_trials.csv.",
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] cue_memory_examples.png: {e!r}")


def write_all(*_, **__) -> None:
    raise NotImplementedError(
        "use write_all_plots(summary, trials, plots_dir) instead."
    )
