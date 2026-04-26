"""Cross-run analysis tool for the M3 observer-worlds experiment suite.

Loads a collection of run directories produced by the experiment scripts
(4D, 2D baseline, shuffled-4D), groups them by ``world_kind`` (extracted
from the per-run ``summary.md``), and produces comparison plots that
test the central M3 hypothesis: do 4D-projected structures score higher
on observer-likeness than the 2D baseline and shuffled-4D baseline?

Public API
----------
- :func:`load_run`, :func:`load_runs` — load run snapshots
- :func:`plot_observer_score_histogram`
- :func:`plot_score_vs_age`
- :func:`plot_baseline_comparison`
- :func:`summary_table`
- :func:`write_summary_md`
- :func:`main` — CLI entry point
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stable display order for world kinds.
_KIND_ORDER: tuple[str, ...] = ("4d", "shuffled_4d", "2d", "unknown")

# Fixed color mapping per world_kind for consistent visual identity across
# all the cross-run plots.
_KIND_COLORS: dict[str, str] = {
    "4d": "#1f77b4",
    "shuffled_4d": "#ff7f0e",
    "2d": "#2ca02c",
    "unknown": "#7f7f7f",
}

_WORLD_KIND_REGEX = re.compile(r"World kind:\s*\*\*(\w+)\*\*")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Loaded snapshot of a single run."""

    run_dir: Path
    label: str
    world_kind: str  # "4d" | "2d" | "shuffled_4d" | "unknown"
    seed: int
    timesteps: int
    grid_shape: tuple  # 2D for "2d" runs, 4D otherwise
    n_tracks: int
    n_candidates: int
    observer_scores: list[dict] = field(default_factory=list)
    candidate_ages: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _parse_world_kind(summary_text: str) -> str:
    m = _WORLD_KIND_REGEX.search(summary_text)
    if m is None:
        return "unknown"
    return m.group(1)


def _maybe_float(value: str) -> float | None:
    """Parse a CSV cell as float, returning None for empty strings."""
    if value is None:
        return None
    s = value.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_observer_scores(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return rows
        for raw_row in reader:
            # Pad / truncate so we always have one cell per header column.
            cells = list(raw_row) + [""] * max(0, len(header) - len(raw_row))
            cells = cells[: len(header)]
            row: dict = {}
            for col, val in zip(header, cells):
                row[col] = _maybe_float(val)
            rows.append(row)
    return rows


def _read_candidates(path: Path) -> tuple[int, list[int]]:
    """Return (n_candidates, candidate_ages).

    n_candidates counts rows with is_candidate == "True".
    candidate_ages contains the integer age of those rows.
    """
    n_cand = 0
    ages: list[int] = []
    if not path.exists():
        return n_cand, ages
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_cand = (row.get("is_candidate") or "").strip()
            if is_cand == "True":
                n_cand += 1
                age_str = (row.get("age") or "").strip()
                try:
                    ages.append(int(float(age_str)))
                except (TypeError, ValueError):
                    continue
    return n_cand, ages


def _grid_shape_from_config(cfg: dict, world_kind: str) -> tuple:
    world = cfg.get("world", {}) or {}
    nx = int(world.get("nx", 0))
    ny = int(world.get("ny", 0))
    nz = int(world.get("nz", 0))
    nw = int(world.get("nw", 0))
    if world_kind == "2d":
        return (nx, ny)
    return (nx, ny, nz, nw)


def _count_tracks(tracks_csv: Path) -> int:
    """Number of distinct track_ids in tracks.csv."""
    if not tracks_csv.exists():
        return 0
    seen: set[int] = set()
    with tracks_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid_str = (row.get("track_id") or "").strip()
            if tid_str == "":
                continue
            try:
                seen.add(int(tid_str))
            except ValueError:
                continue
    return len(seen)


def load_run(run_dir: str | Path) -> RunSummary:
    """Load summary.md + config.json + observer_scores.csv + candidates.csv.

    Determine ``world_kind`` from summary.md by regex-matching
    ``World kind: **<kind>**``. Falls back to ``'unknown'`` if not found.

    All numeric fields in observer_scores.csv that are non-empty are
    converted to float. Empty cells become ``None``.
    """
    run_path = Path(run_dir)
    summary_path = run_path / "summary.md"
    config_path = run_path / "config.json"
    obs_path = run_path / "data" / "observer_scores.csv"
    cand_path = run_path / "data" / "candidates.csv"
    tracks_path = run_path / "data" / "tracks.csv"

    summary_text = summary_path.read_text() if summary_path.exists() else ""
    world_kind = _parse_world_kind(summary_text)

    if config_path.exists():
        cfg = json.loads(config_path.read_text())
    else:
        cfg = {}

    label = str(cfg.get("label", run_path.name))
    seed = int(cfg.get("seed", 0))
    timesteps = int((cfg.get("world") or {}).get("timesteps", 0))
    grid_shape = _grid_shape_from_config(cfg, world_kind)

    observer_scores = _read_observer_scores(obs_path)
    n_candidates, candidate_ages = _read_candidates(cand_path)
    n_tracks = _count_tracks(tracks_path)

    return RunSummary(
        run_dir=run_path,
        label=label,
        world_kind=world_kind,
        seed=seed,
        timesteps=timesteps,
        grid_shape=tuple(grid_shape),
        n_tracks=n_tracks,
        n_candidates=n_candidates,
        observer_scores=observer_scores,
        candidate_ages=candidate_ages,
    )


def load_runs(
    output_root: str | Path = "outputs",
    *,
    glob_pattern: str = "*",
) -> list[RunSummary]:
    """Load every run directory under ``output_root`` matching ``glob_pattern``.

    Skips directories that don't contain a ``config.json`` or
    ``data/observer_scores.csv``. Sorted by directory name.
    """
    root = Path(output_root)
    if not root.exists():
        return []

    candidates = sorted(p for p in root.glob(glob_pattern) if p.is_dir())
    runs: list[RunSummary] = []
    for run_dir in candidates:
        if not (run_dir / "config.json").exists():
            continue
        if not (run_dir / "data" / "observer_scores.csv").exists():
            continue
        try:
            runs.append(load_run(run_dir))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] failed to load {run_dir}: {exc}", file=sys.stderr)
            continue
    return runs


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _group_by_kind(runs: list[RunSummary]) -> dict[str, list[RunSummary]]:
    grouped: dict[str, list[RunSummary]] = {}
    for run in runs:
        grouped.setdefault(run.world_kind, []).append(run)
    return grouped


def _ordered_kinds(grouped: dict[str, list[RunSummary]]) -> list[str]:
    """Return present kinds in stable display order."""
    out = [k for k in _KIND_ORDER if k in grouped]
    # Append any unexpected kind names at the end (alphabetical).
    extras = sorted(k for k in grouped if k not in _KIND_ORDER)
    return out + extras


def _combined_scores_for_runs(runs: list[RunSummary]) -> list[float]:
    out: list[float] = []
    for run in runs:
        for row in run.observer_scores:
            v = row.get("combined")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                out.append(fv)
    return out


def _score_age_pairs(runs: list[RunSummary]) -> list[tuple[float, int]]:
    """Pair (combined score, candidate age) per scored candidate.

    Pairing is by index: the i-th observer_scores row is matched with the
    i-th candidate-only age. If counts differ, pair up to the shorter list.
    """
    pairs: list[tuple[float, int]] = []
    for run in runs:
        scores = [row.get("combined") for row in run.observer_scores]
        ages = list(run.candidate_ages)
        n = min(len(scores), len(ages))
        for s, a in zip(scores[:n], ages[:n]):
            if s is None:
                continue
            try:
                fs = float(s)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fs):
                continue
            pairs.append((fs, int(a)))
    return pairs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _placeholder_panel(ax: plt.Axes, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _empty_figure(message: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.set_axis_off()
    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_observer_score_histogram(
    runs: list[RunSummary],
    out_path: str | Path,
    *,
    bins: int = 25,
) -> None:
    """Histogram of ``combined`` observer scores, one panel per world_kind.

    Empty panels render a "no candidates" placeholder rather than crashing.
    """
    if not runs:
        _empty_figure("no runs loaded", out_path)
        return

    grouped = _group_by_kind(runs)
    kinds = _ordered_kinds(grouped)
    n = len(kinds)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    # Compute a shared x-range across kinds so panels are visually comparable.
    all_scores: list[float] = []
    for k in kinds:
        all_scores.extend(_combined_scores_for_runs(grouped[k]))
    if all_scores:
        lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        bin_edges = np.linspace(lo, hi, bins + 1)
    else:
        bin_edges = bins  # type: ignore[assignment]

    for ax, kind in zip(axes, kinds):
        scores = _combined_scores_for_runs(grouped[kind])
        color = _KIND_COLORS.get(kind, _KIND_COLORS["unknown"])
        if not scores:
            _placeholder_panel(ax, f"{kind}\nno candidates")
            ax.set_title(kind)
            continue
        ax.hist(
            scores,
            bins=bin_edges,
            color=color,
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title(f"{kind}  (n={len(scores)})")
        ax.set_xlabel("combined observer_score")
        ax.set_ylabel("count")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Observer-score histograms by world kind")
    _save_fig(fig, out_path)


def plot_score_vs_age(
    runs: list[RunSummary],
    out_path: str | Path,
) -> None:
    """Scatter of combined observer_score vs candidate age, colored by world_kind."""
    if not runs:
        _empty_figure("no runs loaded", out_path)
        return

    grouped = _group_by_kind(runs)
    kinds = _ordered_kinds(grouped)

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted_any = False
    for kind in kinds:
        pairs = _score_age_pairs(grouped[kind])
        if not pairs:
            continue
        scores = [p[0] for p in pairs]
        ages = [p[1] for p in pairs]
        color = _KIND_COLORS.get(kind, _KIND_COLORS["unknown"])
        ax.scatter(
            ages,
            scores,
            color=color,
            label=f"{kind} (n={len(pairs)})",
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
            s=40,
        )
        plotted_any = True

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "no scored candidates",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_axis_off()
    else:
        ax.set_xlabel("candidate age (frames)")
        ax.set_ylabel("combined observer_score")
        ax.set_title("Observer-score vs candidate age, by world kind")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=9)

    _save_fig(fig, out_path)


def plot_baseline_comparison(
    runs: list[RunSummary],
    out_path: str | Path,
) -> None:
    """Box-and-whiskers of combined observer_score grouped by world_kind."""
    if not runs:
        _empty_figure("no runs loaded", out_path)
        return

    grouped = _group_by_kind(runs)
    kinds = _ordered_kinds(grouped)

    # Restrict to kinds that actually have scored candidates.
    nonempty: list[tuple[str, list[float]]] = []
    for kind in kinds:
        scores = _combined_scores_for_runs(grouped[kind])
        if scores:
            nonempty.append((kind, scores))

    fig, ax = plt.subplots(figsize=(8, 5))

    if not nonempty:
        ax.text(
            0.5,
            0.5,
            "no scored candidates across any world kind",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        _save_fig(fig, out_path)
        return

    labels = [k for k, _ in nonempty]
    data = [s for _, s in nonempty]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, kind in zip(bp["boxes"], labels):
        color = _KIND_COLORS.get(kind, _KIND_COLORS["unknown"])
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Annotate counts under each label.
    tick_labels = [f"{k}\n(n={len(s)})" for k, s in nonempty]
    ax.set_xticklabels(tick_labels)

    if len(nonempty) == 1:
        ax.set_title(
            f"Observer-score distribution — only world_kind '{labels[0]}' has data"
        )
    else:
        ax.set_title("Observer-score distribution by world kind")

    ax.set_ylabel("combined observer_score")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Tabular summary + markdown
# ---------------------------------------------------------------------------


def summary_table(runs: list[RunSummary]) -> dict[str, dict[str, float | int]]:
    """Per-world-kind summary stats.

    Keys:
        n_runs, n_candidates, mean_score, median_score, max_score, std_score,
        mean_age, n_tracks_total

    ``mean_score``, ``median_score``, ``max_score``, ``std_score`` only consider
    candidates with a finite ``combined`` score; if none exist for a kind, those
    fields are NaN.
    """
    grouped = _group_by_kind(runs)
    table: dict[str, dict[str, float | int]] = {}

    for kind in _ordered_kinds(grouped):
        kind_runs = grouped[kind]
        scores = _combined_scores_for_runs(kind_runs)
        ages: list[int] = []
        for r in kind_runs:
            ages.extend(int(a) for a in r.candidate_ages)
        n_candidates = sum(int(r.n_candidates) for r in kind_runs)
        n_tracks_total = sum(int(r.n_tracks) for r in kind_runs)

        if scores:
            mean_score = float(np.mean(scores))
            median_score = float(np.median(scores))
            max_score = float(np.max(scores))
            std_score = float(np.std(scores, ddof=0))
        else:
            mean_score = float("nan")
            median_score = float("nan")
            max_score = float("nan")
            std_score = float("nan")

        mean_age = float(np.mean(ages)) if ages else float("nan")

        table[kind] = {
            "n_runs": int(len(kind_runs)),
            "n_candidates": int(n_candidates),
            "mean_score": mean_score,
            "median_score": median_score,
            "max_score": max_score,
            "std_score": std_score,
            "mean_age": mean_age,
            "n_tracks_total": int(n_tracks_total),
        }

    return table


def _fmt_float(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.3f}" if x < 0 else f"{x:.3f}"


def write_summary_md(
    runs: list[RunSummary],
    out_path: str | Path,
    *,
    plots_dir: str | Path | None = None,
) -> None:
    """Write a cross-run markdown summary listing all runs and the table."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = summary_table(runs)
    grouped = _group_by_kind(runs)

    lines: list[str] = []
    lines.append("# Cross-run summary")
    lines.append("")
    lines.append(f"- Total runs: {len(runs)}")
    lines.append(f"- World kinds present: {', '.join(_ordered_kinds(grouped)) or 'none'}")
    lines.append("")

    lines.append("## Per-kind summary")
    lines.append("")
    lines.append(
        "| world_kind | n_runs | n_candidates | n_tracks_total | "
        "mean_score | median_score | max_score | std_score | mean_age |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for kind, stats in table.items():
        lines.append(
            f"| {kind} | {stats['n_runs']} | {stats['n_candidates']} | "
            f"{stats['n_tracks_total']} | {_fmt_float(stats['mean_score'])} | "
            f"{_fmt_float(stats['median_score'])} | {_fmt_float(stats['max_score'])} | "
            f"{_fmt_float(stats['std_score'])} | {_fmt_float(stats['mean_age'])} |"
        )
    lines.append("")

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        try:
            rel = plots_dir.relative_to(out_path.parent)
            prefix = rel.as_posix()
        except ValueError:
            prefix = plots_dir.as_posix()
        lines.append("## Plots")
        lines.append("")
        for fname in (
            "observer_score_histogram.png",
            "score_vs_age.png",
            "baseline_comparison.png",
        ):
            lines.append(f"- `{prefix}/{fname}`")
        lines.append("")

    lines.append("## Runs")
    lines.append("")
    lines.append("| run_dir | label | world_kind | seed | timesteps | n_tracks | n_candidates |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in sorted(runs, key=lambda r: (r.world_kind, r.run_dir.name)):
        lines.append(
            f"| `{run.run_dir.name}` | {run.label} | {run.world_kind} | "
            f"{run.seed} | {run.timesteps} | {run.n_tracks} | {run.n_candidates} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_table_for_stdout(table: dict[str, dict[str, float | int]]) -> str:
    if not table:
        return "(no runs)"
    cols = [
        "world_kind",
        "n_runs",
        "n_candidates",
        "n_tracks_total",
        "mean_score",
        "median_score",
        "max_score",
        "std_score",
        "mean_age",
    ]
    rows: list[list[str]] = [cols]
    for kind, stats in table.items():
        rows.append(
            [
                kind,
                str(stats["n_runs"]),
                str(stats["n_candidates"]),
                str(stats["n_tracks_total"]),
                _fmt_float(stats["mean_score"]),
                _fmt_float(stats["median_score"]),
                _fmt_float(stats["max_score"]),
                _fmt_float(stats["std_score"]),
                _fmt_float(stats["mean_age"]),
            ]
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    out_lines: list[str] = []
    for i, row in enumerate(rows):
        out_lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            out_lines.append("  ".join("-" * widths[j] for j in range(len(cols))))
    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="observer_worlds.analysis.summarize_results",
        description="Cross-run analysis for observer_worlds runs.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Directory containing run subdirectories (default: outputs).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/_summary",
        help="Directory where plots and summary.md are written.",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern matched against directory names under --output-root.",
    )
    args = parser.parse_args(argv)

    runs = load_runs(args.output_root, glob_pattern=args.glob)
    if not runs:
        print(f"No runs found under {args.output_root!r} matching {args.glob!r}.")
    else:
        print(f"Loaded {len(runs)} run(s) from {args.output_root}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_path = out_dir / "observer_score_histogram.png"
    scatter_path = out_dir / "score_vs_age.png"
    box_path = out_dir / "baseline_comparison.png"

    plot_observer_score_histogram(runs, hist_path)
    plot_score_vs_age(runs, scatter_path)
    plot_baseline_comparison(runs, box_path)

    summary_md_path = out_dir / "summary.md"
    write_summary_md(runs, summary_md_path, plots_dir=out_dir)

    table = summary_table(runs)
    print()
    print("Per-world-kind summary:")
    print(_format_table_for_stdout(table))
    print()
    print(f"Plots written to: {out_dir}")
    print(f"Markdown summary: {summary_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
