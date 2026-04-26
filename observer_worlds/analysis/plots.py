from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from observer_worlds.detection.tracking import Track


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
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_lifetimes(
    tracks: list,
    out_path: str | Path,
    min_age_threshold: int = 0,
) -> None:
    if not tracks:
        _empty_figure("no tracks", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by birth_frame for a tidier plot.
    sorted_tracks = sorted(tracks, key=lambda t: (t.birth_frame, t.track_id))

    y_labels: list[str] = []
    for i, tr in enumerate(sorted_tracks):
        birth = int(tr.birth_frame)
        last = int(tr.last_frame)
        width = max(last - birth + 1, 1)
        color = "tab:green" if int(tr.age) >= int(min_age_threshold) else "tab:gray"
        ax.barh(i, width, left=birth, color=color, edgecolor="none", height=0.7)
        y_labels.append(str(int(tr.track_id)))

    ax.set_yticks(range(len(sorted_tracks)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("frame")
    ax.set_ylabel("track_id")
    ax.set_title(f"Track lifetimes (green: age >= {min_age_threshold})")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.invert_yaxis()  # earliest tracks at the top.

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_area_vs_time(
    tracks: list,
    out_path: str | Path,
    top_n: int = 20,
) -> None:
    if not tracks:
        _empty_figure("no tracks", out_path)
        return

    # Pick the top-N tracks by length (number of observed frames).
    sorted_tracks = sorted(tracks, key=lambda t: len(t.frames), reverse=True)
    selected = sorted_tracks[: max(int(top_n), 0)]

    if not selected:
        _empty_figure("no tracks", out_path)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.get_cmap("tab20")
    for i, tr in enumerate(selected):
        frames = np.asarray(tr.frames, dtype=int)
        areas = np.asarray(tr.area_history, dtype=float)
        if frames.size == 0:
            continue
        ax.plot(
            frames,
            areas,
            label=f"id={int(tr.track_id)}",
            color=cmap(i % 20),
            linewidth=1.2,
        )

    ax.set_xlabel("frame")
    ax.set_ylabel("area (cells)")
    ax.set_title(f"Area vs time (top {len(selected)} longest tracks)")
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(selected) <= 12:
        ax.legend(loc="best", fontsize=7)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
