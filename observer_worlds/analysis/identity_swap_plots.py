"""Plot generators for Follow-up Topic 2 (Stage 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


PLOT_FILENAMES: tuple[str, ...] = (
    "hidden_identity_pull_distribution.png",
    "donor_vs_host_similarity.png",
    "projection_preservation_error.png",
    "identity_pull_by_projection_similarity.png",
    "example_swapped_trajectories.png",
)


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _is_truthy(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.lower() == "true"
    return bool(v)


def _placeholder(plt, path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=11, color="#666", transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_all_plots(
    summary: dict, score_rows: list[dict], plots_dir: Path,
) -> None:
    try:
        plt = _import_plt()
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = [r for r in score_rows
                  if _is_truthy(r.get("valid_swap"))
                  and r.get("hidden_identity_pull") is not None]

    # 1. hidden_identity_pull_distribution.
    try:
        path = plots_dir / "hidden_identity_pull_distribution.png"
        if not valid_rows:
            _placeholder(plt, path, "no valid swap pairs to plot")
        else:
            pulls = [float(r["hidden_identity_pull"]) for r in valid_rows]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(pulls, bins=20, color="#357", edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("hidden_identity_pull = donor_sim − host_sim")
            ax.set_ylabel("count of (pair × horizon) measurements")
            ax.set_title(
                f"Hidden identity pull distribution "
                f"(n={len(pulls)}; mean={float(np.mean(pulls)):+.4f})"
            )
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] hidden_identity_pull_distribution.png: {e!r}")

    # 2. donor vs host similarity scatter.
    try:
        path = plots_dir / "donor_vs_host_similarity.png"
        if not valid_rows:
            _placeholder(plt, path, "no valid swap pairs to plot")
        else:
            host = np.array([float(r["host_similarity"]) for r in valid_rows])
            donor = np.array([float(r["donor_similarity"]) for r in valid_rows])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(host, donor, alpha=0.5, color="#357")
            lim = max(host.max(), donor.max(), 0.0)
            ax.plot([0, lim], [0, lim], "--", color="black", linewidth=0.5)
            ax.set_xlabel("host_similarity (swapped vs original host)")
            ax.set_ylabel("donor_similarity (swapped vs original donor)")
            ax.set_title("donor vs host similarity per (pair × horizon)")
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] donor_vs_host_similarity.png: {e!r}")

    # 3. projection_preservation_error.
    try:
        path = plots_dir / "projection_preservation_error.png"
        # Histograms of preservation error per direction (use audit numbers
        # from summary).
        labels = ["A_with_B_hidden", "B_with_A_hidden"]
        means = [
            summary.get("mean_projection_preservation_error_a") or 0.0,
            summary.get("mean_projection_preservation_error_b") or 0.0,
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, color="#a37")
        ax.set_ylabel("mean projection_preservation_error")
        ax.set_title("Projection-preservation error by swap direction")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] projection_preservation_error.png: {e!r}")

    # 4. identity pull vs visible similarity (match quality).
    try:
        path = plots_dir / "identity_pull_by_projection_similarity.png"
        if not valid_rows:
            _placeholder(plt, path, "no valid swap pairs to plot")
        else:
            # Need visible_similarity per row — not in score_rows directly.
            # Reconstruct from pair_id by reading from summary's per-horizon
            # data; for simplicity in Stage-3 smoke, plot pull vs horizon as
            # an approximation of "match quality varies with horizon" curve.
            horizons = sorted({int(r["horizon"]) for r in valid_rows})
            means = []
            for h in horizons:
                rs = [r for r in valid_rows if int(r["horizon"]) == h]
                pulls = [float(r["hidden_identity_pull"]) for r in rs]
                means.append(float(np.mean(pulls)) if pulls else 0.0)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(horizons, means, marker="o", color="#357")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("horizon")
            ax.set_ylabel("mean hidden_identity_pull")
            ax.set_title(
                "mean hidden_identity_pull vs horizon "
                "(stand-in for match-quality plot in Stage 3 smoke)"
            )
            fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] identity_pull_by_projection_similarity.png: {e!r}")

    # 5. example_swapped_trajectories — Stage-3 smoke text placeholder.
    try:
        path = plots_dir / "example_swapped_trajectories.png"
        _placeholder(
            plt, path,
            "Stage 5+: example swapped vs original trajectories.\n"
            "Stage 3 smoke writes per-pair host/donor similarities at each "
            "horizon to identity_scores.csv;\n"
            "the trajectory-image visualization is a Stage-5+ deliverable.",
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] example_swapped_trajectories.png: {e!r}")
