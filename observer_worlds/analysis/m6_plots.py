"""M6 — Hidden Causal Dependence plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observer_worlds.experiments._m6_hidden_causal import HiddenCausalReport


PERTURBATION_COLORS: dict[str, str] = {
    "hidden_invisible":    "#1f77b4",   # blue
    "visible_match_count": "#d62728",   # red
}
CONDITION_COLORS_M6: dict[str, str] = {
    "coherent": "#1f77b4",
    "shuffled": "#ff7f0e",
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
# Per-condition plots
# ---------------------------------------------------------------------------


def plot_aggregate_divergence_per_perturbation(
    reports: list[HiddenCausalReport],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """For one condition: mean ± std divergence trajectory, hidden_invisible
    vs visible_match_count."""
    out_path = Path(out_path)
    valid = [r for r in reports if r.hidden_invisible.n_replicates > 0]
    if not valid:
        _empty_figure("no reports", out_path)
        return
    n_steps = valid[0].n_steps
    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, color in PERTURBATION_COLORS.items():
        traj_attr = "hidden_invisible" if kind == "hidden_invisible" else "visible_match_count"
        # Stack per-step means across reports.
        means_per_report = np.array([
            getattr(r, traj_attr).full_grid_l1_mean for r in valid
            if len(getattr(r, traj_attr).full_grid_l1_mean) == n_steps
        ])
        if means_per_report.size == 0:
            continue
        m = means_per_report.mean(axis=0)
        s = means_per_report.std(axis=0)
        ax.plot(steps, m, color=color, linewidth=1.8, label=kind)
        ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("mean ± stdev projected L1")
    ax.set_title(title or f"Per-perturbation divergence (N={len(valid)} candidates)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_hce_distribution(
    reports: list[HiddenCausalReport],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Histogram of per-candidate HCE values, with a vertical line at 0
    and at the visible-final mean for context."""
    out_path = Path(out_path)
    if not reports:
        _empty_figure("no reports", out_path)
        return
    hce = np.array([r.HCE for r in reports])
    vis = np.array([r.visible_final_l1 for r in reports])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hce, bins=15, color="#1f77b4", alpha=0.6, label="HCE (hidden_invisible)")
    ax.hist(vis, bins=15, color="#d62728", alpha=0.4,
            label="visible_match_count final L1")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.axvline(hce.mean(), color="#1f77b4", linewidth=1.5, linestyle=":",
               label=f"mean HCE = {hce.mean():.4f}")
    ax.set_xlabel("final-step projected L1 divergence")
    ax.set_ylabel("count")
    ax.set_title(title or f"HCE distribution across {len(reports)} candidates")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_hce_vs_visible_scatter(
    reports: list[HiddenCausalReport],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Per-candidate scatter: x = visible_final_l1, y = HCE. Diagonal y=x.
    Points above the diagonal mean hidden perturbations are MORE
    consequential than equal-magnitude visible ones."""
    out_path = Path(out_path)
    if not reports:
        _empty_figure("no reports", out_path)
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    x = np.array([r.visible_final_l1 for r in reports])
    y = np.array([r.HCE for r in reports])
    ax.scatter(x, y, s=40, alpha=0.7, color="#1f77b4")
    lim = max(float(x.max()), float(y.max()), 0.01) * 1.1
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", alpha=0.6,
            label="y = x (equal effect)")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("visible_match_count final L1")
    ax.set_ylabel("HCE (hidden_invisible final L1)")
    ax.set_title(title or "HCE vs bit-matched visible perturbation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Coherent vs shuffled comparison plots
# ---------------------------------------------------------------------------


def plot_hce_coherent_vs_shuffled_paired(
    coherent: list[HiddenCausalReport],
    shuffled: list[HiddenCausalReport],
    out_path: str | Path,
) -> None:
    """Paired scatter: x = shuffled HCE, y = coherent HCE, matched by track_id."""
    out_path = Path(out_path)
    sh_by_id = {r.track_id: r for r in shuffled}
    pairs = [(c, sh_by_id[c.track_id]) for c in coherent if c.track_id in sh_by_id]
    if not pairs:
        _empty_figure("no matched pairs", out_path)
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    x = np.array([s.HCE for _, s in pairs])
    y = np.array([c.HCE for c, _ in pairs])
    ax.scatter(x, y, s=50, alpha=0.7, color="#1f77b4")
    lim = max(float(x.max()), float(y.max()), 0.01) * 1.1
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", alpha=0.6,
            label="coh = shuf")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("shuffled HCE")
    ax.set_ylabel("coherent HCE")
    n_coh_wins = int((y > x).sum())
    ax.set_title(
        f"HCE coherent vs shuffled (paired, N={len(pairs)})  "
        f"coherent wins {n_coh_wins}/{len(pairs)}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_hce_boxplot_by_condition(
    reports_by_condition: dict[str, list[HiddenCausalReport]],
    out_path: str | Path,
) -> None:
    """Boxplot of HCE distribution per condition (coherent / shuffled)."""
    out_path = Path(out_path)
    data = []
    labels = []
    for cond, recs in reports_by_condition.items():
        if recs:
            data.append([r.HCE for r in recs])
            labels.append(f"{cond}\n(N={len(recs)})")
    if not data:
        _empty_figure("no reports", out_path)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True,
                    meanprops={"marker": "^", "markerfacecolor": "white",
                               "markeredgecolor": "black"})
    for i, (cond, _) in enumerate(zip(reports_by_condition.keys(), data)):
        bp["boxes"][i].set_facecolor(CONDITION_COLORS_M6.get(cond, "#7f7f7f"))
        bp["boxes"][i].set_alpha(0.6)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("HCE")
    ax.set_title("HCE distribution by condition")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def write_all_m6_plots(
    reports_by_condition: dict[str, list[HiddenCausalReport]],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond, recs in reports_by_condition.items():
        plot_aggregate_divergence_per_perturbation(
            recs, out_dir / f"aggregate_divergence_{cond}.png",
            title=f"{cond}: per-perturbation divergence (N={len(recs)})",
        )
        plot_hce_distribution(
            recs, out_dir / f"hce_distribution_{cond}.png",
            title=f"{cond}: HCE distribution (N={len(recs)})",
        )
        plot_hce_vs_visible_scatter(
            recs, out_dir / f"hce_vs_visible_{cond}.png",
            title=f"{cond}: HCE vs visible (N={len(recs)})",
        )
    if "coherent" in reports_by_condition and "shuffled" in reports_by_condition:
        plot_hce_coherent_vs_shuffled_paired(
            reports_by_condition["coherent"], reports_by_condition["shuffled"],
            out_dir / "hce_coherent_vs_shuffled_paired.png",
        )
        plot_hce_boxplot_by_condition(
            reports_by_condition, out_dir / "hce_boxplot_by_condition.png",
        )
