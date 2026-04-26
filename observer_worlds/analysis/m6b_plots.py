"""M6B — plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INTERVENTION_PALETTE: dict[str, str] = {
    "sham":                    "#bbbbbb",
    "hidden_invisible_local":  "#1f77b4",
    "one_time_scramble_local": "#9467bd",
    "fiber_replacement_local": "#17becf",
    "hidden_invisible_far":    "#7f7f7f",
    "visible_match_count":     "#d62728",
}
CONDITION_PALETTE: dict[str, str] = {
    "coherent_4d":               "#1f77b4",
    "per_step_hidden_shuffled_4d": "#ff7f0e",
}


def _empty(out_path: Path, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save(fig, out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _filter(rows, **fields):
    out = []
    for r in rows:
        ok = True
        for k, v in fields.items():
            if getattr(r, k) != v:
                ok = False; break
        if ok:
            out.append(r)
    return out


def plot_hce_by_condition_boxplot(rows, out_path, *, horizon: int) -> None:
    """Boxplot of HCE per condition × intervention at the given horizon."""
    out_path = Path(out_path)
    data, labels, colors = [], [], []
    for cond in sorted({r.condition for r in rows}):
        for intv in sorted({r.intervention_type for r in rows}):
            sub = _filter(rows, condition=cond, intervention_type=intv, horizon=horizon)
            if not sub:
                continue
            data.append([r.hidden_causal_dependence for r in sub])
            labels.append(f"{cond[:4]}/{intv}")
            colors.append(INTERVENTION_PALETTE.get(intv, "#999999"))
    if not data:
        _empty(out_path, "no rows")
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(data)), 6))
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(0.6)
    ax.set_ylabel("HCE")
    ax.set_title(f"HCE by condition × intervention (h={horizon})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    _save(fig, out_path)


def plot_future_divergence_by_condition(rows, out_path, *, horizon: int) -> None:
    """Boxplot of raw future_projection_divergence per condition × intervention."""
    out_path = Path(out_path)
    data, labels, colors = [], [], []
    for cond in sorted({r.condition for r in rows}):
        for intv in sorted({r.intervention_type for r in rows}):
            sub = _filter(rows, condition=cond, intervention_type=intv, horizon=horizon)
            if not sub:
                continue
            data.append([r.future_projection_divergence for r in sub])
            labels.append(f"{cond[:4]}/{intv}")
            colors.append(INTERVENTION_PALETTE.get(intv, "#999999"))
    if not data:
        _empty(out_path, "no rows"); return
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(data)), 6))
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(0.6)
    ax.set_ylabel("future projection divergence")
    ax.set_title(f"Raw future divergence (h={horizon})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    _save(fig, out_path)


def plot_hidden_vs_sham_delta(rows, out_path, *, horizon: int) -> None:
    out_path = Path(out_path)
    data, labels, colors = [], [], []
    for intv in ("hidden_invisible_local", "one_time_scramble_local",
                 "fiber_replacement_local", "hidden_invisible_far",
                 "visible_match_count"):
        sub = _filter(rows, condition="coherent_4d",
                      intervention_type=intv, horizon=horizon)
        if not sub: continue
        data.append([r.hidden_vs_sham_delta for r in sub])
        labels.append(intv)
        colors.append(INTERVENTION_PALETTE.get(intv, "#999999"))
    if not data:
        _empty(out_path, "no rows"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c); box.set_alpha(0.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("future_div - sham_div")
    ax.set_title(f"Sham-subtracted divergence by intervention (h={horizon})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    _save(fig, out_path)


def plot_hidden_vs_far_control_delta(rows, out_path, *, horizon: int) -> None:
    out_path = Path(out_path)
    sub = _filter(rows, condition="coherent_4d",
                  intervention_type="hidden_invisible_local", horizon=horizon)
    if not sub:
        _empty(out_path, "no rows"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = np.array([r.hidden_vs_far_delta for r in sub])
    ax.hist(diffs, bins=25, color="#1f77b4", alpha=0.7)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(diffs.mean(), color="red", linewidth=1.5,
               label=f"mean = {diffs.mean():+.4f}")
    ax.set_xlabel("local_future_div(local) - local_future_div(far)")
    ax.set_ylabel("count")
    ax.set_title(f"Localization control: local hidden vs far hidden (h={horizon})")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_hce_by_horizon(rows, out_path) -> None:
    out_path = Path(out_path)
    horizons = sorted({r.horizon for r in rows})
    if not horizons:
        _empty(out_path, "no rows"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in sorted({r.condition for r in rows}):
        for intv in ("hidden_invisible_local", "one_time_scramble_local",
                     "fiber_replacement_local", "hidden_invisible_far"):
            xs, ys = [], []
            for h in horizons:
                sub = _filter(rows, condition=cond, intervention_type=intv, horizon=h)
                if not sub: continue
                xs.append(h)
                ys.append(np.mean([r.future_projection_divergence for r in sub]))
            if not xs: continue
            color = INTERVENTION_PALETTE.get(intv, "#999999")
            ls = "-" if cond == "coherent_4d" else "--"
            ax.plot(xs, ys, marker="o", linestyle=ls, color=color,
                    label=f"{cond[:4]}/{intv}", linewidth=1.6)
    ax.set_xlabel("rollout horizon")
    ax.set_ylabel("mean future_projection_divergence")
    ax.set_title("Future divergence vs horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_path)


def plot_scatter_vs_observer_score(rows, out_path, *, horizon: int) -> None:
    out_path = Path(out_path)
    sub = _filter(rows, condition="coherent_4d",
                  intervention_type="hidden_invisible_local", horizon=horizon)
    sub = [r for r in sub if r.observer_score is not None]
    if len(sub) < 2:
        _empty(out_path, "insufficient rows with observer_score"); return
    obs = np.array([r.observer_score for r in sub])
    fut = np.array([r.future_projection_divergence for r in sub])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(obs, fut, s=20, alpha=0.7, color="#1f77b4")
    if obs.std() > 1e-12:
        r = np.corrcoef(obs, fut)[0, 1]
        ax.set_title(f"future div vs observer_score (h={horizon}, Pearson r={r:+.2f})")
    ax.set_xlabel("observer_score")
    ax.set_ylabel("future_projection_divergence")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_scatter_vs_lifetime(rows, out_path, *, horizon: int) -> None:
    out_path = Path(out_path)
    sub = _filter(rows, condition="coherent_4d",
                  intervention_type="hidden_invisible_local", horizon=horizon)
    if len(sub) < 2:
        _empty(out_path, "no rows"); return
    age = np.array([r.candidate_lifetime for r in sub])
    fut = np.array([r.future_projection_divergence for r in sub])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(age, fut, s=20, alpha=0.7, color="#1f77b4")
    if age.std() > 1e-12:
        r = np.corrcoef(age, fut)[0, 1]
        ax.set_title(f"future div vs candidate lifetime (Pearson r={r:+.2f})")
    ax.set_xlabel("candidate lifetime"); ax.set_ylabel("future_div")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_hce_by_rule_source(rows, out_path, *, horizon: int) -> None:
    out_path = Path(out_path)
    by_src: dict[str, list[float]] = defaultdict(list)
    sub = _filter(rows, condition="coherent_4d",
                  intervention_type="hidden_invisible_local", horizon=horizon)
    for r in sub:
        by_src[r.rule_source].append(r.future_projection_divergence)
    if not by_src:
        _empty(out_path, "no rows"); return
    sources = sorted(by_src)
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([by_src[s] for s in sources], tick_labels=sources,
                    showmeans=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#1f77b4"); box.set_alpha(0.6)
    ax.set_ylabel("future_projection_divergence")
    ax.set_title(f"Future divergence by rule source (h={horizon})")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_coherent_vs_per_step_shuffled_paired(rows, out_path, *, horizon: int) -> None:
    """Per (rule, candidate), x = shuf future div, y = coh future div."""
    out_path = Path(out_path)
    by_key_coh: dict = {}
    by_key_shuf: dict = {}
    for r in rows:
        if (r.intervention_type == "hidden_invisible_local"
                and r.horizon == horizon):
            key = (r.rule_id, r.seed, r.candidate_id)
            if r.condition == "coherent_4d":
                by_key_coh.setdefault(key, []).append(r.future_projection_divergence)
            elif r.condition == "per_step_hidden_shuffled_4d":
                by_key_shuf.setdefault(key, []).append(r.future_projection_divergence)
    common = sorted(set(by_key_coh) & set(by_key_shuf))
    if not common:
        # Fall back to per-rule means.
        coh_by_rule: dict = defaultdict(list); shuf_by_rule: dict = defaultdict(list)
        for r in rows:
            if r.intervention_type != "hidden_invisible_local" or r.horizon != horizon:
                continue
            if r.condition == "coherent_4d":
                coh_by_rule[r.rule_id].append(r.future_projection_divergence)
            elif r.condition == "per_step_hidden_shuffled_4d":
                shuf_by_rule[r.rule_id].append(r.future_projection_divergence)
        common_rules = sorted(set(coh_by_rule) & set(shuf_by_rule))
        if not common_rules:
            _empty(out_path, "no matched coh/shuf data"); return
        x = np.array([np.mean(shuf_by_rule[r]) for r in common_rules])
        y = np.array([np.mean(coh_by_rule[r]) for r in common_rules])
        ttl = f"per-rule means (N={len(common_rules)})"
    else:
        x = np.array([np.mean(by_key_shuf[k]) for k in common])
        y = np.array([np.mean(by_key_coh[k]) for k in common])
        ttl = f"per-(rule,seed,candidate) (N={len(common)})"
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.7, color="#1f77b4")
    lim = max(float(x.max()) if x.size else 0, float(y.max()) if y.size else 0, 0.001) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.6, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("per_step_shuffled future_div")
    ax.set_ylabel("coherent future_div")
    n_coh_wins = int((y > x).sum())
    ax.set_title(f"coherent vs per-step shuffled paired ({ttl}); coh wins {n_coh_wins}/{len(x)}")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out_path)


def plot_local_vs_far_paired(rows, out_path, *, horizon: int) -> None:
    """Per candidate: x = local_future_div(far), y = local_future_div(local)."""
    out_path = Path(out_path)
    local_by_key: dict = {}
    far_by_key: dict = {}
    for r in rows:
        if r.condition != "coherent_4d" or r.horizon != horizon: continue
        key = (r.rule_id, r.seed, r.candidate_id)
        if r.intervention_type == "hidden_invisible_local":
            local_by_key.setdefault(key, []).append(r.local_future_divergence)
        elif r.intervention_type == "hidden_invisible_far":
            far_by_key.setdefault(key, []).append(r.local_future_divergence)
    common = sorted(set(local_by_key) & set(far_by_key))
    if not common:
        _empty(out_path, "no matched local/far data"); return
    x = np.array([np.mean(far_by_key[k]) for k in common])
    y = np.array([np.mean(local_by_key[k]) for k in common])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.7, color="#1f77b4")
    lim = max(float(x.max()) if x.size else 0, float(y.max()) if y.size else 0, 0.001) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.6, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("local_future_div under far hidden")
    ax.set_ylabel("local_future_div under local hidden")
    n_local_wins = int((y > x).sum())
    ax.set_title(f"local vs far hidden (per candidate, N={len(common)}); "
                 f"local wins {n_local_wins}/{len(common)}")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out_path)


def plot_hidden_vs_visible_ratio_distribution(rows, out_path, *, horizon: int) -> None:
    out_path = Path(out_path)
    sub = _filter(rows, condition="coherent_4d",
                  intervention_type="hidden_invisible_local", horizon=horizon)
    if not sub:
        _empty(out_path, "no rows"); return
    ratios = np.array([r.hidden_vs_visible_ratio for r in sub])
    ratios = ratios[np.isfinite(ratios)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=30, color="#1f77b4", alpha=0.7)
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.6, label="ratio=1 (parity)")
    ax.axvline(ratios.mean(), color="red", linewidth=1.5,
               label=f"mean = {ratios.mean():+.3f}")
    ax.set_xlabel("hidden / visible ratio")
    ax.set_ylabel("count")
    ax.set_title(f"hidden_vs_visible_ratio distribution (h={horizon})")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_initial_projection_delta_histogram(rows, out_path) -> None:
    """All hidden-invisible interventions should have init_delta ≈ 0.
    Visible should have init_delta > 0. This plot is a regression sanity check."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for intv in INTERVENTION_PALETTE:
        sub = _filter(rows, intervention_type=intv)
        if not sub: continue
        deltas = np.array([r.initial_projection_delta for r in sub])
        ax.hist(deltas, bins=30, alpha=0.5,
                label=f"{intv} (mean={deltas.mean():.4f})",
                color=INTERVENTION_PALETTE[intv])
    ax.set_xlabel("initial_projection_delta (full-grid L1)")
    ax.set_ylabel("count")
    ax.set_title("Initial projection delta — regression sanity check")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def write_all_m6b_plots(rows, out_dir, *, horizon: int) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_hce_by_condition_boxplot(rows, out_dir / "hce_by_condition_boxplot.png", horizon=horizon)
    plot_future_divergence_by_condition(rows, out_dir / "future_divergence_by_condition.png", horizon=horizon)
    plot_hidden_vs_sham_delta(rows, out_dir / "hidden_vs_sham_delta_by_condition.png", horizon=horizon)
    plot_hidden_vs_far_control_delta(rows, out_dir / "hidden_vs_far_control_delta.png", horizon=horizon)
    plot_hce_by_horizon(rows, out_dir / "hce_by_horizon.png")
    plot_scatter_vs_observer_score(rows, out_dir / "hce_vs_observer_score.png", horizon=horizon)
    plot_scatter_vs_lifetime(rows, out_dir / "hce_vs_candidate_lifetime.png", horizon=horizon)
    plot_hce_by_rule_source(rows, out_dir / "hce_by_rule_source.png", horizon=horizon)
    plot_coherent_vs_per_step_shuffled_paired(
        rows, out_dir / "coherent_vs_per_step_shuffled_paired.png", horizon=horizon
    )
    plot_local_vs_far_paired(
        rows, out_dir / "local_hidden_vs_far_hidden_paired.png", horizon=horizon
    )
    plot_hidden_vs_visible_ratio_distribution(
        rows, out_dir / "hidden_vs_visible_ratio_distribution.png", horizon=horizon
    )
    plot_initial_projection_delta_histogram(
        rows, out_dir / "initial_projection_delta_histogram.png"
    )
