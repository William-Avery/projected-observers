"""Per-projection statistical aggregation for Follow-up Topic 1.

The runner produces one row per (rule, seed, projection, candidate)
under ``candidate_metrics.csv``; this module aggregates that flat list
into per-projection means, standard deviations, and counts that the
summary report and plots consume.

Stage 2: smoke-quality means + per-projection candidate counts. No
grouped-bootstrap CIs yet — those are Stage 5+ work and require enough
groups to be meaningful.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# Metric inventory for documentation / tests.
PROJECTION_METRICS: tuple[str, ...] = (
    "n_candidates",
    "mean_HCE",
    "mean_far_HCE",
    "mean_hidden_vs_far_delta",
    "mean_hidden_vs_sham_delta",
    "mean_initial_projection_delta",
    "fraction_clean_initial_projection",  # init delta < 1e-6
    "mean_lifetime",
)


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _safe_std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def project_metrics_template(projections: Iterable[str]) -> dict:
    """Empty-result template; preserved from Stage 1 for tests."""
    return {
        proj: {m: None for m in PROJECTION_METRICS}
        for proj in projections
    }


def aggregate_per_projection(
    candidate_rows: list[dict],
    projections: Iterable[str],
) -> dict:
    """Compute per-projection summary stats from ``candidate_metrics.csv``
    rows.

    Stage 2B: a candidate is **valid** iff its hidden-invisible
    perturbation was accepted (``row["valid"] == True``). Mean HCE /
    far / sham / delta are computed **only over valid candidates**.
    Invalid candidates are counted, their invalid reasons are
    aggregated, and ``mean_initial_projection_delta`` is reported over
    valid candidates (where it should be ~0 by construction).
    """
    def _truthy_valid(r):
        v = r.get("valid")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return False

    by_proj: dict[str, list[dict]] = defaultdict(list)
    for r in candidate_rows:
        by_proj[r["projection"]].append(r)
    per_projection: dict[str, dict] = {}
    for proj in projections:
        rs = by_proj.get(proj, [])
        n_total = len(rs)
        valid_rows = [r for r in rs if _truthy_valid(r)]
        n_valid = len(valid_rows)
        n_invalid = n_total - n_valid

        # Invalid-reason histogram.
        reason_counts: dict[str, int] = {}
        for r in rs:
            if _truthy_valid(r):
                continue
            reason = r.get("invalid_reason") or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if n_total == 0:
            per_projection[proj] = {
                "n_candidates_total": 0,
                "n_valid_hidden_invisible": 0,
                "n_invalid_hidden_invisible": 0,
                "invalid_reason_counts": {},
                "n_clean_initial_projection": 0,
                "fraction_clean_initial_projection": None,
                "mean_HCE": None, "std_HCE": None,
                "mean_far_HCE": None,
                "mean_hidden_vs_far_delta": None,
                "mean_hidden_vs_sham_delta": None,
                "mean_initial_projection_delta": None,
                "mean_lifetime": None,
                "_status": "no candidates measured",
            }
            continue

        def col(rows, k):
            return [float(r[k]) for r in rows
                    if r.get(k) not in (None, "", "None")]

        # Means computed only over valid candidates.
        hce_values = col(valid_rows, "HCE")
        far_values = col(valid_rows, "far_HCE")
        delta_far = col(valid_rows, "hidden_vs_far_delta")
        delta_sham = col(valid_rows, "hidden_vs_sham_delta")
        init_delta_valid = col(valid_rows, "initial_projection_delta")
        lifetimes = col(valid_rows, "lifetime")

        n_clean = sum(1 for v in init_delta_valid if v < 1e-6)

        per_projection[proj] = {
            "n_candidates_total": n_total,
            "n_valid_hidden_invisible": n_valid,
            "n_invalid_hidden_invisible": n_invalid,
            "invalid_reason_counts": reason_counts,
            "n_clean_initial_projection": n_clean,
            "fraction_clean_initial_projection":
                (n_clean / n_valid) if n_valid else None,
            "mean_HCE": _safe_mean(hce_values),
            "std_HCE": _safe_std(hce_values),
            "mean_far_HCE": _safe_mean(far_values),
            "mean_hidden_vs_far_delta": _safe_mean(delta_far),
            "mean_hidden_vs_sham_delta": _safe_mean(delta_sham),
            "mean_initial_projection_delta": _safe_mean(init_delta_valid),
            "mean_lifetime": _safe_mean(lifetimes),
            "_status": "ok" if n_valid > 0 else
                       "all candidates invalid under this projection",
        }
    return {
        "stage": 2,
        "metrics_recorded": list(PROJECTION_METRICS),
        "per_projection": per_projection,
        "candidate_count_by_projection": {
            p: per_projection[p]["n_candidates_total"] for p in projections
        },
    }


# ---------------------------------------------------------------------------
# Cross-source extensions (Stage 5C)
# ---------------------------------------------------------------------------


SOURCES_DEFAULT: tuple[str, ...] = (
    "M7_HCE_optimized",
    "M4C_observer_optimized",
    "M4A_viability",
)


def _truthy_valid(r):
    v = r.get("valid")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return False


def aggregate_per_projection_and_source(
    candidate_rows: list[dict], projections, sources=SOURCES_DEFAULT,
) -> dict:
    """Per (projection × source) means over **valid** candidates.

    Returns ``{"per_projection_source": {projection: {source: {...}}}}``.
    Used by the Stage-5C summary report and plots.
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in candidate_rows:
        if not _truthy_valid(r):
            continue
        by_key[(r["projection"], r["rule_source"])].append(r)
    out: dict[str, dict[str, dict]] = {}
    for proj in projections:
        out[proj] = {}
        for src in sources:
            rs = by_key.get((proj, src), [])
            if not rs:
                out[proj][src] = {"n": 0}
                continue
            def col(k):
                return [float(r[k]) for r in rs
                        if r.get(k) not in (None, "", "None")]
            hces = col("HCE")
            fars = col("far_HCE")
            init = col("initial_projection_delta")
            d_far = col("hidden_vs_far_delta")
            cli = col("candidate_locality_index") \
                if any("candidate_locality_index" in r for r in rs) else []
            # Scale-invariant relative HCE: HCE / (HCE + far_HCE) per
            # candidate, then mean. Works across binary and continuous
            # projections.
            rel_hce = []
            for h, f in zip(hces, fars):
                denom = abs(h) + abs(f)
                if denom > 1e-12:
                    rel_hce.append(h / denom)
            out[proj][src] = {
                "n": len(rs),
                "mean_HCE": _safe_mean(hces),
                "mean_far_HCE": _safe_mean(fars),
                "mean_hidden_vs_far_delta": _safe_mean(d_far),
                "mean_initial_projection_delta": _safe_mean(init),
                "mean_relative_HCE": _safe_mean(rel_hce),
            }
    return {"per_projection_source": out}


def _grouped_bootstrap_diff(
    a_vals, a_groups, b_vals, b_groups, *, n_boot: int = 2000, seed: int = 0,
):
    """Mean(a) − Mean(b) bootstrap CI by resampling whole ``(rule, seed)``
    groups with replacement (cf. M8F)."""
    import numpy as np
    if len(a_vals) == 0 or len(b_vals) == 0:
        return None
    a_vals = np.asarray(a_vals, dtype=np.float64)
    b_vals = np.asarray(b_vals, dtype=np.float64)
    a_groups = np.asarray(a_groups)
    b_groups = np.asarray(b_groups)
    rng = np.random.default_rng(int(seed))
    ua = np.unique(a_groups); ub = np.unique(b_groups)
    if ua.size < 2 or ub.size < 2:
        return {
            "mean_a": float(a_vals.mean()),
            "mean_b": float(b_vals.mean()),
            "mean_diff": float(a_vals.mean() - b_vals.mean()),
            "ci_low": None, "ci_high": None,
            "n_a": int(a_vals.size), "n_b": int(b_vals.size),
            "n_groups_a": int(ua.size), "n_groups_b": int(ub.size),
            "_status": "too few groups for bootstrap",
        }
    idx_a = {g: np.where(a_groups == g)[0] for g in ua}
    idx_b = {g: np.where(b_groups == g)[0] for g in ub}
    diffs = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        sa = rng.choice(ua, size=ua.size, replace=True)
        sb = rng.choice(ub, size=ub.size, replace=True)
        ia = np.concatenate([idx_a[g] for g in sa])
        ib = np.concatenate([idx_b[g] for g in sb])
        diffs[i] = float(a_vals[ia].mean() - b_vals[ib].mean())
    return {
        "mean_a": float(a_vals.mean()),
        "mean_b": float(b_vals.mean()),
        "mean_diff": float(a_vals.mean() - b_vals.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_a": int(a_vals.size), "n_b": int(b_vals.size),
        "n_groups_a": int(ua.size), "n_groups_b": int(ub.size),
    }


def _cliffs_delta(a, b):
    import numpy as np
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(((a[:, None] > b[None, :]).sum()
                   - (a[:, None] < b[None, :]).sum())
                  / (a.size * b.size))


def compare_m7_vs_baselines_by_projection(
    candidate_rows: list[dict], projections,
    baselines=("M4C_observer_optimized", "M4A_viability"),
    *, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Per projection, compute M7-vs-baseline grouped-bootstrap CI on
    HCE, hidden_vs_far_delta, and relative_HCE."""
    out: dict[str, dict[str, dict]] = {}
    for proj in projections:
        rows_by_src: dict[str, list[dict]] = defaultdict(list)
        for r in candidate_rows:
            if r["projection"] != proj or not _truthy_valid(r):
                continue
            rows_by_src[r["rule_source"]].append(r)
        m7 = rows_by_src.get("M7_HCE_optimized", [])
        if not m7:
            out[proj] = {b: {"_status": "no M7 candidates"} for b in baselines}
            continue
        out[proj] = {}
        for b in baselines:
            base = rows_by_src.get(b, [])
            if not base:
                out[proj][b] = {"_status": f"no {b} candidates"}
                continue
            entry: dict[str, dict] = {}
            for metric in ("HCE", "hidden_vs_far_delta"):
                a_vals = [float(r[metric]) for r in m7
                          if r.get(metric) not in (None, "", "None")]
                b_vals = [float(r[metric]) for r in base
                          if r.get(metric) not in (None, "", "None")]
                a_groups = [f"{r['rule_id']}|{r['seed']}" for r in m7
                            if r.get(metric) not in (None, "", "None")]
                b_groups = [f"{r['rule_id']}|{r['seed']}" for r in base
                            if r.get(metric) not in (None, "", "None")]
                cmp = _grouped_bootstrap_diff(
                    a_vals, a_groups, b_vals, b_groups,
                    n_boot=n_boot, seed=seed,
                )
                if cmp is not None:
                    cmp["cliffs_delta"] = _cliffs_delta(a_vals, b_vals)
                    cmp["ci_excludes_zero"] = (
                        cmp.get("ci_low") is not None
                        and (cmp["ci_low"] > 0.0 or cmp["ci_high"] < 0.0)
                    )
                entry[metric] = cmp
            out[proj][b] = entry
    return {"m7_vs_baselines_by_projection": out}


def summarize(per_projection: dict) -> dict:
    """Stage-1 compatibility shim used by ``test_projection_robustness``.

    Wraps a per-projection metrics template so the existing import
    surface keeps working while Stage 2 adds the real
    :func:`aggregate_per_projection` for the runner.
    """
    return {
        "stage": 1 if (
            isinstance(per_projection, dict)
            and any(isinstance(v, dict) and all(vv is None for vv in v.values())
                    for v in per_projection.values())
        ) else 2,
        "metrics_recorded": list(PROJECTION_METRICS),
        "per_projection": per_projection,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(summary: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Follow-up Topic 1 — projection robustness (Stage 2)")
    lines.append("")
    lines.append(
        "Reuses the Stage-1 scaffold (`docs/FOLLOWUP_RESEARCH_ROADMAP.md`). "
        "Each (rule, seed) cell runs the 4D substrate **once**; every "
        "requested projection consumes the same in-memory state stream. "
        "Per-candidate HCE is measured by candidate-local hidden "
        "perturbations; the `far` control varies the perturbation "
        "location; the `sham` value is identically zero by construction.\n"
    )
    lines.append("")
    lines.append("**Read every comparison below as classifier-conditional and "
                 "projection-conditional.** Mechanism-class fractions are "
                 "deferred to Stage 5+ (full M8 classifier integration "
                 "across projections is non-trivial).\n")
    lines.append("")

    # Headline table — gated on validity (Stage 2B)
    lines.append("## Per-projection summary (HCE means over **valid** candidates only)")
    lines.append("")
    lines.append(_md_row([
        "projection", "n_total", "n_valid", "n_invalid",
        "mean HCE (valid)", "mean far_HCE (valid)",
        "mean (HCE − far)", "mean init_delta (valid)",
        "frac_clean_init",
    ]))
    lines.append(_md_row(["---"] + ["---:"] * 8))
    for proj, agg in summary["per_projection"].items():
        n_total = agg.get("n_candidates_total", 0)
        if n_total == 0:
            lines.append(_md_row([proj, "0", "0", "0",
                                   "—", "—", "—", "—", "—"]))
            continue
        cells = [
            proj,
            str(n_total),
            str(agg["n_valid_hidden_invisible"]),
            str(agg["n_invalid_hidden_invisible"]),
            f"{agg['mean_HCE']:+.4f}" if agg["mean_HCE"] is not None else "—",
            f"{agg['mean_far_HCE']:+.4f}" if agg["mean_far_HCE"] is not None else "—",
            f"{agg['mean_hidden_vs_far_delta']:+.4f}"
                if agg["mean_hidden_vs_far_delta"] is not None else "—",
            f"{agg['mean_initial_projection_delta']:+.4f}"
                if agg["mean_initial_projection_delta"] is not None else "—",
            f"{agg['fraction_clean_initial_projection']:.2f}"
                if agg["fraction_clean_initial_projection"] is not None else "—",
        ]
        lines.append(_md_row(cells))
    lines.append("")

    # Invalid reasons.
    any_invalid = any(
        agg["n_invalid_hidden_invisible"] > 0
        for agg in summary["per_projection"].values()
    )
    if any_invalid:
        lines.append("## Invalid hidden-invisible perturbations by projection")
        lines.append("")
        lines.append(_md_row(["projection", "n_invalid", "reasons"]))
        lines.append(_md_row(["---", "---:", "---"]))
        for proj, agg in summary["per_projection"].items():
            n_inv = agg.get("n_invalid_hidden_invisible", 0)
            if n_inv == 0:
                continue
            reasons = agg.get("invalid_reason_counts", {})
            reason_str = "; ".join(
                f"{k} ({v})" for k, v in sorted(reasons.items(),
                                                key=lambda x: -x[1])
            ) or "—"
            lines.append(_md_row([proj, str(n_inv), reason_str]))
        lines.append("")

    # Caveats per projection.
    lines.append("## Per-projection caveats")
    lines.append("")
    lines.append("* **mean_threshold / sum_threshold** — natural threshold "
                 "margin; standard hidden-invisible perturbation logic "
                 "applies.")
    lines.append("* **max_projection / parity_projection** — no natural "
                 "threshold margin; threshold-audit metrics are N/A. The "
                 "hidden-invisible perturbation here is empirical (we "
                 "verify post-hoc that t = peak projection is unchanged "
                 "by checking `initial_projection_delta`).")
    lines.append("* **random_linear_projection** — continuous output; the "
                 "binary detector consumes a per-frame median threshold. "
                 "Smoke-level binarisation; production refinement needed.")
    lines.append("* **multi_channel_projection** — channel 0 is consumed by "
                 "the binary detector. Smoke-level reduction; production "
                 "should treat all channels.")
    lines.append("")
    if summary.get("wall_time_seconds_sweep") is not None:
        lines.append(
            f"Wall time (sweep): "
            f"{summary['wall_time_seconds_sweep']:.1f}s, "
            f"{summary.get('n_cells', 0)} cells, "
            f"{summary.get('n_candidate_rows', 0)} candidate rows.\n"
        )

    # Stage 5C: cross-source tables.
    pps = summary.get("per_projection_source", {})
    if pps:
        lines.append("")
        lines.append("## Per (projection × source) means (valid candidates only)")
        lines.append("")
        sources = summary.get("sources_present", [])
        # mean HCE table.
        lines.append(_md_row(["projection"]
                              + [f"{s} n / mean HCE" for s in sources]))
        lines.append(_md_row(["---"] + ["---:"] * len(sources)))
        for proj, by_src in pps.items():
            cells = [proj]
            for s in sources:
                d = by_src.get(s, {})
                if d.get("n", 0) == 0:
                    cells.append("0 / —")
                else:
                    h = d.get("mean_HCE")
                    cells.append(
                        f"{d['n']} / "
                        f"{h:+.4f}" if h is not None else f"{d['n']} / —")
            lines.append(_md_row(cells))
        lines.append("")
        # relative_HCE table — scale-invariant.
        lines.append("### Mean relative_HCE = HCE / (HCE + far_HCE), per (projection × source)")
        lines.append("")
        lines.append(
            "Scale-invariant in [0, 1]; > 0.5 ⇒ candidate-local "
            "(hidden perturbation has more effect than far)."
        )
        lines.append("")
        lines.append(_md_row(["projection"] + list(sources)))
        lines.append(_md_row(["---"] + ["---:"] * len(sources)))
        for proj, by_src in pps.items():
            cells = [proj]
            for s in sources:
                d = by_src.get(s, {})
                v = d.get("mean_relative_HCE")
                cells.append(f"{v:+.3f}" if v is not None else "—")
            lines.append(_md_row(cells))
        lines.append("")

    cmps = summary.get("m7_vs_baselines_by_projection", {})
    if cmps:
        lines.append("## M7 vs baseline by projection (grouped bootstrap)")
        lines.append("")
        lines.append(_md_row([
            "projection", "comparison", "metric",
            "mean_M7", "mean_base", "diff", "95% CI",
            "CI excl 0", "Cliff's δ",
        ]))
        lines.append(_md_row(["---"] * 9))
        for proj, by_b in cmps.items():
            for b, by_metric in by_b.items():
                if isinstance(by_metric, dict) and "_status" in by_metric:
                    lines.append(_md_row([
                        proj, f"M7_vs_{b}", "—",
                        "—", "—", "—", "—", "—", "—",
                    ]))
                    continue
                if not isinstance(by_metric, dict):
                    continue
                for metric, cmp in by_metric.items():
                    if not isinstance(cmp, dict):
                        continue
                    if cmp.get("_status"):
                        ci = "(too few groups)"
                        excl = "—"
                    elif cmp.get("ci_low") is None:
                        ci = "—"; excl = "—"
                    else:
                        ci = f"[{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}]"
                        excl = "yes" if cmp.get("ci_excludes_zero") else "no"
                    lines.append(_md_row([
                        proj, f"M7_vs_{b.split('_')[0]}", metric,
                        f"{cmp.get('mean_a', 0):+.4f}",
                        f"{cmp.get('mean_b', 0):+.4f}",
                        f"{cmp.get('mean_diff', 0):+.4f}",
                        ci, excl,
                        f"{cmp.get('cliffs_delta', 0):+.3f}"
                            if cmp.get("cliffs_delta") is not None else "—",
                    ]))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
