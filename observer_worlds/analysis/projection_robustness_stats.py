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

    path.write_text("\n".join(lines), encoding="utf-8")
