"""Aggregator + summary writer for Follow-up Topic 2 (Stage 3)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean as _mean


IDENTITY_METRICS: tuple[str, ...] = (
    "projection_preservation_error",
    "future_similarity_to_host",
    "future_similarity_to_donor",
    "centroid_trajectory_distance",
    "area_trajectory_distance",
    "shape_trajectory_distance",
    "lifetime_change",
    "future_divergence",
    "identity_follow_hidden_score",
    "identity_follow_visible_score",
    "hidden_identity_pull",
)


def _safe_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(_mean(xs)) if xs else None


def _is_truthy(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.lower() == "true"
    return bool(v)


def aggregate_identity_results(results, score_rows: list[dict]) -> dict:
    """Build the stats_summary.json payload from per-pair results.

    Counts per direction (A_with_B_hidden / B_with_A_hidden):
      n_attempted, n_valid, n_invalid, mean projection_preservation_error,
      mean hidden_identity_pull (over valid pairs at all horizons),
      mean host_similarity, mean donor_similarity.

    Per-horizon aggregates also reported.
    """
    n_pairs = len(results)
    valid_a = [p for p in results if p.valid_swap_a]
    valid_b = [p for p in results if p.valid_swap_b]
    invalid_reasons_a: Counter = Counter()
    invalid_reasons_b: Counter = Counter()
    for p in results:
        if not p.valid_swap_a and p.invalid_reason_a:
            invalid_reasons_a[p.invalid_reason_a] += 1
        if not p.valid_swap_b and p.invalid_reason_b:
            invalid_reasons_b[p.invalid_reason_b] += 1

    # Pull host/donor/pull values from valid score rows.
    valid_score_rows = [r for r in score_rows
                        if _is_truthy(r.get("valid_swap"))
                        and r.get("hidden_identity_pull") is not None]
    rows_a = [r for r in valid_score_rows if r["direction"] == "A_with_B_hidden"]
    rows_b = [r for r in valid_score_rows if r["direction"] == "B_with_A_hidden"]

    # Per-horizon breakdown (across both directions).
    horizons = sorted({int(r["horizon"]) for r in valid_score_rows})
    per_horizon: dict[int, dict] = {}
    for h in horizons:
        rs = [r for r in valid_score_rows if int(r["horizon"]) == h]
        per_horizon[h] = {
            "n": len(rs),
            "mean_host_similarity": _safe_mean(r["host_similarity"] for r in rs),
            "mean_donor_similarity": _safe_mean(r["donor_similarity"] for r in rs),
            "mean_hidden_identity_pull":
                _safe_mean(r["hidden_identity_pull"] for r in rs),
        }

    return {
        "stage": 3,
        "metrics_recorded": list(IDENTITY_METRICS),
        "n_pairs_attempted": n_pairs,
        "n_valid_swap_a": len(valid_a),
        "n_valid_swap_b": len(valid_b),
        "n_invalid_swap_a": n_pairs - len(valid_a),
        "n_invalid_swap_b": n_pairs - len(valid_b),
        "invalid_reasons_a": dict(invalid_reasons_a),
        "invalid_reasons_b": dict(invalid_reasons_b),
        "mean_projection_preservation_error_a":
            _safe_mean(p.projection_preservation_error_a for p in results),
        "mean_projection_preservation_error_b":
            _safe_mean(p.projection_preservation_error_b for p in results),
        "mean_match_distance":
            _safe_mean(p.match_distance for p in results),
        "mean_visible_similarity":
            _safe_mean(p.visible_similarity for p in results),
        "mean_hidden_distance":
            _safe_mean(p.hidden_distance for p in results),
        "n_score_rows_total": len(score_rows),
        "n_score_rows_valid": len(valid_score_rows),
        "mean_host_similarity_a":
            _safe_mean(r["host_similarity"] for r in rows_a),
        "mean_donor_similarity_a":
            _safe_mean(r["donor_similarity"] for r in rows_a),
        "mean_hidden_identity_pull_a":
            _safe_mean(r["hidden_identity_pull"] for r in rows_a),
        "mean_host_similarity_b":
            _safe_mean(r["host_similarity"] for r in rows_b),
        "mean_donor_similarity_b":
            _safe_mean(r["donor_similarity"] for r in rows_b),
        "mean_hidden_identity_pull_b":
            _safe_mean(r["hidden_identity_pull"] for r in rows_b),
        "per_horizon": per_horizon,
    }


def _md_row(cells): return "| " + " | ".join(str(c) for c in cells) + " |"


def write_summary_md(summary: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Follow-up Topic 2 — hidden identity swap (Stage 3)")
    lines.append("")
    lines.append(
        "Pairs ``(A, B)`` come from independent rollouts of the **same** "
        "rule with different seeds. The hybrid for "
        "``A_with_B_hidden`` is built by grafting B's hidden ``(z, w)`` "
        "fibres into A's mask cells, but only at locations where the "
        "per-cell projection matches (so the visible projection of the "
        "hybrid equals A's at every cell of A's mask). If the per-cell "
        "projection disagreement covers the whole mask, the swap is "
        "reported invalid and excluded from the identity-pull summary."
    )
    lines.append("")
    lines.append(
        "**This is a functional causal-identity test.** It is not a "
        "personal-identity or consciousness claim."
    )
    lines.append("")

    # Headline counts
    lines.append("## Pair counts")
    lines.append("")
    lines.append(_md_row(["direction", "attempted", "valid", "invalid"]))
    lines.append(_md_row(["---", "---:", "---:", "---:"]))
    lines.append(_md_row([
        "A_with_B_hidden",
        summary["n_pairs_attempted"],
        summary["n_valid_swap_a"],
        summary["n_invalid_swap_a"],
    ]))
    lines.append(_md_row([
        "B_with_A_hidden",
        summary["n_pairs_attempted"],
        summary["n_valid_swap_b"],
        summary["n_invalid_swap_b"],
    ]))
    lines.append("")

    # Invalid reasons (if any).
    for label, key in (("A_with_B_hidden", "invalid_reasons_a"),
                       ("B_with_A_hidden", "invalid_reasons_b")):
        reasons = summary.get(key, {})
        if not reasons:
            continue
        lines.append(f"### Invalid reasons — {label}")
        lines.append("")
        lines.append(_md_row(["count", "reason"]))
        lines.append(_md_row(["---:", "---"]))
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(_md_row([count, reason]))
        lines.append("")

    # Headline identity pull.
    lines.append("## Headline — host vs donor similarity (valid pairs only)")
    lines.append("")
    lines.append(_md_row([
        "direction", "mean host_sim", "mean donor_sim",
        "mean hidden_identity_pull",
    ]))
    lines.append(_md_row(["---", "---:", "---:", "---:"]))
    for direction, host_key, donor_key, pull_key in (
        ("A_with_B_hidden",
         "mean_host_similarity_a", "mean_donor_similarity_a",
         "mean_hidden_identity_pull_a"),
        ("B_with_A_hidden",
         "mean_host_similarity_b", "mean_donor_similarity_b",
         "mean_hidden_identity_pull_b"),
    ):
        host = summary.get(host_key)
        donor = summary.get(donor_key)
        pull = summary.get(pull_key)
        lines.append(_md_row([
            direction,
            f"{host:+.4f}" if host is not None else "—",
            f"{donor:+.4f}" if donor is not None else "—",
            f"{pull:+.4f}" if pull is not None else "—",
        ]))
    lines.append("")

    # Per-horizon table
    if summary.get("per_horizon"):
        lines.append("## Per-horizon means (both directions pooled, valid pairs)")
        lines.append("")
        lines.append(_md_row(["horizon", "n", "mean host_sim",
                               "mean donor_sim", "mean pull"]))
        lines.append(_md_row(["---", "---:", "---:", "---:", "---:"]))
        for h in sorted(summary["per_horizon"]):
            agg = summary["per_horizon"][h]
            lines.append(_md_row([
                h, agg["n"],
                f"{agg['mean_host_similarity']:+.4f}"
                    if agg['mean_host_similarity'] is not None else "—",
                f"{agg['mean_donor_similarity']:+.4f}"
                    if agg['mean_donor_similarity'] is not None else "—",
                f"{agg['mean_hidden_identity_pull']:+.4f}"
                    if agg['mean_hidden_identity_pull'] is not None else "—",
            ]))
        lines.append("")

    # Audit
    lines.append("## Audit")
    lines.append("")
    lines.append(_md_row([
        "metric", "value",
    ]))
    lines.append(_md_row(["---", "---:"]))
    for k in ("mean_projection_preservation_error_a",
             "mean_projection_preservation_error_b",
             "mean_match_distance", "mean_visible_similarity",
             "mean_hidden_distance"):
        v = summary.get(k)
        lines.append(_md_row([k, f"{v:.4g}" if v is not None else "—"]))
    lines.append("")

    # Pre-committed interpretation rules.
    lines.append("## Activated interpretation")
    lines.append("")
    if summary["n_valid_swap_a"] == 0 and summary["n_valid_swap_b"] == 0:
        lines.append(
            "* **No valid swaps under this projection / matching mode.** "
            "The current candidate population does not support clean "
            "identity-swap testing."
        )
    else:
        pull_a = summary.get("mean_hidden_identity_pull_a")
        pull_b = summary.get("mean_hidden_identity_pull_b")
        pulls = [p for p in (pull_a, pull_b) if p is not None]
        if pulls:
            mean_pull = sum(pulls) / len(pulls)
            if mean_pull > 0.02:
                lines.append(
                    f"* **Mean hidden_identity_pull = {mean_pull:+.4f} > 0** — "
                    "smoke suggests futures are pulled toward hidden donor "
                    "state more than visible host state."
                )
            elif mean_pull < -0.02:
                lines.append(
                    f"* **Mean hidden_identity_pull = {mean_pull:+.4f} < 0** — "
                    "smoke suggests visible host structure dominates future "
                    "trajectory under this swap."
                )
            else:
                lines.append(
                    f"* **Mean hidden_identity_pull = {mean_pull:+.4f} ~ 0** — "
                    "no clear identity pull in this smoke; signal may "
                    "depend on candidate match quality, projection, or "
                    "mechanism class."
                )
        if summary.get("mean_projection_preservation_error_a") and \
                summary["mean_projection_preservation_error_a"] > 1e-6:
            lines.append(
                "* mean_projection_preservation_error_a is non-zero; "
                "swap was not fully clean — identity-pull should not be "
                "interpreted for those pairs."
            )

    path.write_text("\n".join(lines), encoding="utf-8")
