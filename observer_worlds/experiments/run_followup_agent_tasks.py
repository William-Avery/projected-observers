"""Follow-up Topic 3 — agent-task experiment runner (Stage 4).

Discovers candidates, runs three minimal task evaluators (``repair``,
``foraging``, ``memory``), aggregates per-task scores, fits simple
HCE -> task_score and observer_score -> task_score regressions, and
writes the documented bundle.

Stage 4 is smoke-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from observer_worlds.analysis.agent_task_plots import write_all_plots
from observer_worlds.analysis.agent_task_stats import (
    aggregate_agent_task_results, write_summary_md,
)
from observer_worlds.environments import KNOWN_TASKS
from observer_worlds.experiments._followup_agent_tasks import (
    TaskTrial, run_tasks_for_candidate,
)
from observer_worlds.experiments._followup_identity_swap import (
    discover_candidates_for_cell,
)
from observer_worlds.search.rules import FractionalRule


REPO = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Follow-up Topic 3: agent-task experiment.",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--backend", type=str, default=None,
                   choices=["numpy", "numba", "cupy", "cuda-batched"])
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--out-root", type=Path, default=REPO / "outputs")
    p.add_argument("--label", type=str, default="followup_agent_tasks")
    p.add_argument("--n-rules-per-source", type=int, default=None)
    p.add_argument("--test-seeds", type=int, nargs="+", default=None)
    p.add_argument("--tasks", nargs="+", default=None,
                   choices=tuple(KNOWN_TASKS.keys()))
    p.add_argument("--projection", type=str, default=None,
                   help="Single projection used for the trial rollouts.")
    p.add_argument("--replicates", type=int, default=None)
    p.add_argument("--grid", type=int, nargs=4, default=None,
                   metavar=("NX", "NY", "NZ", "NW"))
    p.add_argument("--rules-json", type=Path,
                   default=REPO / "release" / "rules" / "m7_top_hce_rules.json")
    p.add_argument("--profile", action="store_true")
    return p


def _full_defaults() -> dict:
    return {
        "n_workers": max(1, (os.cpu_count() or 2) - 2),
        "backend": "numpy",
        "max_candidates": 50,
        "timesteps": 500,
        "horizons": [5, 10, 20, 40, 80],
        "n_rules_per_source": 5,
        "test_seeds": list(range(6000, 6020)),
        "tasks": list(KNOWN_TASKS),
        "projection": "mean_threshold",
        "replicates": 3,
        "grid": [64, 64, 8, 8],
    }


def _smoke_defaults() -> dict:
    return {
        "n_rules_per_source": 1,
        "test_seeds": [6000, 6001, 6002, 6003, 6004],
        "timesteps": 100,
        "max_candidates": 10,
        "tasks": ["repair", "memory"],   # foraging is opt-in for smoke
        "horizons": [5, 10],
        "replicates": 1,
        "grid": [16, 16, 4, 4],
    }


def _resolve_config(args: argparse.Namespace) -> dict:
    cfg = dict(_full_defaults())
    if args.quick:
        cfg.update(_smoke_defaults())
    for key in ("n_workers", "backend", "max_candidates", "timesteps",
                "horizons", "n_rules_per_source", "test_seeds",
                "tasks", "projection", "replicates", "grid"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = list(v) if isinstance(v, list) else v
    cfg["rules_json"] = str(args.rules_json)
    return cfg


def _make_out_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_root / f"{args.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def _build_frozen_manifest(cfg: dict) -> dict:
    def _safe(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(REPO), timeout=10)
            return r.stdout.strip()
        except Exception:
            return ""
    return {
        "stage": 4,
        "experiment": "followup_agent_tasks",
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
        "git": {
            "commit": _safe(["git", "rev-parse", "HEAD"]),
            "branch": _safe(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_safe(["git", "status", "--porcelain"])),
        },
        "config": cfg,
        "platform": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
    }


def _load_rules(path: Path, n: int) -> list[FractionalRule]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FractionalRule.from_dict(r) for r in raw[:int(n)]]


def _write_trial_csv(trials: list[TaskTrial], path: Path) -> None:
    fields = [
        "trial_id", "rule_id", "rule_source", "seed", "candidate_id",
        "track_id", "task_name", "horizon", "projection_name",
        "survived", "survival_time",
        "hce", "observer_score",
        "repair_score", "resource_contact_score",
        "movement_toward_resource", "cue_memory_score",
        "task_score",
        "hidden_intervention_task_delta",
        "visible_intervention_task_delta",
        "mechanism_class",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trials:
            w.writerow({k: getattr(t, k) for k in fields})


def _write_simple_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = _resolve_config(args)
    out = _make_out_dir(args)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "frozen_manifest.json").write_text(
        json.dumps(_build_frozen_manifest(cfg), indent=2), encoding="utf-8",
    )
    rules = _load_rules(args.rules_json, cfg["n_rules_per_source"])
    rule_records = []
    for i, r in enumerate(rules):
        rid = f"M7_HCE_optimized_rank{i+1:02d}"
        rule_records.append({"rule": r, "rule_id": rid,
                              "rule_source": "M7_HCE_optimized"})

    print("=" * 72)
    print("Follow-up Topic 3: agent-task environments — Stage 4")
    print("=" * 72)
    print(f"  out         = {out}")
    print(f"  backend     = {cfg['backend']}")
    print(f"  n_workers   = {cfg['n_workers']}")
    print(f"  rules       = {len(rule_records)}")
    print(f"  seeds       = {len(cfg['test_seeds'])} ({cfg['test_seeds'][0]}..{cfg['test_seeds'][-1]})")
    print(f"  timesteps   = {cfg['timesteps']}")
    print(f"  grid        = {cfg['grid']}")
    print(f"  projection  = {cfg['projection']}")
    print(f"  tasks       = {cfg['tasks']}")
    print(f"  horizons    = {cfg['horizons']}")
    print(f"  replicates  = {cfg['replicates']}")

    t_total = time.time()

    discovery_tasks = [
        (rec, seed) for rec in rule_records for seed in cfg["test_seeds"]
    ]
    def _discover(rec, seed):
        return discover_candidates_for_cell(
            rule_bs=rec["rule"].to_bsrule(),
            rule_id=rec["rule_id"], rule_source=rec["rule_source"],
            seed=int(seed),
            grid_shape=tuple(cfg["grid"]),
            timesteps=int(cfg["timesteps"]),
            backend=cfg["backend"],
            projection_name=cfg["projection"],
            max_candidates=int(cfg["max_candidates"]),
            initial_density=float(rec["rule"].initial_density),
        )
    if int(cfg["n_workers"]) > 1 and len(discovery_tasks) > 1:
        per_cell_candidates = Parallel(
            n_jobs=int(cfg["n_workers"]), verbose=0, backend="loky",
        )(delayed(_discover)(rec, seed) for rec, seed in discovery_tasks)
    else:
        per_cell_candidates = [_discover(rec, seed)
                               for rec, seed in discovery_tasks]
    all_candidates = [c for cell in per_cell_candidates for c in cell]
    print(f"\nDiscovered {len(all_candidates)} candidates across "
          f"{len(per_cell_candidates)} cells.")

    rule_bs_by_id = {rec["rule_id"]: rec["rule"].to_bsrule()
                      for rec in rule_records}
    def _measure(idx_cic):
        idx, cic = idx_cic
        rng = np.random.default_rng(
            (int(cic.seed) ^ (idx * 7919) ^ 0xA9E47) & 0xFFFFFFFF
        )
        return run_tasks_for_candidate(
            cic=cic, rule_bs=rule_bs_by_id[cic.rule_id],
            projection_name=cfg["projection"],
            horizons=tuple(int(h) for h in cfg["horizons"]),
            backend=cfg["backend"],
            tasks=cfg["tasks"],
            rng=rng,
        )

    if int(cfg["n_workers"]) > 1 and len(all_candidates) > 1:
        per_candidate_trials = Parallel(
            n_jobs=int(cfg["n_workers"]), verbose=0, backend="loky",
        )(delayed(_measure)((i, c)) for i, c in enumerate(all_candidates))
    else:
        per_candidate_trials = [_measure((i, c))
                                 for i, c in enumerate(all_candidates)]

    trials: list[TaskTrial] = []
    next_id = 0
    for ts in per_candidate_trials:
        for t in ts:
            t.trial_id = next_id
            next_id += 1
            trials.append(t)

    print(f"Computed {len(trials)} task trials.")

    # Write CSVs.
    _write_trial_csv(trials, out / "task_trials.csv")

    # task_scores.csv: trial_id + task + horizon + task_score + hce + observer
    task_score_rows = [{
        "trial_id": t.trial_id, "task_name": t.task_name,
        "horizon": t.horizon, "rule_id": t.rule_id, "seed": t.seed,
        "candidate_id": t.candidate_id,
        "task_score": t.task_score, "hce": t.hce,
        "observer_score": t.observer_score, "survived": t.survived,
    } for t in trials]
    _write_simple_csv(task_score_rows, out / "task_scores.csv", fields=[
        "trial_id", "task_name", "horizon", "rule_id", "seed",
        "candidate_id", "task_score", "hce", "observer_score", "survived",
    ])

    # hce_task_joined.csv: per (candidate, task) row with mean task score
    join_rows = []
    by_cand_task: dict[tuple, list[TaskTrial]] = {}
    for t in trials:
        key = (t.rule_id, t.seed, t.candidate_id, t.task_name)
        by_cand_task.setdefault(key, []).append(t)
    for (rule_id, seed, cid, task_name), ts in by_cand_task.items():
        scores = [t.task_score for t in ts if t.task_score is not None]
        if not scores:
            continue
        join_rows.append({
            "rule_id": rule_id, "seed": int(seed), "candidate_id": cid,
            "task_name": task_name,
            "n_horizons": len(scores),
            "mean_task_score": float(np.mean(scores)),
            "max_task_score": float(np.max(scores)),
            "hce": ts[0].hce, "observer_score": ts[0].observer_score,
            "any_survived": any(t.survived for t in ts),
        })
    _write_simple_csv(join_rows, out / "hce_task_joined.csv", fields=[
        "rule_id", "seed", "candidate_id", "task_name",
        "n_horizons", "mean_task_score", "max_task_score",
        "hce", "observer_score", "any_survived",
    ])

    summary = aggregate_agent_task_results(trials, join_rows)
    summary["wall_time_seconds"] = float(time.time() - t_total)
    summary["n_trials"] = len(trials)
    summary["n_candidates"] = len(all_candidates)
    summary["projection"] = cfg["projection"]
    summary["tasks"] = list(cfg["tasks"])

    (out / "stats_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o:
                   float(o) if isinstance(o, np.floating) else
                   (int(o) if isinstance(o, np.integer) else
                    (o.tolist() if isinstance(o, np.ndarray) else str(o)))),
        encoding="utf-8",
    )
    (out / "regression_results.json").write_text(
        json.dumps(summary.get("regressions", {}), indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating)
                                       else (int(o) if isinstance(o, np.integer)
                                              else str(o))),
        encoding="utf-8",
    )
    write_summary_md(summary, out / "summary.md")
    try:
        write_all_plots(summary, trials, out / "plots")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] plot generation failed: {e!r}")

    print(f"\nDone in {summary['wall_time_seconds']:.1f}s. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
