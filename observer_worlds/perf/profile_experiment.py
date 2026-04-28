"""Run a follow-up experiment under :class:`observer_worlds.perf.Profiler`.

Stage-1 skeleton. Does not yet actually launch the experiment runner —
that requires the runner-side smoke commands which only stub their
behavior at this stage. What it does today:

* Accepts the standard CLI surface (``--experiment``, ``--quick``,
  ``--n-workers``, ``--backend``).
* Resolves the experiment name to a runnable module path.
* Prints what it would launch and writes a stub profiler JSON.

Stage 2+ will replace the stub with a ``subprocess.run`` of the
experiment runner, with phase timing collected via the runner's own
``--profile`` flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from observer_worlds.perf.profiler import Profiler


# Map ``--experiment`` short names to runner module paths.
EXPERIMENT_RUNNERS = {
    "projection_robustness":
        "observer_worlds.experiments.run_followup_projection_robustness",
    "hidden_identity_swap":
        "observer_worlds.experiments.run_followup_hidden_identity_swap",
    "agent_tasks":
        "observer_worlds.experiments.run_followup_agent_tasks",
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
    )
    p.add_argument("--experiment", required=True,
                   choices=sorted(EXPERIMENT_RUNNERS),
                   help="Which follow-up experiment to profile.")
    p.add_argument("--quick", action="store_true",
                   help="Pass --quick to the runner (smoke run).")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--backend", type=str, default=None,
                   choices=["numpy", "numba", "cupy", "cuda-batched"])
    p.add_argument("--out-root", type=Path, default=None,
                   help="Where to write the perf JSON. Default: alongside "
                        "the runner's outputs/.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runner = EXPERIMENT_RUNNERS[args.experiment]

    cmd = [sys.executable, "-m", runner]
    if args.quick:
        cmd.append("--quick")
    if args.n_workers is not None:
        cmd += ["--n-workers", str(args.n_workers)]
    if args.backend is not None:
        cmd += ["--backend", args.backend]

    prof = Profiler(label=f"{args.experiment}_profile")
    print(f"profile_experiment skeleton")
    print(f"  experiment: {args.experiment}")
    print(f"  runner    : {runner}")
    print(f"  command   : {' '.join(cmd)}")
    print(f"  status    : Stage 1 — not yet executing the runner.")
    print(f"               Stage 2+ will subprocess-run and time phases.")

    out_root = args.out_root or Path("outputs/perf")
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"perf_{args.experiment}_skeleton.json"
    prof.write_json(out_path)
    print(f"  perf json : {out_path}")

    # Also write the planned command for transparency.
    plan_path = out_root / f"plan_{args.experiment}.json"
    plan_path.write_text(json.dumps({
        "experiment": args.experiment,
        "runner": runner,
        "command": cmd,
        "stage": 1,
        "note": "skeleton; runner not yet executed",
    }, indent=2), encoding="utf-8")
    print(f"  plan json : {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
