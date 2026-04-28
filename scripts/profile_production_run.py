"""Thin wrapper around ``observer_worlds.perf.profile_experiment``.

Lets users invoke the profiler harness as
``python scripts/profile_production_run.py ...`` if they prefer that
form. The canonical invocation is still
``python -m observer_worlds.perf.profile_experiment ...``.

Stage-1 skeleton; the harness it wraps does not yet execute the
runners (see ``observer_worlds/perf/profile_experiment.py``).
"""
from __future__ import annotations

import sys

from observer_worlds.perf.profile_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
