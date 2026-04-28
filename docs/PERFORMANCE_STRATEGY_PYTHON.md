# Performance strategy for follow-up research (Python only)

This document codifies how we make follow-up experiments fast enough to
be useful, **without leaving Python**. It is the contract between the
research roadmap and the implementation: every follow-up experiment is
expected to follow these rules unless it explicitly justifies an
exception.

## Hard rules

1. **Keep Python.** Do not introduce Rust, C++, or any non-Python
   compiled component. The framework is a research orchestration layer;
   speed comes from layering, batching, caching, and the three
   acceleration backends below — not from a rewrite.
2. **Backend interface.** Every long-running primitive (CA step,
   projection, candidate evaluation) accepts a backend selector:
   ```
   backend ∈ {"numpy", "numba", "cupy"}
   ```
   The default for a fresh smoke run is `numpy`. The default for a
   production run is `numba` on CPU and `cupy` (or its `cuda-batched`
   variant) on GPU when CUDA is available.
3. **No CPU↔GPU copy every timestep.** When CuPy is in use, arrays
   stay on the device across many timesteps. The CPU side only
   touches device arrays at projection sampling, candidate extraction,
   or end-of-rollout reduction.
4. **Process-level parallelism for independent work.** Independent
   `(rule, seed, projection, perturbation, candidate)` work units run
   in `joblib` / `multiprocessing` process pools. Within a worker we
   stay sequential to avoid double-parallel oversubscription.
5. **Numba `cache=True` for stable kernels.** A kernel that does not
   change shape/dtype on hot paths must be Numba-compiled with
   `cache=True` so the JIT pays its cost once and reuses on subsequent
   runs.
6. **Vectorize before parallelizing.** Inner loops should be
   array-shaped before they are wrapped in joblib. A vectorized loop
   on one core is usually better than a Python-loop loop on N cores.
7. **No intermediate writes inside hot inner loops.** Plotting and CSV
   writes happen at the end of an experiment, not per (rule, seed)
   pair. Long-form metrics are accumulated in memory or appended in
   bulk after the sweep.

## Caching and reuse rules

8. **Frozen manifests.** Every long experiment writes
   `frozen_manifest.json` recording the git commit, dirty state, and
   command line. Without this, run reproducibility breaks the moment
   the repo moves.
9. **Reusable candidate snapshots.** Candidate-detection state is
   saved per `(rule, seed)` so downstream stages
   (projection-robustness, identity swap, agent tasks) can replay
   from a saved snapshot rather than re-running 4D simulation.
10. **Reusable state checkpoints.** When multiple projections are
    evaluated against the same 4D substrate (Topic 1), the substrate
    runs once; its state stream (or sampled checkpoints) is reused for
    every projection.
11. **Cache projected frames** per `(rule, seed, projection)` — the
    expensive part is candidate tracking on top of a projected frame
    stream, not the projection itself, but caching avoids recomputing
    the projection during analysis.
12. **No-simulation analyses are first-class.** Where possible, follow-
    ups operate on existing artifacts (the M8E artifacts at
    `outputs/m8e_cross_source_20260428T150640Z/` are the reference).
    M8F and M8G already do this; future analyses should as well.

## Profiling

13. **Profile before optimizing unknown bottlenecks.** Premature
    optimization is wasted work. Use `observer_worlds.perf.profiler`
    to measure first.
14. **Phase breakdown.** Every long experiment, when profiled, reports
    per-phase wall time:
    * substrate simulation (4D CA stepping)
    * projection
    * candidate detection / tracking
    * intervention rollouts
    * stats / aggregation
    * plotting / IO
    Profiling builds a budget that the experiment is then expected to
    stay within for follow-up production runs.
15. **What the profiler measures.** Wall time per phase, total
    candidates / rollouts processed, candidates per second, rollouts
    per second, optional process RSS, optional GPU memory if CuPy is
    active. The profiler is best-effort — psutil and cupy are imported
    lazily and treated as optional.

## Standard CLI surface

Every follow-up experiment runner exposes at least:

```
--quick                Quick smoke run with reduced defaults.
--n-workers N          Process-pool size; default os.cpu_count() - 2.
--backend BACKEND      One of numpy | numba | cupy | cuda-batched.
--max-candidates N     Cap on per-cell candidate count.
--timesteps N          Rollout length.
--horizons H ...       Horizons to evaluate (multi-arg).
--out-root PATH        Where to write outputs/.
--label LABEL          Run label folded into the output dir name.
```

Optional but encouraged:

```
--n-rules-per-source N
--test-seeds S ...
--hce-replicates N
--profile              Wrap the run with the M-perf profiler.
```

## Benchmark harness

Run the production-time profile as:

```bash
python -m observer_worlds.perf.profile_experiment \
    --experiment projection_robustness \
    --quick
```

This:
1. Resolves the experiment runner module.
2. Wraps it in a profiler context.
3. Executes the runner as a subprocess so worker pools start clean.
4. Writes a `perf_<experiment>_<timestamp>.json` next to the run's
   output directory containing the per-phase wall-time breakdown and
   `candidates/sec`, `rollouts/sec`, `timesteps/sec`.

## Hardware reference

The captured production-machine baseline is:

* Intel i9-14900K (32 logical threads)
* 64 GiB DDR5
* RTX 3080 Ti, 12 GiB VRAM, NVIDIA driver 595.79, CUDA runtime 12.9
* Windows 11, Python 3.12.10, numpy 2.4.4, numba 0.65.1, cupy 14.0.1

Reference baseline: `m8_m7b_class_numpy` in
`tests/perf/baselines.json` — M8 mechanism discovery at production
scale (5 × 20 cells × T = 500 × 64 × 64 × 8 × 8) takes ~1843 s on
numpy with 30 workers. Cuda-batched on the same config has not yet
been captured.

## What does not belong here

* New language runtimes (Rust, C++, Cython that compiles outside the
  Python build).
* New parallel frameworks beyond what the project already uses
  (joblib, multiprocessing, pytest-xdist, CuPy). Adding Dask or Ray
  is a separate proposal that is out of scope for the follow-up
  topics.
* Heroic single-kernel optimizations on un-profiled paths. If a
  bottleneck has not been measured, do not pre-optimize it.

## Compliance

Every follow-up experiment runner committed under
`observer_worlds/experiments/run_followup_*.py` must comply with the
hard rules above. The Stage 1 skeletons are explicit about which rules
apply where. CI does not yet enforce all of them; reviewer judgement
plus the M8 audit-style discipline is the current backstop.
