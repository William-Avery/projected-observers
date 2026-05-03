# GPU backend plan (Stage G1+)

## Why GPU is being added

The Stage 6 fresh-seed replication program took ≈ 10.7 wall-clock
hours on numpy backend / 30 CPU workers (Stage 6C alone was 6.48 h).
A profile of the production runners shows the dominant cost is in
**batched 4D Moore-r1 CA rollouts** (and the projection / divergence
passes computed on the rollout output): every candidate × replicate ×
perturbation-condition × horizon trajectory is a small-grid CA stepped
500–580 times. There is plenty of independent parallelism in the
condition / replicate / candidate axes — exactly the shape a GPU
likes.

The Stage G program adds a CuPy/CUDA backend so this work can be
batched on a single GPU instead of scattered across 30 CPU workers.
The aim is **scientific equivalence first, throughput second**: every
GPU primitive must agree with the CPU reference within a documented
tolerance before any production claim is made on GPU output.

## What is accelerated

The new ``observer_worlds.backends`` layer (Stage G1) exposes batched
GPU primitives in the canonical shape ``(B, Nx, Ny, Nz, Nw)``:

* **``step_4d_batch``** — batched 4D Moore-r1 CA step. Reuses the
  existing batched RawKernel from ``observer_worlds.worlds.ca4d_batch``.
* **``project_batch``** — all six Stage-5 projections
  (``mean_threshold``, ``sum_threshold``, ``max_projection``,
  ``parity_projection``, ``random_linear_projection``,
  ``multi_channel_projection``) run on-device.
* **``apply_hidden_perturbations_batch``** —
  count-preserving deterministic swap (preserves all four
  count-based projections exactly).
* **``apply_visible_perturbations_batch``** — deterministic
  count-changing flip used as the visible-perturbation control.
* **``compute_candidate_local_l1_batch``** — candidate-local L1
  divergence on projected frames.
* **``rollout_hce_batch``** — full original-vs-perturbed rollout that
  steps both states in lockstep, projects at the requested horizons,
  and returns a compact ``(B, len(horizons))`` metric. **States stay
  on device for the entire loop.** Only the compact metric crosses
  the device boundary.

## What remains CPU

The CPU keeps everything that is *not* batched rollout-heavy work:

* All orchestration: argument parsing, run-dir layout, frozen
  manifest construction, work-list building.
* Candidate discovery from a single full rollout (small-grid +
  tracking + connected-component IO; not a kernel-bound workload).
* All RNG-driven strategies in
  ``observer_worlds.projection.invisible_perturbations`` and
  ``visible_perturbations``. Production runners continue to use these
  unchanged; the deterministic GPU primitives above are for the
  benchmark + equivalence layer.
* CSV / JSON / Markdown writes; bootstrap CIs;
  Cliff's δ; Pearson regressions; per-source / per-horizon
  posthoc analyzers.
* All plot generation.
* All summary interpretation logic.

## Why we do not run many CPU workers against one GPU

A single GPU is a contended resource. If 30 joblib workers each try
to stage their own state arrays onto the same device, three things
go wrong:

1. **VRAM thrash.** Each worker's CuPy memory pool is process-local;
   30 pools × per-worker allocations exceed 12 GB instantly.
2. **Kernel-launch serialization.** GPU work from N processes
   serializes through the driver, so most processes stall.
3. **Tiny batches.** Each worker holds a slice of the work-list, so
   the per-launch ``B`` shrinks to single-digit values — exactly the
   regime where GPU is slowest relative to CPU.

The G2 design: **one GPU controller process** that owns the device,
holds all state arrays on-device, and consumes the same flat work-list
the CPU runner produces. CPU helpers (orchestration, posthoc) stay
parallel; GPU work is single-process.

## Supported backend flags (G1)

The new flags below are wired through the Stage G1 benchmark harness
and will be added to GPU-aware runners in subsequent stages. They do
not modify any Stage 5 / Stage 6 production runner — those continue
to use the legacy ``--backend numpy|numba|cuda`` flag of
``observer_worlds.worlds.ca4d.CA4D`` unchanged.

| flag | purpose |
|---|---|
| ``--backend numpy\|cupy`` | pick the new backend layer's implementation |
| ``--gpu-batch-size`` | override the planner's batch size |
| ``--gpu-memory-target-gb`` | VRAM cap for the planner (default 9.5 on a 12 GB card) |
| ``--gpu-device`` | CUDA device id (default 0) |
| ``--gpu-smoke-only`` | run only the smallest configured batch (CI-safe) |

## Memory strategy for 12 GB GPU

The planner ``estimate_max_safe_batch_size`` returns the largest
batch ``B`` such that

```
B × (n_perturbation_conditions × bytes_per_state
     + n_projection_frames_per_state × bytes_per_projection_frame)
   ≤ target_gb × safety_factor (default 0.6) × GiB
```

For the standard production grid 64 × 64 × 8 × 8 (uint8, ~256 KB per
state), with 5 perturbation conditions (original / hidden / far /
sham / visible) and 8 horizons, the planner returns ≈ 4500 — well
above any reasonable production batch size. Realistic batches are
expected to be limited by per-kernel scratch and projection
intermediates, not by raw state-array size.

The ``evolve_chunked`` helper in ``observer_worlds.worlds.ca4d_batch``
already implements OOM-retry-with-halved-chunk for the legacy CUDA
path. The G2 GPU-aware HCE runner will use the same pattern around
the new ``rollout_hce_batch``.

## Benchmark instructions

```bash
# Full numpy-vs-cupy sweep at production grid:
PYTHONIOENCODING=utf-8 python -m observer_worlds.perf.benchmark_gpu_backend \
    --grid 64 64 8 8 \
    --batch-sizes 16 32 64 128 256 \
    --timesteps 100 \
    --backend both \
    --gpu-memory-target-gb 9.5

# CuPy-only smoke (CI-safe):
python -m observer_worlds.perf.benchmark_gpu_backend \
    --backend cupy --gpu-smoke-only --batch-sizes 32
```

Output dir: ``outputs/gpu_benchmark_<timestamp>/`` containing
``config.json``, ``benchmark_results.csv``, ``summary.md``.

## CPU/GPU equivalence policy

* **Binary projections** (``mean_threshold``, ``sum_threshold``,
  ``max_projection``, ``parity_projection``, ``multi_channel``):
  the GPU output must be **bit-identical** to the CPU output on tiny
  deterministic inputs. Tested in
  ``tests/test_gpu_backend.py`` via ``np.testing.assert_array_equal``.
* **Continuous projection** (``random_linear_projection``): float32
  agreement at ``rtol=1e-5, atol=1e-5``. Reduction order may
  legitimately differ between numpy's accumulator and CuPy's
  warp-level sum.
* **Step ``step_4d_batch``**: bit-identical multi-step agreement on
  a tiny deterministic state. Tested for 1 step and 5 steps.
* **Hidden / visible perturbations**: deterministic in this layer
  (no RNG), so bit-identical agreement is required.
* **Compact HCE metric**: bit-identical for binary projections;
  ``rtol=1e-4, atol=1e-3`` for continuous projections.
* **Memory planner**: smoke-tested for sane order-of-magnitude (≥ 1,
  monotone vs target_gb).

The full equivalence battery runs in
``tests/test_gpu_backend.py`` and skips cleanly if cupy is missing.
On a CPU-only machine, the CPU test suite (currently 452 passed)
still passes without cupy as a hard dependency. **GPU production
results are not scientific until per-primitive equivalence has
passed.**

## Warning

Stage G1 delivers backend + benchmark + equivalence only. **No
Stage 5 / Stage 6 production claim has been retroactively re-derived
on GPU output.** Any future GPU-derived production result must:

1. Pass the per-primitive equivalence battery in
   ``tests/test_gpu_backend.py`` for the full Stage-6 grid + horizons.
2. Reproduce the CPU baseline numbers for at least one Stage-5
   small-scale audit (e.g. preflight Stage 6A) within the documented
   tolerance.
3. Be flagged in the run's ``config.json`` as ``backend: cupy`` so
   downstream readers can audit the backend used.

Until those three conditions hold, GPU is for benchmarking and
exploration only — not for production claims.

## Stage status (G1 → G4)

| stage | what shipped | result |
|---|---|---|
| **G1** | `observer_worlds.backends` (numpy + cupy primitives), benchmark harness, 19 equivalence tests | RTX 3080 Ti benchmark: 500-664× step-loop speedup at B≥32; all binary projections bit-identical; random_linear within 6e-7 |
| **G2** | GPU-aware projection-robustness runner; preflight equivalence audit | preflight 240/240 candidates matched, max abs HCE delta 6.4e-7 |
| **G2B** | production-grid equivalence audit (T=500, max_h=80) | 870/870 candidates matched bit-identical for binary; random_linear max 5.4e-7 |
| **G3** | parallel CPU discovery via ProcessPoolExecutor + as_completed; full Stage 6C-equivalent GPU run | scientifically bit-identical to Stage 6C (26 139/26 139 candidates; posthoc verdict 17/18 + 16/18 matches); end-to-end wall **6.63 h vs Stage 6C CPU 6.48 h** — neutral |

**G3 finding.** GPU rollout itself is solved (499 s vs ≈ 17 500 s of
equivalent CPU rollout work — 35× phase speedup). The new bottleneck is
the CPU candidate-discovery half (substrate + projection-stream + tracker
+ perturbation construction, 23 271 s = 97.5% of G3 wall). The runner is
capped at ≈ 8 effective workers on Windows because joblib loky and
`concurrent.futures.ProcessPoolExecutor` both crash intermittently with
Windows access violations in `numpy._sum` (inside `tracking._iou`) and
`projection.invisible_perturbations._weight_canceling_pair_swap` once
worker count exceeds ~8 — the same fault pattern observed in Stage 6D
first attempt on the CPU production runner.

**Next experiment (G4-Linux).** Test whether the worker-count ceiling is
OS-specific by running the same G3 binary on a Linux host with
n_workers=30. See `docs/G4_LINUX_GPU_RUNBOOK.md` for the full
runbook (environment setup, three sequenced runs, equivalence check,
reporting template). If Linux is clean at n=30, end-to-end wall is
expected to drop to ≈ 2 h (≈ 3× speedup vs Stage 6C CPU baseline) without
any code change. If Linux also hits the ceiling, the GPU-discovery
refactor (G5: port per-frame projection / tracker hot loops to GPU) is
the next move.
