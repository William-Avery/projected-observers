# Stage G4-Linux runbook

## What this run is, and isn't

Stage G3 validated that the CuPy-based GPU pipeline reproduces the
Stage 6C scientific output bit-exactly (binary projections) or within
the documented continuous-projection tolerance (random_linear). The
end-to-end wall time on Windows was **6.63 h vs the 6.48 h CPU
baseline — i.e., neutral**. The reason is that CPU candidate
discovery (substrate rollout + projection-stream + tracker +
perturbation construction) is the dominant cost (97.5% of the GPU
runner's wall) and is bottlenecked by a Windows-specific worker-count
ceiling: `joblib loky` and `concurrent.futures.ProcessPoolExecutor`
both die intermittently with **Windows access violations** in
`numpy._sum` (inside `tracking._iou`) and in
`projection.invisible_perturbations._weight_canceling_pair_swap` once
worker count exceeds ~8. The same fault was observed in Stage 6D
first attempt on the CPU production runner and is documented there.

This runbook is a Linux deployment plan to test the hypothesis that
the worker-count ceiling is **OS-specific**, not a code defect. If
30-worker CPU discovery is stable on Linux, the G3 GPU pipeline
should drop to ≈ 2 h end-to-end without any code change.

This runbook is documentation only. **Do not run from Windows.**
The repository commit + tag identify the exact code state to test.

## 1. Environment setup

### 1.1 Hardware

* One NVIDIA GPU with at least 8 GB VRAM (12 GB matches the dev box
  RTX 3080 Ti). The G3 production-grid run used 850 MB pool peak, so
  any modern NVIDIA card with ≥ 4 GB will fit.
* ≥ 16 cores recommended for the n_workers=30 stress test; ≥ 8 cores
  acceptable.
* ≥ 32 GB system RAM (workers each hold a substrate stream
  ≈ 125 MB; 30 × ~2 GB peak per-worker memory headroom = ≈ 60 GB).

### 1.2 OS

* Ubuntu 22.04 LTS or Debian 12 are the reference targets. Any glibc
  Linux that supports CUDA 12.x should work.

### 1.3 Drivers + CUDA

* NVIDIA driver supporting CUDA 12 (≥ 525.x). Verify:

      nvidia-smi

  Expect a populated table with the GPU listed and the driver version
  matching CUDA 12.x compatibility. If `nvidia-smi` is missing, install
  `nvidia-driver-535` (Ubuntu) or equivalent before continuing.

* The repo uses CuPy 14.0.1 against CUDA 12 via the
  ``nvidia-cuda-nvrtc-cu12`` pip wheel (no system CUDA toolkit
  required for compilation — the repo's RawKernels are JIT-compiled).

### 1.4 Python

* Python ≥ 3.10. The dev box ran 3.12.10. Linux should run the same
  minor version for closest behavioural match.

      python3.12 --version    # expect: Python 3.12.x

### 1.5 venv + dependencies

      git clone https://github.com/William-Avery/projected-observers.git
      cd projected-observers
      git checkout perf/gpu-backend
      git rev-parse HEAD       # expect: 4cd1cb9 or later on perf/gpu-backend
      git tag --list 'stage6-*' | head
      # expect: stage6-replication-complete

      python3.12 -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip
      pip install -e ".[dev,gpu]"

The `[gpu]` extra installs `cupy-cuda12x>=13.0` and
`nvidia-cuda-nvrtc-cu12`. Pip will pull `cupy-cuda12x` 14.x; if a
specific version is needed for reproducibility, pin to **14.0.1** to
match the dev-box config.

### 1.6 Verify CuPy sees the GPU

      python -c "
      import cupy as cp
      print('cupy', cp.__version__)
      print('device count', cp.cuda.runtime.getDeviceCount())
      props = cp.cuda.runtime.getDeviceProperties(0)
      print('GPU 0:', props['name'].decode())
      free, total = cp.cuda.Device(0).mem_info
      print(f'mem: {free / 1024**2:.0f} / {total / 1024**2:.0f} MB free')
      "

Expected:

      cupy 14.0.x
      device count 1
      GPU 0: NVIDIA GeForce RTX 3080 Ti     # or whatever card
      mem: 11000 / 12287 MB free

If `device count 0` or an `ImportError`, fix the CUDA install before
proceeding — the G4 hypothesis can only be tested with a working GPU.

## 2. Repo verification

      pytest tests/ -q

Expected on Linux: **479 passed, 3 skipped** (matching the dev-box
result; `test_cupy_backend_unavailable_*` skips when cupy *is*
installed). If you see `cupy not available` skips elsewhere, the GPU
test path will be silently skipped and G4 is moot — fix the venv
first.

      pytest tests/test_gpu_backend.py tests/test_projection_robustness_gpu.py -q

Expected: **27 passed, 2 skipped**. The G1 + G2 + G3 GPU equivalence
suite. If any of the binary-projection equivalence tests *fail* on
Linux, the GPU backend output is no longer bit-identical to numpy on
this host and the run should not proceed.

## 3. The three Linux runs

All commands below assume the venv is active and the working directory
is the repo root.

### 3.1 Backend tests (≈ 1 minute)

      PYTHONIOENCODING=utf-8 python -m pytest \
          tests/test_gpu_backend.py \
          tests/test_projection_robustness_gpu.py -q

Pass criterion: 27 passed, 2 skipped.

### 3.2 Linux discovery stress test (≈ 10–15 minutes)

This is the **load-bearing test**. Same medium smoke as G3 (1 rule per
source × 3 seeds × T=500 × max_candidates=20 × hce_replicates=3 × 6
projections), but with **`--n-workers 30`** — the count that reliably
crashes within seconds on Windows.

      PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_projection_robustness_gpu \
          --rules-from release/rules/m7_top_hce_rules.json \
          --m4c-rules release/rules/m4c_evolve_leaderboard.json \
          --m4a-rules release/rules/m4a_search_leaderboard.json \
          --n-rules-per-source 1 \
          --seeds 7000..7002 \
          --timesteps 500 \
          --grid 64 64 8 8 \
          --max-candidates 20 \
          --hce-replicates 3 \
          --horizons 1 2 3 5 10 20 40 80 \
          --projections mean_threshold sum_threshold max_projection \
                        parity_projection random_linear_projection \
                        multi_channel_projection \
          --backend cupy \
          --n-workers 30 \
          --gpu-batch-size 64 \
          --gpu-memory-target-gb 9.5 \
          --label g4_linux_smoke_n30 \
          --profile

Pass criteria:

* Returncode 0.
* `[discovery] pool attempt 1 with workers=30 for 9 pending cells.`
  appears once and **no** `BrokenProcessPool`,
  `[discovery] pool died`, or `[discovery] retry` log lines after it.
* `cpu_discovery_workers = 30` in the printed banner.
* `discovery_stats.n_pool_restarts == 1` in
  `outputs/g4_linux_smoke_n30_*/stats_summary.json`'s
  `g3_perf` block (one attempt, no restarts).
* Bit-identical output to G2B / G3 — verify with the comparison
  snippet in §3.4.

If this run shows even one pool restart, the Linux stack also has the
worker-count ceiling and §3.3 is not worth running yet (drop n_workers
to 8 or rebuild the GPU-discovery refactor).

If this run is clean, the path forward is the full run in §3.3.

### 3.3 Full G3-equivalent Linux run (estimated ≈ 1.5–2.5 h)

Identical to the G3 production run but on Linux at `--n-workers 30`.

      PYTHONIOENCODING=utf-8 python -m observer_worlds.experiments.run_followup_projection_robustness_gpu \
          --rules-from release/rules/m7_top_hce_rules.json \
          --m4c-rules release/rules/m4c_evolve_leaderboard.json \
          --m4a-rules release/rules/m4a_search_leaderboard.json \
          --n-rules-per-source 5 \
          --seeds 7000..7019 \
          --timesteps 500 \
          --grid 64 64 8 8 \
          --max-candidates 20 \
          --hce-replicates 3 \
          --horizons 1 2 3 5 10 20 40 80 \
          --projections mean_threshold sum_threshold max_projection \
                        parity_projection random_linear_projection \
                        multi_channel_projection \
          --backend cupy \
          --n-workers 30 \
          --gpu-batch-size 64 \
          --gpu-memory-target-gb 9.5 \
          --label g4_linux_projection_robustness_gpu_stage6c_equivalent \
          --profile

Pass criteria:

* Returncode 0.
* All Stage 6C scientific conclusions preserved (verify in §3.4).
* `total_wall_s < 4 h` (acceptable); **< 3 h preferred**; ideal
  ≈ 2 h based on the linear extrapolation 23 271 s / 8 * 30 → 6 200 s
  + GPU 500 s ≈ 6 700 s ≈ 1.86 h.
* `gpu_rollout_wall_s < 15 min` (already 8.3 min on Windows; should
  be the same or slightly faster on Linux due to no Windows scheduler
  jitter).
* `cpu_discovery_wall_s` substantially below 23 271 s. If the linear
  extrapolation holds, expect 6 000–8 000 s.

Output dir:
`outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_<ts>/`.

### 3.4 Equivalence check vs Stage 6C CPU baseline

After §3.3 finishes, run the row-by-row comparison and the posthoc
analyzer:

      G4_RUN=outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_*/
      S6C=outputs/stage6c_projection_robustness_seed7000_20260501T035342Z

      # Row-by-row equivalence (binary projections must be bit-identical;
      # random_linear must be within ~1e-6).
      python - <<'PY'
      import csv, math, glob
      g4 = sorted(glob.glob("outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_*/"))[-1]
      s6c = "outputs/stage6c_projection_robustness_seed7000_20260501T035342Z/"
      def idx(p):
          rows = list(csv.DictReader(open(p + "candidate_metrics.csv")))
          return {(r['rule_id'], int(r['seed']), r['projection'], int(r['candidate_id'])): r for r in rows}
      a = idx(s6c); b = idx(g4)
      print(f"6C rows: {len(a)}, G4 rows: {len(b)}, matched: {len(set(a) & set(b))}")
      def f(x):
          if x in ('','None'): return None
          try: return float(x)
          except: return None
      hce_max = far_max = init_max = 0.0
      bin_match = 0; bin_n = 0
      for k in set(a) & set(b):
          for fn, store in (('HCE',1),('far_HCE',2),('initial_projection_delta',3)):
              va = f(a[k][fn]); vb = f(b[k][fn])
              if va is None or vb is None: continue
              if math.isnan(va) or math.isnan(vb): continue
              d = abs(va - vb)
              if fn == 'HCE': hce_max = max(hce_max, d)
              elif fn == 'far_HCE': far_max = max(far_max, d)
              else: init_max = max(init_max, d)
          if 'random_linear' not in k[2]:
              bin_n += 1
              if f(a[k]['HCE']) == f(b[k]['HCE']): bin_match += 1
      print(f"binary HCE bit-identical: {bin_match}/{bin_n}")
      print(f"max abs HCE delta:        {hce_max:.3e}")
      print(f"max abs far_HCE delta:    {far_max:.3e}")
      print(f"max abs init_proj delta:  {init_max:.3e}")
      PY

      # Headline scientific verdict.
      python -m observer_worlds.analysis.projection_robustness_posthoc \
          --run-dir "$(ls -td outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_*/ | head -1)"

Pass criteria:

* Matched candidate count = 26 139.
* Binary HCE bit-identical: 20 139 / 20 139.
* Max abs HCE delta: ≤ 1e-6 (continuous projection only).
* Max abs `initial_projection_delta` delta: 0.000e+00.
* Posthoc summary: cells with normalized_HCE > 0.5 = **17/18**;
  cells with CI lower bound > 0.5 = **16/18**.
* `diff` of the two posthoc `summary.md` files (Linux G4 vs Stage 6C
  CPU baseline) should be empty for the per-projection table.

## 4. Reporting template

After §3.3 + §3.4 complete, fill in this template for the report:

      # Stage G4-Linux report

      ## System
      * OS / kernel:                       e.g. Ubuntu 22.04.4 LTS, 6.5.0-x
      * GPU:                               e.g. NVIDIA GeForce RTX 3080 Ti
      * GPU driver:                        from `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
      * CUDA runtime via cupy:             from `cupy.cuda.runtime.runtimeGetVersion()`
      * Cupy version:                      e.g. 14.0.1
      * Python:                            e.g. 3.12.10
      * CPU model + cores:                 from `lscpu | head -20`
      * RAM total:                         from `free -h`
      * Repo commit / branch:              from `git rev-parse --short HEAD`, `git rev-parse --abbrev-ref HEAD`

      ## Tests
      * `pytest tests/ -q`:                 e.g. 479 passed, 3 skipped
      * `pytest tests/test_gpu_backend.py tests/test_projection_robustness_gpu.py -q`:
                                            e.g. 27 passed, 2 skipped

      ## Linux smoke (n_workers=30)
      * Output dir:                        outputs/g4_linux_smoke_n30_<ts>/
      * Pool restarts:                     <count>
      * Returncode:                        <int>
      * Wall:                              <float> s
      * Equivalence to G2B GPU:            bit-identical / max abs HCE delta = <float>

      ## Full Linux G3-equivalent run
      * Output dir:                        outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_<ts>/
      * total_wall_s:                      <float>
      * cpu_discovery_wall_s:              <float>
      * gpu_rollout_wall_s:                <float>
      * gpu_compute_s:                     <float>
      * cpu_discovery_workers:             <int> (target: 30)
      * pool restarts (g3_perf.n_pool_restarts):  <int> (target: 0 or 1)
      * gpu_memory_peak_mb:                <float>
      * Speedup vs Windows G3 (23 870 s):  <float>x
      * Speedup vs Stage 6C CPU (23 318 s): <float>x

      ## Equivalence vs Stage 6C CPU baseline
      * matched candidate keys:            26 139 / 26 139
      * binary HCE bit-identical:          20 139 / 20 139
      * max abs HCE delta:                 <float>
      * max abs far_HCE delta:             <float>
      * max abs init_proj delta:           <float>
      * cells normalized_HCE > 0.5:        17 / 18
      * cells CI lower bound > 0.5:        16 / 18
      * Stage 6C scientific conclusions:   preserved / changed (one of)

      ## Recommendation
      One of:
      * **Linux deployment** — if Linux removes the worker ceiling (smoke
        clean at n=30, full run in 1.5–2.5 h), the project should adopt
        Linux as the production target for GPU runs and shelve the
        GPU-discovery refactor; the Windows runner stays available for
        developer iteration.
      * **GPU-discovery refactor (G5)** — if Linux still hits the
        worker ceiling (smoke shows pool restarts at n=30), the OS is
        not the binding constraint and the next move is to port the
        per-frame projection / tracker hot loops to GPU. This is a
        larger refactor; the runbook should be updated with the new
        baseline numbers from the Linux smoke before scoping G5.
      * **Mixed** — if the smoke is clean but the full run hits
        restarts at scale, document the safe `--n-workers` ceiling for
        the Linux host and use that for production. Speedup may be
        intermediate (e.g. 2x rather than 3x).

## 4b. One-shot bootstrap script

If you'd rather not step through §1–4 by hand, the repo ships a
single-shot bootstrap script that does everything end-to-end:

      chmod +x scripts/g4_linux_bootstrap.sh
      ./scripts/g4_linux_bootstrap.sh \
          --stage6c-baseline-dir /path/to/outputs/stage6c_projection_robustness_seed7000_<ts>/

Without the baseline path, the script still runs the smoke + full
production-grid GPU run but skips the equivalence comparison and
clearly flags it in the report as "equivalence not run: baseline
dir missing".

What the script does (in order):

1. Checks `uname -s == Linux`; refuses to run on anything else.
2. Verifies repo root, prints git branch / commit / dirty state, and
   confirms the `stage6-replication-complete` tag is reachable.
3. Verifies `nvidia-smi` works and runs a CuPy probe to confirm the
   GPU is visible. Writes the probe output to
   `outputs/g4_linux_cupy_info_<ts>.txt`.
4. Creates `.venv/` if missing and `pip install -e ".[dev,gpu]"`.
   Idempotent via the `.g4_state/pip_done` marker — re-running the
   script after a partial failure skips the install step. Delete that
   marker to force a reinstall.
5. Runs `pytest tests/test_gpu_backend.py tests/test_projection_robustness_gpu.py -q`
   then `pytest tests/ -q`. Records both into per-run log files.
6. Runs the **n_workers=30 stress smoke** (the gate from §3.2).
   Parses `stats_summary.json` for the G3 perf block and counts
   `[discovery] pool died` lines in the log. If any pool restart is
   detected, the script prints a clear failure message and **does
   not start the full production run** — exit code 2.
7. If the smoke gate passes, runs the **full Stage 6C-equivalent
   production run** (§3.3). Parses its perf block likewise.
8. If `--stage6c-baseline-dir` was supplied, calls
   `scripts/g4_compare_to_stage6c.py` to do the row-by-row + posthoc
   equivalence check and emit `outputs/g4_linux_equivalence_<ts>.{json,md}`.
9. Writes the final markdown report to
   `outputs/g4_linux_report_<ts>.md` with system info, all phase
   metrics, speedup vs Windows G3 (6.63 h) and Stage 6C CPU baseline
   (6.48 h), the equivalence verdict, and a **recommendation** chosen
   by:

   * smoke crashed at n=30 → **G5 GPU-discovery refactor** (the
     ceiling isn't Windows-specific).
   * full run < 3 h + equivalence pass → **adopt Linux as production
     target; shelve G5**.
   * full run 3–4 h + equivalence pass → **acceptable; G5 still worth
     scoping** for the next workload size increase.
   * full run > 4 h → **Linux removes crashes but not bottleneck;
     plan G5**.

   The script's exit code mirrors the recommendation:
   0=success, 2=smoke gate failed, 3=full run failed,
   4=equivalence failed.

Useful flags:

      --n-workers N              # default 30; lower if the smoke gate fails
      --gpu-batch-size N         # default 64
      --gpu-memory-target-gb F   # default 9.5
      --skip-tests               # skip both pytest invocations
      --skip-smoke               # skip the gate (use with care)
      --skip-full                # do not run the full production run
      --smoke-only               # run only the smoke gate
      -h | --help                # print usage

Output you can `scp` back to the dev box for analysis:

      outputs/g4_linux_report_<ts>.md             # human-readable report
      outputs/g4_linux_equivalence_<ts>.json      # machine-readable equiv
      outputs/g4_linux_equivalence_<ts>.md
      outputs/g4_linux_bootstrap_<ts>.log         # full transcript
      outputs/g4_linux_pytest_*_${{ts}}.log         # both pytest runs
      outputs/g4_linux_cupy_info_<ts>.txt         # GPU probe output
      outputs/<full_run_dir>/stats_summary.json   # the run's perf block
      outputs/<full_run_dir>/perf_profile.json
      outputs/<full_run_dir>/candidate_metrics.csv

(If disk is tight, the `_workitems/` subdir under the full run
directory holds the per-cell .npz scratch — about 60 GB. Safe to
delete after equivalence is verified.)

## 5. Notes / caveats

* Do not run from Windows. This file is documentation; the dev box's
  Windows joblib stack reproducibly hits the access violation at
  n_workers > 8, and re-running here would only re-confirm what G3
  already showed.
* The GPU backend imports `cupy` at module load time (via
  ``observer_worlds.worlds._cuda_bootstrap``). On Linux the bootstrap
  is a no-op (the `nvidia-cuda-nvrtc-cu12` wheel layout doesn't need
  the Windows-specific `CUDA_PATH` patch); cupy auto-discovers CUDA.
* All three runs above write to `outputs/<label>_<timestamp>/`. Disk:
  the full run produces ~60 GB of npz scratch under
  `outputs/<run>/_workitems/`. After the equivalence check you can
  delete it with `rm -rf outputs/g4_linux_projection_robustness_gpu_stage6c_equivalent_*/_workitems`.
* Scientific equivalence is checked against the Stage 6C CPU baseline
  output dir, which **must** be present in `outputs/`. If the Linux
  host is fresh, copy `outputs/stage6c_projection_robustness_seed7000_20260501T035342Z/`
  from the Windows dev box (or rerun Stage 6C on Linux first; that's
  ~6.5 h and not the point of G4).
* As with all GPU runs, **no scientific conclusion is updated from G4
  output**. The G4 report's job is to characterize wall time and
  confirm the GPU pipeline still matches the Stage 6C CPU baseline,
  nothing more.
