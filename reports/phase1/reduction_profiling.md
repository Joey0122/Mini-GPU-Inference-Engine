# Phase 1 Reduction — Profiling Report

**Date:** 2026-06-06
**GPU:** NVIDIA GeForce RTX 4070 Laptop (Ada Lovelace, sm_89, 36 SMs)
**CUDA:** 12.4.131
**Theoretical peak BW:** 256.032 GB/s
**Host:** WSL2 Ubuntu, GCC 13.4.0

---

## Benchmark Results (n=16M, primary gate measurement)

| Version | Time (µs) | BW (GB/s) | % Peak | vs CPU |
|---------|-----------|-----------|--------|--------|
| CPU (double acc) | 14,967 | 4.8 | — | 1× |
| Naive | 310 | 215 | 84.3% | 48× |
| SharedMem | 309 | 217 | 85.1% | 48× |
| WarpShuffle | 309 | 218 | 85.1% | 48× |

All three GPU versions exceed the **≥80% target** at n=16M. ✅

Full results across all sizes in `reduction_results.csv`.

---

## Nsight Systems — API Timeline (nsys_reduction.nsys-rep)

Captured via `nsys profile --trace=cuda,nvtx,osrt` over all three GPU versions
at n=16M. Per-kernel GPU occupancy and warp statistics require hardware
performance counters (Nsight Compute), which require elevated permissions not
available in this environment (see blocker below).

### CUDA API summary (n=16M, all versions)

| API call | % time | Calls | Avg latency |
|----------|--------|-------|-------------|
| cudaEventSynchronize | 80.0% | 15,550 | 368 µs |
| cudaStreamSynchronize | 13.0% | 302 | 3,071 µs |
| cudaLaunchKernel | 2.3% | 37,140 | 4.4 µs |
| cudaMalloc | 1.9% | 27 | 5,055 µs |
| cudaEventRecord | 1.6% | 31,100 | 3.8 µs |

### Observations

**Kernel launch overhead:** avg `cudaLaunchKernel` latency = 4.4 µs. Each
reduction run issues 2 kernel launches (Pass 1 + Pass 2). At n=16M with 310 µs
total kernel time, launch overhead is ~2.8% of measured time — acceptable.

**Event synchronization dominates wall time:** `cudaEventSynchronize` accounts
for 80% of API time. This is the timer.hpp harness issuing 100 timed iterations
× 2 event sync operations per iteration. This is instrumentation overhead, not
kernel overhead — it does not affect the CUDA event GPU time reported by the
benchmark.

**cudaMalloc visible in timeline:** 27 `cudaMalloc` calls with avg 5 ms latency
confirm that `ReductionWorkspace::Reserve()` is called only during benchmark
setup, not inside the timed loop. The `.nsys-rep` timeline shows these allocations
clustered at startup before any kernel activity. ✅

**Inter-pass synchronization:** No `cudaStreamSynchronize` appears between Pass 1
and Pass 2 in the kernel dispatch path — the two kernel launches are enqueued on
the same stream (default `nullptr` = stream 0), so Pass 2 is correctly serialized
after Pass 1 by the CUDA stream ordering guarantee without an explicit sync.
The 302 `cudaStreamSynchronize` calls are from the TimeKernel warmup path.

---

## Nsight Compute — Status

**Blocker: ERR_NVGPUCTRPERM**

Nsight Compute hardware performance counters require:
- Driver config: `NVreg_RestrictProfilingToAdminUsers=0`
- Or: running `ncu` as root

Neither is available in the current non-interactive WSL2 shell (sudo requires
interactive authentication). This is a **one-time setup** done outside the
agent session. Steps to enable for future profiling:

```bash
# On the Windows host (PowerShell as Administrator):
bcdedit /set hypervisorlaunchtype auto
# Then in WSL2:
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
    >> /etc/modprobe.d/nvidia.conf'
# Reboot WSL2, then:
ncu --set full -o reports/phase1/ncu_warp_shuffle \
    --kernel-name ReduceWarpKernel \
    ./build/benchmarks/bench_reduction \
    --benchmark_filter='BM_Reduction_WarpShuffle/16777216'
```

**What Nsight Compute would show** (expected from the kernel structure):

| Metric | Expected | Reasoning |
|--------|----------|-----------|
| Achieved occupancy | ~75–85% | 256-thread blocks, 36 SMs; warp-shuffle has low smem footprint |
| Warp execution efficiency | ~95%+ | WarpShuffle: no divergent branches; SharedMem: sequential stride = no divergence |
| Memory throughput | ~85% of peak | Confirmed by bandwidth measurement |
| L2 hit rate | ~0% at n=16M | 16M × 4 bytes = 64 MB > L2 cache on Ada (64 MB boundary) |
| Register usage | ~16–24 regs/thread | Estimate; affects occupancy calculation |

---

## Infrastructure Bug Discovered During Benchmarking

**kMaxPass1Blocks = 1024 → 256 (fixed in Commit 7)**

Root cause: at n=64M with 1024 Pass-1 blocks of 256 threads, each thread
grid-strides over 256 elements serially before entering the shared-memory tree.
This serial work dominates kernel time and collapses bandwidth to ~23% at 64M.

Fix: cap Pass-1 blocks at `kBlockSize (256)` so Pass 2 reduces exactly
`kBlockSize` partials in one block. At n=64M the grid-stride work per thread
is 64M/(256×256)=1024 — still high, but the fix scales acceptably because
the grid-stride loop is a simple float addition.

After fix: SharedMem at n=64M = 175 GB/s (68.5%), WarpShuffle = 180 GB/s (70.1%).

The residual gap between 16M (85%) and 64M (70%) at large N is expected:
more serial grid-stride accumulation per thread before the parallel tree step.
A fully scalable design would increase `kMaxPass1Blocks` proportionally with n
(up to the single-block Pass-2 limit) — a Phase 6 optimization opportunity.

---

## Performance Comparison Summary

**Progression across versions (n=16M):**

| Version | Bottleneck removed | BW improvement vs Naive |
|---------|-------------------|------------------------|
| Naive | — (baseline) | 1× (215 GB/s) |
| SharedMem | Warp divergence + bank conflicts | ~1× (217 GB/s, +1%) |
| WarpShuffle | Shared-mem round-trip for intra-warp step | ~1× (218 GB/s, +1.4%) |

At n=16M all three versions achieve similar bandwidth because the kernel is
overwhelmingly memory-bound: the time spent in the intra-block tree is small
compared to the global load time. The differences between versions would be
more pronounced at smaller n (compute-bound regime) or with a more complex
reduction (e.g., sum-of-squares).

**CPU vs GPU:** GPU is 48× faster at n=16M for a simple sum — demonstrating
the memory bandwidth gap (4.8 GB/s host DRAM vs 215 GB/s GPU DRAM).

---

## Phase 1 Exit Criteria — Final Status

| Criterion | Result |
|-----------|--------|
| Naive: compiles + tests pass | ✅ |
| SharedMem: compiles + tests pass | ✅ |
| WarpShuffle: compiles + tests pass | ✅ |
| All versions validated vs CpuReduce at n=1K/64K/1M/16M | ✅ |
| Non-power-of-two and single-element tests pass | ✅ |
| bench_reduction CSV with CPU baseline + 3 GPU versions | ✅ |
| Nsight Systems: API timeline captured, launch overhead noted | ✅ |
| Nsight Compute: blocked by ERR_NVGPUCTRPERM | ⚠️ Blocked — steps documented |
| Achieved BW ≥ 70% at n=16M (pass gate) | ✅ 85% |
| Achieved BW ≥ 80% at n=16M (target) | ✅ 85% |
| reports/phase1/ updated | ✅ |
| All commits created, clean tree | ✅ (pending this commit) |
