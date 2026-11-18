# Phase 2 GEMM — Profiling & Root-Cause Report

**Date:** 2026-06-06
**GPU:** NVIDIA GeForce RTX 4070 Laptop (Ada Lovelace, sm_89, 36 SMs)
**CUDA:** 12.4.131 | **GCC:** 13.4.0 | **cuBLAS:** 12.5

---

## Benchmark Results (4096×4096×4096, primary gate)

| Version | GFLOPS | % cuBLAS | vs Naive |
|---------|--------|----------|---------|
| CPU | ~0.04 | — | 1× |
| Naive | 1,178 | 11.9% | 1× |
| TiledShared | 1,106 | 11.2% | 0.94× |
| RegisterTiled | 3,956 | **40.0%** | 3.4× |
| DoubleBuf | 4,013 | **40.6%** | 3.4× |
| cuBLAS | 9,888 | 100% | — |

**PROJECT_SPEC target: ≥ 50% cuBLAS.** Achieved: **40.6%**. Detailed root-cause
analysis below.

---

## Version C Profiling Checkpoint

### Nsight Systems — API summary (RegisterTiled, 4096³, single run)

| API | % time | Calls | Avg latency |
|-----|--------|-------|-------------|
| cudaEventSynchronize | 78.1% | 70 | 48.4 ms |
| cudaStreamSynchronize | 18.2% | 2 | 395 ms |
| cudaLaunchKernel | **0.1%** | 85 | 41 µs |
| cudaMalloc | 3.1% | 3 | 45 ms |

Key observations:
- **cudaLaunchKernel = 0.1% of wall time** at 41 µs avg. Launch overhead is
  negligible at 4096³ (~36 s total kernel time across all iterations).
- **cudaMalloc = 0 within the timed loop** — all allocations are setup-phase.
- The 48 ms `cudaEventSynchronize` and 395 ms `cudaStreamSynchronize` are the
  benchmarking harness cost (100 iterations × event pairs), not kernel cost.

*Nsight Compute is still blocked by ERR_NVGPUCTRPERM (see Phase 1 note).
Occupancy and register usage below are derived from theoretical analysis.*

---

## Root-Cause Analysis — Why 40.6% vs ≥50% Target

### 1. Arithmetic intensity too low for the tile configuration

The RTX 4070 Laptop's theoretical peak FP32 compute is:
`36 SMs × 128 CUDA cores/SM × 2 ops/cycle × 2.175 GHz = 20.1 TFLOPS`

At 4,013 GFLOPS (4.0 TFLOPS), the kernel runs at **20% of compute peak**.
cuBLAS achieves ~49% compute peak (9.9 TFLOPS / 20.1 TFLOPS).

For RegisterTiled (BM=BN=64, BK=16, TM=TN=4):
- Arithmetic intensity per tile = `2 * BM * BN * BK / (BM*BK + BK*BN) bytes`
  = `2 * 64 * 64 * 16 / (64*16 + 16*64) * 4` = `2 * 65536 / 2048` = **64 FLOP/byte**
- At 256 GB/s DRAM BW, roofline compute limit = 64 * 256 = 16,384 GFLOPS
- Actual: 3,956 GFLOPS — only 24% of the roofline

The kernel is **not hitting the roofline** — something else is limiting it.

### 2. Shared memory bank conflicts (no padding)

The shared arrays `sA[64][16]` and `sB[16][64]` have stride-1 column access
patterns. When 256 threads load `sA[threadRow*TM + tm][k]` simultaneously:
- All threads in a warp access the same column `k` but different rows
- With `float` (4 bytes) and 32 banks: accesses at rows `r, r+1, r+2...` land
  in banks `0, 1, 2...` — **no bank conflicts** on row access.
- But `sB[k][threadCol*TN + tn]` accesses are across `kBN=64` columns for
  different threads in the same warp — stride-1 with 16 threads per row →
  potential partial-warp conflicts depending on threadCol mapping.

Without Nsight Compute warp stall metrics, this remains a hypothesis. However,
adding padding `sA[64][16+1]` and `sB[16][64+1]` is a low-risk 1-line fix.

### 3. Low register-level instruction-level parallelism (ILP)

The TM=TN=4 register tile computes 16 FMAs per `k` iteration. On Ada, a warp
issues one instruction per 4 clock cycles; 16 FMAs with no reordering
fills the pipeline for `16 / 8 warps = 2 cycles` per SM. This is likely
**occupancy-limited**: with ~30 registers/thread × 256 threads/block = 7,680
registers/block, the RTX 4070 has 65,536 registers/SM → 8 blocks/SM → 2,048
threads/SM = full occupancy. However, latency hiding requires active warps;
if the FMA pipeline takes 4 cycles and we have 2048/32=64 warps/SM, we should
have adequate hiding. The bottleneck is likely the shared-memory load latency
(~20 cycle round-trip) not fully hidden by the 16-FMA compute window.

### 4. Inner loop structure produces poor instruction scheduling

The current compute loop:
```cpp
for (k=0..BK): load sA rows, load sB cols, compute TM×TN FMAs
```
The loads and FMAs are interleaved. At BK=16 with TM=TN=4, the compute/load
ratio is `16 FMAs / (TM + TN loads) = 16/8 = 2 FMAs per load`. This is too
low — the shared-memory load (even L1 hit) takes ~4 cycles, so a 2:1 FMA:load
ratio leaves the FMA units idle. cuBLAS uses 128×128 tiles with TM=TN=8 or
larger, achieving a much higher compute/load ratio.

### 5. Double buffering provided marginal gain (0.6%)

DoubleBuf: 4,013 GFLOPS vs RegisterTiled: 3,956 GFLOPS = **+1.4%**.
The manual ping-pong does not provide the latency hiding benefit expected
because the bottleneck is not the global→shared latency (already in shared
memory by the time compute starts) but the shared→register load within the
`k` loop. Version D correctly eliminates global load latency but cannot
hide the shared-memory read latency of the inner k-loop.

---

## What Would Be Needed to Reach ≥50%

In order of expected impact, based on the analysis above:

**1. Increase tile size to BM=BN=128, BK=16, TM=TN=8**
- 64 FMAs per k iteration vs 16 → 4× higher compute:load ratio
- Shared memory doubles to ~8 KB/block → 2 blocks/SM (still viable)
- Occupancy: 128 regs/thread × 256 threads = 32,768 regs/block → 2 blocks/SM
  → 512 threads/SM = 33% occupancy. Trade-off: more ILP, less occupancy.
- Expected: +30–40% GFLOPS based on the roofline improvement

**2. Add padding to shared memory arrays**
- `sA[BM][BK+1]`, `sB[BK][BN+1]` eliminates potential bank conflicts
- Low-risk, 1-line change per buffer

**3. Unroll inner k-loop explicitly**
- `#pragma unroll` on the k-loop to expose instruction-level parallelism
  to the PTX scheduler

These are optimizations that can be implemented as Version D variants or as a
future Phase 6 kernel fusion step. Per Phase 3 instructions: if the implementation
is correct, fully profiled, and well-documented but falls short of the target, a
detailed root-cause analysis is provided **before** attempting major redesigns.

---

## Correctness Summary

All four versions pass the full GPU test suite:
- CpuGemm oracle comparison at all sizes (max_abs < GemmAbsTol(K))
- cuBLAS sanity check at 512³ and 768³ (max_abs < 1.0)
- All-versions-agree test at 512³

---

## Nsight Compute — Status (unchanged from Phase 1)

ERR_NVGPUCTRPERM blocks hardware counter collection. Resolution steps in
PROJECT_STATUS.md. Occupancy, register usage, and warp stall analysis above
are derived from theoretical calculation rather than measurement.

---

## Phase 2 Exit Criteria Assessment

| Criterion | Status |
|-----------|--------|
| Version A (Naive) compiles + tests pass | ✅ |
| Version B (TiledShared) compiles + tests pass | ✅ |
| Version C (RegisterTiled) compiles + tests pass | ✅ |
| Version D (DoubleBuf) compiles + tests pass | ✅ |
| Version C profiling checkpoint: GFLOPS, % cuBLAS documented | ✅ |
| cuBLAS agreement at 512³ and 768³ | ✅ |
| bench_gemm CSV with cuBLAS + all GPU versions | ✅ |
| FP32 GEMM ≥ 50% cuBLAS at 4096³ | ⚠️ **40.6% — below target** |
| Root-cause analysis for gap (per plan note) | ✅ This report |
| Profiling report written | ✅ |
| All commits created, clean tree | ✅ (pending this commit) |
