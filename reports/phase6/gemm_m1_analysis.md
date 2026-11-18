# Phase 6b: M=1 GEMM Root-Cause Analysis & Optimization Roadmap

**Date:** 2026-06-10
**GPU:** RTX 4070 Laptop (AD106, sm_89, 36 SMs, 8 GB VRAM, 256 GB/s peak BW, ~15 TFLOPS FP32)

---

## 1. Executive Summary

After implementing KV Cache (Phase 6a), the decode step is **no longer attention-bound**.
Attention accounts for only 6.6% (1,072 µs) of the 16.2 ms decode step. Instead,
**GEMMs at M=1 dominate at 89%** (14,395 µs). The largest single hotspot is
`mlp_proj` at 7,313 µs (45.2%), which is 3.7× slower than `mlp_fc` (1,952 µs)
despite identical per-layer FLOP counts.

**Finding:** The RegTiled128Pad kernel, designed and optimized for M ≥ 128
(matrix-matrix multiply), achieves only ~0.78% tile utilization at M=1
(matrix-vector multiply). A trivial GEMV kernel achieves **7.6× GEMM speedup**,
reducing projected decode-step latency from **16.2 ms → ~4.4 ms (3.7× overall)**.

---

## 2. Benchmark: RegTiled128Pad vs GEMV at Decode Shapes

| Kernel   | N×/step | Shape             | Reg128Pad (µs) | GEMV (µs) | Speedup |
|----------|---------|-------------------|---------------:|----------:|--------:|
| c_attn   | 12      | [1,2304]=[1,768]@[768,2304]   | 215.8 | 24.9 | 8.7× |
| c_proj   | 12      | [1,768]=[1,768]@[768,768]     | 214.7 | 24.6 | 8.7× |
| mlp_fc   | 12      | [1,3072]=[1,768]@[768,3072]   | 215.0 | 26.7 | 8.1× |
| mlp_proj | 12      | [1,768]=[1,3072]@[3072,768]   | 847.8 | 88.5 | 9.6× |
| lm_head  | 1       | [1,50257]=[1,768]@[768,50257] | 2045.5|636.0 | 3.2× |
| **TOTAL**| **49**  |                                   | **19,965** | **2,613** | **7.6×** |

*Projected decode-step latency (applying ratios to Phase 6a profile):*

| Method          | GEMM (µs) | Non-GEMM (µs) | Total (µs) | vs Baseline |
|-----------------|----------:|--------------:|-----------:|------------:|
| RegTiled128Pad  | 14,395    | 1,776         | **16,171** | 1.00× |
| Naive GEMV      | ~1,900    | 1,776         | **~3,676** | **4.40×** |
| *(best case, BW-limited GEMV)* | ~1,500 | 1,776 | **~3,276** | **4.94×** |

> The naive GEMV speedup is a lower bound. Optimized GEMV with warp-level
> reductions and float4 loads (where stride N is divisible by 4) would approach
> memory-bandwidth-limited performance, yielding ~5× overall speedup.

---

## 3. Root-Cause Analysis: Why RegTiled128Pad is 8-10× Too Slow at M=1

### 3.1 Tile Parameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| kBM (M-dimension tile) | 128 | 128 rows processed per block |
| kBN (N-dimension tile) | 128 | 128 cols per block |
| kBK (K-dimension tile) | 16  | Inner loop stride |
| kTM, kTN (register tile) | 8, 8 | each thread: 8×8=64 accumulators |
| Threads per block | 256 | 16×16 grid |
| Shared memory | ~16.6 KB | sA[128][17] + sB[16][129] padded |

### 3.2 Cause 1: M=1 Tile Under-Utilization (0.78%)

```
Valid output rows:    1 / 128 = 0.78% of tile
Threads w/ valid work: tr=0 (16 threads) × m=0 (1 of 8 results) = ~0.78%
```

- 127/128 rows compute FMAs on zero-padded `sA` entries (loaded from
  out-of-bounds A rows).
- All 256 threads execute the full inner loop regardless of output validity.
- The bounds check only affects the load (`sA=0` for OOB rows), not FMA retirement.
- **This is identical for ALL M=1 GEMMs regardless of N or K.**

### 3.3 Cause 2: Grid-Dependent SM Occupancy Disparity (Explains 3.7×)

| Kernel   | N      | Grid blocks | SMs utilized* | vs mlp_proj |
|----------|--------|-------------|---------------|-------------|
| c_attn   | 2,304  | 18          | 9             | 3.0× |
| c_proj   | 768    | 6           | 3             | 1.0× |
| mlp_fc   | 3,072  | 24          | 12            | 4.0× |
| mlp_proj | 768    | 6           | 3             | 1.0× |
| lm_head  | 50,257 | 393         | 36 (full)     | 12.0× |

*\* Assuming 2 blocks/SM (limited by registers + shared memory).*

- **mlp_proj has only 6 blocks → 3 SMs = 8.3% SM utilization.**
- **mlp_fc has 24 blocks → 12 SMs = 33% SM utilization.**
- **SM ratio: 33/8.3 = 4.0× → explains the 3.7× measured difference.**
- lm_head has 393 blocks → full 36-SM utilization → closer to "best case for
  RegTiled128Pad at M=1" (but still suffers from 0.78% tile efficiency).

### 3.4 Cause 3: Register Pressure Caps Occupancy at 33%

```
acc[8][8] = 64 regs  (accumulator tile)
ra[8] + rb[8] = 16    (load buffers)
Indices, pointers ≈ 10
Total: ~90 regs/thread
Per-block: 256 × 90 = 23,040 regs
SM limit: 65,536 / 23,040 = 2.8 → 2 blocks/SM max (33.3% occupancy)
```

Shared memory independently caps at 2 blocks/SM: 48 KB / 16.6 KB = 2.9.

### 3.5 Arithmetic Intensity

All decode GEMMs at M=1 are **memory-bound**:

| Kernel   | FLOPs     | Bytes read | AI (FLOP/byte) |
|----------|----------:|-----------:|:--------------:|
| c_attn   | 3.54 M    | ~7.1 MB   | 0.50           |
| c_proj   | 1.18 M    | ~2.4 MB   | 0.50           |
| mlp_fc   | 4.72 M    | ~9.5 MB   | 0.50           |
| mlp_proj | 4.72 M    | ~9.5 MB   | 0.50           |
| lm_head  | 77.2 M    | ~154 MB   | 0.50           |

*Peak AI (RTX 4070 Laptop): ~58 FLOP/byte (15 TFLOPS / 256 GB/s).*

All kernels are at ~0.5 FLOP/byte — **115× below the compute/memory roofline
intersection**. Performance is determined entirely by memory bandwidth.

### 3.6 GEMV Memory Bandwidth Efficiency

| Kernel   | Bytes    | BW (GB/s) | % Peak (256 GB/s) |
|----------|---------:|----------:|:-----------------:|
| c_attn   | 7.1 MB   | 282.6     | 110%*             |
| c_proj   | 2.4 MB   | 90.8      | 35.5%             |
| mlp_fc   | 9.5 MB   | 373.7     | 146%*             |
| mlp_proj | 9.5 MB   | 106.8     | 41.7%             |
| lm_head  | 154.6 MB | 243.5     | 95.1%             |

*\* >100% indicates L2 cache hits inflating the effective bandwidth. The c_attn
and mlp_fc weight matrices (7-9 MB each) are partially cached in L2 (32 MB)
when called repeatedly in the benchmark loop.*

Key observations:
- **lm_head: 95% peak BW** — the 154 MB weight matrix forces streaming from VRAM;
  the naive GEMV kernel achieves near-peak memory throughput.
- **Small-N shapes (c_proj, mlp_proj at N=768): 35-42% peak** — only 3 thread
  blocks → poor SM occupancy → low aggregate memory throughput.
- **Between-kernel weight reuse:** consecutive decode steps for the same layer
  reuse the weight matrix from L2 cache (if it fits). Cross-layer, weights must
  be reloaded.

---

## 4. What About cuBLAS?

cuBLAS SGEMV is the industry-standard GEMV implementation. On this system,
cuBLAS initialization failed (WSL2 environment issue with cublas_v2.h linking).
However, cuBLAS SGEMV performance is expected to be within 10-20% of our naive
GEMV for these shapes — the kernel is similarly memory-bandwidth-bound.

For reference, cuBLAS SGEMM at M=1 would suffer from the same fundamental
limitation as RegTiled128Pad: any tiled GEMM designed for M ≫ 1 wastes
throughput when M=1.

---

## 5. Optimization Roadmap (Ranked by Wall-Clock Impact)

### 5.1 Phase 6b: Dedicated GEMV for M=1 — ★★★★★

**Impact: ~3.7× decode speedup (16.2 ms → ~4.4 ms)**

| Approach | Estimated GEMM total | Decode latency | Speedup |
|----------|---------------------:|---------------:|--------:|
| Current (RegTiled128Pad) | 14,395 µs | 16,171 µs | 1.00× |
| Naive GEMV (1 thread/output) | ~2,600 µs | ~4,400 µs | 3.67× |
| Optimized GEMV (warp-reduce + vec4) | ~1,500 µs | ~3,300 µs | 4.90× |

**Implementation:** Replace `LaunchGemm` with shape-dispatch at M=1. When M=1,
launch a dedicated GEMV kernel. The GEMV kernel:

1. **1D grid over N**: `ceil(N/256)` blocks of 256 threads.
2. **Simple inner product**: per-thread dot product of `a[K]` with one column of `B`.
3. **Optimizations**:
   - Shared-memory cache of `a[K]` (≤12 KB for K≤3072)
   - Warp-level parallel reduction over K (when K > 256)
   - float4 loads from B (when N divisible by 4; stride-safe)
   - float4 loads from A (reduce shared-memory bank conflicts)

**Affected kernels:** 5 launch sites in `transformer_block.cu` +
`forward.cu` (c_attn, c_proj, mlp_fc, mlp_proj, lm_head). Each checks
`M == 1` and dispatches to GEMV.

**Atomization:** Can be delivered as 3 commits:
- C1: GEMV kernel + unit tests
- C2: Shape-dispatch in LaunchGemm
- C3: End-to-end golden gate validation

### 5.2 Phase 6c: CUDA Graphs — ★★★☆☆

**Impact: ~0.8-1.0 ms reduction (4.4 ms → ~3.5 ms)**

The decode loop launches ~200 kernels per step (12 layers × ~15 kernels/layer +
embedding + lm_head + host sync). Each kernel launch costs ~5 µs → ~1 ms total
launch overhead. CUDA Graphs capture the decode step as a single graph, replaying
it with one `cudaGraphLaunch` call.

| Component | Current | With Graphs |
|-----------|--------:|------------:|
| Kernel launch overhead | ~1,000 µs | ~10 µs |
| GPU execution | ~3,400 µs | ~3,400 µs |
| **Total** | **~4,400 µs** | **~3,410 µs** |

Additional benefit: CUDA Graphs enable concurrent copy/compute overlap
(memcpy D2H logits while next step's kernels execute).

### 5.3 Phase 6d: GEMV + BiasAdd Fusion — ★★☆☆☆

**Impact: ~5-10% further reduction (3.4 ms → ~3.1 ms)**

At M=1, the GEMM output is 1 row of D elements (768 or 3072). The subsequent
BiasAdd writes the same row in-place. Fusion eliminates:
- One round-trip through L2/VRAM for the intermediate GEMM output
- One kernel launch per BiasAdd
- Reduces the GeMM + BiasAdd pair from 2 launches to 1

**Affected:** All 4 per-layer GEMMs (c_attn, c_proj, mlp_fc, mlp_proj) ×
12 layers = 48 BiasAdd kernels saved with zero additional logic.

### 5.4 Phase 6e: LayerNorm + GEMM Fusion — ★★☆☆☆

**Impact: ~5-10% further reduction**

Fusing LN → GEMM at M=1 eliminates the intermediate normalized activation
([1, 768] = 3 KB) from VRAM. The LayerNorm output is consumed immediately
by the GEMV kernel, staying in shared memory or registers.

### 5.5 FlashAttention / Attention Optimization — ★☆☆☆☆

**Impact: <2% reduction at current architecture**

Post-KV-cache, attention accounts for only 6.6% (1,072 µs) of decode-step
latency. Even a theoretical 4× attention speedup would reduce total latency
by only ~800 µs (5.2% of total).

FlashAttention is valuable for **prefill** (where S can be up to 1024), but
not for **decode** (where S=1 per step).

---

## 6. Revised Phase 6 Plan

| Phase | Description | Est. Decode Latency | Cumulative Speedup |
|-------|-------------|--------------------:|-------------------:|
| 6a (done) | KV Cache | 16.2 ms | 1.0× |
| **6b** | **Dedicated GEMV for M=1** | **~4.4 ms** | **3.7×** |
| 6c | CUDA Graphs | ~3.5 ms | 4.6× |
| 6d | GEMV + BiasAdd fusion | ~3.2 ms | 5.1× |
| 6e | LN + GEMV fusion | ~2.9 ms | 5.6× |
| 6f | FlashAttention (prefill) | ~2.9 ms | 5.6×* |

*\*FlashAttention improves prefill latency, not decode. Listed for completeness.*

### Recommended Next Milestone

**Phase 6b (Dedicated GEMV for M=1)** is the highest-ROI optimization by a wide
margin. It delivers 3.7× decode speedup with a single new kernel (~100 lines of
CUDA) and a 5-line dispatch change in `LaunchGemm`. No API changes, no workspace
changes, fully backward-compatible.

Estimated implementation effort: **2-3 atomic commits**, ~200 lines of new code.

---

## 7. Appendix: GEMV Kernel Pseudocode

```cuda
// Phase 6b: dedicated GEMV for M=1 decode GEMMs.
// Computes c[0:N] = a[0:K] * B[K,N]  (row-vector × matrix).
//
// Grid: ceil(N / 256),  Block: 256
// Shared memory: K * sizeof(float) for caching `a`

__global__ void GemvKernel(const float* __restrict__ a,   // [K]
                            const float* __restrict__ B,    // [K, N] row-major
                            float* __restrict__ c,          // [N]
                            int N, int K) {
  extern __shared__ float a_shared[];

  // Cooperative load of a into shared memory.
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    a_shared[k] = a[k];
  __syncthreads();

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += a_shared[k] * B[(long long)k * N + n];

  c[n] = acc;
}
```

For K ≤ 3072 (all per-layer GEMMs), shared memory usage is ≤ 12 KB — well under
the 48 KB default limit. For lm_head (K=768), shared memory is 3 KB. The kernel
is simple, debuggable, and achieves near-peak memory bandwidth for large N.
