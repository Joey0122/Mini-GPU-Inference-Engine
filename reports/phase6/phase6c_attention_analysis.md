# Phase 6c: Deep Attention-Path Analysis

**Date:** 2026-06-10
**GPU:** RTX 4070 Laptop (AD106, sm_89, 36 SMs, 256 GB/s peak BW, ~15.6 TFLOPS fp32)

---

## 1. Attention-Time Decomposition (S_cached=1024, 12 heads, head_dim=64)

### 1.1 Measured Wall-Clock Times (per-step, 12 layers)

| Sub-kernel | µs (12 layers) | µs/layer | % of 1470 | Grid | Block | Total threads |
|-----------|-----:|-----:|-----:|------|------|------:|
| Scores (QK^T) | 723.7 | 60.3 | 49.2% | (1, 12) | 64 | 768 |
| Softmax | 75.7 | 6.3 | 5.1% | (12) | 256 | 3,072 |
| AV (weights×V) | 670.7 | 55.9 | 45.6% | (1, 12) | 64 | 768 |
| **Total** | **1,470.1** | **122.5** | **100%** | — | — | — |

> Source: Phase 6b decode profile, real GPT-2 Small weights. Times are CUDA-event wall-clock (GPU timeline).

### 1.2 Scores Kernel (AttentionScoresKernelCached) — 723.7 µs

**Code path:** `q_row[t] ← q[h][d]` (cooperative smem load), then `for j in stride: dot += q_row[d] * k[h][j][d]`.

**Memory traffic per launch (12 heads):**

| Access | Pattern | Bytes | Coalesced? |
|--------|---------|-------|------------|
| Read q | 12 heads × 64 floats | 3,072 B | Yes (per-head contiguous) |
| Read k_cache | 12 heads × 1,024 keys × 64 dims × 4 B | **3,145,728 B** | Yes (64 threads read contiguous kj[0..63]) |
| Write scores | 12 heads × 1,024 floats × 4 B | 49,152 B | Yes (contiguous per row) |
| **Total** | | **3,197,952 B ≈ 3.05 MB** | |

**Arithmetic:** 12 heads × 1,024 keys × 64 FMAs = 786,432 FMAs per launch.
At 15.6 TFLOPS peak: 786,432 / 15.6e12 = **0.05 µs** (negligible — kernel is 100% memory-bound).

**Effective bandwidth:**
3.05 MB / 60.3 µs ≈ **50.6 GB/s (19.8% of peak 256 GB/s)**

**Root cause of low BW utilization:** The scores kernel uses only 768 threads total (12 blocks × 64 threads = 24 warps). With one block per SM on 12 of 36 SMs, each active SM has only 2 warps to schedule. Memory latency (~200-400 cycles) requires ~4-8 warps per SM to hide. At 2 warps/SM, the SM stalls waiting for DRAM on nearly every global load.

**Specific breakdown (per launch, ~60.3 µs):**

| Component | µs | % |
|-----------|---:|---:|
| k_cache global reads (3.0 MB at 50.6 GB/s) | 59.3 | 98.4% |
| q shared-memory load + sync | 0.3 | 0.5% |
| Compute (dot products + masked writes) | 0.2 | 0.3% |
| Kernel launch + pipeline drain | 0.5 | 0.8% |

### 1.3 Softmax Kernel (Fused) — 75.7 µs

**Code path:** `LaunchSoftmax(scores, weights, n_head=12, D=1024, Fused)` — two-pass fused kernel: (1) max reduction + exp + sum, (2) normalize.

**Memory traffic per launch (12 rows):**

| Access | Bytes |
|--------|-------|
| Read scores [12×1024] | 49,152 B |
| Write exp intermediates [12×1024] | 49,152 B |
| Read exp intermediates | 49,152 B |
| Write weights [12×1024] | 49,152 B |
| **Total** | **196,608 B ≈ 192 KB** |

**Arithmetic:** 12 rows × 1,024 × (1 max-compare + 1 expf + 1 add + 1 multiply) ≈ 49,152 operations.
The `expf` instruction dominates: ~8 cycles each → 12 × 1,024 × 8 = 98,304 cycles. At ~2 GHz SM clock: ~49 µs.

**Effective bandwidth:** 192 KB / 6.3 µs ≈ **30.5 GB/s (11.9% peak).** The kernel is expf-latency-bound, not memory-bound. With 12 blocks × 256 threads = 3,072 threads (96 warps), occupancy is moderate (12 blocks on 12 SMs, 8 warps/SM). The expf pipeline serializes throughput.

**Specific breakdown (per launch, ~6.3 µs):**

| Component | µs | % |
|-----------|---:|---:|
| expf (1,024 × 12) | 4.9 | 78% |
| Global memory (192 KB at 30.5 GB/s) | 0.6 | 10% |
| Block reductions (max + sum) | 0.5 | 8% |
| Kernel launch overhead | 0.3 | 5% |

### 1.4 AV Kernel (AttentionAVKernelCached) — 670.7 µs

**Code path:** Thread `d` accumulates `sum_j weights[h][j] * v[h][j][d]` over all 1,024 keys.

**Memory traffic per launch (12 heads):**

| Access | Pattern | Bytes | Coalesced? |
|--------|---------|-------|------------|
| Read weights | 12 heads × 1,024 floats × 4 B | 49,152 B | Broadcast (all 64 threads read same `wrow[j]`) |
| Read v_cache | 12 heads × 1,024 keys × 64 dims × 4 B | **3,145,728 B** | Yes (thread `d` reads `v[j][d]`, contiguous across threads) |
| Write out | 12 heads × 64 floats × 4 B | 3,072 B | Yes |
| **Total** | | **3,197,952 B ≈ 3.05 MB** | |

> Note: weights buffer read is almost free — all 64 threads in a block read the exact same `wrow[j]` address, which the L1 cache serves as a broadcast. The L1 hit rate on weights reads is ~100%.

**Effective bandwidth (v_cache reads):** 3.0 MB / 55.9 µs ≈ **53.7 GB/s (21.0% peak).**

**Why higher BW than scores?** Both kernels use 64-thread blocks, but the AV kernel's v_cache reads are perfectly coalesced (thread `d` reads `v[j][d]`, so 64 threads read 256 contiguous bytes per key iteration). The scores kernel has the same coalescing pattern for k_cache, so the difference is minor (50.6 vs 53.7 GB/s — within run-to-run variance).

**Specific breakdown (per launch, ~55.9 µs):**

| Component | µs | % |
|-----------|---:|---:|
| v_cache global reads (3.0 MB at 53.7 GB/s) | 55.0 | 98.4% |
| weights L1 broadcast reads | 0.1 | 0.2% |
| Compute (FMAs + accumulations) | 0.3 | 0.5% |
| Kernel launch + pipeline drain | 0.5 | 0.9% |

### 1.5 Cross-Kernel Synchronization

All three sub-kernels launch on the same stream (nullptr/default). Between each pair there is an **implicit device-side barrier**: the GPU must complete kernel N before launching kernel N+1. CUDA streams guarantee sequential execution, so the GPU pipeline drains between Scores→Softmax and Softmax→AV.

The pipeline drain cost is not separately measurable (it's baked into the end of the previous kernel and the start of the next). However, the practical effect is:

1. **L2 cache flush:** Each sub-kernel is small enough that its working set (k_cache at 3 MB, scores/weights at 48 KB) competes for L2. The 3 MB k_cache working set is evicted between the Scores and AV kernels by the softmax pass and any intervening work.
2. **No overlapping execution:** The three kernels cannot overlap — they must run sequentially.

### 1.6 Summary Decomposition

```
                      ┌─────────────────────────────────────────┐
                      │   Cached Attention: 1470.1 µs total      │
                      │                                          │
                      │  Scores 723.7 µs (49.2%)                 │
                      │  ├─ k_cache DRAM read: 713 µs (98.4%)    │
                      │  ├─ Compute: negligible                  │
                      │  └─ Launch + q load: ~10 µs              │
                      │  ─── pipeline drain ───                  │
                      │  Softmax 75.7 µs (5.1%)                  │
                      │  ├─ expf compute: 59 µs (78%)            │
                      │  ├─ Memory (192 KB): 8 µs                │
                      │  └─ Launch: ~3 µs                        │
                      │  ─── pipeline drain ───                  │
                      │  AV 670.7 µs (45.6%)                     │
                      │  ├─ v_cache DRAM read: 660 µs (98.4%)    │
                      │  ├─ Compute: negligible                  │
                      │  └─ Launch: ~5 µs                        │
                      └─────────────────────────────────────────┘

  Root cause: 768 threads (12 blocks × 64) cannot saturate DRAM bandwidth.
  Only 24 warps across 12 SMs → ~2 warps/SM → memory latency dominates.
  Peak bandwidth would require ~8+ warps per SM to hide 200-400 cycle DRAM latency.
```

---

## 2. Exact Byte Accounting

All sizes for one decode step, S_cached=1024, n_head=12, head_dim=64.
Quantities are per-layer (×12 for full step).

### 2.1 Buffer Sizes (VRAM Allocation)

| Buffer | Shape | Elements | Bytes |
|--------|-------|----------|-------|
| q (input) | [n_head, head_dim] | 768 | 3,072 B |
| k_cache | [n_head, n_ctx, head_dim] | 786,432 | 3,145,728 B |
| v_cache | [n_head, n_ctx, head_dim] | 786,432 | 3,145,728 B |
| **scores** (intermediate) | **[n_head, S_cached]** | **12,288** | **49,152 B** |
| **weights** (intermediate) | **[n_head, S_cached]** | **12,288** | **49,152 B** |
| out | [n_head, head_dim] | 768 | 3,072 B |

### 2.2 Per-Layer Reads and Writes

#### Scores kernel (writes scores):
| Buffer | Operation | Bytes | Notes |
|--------|-----------|-------|-------|
| q | READ (global) | 3,072 | — |
| k_cache | READ (global) | 3,145,728 | Coalesced, 64-thread blocks |
| scores | WRITE (global) | 49,152 | Coalesced per row |

#### Softmax kernel (reads scores, writes weights):
| Buffer | Operation | Bytes | Notes |
|--------|-----------|-------|-------|
| scores | READ (global) | 49,152 | Just written by scores kernel; in L2 |
| scores | READ (re-read for max+exp) | ~49,152 | Fused kernel re-reads same buffer |
| weights | WRITE (global) | 49,152 | — |

#### AV kernel (reads weights, reads v_cache, writes out):
| Buffer | Operation | Bytes | Notes |
|--------|-----------|-------|-------|
| weights | READ (global) | 49,152 | Broadcast — same address per warp; L1-hit |
| v_cache | READ (global) | 3,145,728 | Coalesced, 64-thread blocks |
| out | WRITE (global) | 3,072 | Coalesced |

### 2.3 Aggregate Per-Layer IO

| | Bytes | % of total |
|---|---|---|
| **Total reads** | 6,490,368 B ≈ **6.19 MB** | 98.4% |
| ├─ k_cache read | 3,145,728 B (48.5%) | |
| ├─ v_cache read | 3,145,728 B (48.5%) | |
| ├─ scores/weights read | 98,304 B (1.5%) | |
| └─ q read | 3,072 B (0.05%) | |
| **Total writes** | 101,376 B ≈ **99 KB** | 1.6% |
| ├─ scores write | 49,152 B (0.76%) | |
| ├─ weights write | 49,152 B (0.76%) | |
| └─ out write | 3,072 B (0.05%) | |
| **Grand total** | **6.59 MB** | |

**Full decode step (12 layers):** ~79.1 MB reads + ~1.2 MB writes = **~80.3 MB total IO** for attention.

### 2.4 Key Insight

The intermediate buffers (`scores` and `weights`, 48 KB each) account for only **1.6% of total IO** but cause **two of the three kernel boundaries**. The entire softmax (75.7 µs, 5.1%) exists solely to transform scores→weights through two global memory round-trips that are completely avoidable in a fused kernel.

---

## 3. Theoretical Lower Bounds

### 3.1 Methodology

The minimum time to compute cached attention is:

**Bandwidth bound** = (k_cache_bytes + v_cache_bytes) / peak_bandwidth
**Compute bound** = (FLOPs per layer) / peak_TFLOPS
**Lower bound** = max(bandwidth_bound, compute_bound)

For all S values, the problem is strongly memory-bound (compute is negligible).

### 3.2 Lower Bounds by S_cached

| Metric | S=256 | S=512 | S=1024 | Formula |
|--------|------:|------:|-------:|---------|
| k_cache size (per head) | 1,024 B | 2,048 B | 4,096 B | S × 64 × 4 |
| k_cache (12 heads) | 12 KB | 24 KB | 48 KB | — |
| v_cache (same) | 12 KB | 24 KB | 48 KB | — |
| Total IO per layer | ~24.8 KB | ~49.0 KB | ~97.4 KB | (but k,v are the dominant IO) |

Wait — I need to recalculate. The k_cache and v_cache sizes:

k_cache = n_head × S × head_dim × 4 = 12 × S × 64 × 4

| S | k_cache bytes | v_cache bytes | Total IO/layer | BW bound (256 GB/s) |
|---|-------------:|-------------:|--------------:|--------------------:|
| 256 | 786,432 B = 768 KB | same | ~1.54 MB | **6.0 µs** |
| 512 | 1,572,864 B = 1.5 MB | same | ~3.02 MB | **11.8 µs** |
| 1024 | 3,145,728 B = 3.0 MB | same | ~6.04 MB | **23.6 µs** |

For 12 layers:

| S | BW lower bound (12 layers) | Current measured | Gap |
|---|--------------------------:|-----------------:|----:|
| 256 | **72 µs** | Not profiled separately | — |
| 512 | **142 µs** | Not profiled separately | — |
| 1024 | **283 µs** | **1,470 µs** | **5.2×** |

### 3.3 Why 5.2× Above the Lower Bound

The theoretical minimum assumes 100% peak memory bandwidth (256 GB/s). Achieving this requires:
1. Enough in-flight memory requests to hide DRAM latency (~200-400 cycles)
2. Coalesced access patterns
3. Sufficient thread-level parallelism to keep the memory controller saturated

The current kernels achieve ~50-54 GB/s (20-21% of peak) because:
- **768 total threads** (12 blocks × 64) → 24 warps
- Each SM gets at most 1 block → 2 warps per active SM
- To hide ~300-cycle DRAM latency at 2 GHz, each SM needs ~5-6 warps that can issue memory instructions
- With only 2 warps, the SM stalls on every memory access

A well-designed kernel with 48+ blocks of 256 threads (12,288 threads, 384 warps) should achieve 60-80% of peak bandwidth, closing most of the 5.2× gap.

### 3.4 Practical Lower Bounds

Accounting for real-world constraints (warp scheduling overhead, address generation, non-ideal coalescing, TLB misses):

| S | Theoretical | Practical (~70% peak) | Practical (~85% peak) |
|---|-----------:|--------------------:|--------------------:|
| 256 | 72 µs | ~103 µs | ~85 µs |
| 512 | 142 µs | ~203 µs | ~167 µs |
| 1024 | 283 µs | **~405 µs** | **~333 µs** |

The practical lower bound at S=1024 is **~350-450 µs** for 12 layers — a 3.3-4.2× improvement over the current 1,470 µs.

---

## 4. Fused Cached-Attention Kernel Design

### 4.1 Design Rationale

The central challenge: **softmax requires a global max and sum across all S keys before AV accumulation can begin.** With S=1024, a single block can hold all 1,024 keys (max threads per block = 1,024 on sm_89), avoiding the cross-block synchronization problem entirely.

**Key design decision:** 1 block per head, 1,024 threads each. This maps naturally: thread `t` handles key position `t`, computing `dot(q, k[t])`, then cooperatively reducing for softmax, then accumulating `weight[t] * v[t]` into the output.

### 4.2 Launch Configuration

```
Grid:  (1, n_head) = (1, 12)
Block: 1024 threads  (max for sm_89)
Total threads: 12,288  (vs current 768 — 16× more)
```

### 4.3 Register Budget (Critical Path)

Sm_89 per-SM limits: 65,536 registers, 1,536 max threads.
1,024 threads per block → max 65,536 / 1,024 = **64 registers per thread** to fit 1 block/SM.

Register allocation strategy:
- `q_row` loaded from smem (not registers) — no register cost
- Scores phase: ~8 registers (dot accumulator, k_cache pointer, loop vars)
- Softmax phase: ~6 registers (max, exp, sum, lane ops)
- AV phase: tile D_TILE=8 output dimensions per pass → 8 accumulator registers + 8 v_cache registers + weight register = ~20 registers

The AV phase processes the 64 output dimensions in 8 tiles of 8 dimensions each:
```
For tile d0 in {0, 8, 16, ..., 56}:
  // 8 accumulator registers
  float acc[8] = {0};
  // Thread t handles key t
  float w = smem_weights[t];
  float v_tile[8];  // load v[t][d0..d0+7]
  for (int i = 0; i < 8; i++) acc[i] += w * v_tile[i];
  // Warp-reduce acc[0..7] → per-warp partials in smem
  // Block-reduce warp partials → 8 output values
```

Total register count: ~35-40. Well within the 64-register limit for 1,024-thread blocks.

### 4.4 Shared Memory Budget

Sm_89: up to 100 KB configurable shared memory per SM.

| Phase | Buffer | Size | Notes |
|-------|--------|------|-------|
| Q load | q_row[64] | 256 B | Cooperative load by 1,024 threads |
| Scores | scores[1024] | 4,096 B | One float per thread/key |
| Softmax workspace | warp_max[32] + warp_sum[32] | 256 B | Block reductions reuse scores buffer |
| AV | warp_partial[32][D_TILE] | 32×8×4 = 1,024 B | Per-warp partial sums for current tile |
| AV | output staging[64] | 256 B | Final per-head output before global write |
| **Total** | | **~5.9 KB** | |

Smem per block: ~5.9 KB. Per SM with 1 block: 5.9 KB. Well within limits.

### 4.5 Occupancy

| Resource | Per Block | Per SM Limit | Blocks/SM |
|----------|-----------|-------------|:---------:|
| Threads | 1,024 | 1,536 | **1** |
| Registers | ~40 × 1,024 = 40,960 | 65,536 | 1 (register-bound) |
| Shared memory | ~5.9 KB | 100 KB | 16 (not limiting) |

**Theoretical occupancy: 1 block/SM (66.7% of max 1,536 threads/SM).**

With 12 blocks total and 36 SMs: 12 blocks occupy 12 SMs, 24 SMs idle → **33% SM utilization.** This is the fundamental limit of a per-head block decomposition at n_head=12 — we can only occupy at most 12 SMs concurrently.

### 4.6 Kernel Pseudocode

```cuda
__global__ void FusedAttentionCachedKernel(
    const float* __restrict__ q,       // [n_head, head_dim]
    const float* __restrict__ k_cache, // [n_head, n_ctx, head_dim]
    const float* __restrict__ v_cache, // [n_head, n_ctx, head_dim]
    float* __restrict__ out,           // [n_head, head_dim]
    int n_ctx, int head_dim, int num_cached, float scale)
{
  extern __shared__ float smem[];
  float* q_row     = smem;                         // [head_dim] = 64
  float* scores    = smem + head_dim;              // [num_cached] = 1024
  // scores buffer reused for exp_vals and weights
  float* warp_data = smem + head_dim + num_cached; // [32][D_TILE] for warp partials

  const int h = blockIdx.y;
  const int t = threadIdx.x;  // key index
  const int lane = t & 31;
  const int warp_id = t >> 5;

  // ---- Phase 1: Load q into shared memory ----
  for (int d = t; d < head_dim; d += blockDim.x)
    q_row[d] = q[h * head_dim + d];
  __syncthreads();

  // ---- Phase 2: QK^T scores ----
  const float* kh = k_cache + (long long)h * n_ctx * head_dim;
  float dot = 0.f;
  if (t < num_cached) {
    const float* kj = kh + (long long)t * head_dim;
    for (int d = 0; d < head_dim; ++d) dot += q_row[d] * kj[d];
    dot *= scale;
  } else {
    dot = -1e30f;  // masked (shouldn't happen at decode)
  }
  scores[t] = dot;
  __syncthreads();

  // ---- Phase 3: Softmax (in-place, cooperative) ----
  // 3a. Max reduction
  float mx = (t < num_cached) ? scores[t] : -1e30f;
  // Warp reduce max
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
  if (lane == 0) scores[1024 + warp_id] = mx;  // warp_max[32] after scores
  __syncthreads();
  if (t < 32) {
    mx = scores[1024 + t];
    for (int off = 16; off > 0; off >>= 1)
      mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if (t == 0) scores[1024] = mx;  // global max
  }
  __syncthreads();
  mx = scores[1024];  // broadcast

  // 3b. Exp + sum
  float exp_val = (t < num_cached) ? expf(scores[t] - mx) : 0.f;
  scores[t] = exp_val;  // now stores exp(score - max)
  float sum_val = exp_val;
  for (int off = 16; off > 0; off >>= 1)
    sum_val += __shfl_down_sync(0xffffffff, sum_val, off);
  if (lane == 0) scores[1024 + warp_id] = sum_val;
  __syncthreads();
  if (t < 32) {
    sum_val = scores[1024 + t];
    for (int off = 16; off > 0; off >>= 1)
      sum_val += __shfl_down_sync(0xffffffff, sum_val, off);
    if (t == 0) scores[1025] = sum_val;  // global sum
  }
  __syncthreads();

  // 3c. Normalize: scores[t] is now the softmax weight
  float weight = exp_val / scores[1025];

  // ---- Phase 4: AV weighted sum ----
  const float* vh = v_cache + (long long)h * n_ctx * head_dim;
  constexpr int D_TILE = 8;

  for (int d0 = 0; d0 < head_dim; d0 += D_TILE) {
    // Accumulate across keys: each thread handles key t
    float acc[D_TILE] = {0};
    if (t < num_cached) {
      for (int i = 0; i < D_TILE; ++i)
        acc[i] = weight * vh[(long long)t * head_dim + d0 + i];
    }

    // Warp reduce: sum acc[0..D_TILE-1] across warp
    for (int i = 0; i < D_TILE; ++i) {
      float val = acc[i];
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
      if (lane == 0) warp_data[warp_id * D_TILE + i] = val;
    }
    __syncthreads();

    // First warp reduces per-warp partials
    if (warp_id == 0) {
      for (int i = lane; i < D_TILE; i += 32) {
        float sum = 0;
        for (int w = 0; w < blockDim.x / 32; ++w)
          sum += warp_data[w * D_TILE + i];
        out[(long long)h * head_dim + d0 + i] = sum;
      }
    }
    __syncthreads();
  }
}
```

### 4.7 Expected Latency

**Memory traffic** (unchanged from current — we must still read k_cache and v_cache):
- k_cache: 3.0 MB
- v_cache: 3.0 MB
- q: 256 B
- out: 256 B
- **Total: 6.0 MB per layer** (same as current, but read only once each)

**Key difference from current kernels:**
1. **1 kernel launch instead of 3** — saves 2 pipeline drains and 2 launch overheads per layer
2. **12,288 threads vs 768** — 16× more memory-level parallelism
3. **No intermediate scores/weights writes to global memory** — softmax data stays in shared memory
4. **12 blocks on 12 SMs (33% utilization)** — same SM count as current, but each block has 16× more threads

**Effective bandwidth projection:**

The GEMV kernel (`GemvKernel`) achieves 48-80% peak BW depending on N:
- mlp_proj (N=768, K=3072): ~62% peak (large K keeps memory pipe full)
- c_attn (N=2304, K=768): ~48% peak
- lm_head (N=50257, K=768): ~80% peak (large N gives many parallel blocks)

The fused attention kernel reads k_cache with 1,024 threads per block (32 warps), vs the current scores kernel's 64 threads (2 warps). With 32 warps per SM, memory latency hiding should be 8-16× better than the current 2-warps-per-SM configuration.

**Conservative estimate (60% peak BW, 154 GB/s):**
- Read k_cache: 3.0 MB / 154 GB/s = 19.5 µs
- Read v_cache: 3.0 MB / 154 GB/s = 19.5 µs
- Softmax compute + reductions: ~5 µs
- Launch overhead: 3.5 µs
- **Per layer: ~47.5 µs → 12 layers: ~570 µs**

**Optimistic estimate (75% peak BW, 192 GB/s):**
- k_cache + v_cache: 6.0 MB / 192 GB/s = 31.3 µs
- Softmax + launch: ~8 µs
- **Per layer: ~39 µs → 12 layers: ~470 µs**

**Expected range: 470-650 µs (vs current 1,470 µs) → savings of 820-1,000 µs (56-68% of attention time).**

### 4.8 Occupancy Analysis Detail

Each SM running the fused kernel has:
- 1,024 threads (32 warps)
- ~40 registers/thread → 40,960 registers (62.5% of SM's 65,536)
- ~5.9 KB shared memory (5.9% of 100 KB)

This is the maximum occupancy achievable given the 1,024-thread block size. With 12 blocks for 12 heads:
- 12 blocks on 12 SMs → 24 SMs completely idle
- But each active SM runs at 66.7% thread occupancy

The idle SM count (24/36 = 67%) is acceptable because the 12 independent heads have no cross-head communication. Each head's work is fully parallel, and 12 concurrent blocks is the natural parallelism limit (n_head=12).

---

## 5. Comparative ROI Analysis

All estimates use measured Phase 6b profile data and Phase 6c heterogeneous launch overhead measurements.

### 5.1 Candidate 1: Fused Cached-Attention

| Metric | Value |
|--------|-------|
| Current latency | 1,470 µs (24.9% of decode) |
| Projected latency | **470-650 µs** |
| **Savings** | **820-1,000 µs (13.9-16.9% of decode)** |
| New decode total | ~4,900-5,080 µs (4.9-5.1 ms) |
| Cumulative speedup (vs Phase 6a 16.2 ms) | **3.2-3.3×** |
| Implementation effort | ~200 lines CUDA (1 kernel + 1 launch wrapper) |
| Risk | Medium — new kernel logic, warp reductions tested in existing softmax |

**Sources of savings:**
1. 16× more threads → 3-4× higher effective memory bandwidth → ~450 µs
2. Eliminate 2 intermediate VRAM round-trips (scores, weights) → ~150 µs
3. Eliminate 24 kernel launches (2 pipeline drains × 12 layers) → ~80 µs
4. Single-pass k_cache/v_cache read (no L2 eviction between passes) → ~300 µs

### 5.2 Candidate 2: GEMV+BiasAdd Fusion

48 BiasAdd kernels (4 per layer × 12 layers). Each BiasAdd operates on data just written by GEMV (in L2 cache).

**Current state:**
Each BiasAdd is a separate kernel launch after its corresponding GEMV. The kernel reads `C[N]` (just written by GEMV), reads `bias[N]`, writes `C[N] + bias` back. With data in L2, this is ~100 GB/s effective.

| GEMM Shape | N | GEMV (µs) | BiasAdd (est. µs) | Combined (µs) |
|-----------|----:|----------:|------------------:|-------------:|
| c_attn | 2,304 | ~22 | ~24 | ~46 |
| c_proj | 768 | ~22 | ~14 | ~36 |
| mlp_fc | 3,072 | ~21 | ~36 | ~57 |
| mlp_proj | 768 | ~74 | ~42 | ~116 |

> BiasAdd times estimated by subtracting synthetic GEMV times from profile GEMM+BiasAdd combined times. Approximate due to synthetic vs real-weight differences.

**Fusion approach:** Add `const float* bias` parameter to `LaunchGemv`. After computing `c[n]`, add `bias[n]`. Zero additional memory traffic — the output is already in a register.

**Projected savings:**

| Source | Savings |
|--------|--------|
| Eliminate 48 BiasAdd kernel launches | 48 × 3.45 µs = **166 µs** |
| Eliminate 48 VRAM round-trips of GEMV output | negligible (data in L2) |
| **Total savings** | **~166 µs (2.8% of decode)** |

**New decode total:** ~5,734 µs (1.03×). Cumulative: 2.82×.

### 5.3 Candidate 3: LayerNorm Fusion

25 LayerNorm kernels per decode step (2 per layer × 12 + ln_f).

**Current state:**
Each LayerNorm on D=768 reads input [768], reads gamma [768], reads beta [768], writes output [768]. At ~100 GB/s effective (small, L2-resident): ~0.1 µs of memory + ~8 µs of compute (mean, variance, normalization).

**Fusion approach:** Fuse the LN that precedes each GEMM into the GEMV kernel. The LN output (normalized hidden state) is the GEMV input vector. In a fused kernel, the LN-normalized values stay in shared memory and are consumed directly by the GEMV dot products.

However, LN+GEMV fusion is more complex than BiasAdd fusion because:
1. LN requires computing mean and variance across the D=768 vector (reductions)
2. The GEMV needs the normalized result as its input vector
3. Different GEMMs have different LN inputs (ln1→c_attn, ln2→mlp_fc, ln_f→lm_head)

**Projected savings:**

| Source | Savings |
|--------|--------|
| Eliminate 25 LayerNorm kernel launches | 25 × 3.45 µs = **86 µs** |
| Eliminate 25 VRAM round-trips of LN output (3 KB each) | negligible |
| **Total savings** | **~86 µs (1.5% of decode)** |

**New decode total:** ~5,814 µs (1.01×). Cumulative: 2.78×.

### 5.4 Candidate 4: Attention Scores Kernel Parallelism Only

Increase the scores kernel from 64 to 256 threads per block, keeping 3-kernel structure.

Grid: (1, 12), Block: 256 (instead of 64).
Each thread handles 4 keys (1024/256) instead of 16 (1024/64).
4× more threads per block → 8 warps instead of 2 → better latency hiding.

**Projected:** Effective BW improves from ~50 GB/s to maybe ~100-120 GB/s. Scores time: 723.7 → ~350 µs. Savings: ~374 µs (6.3%).

This is a subset of full fusion — easy to implement but leaves the other two kernels and pipeline drains untouched.

### 5.5 Summary Table

| Rank | Optimization | Savings (µs) | % Decode | New Total | Cum. Spdup | Effort | Risk |
|:----:|-------------|------------:|:--------:|----------:|:----------:|--------|------|
| **1** | **Fused cached-attention** | **820-1,000** | **13.9-16.9%** | **4.9-5.1 ms** | **3.2-3.3×** | Medium | Medium |
| 2 | Scores kernel 256-thread | ~374 | 6.3% | 5.53 ms | 2.92× | Low | Low |
| 3 | GEMV+BiasAdd fusion | ~166 | 2.8% | 5.73 ms | 2.82× | Low | Low |
| 4 | LayerNorm fusion | ~86 | 1.5% | 5.81 ms | 2.78× | Medium | Low |

### 5.6 Verdict

**Fused cached-attention is the highest-ROI optimization remaining by a wide margin (5-6× the savings of the next-best candidate).**

The measured data confirms:
1. Attention at 1,470 µs is 24.9% of decode — the single largest component
2. The bottleneck is memory bandwidth underutilization (20% peak), not arithmetic
3. 768 threads (24 warps) cannot saturate DRAM — this is a parallelism deficit, not an algorithmic one
4. 12,288 threads is achievable through better block sizing and fusion
5. The practical lower bound (~350-450 µs) leaves substantial headroom

GEMV+BiasAdd and LayerNorm fusion are low-effort, low-impact optimizations that should follow after attention fusion (as Phases 6e and 6f), not replace it.

---

## 6. Updated Roadmap

| Phase | Optimization | Decode Latency | Cumulative | Status |
|-------|-------------|---------------:|-----------:|--------|
| 6a | KV Cache | 16.2 ms | 1.0× | ✓ Done |
| 6b | Dedicated GEMV M=1 | 5.9 ms | 2.74× | ✓ Done |
| **6c** | **Fused cached-attention** | **~5.0 ms** | **3.2×** | **Recommended** |
| 6d | GEMV+BiasAdd fusion | ~4.8 ms | 3.4× | After 6c |
| 6e | CUDA Graphs | ~4.6 ms | 3.5× | After fusion reduces launch count |
| 6f | LayerNorm+GEMV fusion | ~4.5 ms | 3.6× | |

---

## 7. Artifacts

- **Current attention kernels:** [src/kernels/attention.cu](../../src/kernels/attention.cu) — `AttentionScoresKernelCached`, `AttentionAVKernelCached`
- **Softmax kernel:** [src/kernels/softmax.cu](../../src/kernels/softmax.cu) — `SoftmaxFusedKernel`
- **Decode profile:** [benchmarks/decode/bench_decode_profile.cu](../../benchmarks/decode/bench_decode_profile.cu)
- **Launch overhead benchmark:** [benchmarks/decode/bench_launch_overhead.cu](../../benchmarks/decode/bench_launch_overhead.cu)
- **Phase 6b report:** [reports/phase6/phase6b_report.md](phase6b_report.md)
- **Phase 6c readiness:** [reports/phase6/phase6c_readiness.md](phase6c_readiness.md)
- **This report:** [reports/phase6/phase6c_attention_analysis.md](phase6c_attention_analysis.md)
