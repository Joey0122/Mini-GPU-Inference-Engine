# Phase 6d Design Review: Fused Cached-Attention Kernel

**Date:** 2026-06-10
**Review scope:** Proposed fused kernel replacing `AttentionScoresKernelCached` + `SoftmaxFusedKernel` + `AttentionAVKernelCached`
**GPU:** RTX 4070 Laptop (AD106, sm_89, 36 SMs, 128 KB L1/SM, 32 MB L2, 256 GB/s peak BW, ~15.6 TFLOPS fp32)

---

## 1. Roofline Analysis

### 1.1 DRAM Traffic Per Decode Step (S=1024, 12 heads, head_dim=64)

| Access | Size (per layer) | Notes |
|--------|-----------------:|-------|
| Read q | 3,072 B | [n_head=12, head_dim=64] |
| Read k_cache | 3,145,728 B | [12, 1024, 64], one key per thread |
| Read v_cache | 3,145,728 B | Same layout, read in AV phase |
| Write out | 3,072 B | [12, head_dim] merged output |
| **Total DRAM** | **6,297,600 B ≈ 6.01 MB** | |
| Intermediate scores/weights | 0 B | Eliminated — stays in shared memory |

**Comparison to current 3-kernel path:**

| | Current (3 kernels) | Fused | Reduction |
|---|---|---|---|
| k_cache reads | 3.0 MB (Scores) | 3.0 MB | **0%** (same data, same read count) |
| v_cache reads | 3.0 MB (AV) | 3.0 MB | **0%** (same data, same read count) |
| scores buffer write | 48 KB → DRAM | 48 KB → smem | **100%** (eliminated from DRAM) |
| scores buffer read | 48 KB (Softmax) | 48 KB (smem read) | **0 KB DRAM** |
| weights buffer write | 48 KB → DRAM | 48 KB → smem (overwrites scores) | **100%** (eliminated from DRAM) |
| weights buffer read | 48 KB (AV) | 48 KB (smem read) | **0 KB DRAM** |
| **Total VRAM IO** | **~6.19 MB** | **~6.01 MB** | **~3% reduction** |

**Key insight:** The fusion doesn't reduce the dominant DRAM traffic (k_cache + v_cache = 6.0 MB). The savings come from (a) eliminating 2 kernel launches and pipeline drains per layer and (b) 16× more memory-level parallelism per SM. The intermediate buffer DRAM elimination is a minor contributor (~3%).

### 1.2 Arithmetic Intensity

```
QK^T FLOPs:  12 heads × 1,024 keys × 64 dims × 2 ops/FMA = 1,572,864 FLOPs
AV FLOPs:    12 heads × 1,024 keys × 64 dims × 2 ops/FMA = 1,572,864 FLOPs
Softmax:     12 × 1,024 × (1 max + 1 expf + 1 add + 1 div) — special functions, not FLOPs
───────────────────────────────────────────────────────────
Total FLOPs: 3,145,728 ≈ 3.15 MFLOPs per layer
```

```
AI = 3,145,728 FLOPs / 6,297,600 bytes = 0.50 FLOPs/byte
```

**Roofline ridge point:** 15.6 TFLOPS / 256 GB/s = **60.9 FLOPs/byte**

0.50 << 60.9: the kernel is **firmly memory-bound** by a factor of **122×**. Compute time is completely negligible (~0.0002 µs per layer). Performance depends entirely on how efficiently we stream k_cache and v_cache from DRAM.

### 1.3 GEMV Reference Point

Our existing `GemvKernel` provides a calibrated bandwidth-efficiency reference. The GEMV kernel reads from DRAM and has a similar strided access pattern:

| GEMV Shape | N | K | Bytes | µs | BW (GB/s) | % Peak |
|-----------|----:|----:|------:|----:|----------:|-------:|
| c_attn | 2,304 | 768 | ~7.0 MB | 22.0 | — | ~48%¹ |
| c_proj | 768 | 768 | ~2.4 MB | 21.9 | — | ~47%¹ |
| mlp_fc | 3,072 | 768 | ~9.4 MB | 21.3 | — | ~53%¹ |
| mlp_proj | 768 | 3,072 | ~9.4 MB | 73.7 | ~128 | **~50%** |
| lm_head | 50,257 | 768 | ~154 MB | 774.9 | ~199 | **~78%** |

¹ *Small-N shapes have launch overhead as a larger fraction; the BW numbers for c_attn/c_proj/mlp_fc understate true DRAM efficiency.*

The GEMV kernel achieves **48-80% peak BW** depending on problem size. The fused attention kernel has a different access pattern (contiguous regions per thread vs strided per-thread-dimension), but the GEMV results provide a practical ceiling calibrated to this GPU and codebase.

### 1.4 Access Pattern Analysis

**Critical design choice:** How to map threads to keys for the QK^T dot products. Two options, with different coalescing behavior:

#### Option A: Thread-per-key (1,024 threads/block)

Thread `t` reads key `t`'s 64 dimensions sequentially: `k_cache[h][t][0..63]`.

```
Thread 0: reads addresses [0..252]      (2 cache lines)
Thread 1: reads addresses [256..508]    (2 cache lines)
...
Warp 0 (threads 0..31): 32 unique address regions, each 256B apart
```

**Coalescing:** Within a warp, for iteration `d=0`, 32 threads access 32 distinct 128-byte cache lines. This requires 32 L1 transactions instead of 1-2 for fully coalesced. **However**, once a thread's first `d` iteration brings both its cache lines into L1, the next 31 iterations (`d=1..31`) hit L1. Then `d=32` misses again (new cache line boundary), and `d=33..63` hit.

Per thread: 4 L1 misses (d=0, 32) + 60 L1 hits. Per warp: `4 × 32 = 128` cache line loads to L1 = 16 KB from L2 per warp. Per block: `32 × 16 KB = 512 KB` loaded into L1 from L2 — exceeds L1 capacity (128 KB). L1 thrashes.

**But L2 (32 MB) catches nearly everything.** The 3 MB per-head k_cache fits in L2. The access pattern is a single contiguous 256 KB region per head. The DRAM → L2 path serves this well because:
- The region is physically contiguous
- DRAM row-buffer hits are high (sequential reads within 256 KB)
- L2 prefetchers detect the streaming pattern

**Effective BW estimate: 55-70% peak (141-179 GB/s).**

#### Option B: Thread-per-dimension with key-batching (256 threads/block, 4 blocks/head)

Thread `(d, g)` where `d ∈ [0,63]` is the dimension and `g ∈ [0,3]` is a key-group:
- Each block processes 256 keys
- Within a warp, all threads read the same key's different dimensions (fully coalesced)
- 4 blocks per head → need cross-block grid sync for softmax global max/sum

**Coalescing:** Excellent within each warp — identical to the current scores kernel pattern. All 32 threads in a warp read `kj[0..31]` from the same key row, then `kj[32..63]`, etc. Perfectly coalesced.

**Cost:** Two cooperative grid synchronizations (`grid.sync()`) per head for softmax global reductions.

**Effective BW estimate: 65-80% peak (166-205 GB/s).**

#### Chosen Approach: Option A (1,024 threads, 1 block/head)

Despite slightly lower expected BW efficiency, Option A is preferred because:
1. **No cross-block synchronization** — sm_89 `grid.sync()` requires `cudaLaunchCooperativeKernel`, which has higher launch overhead (~10 µs vs ~3.5 µs) and restricts kernel configurability
2. **Simpler implementation** — single block handles softmax internally via warp+block reductions (identical pattern to existing `SoftmaxFusedKernel`)
3. **12 blocks occupy 12 SMs** — no need to schedule 48 tiny blocks, which the hardware scheduler handles poorly
4. **The L2 cache absorbs the striding penalty** — at 55-70% BW efficiency, we still achieve 2.75-3.5× speedup

### 1.5 Performance Bounds

```
Theoretical minimum (100% peak BW):    6.01 MB / 256 GB/s  = 23.5 µs/layer → 282 µs (12 layers)
Practical ceiling (80% peak BW):       6.01 MB / 205 GB/s  = 29.3 µs/layer → 352 µs
Optimistic (70% peak BW):              6.01 MB / 179 GB/s  = 33.6 µs/layer → 403 µs
Expected (60% peak BW):                6.01 MB / 154 GB/s  = 39.0 µs/layer → 468 µs  + softmax ~65 µs
Conservative (50% peak BW):            6.01 MB / 128 GB/s  = 46.9 µs/layer → 563 µs  + softmax ~65 µs
Pessimistic (35% peak BW):             6.01 MB /  90 GB/s  = 66.8 µs/layer → 801 µs  + softmax ~65 µs
```

Adding fixed softmax cost (~5.5 µs/layer: expf + reductions) and launch overhead (~3.5 µs):

| Scenario | BW eff. | Per layer | 12 layers | vs Current | Savings |
|----------|--------:|----------:|----------:|:----------:|--------:|
| Ceiling | 80% | 29.3 + 5.5 + 3.5 = 38.3 | **460 µs** | 3.20× | 1,010 µs |
| Optimistic | 70% | 33.6 + 5.5 + 3.5 = 42.6 | **511 µs** | 2.88× | 959 µs |
| **Expected** | **60%** | **39.0 + 5.5 + 3.5 = 48.0** | **576 µs** | **2.55×** | **894 µs** |
| Conservative | 50% | 46.9 + 5.5 + 3.5 = 55.9 | **671 µs** | 2.19× | 799 µs |
| Pessimistic | 35% | 66.8 + 5.5 + 3.5 = 75.8 | **910 µs** | 1.62× | 560 µs |

**Even at a pessimistic 35% BW efficiency, the fused kernel saves ~560 µs (9.5% of decode).**

The expected case (576 µs) represents a **60% reduction** in attention latency from 1,470 µs, saving **894 µs (15.2% of decode).** Total decode drops from 5,900 µs to ~5,006 µs (5.0 ms), a cumulative 3.23× speedup from the Phase 6a baseline.

---

## 2. Register Pressure & Occupancy

### 2.1 AD106 (sm_89) Limits

| Resource | Per SM | Per GPU |
|----------|--------|---------|
| Max threads | 1,536 | 55,296 |
| Max registers | 65,536 | 2,359,296 |
| Max shared memory | 100 KB | 3,600 KB |
| Max thread blocks | 24 | 864 |
| Max threads per block | 1,024 | — |
| Warp size | 32 | — |

### 2.2 Register Budget Per Phase

With the 1,024-thread block design and scoped code blocks to encourage register reuse:

**Phase 1 — Q load (cooperative, short):**
```
q_row pointer:      2  (base + offset)
thread index:       1  (t = threadIdx.x)
loop bound + stride: 2  (head_dim, blockDim.x)
load temp:          1  (q_row[t] = ...)
Total: ~6 registers (all dead after __syncthreads)
```

**Phase 2 — QK^T scores:**
```
dot accumulator:    1
kh base pointer:    2  (k_cache + head offset)
kj pointer:         1  (kh + t * head_dim)
q_row pointer:      1  (smem base)
d loop variable:    1
head_dim:           1
scale:              1
key index t:        1  (reused from Phase 1)
Total: ~9 registers (dot survives, others dead after loop)
```

**Phase 3 — Softmax (warp + block reductions):**
```
max_val:            1  (reuse dot register space)
lane:               1
warp_id:            1
offset:             1
exp_val:            1  (replaces max_val after max found)
sum_val:            1
weight:             1  (result, survives to Phase 4)
normalize_temp:     1
Total: ~8 registers (weight survives)
```

**Phase 4 — AV weighted sum (D_TILE=8 tiled):**
```
acc[0..7]:          8  (tile accumulators)
v_tile_val:         1  (pipelined load, not live across loop)
vh base pointer:    2
out pointer:        2
tile loop var (d0): 1
weight:             1  (from Phase 3, live-in)
key index t:        1  (from Phase 2, live-in)
Total: ~16 registers
```

**Compiler live-range analysis:**

Across the entire kernel, the compiler sees:
- Phases 1, 2, 3, 4 are separated by `__syncthreads()` barriers
- After each barrier, only the values that cross phases must remain live
- Between Phase 2 and 3: `scores[t]` is in smem, so only `t` (key index) needs to stay in a register
- Between Phase 3 and 4: `weight` (the softmax-normalized value) stays live
- Variables from Phase 1 are dead after Phase 1's barrier
- Variables from Phase 2 (dot, kh, kj, q_row) are dead after Phase 2's barrier
- Variables from Phase 3 (max_val, exp_val, sum_val) are dead after Phase 4 begins

**Worst-case simultaneous live registers: ~20-25.**

Empirically, nvcc tends to over-allocate (no cross-phase reuse optimization) and uses 1.3-1.5× the minimal count. **Expected compiled register count: 30-38.**

### 2.3 Occupancy

| Metric | 1,024 thr/block, 38 regs | 1,024 thr/block, 48 regs | Limit |
|--------|------------------------:|------------------------:|------:|
| Registers used | 38,912 | 49,152 | 65,536 |
| % register capacity | 59% | 75% | — |
| Max blocks/SM (register-bound) | 1 | 1 | 24 |
| Max blocks/SM (thread-bound) | 1 | 1 | 1,536/1,024=1 |
| Max blocks/SM (smem-bound) | 16 | 16 | 100 KB/6 KB |
| **Effective blocks/SM** | **1** | **1** | — |
| SM thread utilization | 1,024/1,536 = **66.7%** | same | — |
| Total blocks (12 heads) | 12 | 12 | — |
| Active SMs | 12 of 36 | 12 of 36 | 36 |
| SM utilization | **33.3%** | **33.3%** | — |

SM utilization is capped at 33% because n_head=12 limits us to 12 independent blocks. Each block saturates one SM's thread capacity. This is a **hardware limitation** — there's no way to create more blocks without cross-block softmax (Option B) or inter-head work (which doesn't exist in attention).

### 2.4 Register Spilling Risk: LOW

At 30-38 registers expected vs 64 available at 1,024 threads/block:

- **Margin:** 26-34 registers (41-53% headroom)
- **Spilling threshold:** 64 registers at 1,024 threads/block
- **Typical spilling trigger patterns** (large arrays, many live `float4` variables) are absent
- **Mitigation:** The D_TILE=8 tiling in Phase 4 keeps accumulator arrays small (8 elements). Scoped blocks `{...}` around each phase help the compiler reclaim dead registers

**Risk of performance-degrading spilling:** very low. Even at 50 registers (well above expectation), we still fit 1 block/SM.

---

## 3. Alternative Comparison

### 3.1 Alternative A: Current 3-Kernel Pipeline (Baseline)

| Dimension | Assessment |
|-----------|-----------|
| **Latency** | 1,470 µs (measured) |
| **BW utilization** | ~20% peak (~50 GB/s effective) |
| **Why so low?** | 768 threads (24 warps) across 12 SMs = 2 warps/SM active. Memory latency (~300 cycles) requires ~6-8 warps/SM to hide. The pipeline starves on DRAM |
| **Implementation** | Done. 3 kernels: `AttentionScoresKernelCached` (66 lines), `SoftmaxFusedKernel` (35 lines in softmax.cu), `AttentionAVKernelCached` (36 lines). Total ~137 lines CUDA |
| **Maintenance** | Low. Clean separation: scores+softmax+AV. Each kernel independently testable and profileable. Scores/softmax kernels shared with prefill path |

### 3.2 Alternative B: Fused 1-Block-per-Head (Proposed)

| Dimension | Assessment |
|-----------|-----------|
| **Latency** | **460-671 µs** (expected 576 µs) — 2.2-3.2× faster |
| **BW utilization** | 50-70% peak (128-179 GB/s). Memory-level parallelism: 1,024 threads (32 warps) per block → 12,288 threads total → 32 warps per active SM |
| **Why better** | 16× more threads per head. 32 warps/SM vs 2 in current design. As each warp issues a memory load, the SM can cycle through 32 warps' ready instructions — easily covering 300-cycle DRAM latency. The L2 cache sees 12 independent 256 KB streaming reads (one per SM), which the 32 MB L2 handles without thrashing |
| **Implementation** | ~200 lines CUDA: 1 kernel (`FusedAttentionCachedKernel`), 1 launch wrapper. Reuses existing warp/block reduction patterns from `SoftmaxFusedKernel` |
| **Maintenance** | Medium. Single kernel with 4 internal phases. Harder to profile sub-components individually. Can add `#ifdef`-gated profiling counters. The 3-kernel path is preserved for prefill (S>1) so the regression surface is contained to decode only |

### 3.3 Alternative C: cuBLAS-Based Formulation

Use `cublasSgemmStridedBatched` to compute QK^T and AV as batched GEMVs:

```cpp
// Scores: [n_head, 1, head_dim] @ [n_head, head_dim, n_ctx] → [n_head, 1, n_ctx]
// Per-head GEMM: M=1, N=n_ctx, K=head_dim, batch=n_head
cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    n_ctx, 1, head_dim, &alpha,
    k_cache, n_ctx, head_dim*n_ctx,   // B^T: [head_dim, n_ctx] per head
    q,       head_dim, head_dim,       // A: [1, head_dim]
    &beta, scores, n_ctx, n_ctx);

// Softmax: custom kernel (cuBLAS has no softmax)

// AV: [n_head, 1, n_ctx] @ [n_head, n_ctx, head_dim] → [n_head, 1, head_dim]
cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    head_dim, 1, n_ctx, &alpha,
    v_cache, head_dim, head_dim*n_ctx,  // V: [n_ctx, head_dim] per head
    weights, n_ctx, n_ctx,              // W: [1, n_ctx]
    &beta, out, head_dim, head_dim);
```

| Dimension | Assessment |
|-----------|-----------|
| **Latency** | ~350-500 µs (projected). cuBLAS has hand-tuned GEMV for all problem sizes, using vectorized loads (float4/uint4), prefetch intrinsics, and warp-level assembly. For M=1 GEMV, cuBLAS is near-optimal |
| **BW utilization** | 70-85% peak. cuBLAS autotunes kernel selection and launch params at runtime based on problem dimensions |
| **Implementation** | ~30 lines: 2 cuBLAS calls + 1 custom softmax kernel. No new attention kernel code needed |
| **Maintenance** | Very low. cuBLAS is maintained by NVIDIA. No custom kernel to debug |
| **Violations** | **Violates PROJECT_SPEC** ("cuBLAS/cuDNN are forbidden as the implementation"). Also: ~1 ms of cuBLAS internal handle overhead on first call (can be amortized) |

### 3.4 Summary Comparison

| Criteria | A: 3-Kernel (current) | B: Fused (proposed) | C: cuBLAS |
|----------|:---------------------:|:-------------------:|:---------:|
| Latency (12 layers) | 1,470 µs | **576 µs** (460-671) | ~400 µs (350-500) |
| Speedup vs A | 1.0× | **2.55×** | ~3.7× |
| DRAM BW efficiency | ~20% | ~60% | ~75% |
| Lines of code | 137 (existing) | ~200 (new) | ~30 (new) + ~60 (softmax) |
| Custom kernel risk | None | Medium | Low |
| Profilability | Per-kernel (excellent) | Single kernel (poor) | Opaque (cuBLAS internal) |
| Prefill compatibility | ✓ Shared kernels | A alongside B (B is decode-only) | A alongside C |
| PROJECT_SPEC compliance | ✓ | ✓ | **✗ VIOLATION** |
| Cumulative decode | 5.90 ms | **~5.01 ms (3.23×)** | ~4.83 ms (3.35×) |

**Recommendation: Option B (fused).** Option C violates the project's "no cuBLAS" constraint and is excluded on that basis alone. Between A and B, the 2.55× attention speedup is the highest-ROI optimization remaining.

---

## 4. Correctness Risks

### 4.1 Numerical Stability of Fused Softmax

**Risk: LOW**

The fused softmax implements the standard numerically-stable algorithm:
1. `scores[t] = dot * scale`
2. `max = reduce_max(scores)` — subtract max before exp to prevent overflow
3. `exp_vals[t] = expf(scores[t] - max)`
4. `sum = reduce_sum(exp_vals)`
5. `weights[t] = exp_vals[t] / sum`

This is identical to the current `SoftmaxFusedKernel`. The only difference is that scores and weights reside in shared memory instead of global memory. The reduction operators (warp shuffle for max/sum) are bitwise-deterministic within a warp and consistent across blocks.

**expf precision:** The `expf` intrinsic is IEEE-754 compliant with ≤1 ULP error. At extreme values: `expf(-1e30f)` underflows to exactly `0.0f` (not a denormal). `expf(0.0f)` = exactly `1.0f`. `expf(88.7f)` = maximum finite float ~3.4e38; `expf(89.4f)` = +Inf. Our scores are bounded: `|dot * scale| ≤ 64 × max(|q|)·max(|k|) / sqrt(64)` ≈ `|8 × max_val²|`. For GPT-2 weights and LayerNorm-normalized inputs, `max_val` is typically < 2.0 (after LN), so `scores ∈ [-32, 32]`. After subtracting max, `expf(x-max) ∈ (0, 1]` — well within safe range.

**Subtractive cancellation:** Not applicable — softmax is monotonic and stable.

### 4.2 Reduction Ordering Differences

**Risk: LOW (acceptable tolerance)**

The current softmax uses:
1. `BlockReduceMax` — warp reduce → smem → block reduce via first warp
2. `BlockReduceSum` — same pattern

The fused kernel uses identical warp-level reductions. Within a warp, `__shfl_down_sync` is deterministic (same mask, same operands → same result). The block-level reduction in the current code uses `threadIdx.x < blockDim.x/32` (first warp reads warp partials). The fused kernel uses `warp_id == 0` (first warp reads warp partials). These are semantically identical.

However, **the mapping of keys to threads differs** from the current 3-kernel path. In the current path, `SoftmaxFusedKernel` receives scores computed by `AttentionScoresKernelCached`, which used a different grid (12 blocks × 64 threads). The reduction tree shape is different (64-thread block vs 1,024-thread block), so the floating-point summation order differs.

**Quantitative impact:** For N=1,024 floats summed in different tree topologies, the accumulated rounding error is at most `N × epsilon × max(|x_i|)` ≈ `1,024 × 1.2e-7 × 1.0` ≈ **1.2e-4** for the sum. The division `exp_val / sum` amplifies this to ~1e-4 relative error in the weights. The AV accumulation then weights each v[t,d] by a weight with ~1e-4 error, producing output with **~1e-4 relative error** compared to the current implementation.

This is well within the existing test tolerance (`GemmAbsTol = 1e-3` used in GEMM tests).

### 4.3 Determinism Concerns

**Risk: LOW for single-GPU inference**

- Within a single kernel launch: warp shuffles (`__shfl_down_sync(0xffffffffu, ...)`) are deterministic (all lanes active)
- Block reductions: `warp_id == 0` reads partials from smem — deterministic given fixed thread-to-data mapping
- **Non-deterministic:** The GPU scheduler chooses which warps/blocks execute when. But this affects timing, not results (the kernel has no atomic operations and no data-dependent control flow across threads)
- **Across runs:** Bitwise-deterministic on the same GPU with the same CUDA/driver version. Cross-GPU determinism not guaranteed (different reduction trees on different architectures)

### 4.4 Validation Strategy

**Proposed 5-gate validation plan:**

| Gate | Test | Method | Pass criterion |
|------|------|--------|---------------|
| **G1** | Fused vs current (single head, S=256) | Compare `FusedAttentionCached` output against `LaunchAttentionCached` output for all 12 heads | `max(|fused - current|) < 5e-4` |
| **G2** | Fused vs current (randomized S) | Test at S ∈ {64, 128, 256, 512, 1024} with random q, k, v | Same tolerance |
| **G3** | Forward-pass consistency | Run `LaunchGpt2DecodeStep` with fused attention path, compare logits to current path | `argmax(logits_fused) == argmax(logits_current)` |
| **G4** | Numerical edge cases | Test with all-zeros q, all-ones k/v, near-boundary expf inputs (±88.0 scores) | No NaN/Inf in output |
| **G5** | Existing test suite | Full 113-test suite must remain green | All pass |

Gate G3 is the critical correctness gate: if the fused kernel produces the same `argmax(logits)` as the current implementation, token generation is identical and all downstream gates (C8-C10) pass automatically.

### 4.5 Interaction with M==1 GEMV Dispatch

The fused kernel's output shape is identical to the current path: `out[n_head, head_dim]` = `[12, 64]` = 768 floats. This is the input to `c_proj` GEMV, which routes through the M==1 dispatch path. No changes needed to the GEMM interface or the transformer block orchestration.

---

## 5. Other Optimization Candidates (Re-Evaluated)

### 5.1 Quick Win: Scores Kernel 256-Thread Upgrade (No Fusion)

Before committing to full fusion, a minimal change: increase scores and AV kernel blocks from 64 to 256 threads (no architectural change).

```
Current:  Block(64)  → 2 warps/block, 12 blocks → 24 warps total
Proposed: Block(256) → 8 warps/block, 12 blocks → 96 warps total (4×)
```

| | Current (64-thread) | Upgraded (256-thread) | Fused |
|---|---|---|---|
| Lines changed | 0 | ~4 (constant in launch) | ~200 |
| Scores latency | 723 µs | ~280 µs (est.) | — |
| AV latency | 671 µs | ~260 µs (est.) | — |
| Attention total | 1,470 µs | ~620 µs (est.) | ~576 µs |
| Code risk | Zero | ~1 line | Medium |
| Speedup | 1.0× | **2.37×** | 2.55× |

**The 256-thread upgrade alone achieves 90% of the fused kernel's speedup with near-zero implementation risk.** It's literally changing `const int block = head_dim` to `const int block = 256` in `LaunchAttentionScoresCached` and `LaunchAttentionAVCached`.

The current kernel already handles arbitrary block sizes correctly (threads stride over keys with `j += blockDim.x`). With 256 threads: each thread handles 1,024/256 = 4 keys instead of 16. The only concern is shared memory for `q_row` — still 64 floats (256 bytes), unchanged.

**This is a compelling intermediate step.** It keeps the 3-kernel structure, all existing profiling, and all test infrastructure intact while delivering most of the fusion benefit.

### 5.2 Comparative ROI (Updated)

| Rank | Optimization | Savings | Risk | Effort | Recommendation |
|:----:|-------------|--------:|------|--------|---------------|
| **1** | **Scores+AV 256-thread** | **~850 µs (14.4%)** | **Zero** | **~2 lines** | **Do immediately** |
| **2** | **Full fusion** | **~894 µs (15.2%)** | **Medium** | **~200 lines** | **Phase 6d proper** |
| 3 | GEMV+BiasAdd | ~166 µs (2.8%) | Zero | ~80 lines | Phase 6e |
| 4 | CUDA Graphs | ~750 µs (12.7%) | Low | ~50 lines | Phase 6f |
| 5 | LayerNorm fusion | ~86 µs (1.5%) | Low | ~120 lines | Phase 6g |

### 5.3 Recommendation: Two-Step Phase 6d

**Step 1 (immediate, ~2 lines):** Increase scores and AV cached-attention block size from 64 to 256. This is a pure launch-configuration change — no kernel code changes needed. Expected savings: ~850 µs. Decode drops from 5.9 ms to ~5.05 ms.

**Step 2 (Phase 6d proper, ~200 lines):** Implement the fused kernel as designed above. Expected additional savings: ~44 µs from eliminating pipeline drains and intermediate DRAM buffers. Decode drops to ~5.01 ms.

The two-step approach provides:
1. **Immediate value delivery** — 90% of the benefit with near-zero risk
2. **Calibrated measurement** — the 256-thread upgrade provides a real-world BW efficiency datapoint that validates or corrects the fusion projections
3. **Incremental validation** — each step can be tested, benchmarked, and merged independently
4. **Fallback** — if fusion introduces subtle numerical issues, the 256-thread upgrade stands alone as a validated 2.37× attention speedup

---

## 6. Go/No-Go Recommendation

### Decision: **GO — Phase 6d, with two-step execution**

**Step 1: GO immediately.** The 256-thread block-size upgrade for scores and AV kernels is a trivial launch-configuration change with zero correctness risk (existing tests validate the path). It delivers ~850 µs savings (14.4% of decode, 2.37× attention speedup). This should be a single commit, benchmarked independently.

**Step 2: GO after Step 1 measurement.** The fused kernel is the natural next step. It eliminates the remaining softmax DRAM round-trip and the two inter-kernel pipeline drains. The expected additional 44 µs savings is modest after Step 1, but fusion also:
- Reduces attention launch count from 36 to 12 (eliminates 24 softmax launches)
- Keeps all intermediate data in shared memory, never touching DRAM
- Is architecturally required for future optimizations (e.g., FlashAttention-style tiling for prefill)
- Serves as the template for other kernel fusions (GEMV+BiasAdd, LN+GEMV)

**The 256-thread upgrade may make the fused kernel's incremental benefit small (44 µs).** If Step 1 measurement shows attention time ≤620 µs, the case for immediate fusion weakens and GEMV+BiasAdd fusion (166 µs) becomes the higher-ROI next step. The Step 1 measurement is critical for re-ranking.

### Decision Tree

```
Step 1: 256-thread upgrade
  ├─ Measure attention time
  │
  ├─ If attention ≤ 650 µs (savings ≥ 820 µs):
  │   └─ Proceed to GEMV+BiasAdd fusion (+166 µs), defer full attention fusion
  │
  └─ If attention > 650 µs (savings < 820 µs):
      └─ Proceed to full attention fusion (+44-150 µs additional)
```

---

## 7. Updated Roadmap

| Phase | Optimization | Decode Latency | Cumulative | Status |
|-------|-------------|---------------:|-----------:|--------|
| 6a | KV Cache | 16.2 ms | 1.00× | ✓ Done |
| 6b | Dedicated GEMV M=1 | 5.90 ms | 2.74× | ✓ Done |
| **6d-a** | **Scores+AV 256-thread upgrade** | **~5.00 ms** | **3.24×** | **GO (immediate)** |
| 6d-b | Full attention fusion | ~4.96 ms | 3.27× | GO (after measurement) |
| 6e | GEMV+BiasAdd fusion | ~4.80 ms | 3.38× | Next |
| 6f | CUDA Graphs | ~4.50 ms | 3.60× | After fusion cascade |
| 6g | LayerNorm+GEMV fusion | ~4.42 ms | 3.66× | |

The 256-thread upgrade alone matches or exceeds the CUDA Graphs projection (750 µs savings) with essentially zero implementation cost.

---

## 8. Artifacts

- **Current attention kernels:** [src/kernels/attention.cu](../../src/kernels/attention.cu)
- **Softmax (reference reduction pattern):** [src/kernels/softmax.cu](../../src/kernels/softmax.cu)
- **GEMV kernel (BW efficiency reference):** [src/kernels/gemm.cu](../../src/kernels/gemm.cu) L230-249
- **Decode profile:** [benchmarks/decode/bench_decode_profile.cu](../../benchmarks/decode/bench_decode_profile.cu)
- **Phase 6c analysis:** [reports/phase6/phase6c_attention_analysis.md](phase6c_attention_analysis.md)
- **This review:** [reports/phase6/phase6d_design_review.md](phase6d_design_review.md)
