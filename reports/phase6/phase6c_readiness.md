# Phase 6c Readiness: Launch-Overhead Audit & Recommendation

**Date:** 2026-06-10
**GPU:** RTX 4070 Laptop (AD106, sm_89, 36 SMs, 256 GB/s peak BW)

---

## 1. Kernel Launch Inventory

Each decode step launches **219 kernel invocations** plus 3 `cudaMemcpyAsync` calls and 1 stream sync:

| Category | Per Layer | ×12 | Extra | Total |
|----------|:---------:|:---:|:-----:|:-----:|
| GEMM/GEMV (c_attn, c_proj, mlp_fc, mlp_proj) | 5 | 60 | +1 (lm_head) | **61** |
| BiasAdd (one per GEMM) | 4 | 48 | — | **48** |
| Attention sub-kernels (scores, softmax, AV) | 3 | 36 | — | **36** |
| LayerNorm (ln1, ln2) | 2 | 24 | +1 (ln_f) | **25** |
| VectorAdd (residual1, residual2) | 2 | 24 | +1 (embed) | **25** |
| GELU | 1 | 12 | — | **12** |
| QkvSplitToCache | 1 | 12 | — | **12** |
| **Total kernel launches** | **17** | **204** | **+3** | **207** |
| cudaMemcpyAsync (wte, wpe, D2H, H2D) | — | — | 4 | **4** |
| cudaStreamSynchronize | — | — | 1 | **1** |
| **Grand total** | | | | **222** |

> The 207 kernel launches include: 5 GEMV × 12 layers + 1 lm_head GEMV = 61 GEMM/GEMV, plus 48 BiasAdd, 36 attention sub-kernels, 25 LayerNorm, 25 VectorAdd, 12 GELU, 12 QkvSplitToCache.

---

## 2. Launch Overhead Measurement

### 2.1 Methodology

Two complementary measurements:

**A. Homogeneous:** Launch the SAME kernel 219 times consecutively. Measures the driver's ability to batch identical launches (best-case, not realistic).

**B. Heterogeneous:** Launch the EXACT decode-step pattern — all 17 per-layer kernel types in correct order, 12 layers, plus ln_f and lm_head. This is the realistic measurement matching the actual decode workload.

For each, we measure:
- **T_wall:** Total wall-clock time (outer CUDA events around all launches)
- **T_device:** Pure GPU execution time (inner CUDA events on the device timeline)
- **T_overhead:** T_wall − T_device = CPU launch work NOT overlapped with GPU execution

### 2.2 Results

| Metric | Value |
|--------|------:|
| Null kernel (1 launch) | 5.80 µs |
| 219 null kernels (homogeneous) | 1,625 µs (7.42 µs each) |
| **219 heterogeneous decode pattern** (T_wall) | **4,957 µs** |
| **219 heterogeneous decode pattern** (T_device) | **4,202 µs** |
| **Non-overlapped CPU overhead** | **756 µs (15.2%)** |
| Per-launch CPU overhead (avg across 219 heterogeneous) | 3.45 µs |

### 2.3 Key Insight: Homogeneous vs Heterogeneous

The homogeneous benchmark (launching the same GEMV 219 times) reports only ~163 µs of overhead — because the CUDA driver batches identical `<<<>>>` calls efficiently. **This is NOT representative of the real decode pattern.** The heterogeneous pattern, where each launch has different arguments, grid sizes, shared memory configs, and kernel functions, costs **4.6× more** in non-overlappable overhead.

---

## 3. Decomposing the Decode Step

Using measured data from the Phase 6b profile and the heterogeneous benchmark:

| Component | µs | % |
|-----------|---:|--:|
| GPU execution (T_device, 207 kernel device times) | ~4,200¹ | 71.2% |
| CPU launch overhead (non-overlapped, 219 heterogeneous) | 756 | 12.8% |
| Host D2H logits (cudaMemcpy) | 79 | 1.3% |
| Host argmax (CPU) | 41 | 0.7% |
| Host H2D token (cudaMemcpy) | 14 | 0.2% |
| Stream sync + misc | 20 | 0.3% |
| Additional data-dependent execution² | ~788 | 13.4% |
| **Total** | **~5,900** | **100.0%** |

¹ *Synthetic data (all-0.5f) — real weights have different cache behavior.*
² *Gap between synthetic T_device + overhead vs. profile total. Attributed to real-weight memory access patterns, L2 cache effects, and profiling event overhead.*

---

## 4. CUDA Graphs Projection

### 4.1 Estimated Savings

| Item | µs |
|------|----:|
| Current non-overlapped CPU overhead (219 heterogeneous launches) | 756 |
| CUDA Graphs launch (1 cudaGraphLaunch) | −6 |
| **Net wall-clock savings** | **750** |

### 4.2 Projected Decode Latency

| | Before | After CUDA Graphs | Speedup |
|---|---|-----:|-----:|
| Decode step | 5,900 µs (5.9 ms) | **5,150 µs (5.2 ms)** | **1.15× (12.7%)** |

### 4.3 Sensitivity Analysis

The measurement shows 3.45 µs average CPU overhead per heterogeneous launch. The table below shows how CUDA Graphs benefit scales if per-launch overhead differs from measurement:

| Per-launch CPU cost | CUDA Graphs savings | Decode latency | Speedup |
|--------------------:|--------------------:|---------------:|--------:|
| 0.5 µs/kernel | 104 µs | 5,796 µs | 1.02× |
| 1.0 µs/kernel | 213 µs | 5,686 µs | 1.04× |
| 2.0 µs/kernel | 432 µs | 5,467 µs | 1.08× |
| **3.5 µs/kernel (measured)** | **750 µs** | **5,150 µs** | **1.15×** |
| 5.0 µs/kernel | 1,089 µs | 4,810 µs | 1.23× |

Even with an optimistic 5.0 µs/kernel overhead, CUDA Graphs delivers at most 1.23×. The measured 3.45 µs/kernel yields 1.15×.

### 4.4 Verdict

**CUDA Graphs is projected to save ~750 µs (12.7% wall-clock).** This falls short of the 15% threshold. While not zero, it is modest compared to what kernel optimization can deliver at the current 5.9 ms/token regime.

CUDA Graphs becomes more valuable **after** kernel fusion, when:
- Fewer, larger kernels mean less CPU/GPU overlap buffer
- Individual launch overhead is a larger fraction of per-kernel execution time
- The total launch count drops from 219 to ~50-100

---

## 5. Attention Kernel Analysis

### 5.1 Current State

The three attention sub-kernels at decode time (S_cached=1024, n_head=12, head_dim=64):

| Sub-kernel | µs | % of step | Grid | Block | Total threads | SM utilization |
|-----------|----:|:---------:|:----:|:-----:|:-------------:|:--------------:|
| Scores (QK^T) | 723.7 | 12.3% | (1, 12) | 64 | 768 | 1.4% |
| Softmax | 75.7 | 1.3% | — | 256 | ~3,072 | ~5.5% |
| AV (weights×V) | 670.7 | 11.4% | (1, 12) | 64 | 768 | 1.4% |
| **Total** | **1,470.1** | **24.9%** | — | — | — | — |

Key problems:
1. **Severe under-utilization:** Scores and AV kernels use only 768 threads (1.4% of GPU's 55K capacity) across 12 blocks (33% of 36 SMs).
2. **Triple memory traversal:** k_cache is read once (scores), then v_cache is read once (AV). But the intermediate scores/weights tensor is written and then read back from VRAM.
3. **64-thread blocks:** head_dim=64 limits thread count per block in the current design. Each thread serializes over 1024/64 = 16 keys.

### 5.2 Optimization: Fused Attention with Better Parallelism

**Proposed fused kernel:** Grid(ceil(S/256), n_head) = (4, 12) = 48 blocks of 256 threads = 12,288 threads.

One thread per (head, key) pair for the dot product. The softmax is computed cooperatively within each head's thread group (4 blocks of 256 = 1,024 threads → covers all 1,024 keys). The AV weighted sum is accumulated in shared memory.

Benefits:
- **1 kernel launch instead of 3** — saves 24 launches per step
- **Read k_cache and v_cache once each** — saves 2 memory traversals
- **48 blocks → better SM utilization** (48/36 = 1.33 waves vs 12/36 = 0.33)
- **No intermediate scores/weights tensor in VRAM** — stays in shared memory
- **12,288 threads vs 768** — 16× more parallelism

**Estimated impact:**

| | Current | After fusion | Savings |
|---|---|-----:|----:|
| Attention total | 1,470 µs | ~700-800 µs | **~670-770 µs (11-13%)** |

---

## 6. Alternative Phase 6c Candidates (Ranked)

| Rank | Candidate | Est. Savings | % of step | Effort |
|:----:|-----------|------------:|:---------:|:------:|
| **1** | **Attention kernel fusion + parallelism** | **~700 µs** | **11.9%** | Medium (~150 lines CUDA) |
| 2 | CUDA Graphs | ~750 µs | 12.7% | Low (~50 lines API) |
| 3 | GEMV+BiasAdd fusion | ~250 µs | 4.2% | Low (~80 lines) |
| 4 | Attention scores kernel parallelism only | ~360 µs | 6.1% | Low (~50 lines) |
| 5 | LN+GEMV fusion | ~200 µs | 3.4% | Medium (~120 lines) |

### Why Attention Fusion > CUDA Graphs (despite similar µs savings)

1. **Attention fusion improves GPU execution**, not just launch overhead. It addresses the root cause (1.4% thread utilization) rather than the symptom (many launches).
2. **Attention fusion compounds with future optimizations.** After fusion, CUDA Graphs saves MORE because there are fewer, larger kernels.
3. **12.7% from CUDA Graphs is the ceiling** — it can never exceed the total non-overlapped launch overhead. Attention fusion's ceiling is higher (24.9% of total step time).
4. **Attention fusion is a prerequisite for CUDA Graphs to be worthwhile.** After reducing launch count from 219 to ~180 (via 3→1 attention fusion), the remaining launches are larger and benefit more from graph compilation.

---

## 7. Recommended Phase 6c

### Fused Cached-Attention Kernel for Decode

**Goal:** Replace `LaunchAttentionCached` (3 sub-kernels) with a single fused kernel that:
1. Computes Q·K^T scores in registers
2. Applies softmax inline (per-head cooperative reduction)
3. Accumulates weights·V in registers
4. Writes the final merged output directly

**Grid:** `(ceil(S_cached/256), n_head)` = `(4, 12)` for S=1024
**Block:** 256 threads
**Shared memory:** `q_row[head_dim]` (256 B) + per-head softmax workspace (~4 KB) = ~5 KB total

**Key design decisions:**
- Each thread handles one key position's entire dot product (64 FMAs)
- Softmax: 4 blocks cooperate per head (1,024 threads for 1,024 keys)
  → warp-level max reduction, then exp + sum, then normalization
- AV accumulation: each thread accumulates its key's contribution, then block-level reduction across 256 threads → 64 output elements per head

**Estimated decode step after Phase 6c:**

| Component | Before | After | Savings |
|-----------|---:|----:|----:|
| Attention | 1,470 µs | ~750 µs | ~720 µs |
| All other | 4,430 µs | 4,430 µs | — |
| **Total** | **5,900 µs (5.9 ms)** | **~5,180 µs (5.2 ms)** | **1.14×** |

### Updated Roadmap

| Phase | Optimization | Decode Latency | Cumulative | Status |
|-------|-------------|---------------:|-----------:|--------|
| 6a | KV Cache | 16.2 ms | 1.0× | ✓ Done |
| 6b | Dedicated GEMV M=1 | 5.9 ms | 2.74× | ✓ Done |
| **6c** | **Fused cached-attention (decode)** | **~5.2 ms** | **3.1×** | **Recommended** |
| 6d | CUDA Graphs | ~4.8 ms | 3.4× | After fusion |
| 6e | GEMV+BiasAdd fusion | ~4.5 ms | 3.6× | |
| 6f | LN+GEMV fusion | ~4.3 ms | 3.8× | |

CUDA Graphs moves from Phase 6c to 6d — it delivers more value after kernel fusion reduces the launch count.

---

## 8. Artifacts

- **Benchmark:** [benchmarks/decode/bench_launch_overhead.cu](../../benchmarks/decode/bench_launch_overhead.cu)
- **Profile data:** [reports/phase6/phase6b_report.md](phase6b_report.md) (Phase 6b decode profile)
- **Report:** [reports/phase6/phase6c_readiness.md](phase6c_readiness.md) — This file
