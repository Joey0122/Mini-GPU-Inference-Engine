# Phase 5 Final Report — GPT-2 Inference Engine

**Date:** 2026-06-10
**Scope:** Commit C1 through C10 — complete GPT-2 Small (124M) inference pipeline
**Status:** COMPLETE

---

## 1. Architecture Summary

Phase 5 implements a full GPT-2 Small (124M) autoregressive inference engine on a single
CUDA stream. The architecture follows a clean layered design:

```
┌─────────────────────────────────────────┐
│  decode.cu  (C9/C10)                    │
│  autoregressive loop: forward → sample  │
├─────────────────────────────────────────┤
│  forward.cu (C8)                        │
│  embedding → 12×block → ln_f → lm_head  │
├─────────────────────────────────────────┤
│  transformer_block.cu (C7)              │
│  ln→c_attn→QKV→attn→c_proj→res→        │
│  ln→mlp_fc→GELU→mlp_proj→res           │
├─────────────────────────────────────────┤
│  Kernels (C4-C6, Phases 1-4)           │
│  embedding, bias_add, qkv_split,        │
│  layernorm, gemm, softmax, gelu,        │
│  attention(scores/AV), vector_add       │
├─────────────────────────────────────────┤
│  workspace.cu (C8)                      │
│  pre-allocated scratch, zero mallocs    │
├─────────────────────────────────────────┤
│  loader.cpp (C2-C3)                     │
│  manifest + flat binary → DeviceBuffers │
└─────────────────────────────────────────┘
```

### Design Principles

1. **Naive-first kernels** — correctness before optimization
2. **Pre-allocated workspace** — zero `cudaMalloc`/`cudaFree` in forward/decode hot paths
3. **Single CUDA stream** — all ops sequential, no concurrency bugs
4. **RAII memory ownership** — `DeviceBuffer<T>` move-only, no raw pointer leaks
5. **Host-side sampling** — argmax/softmax/categorical on CPU (200 KB D2H copy per step)
6. **HF Conv1D convention** — weights stored as `[in_features, out_features] = [K, N]`, directly compatible with `LaunchGemm(A[M,K], B[K,N], C[M,N])` without transpose

---

## 2. Implemented Components

| Commit | Component | Files | What It Does |
|--------|-----------|-------|--------------|
| C1 | Config | `include/model/gpt2_config.hpp` | GPT-2 Small architectural constants |
| C2 | Export | `scripts/export_gpt2.py` | HF GPT-2 → flat binary + manifest |
| C3 | Loader | `src/model/loader.cpp` | Manifest + binary → `Gpt2Weights` (RAII) |
| C4 | Embedding | `src/kernels/embedding.cu` | `wte[ids] + wpe` gather+add |
| C5 | BiasAdd | `src/kernels/bias_add.cu` | Row-wise bias broadcast, 1D grid |
| C6 | QKV Split | `src/kernels/qkv_split.cu` | `[S,3D]` → head-major `[H,S,d]` × 3 |
| C7 | Transformer Block | `src/model/transformer_block.cu` | Pre-LN block: 11 kernel launches in order |
| C8 | Forward Pass | `src/model/forward.cu` | Full pipeline: embed→12×block→ln_f→lm_head |
| C8 | Workspace | `src/model/workspace.cu` | Pre-allocated scratch buffers |
| C9 | Greedy Decode | `src/model/decode.cu` | Autoregressive loop with argmax |
| C10 | Temperature Sample | `src/model/decode.cu` | Softmax + categorical sampling (Mersenne Twister) |

### Key Kernel Design Decisions

| Kernel | Grid | Block | Strided Loop | Rationale |
|--------|------|-------|-------------|-----------|
| Embedding | `[S]` | 256 | over `n_embd` | Naive gather, ~3 iter at D=768 |
| BiasAdd | `[N]` | 256 | over D | Row-parallel, bias in constant cache |
| QKV Split | `[n_head, S]` | head_dim | none (1 iter at d=64) | Exact coverage, head-major output |
| Attention Scores | `[n_head, S]` | 256 | over S | Causal mask pre-applied |
| Attention AV | `[n_head, S]` | 64 | over head_dim | Heads merged in output |

---

## 3. Memory Footprint

### Workspace (S = 1024, GPT-2 Small)

| Buffer | Shape | Elements | Size |
|--------|-------|----------|------|
| `hidden` | [S, D] | 786,432 | 3.15 MB |
| `logits` | [S, V] | 51,463,168 | 205.85 MB |
| `ln1_out` | [S, D] | 786,432 | 3.15 MB |
| `c_attn_out` | [S, 3D] | 2,359,296 | 9.44 MB |
| `q` | [H, S, d] | 786,432 | 3.15 MB |
| `k` | [H, S, d] | 786,432 | 3.15 MB |
| `v` | [H, S, d] | 786,432 | 3.15 MB |
| `scores` | [H, S, S] | 12,582,912 | **50.33 MB** |
| `weights` | [H, S, S] | 12,582,912 | **50.33 MB** |
| `attn_out` | [S, D] | 786,432 | 3.15 MB |
| `proj_out` | [S, D] | 786,432 | 3.15 MB |
| `ln2_out` | [S, D] | 786,432 | 3.15 MB |
| `fc_out` | [S, M] | 3,145,728 | 12.58 MB |
| **Total workspace** | | | **~354 MB** |
| **Weights** (124M × 4B) | | | **~496 MB** |
| **Total GPU memory** | | | **~854 MB** |

Fits comfortably in 8 GB VRAM. The dominant consumers are `scores`/`weights` (O(S²) attention, 100 MB combined) and `logits` (O(S×V), 206 MB). Workspace is allocated once at startup for `S_max = prompt_len + max_new_tokens`.

### Per-decode-step D2H/H2D Transfer

| Transfer | Direction | Size |
|----------|-----------|------|
| Logit row | D2H | 200 KB (50,257 × 4B) |
| Next token | H2D | 4 B |

Total per step: ~200 KB — negligible.

---

## 4. Validation Coverage

### 4.1 Test Infrastructure

Three-level validation hierarchy:

| Level | What | Gate | Scope |
|-------|------|------|-------|
| **CPU reference** | `CpuKernel()` vs GPU | Per-op FP32 tolerance | All kernels, in-process GTest |
| **PyTorch reference** | `.npy` files → `validate_runner.cu` | FP32 `max_abs_error < 1e-4` | Per-kernel, file-based |
| **HF golden** | `golden_test.py` vs HuggingFace GPT-2 | Exact argmax/token match | End-to-end, real weights |

### 4.2 Test Suite (117/117 passing)

| Category | Tests | Description |
|----------|-------|-------------|
| VectorAdd | 2 | Phase 0 |
| Reduction | 8 | Phase 1 |
| GEMM | 10 | Phase 2 |
| LayerNorm | 5 | Phase 3 |
| Softmax | 7 | Phase 3 |
| GELU | 3 | Phase 3 |
| Attention | 7 | Phase 4 |
| Embedding | 5 | C4 |
| BiasAdd | 7 | C5 |
| QKV Split | 6 | C6 |
| Transformer Block | 12 | C7 |
| Forward | 3 | C8 |
| Decode | 6 | C9/C10 |
| Loader | 4 | C3 |
| CPU Reference | 17 | Phases 0-4 |
| Validation Runner | 1 | All (42 PyTorch cases) |

### 4.3 Golden Validation (ALL GATES PASS)

| Gate | Cases | Result |
|------|-------|--------|
| **C8 Logit Gate** | 53/53 argmax matches | PASS (worst max_abs_error = 2.59e-4, gate: 5e-3) |
| **C9 Decode Gate** | 113/113 token matches | PASS (4 prompts × 25+ steps each) |
| **C10 Sample Gate** | 0/60 implausible samples | PASS (distribution-level, 3 prompts × 20 steps) |

Golden prompts: S=1, 4, 16, 32 (logits); up to 33+ tokens (decode).

### 4.4 PyTorch Kernel Validation (42/42 cases)

| Kernel | Versions | Shapes | Worst max_abs_error | Margin |
|--------|----------|--------|---------------------|--------|
| LayerNorm | Naive, Welford | 6 cases | 9.5e-7 | 105× |
| Softmax | Naive, Fused | 4 cases | 6.0e-7 | 167× |
| GELU | NewGELU | 2 cases | 2.4e-7 | 417× |
| Attention | Integrated | 4 cases | 1.8e-6 | 54× |
| Embedding | Naive | 4 cases | 0.0 | ∞ |
| BiasAdd | Naive | 5 cases | 0.0 | ∞ |
| QKV Split | q/k/v | 15 cases | 0.0 | ∞ |
| Transformer Block | Full block | 2 cases | 3.9e-6 | 26× |
| **Gate** | | | **1e-4** | **26× minimum** |

Embedding, BiasAdd, and QKV Split achieve exact zero error because they involve
only gathers/adds with no fused multiply-add accumulation — each output element is
a single FMUL/FADD operation, producing bit-exact results between PyTorch and CUDA.

---

## 5. Benchmark Results

### Full Forward Pass (RTX 4070 Laptop, FP32)

| S | Latency (ms) | Notes |
|--:|-------------:|-------|
| 1 | 15.1 | Overhead-dominated (kernel launch + small GEMMs) |
| 16 | 15.7 | Near-constant up to S=64 |
| 64 | 16.3 | |
| 128 | 18.2 | Attention O(S²) starts to become visible |
| 256 | 26.8 | |
| 512 | 53.5 | |
| 1024 | 142.5 | O(S²) attention dominates |

Key observation: S=1 latency is 15.1 ms for a single-token forward pass. With KV
cache (Phase 6), per-step decode latency should drop from ~15 ms to well under
1 ms, enabling ~1000+ tokens/sec generation.

### Subsystem Benchmarks (representative)

| Kernel | Configuration | Effective BW (GB/s) | % Peak |
|--------|--------------|---------------------|--------|
| Embedding | S=1024, D=768 | 931* | — |
| BiasAdd | S=1024, D=768 | 255 | 100% |
| QKV Split | S=1024, H=12 | 250 | 98% |
| LayerNorm Naive | 256×768 | 199 | 78% |
| Softmax Fused | 64×4096 | 237 | 93% |
| GELU | 64×3072 | 230 | 90% |
| GEMM RegTiled128Pad | 4096³ | — | 60.5% cuBLAS |
| Attention (integrated) | S=1024 | 35.2 | compute-bound |

*Embedding bandwidth at S=1024 exceeds DRAM peak (256 GB/s) because the 9.4 MB
working set fits entirely in L2 cache. The metric `BandwidthGBs` measures total
data through the memory hierarchy, not DRAM throughput.

---

## 6. Architecture Review Findings

### 6.1 Correctness — PASS

- **Pre-LN block order:** Verified. `ln1→c_attn→QKVsplit→attention→c_proj→+residual→ln2→mlp_fc→GELU→mlp_proj→+residual`. Matches HF GPT-2 exactly.
- **Forward pass pipeline:** Verified. `embedding→12×block→ln_f→lm_head`. Single stream, sequential.
- **Conv1D/GEMM orientation:** All weight matrices stored as `[K,N] = [in_features, out_features]`, directly compatible with `LaunchGemm(A[M,K], B[K,N], C[M,N])` without transpose. Verified by 4 dedicated orientation tests.
- **QKV→attention layout:** QKV split produces head-major `[H,S,d]` layout matching `AttentionScoresKernel` indexing `q[h*S+s]*head_dim + d]`. Verified by deterministic layout test.
- **WTE transpose:** `wte_T` stored separately as `[D,V]` for LM head GEMM. Verified by `WteTTransposeIsCorrect` loader test.
- **Golden validation:** 53/53 argmax + 113/113 tokens + 0/60 implausible samples. Authoritative.

### 6.2 Memory Ownership — PASS

- `DeviceBuffer<T>`: move-only RAII, `cudaFree` in destructor, zero raw pointers retained.
- `Gpt2Weights`: owns all weight tensors. Loaded once at startup, reused for lifetime.
- `Gpt2Workspace`: owns all scratch buffers. Created once for S_max, reused across forward calls.
- No `cudaMalloc`/`cudaFree` in `LaunchGpt2Forward`, `LaunchTransformerBlock`, or `LaunchSamplingDecode`.
- `loader.cpp` `Upload()` helper: clear ownership transfer (host vector → DeviceBuffer).

### 6.3 Workspace Lifetime — PASS

- Created: `Gpt2Workspace::Create(S_max, cfg)` before decode/forward loop.
- Reused: same workspace passed to every `LaunchTransformerBlock` call (12 layers, sequential).
- `proj_out` reused for both c_proj and mlp_proj outputs — safe because they execute sequentially.
- Destroyed: when `Gpt2Workspace` goes out of scope (RAII).
- No use-after-free: all kernel launches complete before workspace is reused in next layer.

### 6.4 Unnecessary Synchronizations — PASS

- `decode.cu`: `cudaStreamSynchronize(stream)` after D2H logit copy — **necessary**: host-side sampling requires logits on CPU before selecting next token.
- `forward.cu`, `transformer_block.cu`: no synchronizations. All kernels on same stream → implicit ordering.
- Tests: `CUDA_CHECK_LAST_SYNC()` used appropriately — catches both launch and execution errors at test boundaries. Not used in production paths.
- No `cudaDeviceSynchronize()` in any hot path.

### 6.5 Hidden O(S²) Allocations — PASS (Bounded)

- `scores` and `weights` buffers: `H × S × S × 4B` = 50.33 MB each at S=1024. These are the classic attention O(S²) memory cost, pre-allocated in workspace.
- `logits` buffer: `S × V × 4B` = 205.85 MB at S=1024. Pre-allocated.
- All buffers sized at creation time for S_max. No reallocation during operation.
- S is capped at `n_ctx = 1024`, so worst-case memory is bounded.
- **Phase 6 KV cache will add** ~12 × 2 × 1024 × 768 × 4B = 75.5 MB (K+V cache for all layers).

### 6.6 API Consistency — PASS

- All `Launch*` functions: `const` inputs first, outputs next, config/context params, `cudaStream_t stream` last with default `nullptr`.
- Consistent namespace: `mgie::`
- Consistent error handling: `CUDA_CHECK` for all runtime calls, `CUDA_CHECK_LAST` after kernel launches.
- `LaunchGreedyDecode` is a thin wrapper calling `LaunchSamplingDecode(temp=0)` — clean composition.
- Return conventions: decode functions return `int` (tokens generated), forward/block return `void`.

### 6.7 Error Handling — PASS

- `CUDA_CHECK(cudaMalloc/Free/Memcpy/StreamSync)`: all wrapped.
- `loader.cpp`: validates every tensor shape against config, throws `std::runtime_error` on mismatch.
- `embedding.cu`: bounds-checks token IDs with `safe_tid` clamp (defense-in-depth).
- `decode.cu`: clamps negative temperature to 0, caps `gen_len` to `n_ctx`.
- `forward.cu`, `transformer_block.cu`: early return on `S <= 0`.
- `DeviceBuffer`: constructor with `count > 0` check. Move operations null out source.
- No `cudaFree` in destructor wrapped in try-catch (best-effort, as documented).

### 6.8 CUDA Stream Correctness — PASS

- Single stream throughout (default stream `nullptr` or caller-provided).
- `forward.cu`: all 4 pipeline stages on same stream → sequential execution guaranteed.
- `transformer_block.cu`: all 11 kernel launches + bias adds on same stream.
- `decode.cu`: forward and memcpy ops on same stream, `cudaStreamSynchronize` before host-side sampling, then next iteration.
- No concurrent streams, no inter-stream dependencies to manage.
- Correct for the single-stream model.

---

## 7. Audit Recommendation Revisit

### R1: Longer-Sequence Golden Validation (S=128, S=256, S=512)

**Classification: SHOULD FIX**

The C8 golden validation covers S=1, 4, 16, 32 — all short. The synthetic CPU
reference extends block-level testing to S=1024, and per-kernel PyTorch validation
covers all expected shapes. However, end-to-end HF comparison at S=128 would
provide stronger confidence for the full context range.

**Justification:** Not a MUST because:
1. The forward pass is a purely sequential chain — longer S uses the same code path
   with strided loops; there is no branching on sequence length.
2. All individual kernels are validated at their target S ranges (up to S=1024).
3. The golden test already proves end-to-end correctness with real weights.
4. Adding S=128 would require HF to generate reference outputs, which is compute-heavy
   but straightforward.

**Recommended Phase 6 action:** Add one S=128 golden prompt during Phase 6 validation
pass. Estimated effort: ~30 minutes.

### R2: Benchmark Bandwidth Naming/Documentation

**Classification: SHOULD FIX**

The embedding benchmark reports 930.9 GB/s at S=1024, which exceeds the RTX 4070
Laptop's ~256 GB/s DRAM bandwidth. The 9.4 MB working set fits in L2 cache;
`BandwidthGBs` measures total data through the hierarchy, not DRAM throughput.

**Justification:** Not a MUST because:
1. This is a measurement artifact, not a bug.
2. The forward benchmark already handles this correctly (sets 0.0 gflops/bandwidth
   for composite ops).
3. Only affects benchmark CSV output, not functionality.

**Recommended Phase 6 action:** Rename embedding benchmark metric to
`effective_bandwidth_gbs` and add a comment documenting the L2 cache effect.
Alternatively, flush L2 between iterations for accurate DRAM measurement.
Estimated effort: ~15 minutes.

### R3: SyntheticGreedyDecodeProducesValidTokens Misleading Name

**Classification: DEFER TO PHASE 6**

The test name suggests a CPU-reference comparison, but the test only validates
that generated tokens fall within `[0, V)`. There is no CPU reference for
synthetic decode. The original intent (diversity check) was removed when random
weights produced degenerate identical tokens.

**Justification:** DEFER because:
1. The test passes and validates token range correctness.
2. The real-weight golden test (C9 Decode Gate) is the authoritative gate.
3. A CPU reference for synthetic decode would require a full C++ autoregressive
   loop implementation — non-trivial effort.
4. No correctness risk from the misleading name.

**Recommended Phase 6 action:** Either rename to `SyntheticGreedyDecodeProducesTokensInRange`
or add a CPU reference for synthetic decode. Estimated effort: ~5 minutes (rename)
or ~1 hour (CPU reference).

### R4: wte_T Integrity Check in Loader

**Classification: SHOULD FIX**

The loader reads `wte_T` as a separate tensor from the export script. If the
export script had a bug causing `wte_T ≠ transpose(wte)`, the LM head GEMM
would produce incorrect logits. The C8 golden gate would catch this (argmax
mismatch with HF), but the error message would be opaque.

**Justification:** Not a MUST because:
1. The C8 golden test already catches this (and passes).
2. The loader test `WteTTransposeIsCorrect` validates the transpose relationship.
3. An in-loader check would provide a clearer error message if the invariant
   were ever violated.

**Recommended Phase 6 action:** Add optional `#ifndef NDEBUG` check in loader
comparing `wte_T` against transpose of `wte`. Estimated effort: ~30 minutes.

---

## 8. Known Limitations

### 8.1 Attention O(S²) Memory
At S=1024, `scores` and `weights` consume 100 MB combined. This is pre-allocated
and bounded, but limits scalability beyond n_ctx=1024. FlashAttention-style tiling
(Phase 6 optimization item) would reduce this to O(S).

### 8.2 No KV Cache
Every decode step recomputes the full forward pass over the entire growing sequence
(S=1, 2, 3, ...). This makes generation O(S²) in total compute. KV cache (Phase 6)
will cache K,V per layer, reducing per-step work to O(S) attention over cached
keys/values.

### 8.3 S=1 Decode Latency
Single-token forward pass latency is 15.1 ms due to kernel launch overhead and
small-GEMM inefficiency. KV cache + kernel fusion (Phase 6) should reduce this
to <1 ms.

### 8.4 Golden Validation Coverage Limit
HF golden validation covers S ≤ 32 for logits and up to 33 tokens for decoding.
No HF comparison at S=128, 256, 512. See R1.

### 8.5 No CUDA Graphs
The decode loop launches ~120+ kernels per step (4 pipeline stages + 12 blocks × 11 ops).
CUDA graphs (Phase 6) would amortize launch overhead.

### 8.6 Host-Side Sampling
Softmax + categorical sampling on CPU requires a 200 KB D2H copy and
`cudaStreamSynchronize` per step. A GPU sampling kernel would eliminate the
sync, but the benefit is small (host-side softmax over 50K elements is ~10 µs).

### 8.7 No FP16/BF16 Support
All computation is FP32. Mixed precision would reduce memory bandwidth pressure
and enable larger batch sizes, but is out of scope.

---

## 9. Proposed Phase 6 Roadmap

### 9.1 Phase 6a: KV Cache

The highest-impact optimization. Cache K and V tensors per layer across decode
steps, avoiding recomputation of all previous positions.

- **Memory:** ~75.5 MB for full cache (12 layers × 2 × 1024 × 768 × 4B)
- **Per-step speedup:** from O(S²) compute to O(S) attention + O(1) projections
- **Expected S=1 decode latency:** ~15 ms → <1 ms (15× speedup)
- **Risk:** Medium. Requires attention kernel modification to handle mixed
  (cached + new) K/V. The layout change from `[H,S,d]` to `[H,n_ctx,d]` with
  a `cur_S` pointer is straightforward.

### 9.2 Phase 6b: Kernel Fusion

Fuse adjacent elementwise operations to reduce kernel launch overhead and
memory traffic:

- **BiasAdd + residual** (currently 2 launches: `BiasAdd` + `VectorAdd`)
- **GELU + BiasAdd** (MLP path: currently `GELU` → `BiasAdd`)
- **QKV Split + Attention Scores** (eliminate QKV write→read round trip)
- **LayerNorm + GEMM** (common pattern: ln→c_attn, ln→mlp_fc)

### 9.3 Phase 6c: CUDA Graphs

Capture the decode-step kernel DAG into a CUDA graph. The graph structure is
static (same kernels, same shapes per step with KV cache), so graph capture
amortizes 120+ kernel launches into a single graph launch.

### 9.4 Phase 6d: Attention Optimization

- **FlashAttention-style tiling:** Reduce scores/weights workspace from O(S²)
  to O(S) by computing softmax in blocks. Enables S > 1024.
- **Online softmax:** Fuse `scores→softmax→AV` into a single kernel, eliminating
  the 100 MB scores/weights round-trip.

### 9.5 Phase 6e: GEMM Tuning

- **Autotune tile sizes** for GPT-2 shapes (M=S=1..1024, K=768, N=768/2304/3072/50257)
- **Split-K for LM head** (K=768, N=50257) to improve occupancy

### 9.6 Phase 6f: Address Audit Recommendations

- R1: Add S=128 HF golden validation
- R2: Rename bandwidth metric / document L2 artifact
- R4: Optional wte_T integrity check in loader
- R3: Rename or enhance synthetic decode test

### Priority Order

```
1. KV Cache        (6a) — largest speedup, enables real-time generation
2. Kernel Fusion   (6b) — reduces launch overhead, second-largest impact
3. CUDA Graphs     (6c) — amortizes remaining launch overhead
4. Attention Opt   (6d) — enables longer context, reduces memory
5. GEMM Tuning     (6e) — marginal improvement for GPT-2 shapes
6. Audit Items     (6f) — documentation and coverage improvements
```

### Estimated Impact

| Optimization | S=1 Decode Latency | S=1024 Forward Latency | Memory |
|-------------|---------------------|------------------------|--------|
| Baseline (Phase 5) | 15.1 ms | 142.5 ms | 854 MB |
| +KV Cache | <1 ms | 142.5 ms | +76 MB |
| +Fusion | <0.5 ms | ~100 ms | — |
| +CUDA Graphs | <0.3 ms | ~90 ms | — |
| +FlashAttention | <0.3 ms | ~60 ms | −100 MB (scores/weights) |

---

## 10. Build & Test Reproducibility

### Clean Build Confirmation (2026-06-10)

```
Build:  cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/wsl2-gcc13.cmake
        103/103 targets compiled, 0 errors
Tests:  ctest --test-dir build --output-on-failure
        117/117 passing (100%)
Golden: python scripts/golden_test.py
        C8 Logit Gate:   PASS (53/53 argmax, max_err=2.59e-4)
        C9 Decode Gate:  PASS (113/113 tokens)
        C10 Sample Gate: PASS (0/60 implausible)
PyT:    python scripts/validate_kernels.py --kernel all
        42/42 cases (worst max_abs_err=3.87e-6, gate=1e-4, margin=26×)
```

### Build Warnings (benign)

| Warning | Source | Severity |
|---------|--------|----------|
| C++ designated initializers (`-Wc++20-extensions`) | `gpt2_config.hpp` | Benign — works with g++-13 |
| Float equality comparison (`-Wfloat-equal`) | `main.cpp` (Phase 0) | Legacy code, not Phase 5 |
| `system()` return value unused (`-Wunused-result`) | `test_forward.cu`, `test_decode.cu` | Intentional — check via file existence |
| NVLink incompatible librt/pthread/dl | Link step | Benign — 32-bit libs skipped, 64-bit used |

---

## 11. Verdict

**Phase 5 is complete and correct.** All 117 tests pass, all golden gates pass,
and the architecture review finds no blocking issues. The engine can perform
autoregressive GPT-2 inference with both greedy and temperature-sampled decoding
using real HuggingFace weights. The codebase is clean, follows consistent
conventions, and has no memory leaks or synchronization bugs.

The four audit recommendations (R1-R4) are all non-blocking — zero MUST FIX items.
Next: Phase 6 (Optimization).
