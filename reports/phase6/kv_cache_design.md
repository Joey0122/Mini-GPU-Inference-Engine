# Phase 6a: KV Cache — Design Review & Commit Plan

> Historical Phase 6 design notes. The published implementation reuses hidden
> row 0 for decode and sizes each attention scratch buffer for both prefill and
> cached decode. See [the workspace API](../../include/model/workspace.hpp).

**Status:** DESIGN REVIEW (no code changes)
**Date:** 2026-06-10
**Author:** Claude Code

---

## 1. Current Decode Complexity

### 1.1 Per-Step Computational Complexity

The current `LaunchSamplingDecode` ([src/model/decode.cu:63-127](src/model/decode.cu)) runs a full
`LaunchGpt2Forward` at each autoregressive step. The forward pass
([src/model/forward.cu:24-56](src/model/forward.cu)) processes all S=prompt_len+step
positions end-to-end:

| Operation | FLOPs per step | Dominant term |
|---|---|---|
| Embedding | S·D | Θ(S) |
| 12× c_attn GEMM | 12 · 2·S·3D·D = 72·S·D² | Θ(S) |
| 12× QKV Split | 12 · 3·S·D (memcpy) | Θ(S) |
| 12× Attention Scores | 12 · H · S² · (2·d) = 24·H·S²·d | **Θ(S²)** |
| 12× Softmax | 12 · H·S·S (exp + sum) | **Θ(S²)** |
| 12× Attention AV | 12 · H · S² · (2·d) = 24·H·S²·d | **Θ(S²)** |
| 12× c_proj GEMM | 12 · 2·S·D·D = 24·S·D² | Θ(S) |
| 12× mlp_fc GEMM | 12 · 2·S·M·D = 24·S·M·D | Θ(S) |
| 12× GELU | 12 · S·M | Θ(S) |
| 12× mlp_proj GEMM | 12 · 2·S·M·D = 24·S·M·D | Θ(S) |
| lm_head GEMM | 2·S·V·D | Θ(S) |

Plugging GPT-2 Small constants (D=768, H=12, d=64, M=3072, V=50257):

| S | Attention (S² terms) | Linear (S terms) | Total FLOPs | Measured latency |
|---:|---:|---:|---:|---:|
| 1 | ~0.12 M | ~2.84 G | ~2.84 G | 15.1 ms |
| 16 | ~30 M | ~45 G | ~45 G | — |
| 64 | ~483 M | ~182 G | ~182 G | — |
| 256 | ~7.7 G | ~726 G | ~734 G | — |
| 512 | ~61.8 G | ~1.45 T | ~1.51 T | — |
| 1024 | ~494 G | ~2.90 T | ~3.39 T | 142.5 ms |

At S=1024, the quadratic attention terms contribute ~15% of total FLOPs but
~50% of memory traffic (the O(S²) scores/weights tensors at 50 MB dominate
DRAM bandwidth).

### 1.2 Autoregressive Decode Total Cost

For a prompt of length P generating N new tokens:

```
Total FLOPs = Σ_{k=0}^{N-1} f(P + k)
            ≈ N · f(P) + O(N²) extra

where f(S) = O(S²) from attention + O(S) from linear layers
```

For P=4, N=25 (typical C9 golden test): total ~25 · f(4..29) ≈
25 × average ~15ms = ~375ms. But really it's worse because S grows.

For P=256, N=256: total =  Σ_{k=0}^{255} f(256 + k). At S=512 this is
1.51 TFLOPs × 256 ≈ 387 PFLOPs — **307× more work than with KV cache**.

### 1.3 Where Recomputation Occurs

At decode step k (S = P + k), the forward pass recomputes:

1. **c_attn GEMM + BiasAdd**: ln1_out[S, D] → [S, 3D]. All S-1 previous
   rows produce the same Q, K, V values as step k-1. Only row S-1 (the new
   token's hidden state) is new.
2. **QKV Split**: Rearranges [S, 3D] → q/k/v[H, S, d]. All S-1 previous rows
   are identical to step k-1's output.
3. **Attention Scores**: Q[H, S, d] × Kᵀ[H, d, S] → scores[H, S, S].
   All S-1 query rows × all S-1 key columns are identical to the (S-1)×(S-1)
   submatrix from step k-1. Only the new last row and column are new.
4. **Softmax + AV**: Same redundancy — all previous output rows are identical.
5. **c_proj, MLP**: All linear layers recompute S-1 rows identically.
6. **Residual adds**: The first S-1 hidden rows are only modified in-place;
   the residual contributions are identical to step k-1.

**Root cause**: There is no memory of past K and V tensors. At each step,
every position's keys and values are derived from scratch via the full
c_attn → QKV Split → attention pipeline.

---

## 2. KV Cache Architecture

### 2.1 Core Insight

During autoregressive decoding, the key and value tensors for position `pos`
depend ONLY on the hidden state at that position. Since hidden[:pos] is
unchanged across decode steps (residual adds are position-independent —
hidden[pos] only gets contributions from attention over positions 0..pos), K
and V for positions 0..pos-1 are *identical* to the previous step.

We store them once and reuse them. The attention computation for the new
token becomes:

```
q_new = c_attn_split(ln1(hidden[new_pos]))   — [H, 1, d]
scores[h, 0, j] = scale · q_new[h,0,:] · K_cache[layer][h, j, :]ᵀ   — for j in 0..cur_S
weights = softmax(scores)
attn_out = weights · V_cache[layer]
```

The "all previous positions" work vanishes. Per-step FLOPs drop from
O(S² + S·D²) to O(S + D²). At S=1024, attention drops from 494 GFLOPs to
0.48 GFLOPs — a **1024× reduction** in attention computation.

### 2.2 Cache Tensor Layouts

Each of the 12 transformer blocks stores its own K and V cache:

```
K_cache[layer]: [n_head, n_ctx, head_dim]  — row-major, head-major
V_cache[layer]: [n_head, n_ctx, head_dim]  — row-major, head-major

Linear index: ((layer * n_head + h) * n_ctx + pos) * head_dim + d
```

This layout is chosen to match the existing head-major convention in
`LaunchAttentionScores` and `LaunchAttentionAV`:

```cpp
// In AttentionScoresKernel (current):
const float* kh = k + static_cast<long long>(h) * S * head_dim;
// Element k[h, pos, d] is at offset h*S*head_dim + pos*head_dim + d

// In KV cache:
const float* kh = k_cache + ((layer * n_head + h) * n_ctx + 0) * head_dim;
// Element k_cache[layer][h, pos, d] at same relative offset within head h
```

The attention kernel reads K[h, 0..cur_S, :] as a contiguous block of
cur_S × head_dim floats. With the cache layout, this is exactly a pointer
into the cache at head h, starting at position 0. **No stride change —
the cache layout is a drop-in replacement for the k tensor layout in the
attention kernel.**

Decision: pack all layers into two flat buffers (one for K, one for V)
rather than 24 separate `DeviceBuffer`s. Rationale: fewer allocations,
simpler memory accounting, and the offset math is trivial.

```cpp
struct KVCache {
    DeviceBuffer<float> k;   // [n_layer * n_head * n_ctx * head_dim]
    DeviceBuffer<float> v;   // [n_layer * n_head * n_ctx * head_dim]
    int num_positions;       // number of cached positions (0 = empty, grows during prefill)
};
```

### 2.3 Ownership Model

```
┌─────────────────────────────────────────────────────┐
│                  Inference Engine                     │
│                                                       │
│  Gpt2Weights (owned, loaded once)                     │
│  ├── wte, wpe, wte_T                                  │
│  ├── 12× BlockWeights                                 │
│  └── ln_f_w, ln_f_b, config                           │
│                                                       │
│  Gpt2Workspace (owned, allocated once)                │
│  ├── hidden, logits                                   │
│  └── blk (TransformerBlockWorkspace)                  │
│                                                       │
│  KVCache (owned, allocated once)          ← NEW       │
│  ├── k [12 × 12 × 1024 × 64] = 72 MB                 │
│  └── v [12 × 12 × 1024 × 64] = 72 MB                 │
│                                                       │
│  output_ids [device, int32]                           │
└─────────────────────────────────────────────────────┘
```

`KVCache` follows the same move-only RAII pattern as `DeviceBuffer` and
`Gpt2Workspace`. It is allocated once at inference startup alongside
`Gpt2Weights` and `Gpt2Workspace`. No allocation occurs in the hot path.

The cache is logically write-only during prefill (populating K,V for all
prompt positions) and write-one-row during decode (appending K,V for the
new token). K and V for position pos are written exactly once and never
modified thereafter — this is the fundamental property that makes the cache
correct. The current `num_positions` advances monotonically (0 → P → P+N).

### 2.4 Allocation Strategy

Factory function following the existing `::Create` pattern:

```cpp
// include/model/kv_cache.hpp
struct KVCache {
    DeviceBuffer<float> k;
    DeviceBuffer<float> v;
    int num_positions = 0;

    static KVCache Create(int n_layer, int n_head, int n_ctx, int head_dim);

    // Convenience helpers for indexing
    float* k_layer(int layer) {
        return k.data() + static_cast<size_t>(layer) * n_head * n_ctx * head_dim;
    }
    float* v_layer(int layer) {
        return v.data() + static_cast<size_t>(layer) * n_head * n_ctx * head_dim;
    }
};

// src/model/kv_cache.cu
KVCache KVCache::Create(int n_layer, int n_head, int n_ctx, int head_dim) {
    KVCache cache;
    const size_t per_layer = static_cast<size_t>(n_head) * n_ctx * head_dim;
    cache.k = DeviceBuffer<float>(per_layer * n_layer);
    cache.v = DeviceBuffer<float>(per_layer * n_layer);
    cache.num_positions = 0;
    return cache;
}
```

For GPT-2 Small: `KVCache::Create(12, 12, 1024, 64)` → two 72 MB buffers.

### 2.5 Workspace Changes

The current `TransformerBlockWorkspace` allocates q, k, v, scores, and
weights sized for `S` (full sequence). For decode steps (S=1 for the new
token), most of this space is unused. **No change is needed** — the
existing workspace is already oversized for decode steps, and the memory
is already allocated.

However, we add a **second, smaller workspace** for decode steps to avoid
pinning the full-S workspace. This is optional but recommended:

| Buffer | Prefill workspace (S=1024) | Decode workspace (S'=1) |
|---|---|---|
| q | H·S·d = 768 KB | H·1·d = 768 × 1 × 64 = **48 KB** |
| k | 768 KB | 48 KB |
| v | 768 KB | 48 KB |
| scores | H·S² = 12·1024² = 50 MB | H·1·S_cached ≈ **48 KB** (at S=1, scores is H×1×1) |
| weights | 50 MB | 48 KB |

Wait — for decode step, scores and weights are `[n_head, 1, cur_S]` where
cur_S grows from P to P+N. At S_cached=1024, scores is 12 × 1 × 1024 = 48 KB.
Much smaller than 50 MB!

**Decision**: Keep the existing `TransformerBlockWorkspace` for prefill
(full-S). For decode steps, either (a) reuse it (works but wastes 50 MB of
allocated space), or (b) create a lightweight `DecodeStepWorkspace` with
buffers sized for S=1 for linear ops and S=1024 for scores/weights.

**Recommended**: Option (a) — reuse the existing workspace. The memory is
already allocated. Creating a separate workspace adds API surface without
saving memory (the full workspace must still exist for prefill). After
Phase 6, if memory pressure is a concern, we can resize.

**One actual change**: the workspace is currently created at `max_seq =
prompt_len + max_new_tokens` ([test_decode.cu:247](tests/unit/test_decode.cu#L247)).
With KV cache, the prefill workspace only needs `S = prompt_len`, and the
decode workspace only needs `S = max_seq` for scores/weights (but linear
ops at S=1). The simplest path: keep the workspace at `max_seq` but only
pass `S=1` to the linear-layer portions of the decode step.

---

## 3. Integration Plan

### 3.1 Changes to Attention

**New kernel**: `LaunchAttentionCached` — single-token attention using
cached K and V.

```
Current: LaunchAttention(q[S,H,d], k[S,H,d], v[S,H,d]) → out[S,D]
  → AttentionScoresKernel: grid(S, H), block(head_dim)
     Computes scores[h,i,j] for ALL i in 0..S-1, j in 0..S-1
  → LaunchSoftmax: row-wise over [H*S, S]
  → AttentionAVKernel: grid(S, H), block(head_dim)
     Computes out[h,i,d] for ALL i in 0..S-1

New: LaunchAttentionCached(q_new[1,H,d], k_cache[S,H,d], v_cache[S,H,d]) → out[1,D]
  → AttentionScoresKernel (same kernel, different grid): grid(1, H), block(head_dim)
     Computes scores[h,0,j] for j in 0..S-1 (S = cached positions)
     Only ONE query row per head. The existing kernel already supports this —
     blockIdx.x ranges over query rows; we launch with grid(1, n_head).
  → LaunchSoftmax: row-wise over [H*1, S_cached] = H rows of S_cached elements
  → AttentionAVKernel (same kernel): grid(1, H), block(head_dim)
     Computes out[h,0,d] for the single query row.
```

**Key insight**: The existing `AttentionScoresKernel` and `AttentionAVKernel`
already support arbitrary S grid dimensions. The kernel that processes query
row `i = blockIdx.x` only reads K rows for j ≤ i. If we pass `S` as the
number of cached positions and launch with `grid(1, n_head)`, the existing
kernel computes exactly what we need — **no kernel changes required**.

Similarly, `AttentionAVKernel` loops `j ≤ i` — with `i = 0` (the only query
row), it sums over all cached V rows 0..S-1, which is correct.

**What changes**:
- The attention orchestration function needs a variant that passes
  `k_cache` and `v_cache` as the key/value sources instead of freshly-split
  k and v tensors.
- `S` in the attention call means "number of cached positions" (K/V
  dimension), NOT "query dimension".

```cpp
// New function (add to include/kernels/attention.hpp):
// Single-query-row causal attention against cached K, V.
// q_new:   [n_head, 1, head_dim] — one query row per head (new token only)
// k_cache: [n_head, S, head_dim] — cached keys for positions 0..S-1
// v_cache: [n_head, S, head_dim] — cached values for positions 0..S-1
// out:     [1, n_head*head_dim]  — single-row output
// S:       number of cached positions (all valid as keys — causal mask
//          is trivially satisfied: query pos S-1 ≥ all cached keys)
// scores, weights: scratch, each [n_head, 1, S]
void LaunchAttentionCached(const float* q_new,
                           const float* k_cache,
                           const float* v_cache,
                           float* out,
                           float* scores,
                           float* weights,
                           int n_head, int S, int head_dim,
                           cudaStream_t stream = nullptr);
```

The implementation is:
```cpp
void LaunchAttentionCached(...) {
    // Step 1: Scores with grid(1, n_head) — only 1 query row
    // The existing kernel works because blockIdx.x = 0 is the query row,
    // and j > 0 is never true (0 ≤ 0) so causal mask is a no-op.
    // ...but wait: we need to pass S correctly.
    // The kernel indexes k[j] for j in 0..S-1. If S is the cache size,
    // and the query is at position cur_S-1, then j <= cur_S-1 is always
    // true for j in 0..cur_S-1. The causal mask is satisfied.
    // Actually, the kernel checks `j > i` where i = blockIdx.x.
    // With grid(1, H), blockIdx.x = 0 (not cur_S-1).
    // We need blockIdx.x to represent the query's absolute position.
}
```

**Correction**: The causal mask `j > i` in `AttentionScoresKernel` depends
on `i = blockIdx.x` being the absolute query position. If we launch with
`grid(1, H)` and `blockIdx.x = 0`, the kernel will incorrectly mask all
j > 0. We need to either:

**Option A**: Pass the query's absolute position and use a modified kernel.
**Option B**: Since all cached positions j ≤ cur_S-1 are valid keys for
the new token (causal masking is satisfied by construction), skip the
causal mask entirely. The new query is at position cur_S, and all cached
keys are at positions 0..cur_S-1 ≤ cur_S. So j ≤ i is always true.

**Decision: Option B**. Add a flag `causal=false` to the scores kernel
or, more cleanly, add a `LaunchAttentionScoresUnmasked` variant. In the
unmasked variant, `j ≤ i` is not checked; all keys are valid.

Actually, even simpler: the current kernel already works if we make `i = S-1`
(the last cached position). Wait, no — the kernel uses `blockIdx.x` as `i`.
With a 1×H grid, `i = 0`. We'd need `i = S-1`.

**Best approach**: Modify `AttentionScoresKernel` to accept the query's
absolute position as a kernel parameter rather than deriving it from
`blockIdx.x`. Or, add a simple unmasked variant.

Given the project's "clean first" approach, I recommend a dedicated
**`AttentionScoresKernelCached`** that omits the causal mask:

```cpp
__global__ void AttentionScoresKernelCached(
    const float* __restrict__ q,    // [n_head, 1, head_dim]
    const float* __restrict__ k,    // [n_head, S, head_dim]
    float* __restrict__ scores,     // [n_head, 1, S]
    int S, int head_dim, float scale) {

    const int h = blockIdx.y;
    const int t = threadIdx.x;

    extern __shared__ float q_row[];
    const float* qh = q + (static_cast<long long>(h) * 1 + 0) * head_dim;
    const float* kh = k + static_cast<long long>(h) * S * head_dim;
    float* srow = scores + static_cast<long long>(h) * 1 * S;

    // Stage the single query row
    for (int d = t; d < head_dim; d += blockDim.x) q_row[d] = qh[d];
    __syncthreads();

    // Dot against all cached keys — NO causal mask
    for (int j = t; j < S; j += blockDim.x) {
        const float* kj = kh + static_cast<long long>(j) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += q_row[d] * kj[d];
        srow[j] = dot * scale;
    }
}
```

This is a minor variant — the kernel body is actually *simpler* than the
original (no `if (j > i)` branch). We can share most of the code or keep
it separate for clarity.

### 3.2 Changes to Transformer Block

Two new functions (or one overloaded function with a cache parameter):

**`LaunchTransformerBlockPrefill`** — existing behavior + cache population.

This is the existing `LaunchTransformerBlock` with one addition: after the
QKV split, the full K and V tensors are copied into the cache for this
layer:

```cpp
// After QkvSplit, before Attention:
// Copy K and V into cache at positions [0, S)
const size_t per_head = static_cast<size_t>(S) * head_dim;  // K/V rows per head
for (int h = 0; h < H; ++h) {
    const size_t src_off = static_cast<size_t>(h) * S * head_dim;
    const size_t dst_off = (static_cast<size_t>(layer_idx) * H + h) * n_ctx * head_dim;
    cudaMemcpyAsync(kv_cache.k.data() + dst_off,
                    k + src_off,
                    per_head * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    // same for V
}
```

Wait — this is H separate cudaMemcpyAsync calls per layer (one per head).
At 12 layers × 12 heads × 2 (K+V) = 288 small memcpys. That's fine for a
one-time prefill but ugly.

**Better**: Write K and V into the cache directly during the QKV split.
Currently, `LaunchQkvSplit` writes to workspace q/k/v buffers. We can add
an alternative output path that writes K and V directly to the cache:

```cpp
// New variant: QkvSplitToCache
// Writes Q to q_ws (needed for attention this step only),
// writes K and V directly to the KV cache at the specified position range.
void LaunchQkvSplitToCache(const float* x,
                           float* q_ws,
                           float* k_cache,   // cache destination
                           float* v_cache,   // cache destination
                           int S, int pos_offset,  // write to cache[pos_offset..pos_offset+S)
                           int n_embd, int n_head, int n_ctx,
                           cudaStream_t stream);
```

**`LaunchTransformerBlockDecodeStep`** — single-position forward using cache.

Processes only position `pos` (the new token), reading K and V from the
cache and writing the new K and V for this position:

```
1. ln_1: LayerNorm(hidden[pos:pos+1, :]) → ln1_out[1, D]
   (reads only 1 row of hidden; writes 1 row)
2. c_attn: ln1_out[1,D] @ c_attn_w[D,3D] + bias → c_attn_out[1,3D]
   (1-row GEMM — existing kernel handles this)
3. QKV split: c_attn_out[1,3D] → q[1,H,d], k_new[1,H,d], v_new[1,H,d]
   (tiny — writes to workspace q/k/v at S=1)
4. Write k_new → K_cache[layer][:, pos, :], v_new → V_cache[layer][:, pos, :]
   (H small cudaMemcpyAsync calls)
5. Attention: q[1,H,d] queries against K_cache[layer][:, 0..pos, :]
   → attn_out[1, D]
6. c_proj: attn_out[1,D] @ c_proj_w[D,D] + bias → proj_out[1,D]
7. Residual: hidden[pos, :] += proj_out[0, :]
8. ln_2, mlp_fc, GELU, mlp_proj: all on 1 row → proj_out[1,D]
9. Residual: hidden[pos, :] += proj_out[0, :]
```

This is exactly the same sequence of operations as `LaunchTransformerBlock`,
but every kernel processes S=1 instead of S. The `hidden` buffer contains
all past positions' states; only row `pos` is read and written.

### 3.3 Changes to Forward

**`LaunchGpt2Prefill`** — full forward pass + cache population.

Nearly identical to the existing `LaunchGpt2Forward`, but with a
`KVCache&` parameter and layer index threading:

```cpp
void LaunchGpt2Prefill(const int* token_ids,
                       const Gpt2Weights& weights,
                       float* logits,
                       Gpt2Workspace& ws,
                       KVCache& kv_cache,
                       const Gpt2Config& cfg,
                       int S,
                       cudaStream_t stream);
```

The implementation is `LaunchGpt2Forward` with one addition: each
`LaunchTransformerBlock` call is swapped for `LaunchTransformerBlockPrefill`,
which populates the cache.

**`LaunchGpt2DecodeStep`** — single-position forward using cache.

```cpp
void LaunchGpt2DecodeStep(int new_token,
                          int position,       // absolute position in sequence (0-indexed)
                          const Gpt2Weights& weights,
                          float* logits_row,  // [vocab_size] — single row of logits
                          Gpt2Workspace& ws,
                          const KVCache& kv_cache,
                          const Gpt2Config& cfg,
                          cudaStream_t stream);
```

Implementation:
```
1. Embedding for single token:
   hidden[pos, :] = wte[new_token, :] + wpe[pos, :]
   (writes 1 row — use cudaMemcpy or a small kernel)
2. For each layer 0..11:
   LaunchTransformerBlockDecodeStep(hidden, weights.blocks[l], kv_cache,
                                     ws.blk, cfg, layer_idx, position, stream)
3. Final LayerNorm: only row `position` of hidden — but wait, ln_f needs to
   normalize row `position` ONLY. We can call LaunchLayerNorm with S=1 on
   hidden[pos:pos+1, :]:
   LaunchLayerNorm(hidden + pos*D, hidden + pos*D, ln_f_w, ln_f_b,
                   1, D, eps, Naive, stream)
4. LM head: hidden[pos:pos+1, :] @ wte_T → logits_row[1, V]
   LaunchGemm(hidden + pos*D, wte_T, logits_row, 1, V, D, ...)
```

### 3.4 Changes to Decode

The decode loop in [src/model/decode.cu:63-127](src/model/decode.cu) is
restructured into two phases:

```cpp
int LaunchSamplingDecodeKV(const int* prompt_ids,
                           int* output_ids,
                           const Gpt2Weights& weights,
                           Gpt2Workspace& ws,
                           KVCache& kv_cache,
                           const Gpt2Config& cfg,
                           int prompt_len,
                           int max_new_tokens,
                           float temperature,
                           unsigned int seed,
                           cudaStream_t stream) {

    // --- Phase 1: Prefill ---
    // Copy prompt to output_ids
    cudaMemcpyAsync(output_ids, prompt_ids, prompt_len * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    // Full forward pass on prompt, populating KV caches
    LaunchGpt2Prefill(output_ids, weights, ws.logits.data(),
                      ws, kv_cache, cfg, prompt_len, stream);
    kv_cache.num_positions = prompt_len;

    // --- Phase 2: Autoregressive Decode ---
    std::vector<float> last_logits(V);
    std::mt19937 rng(seed != 0 ? seed : std::random_device{}());

    for (int step = 0; step < max_new_tokens; ++step) {
        const int cur_pos = prompt_len + step;

        // Copy last logit row to host
        cudaMemcpyAsync(last_logits.data(),
                        ws.logits.data() + (cur_pos - 1) * V,
                        V * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Sample next token
        int next_token = (temperature > 1e-6f)
            ? SampleNextToken(last_logits.data(), V, temperature, rng)
            : GreedyNextToken(last_logits.data(), V);

        // Write to output buffer
        cudaMemcpyAsync(output_ids + cur_pos, &next_token,
                        sizeof(int), cudaMemcpyHostToDevice, stream);

        // Decode step: process the new token using KV cache
        LaunchGpt2DecodeStep(next_token, cur_pos, weights,
                             ws.logits.data() + cur_pos * V,
                             ws, kv_cache, cfg, stream);
        kv_cache.num_positions = cur_pos + 1;
    }

    cudaStreamSynchronize(stream);
    return max_new_tokens;
}
```

Key differences from current `LaunchSamplingDecode`:
1. The prefill processes the full prompt at once.
2. Each decode step calls `LaunchGpt2DecodeStep` (S=1 for the new position)
   instead of `LaunchGpt2Forward` (full S-sequence forward).
3. The first decode step reads logits from the prefill output (position
   `prompt_len-1`); subsequent steps read from the previous decode step's
   output (position `prompt_len+step-1`).

---

## 4. Validation Strategy

### 4.1 Numerical Identity Proof

**Claim**: For any prompt of length P and any sequence of sampled tokens,
the KV-cache decode produces logits that are **bitwise identical** to the
full-S recomputation forward pass.

**Proof sketch** (by induction on the sequence position):

The GPT-2 Pre-LN transformer block is:
```
h_next = h + c_proj(Attn(QKV(c_attn(ln1(h)))))
       + mlp_proj(GELU(mlp_fc(ln2(h + ...))))
```

**Base case** (prefill, S = P): The prefill forward pass is identical to
the existing `LaunchGpt2Forward`. All hidden states and the P-th position's
logits are identical. K_cache and V_cache hold the exact K and V tensors
from the c_attn → QKV split of the prefill pass.

**Inductive step**: Assume the cache holds correct K, V for positions
0..pos-1 and hidden[0..pos-1, :] are correct. At position pos:

1. `hidden[pos, :] = wte[token_ids[pos]] + wpe[pos]` — identical to full
   recomputation (embedding depends only on the token ID and position index).
2. For each layer:
   - `ln1_out = LayerNorm(hidden[pos:pos+1, :])` — reads only hidden[pos,:],
     which is identical to the full recomputation. LayerNorm per-row is
     independent; row pos depends only on hidden[pos,:].
   - `c_attn_out = ln1_out @ c_attn_w + bias` — row-wise GEMM; row pos
     depends only on ln1_out[pos,:].
   - QKV split produces q[H,1,d], k_new[H,1,d], v_new[H,1,d] — identical
     to the corresponding rows in the full recomputation.
   - `k_new` written to K_cache at position pos. In the full recomputation,
     this same value would be at K[pos,:,:]. Identical.
   - `scores = q_new · K_cache[pos]ᵀ` — q_new is identical to Q[pos,:,:]
     in the full recomputation; K_cache[:pos] is identical to K[:pos,:,:]
     from the previous recomputation step. The dot products are identical.
   - `weights = softmax(scores)` — identical inputs → identical outputs.
   - `attn_out = weights · V_cache[:pos]` — V_cache[:pos] is identical to
     V[:pos,:,:] from the previous step. The weighted sum is identical.
   - c_proj, residual, ln2, mlp, residual2 — all row-wise, depend only on
     the row's inputs. Identical.
3. `ln_f(hidden[pos,:])` — identical.
4. `logits[pos,:] = hidden[pos,:] @ wte_T` — identical.

Therefore, logits[pos,:] from KV-cache decode equals logits[pos,:] from
the full S-sequence forward pass. ∎

### 4.2 Unit Tests

**Test 1: KVCache Allocation**
- Create KVCache, verify sizes, verify pointers are non-null.
- Create with various dimensions (GPT-2 small, GPT-2 medium).
- Zero-size n_layer/n_head returns empty buffers (no-op).

**Test 2: Cache Write + Read**
- Write known K, V for a single position, read back via cudaMemcpy.
- Verify bit-exact round-trip.

**Test 3: Cached Attention = Full Attention (Last Row)**
- Generate random hidden states for S positions.
- Run full `LaunchTransformerBlock` on S positions → capture output[S-1, :].
- Run `LaunchTransformerBlockDecodeStep` on position S-1 with K/V cache
  populated from the full block's QKV split → capture output.
- Assert bit-level equality (or within FP32 exact-reproducibility tolerance).
- **This is the critical safety test.** If this fails, all bets are off.

**Test 4: Prefill + Decode Step Matches Full Forward**
- S=prompt_len=4, generate one additional token.
- Run full `LaunchGpt2Forward` at S=5 → capture logits[4, :].
- Run prefill at S=4 → run decode step for token at position 4 → capture logits.
- Assert bit-level equality of logits.

**Test 5: Multi-Step Decode Matches Full Forward**
- Prompt length 4, generate 8 tokens.
- Full recomputation: call LaunchGpt2Forward for S=4,5,...,12.
- KV-cache: prefill at S=4 + 8 decode steps.
- Compare all generated logits at each step.
- All must match exactly.

**Test 6: Identity with Real Weights**
- Load real GPT-2 weights.
- Run prefill + 10 decode steps with KV cache.
- Run the existing `LaunchGpt2Forward`-based decode for 10 steps.
- Compare token sequences — must be **identical** (both use greedy, no randomness).

### 4.3 Golden Tests

The existing golden tests in [tests/unit/test_decode.cu](tests/unit/test_decode.cu)
must pass **unchanged** against the KV-cache decode:

- `MultiPromptGreedyDecode`: 113/113 token matches (C9)
- `MultiPromptSamplingDecode`: 0/60 implausible (C10)
- `TemperatureZeroReproducesGreedy`: identical output
- `SameSeedProducesIdenticalOutput`: determinism preserved

Since the numerical output is identical (Section 4.1), these tests continue
to pass without modification. The only change is that the underlying
implementation uses KV cache instead of full recomputation.

Additionally, a new golden test compares KV-cache decode against full
recomputation for a few prompts using real weights:

```cpp
TEST_F(DecodeGoldenTest, KVCacheMatchesFullRecompute) {
    // For each prompt in {singleton, short4, medium16}:
    //   1. Run full forward pass at each step (current method)
    //   2. Run KV-cache decode (prefill + steps)
    //   3. Assert: all logits match within 0 (bit-identical)
    //   4. Assert: token sequences match
}
```

### 4.4 Performance Tests

Benchmark comparing old vs new decode:

```cpp
// benchmarks/decode/bench_decode.cu
// BM_DecodeFullRecompute: current method
// BM_DecodeKVCache: prefill + KV-cache steps
// Sweep: prompt_len = {1, 4, 16, 64, 128}, num_steps = {25, 100}
```

Expected results (RTX 4070 Laptop, GPT-2 Small):

| Prompt | Steps | Old (ms) | New (ms) | Speedup |
|---|---:|---:|---:|---:|
| 1 | 25 | ~375 | ~25 (15 prefill + 25×0.4) | **15×** |
| 4 | 25 | ~400 | ~28 (15 prefill + 25×0.5) | **14×** |
| 16 | 25 | ~650 | ~35 (18 prefill + 25×0.7) | **19×** |
| 64 | 25 | ~2,200 | ~55 (25 prefill + 25×1.2) | **40×** |
| 128 | 100 | ~18,000 | ~200 (35 prefill + 100×1.7) | **90×** |
| 256 | 100 | ~70,000 | ~500 (60 prefill + 100×4.4) | **140×** |

The speedup grows with prompt length because the old method pays O(S²)
every step while the new method pays O(S) in the decode phase.

---

## 5. Memory Analysis

### 5.1 KV Cache Size — Exact

For GPT-2 Small (n_layer=12, n_head=12, n_ctx=1024, head_dim=64):

```
K_cache = 12 × 12 × 1024 × 64 × 4 bytes = 37,748,736 bytes = 36.0 MB
V_cache = 12 × 12 × 1024 × 64 × 4 bytes = 37,748,736 bytes = 36.0 MB
Total   = 72 MB (75,497,472 bytes)
```

Per-layer: 6 MB (3 MB K + 3 MB V).
Per-head per-layer: 256 KB (1024 × 64 × 4 bytes).

### 5.2 Total GPU Memory — Before vs After

| Component | Before (Phase 5) | After (+KV Cache) |
|---|---|---|
| Weights (wte + wpe + 12×blocks + ln_f) | ~497 MB | 497 MB |
| Workspace (S=1024, full prefill) | ~357 MB | 357 MB |
| output_ids (S=1024 int32) | 4 KB | 4 KB |
| **KV Cache (K+V)** | — | **72 MB** |
| **Total** | **~854 MB** | **~926 MB** |

On an 8 GB RTX 4070 Laptop, this leaves ~7 GB for CUDA context, driver,
and other processes — comfortably within budget.

### 5.3 Future Scaling

For larger models, KV cache dominates. At GPT-2 Medium (n_layer=24,
n_head=16, n_ctx=1024, head_dim=64):

```
K_cache = 24 × 16 × 1024 × 64 × 4 = 96 MB
V_cache = 96 MB
Total   = 192 MB
```

At GPT-2 Large (n_layer=36, n_head=20, n_ctx=1024, head_dim=64):

```
Total = 2 × 36 × 20 × 1024 × 64 × 4 = 360 MB
```

For these larger models, the KV cache becomes the dominant memory consumer
after weights. FlashAttention-style tiling (Phase 6d) can eliminate the
50 MB scores/weights scratch buffers, offsetting the KV cache footprint.

---

## 6. Commit Plan

Each commit must compile and pass all existing tests. New tests are added
alongside the feature they validate.

### Commit 1: KVCache Data Structure

**Files:**
- NEW: `include/model/kv_cache.hpp` — struct definition, `Create` declaration
- NEW: `src/model/kv_cache.cu` — `KVCache::Create` implementation
- NEW: `tests/unit/test_kv_cache.cu` — allocation tests
- MODIFY: `CMakeLists.txt` — add new source files and test

**Tests:**
- `KVCache.Creation`: Creates cache, verifies sizes, verifies non-null pointers.
- `KVCache.ZeroLayerIsNoop`: n_layer=0 returns empty buffers.
- `KVCache.WriteAndRead`: Writes one position's K and V, reads back, verifies
  bit-level accuracy.

**Verification:** `ctest --test-dir build` — all 117 existing tests pass.

---

### Commit 2: Cached Attention Kernel

**Files:**
- MODIFY: `include/kernels/attention.hpp` — add `LaunchAttentionCached` declaration
- MODIFY: `src/kernels/attention.cu` — add `AttentionScoresKernelCached` kernel
  and `LaunchAttentionCached` function
- MODIFY: `tests/unit/test_attention.cu` — add cached attention unit tests
- MODIFY: `tests/common/test_utils.hpp` — add `CpuAttentionCached` reference (if needed)

**Kernel:** `AttentionScoresKernelCached` — identical to `AttentionScoresKernel`
but without the `j > i` causal mask branch. All cached keys are valid by
construction. Grid: `(1, n_head)`.

**Tests:**
- `Attention.CachedMatchesFullLastRow`: Random Q/K/V. Run full attention at
  S positions; run cached attention with Q last row + full K/V cache.
  Assert output matches (within FP32 reproducibility).
- `Attention.CachedZeroLengthIsNoop`: S=0 returns immediately.
- `Attention.CachedSingleKey`: S=1, verify output equals scale·dot·V (no
  softmax effect with single element — softmax of single element = 1.0).

**Verification:** All existing tests pass. New cached attention tests pass.

---

### Commit 3: QKV Split to Cache

**Files:**
- MODIFY: `include/kernels/qkv_split.hpp` — add `LaunchQkvSplitToCache` declaration
- MODIFY: `src/kernels/qkv_split.cu` — add kernel that writes K/V directly to cache

**Kernel:** `QkvSplitToCacheKernel` — same as `QkvSplitKernel` but K and V
output pointers point to cache positions instead of workspace tensors. The
output offset formula uses `(h * n_ctx + pos) * head_dim` instead of
`(h * S + s) * head_dim`.

**Tests:**
- `QkvSplit.CacheWrite`: Split c_attn output, write K/V to simulated cache.
  Read back and compare against expected values from CPU reference.

**Verification:** All existing tests pass.

---

### Commit 4: Transformer Block — Prefill + Decode Step

**Files:**
- MODIFY: `include/model/transformer_block.hpp` — add `LaunchTransformerBlockPrefill`
  and `LaunchTransformerBlockDecodeStep` declarations
- MODIFY: `src/model/transformer_block.cu` — implement both new functions
- MODIFY: `tests/unit/test_transformer_block.cu` — add cache tests

**`LaunchTransformerBlockPrefill(hidden, blk, ws, kv_cache, layer_idx, cfg, S, stream)`:**
Identical to `LaunchTransformerBlock` except: after QKV split, writes K and V
into `kv_cache` for this layer at positions [0, S). Uses the new
`LaunchQkvSplitToCache` variant.

**`LaunchTransformerBlockDecodeStep(hidden, blk, kv_cache, ws, cfg, layer_idx, pos, stream)`:**
Processes only position `pos` (S=1 for the new token):
1. ln1 on hidden[pos:pos+1, :]
2. c_attn GEMM + bias (S=1)
3. QKV split (S=1) → workspace q/k/v
4. Write k/v to kv_cache at position pos (H small cudaMemcpyAsync calls,
   or a micro-kernel)
5. LaunchAttentionCached: q_new queries against K_cache[:, 0..pos, :]
6. c_proj GEMM + bias (S=1)
7. VectorAdd hidden[pos,:] += proj_out
8. MLP path: all at S=1
9. VectorAdd hidden[pos,:] += proj_out

**Tests:**
- `Block.CachedMatchesFullLastRow`: Run full block at S positions. Run
  decode step for position S-1 with cache populated from full block's QKV.
  Assert hidden[S-1, :] matches exactly.
- `Block.PrefillPopulatesCache`: Run prefill at S=4. Read back K/V cache.
  Assert values match the QKV split of a full block run.
- `Block.DecodeStepMultipleLayers`: Run decode step through all 12 layers
  with real-shaped buffers. Verify output is valid (no NaN, all finite).

**Verification:** All existing block tests pass. New cache block tests pass.

---

### Commit 5: Forward — Prefill + Decode Step

**Files:**
- MODIFY: `include/model/forward.hpp` — add `LaunchGpt2Prefill` and
  `LaunchGpt2DecodeStep` declarations
- MODIFY: `src/model/forward.cu` — implement both new functions
- MODIFY: `tests/unit/test_forward.cu` — add cache validation tests

**`LaunchGpt2Prefill`:** `LaunchGpt2Forward` + KVCache population via
`LaunchTransformerBlockPrefill` for each layer.

**`LaunchGpt2DecodeStep`:** Embedding for one token → 12×
`LaunchTransformerBlockDecodeStep` → ln_f on one row → lm_head for one row.

**Tests:**
- `Forward.PrefillMatchesFullForward`: Run prefill at S=8. Compare all logits
  against `LaunchGpt2Forward` at S=8. Assert bit-identical.
- `Forward.DecodeStepMatchesFullForward`: Run prefill at S=4, then decode
  step for position 4. Compare logits against `LaunchGpt2Forward` at S=5.
  Assert bit-identical for the last row.
- `Forward.MultiStepDecodeMatchesFullForward`: Prefill at S=4, then 5 decode
  steps (positions 4..8). At each step, compare logits against full forward
  at S=4..9. All must match exactly.

**Verification:** All existing forward tests pass. New cache forward tests pass.

---

### Commit 6: Rewire Decode Loop

**Files:**
- MODIFY: `src/model/decode.cu` — restructure `LaunchSamplingDecode` to use
  prefill + KV-cache decode steps
- MODIFY: `include/model/decode.hpp` — add `KVCache&` parameter (or keep
  the old API as a thin wrapper)

**API decision**: We have two options:
1. **Add new API**: `LaunchSamplingDecodeKV` / `LaunchGreedyDecodeKV`
   (new functions). Old API preserved for backward compatibility.
2. **Modify existing API**: Add `KVCache&` parameter to
   `LaunchSamplingDecode` / `LaunchGreedyDecode`. All callers updated.

**Recommendation: Option 2** — the old full-recompute decode path has no
use case once KV cache exists. It's strictly slower and produces identical
output. Delete the old path; update all callers (tests, benchmarks).

**Implementation**:
```cpp
int LaunchSamplingDecode(const int* prompt_ids,
                         int* output_ids,
                         const Gpt2Weights& weights,
                         Gpt2Workspace& ws,
                         KVCache& kv_cache,           // NEW parameter
                         const Gpt2Config& cfg,
                         int prompt_len,
                         int max_new_tokens,
                         float temperature,
                         unsigned int seed,
                         cudaStream_t stream);
```

Body: prefill + loop of {sample, decode_step} as described in Section 3.4.

**Tests:**
- Update all test files to allocate and pass `KVCache`.
- All existing decode tests (SyntheticGreedyDecodeProducesValidTokens,
  TemperatureZeroReproducesGreedy, SameSeedProducesIdenticalOutput,
  DifferentSeedsMayDiffer, MultiPromptGreedyDecode,
  MultiPromptSamplingDecode) pass unchanged.
- Golden tests pass: C8 (53/53 argmax), C9 (113/113 tokens), C10 (0/60
  implausible).

**Verification:** `ctest --test-dir build` — 117/117 + new tests = all passing.
`python scripts/golden_test.py` — ALL GATES PASS.

---

### Commit 7: Benchmark + Documentation

**Files:**
- NEW: `benchmarks/decode/bench_decode_kv.cu` — KV-cache decode benchmark
- MODIFY: `benchmarks/forward/bench_forward.cu` — add prefill benchmark
- MODIFY: `reports/phase6/kv_cache_design.md` — update to IMPLEMENTED status
  with actual benchmark numbers
- MODIFY: `PROJECT_STATUS.md` — update Phase 6 status

**Benchmarks:**
- Prefill latency at various S (should match existing forward benchmark
  + small cache-write overhead).
- Decode step latency at various cache sizes (S=1..1024).
- End-to-end: prefill + N decode steps.
- Compare against old full-recompute decode (measured or extrapolated).

**Verification:** Clean build, all tests pass, benchmarks produce valid CSV.

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Numerical non-determinism between prefill and decode step | Low | High (golden tests fail) | Commit 5's tests compare bit-level output; any discrepancy caught immediately |
| Cache index bug (off-by-one in position) | Medium | High (wrong attention) | Commit 3 tests verify exact values; Commit 4 tests compare against full block |
| Memory regression (72 MB KV cache) | Low | Low | 926 MB total on 8 GB GPU — fine. Documented in memory analysis. |
| Existing tests break due to API changes | High | Medium | Each commit runs full test suite. Commit 6 updates all callers. |
| Performance regression for very short prompts (KV cache overhead > benefit) | Low | Low | Even at S=1, prefill+step is competitive with full forward. The cache write/read is O(H·d·n_layer) = negligible. |

---

## 8. Summary

The KV cache reduces per-step decode from O(S²) to O(S) by storing K and V
tensors for all past positions. With GPT-2 Small, this adds 72 MB of GPU
memory (total ~926 MB) and delivers an estimated **15–140× speedup**
depending on prompt length and number of decode steps.

**Key design decisions:**
1. Flat [n_layer, n_head, n_ctx, head_dim] layout — matches existing
   attention kernel stride patterns.
2. Same attention kernel code — the cached variant is a simpler unmasked
   kernel with grid(1, n_head) instead of grid(S, n_head).
3. Single workspace reused for both prefill and decode steps.
4. Separate code paths for prefill (full-S) and decode step (S=1) rather
   than conditional branches — clarity over concision.
5. API change: `KVCache&` parameter added to decode functions. Old
   full-recompute path deleted (identical output, strictly slower).

**7 atomic commits**, each compiling and passing all tests independently,
leading to a verified, golden-test-passing KV cache implementation.
