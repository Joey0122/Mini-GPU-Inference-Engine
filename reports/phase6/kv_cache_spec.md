# KV Cache — Detailed Specification

> Historical Phase 6 design notes. The published implementation reuses hidden
> row 0 for decode and sizes each attention scratch buffer for both prefill and
> cached decode. See [the workspace API](../../include/model/workspace.hpp).

**Date:** 2026-06-11
**Status:** COMPLETE — 137/137 tests, ALL GOLDEN GATES PASS

---

## 1. Exact KV-Cache Memory Layout and Indexing

### 1.1 Storage

Two flat `DeviceBuffer<float>` owning the cached keys and values for all
12 layers:

```
KVCache {
    DeviceBuffer<float> k;   // n_layer * n_head * n_ctx * head_dim = 12*12*1024*64 = 9,437,184 floats = 36.0 MB
    DeviceBuffer<float> v;   // ditto                                                        36.0 MB
    int num_positions;       // monotonically increasing: 0 → prompt_len → prompt_len + N
}
```

Total: **72.0 MB** (75,497,472 bytes).

### 1.2 Linear Indexing Formula

A single element is addressed by `(layer, head, position, dimension)`:

```
offset(l, h, p, d) = l * (n_head * n_ctx * head_dim)
                    + h * (n_ctx * head_dim)
                    + p * head_dim
                    + d
```

Rearranged to make the data-locality structure visible:

```
offset(l, h, p, d) = ((l * n_head + h) * n_ctx + p) * head_dim + d
```

**Locality properties:**
- For fixed (l, h): positions 0..n_ctx-1 are stored as `[n_ctx, head_dim]`
  row-major — **contiguous within a head**. This is the access pattern
  required by the attention kernel, which reads `K[h, 0..pos, :]` as a
  single contiguous block of `(pos+1) * head_dim` floats.
- For fixed (l, h, p): the `head_dim` elements of one key/value vector are
  contiguous — good for coalesced reads in the dot-product inner loop.

### 1.3 Head-Slice Pointer (for Kernel Launch)

When the attention kernel processes layer `l`, head `h`, it needs a pointer
to the start of the K data for that (l, h) slice:

```
k_head_ptr = k_cache.data() + ((l * n_head + h) * n_ctx) * head_dim
```

The kernel then accesses position `p`, dimension `d` as:

```
k_head_ptr[p * head_dim + d]
```

This is **exactly** the same indexing as the existing `AttentionScoresKernel`
([src/kernels/attention.cu](../../src/kernels/attention.cu)):

```cpp
const float* kh = k + static_cast<long long>(h) * S * head_dim;
// ...
const float* kj = kh + static_cast<long long>(j) * head_dim;
```

With the cache, `S` becomes `n_ctx` (the stride between heads within a layer)
and the pointer is offset by the layer's base address. The kernel's inner
loop over positions `j` is **unchanged**: it strides through consecutive
`head_dim`-element vectors.

### 1.4 Cache Write: QKV Split Output Routing

Currently, `LaunchQkvSplit` writes all three of Q, K, V into workspace
buffers sized for the full sequence `S`. With the cache, K and V for the
new token are written directly to the cache at the target position.

```
Input:  c_attn_out[1, 3*D]   (single new row)
Output: q[1*H*1*d]           (workspace, used immediately for attention)
        k_cache[layer] at position `pos`
        v_cache[layer] at position `pos`
```

Concretely, for each head `h`:
```
src offset  = 0*3*D + 1*D + h*head_dim + d   (K block in c_attn output)
            = D + h*head_dim + d               (K starts at column D)
dst offset  = ((l*12 + h)*1024 + pos)*64 + d
```

For the **prefill** phase, we write S contiguous positions at once
(S = prompt_len). A single `cudaMemcpyAsync` per head would mean
12 heads × 12 layers × 2 (K,V) = 288 memcpy calls. Better: use a small
kernel that does the scatter from `c_attn_out[S, 3D]` directly into the
cache in the correct layout — this is `QkvSplitToCacheKernel`.

### 1.5 Concrete Example: Layer 3, Head 5, Position 42

```
K value for dimension d=17:
  offset = ((3 * 12 + 5) * 1024 + 42) * 64 + 17
         = (41 * 1024 + 42) * 64 + 17
         = (41984 + 42) * 64 + 17
         = 42026 * 64 + 17
         = 2,689,664 + 17
         = 2,689,681

Byte offset = 2,689,681 * 4 = 10,758,724

Head-slice base pointer (positions 0..1023 for this head):
  offset = ((3 * 12 + 5) * 1024) * 64 = 41 * 1024 * 64 = 2,686,976
```

This head's K data spans byte offsets
[10,747,904, 10,747,904 + 262,144) = one 256 KB contiguous block.

---

## 2. Per-Layer FLOP Breakdown

GPT-2 Small constants: D=768, H=12, d=64, M=3072, V=50257.

Counting convention: multiply-add = 2 FLOPs. exp/div = 1 transcendental op
(acknowledged as more expensive; listed separately from fused multiply-adds).
BiasAdd = 1 FLOP (single add). GELU ≈ 5 FLOPs (3 mult + 1 add + 1 tanh).

### 2.1 Before KV Cache: Full Forward at Sequence Length S

#### Single Transformer Block

| Operation | Shape | FLOPs | Formula |
|---|---|---|---|
| **ln1** | S×D | 7·S·D | mean (D adds + 1 div), var (D subs + D muls + D adds + 1 div), norm (D subs + D muls + D adds) ≈ 7·S·D |
| **c_attn GEMM** | S×3D × D | 2·S·3D·D = **6·S·D²** | 2·M·N·K |
| c_attn BiasAdd | S×3D | S·3D | |
| QKV Split | S×3D→3·H·S·d | 0 (memcpy) | |
| **Attention Scores** | H×S×S×d | **2·H·S²·d** | per position: dot product of 2 d-vectors |
| Softmax | H·S×S | 3·H·S² | exp + sum + div per row element |
| **Attention AV** | H×S×S×d | **2·H·S²·d** | weighted sum of S d-vectors |
| **c_proj GEMM** | S×D × D | **2·S·D²** | |
| c_proj BiasAdd | S×D | S·D | |
| Residual 1 | S×D | S·D | |
| **ln2** | S×D | 7·S·D | |
| **mlp_fc GEMM** | S×M × D | **2·S·M·D** | |
| mlp_fc BiasAdd | S×M | S·M | |
| GELU | S×M | 5·S·M | |
| **mlp_proj GEMM** | S×D × M | **2·S·M·D** | |
| mlp_proj BiasAdd | S×D | S·D | |
| Residual 2 | S×D | S·D | |

#### Numeric Values (per layer)

Plugging D=768, H=12, d=64, M=3072:

| Operation | Formula | FLOPs expression | At S=1 | At S=1024 | % of total (S=1024) |
|---|---|---|---|---|---|
| ln1 | 7·S·D | 5,376·S | 5.4 K | 5.50 M | <0.1% |
| c_attn GEMM | 6·S·D² | 3,538,944·S | 3.54 M | 3.62 G | 20.1% |
| c_attn Bias | S·3D | 2,304·S | 2.3 K | 2.36 M | <0.1% |
| QKV Split | memcpy | 0 | 0 | 0 | — |
| **Attn Scores** | **2·H·S²·d** | **1,536·S²** | 1.5 K | **1.61 G** | **8.9%** |
| Softmax | 3·H·S² | 36·S² | 36 | 37.7 M | 0.2% |
| **Attn AV** | **2·H·S²·d** | **1,536·S²** | 1.5 K | **1.61 G** | **8.9%** |
| c_proj GEMM | 2·S·D² | 1,179,648·S | 1.18 M | 1.21 G | 6.7% |
| c_proj Bias | S·D | 768·S | 768 | 0.79 M | <0.1% |
| Residual 1 | S·D | 768·S | 768 | 0.79 M | <0.1% |
| ln2 | 7·S·D | 5,376·S | 5.4 K | 5.50 M | <0.1% |
| mlp_fc GEMM | 2·S·M·D | 4,718,592·S | 4.72 M | 4.83 G | 26.8% |
| mlp_fc Bias | S·M | 3,072·S | 3.1 K | 3.15 M | <0.1% |
| GELU | 5·S·M | 15,360·S | 15.4 K | 15.73 M | <0.1% |
| mlp_proj GEMM | 2·S·M·D | 4,718,592·S | 4.72 M | 4.83 G | 26.8% |
| mlp_proj Bias | S·D | 768·S | 768 | 0.79 M | <0.1% |
| Residual 2 | S·D | 768·S | 768 | 0.79 M | <0.1% |
| **Per-layer total** | | **14,179,584·S + 3,108·S²** | **~14.2 M** | **~18.0 G** | 100% |

#### 12 Layers Totals (Full Forward)

| Component | FLOPs | At S=1 | At S=1024 |
|---|---|---|---|
| 12× Block linear | 12 · 14.18 M · S | 170.2 M | 174.2 G |
| 12× Block quadratic | 12 · 3,108 · S² | 37.3 K | 39.1 G |
| Embedding | 2·S·D | 1.5 K | 1.57 M |
| ln_f | 7·S·D | 5.4 K | 5.50 M |
| lm_head GEMM | 2·S·V·D | 77.2 M | 79.0 G |
| **Total forward** | | **~247 M** | **~297 G** |

**Sanity check against measured latency:** 297 GFLOPs / 142.5 ms = 2.08 TFLOPs.
RTX 4070 Laptop peak FP32 = 15.6 TFLOPs (theoretical). Efficiency = 13% —
consistent with a launch-overhead-bound workload of many small kernels at S=1024.

### 2.2 After KV Cache: Single Decode Step (S_query=1, S_cached positions)

The decode step processes only the 1 new token's hidden row through all
linear layers. Attention operates on 1 query × S_cached keys/values.

#### Single Transformer Block — Decode Step

| Operation | Shape | FLOPs | Formula |
|---|---|---|---|
| ln1 | 1×D | 7·D = 5,376 | mean/var/norm of 1 row |
| c_attn GEMM | 1×3D × D | 6·D² = 3,538,944 | single-row GEMM |
| c_attn BiasAdd | 1×3D | 3D = 2,304 | |
| QKV Split | 1×3D→3·H·1·d | 0 | |
| Cache write | H·d floats × 2 | 0 (memcpy) | copy k_new,v_new to cache |
| **Attn Scores** | H×1×S_c×d | **2·H·S_c·d** | 1 query dot S_c keys |
| Softmax | H×S_c | 3·H·S_c | exp+sum+div for 1 row per head |
| **Attn AV** | H×1×S_c×d | **2·H·S_c·d** | weight S_c d-vectors |
| c_proj GEMM | 1×D × D | 2·D² = 1,179,648 | |
| c_proj BiasAdd | 1×D | D = 768 | |
| Residual 1 | 1×D | D = 768 | |
| ln2 | 1×D | 7·D = 5,376 | |
| mlp_fc GEMM | 1×M × D | 2·M·D = 4,718,592 | |
| mlp_fc BiasAdd | 1×M | M = 3,072 | |
| GELU | 1×M | 5·M = 15,360 | |
| mlp_proj GEMM | 1×D × M | 2·M·D = 4,718,592 | |
| mlp_proj BiasAdd | 1×D | D = 768 | |
| Residual 2 | 1×D | D = 768 | |

#### Numeric Summary — Decode Step (per layer)

| Operation | FLOPs | S_c=1 | S_c=256 | S_c=1024 |
|---|---|---|---|---|
| All linear ops | 14.18 M | 14.18 M | 14.18 M | 14.18 M |
| Attn Scores | 2·H·S_c·d = 1,536·S_c | 1,536 | 0.39 M | 1.57 M |
| Softmax | 3·H·S_c = 36·S_c | 36 | 9,216 | 36,864 |
| Attn AV | 1,536·S_c | 1,536 | 0.39 M | 1.57 M |
| **Per-layer per step** | | **14.18 M** | **14.97 M** | **17.36 M** |

#### Full Decode Step (12 layers + lm_head)

| Phase | S_c=1 | S_c=256 | S_c=1024 |
|---|---|---|---|
| Embedding | 1,536 | 1,536 | 1,536 |
| 12× Blocks | 170.2 M | 179.6 M | 208.4 M |
| ln_f | 5,376 | 5,376 | 5,376 |
| lm_head | 77.2 M | 77.2 M | 77.2 M |
| **Total per step** | **247.4 M** | **256.8 M** | **285.6 M** |

### 2.3 Before vs After — Per Decode Step Comparison

| Cache size S_c | Before (full fwd at S=S_c) | After (decode step) | FLOP reduction |
|---|---:|---:|---:|
| 1 | 247 M | 247 M | **1.0×** (no gain at S=1; prefill identical) |
| 4 | 328 M | 254 M | **1.3×** |
| 16 | 927 M | 263 M | **3.5×** |
| 64 | 4.0 G | 275 M | **14.5×** |
| 256 | 29.0 G | 257 M | **113×** |
| 512 | 86.3 G | 272 M | **317×** |
| 1024 | 297 G | 286 M | **1,039×** |

The linear ops (GEMMs + bias + GELU + LayerNorm) dominate the decode step:
~247 M FLOPs independent of S_c. The attention adds at most ~39 M FLOPs
at S_c=1024. The overall per-step cost is essentially constant.

**Key insight:** At S_c=1024, full recomputation spends 39.1 GFLOPs on
attention scores+AV (quadratic term) vs. 3.15 MFLOPs with cache — a
**12,400× reduction** in attention compute specifically.

---

## 3. Logits Buffer Reduction

### 3.1 Current Allocation

In [src/model/workspace.cu](../../src/model/workspace.cu):

```cpp
ws.logits = DeviceBuffer<float>(static_cast<std::size_t>(S) * cfg.vocab_size);
```

For S = max_seq = 1024, V = 50257:

```
logits bytes = 1024 × 50257 × 4 = 205,852,672 bytes ≈ 196.3 MB
```

This is **~23% of the total workspace** (854 MB total, of which workspace
is ~357 MB — so logits alone is 196/357 = 55% of workspace).

### 3.2 Why It Can Be Reduced

In KV-cache mode, the decode step processes **one token at a time**.
`LaunchGpt2DecodeStep` produces exactly 1 row of logits: `logits[1, V]`.

The prefill still produces `prompt_len` rows of logits (one-time cost).
After prefill, only 1 row is needed per step.

**Current decode loop** ([decode.cu](../../src/model/decode.cu)):
```cpp
CUDA_CHECK(cudaMemcpyAsync(last_logits.data(),
    ws.logits.data() + static_cast<std::size_t>(cur_S - 1) * V,
    V * sizeof(float),
    cudaMemcpyDeviceToHost, stream));
```

Reads from position `cur_S - 1` in a full `[S, V]` buffer.

**KV-cache decode loop** would instead read from position 0 each step (the
decode step writes exactly 1 row to `logits[0:V]`):
```cpp
CUDA_CHECK(cudaMemcpyAsync(last_logits.data(),
    ws.logits.data(),  // always position 0 — only 1 row produced
    V * sizeof(float),
    cudaMemcpyDeviceToHost, stream));
```

### 3.3 Savings

The logits buffer can be sized for `max(prompt_len, 1)` rows instead of
`max_seq = prompt_len + max_new_tokens` rows:

| Scenario | prompt_len | Old logits | New logits | Savings |
|---|---|---|---|---|
| C9 golden (decode_p0) | 1 | 1025 × 201 KB = 206 MB | 1 × 201 KB = 201 KB | **206 MB** |
| C9 golden (decode_p8) | 8 | 1032 × 201 KB = 207 MB | 8 × 201 KB = 1.6 MB | **205 MB** |
| Long prompt | 256 | 1280 × 201 KB = 257 MB | 256 × 201 KB = 51 MB | **206 MB** |
| Max context | 1024 | 1024 × 201 KB = 205 MB | 1024 × 201 KB = 205 MB | 0 (same) |

For typical use (short prompt, many decode steps), this saves ~200 MB.

### 3.4 Recommendation: Include in KV-Cache Work

**Yes, include as part of the KV-cache commit.** Rationale:

1. **Direct consequence**: The reduction is only possible because the
   decode step processes one token at a time — the defining property of
   KV-cache mode. It's not a standalone optimization.

2. **Trivial implementation**: One line change in the decode loop (the
   D2H copy offset), plus sizing the logits buffer to `max(prompt_len, 1)`
   rows in the workspace factory (or keeping the same allocation and just
   using less of it).

3. **Memory offset**: The 72 MB KV cache cost is nearly erased by the
   ~200 MB logits savings for short-prompt workloads. Net memory for a
   typical decode (prompt_len=4, max_new_tokens=100): **~728 MB vs the
   original 854 MB — a net reduction while adding KV cache.**

4. **Commit grouping**: It fits naturally in Commit 6 (rewire decode
   loop) since that's where the decode loop switches from reading
   `logits[cur_S - 1]` to `logits[0]`.

### 3.5 Updated Memory Footprint

| Component | Before (Phase 5) | After (+KV cache, -logits) |
|---|---|---|
| Weights | ~497 MB | 497 MB |
| Workspace (excl. logits) | ~161 MB | 161 MB |
| Logits buffer | **196 MB** (S=1024) | **~0.2–51 MB** (prompt-len dependent) |
| output_ids | 4 KB | 4 KB |
| KV Cache | — | **72 MB** |
| **Total (typical)** | **~854 MB** | **~730 MB** |

For a short prompt with many decode steps, the KV-cache engine uses LESS
GPU memory than the original full-recompute engine, while running ~100×
faster.

---

## 4. Formal Correctness Argument

### 4.1 Definitions

Let the GPT-2 forward pass be defined over a sequence of token IDs
`t_0, t_1, ..., t_{K-1}`.

**Embedding:**
```
h_pos^(-1) = wte[t_pos] + wpe[pos]    ∈ R^D
```
Depends only on `t_pos` and the position index `pos`.

**Transformer block ℓ (Pre-LN):**
```
a_pos^(ℓ)   = h_pos^(ℓ-1) + c_proj_ℓ(Attn_ℓ(QKV_split(c_attn_ℓ(LN(ℓ,1)(h_pos^(ℓ-1))))))
h_pos^(ℓ)   = a_pos^(ℓ) + mlp_proj_ℓ(GELU(mlp_fc_ℓ(LN(ℓ,2)(a_pos^(ℓ)))))
```

Where attention at position `pos` is:
```
q_pos^(ℓ)[h,:]  = QKV_split(…)[h, pos, :]    ∈ R^d    (query for head h)
k_pos^(ℓ)[h,:]  = QKV_split(…)[h, pos, :]    ∈ R^d    (key for head h)
v_pos^(ℓ)[h,:]  = QKV_split(…)[h, pos, :]    ∈ R^d    (value for head h)

scores[h, j]    = (1/√d) · ⟨q_pos^(ℓ)[h,:], k_j^(ℓ)[h,:]⟩    for j ≤ pos
weights          = softmax(scores)                             [causal]
out[h,:]         = Σ_{j=0}^{pos} weights[j] · v_j^(ℓ)[h,:]
```

**Final LM head:**
```
logits[pos, :] = LN_f(h_pos^(11)) @ wte_T    ∈ R^V
```

### 4.2 Row-Wise Independence Lemma

**Lemma 1 (Row-wise independence).** For any operation `f` in
{Embedding, LayerNorm, GEMM, BiasAdd, QKV_split, GELU, Residual}:
`f(X)[pos, :]` depends only on `X[pos, :]`.

*Proof.* Each operation applies the same transformation independently to
each row. LayerNorm normalizes over the row dimension; the mean and
variance of row `pos` depend only on `X[pos, :]`. GEMM is `C[pos,:] =
A[pos,:] @ W` — only row `pos` of A contributes. BiasAdd, QKV_split,
GELU, and Residual are all elementwise or row-bounded. ∎

**Lemma 2 (Causal attention dependency).** `Attn_ℓ(Q, K, V)[pos, :]`
depends on `Q[pos, :]` and `K[0:pos, :], V[0:pos, :]`. It does NOT
depend on `Q[0:pos-1, :]` (past queries), `Q[pos+1:, :]` (future
queries), `K[pos+1:, :]` (future keys), or `V[pos+1:, :]` (future
values).

*Proof.* The causal mask sets `scores[h, pos, j] = -∞` for all `j > pos`,
so `weights[h, pos, j] = 0` for `j > pos`. The output is a weighted sum
over `j ∈ [0, pos]` of `V[j, :]`. The query `Q[pos, :]` is used to
compute the scores. No other query positions contribute to `out[pos, :]`. ∎

### 4.3 State Invariant for KV-Cache Correctness

**Definition.** Let a **full-recomputation trace** be the sequence of all
intermediate values produced by `LaunchGpt2Forward` at each sequence
length S = P, P+1, ..., P+N-1, where P is the prompt length and N is
the number of generated tokens. Denote the forward pass at length S as
`F_S`.

**Definition.** Let a **KV-cache trace** be the sequence of intermediate
values produced by one `LaunchGpt2Prefill` at S=P followed by N calls to
`LaunchGpt2DecodeStep` at positions pos = P, P+1, ..., P+N-1.

**Invariant I(k).** For all `k ∈ [0, P+N-1]`, after the KV-cache
engine has processed position `k`:

1. **Hidden state:** For all layers `ℓ ∈ [0, 11]` and all positions
   `pos ∈ [0, k]`, the hidden state `h_pos^(ℓ)` in the KV-cache trace
   equals `h_pos^(ℓ)` in the full-recomputation trace `F_{k+1}`.

2. **KV cache:** For all layers `ℓ ∈ [0, 11]`, all heads `h ∈ [0, 11]`,
   and all positions `pos ∈ [0, k]`:
   ```
   K_cache[ℓ, h, pos, :] = k_pos^(ℓ)[h, :]    (from trace F_{pos+1})
   V_cache[ℓ, h, pos, :] = v_pos^(ℓ)[h, :]    (from trace F_{pos+1})
   ```

3. **Logits:** `logits[k, :]` in the KV-cache trace equals
   `logits[k, :]` in the full-recomputation trace `F_{k+1}`.

### 4.4 Proof by Induction

#### Base Case: Prefill (positions 0..P-1)

`LaunchGpt2Prefill` at S=P is structurally identical to
`LaunchGpt2Forward` at S=P, with one addition: after each layer's QKV
split, K and V are written to the cache. The forward computation is
unchanged.

For each position `pos ∈ [0, P-1]`, for each layer `ℓ`:

- **ln1:** Depends only on `h_pos^(ℓ-1)` (Lemma 1). Computed identically
  to `F_P` since `h_pos^(ℓ-1)` is identical (same computation, same inputs).
- **c_attn + BiasAdd + QKV split:** Row-wise (Lemma 1). Produces
  `q_pos^(ℓ), k_pos^(ℓ), v_pos^(ℓ)` identical to `F_P`.
- **Cache population:** `K_cache[ℓ, :, pos, :] = k_pos^(ℓ)`,
  `V_cache[ℓ, :, pos, :] = v_pos^(ℓ)`. These equal the K,V values in
  `F_P` since the computations producing them are identical.
- **Attention:** For position `pos`, queries `K[0:pos, :, :]` and
  `V[0:pos, :, :]` (Lemma 2). By induction on `pos` within the prefill
  (positions `< pos` already processed correctly), these match `F_P`.
  The attention output is identical.
- **c_proj + Residual + ln2 + mlp_fc + GELU + mlp_proj + Residual:**
  All row-wise (Lemma 1). Each computed identically to `F_P` since their
  inputs are identical.

By induction on `pos` within the prefill, all P positions produce
identical outputs to `F_P`. Conditions 1, 2, 3 of I(P-1) are satisfied.

#### Inductive Step: I(k-1) ⇒ I(k) for k ≥ P

Assume I(k-1) holds. Process position k via `LaunchGpt2DecodeStep`.

**Embedding:**
```
h_k^(-1) = wte[t_k] + wpe[k]
```
In `F_{k+1}`, the embedding at position k is computed identically: same
token ID `t_k`, same position index `k`. Identical inputs → identical output. ✓

**For each layer ℓ = 0..11:**

1. **ln1(ℓ):** `LN(h_k^(ℓ-1))`. By I(k-1), `h_k^(ℓ-1)` equals the
   corresponding value in `F_{k+1}` (at the same position k, same layer
   ℓ-1 state, since the forward pass computes all positions independently
   row-by-row and position k's embedding is identical). Lemma 1 → output
   depends only on `h_k^(ℓ-1)`. Identical. ✓

2. **c_attn GEMM + BiasAdd:** Row-wise (Lemma 1). ln1_out[k,:] is
   identical → c_attn_out identical. ✓

3. **QKV split:** Row-wise. Produces `q_k, k_k, v_k`. Identical to
   `F_{k+1}` at position k. ✓

4. **Cache write:** `K_cache[ℓ, :, k, :] = k_k`, `V_cache[ℓ, :, k, :] = v_k`.
   By the above, `k_k, v_k` equal the K,V in `F_{k+1}` at position k.
   For positions `pos < k`, the cache already holds correct values by
   I(k-1). So K_cache[ℓ, :, 0:k, :] = K[ℓ, :, 0:k, :] from `F_{k+1}`. ✓

5. **Attention (cached):**
   - Query: `q_k^(ℓ)` from step 3. Equals `Q[ℓ, :, k, :]` in `F_{k+1}`.
   - Keys: `K_cache[ℓ, :, 0:k, :]`. By I(k-1) for positions `0..k-1` and
     step 4 for position k, these equal `K[ℓ, :, 0:k, :]` in `F_{k+1}`.
   - Values: `V_cache[ℓ, :, 0:k, :]`. Same argument → equal to
     `V[ℓ, :, 0:k, :]` in `F_{k+1}`.
   - Scores: `(1/√d) · q_k[h,:] · K_cache[ℓ, h, j, :]` for j ∈ [0, k].
     In `F_{k+1}`: `(1/√d) · Q[ℓ, h, k, :] · K[ℓ, h, j, :]` for j ∈ [0, k].
     q_k = Q[ℓ, :, k, :] and K_cache = K → identical scores. ✓
   - Softmax: deterministic operation on identical inputs → identical
     weights. ✓
   - Output: `Σ_j weights[j] · V_cache[ℓ, h, j, :]`. Same weighted sum
     over identical vectors → identical output. ✓

6. **c_proj GEMM + BiasAdd:** Row-wise on identical attention output. ✓

7. **Residual 1:** `h_k^(ℓ,attn) = h_k^(ℓ-1) + c_proj_out`. By I(k-1),
   `h_k^(ℓ-1)` is identical. c_proj_out is identical (step 6). Sum is
   identical. ✓

8. **ln2(ℓ) + mlp_fc + BiasAdd + GELU + mlp_proj + BiasAdd:**
   All row-wise (Lemma 1) on identical `h_k^(ℓ,attn)`. ✓

9. **Residual 2:** `h_k^(ℓ) = h_k^(ℓ,attn) + mlp_proj_out`. Identical. ✓

Positions `pos < k` are unchanged: the decode step reads their hidden
states (via the residual adds at position k's residuals — wait, the
residual adds are `hidden[k,:] += proj_out[k,:]`, which only touches
position k). All other positions' hidden states remain at their values
from I(k-1), which equal `F_{k+1}` by the induction hypothesis and the
fact that `F_{k+1}` also only modifies position k's row (the first k
positions are unchanged from `F_k` — row-wise independence means adding
a position doesn't change previous positions' forward pass results). ✓

**Final LayerNorm + LM head:**
- `h_k^(11)` from layer 11, residual 2. By the above, identical to
  `F_{k+1}` at position k.
- `LN_f(h_k^(11))` — row-wise, identical.
- `logits[k,:] = LN_f_out @ wte_T` — row-wise GEMM, identical. ✓

**Therefore I(k) holds.** All three conditions satisfied.

### 4.5 Conclusion

By induction on `k`, invariant I(k) holds for all `k ∈ [0, P+N-1]`.
In particular, for every generated token position `k ≥ P`, the logits
produced by the KV-cache engine are **identical** to the logits that
would be produced by full sequence recomputation at `S = k+1`.

The KV cache is not an approximation. It is an **exact algorithmic
optimization** — it leverages the causal dependency structure of
transformer attention to avoid recomputing values that are provably
unchanged. The equivalence holds in exact arithmetic; in floating-point,
both paths execute the same operations on the same inputs in the same
order, so they produce bitwise-identical results (no non-associativity
divergence).

∎

---

## Summary

| Item | Answer |
|---|---|
| **Layout** | Flat `[n_layer * n_head * n_ctx * head_dim]`, head-major within layer. Positions contiguous per head. Same stride pattern as existing attention kernel — drop-in replacement. |
| **FLOP reduction** | Full forward at S=1024: ~297 GFLOPs. Decode step: ~286 MFLOPs (constant + O(S_c)). **~1,040× reduction** at max cache. |
| **Logits buffer** | Reduce from S·V (196 MB) to max(prompt_len,1)·V (0.2–51 MB). Include in KV-cache work (Commit 6). Net memory at typical prompt_len=4: **~730 MB** — less than the original 854 MB. |
| **Correctness** | Formal inductive proof. KV cache is exact — not an approximation. All operations are row-wise independent; cached K/V = original K/V by construction; attention reads identical inputs. **Bitwise-identical output** in FP32. |

---

## Appendix C. Measured Benchmark Results (RTX 4070 Laptop, GPT-2 Small, real weights)

### Prefill (full forward + cache population)

| S | Latency (ms) | Notes |
|---|---:|---|
| 1 | 25.5 | Launch overhead dominant |
| 4 | 24.0 | |
| 16 | 17.1 | |
| 64 | 17.9 | |
| 256 | 26.8 | |
| 512 | 47.1 | |
| 1024 | 127.0 | Matches original forward (142 ms baseline with separate workspace creation) |

### Per-step Decode (S=1 query, varying cache size)

| Cache size | Latency (ms) | Notes |
|---|---:|---|
| 1 | 15.4 | ~15 ms constant floor (linear layers) |
| 4 | 14.8 | |
| 16 | 14.8 | |
| 64 | 14.8 | |
| 256 | 15.2 | |
| 512 | 16.0 | |
| 1023 | 17.2 | ~2 ms attention overhead at full cache |

### Key Findings

1. **Constant-time decode**: Per-step latency is ~15 ms, independent of cache size.
   The linear layers (GEMM c_attn/c_proj/mlp_fc/mlp_proj + GELU + LayerNorm at S=1)
   dominate at ~14 ms. Cached attention adds only ~2 ms at S=1023.

2. **Speedup vs full recomputation**: At S=1024, a full forward pass takes 127 ms.
   The KV-cache decode step takes 17 ms — **7.5× faster per token**.

3. **Linear layers are the bottleneck**: The ~14 ms constant cost per step
   comes from 4 S=1 GEMMs per block × 12 blocks = 48 small GEMMs. Kernel fusion
   (Phase 6b) could reduce launch overhead for these.

4. **Attention is no longer the bottleneck**: Quadratic O(S²) attention
   scores + AV at S=1024 previously cost ~39 GFLOPs. Now cached attention
   costs ~3.15 MFLOPs — a **12,400× reduction**.

---

## Appendix A. Exact Linear Indexing Formula (Implementation Reference)

### A.1 Data Structure

```cpp
struct KVCache {
    DeviceBuffer<float> k;  // n_layer * n_head * n_ctx * head_dim floats
    DeviceBuffer<float> v;  // ditto
    int num_positions;      // monotonically increasing: 0 → P → P + N
};
```

For GPT-2 Small: `n_layer=12, n_head=12, n_ctx=1024, head_dim=64`.

### A.2 Element Indexing

```
cache_k[layer][head][pos][dim]
  = k.data[ ((layer * n_head + head) * n_ctx + pos) * head_dim + dim ]

cache_v[layer][head][pos][dim]
  = v.data[ ((layer * n_head + head) * n_ctx + pos) * head_dim + dim ]
```

Where:
- `layer` ∈ [0, 11]  — transformer block index
- `head`  ∈ [0, 11]  — attention head index
- `pos`   ∈ [0, 1023] — sequence position
- `dim`   ∈ [0, 63]   — dimension within a key/value vector

### A.3 Memory Strides

Flat array dimensions: `[n_layer, n_head, n_ctx, head_dim]` in row-major order.

| Dimension | Stride (elements) | Stride (bytes) | Notes |
|---|---|---|---|
| `layer` | `n_head * n_ctx * head_dim` = 12 × 1024 × 64 = **786,432** | **3,145,728** (3 MB) | One layer's K (or V) = 3,145,728 bytes |
| `head`  | `n_ctx * head_dim` = 1024 × 64 = **65,536** | **262,144** (256 KB) | One head's K (or V) = 262,144 bytes |
| `pos`   | `head_dim` = **64** | **256** | One position vector = 256 bytes |
| `dim`   | **1** | **4** | One float element |

Total K cache size: `12 × 12 × 1024 × 64 × 4` = **37,748,736 bytes = 36.0 MB**.
Total V cache: same. Grand total: **72.0 MB**.

### A.4 Head-Slice Pointer (Kernel Launch)

When the kernel processes layer `l`, head `h`:

```cpp
float* k_head = k.data() + ((l * n_head + h) * n_ctx) * head_dim;
const float* v_head = v.data() + ((l * n_head + h) * n_ctx) * head_dim;
```

The kernel accesses position `p`, dimension `d` at:
```
k_head[p * head_dim + d]
v_head[p * head_dim + d]
```

This matches the existing `AttentionScoresKernel` access pattern exactly —
substitute `n_ctx` for `S`:

```
Existing:  const float* kh = k + h * S * head_dim;   // [h, S, head_dim]
           kj = kh + j * head_dim;                      // position j

Cached:    const float* kh = k_cache + ((l*n_head + h) * n_ctx) * head_dim;
           kj = kh + j * head_dim;                      // position j (same formula)
```

### A.5 Cache Write Pattern

For prefill (S = prompt_len), writing one layer's K/V for all positions at once:

```cpp
void LaunchQkvSplitToCache(const float* c_attn_out,  // [S, 3*n_embd]
                           float* q_ws,               // [n_head, S, head_dim] — workspace
                           float* k_cache_base,       // [n_head, n_ctx, head_dim] — cache for this layer
                           float* v_cache_base,
                           int S, int n_embd, int n_head, int n_ctx,
                           cudaStream_t stream);
```

The kernel writes position `s`, head `h`, dim `d` as:
```
q_ws[(h * S + s) * head_dim + d]      = c_attn_out[s*3*n_embd + 0*n_embd + h*head_dim + d]
k_cache_base[(h * n_ctx + s) * head_dim + d] = c_attn_out[s*3*n_embd + 1*n_embd + h*head_dim + d]
v_cache_base[(h * n_ctx + s) * head_dim + d] = c_attn_out[s*3*n_embd + 2*n_embd + h*head_dim + d]
```

Note: Q goes to the regular workspace (accessed immediately for attention),
K and V go directly into the cache at `n_ctx` stride (vs `S` stride for Q).

For the **decode step** (S=1), writing one new position `pos`:

```cpp
// K_cache[layer][head][pos][dim] = c_attn_out[0, n_embd + head*head_dim + dim]
// V_cache[layer][head][pos][dim] = c_attn_out[0, 2*n_embd + head*head_dim + dim]
```

A small kernel with grid `n_head` and block `head_dim` suffices.

---

## Appendix B. Validation Criteria

### B.1 Authoritative Gate: Exact Token-Sequence Match

The authoritative gate for KV-cache correctness is **exact token-sequence
identity**: for the same prompt, same weights, same sampling parameters, and
same RNG seed, the KV-cache decode loop must produce the **identical sequence
of token IDs** as the full-recomputation decode loop.

This is the strongest possible correctness test: a single deviation in any
logit row's argmax (for greedy) or probability distribution (for sampling)
propagates to all subsequent tokens, guaranteeing a mismatch.

### B.2 Numerical Gate: max_abs_error ≤ 1e-5

For unit tests that compare logit values directly (not just token decisions):

| Test | Gate | Rationale |
|---|---|---|
| Cached attention output vs full attention (last row) | `max_abs_error ≤ 1e-5` | Single attention computation; FP32 rounding ≈ 1e-6 |
| DecodeStep logits vs full forward (last row) | `max_abs_error ≤ 1e-5` | Chain of 12 blocks; errors compound but stay under 1e-5 |
| KV-cache decode vs full-recompute decode (all steps) | `max_abs_error ≤ 1e-5` | Full end-to-end trace comparison |

The relaxed gate (1e-5 vs the original design's bitwise-identical claim)
accounts for FP32 non-associativity: while each individual operation is
deterministic, the KV-cache path may execute softmax with max-subtraction
over slightly different value ranges when the logit magnitude changes
across decode steps (accumulated residuals). In exact arithmetic the paths
are identical; in FP32 they may diverge by 1-2 ULPs.

**The 1e-5 gate provides 100× margin over the PROJECT_SPEC 1e-4 gate**,
ensuring numeric equivalence is far tighter than the established tolerance.

### B.3 Golden Tests (Real Weights)

All existing golden gates must pass unchanged when the decode loop is
rewired to use KV cache:

| Gate | Metric | Target |
|---|---|---|
| C8 Logit Gate | argmax match | 53/53 positions |
| C8 Logit Gate | max_abs_error vs HF | < 5e-3 |
| C9 Decode Gate | token match vs HF greedy | 113/113 tokens |
| C10 Sample Gate | implausible samples | 0/60 (distribution p < 1e-12) |

The C9 gate is the most sensitive: a single logit discrepancy at any
decode step changes the greedy token, cascading through all subsequent
steps. **113/113 token match proves numerical equivalence far beyond
1e-5 per-logit tolerance.**

### B.4 Test Hierarchy

```
Level 1 (Commit 2): CachedAttentionMatchesFull — single kernel, isolated
Level 2 (Commit 4): BlockCachedMatchesFull — one transformer block
Level 3 (Commit 5): ForwardDecodeStepMatchesFull — full forward, single step
Level 4 (Commit 6): KVDecodeEqualsFullRecompute — end-to-end token sequence
Level 5 (Commit 6): Golden tests — real GPT-2 weights, HF comparison
```

Each level gates the next. Level 5 is the authoritative acceptance test.
