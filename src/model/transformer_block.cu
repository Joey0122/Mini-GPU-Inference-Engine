// src/model/transformer_block.cu
// single GPT-2 transformer block — kernel orchestration.
//
// Chains the existing sub-kernels exactly as a GPT-2 Pre-LN block:
//   ln_1  →  c_attn  →  QKV split →  attention  →  c_proj  →  residual
//   ln_2  →  mlp_fc  →  GELU      →  mlp_proj  →  residual
//
// Every Conv1D weight is stored as [K, N] = [in_features, out_features],
// matching LaunchGemm's B[K,N] convention. No transpose is ever needed.
//
// all scratch buffers come from the caller-provided workspace. No
// per-call allocation — the same workspace is reused across all 12 layers.

#include "model/transformer_block.hpp"

#include "common/errors.hpp"
#include "cuda/memory.hpp"
#include "kernels/attention.hpp"
#include "kernels/bias_add.hpp"
#include "kernels/gelu.hpp"
#include "kernels/gemm.hpp"
#include "kernels/layernorm.hpp"
#include "kernels/qkv_split.hpp"
#include "kernels/vector_add.hpp"

namespace mgie {

void LaunchTransformerBlock(float* hidden,
                            const BlockWeights& blk,
                            TransformerBlockWorkspace& ws,
                            const Gpt2Config& cfg,
                            int S,
                            cudaStream_t stream) {
  if (S <= 0) return;

  const int D = cfg.n_embd;       // 768
  const int H = cfg.n_head;       // 12
  const int d = cfg.head_dim;     // 64
  const int M = cfg.mlp_inner;    // 3072
  const float eps = cfg.ln_eps;   // 1e-5

  // ---- workspace aliases (raw pointers from pre-allocated buffers) ------------

  float* ln1_out    = ws.ln1_out.data();
  float* c_attn_out = ws.c_attn_out.data();
  float* q          = ws.q.data();
  float* k          = ws.k.data();
  float* v          = ws.v.data();
  float* scores     = ws.scores.data();
  float* weights    = ws.weights.data();
  float* attn_out   = ws.attn_out.data();
  float* proj_out   = ws.proj_out.data();
  float* ln2_out    = ws.ln2_out.data();
  float* fc_out     = ws.fc_out.data();

  // ---- attention path --------------------------------------------------------
  // 1. ln_1: LayerNorm(hidden) -> ln1_out [S, D]
  LaunchLayerNorm(hidden, ln1_out,
                  blk.ln1_w.data(), blk.ln1_b.data(),
                  S, D, eps, LayerNormVersion::Naive, stream);

  // 2. c_attn: ln1_out @ c_attn_w [D, 3D] + c_attn_b [3D] -> c_attn_out [S, 3D]
  LaunchGemm(ln1_out, blk.c_attn_w.data(), c_attn_out,
             S, 3 * D, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(c_attn_out, blk.c_attn_b.data(), c_attn_out,
                S, 3 * D, stream);  // in-place

  // 3. QKV split: c_attn_out [S, 3D] -> q,k,v [H, S, d]
  LaunchQkvSplit(c_attn_out, q, k, v, S, D, H, stream);

  // 4. Attention: q,k,v -> attn_out [S, D] (heads merged)
  LaunchAttention(q, k, v, attn_out, scores, weights, H, S, d, stream);

  // 5. c_proj: attn_out @ c_proj_w [D, D] + c_proj_b [D] -> proj_out [S, D]
  LaunchGemm(attn_out, blk.c_proj_w.data(), proj_out,
             S, D, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(proj_out, blk.c_proj_b.data(), proj_out, S, D, stream);

  // 6. Residual 1: hidden += proj_out  (elementwise, in-place)
  LaunchVectorAdd(hidden, proj_out, hidden, S * D, stream);

  // ---- MLP path --------------------------------------------------------------
  // 7. ln_2: LayerNorm(hidden) -> ln2_out [S, D]
  LaunchLayerNorm(hidden, ln2_out,
                  blk.ln2_w.data(), blk.ln2_b.data(),
                  S, D, eps, LayerNormVersion::Naive, stream);

  // 8. mlp_fc: ln2_out @ mlp_fc_w [D, M] + mlp_fc_b [M] -> fc_out [S, M]
  LaunchGemm(ln2_out, blk.mlp_fc_w.data(), fc_out,
             S, M, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(fc_out, blk.mlp_fc_b.data(), fc_out, S, M, stream);

  // 9. GELU: fc_out -> fc_out (in-place, elementwise)
  LaunchGelu(fc_out, fc_out, S * M, stream);

  // 10. mlp_proj: fc_out @ mlp_proj_w [M, D] + mlp_proj_b [D] -> proj_out [S, D]
  LaunchGemm(fc_out, blk.mlp_proj_w.data(), proj_out,
             S, D, M, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(proj_out, blk.mlp_proj_b.data(), proj_out, S, D, stream);

  // 11. Residual 2: hidden += proj_out
  LaunchVectorAdd(hidden, proj_out, hidden, S * D, stream);
}

// ---- Prefill (full block + cache population) ----------------------
// Runs the full block identically to LaunchTransformerBlock (all intermediate
// tensors at S-stride). After the QKV split, copies the per-position K and V
// vectors from the workspace into the KV cache at n_ctx-stride for future
// decode steps. The extra copy is a one-time cost (prefill → ~0.1% overhead).

void LaunchTransformerBlockPrefill(float* hidden,
                                    const BlockWeights& blk,
                                    TransformerBlockWorkspace& ws,
                                    KVCache& kv_cache,
                                    int layer_idx,
                                    const Gpt2Config& cfg,
                                    int S,
                                    cudaStream_t stream) {
  if (S <= 0) return;

  const int D = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim;
  const int M = cfg.mlp_inner;
  const float eps = cfg.ln_eps;
  const int n_ctx = kv_cache.n_ctx();

  float* ln1_out    = ws.ln1_out.data();
  float* c_attn_out = ws.c_attn_out.data();
  float* q          = ws.q.data();
  float* k_ws       = ws.k.data();
  float* v_ws       = ws.v.data();
  float* scores     = ws.scores.data();
  float* weights    = ws.weights.data();
  float* attn_out   = ws.attn_out.data();
  float* proj_out   = ws.proj_out.data();
  float* ln2_out    = ws.ln2_out.data();
  float* fc_out     = ws.fc_out.data();

  // --- Attention path (identical to original block, using workspace K/V) ---
  LaunchLayerNorm(hidden, ln1_out,
                  blk.ln1_w.data(), blk.ln1_b.data(),
                  S, D, eps, LayerNormVersion::Naive, stream);

  LaunchGemm(ln1_out, blk.c_attn_w.data(), c_attn_out,
             S, 3 * D, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(c_attn_out, blk.c_attn_b.data(), c_attn_out,
                S, 3 * D, stream);

  // Regular QKV split: Q, K, V all to workspace (S-stride).
  LaunchQkvSplit(c_attn_out, q, k_ws, v_ws, S, D, H, stream);

  // Copy K and V from workspace (S-stride) to KV cache (n_ctx-stride).
  // One cudaMemcpyAsync per head — 12 heads × 2 (K,V) = 24 memcpys.
  // Prefill is a one-time cost; the overhead is negligible.
  float* k_cache_base = kv_cache.k_head(layer_idx, 0);
  float* v_cache_base = kv_cache.v_head(layer_idx, 0);
  const std::size_t head_stride_ws = static_cast<std::size_t>(S) * d;
  const std::size_t head_stride_kv = static_cast<std::size_t>(n_ctx) * d;
  for (int h = 0; h < H; ++h) {
    CUDA_CHECK(cudaMemcpyAsync(
        k_cache_base + h * head_stride_kv,
        k_ws + h * head_stride_ws,
        head_stride_ws * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        v_cache_base + h * head_stride_kv,
        v_ws + h * head_stride_ws,
        head_stride_ws * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
  }

  // Regular attention from workspace Q, K, V (S-stride).
  LaunchAttention(q, k_ws, v_ws, attn_out,
                  scores, weights, H, S, d, stream);

  LaunchGemm(attn_out, blk.c_proj_w.data(), proj_out,
             S, D, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(proj_out, blk.c_proj_b.data(), proj_out, S, D, stream);

  LaunchVectorAdd(hidden, proj_out, hidden, S * D, stream);

  // --- MLP path (identical to original) ---
  LaunchLayerNorm(hidden, ln2_out,
                  blk.ln2_w.data(), blk.ln2_b.data(),
                  S, D, eps, LayerNormVersion::Naive, stream);

  LaunchGemm(ln2_out, blk.mlp_fc_w.data(), fc_out,
             S, M, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(fc_out, blk.mlp_fc_b.data(), fc_out, S, M, stream);

  LaunchGelu(fc_out, fc_out, S * M, stream);

  LaunchGemm(fc_out, blk.mlp_proj_w.data(), proj_out,
             S, D, M, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(proj_out, blk.mlp_proj_b.data(), proj_out, S, D, stream);

  LaunchVectorAdd(hidden, proj_out, hidden, S * D, stream);
}

// ---- Decode Step (single position, cached K/V) --------------------
// Processes only the new token at absolute position `pos`. Every linear
// operation runs at S=1. Attention queries the full K_cache[0..pos] and
// V_cache[0..pos]. The new K/V for position `pos` are computed and appended
// to the cache.

void LaunchTransformerBlockDecodeStep(float* h_row,
                                       const BlockWeights& blk,
                                       KVCache& kv_cache,
                                       TransformerBlockWorkspace& ws,
                                       const Gpt2Config& cfg,
                                       int layer_idx,
                                       int pos,
                                       cudaStream_t stream) {
  const int D = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim;
  const int M = cfg.mlp_inner;
  const float eps = cfg.ln_eps;
  const int n_ctx = kv_cache.n_ctx();

  // Workspace: reuse full-sized buffers; only the first S=1 rows are used.
  float* ln1_out    = ws.ln1_out.data();
  float* c_attn_out = ws.c_attn_out.data();
  float* q          = ws.q.data();
  float* scores     = ws.scores.data();
  float* weights    = ws.weights.data();
  float* attn_out   = ws.attn_out.data();
  float* proj_out   = ws.proj_out.data();
  float* ln2_out    = ws.ln2_out.data();
  float* fc_out     = ws.fc_out.data();

  // --- Attention path (S=1) ---
  // 1. ln_1 on the single row.
  LaunchLayerNorm(h_row, ln1_out,
                  blk.ln1_w.data(), blk.ln1_b.data(),
                  1, D, eps, LayerNormVersion::Naive, stream);

  // 2. c_attn on the single row → [1, 3D].
  LaunchGemm(ln1_out, blk.c_attn_w.data(), c_attn_out,
             1, 3 * D, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(c_attn_out, blk.c_attn_b.data(), c_attn_out,
                1, 3 * D, stream);

  // 3. Q → workspace; K, V → cache at position `pos`.
  float* k_cache_base = kv_cache.k_head(layer_idx, 0);
  float* v_cache_base = kv_cache.v_head(layer_idx, 0);
  LaunchQkvSplitToCache(c_attn_out, q, k_cache_base, v_cache_base,
                        pos, 1, D, H, n_ctx, stream);

  // 4. Cached attention: q[1, H, d] against K_cache[0..pos], V_cache[0..pos].
  const int num_cached = pos + 1;
  LaunchAttentionCached(q, k_cache_base, v_cache_base,
                        attn_out, scores, weights,
                        H, n_ctx, d, num_cached, stream);

  // 5. c_proj on the single row → [1, D].
  LaunchGemm(attn_out, blk.c_proj_w.data(), proj_out,
             1, D, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(proj_out, blk.c_proj_b.data(), proj_out, 1, D, stream);

  // 6. Residual 1: hidden_row += proj_out[0,:].
  LaunchVectorAdd(h_row, proj_out, h_row, D, stream);

  // --- MLP path (S=1) ---
  // 7. ln_2 on the single row.
  LaunchLayerNorm(h_row, ln2_out,
                  blk.ln2_w.data(), blk.ln2_b.data(),
                  1, D, eps, LayerNormVersion::Naive, stream);

  // 8. mlp_fc → [1, M].
  LaunchGemm(ln2_out, blk.mlp_fc_w.data(), fc_out,
             1, M, D, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(fc_out, blk.mlp_fc_b.data(), fc_out, 1, M, stream);

  // 9. GELU (1 row, M elements).
  LaunchGelu(fc_out, fc_out, M, stream);

  // 10. mlp_proj → [1, D].
  LaunchGemm(fc_out, blk.mlp_proj_w.data(), proj_out,
             1, D, M, GemmVersion::RegTiled128Pad, stream);
  LaunchBiasAdd(proj_out, blk.mlp_proj_b.data(), proj_out, 1, D, stream);

  // 11. Residual 2: hidden_row += proj_out[0,:].
  LaunchVectorAdd(h_row, proj_out, h_row, D, stream);
}

}  // namespace mgie
