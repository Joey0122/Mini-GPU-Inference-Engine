// src/model/forward.cu
// Phase 5 C8: full GPT-2 Small forward pass.
//
// Chains embedding, 12 transformer blocks, final LayerNorm, and LM head into
// a single call. All intermediate buffers come from the pre-allocated workspace.
//
// The forward pass:
//   hidden = wte[token_ids] + wpe                     (C4 embedding)
//   for i in 0..11:
//       hidden = transformer_block(hidden, blk[i])    (C7 block, 12×)
//   hidden = ln_f(hidden)                              (Phase 3 LayerNorm)
//   logits = hidden @ wte_T                            (Phase 2 GEMM)

#include "model/forward.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/errors.hpp"
#include "kernels/embedding.hpp"
#include "kernels/gemm.hpp"
#include "kernels/layernorm.hpp"
#include "kernels/vector_add.hpp"
#include "model/transformer_block.hpp"

namespace mgie {

void LaunchGpt2Forward(const int* token_ids,
                       const Gpt2Weights& weights,
                       float* logits,
                       Gpt2Workspace& ws,
                       const Gpt2Config& cfg,
                       int S,
                       cudaStream_t stream) {
  if (S <= 0) return;

  const int D = cfg.n_embd;
  const int V = cfg.vocab_size;
  const float eps = cfg.ln_eps;

  // 1. Embedding: hidden = wte[token_ids] + wpe
  LaunchEmbedding(token_ids,
                  weights.wte.data(), weights.wpe.data(),
                  ws.hidden.data(), S, D, V, stream);

  // 2. 12 transformer blocks: hidden modified in-place via residuals
  for (int i = 0; i < cfg.n_layer; ++i)
    LaunchTransformerBlock(ws.hidden.data(),
                           weights.blocks[static_cast<std::size_t>(i)],
                           ws.blk, cfg, S, stream);

  // 3. Final LayerNorm: ln_f(hidden) → hidden (in-place)
  LaunchLayerNorm(ws.hidden.data(), ws.hidden.data(),
                  weights.ln_f_w.data(), weights.ln_f_b.data(),
                  S, D, eps, LayerNormVersion::Naive, stream);

  // 4. LM head: logits = hidden @ wte_T [D, V]
  LaunchGemm(ws.hidden.data(), weights.wte_T.data(), logits,
             S, V, D, GemmVersion::RegTiled128Pad, stream);
}

// ---- Phase 6a: Prefill (full forward + cache population) --------------------
// Same as LaunchGpt2Forward but calls LaunchTransformerBlockPrefill for each
// layer, populating the KV cache. All intermediate tensors use S-stride layout
// for the attention computation; K/V are then copied to n_ctx-stride cache.

void LaunchGpt2Prefill(const int* token_ids,
                       const Gpt2Weights& weights,
                       float* logits,
                       Gpt2Workspace& ws,
                       KVCache& kv_cache,
                       const Gpt2Config& cfg,
                       int S,
                       cudaStream_t stream) {
  if (S <= 0) return;
  if (S > cfg.n_ctx || S > kv_cache.n_ctx())
    throw std::out_of_range("prefill length exceeds the context window");

  const int D = cfg.n_embd;
  const int V = cfg.vocab_size;
  const float eps = cfg.ln_eps;

  // 1. Embedding: hidden = wte[token_ids] + wpe
  LaunchEmbedding(token_ids,
                  weights.wte.data(), weights.wpe.data(),
                  ws.hidden.data(), S, D, V, stream);

  // 2. 12 transformer blocks: each populates the KV cache for its layer.
  for (int i = 0; i < cfg.n_layer; ++i)
    LaunchTransformerBlockPrefill(ws.hidden.data(),
                                   weights.blocks[static_cast<std::size_t>(i)],
                                   ws.blk, kv_cache, i, cfg, S, stream);
  kv_cache.num_positions = S;

  // 3. Final LayerNorm: ln_f(hidden) → hidden (in-place)
  LaunchLayerNorm(ws.hidden.data(), ws.hidden.data(),
                  weights.ln_f_w.data(), weights.ln_f_b.data(),
                  S, D, eps, LayerNormVersion::Naive, stream);

  // 4. LM head: logits = hidden @ wte_T [D, V]
  LaunchGemm(ws.hidden.data(), weights.wte_T.data(), logits,
             S, V, D, GemmVersion::RegTiled128Pad, stream);
}

// ---- Phase 6a: Decode Step (single position, cached K/V) -------------------
// Processes one new token: embedding → 12× decode step → ln_f → lm_head.
// All linear operations run at S=1 (except attention which queries S_cached keys).

void LaunchGpt2DecodeStep(int new_token,
                          int pos,
                          const Gpt2Weights& weights,
                          float* logits_row,
                          Gpt2Workspace& ws,
                          KVCache& kv_cache,
                          const Gpt2Config& cfg,
                          cudaStream_t stream) {
  if (pos < 0 || pos >= cfg.n_ctx || pos >= kv_cache.n_ctx())
    throw std::out_of_range("decode position is outside the context window");
  if (new_token < 0 || new_token >= cfg.vocab_size)
    throw std::out_of_range("decode token is outside the vocabulary");

  const int D = cfg.n_embd;
  const int V = cfg.vocab_size;
  const float eps = cfg.ln_eps;

  // 1. Reuse one hidden row. Only the positional embedding and cache use pos.
  float* h_row = ws.hidden.data();
  CUDA_CHECK(cudaMemcpyAsync(
      h_row,
      weights.wte.data() + static_cast<std::size_t>(new_token) * D,
      D * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  const float* wpe_row = weights.wpe.data() + static_cast<std::size_t>(pos) * D;
  LaunchVectorAdd(h_row, wpe_row, h_row, D, stream);

  // 2. 12 transformer block decode steps.
  for (int i = 0; i < cfg.n_layer; ++i)
    LaunchTransformerBlockDecodeStep(h_row,
                                     weights.blocks[static_cast<std::size_t>(i)],
                                     kv_cache, ws.blk, cfg, i, pos, stream);
  kv_cache.num_positions = std::max(kv_cache.num_positions, pos + 1);

  // 3. Final LayerNorm on the hidden row.
  LaunchLayerNorm(h_row, h_row,
                  weights.ln_f_w.data(), weights.ln_f_b.data(),
                  1, D, eps, LayerNormVersion::Naive, stream);

  // 4. LM head: single row @ wte_T → logits_row [vocab_size].
  LaunchGemm(h_row, weights.wte_T.data(), logits_row,
             1, V, D, GemmVersion::RegTiled128Pad, stream);
}

}  // namespace mgie
