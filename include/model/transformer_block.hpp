// include/model/transformer_block.hpp
//
// Single GPT-2 transformer block — orchestrates the sub-kernels into the
// standard Pre-LayerNorm transformer block structure:
//
//   hidden = hidden + c_proj( Attention( QKV( c_attn( ln_1(hidden) ) ) ) )
//   hidden = hidden + mlp_proj( GELU( mlp_fc( ln_2(hidden) ) ) )
//
// This is a C++ orchestration function, not a fused CUDA kernel. It chains the
// existing per-operation kernels (LayerNorm, GEMM, BiasAdd, QkvSplit,
// Attention, GELU, VectorAdd) in the exact order of a GPT-2 forward pass.
//
// C8: all intermediate scratch buffers are caller-provided via the
// TransformerBlockWorkspace struct. No cudaMalloc/cudaFree occurs inside
// the block. The same workspace is reused across all 12 layers.
//
// Weight layout — every Conv1D weight [in_features, out_features] = [K, N]
// is passed directly to LaunchGemm without transpose.

#pragma once

#include <cuda_runtime.h>

#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"

namespace mgie {

// Forward pass through one GPT-2 Pre-LN transformer block.
//
// hidden: [S, n_embd] — input/output, modified in-place (residual stream).
//   The residual adds are done in-place: hidden := hidden + attn_residual
//   and hidden := hidden + mlp_residual.
//
// blk: device-resident block weights (ln1, c_attn, c_proj, ln2, mlp_fc, mlp_proj).
//
// ws: pre-allocated scratch buffers. Must be sized for the given S and cfg.
//   All raw pointers (ln1_out, c_attn_out, q, k, v, scores, weights,
//   attn_out, proj_out, ln2_out, fc_out) are used — the caller
//   must ensure they hold at least the expected element counts.
//
// S: sequence length (must be <= n_ctx and > 0).
void LaunchTransformerBlock(float* hidden,
                            const BlockWeights& blk,
                            TransformerBlockWorkspace& ws,
                            const Gpt2Config& cfg,
                            int S,
                            cudaStream_t stream = nullptr);

// Phase 6a (KV cache): full transformer block with cache population (prefill).
// Identical to LaunchTransformerBlock, but routes K and V directly into the
// KV cache at positions [0, S) instead of the workspace k/v buffers.
// layer_idx: which transformer layer this is (0..11).
void LaunchTransformerBlockPrefill(float* hidden,
                                   const BlockWeights& blk,
                                   TransformerBlockWorkspace& ws,
                                   KVCache& kv_cache,
                                   int layer_idx,
                                   const Gpt2Config& cfg,
                                   int S,
                                   cudaStream_t stream = nullptr);

// Phase 6a (KV cache): single-position decode step using the cached K/V.
// Updates the supplied hidden_row in-place (one row of n_embd floats).
// K/V at pos are computed fresh and attention reads cache entries [0, pos].
// layer_idx: which transformer layer (0..11).
// pos: absolute position in the sequence.
void LaunchTransformerBlockDecodeStep(float* hidden_row,
                                      const BlockWeights& blk,
                                      KVCache& kv_cache,
                                      TransformerBlockWorkspace& ws,
                                      const Gpt2Config& cfg,
                                      int layer_idx,
                                      int pos,
                                      cudaStream_t stream = nullptr);

}  // namespace mgie
