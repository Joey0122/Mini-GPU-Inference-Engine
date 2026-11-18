// include/model/forward.hpp
//
// Full GPT-2 Small (124M) forward pass.
//
// Orchestrates the complete inference pipeline:
//   1. Embedding:          hidden = wte[token_ids] + wpe
//   2. Transformer blocks: hidden = block_0(hidden) → ... → block_11(hidden)
//   3. Final LayerNorm:    hidden = ln_f(hidden)
//   4. LM head:            logits = hidden @ wte_T
//
// All intermediate scratch buffers are pre-allocated via Gpt2Workspace.
// No cudaMalloc/cudaFree occurs inside the forward pass.

#pragma once

#include <cuda_runtime.h>

#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"

namespace mgie {

// Full GPT-2 forward pass: token_ids → logits.
//
// token_ids: [S] int32, device-resident. Each element must be in [0, vocab_size).
// weights:   all GPT-2 tensors, device-resident (loaded once via LoadGpt2Weights).
// logits:    [S, vocab_size] float, device-resident output. Caller supplies
//            buffer; this function computes into it.
// ws:        pre-allocated workspace (hidden + per-block scratch). Caller
//            creates once via Gpt2Workspace::Create(S, cfg) and reuses
//            across forward calls with the same S.
//
// Ownership: logits buffer is caller-owned (typically inside ws.logits, but
//            passed separately so the caller can specify a different buffer
//            for the autoregressive loop).
void LaunchGpt2Forward(const int* token_ids,
                       const Gpt2Weights& weights,
                       float* logits,
                       Gpt2Workspace& ws,
                       const Gpt2Config& cfg,
                       int S,
                       cudaStream_t stream = nullptr);

// full forward pass with cache population (prefill).
// Identical to LaunchGpt2Forward, but each transformer block populates the
// KV cache for future decode steps. kv_cache.num_positions is updated to S.
void LaunchGpt2Prefill(const int* token_ids,
                       const Gpt2Weights& weights,
                       float* logits,
                       Gpt2Workspace& ws,
                       KVCache& kv_cache,
                       const Gpt2Config& cfg,
                       int S,
                       cudaStream_t stream = nullptr);

// single-position forward step using cached K/V.
// Processes one token at absolute position `pos` in [0, n_ctx). The cache
// must already contain positions [0, pos) and be created with the same cfg.
// The step overwrites the K/V entry at pos and updates num_positions.
// A Gpt2Workspace with S >= 1 suffices. Its first hidden row is reused,
// and logits_row must hold at least vocab_size floats.
void LaunchGpt2DecodeStep(int new_token,
                          int pos,
                          const Gpt2Weights& weights,
                          float* logits_row,
                          Gpt2Workspace& ws,
                          KVCache& kv_cache,
                          const Gpt2Config& cfg,
                          cudaStream_t stream = nullptr);

}  // namespace mgie
