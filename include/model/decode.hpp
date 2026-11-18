// include/model/decode.hpp
//
// C9/C10: Greedy and temperature-sampling autoregressive decoding.
//
// C9: LaunchGreedyDecode — selects argmax(logits[-1,:]) at each step.
// C10: LaunchSamplingDecode — applies temperature scaling + softmax + categorical
//   sampling. Temperature ≤ 1e-6 falls back to greedy. Seed controls the
//   host-side Mersenne Twister RNG deterministically.
//
// Memory model (zero-allocation loop):
//   - Create Gpt2Workspace with S >= prompt_len and KVCache with the same cfg.
//   - prompt_ids and output_ids are device buffers. With a positive return
//     value, output_ids holds prompt_len + min(max_new_tokens, n_ctx - prompt_len)
//     tokens, including the original prompt prefix.
//   - Returns 0 without writing output for an empty prompt, a prompt at or
//     beyond the context limit, or a nonpositive max_new_tokens.
//   - The workspace and cache are reused without device allocations in the loop.

#pragma once

#include <cuda_runtime.h>

#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"
#include "model/forward.hpp"

namespace mgie {

// Greedy autoregressive decoding (C9).
// Equivalent to LaunchSamplingDecode with temperature = 0.
int LaunchGreedyDecode(const int* prompt_ids,
                       int* output_ids,
                       const Gpt2Weights& weights,
                       Gpt2Workspace& ws,
                       KVCache& kv_cache,
                       const Gpt2Config& cfg,
                       int prompt_len,
                       int max_new_tokens,
                       cudaStream_t stream = nullptr);

// Temperature-sampling autoregressive decoding (C10).
//
// At each step:
//   logits = forward(token_ids)
//   if temperature <= 1e-6:  next_token = argmax(logits[-1,:])   (greedy)
//   else:
//     logits[-1,:] /= temperature
//     probs = softmax(logits[-1,:])   (host-side, stable)
//     next_token ~ Categorical(probs) (std::mt19937, seeded)
//
// seed: RNG seed for deterministic sampling. Same seed + same prompt +
//   same temperature → identical output sequence (deterministic Mersenne
//   Twister). Pass 0 for std::random_device seeding.
//
// Returns the number of tokens actually generated.
int LaunchSamplingDecode(const int* prompt_ids,
                         int* output_ids,
                         const Gpt2Weights& weights,
                         Gpt2Workspace& ws,
                         KVCache& kv_cache,
                         const Gpt2Config& cfg,
                         int prompt_len,
                         int max_new_tokens,
                         float temperature,
                         unsigned int seed,
                         cudaStream_t stream = nullptr);

}  // namespace mgie
