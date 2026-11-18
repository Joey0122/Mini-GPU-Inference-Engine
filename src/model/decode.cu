// src/model/decode.cu
// KV-cache autoregressive decoding (greedy or temperature sampling).
//
// Two-phase decode:
//   1. Prefill: full forward pass on the prompt tokens, populating the KV cache
//      for all layers and positions [0, prompt_len).
//   2. Autoregressive loop: for each new token, a single-position decode step
//      (S=1 linear ops + cached attention) produces its logits. The next token
//      is greedily or stochastically selected on the host.
//
// No cudaMalloc/cudaFree inside the loop. Workspace and KVCache are allocated
// once at startup.

#include "model/decode.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common/errors.hpp"

namespace mgie {

// Host-side: select next token from logits via greedy argmax.
inline int GreedyNextToken(const float* logits, int V) {
  int best = 0;
  float best_val = logits[0];
  for (int v = 1; v < V; ++v) {
    if (logits[v] > best_val) {
      best_val = logits[v];
      best = v;
    }
  }
  return best;
}

// Host-side: apply temperature + softmax + categorical sampling.
// logits buffer is modified in-place (divided by temperature, then used
// for exp computation in softmax).
inline int SampleNextToken(float* logits, int V, float temperature,
                           std::mt19937& rng) {
  // Temperature scaling: logits /= temp
  const float inv_temp = 1.0f / temperature;
  float max_val = logits[0] * inv_temp;
  for (int v = 0; v < V; ++v) {
    logits[v] *= inv_temp;
    if (logits[v] > max_val) max_val = logits[v];
  }

  // Softmax: exp(logit - max) for numerical stability.
  // We only need unnormalized weights (discrete_distribution normalizes
  // internally), so we skip the division by sum.
  std::vector<double> weights(V);  // double for discrete_distribution precision
  for (int v = 0; v < V; ++v)
    weights[v] = static_cast<double>(std::exp(logits[v] - max_val));

  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return dist(rng);
}

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
                         cudaStream_t stream) {
  if (prompt_len <= 0 || max_new_tokens <= 0) return 0;
  if (temperature < 0.0f) temperature = 0.0f;

  if (prompt_len >= cfg.n_ctx) return 0;
  const int gen_len = std::min(max_new_tokens, cfg.n_ctx - prompt_len);

  const int V = cfg.vocab_size;
  const bool do_sample = (temperature > 1e-6f);

  // Deterministic RNG (seeded). Mersenne Twister with fixed seed reproduces
  // the exact same sequence across runs.
  std::mt19937 rng(seed != 0 ? seed : std::random_device{}());

  // Copy prompt tokens to output buffer.
  CUDA_CHECK(cudaMemcpyAsync(output_ids, prompt_ids,
                              prompt_len * sizeof(int),
                              cudaMemcpyDeviceToDevice, stream));

  // --- Prefill ---
  // Full forward pass on the prompt, populating the KV cache for all layers.
  kv_cache.num_positions = 0;
  LaunchGpt2Prefill(output_ids, weights, ws.logits.data(),
                     ws, kv_cache, cfg, prompt_len, stream);
  // kv_cache.num_positions is now prompt_len (set inside LaunchGpt2Prefill).

  // Host scratch: one row of logits (V floats), reused per step.
  std::vector<float> last_logits(V);

  // Copy last prefill logit row to host for the first sampling step.
  CUDA_CHECK(cudaMemcpyAsync(last_logits.data(),
                              ws.logits.data()
                                + static_cast<std::size_t>(prompt_len - 1) * V,
                              V * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // --- Autoregressive Decode ---
  for (int step = 0; step < gen_len; ++step) {
    const int cur_pos = prompt_len + step;

    // Select next token from the last logits row (already on host).
    int next_token;
    if (do_sample)
      next_token = SampleNextToken(last_logits.data(), V, temperature, rng);
    else
      next_token = GreedyNextToken(last_logits.data(), V);

    // Write next token to output_ids[cur_pos].
    CUDA_CHECK(cudaMemcpyAsync(output_ids + cur_pos, &next_token,
                                sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    // Decode step: single-position forward pass using the KV cache.
    // Writes one row of logits to ws.logits[0] (reused each step).
    float* logits_row = ws.logits.data();  // position 0, single row
    LaunchGpt2DecodeStep(next_token, cur_pos, weights,
                          logits_row, ws, kv_cache, cfg, stream);

    // Copy the new logits row to host for the next iteration.
    CUDA_CHECK(cudaMemcpyAsync(last_logits.data(), logits_row,
                                V * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return gen_len;
}

// Greedy decoding uses a temperature of zero.
int LaunchGreedyDecode(const int* prompt_ids,
                       int* output_ids,
                       const Gpt2Weights& weights,
                       Gpt2Workspace& ws,
                       KVCache& kv_cache,
                       const Gpt2Config& cfg,
                       int prompt_len,
                       int max_new_tokens,
                       cudaStream_t stream) {
  return LaunchSamplingDecode(prompt_ids, output_ids, weights, ws, kv_cache,
                              cfg, prompt_len, max_new_tokens,
                              0.0f, 0, stream);
}

}  // namespace mgie
