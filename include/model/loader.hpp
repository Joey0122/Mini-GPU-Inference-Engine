// include/model/loader.hpp
//
// GPT-2 weight loader: manifest + flat binary -> per-tensor DeviceBuffers.
//
// LoadGpt2Weights() reads the files produced by scripts/export_gpt2.py:
//   gpt2_manifest.txt  — one line per tensor: "name offset_bytes ndim d0 [d1]"
//   gpt2_weights.bin   — flat FP32 payload, tensors in manifest order
//
// All allocations happen inside LoadGpt2Weights and are transfered into the
// returned Gpt2Weights struct. After the call, the binary file is closed and
// no raw pointers are retained; every tensor lives in an owned DeviceBuffer.
// Do NOT call LoadGpt2Weights in the inference loop — load once at startup
// and reuse the same Gpt2Weights for the lifetime of the engine. (AGENT.md)
//
// Weight layout (mirrors scripts/export_gpt2.py):
//   wte       [vocab_size, n_embd]    token embeddings
//   wte_T     [n_embd, vocab_size]    transposed (tied LM-head GEMM)
//   wpe       [n_ctx,   n_embd]       position embeddings
//   blocks[]  BlockWeights per layer  Conv1D weights [K, N] as LaunchGemm expects
//   ln_f_w/b  [n_embd]               final layer-norm

#pragma once

#include <array>
#include <string>

#include "cuda/memory.hpp"
#include "model/gpt2_config.hpp"

namespace mgie {

// Weights for one transformer block. All buffers are device-resident.
struct BlockWeights {
    DeviceBuffer<float> ln1_w;       // [n_embd]               ln_1 gamma
    DeviceBuffer<float> ln1_b;       // [n_embd]               ln_1 beta
    DeviceBuffer<float> c_attn_w;    // [n_embd, 3*n_embd]     QKV projection (Conv1D)
    DeviceBuffer<float> c_attn_b;    // [3*n_embd]
    DeviceBuffer<float> c_proj_w;    // [n_embd, n_embd]       attn output projection
    DeviceBuffer<float> c_proj_b;    // [n_embd]
    DeviceBuffer<float> ln2_w;       // [n_embd]               ln_2 gamma
    DeviceBuffer<float> ln2_b;       // [n_embd]               ln_2 beta
    DeviceBuffer<float> mlp_fc_w;    // [n_embd, mlp_inner]    MLP first linear (Conv1D)
    DeviceBuffer<float> mlp_fc_b;    // [mlp_inner]
    DeviceBuffer<float> mlp_proj_w;  // [mlp_inner, n_embd]    MLP second linear (Conv1D)
    DeviceBuffer<float> mlp_proj_b;  // [n_embd]
};

// All GPT-2 Small weights, device-resident and move-only.
// The config embedded here is the authoritative record of the shapes that were
// validated at load time.
struct Gpt2Weights {
    DeviceBuffer<float> wte;     // [vocab_size, n_embd]
    DeviceBuffer<float> wte_T;   // [n_embd, vocab_size]   for LM-head GEMM
    DeviceBuffer<float> wpe;     // [n_ctx, n_embd]
    std::array<BlockWeights, 12> blocks;
    DeviceBuffer<float> ln_f_w;  // [n_embd]
    DeviceBuffer<float> ln_f_b;  // [n_embd]
    Gpt2Config config;
};

// Load weights from `weights_dir` (must contain gpt2_manifest.txt and
// gpt2_weights.bin produced by scripts/export_gpt2.py).
// Validates every tensor's shape against `cfg`.
// Throws std::runtime_error on I/O error or shape mismatch.
Gpt2Weights LoadGpt2Weights(const std::string& weights_dir,
                            const Gpt2Config& cfg);

}  // namespace mgie
