// include/model/kv_cache.hpp
//
// Phase 6a: KV cache for autoregressive decoding.
//
// Stores key and value tensors for every transformer layer's attention heads
// across all sequence positions, avoiding recomputation during decoding.
//
// Memory layout (flat DeviceBuffer, row-major over [n_layer, n_head, n_ctx, head_dim]):
//
//   cache_k[layer][head][pos][dim]
//     = k.data[ ((layer * n_head + head) * n_ctx + pos) * head_dim + dim ]
//
//   cache_v[layer][head][pos][dim]
//     = v.data[ ((layer * n_head + head) * n_ctx + pos) * head_dim + dim ]
//
// Strides (GPT-2 Small: n_layer=12, n_head=12, n_ctx=1024, head_dim=64):
//   layer: n_head * n_ctx * head_dim = 786,432 elements (3 MB)
//   head:  n_ctx * head_dim         =  65,536 elements (256 KB)
//   pos:   head_dim                 =      64 elements (256 bytes)
//   dim:   1                        =       1 element  (4 bytes)
//
// Total: 2 × 12 × 12 × 1024 × 64 × 4 bytes = 72 MB.
//
// Ownership: caller creates KVCache once alongside Gpt2Weights and Gpt2Workspace.
// No allocations in the hot path. Move-only RAII via DeviceBuffer members.

#pragma once

#include "cuda/memory.hpp"
#include "model/gpt2_config.hpp"

namespace mgie {

struct KVCache {
  DeviceBuffer<float> k;  // [n_layer * n_head * n_ctx * head_dim]
  DeviceBuffer<float> v;  // ditto
  int num_positions = 0;   // monotonically increasing: 0 → prompt_len → prompt_len + N

  // Allocate for the given config (n_layer, n_head, n_ctx, head_dim).
  static KVCache Create(const Gpt2Config& cfg);

  // Per-layer, per-head base pointer for kernel launches.
  // Returns pointer to the start of (layer, head)'s key/value data.
  // The kernel then accesses position pos, dimension d at:
  //   k_head_ptr[pos * head_dim + d]
  // This matches the existing AttentionScoresKernel access pattern.
  float*       k_head(int layer, int head);
  const float* k_head(int layer, int head) const;
  float*       v_head(int layer, int head);
  const float* v_head(int layer, int head) const;

  // Stored dimensions (set by Create).
  int n_layer()  const { return n_layer_; }
  int n_head()   const { return n_head_; }
  int n_ctx()    const { return n_ctx_; }
  int head_dim() const { return head_dim_; }

 private:
  int n_layer_  = 0;
  int n_head_   = 0;
  int n_ctx_    = 0;
  int head_dim_ = 0;
};

}  // namespace mgie
