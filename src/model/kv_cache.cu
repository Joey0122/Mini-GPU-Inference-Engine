// src/model/kv_cache.cu
// Phase 6a: KV cache allocation factory.

#include "model/kv_cache.hpp"

#include <cassert>

namespace mgie {

KVCache KVCache::Create(const Gpt2Config& cfg) {
  KVCache cache;
  cache.n_layer_  = cfg.n_layer;
  cache.n_head_   = cfg.n_head;
  cache.n_ctx_    = cfg.n_ctx;
  cache.head_dim_ = cfg.head_dim;
  cache.num_positions = 0;

  const std::size_t per_layer = static_cast<std::size_t>(cache.n_head_)
                              * static_cast<std::size_t>(cache.n_ctx_)
                              * static_cast<std::size_t>(cache.head_dim_);
  const std::size_t total = per_layer * cache.n_layer_;

  if (total > 0) {
    cache.k = DeviceBuffer<float>(total);
    cache.v = DeviceBuffer<float>(total);
  }
  return cache;
}

float* KVCache::k_head(int layer, int head) {
  return k.data() + ((static_cast<std::size_t>(layer) * n_head_ + head)
                      * n_ctx_) * head_dim_;
}

const float* KVCache::k_head(int layer, int head) const {
  return k.data() + ((static_cast<std::size_t>(layer) * n_head_ + head)
                      * n_ctx_) * head_dim_;
}

float* KVCache::v_head(int layer, int head) {
  return v.data() + ((static_cast<std::size_t>(layer) * n_head_ + head)
                      * n_ctx_) * head_dim_;
}

const float* KVCache::v_head(int layer, int head) const {
  return v.data() + ((static_cast<std::size_t>(layer) * n_head_ + head)
                      * n_ctx_) * head_dim_;
}

}  // namespace mgie
