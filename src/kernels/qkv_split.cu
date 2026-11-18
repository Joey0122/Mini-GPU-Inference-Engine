// src/kernels/qkv_split.cu
// QKV split kernel — c_attn output to head-major Q, K, V tensors.
//
// Kernel design:
//   - 1D grid: n_head * S blocks, one per (head, sequence position) pair
//   - 1D block: head_dim threads, one per dimension element
//   - Each thread reads 3 values from x (Q, K, V blocks) and writes to q, k, v
//   - Fallback strided loop for head_dim > kBlockDim (generic, though GPT-2's
//     head_dim=64 fits in one block)
//
// Layout transformation:
//   Input  x: [S, 3*n_embd]  row-major, Q|K|V block-concatenated columns
//   Output q: [n_head, S, head_dim]  head-major, then row-major within each head
//
// The output layout matches LaunchAttention's expected input exactly:
//   q[(h * S + s) * head_dim + d] — the same formula used in AttentionScoresKernel.

#include "kernels/qkv_split.hpp"

#include "common/errors.hpp"

namespace mgie {
namespace {

// Default block size for the generic case (head_dim > 256).
constexpr int kBlockDim = 256;

__global__ void QkvSplitKernel(const float* __restrict__ x,
                                float* __restrict__ q,
                                float* __restrict__ k,
                                float* __restrict__ v,
                                int S, int n_embd, int n_head, int head_dim) {
  const int flat = blockIdx.x;
  const int h = flat / S;   // head index
  const int s = flat % S;   // sequence position within head
  if (h >= n_head) return;  // safety: extra grid blocks are no-ops

  const std::size_t src_row = static_cast<std::size_t>(s) * 3 * n_embd;
  const std::size_t head_offset = static_cast<std::size_t>(h) * head_dim;
  const std::size_t tgt_base = (static_cast<std::size_t>(h) * S + s) * head_dim;

  // Strided loop: each thread handles dimensions d, d+blockDim, d+2*blockDim, ...
  // For GPT-2 (head_dim=64 < kBlockDim=256), each thread handles 1 element.
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q[tgt_base + d] = x[src_row + 0 * n_embd + head_offset + d];
    k[tgt_base + d] = x[src_row + 1 * n_embd + head_offset + d];
    v[tgt_base + d] = x[src_row + 2 * n_embd + head_offset + d];
  }
}

}  // namespace

void LaunchQkvSplit(const float* x,
                    float* q, float* k, float* v,
                    int S, int n_embd, int n_head,
                    cudaStream_t stream) {
  if (S <= 0 || n_embd <= 0 || n_head <= 0) return;

  const int head_dim = n_embd / n_head;
  const int block = (head_dim < kBlockDim) ? head_dim : kBlockDim;
  const int grid = n_head * S;

  QkvSplitKernel<<<grid, block, 0, stream>>>(x, q, k, v, S, n_embd, n_head,
                                              head_dim);
}

// ---- QKV Split to KV Cache ----------------------------------------
// Same data rearrangement as QkvSplitKernel, but K and V are written directly
// into the KV cache at n_ctx stride (instead of workspace at S stride).
// Q still goes to workspace (q) since it is consumed immediately by attention.

namespace {

__global__ void QkvSplitToCacheKernel(const float* __restrict__ x,
                                       float* __restrict__ q,
                                       float* __restrict__ k_cache,
                                       float* __restrict__ v_cache,
                                       int S, int n_embd, int n_head,
                                       int n_ctx, int pos_offset, int head_dim) {
  const int flat = blockIdx.x;
  const int h = flat / S;
  const int s = flat % S;
  if (h >= n_head) return;

  const std::size_t src_row = static_cast<std::size_t>(s) * 3 * n_embd;
  const std::size_t head_off = static_cast<std::size_t>(h) * head_dim;

  // Q: workspace layout — S positions per head (same as original).
  const std::size_t q_base = (static_cast<std::size_t>(h) * S + s) * head_dim;

  // K, V: cache layout — n_ctx positions per head.
  const int cache_pos = pos_offset + s;
  const std::size_t kv_base = (static_cast<std::size_t>(h) * n_ctx + cache_pos) * head_dim;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q[q_base + d]        = x[src_row + 0 * n_embd + head_off + d];
    k_cache[kv_base + d] = x[src_row + 1 * n_embd + head_off + d];
    v_cache[kv_base + d] = x[src_row + 2 * n_embd + head_off + d];
  }
}

}  // namespace

void LaunchQkvSplitToCache(const float* x,
                           float* q,
                           float* k_cache,
                           float* v_cache,
                           int pos_offset,
                           int S, int n_embd, int n_head, int n_ctx,
                           cudaStream_t stream) {
  if (S <= 0 || n_embd <= 0 || n_head <= 0) return;

  const int head_dim = n_embd / n_head;
  const int block = (head_dim < 256) ? head_dim : 256;
  const int grid = n_head * S;

  QkvSplitToCacheKernel<<<grid, block, 0, stream>>>(
      x, q, k_cache, v_cache, S, n_embd, n_head, n_ctx, pos_offset, head_dim);
}

}  // namespace mgie
