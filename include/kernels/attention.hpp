// include/kernels/attention.hpp
//
// Launch interface for the GPT-2 attention kernels.
//
// Implements causal multi-head scaled-dot-product attention for batch=1, the
// mechanism only: input projections (c_attn) and the output projection
// (c_proj) are applied separately as GEMMs over loaded weights.
//
// Tensor layout (per-head contiguous):
//   q, k, v : [n_head, S, head_dim]   row-major device pointers
//   out     : [S, n_head*head_dim]    row-major; heads concatenated along cols
//                                      (out[i, h*head_dim + d]) — GPT-2's merged
//                                      multi-head output, ready for c_proj.
//
// The path is decomposed into dedicated batched kernels (one block grid over
// all heads via blockIdx.z):
//
//   1. LaunchAttentionScores : scores = scale * (Q Kᵀ), causal-masked
//   2. (reuse) LaunchSoftmax : row-wise softmax over [n_head*S, S]
//   3. LaunchAttentionAV     : out = weights · V, heads merged
//   4. LaunchAttention       : integrated Q,K,V -> out using caller-provided scratch

#pragma once

#include <cuda_runtime.h>

namespace mgie {

// Step 1: scaled QKᵀ with causal masking.
//   scores[h,i,j] = (1/sqrt(head_dim)) * sum_d q[h,i,d]*k[h,j,d]   for j <= i
//   scores[h,i,j] = -large                                        for j >  i
// q, k     : [n_head, S, head_dim]
// scores   : [n_head, S, S]  (output; viewable as [n_head*S, S] rows)
void LaunchAttentionScores(const float* q, const float* k, float* scores,
                           int n_head, int S, int head_dim,
                           cudaStream_t stream = nullptr);

// Step 3: attention-weighted sum of values, with heads merged into the output.
//   out[i, h*head_dim + d] = sum_j weights[h,i,j] * v[h,j,d]
// weights : [n_head, S, S]  (softmax of masked scores)
// v       : [n_head, S, head_dim]
// out     : [S, n_head*head_dim]
void LaunchAttentionAV(const float* weights, const float* v, float* out,
                       int n_head, int S, int head_dim,
                       cudaStream_t stream = nullptr);

// Step 4: integrated causal multi-head attention.
//   out = Attention(q, k, v) = AV(softmax(mask(scale * q kᵀ)), v), heads merged.
// q, k, v : [n_head, S, head_dim]
// out     : [S, n_head*head_dim]
// scores, weights : caller-provided scratch, each [n_head, S, S]. Passing them
//   in keeps allocation out of the hot path. The model reuses them across layers.
void LaunchAttention(const float* q, const float* k, const float* v, float* out,
                     float* scores, float* weights,
                     int n_head, int S, int head_dim,
                     cudaStream_t stream = nullptr);

// cached attention for single-token decode steps.
//   q_new      : [n_head, 1, head_dim]       single query row per head
//   k_cache, v_cache : [n_head, n_ctx, head_dim]  KV cache for one layer
//   n_ctx      : stride between heads in the cache (= context width)
//   num_cached : actual number of populated positions (≤ n_ctx); loop bound
//   out        : [1, n_head*head_dim]         single-row output
//   scores, weights : caller scratch, each [n_head, num_cached]
//
// No causal mask — all num_cached positions are valid keys for the query
// The query is at position num_cached - 1, and the cache includes its key.
void LaunchAttentionCached(const float* q_new,
                           const float* k_cache,
                           const float* v_cache,
                           float* out,
                           float* scores,
                           float* weights,
                           int n_head, int n_ctx, int head_dim,
                           int num_cached,
                           cudaStream_t stream = nullptr);

}  // namespace mgie
