// include/kernels/qkv_split.hpp
//
// Launch interface for the QKV Split kernel.
//
// After the c_attn GEMM + BiasAdd, the output is a [S, 3*n_embd] tensor where
// the Q, K, V blocks are concatenated column-wise:
//   columns [0, n_embd)          — Q heads interleaved (head h at cols [h*head_dim, (h+1)*head_dim))
//   columns [n_embd, 2*n_embd)   — K heads interleaved
//   columns [2*n_embd, 3*n_embd) — V heads interleaved
//
// This kernel rearranges the data into the head-major layout that
// LaunchAttention expects:
//   q, k, v : [n_head, S, head_dim]
//
// Each output is head-major: all S rows of head 0 are contiguous, then head 1,
// etc. Within each head, the S rows are contiguous and each row has head_dim
// elements.
//
// This IS a data-movement kernel — the stride pattern changes from (3*n_embd, 1)
// to (S*head_dim, head_dim, 1). It is NOT a zero-copy slice.
//
// Shapes:
//   x       [S, 3*n_embd]     — c_attn output, row-major, device-resident
//   q,k,v   [n_head, S, head_dim] — outputs, head-major, device-resident
//   n_embd  = n_head * head_dim   (768 for GPT-2 Small)

#pragma once

#include <cuda_runtime.h>

namespace mgie {

// x: [S, 3*n_embd] — combined QKV (c_attn matmul + bias output)
// q, k, v: [n_head, S, head_dim] — split outputs, head-major layout matching
//   LaunchAttention's expected input.
// S: sequence length; n_embd: hidden dim (== n_head * head_dim)
// n_head: number of attention heads; head_dim: dim per head (== n_embd / n_head)
void LaunchQkvSplit(const float* x,
                    float* q, float* k, float* v,
                    int S, int n_embd, int n_head,
                    cudaStream_t stream = nullptr);

// QKV split routing K and V directly into the cache.
//   x        : [S, 3*n_embd]      c_attn output
//   q        : [n_head, S, head_dim]  Q → workspace (needed immediately for attention)
//   k_cache  : [n_head, n_ctx, head_dim]  K → cache at positions [pos_offset, pos_offset+S)
//   v_cache  : [n_head, n_ctx, head_dim]  V → cache at positions [pos_offset, pos_offset+S)
//   pos_offset: first cache position to write (0 for prefill, pos for decode step)
//   n_ctx    : context width (cache stride between positions within a head)
//
// Position s in [0, S) maps to cache position pos_offset + s.
void LaunchQkvSplitToCache(const float* x,
                           float* q,
                           float* k_cache,
                           float* v_cache,
                           int pos_offset,
                           int S, int n_embd, int n_head, int n_ctx,
                           cudaStream_t stream = nullptr);

}  // namespace mgie
