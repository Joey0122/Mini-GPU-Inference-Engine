// include/kernels/embedding.hpp
//
// Launch interface for the Phase 5 C4 embedding kernel.
//
// GPT-2 embeddings combine a token lookup (wte) with a positional encoding (wpe):
//   out[s, d] = wte[token_ids[s], d] + wpe[s, d]
//
// This is a pure gather+add kernel — no GEMM, no reduction. It produces the
// input tensor that feeds into the first transformer block.
//
// Shapes:
//   token_ids   [S]                — int32 token indices, 0 <= token_ids[s] < vocab_size
//   wte         [vocab_size, n_embd] — token embedding table, device-resident
//   wpe         [n_ctx, n_embd]      — positional embedding table, device-resident
//   out         [S, n_embd]           — output, device-resident
//   S           sequence length, must be <= n_ctx
//   n_embd      embedding dimension (768 for GPT-2 Small)

#pragma once

#include <cuda_runtime.h>

namespace mgie {

// Launch the embedding gather+add kernel.
// Each thread handles one element of the output; grid spans (S, n_embd).
// The wte lookup is an indexed gather — performance is memory-latency-bound
// by design for this naive version (optimization deferred to Phase 6).
void LaunchEmbedding(const int* token_ids, const float* wte, const float* wpe,
                     float* out, int S, int n_embd, int vocab_size,
                     cudaStream_t stream = nullptr);

}  // namespace mgie
