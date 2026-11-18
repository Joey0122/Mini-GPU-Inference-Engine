// src/kernels/embedding.cu
// token + positional embedding gather+add kernel.
//
// Kernel design:
//   - 1D grid over sequence positions (blockIdx.x = s)
//   - 1D block over embedding dimension (threadIdx.x = d), strided if needed
//   - Each thread loads wte[token_ids[s], d] (gather) + wpe[s, d] (direct)
//     and writes the sum to out[s, d].
//
// The wte gather is a scattered read (depends on token_ids[s]); coalescing is
// poor by construction for this naive kernel. Optimization (reshaped lookups,
// vectorized loads) is possible.

#include "kernels/embedding.hpp"

#include "common/errors.hpp"

namespace mgie {
namespace {

// Block size: cover the embedding dimension with threads.
// 256 threads → 3 stride iterations for n_embd=768 (GPT-2 Small).
// 128 threads → 6 stride iterations for n_embd=768.
constexpr int kBlockDim = 256;

__global__ void EmbeddingKernel(const int* __restrict__ token_ids,
                                 const float* __restrict__ wte,
                                 const float* __restrict__ wpe,
                                 float* __restrict__ out,
                                 int S, int n_embd, int vocab_size) {
  const int s = blockIdx.x;
  if (s >= S) return;

  const int tid = token_ids[s];
  // Guard against OOB token IDs (caller contract: 0 <= tid < vocab_size).
  // Clamping prevents undefined behaviour in release builds but the caller
  // (model layer) must ensure validity.
  const int safe_tid = (tid >= 0 && tid < vocab_size) ? tid : 0;

  const std::size_t wte_row = static_cast<std::size_t>(safe_tid) * n_embd;
  const std::size_t wpe_row = static_cast<std::size_t>(s) * n_embd;
  const std::size_t out_row = static_cast<std::size_t>(s) * n_embd;

  // Strided loop: each thread handles elements d, d+blockDim, d+2*blockDim, ...
  for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
    out[out_row + d] = wte[wte_row + d] + wpe[wpe_row + d];
  }
}

}  // namespace

void LaunchEmbedding(const int* token_ids, const float* wte, const float* wpe,
                     float* out, int S, int n_embd, int vocab_size,
                     cudaStream_t stream) {
  if (S <= 0 || n_embd <= 0) return;

  const int block = (n_embd < kBlockDim) ? n_embd : kBlockDim;
  const dim3 grid(S);
  EmbeddingKernel<<<grid, block, 0, stream>>>(token_ids, wte, wpe, out,
                                                S, n_embd, vocab_size);
}

}  // namespace mgie
