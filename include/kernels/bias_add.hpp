// include/kernels/bias_add.hpp
//
// Launch interface for the Phase 5 C5 BiasAdd kernel.
//
// BiasAdd broadcasts a bias vector across every row of an input tensor:
//   out[i, d] = x[i, d] + bias[d]
//
// This is a pure elementwise op — no GEMM, no reduction. It appears in every
// transformer block after the projection matmuls (c_attn, c_proj, mlp_fc,
// mlp_proj) and in the final LayerNorm.
//
// Shapes:
//   x     [N, D]  — input, row-major, device-resident
//   bias  [D]     — bias vector, device-resident
//   out   [N, D]  — output, row-major, device-resident
//   N     number of rows (typically sequence length S, or 1 for layer-norm bias)
//   D     feature dimension (n_embd=768, 3*n_embd=2304, or 4*n_embd=3072)

#pragma once

#include <cuda_runtime.h>

namespace mgie {

// Launch the BiasAdd kernel.
// Each block handles one row; threads stride over the feature dimension D.
void LaunchBiasAdd(const float* x, const float* bias, float* out,
                   int N, int D, cudaStream_t stream = nullptr);

}  // namespace mgie
