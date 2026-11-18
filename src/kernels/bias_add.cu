// src/kernels/bias_add.cu
// Phase 5 C5: row-wise bias addition kernel.
//
// Design (naive-first, following AGENT.md kernel process):
//   - 1D grid over rows (blockIdx.x = i)
//   - 1D block over feature dimension (threadIdx.x = d), strided over D
//   - Each thread handles out[i, d] = x[i, d] + bias[d]
//
// The bias vector is small (D ≤ 3072 for GPT-2 Small) and read by every row;
// it sits in L1/L2 cache after the first row. The row-wise grid design ensures
// coalesced reads from x and coalesced writes to out, with bias reads broadcast
// through the L1 cache of each SM.

#include "kernels/bias_add.hpp"

#include "common/errors.hpp"

namespace mgie {
namespace {

// Block size covers the feature dimension. 256 threads → 3 stride iterations
// for D=768 (GPT-2 Small), 9 for D=2304, 12 for D=3072.
constexpr int kBlockDim = 256;

__global__ void BiasAddKernel(const float* __restrict__ x,
                               const float* __restrict__ bias,
                               float* __restrict__ out,
                               int N, int D) {
  const int i = blockIdx.x;
  if (i >= N) return;

  const std::size_t row = static_cast<std::size_t>(i) * D;

  // Strided loop: each thread handles elements d, d+blockDim, d+2*blockDim, ...
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    out[row + d] = x[row + d] + bias[d];
  }
}

}  // namespace

void LaunchBiasAdd(const float* x, const float* bias, float* out,
                   int N, int D, cudaStream_t stream) {
  if (N <= 0 || D <= 0) return;

  const int block = (D < kBlockDim) ? D : kBlockDim;
  const dim3 grid(N);
  BiasAddKernel<<<grid, block, 0, stream>>>(x, bias, out, N, D);
}

}  // namespace mgie
