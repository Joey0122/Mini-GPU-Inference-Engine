// src/kernels/vector_add.cu
//
// Elementwise addition for residual connections and token embeddings.

#include "kernels/vector_add.hpp"

#include "cuda/launch.hpp"

namespace mgie {

namespace {

__global__ void VectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

void LaunchVectorAdd(const float* a, const float* b, float* c, std::size_t n,
                     cudaStream_t stream) {
  if (n == 0) return;
  const dim3 block(kDefaultBlockSize);
  const dim3 grid = Grid1D(n, kDefaultBlockSize);
  VectorAddKernel<<<grid, block, 0, stream>>>(a, b, c, n);
}

}  // namespace mgie
