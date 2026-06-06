// src/kernels/vector_add.cu
//
// Phase 0 vector-add kernel + launch wrapper.
//
// Proves the CUDA toolchain compiles and runs, and gives the benchmark/test
// harness a real (if trivial) kernel to exercise. Memory-bound by nature:
// 2 reads + 1 write per element, 1 add per element. Used in Phase 0 docs as
// the worked example of bandwidth-limited performance.
//
// Kept naive on purpose — Phase 0 does NOT optimize. The optimization
// methodology (naive -> shared -> warp -> advanced) begins in Phase 1 with
// reduction.

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
