// src/kernels/gelu.cu
// NewGELU — 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// GPT-2 compatible; matches F.gelu(approximate='tanh').

#include "kernels/gelu.hpp"
#include "cuda/launch.hpp"

namespace mgie {
namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCoeff       = 0.044715f;

__global__ void GeluKernel(const float* __restrict__ x,
                            float* __restrict__ out, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float v = x[i];
  out[i] = 0.5f * v * (1.f + tanhf(kSqrt2OverPi * (v + kCoeff * v*v*v)));
}

}  // namespace

void LaunchGelu(const float* x, float* out, int N, cudaStream_t stream) {
  if (N == 0) return;
  const dim3 block(kDefaultBlockSize);
  const dim3 grid(CeilDiv(static_cast<unsigned>(N),
                          static_cast<unsigned>(kDefaultBlockSize)));
  GeluKernel<<<grid, block, 0, stream>>>(x, out, N);
}

}  // namespace mgie
