// src/kernels/softmax.cu
// Softmax — Version A (three-pass) + Version B (fused two-pass).

#include "kernels/softmax.hpp"
#include "common/errors.hpp"
#include <float.h>

namespace mgie {
namespace {

__device__ float WarpReduceMax(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}
__device__ float WarpReduceSum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}
__device__ float BlockReduceMax(float v, float* smem) {
  const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  v = WarpReduceMax(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (threadIdx.x < blockDim.x/32) ? smem[lane] : -FLT_MAX;
  if (wid == 0) v = WarpReduceMax(v);
  __syncthreads();
  if (threadIdx.x == 0) smem[0] = v;
  __syncthreads();
  return smem[0];
}
__device__ float BlockReduceSum(float v, float* smem) {
  const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  v = WarpReduceSum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (threadIdx.x < blockDim.x/32) ? smem[lane] : 0.f;
  if (wid == 0) v = WarpReduceSum(v);
  __syncthreads();
  if (threadIdx.x == 0) smem[0] = v;
  __syncthreads();
  return smem[0];
}

// ---- Version A: three-pass (max, exp+sum, divide) ---------------------------
__global__ void SoftmaxNaiveKernel(const float* __restrict__ x,
                                    float* __restrict__ out, int D) {
  extern __shared__ float smem[];
  const float* xr = x + blockIdx.x * D;
  float*       or_ = out + blockIdx.x * D;

  float mx = -FLT_MAX;
  for (int j = threadIdx.x; j < D; j += blockDim.x) mx = fmaxf(mx, xr[j]);
  mx = BlockReduceMax(mx, smem);

  float s = 0.f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    const float e = expf(xr[j] - mx);
    or_[j] = e;
    s += e;
  }
  s = BlockReduceSum(s, smem);

  for (int j = threadIdx.x; j < D; j += blockDim.x) or_[j] /= s;
}

// ---- Version B: fused two-pass (single launch) ------------------------------
// Pass 1 (inside kernel): read x once to compute max AND exp+sum simultaneously.
//   Phase A: reduce max → smem[0]  (__syncthreads between phases)
//   Phase B: compute exp(x-max) and accumulate partial sums → smem[0]
// Pass 2: normalize using already-computed sum (second read of exp buffer).
// Two global reads of x; one read of the intermediate exp array.
__global__ void SoftmaxFusedKernel(const float* __restrict__ x,
                                    float* __restrict__ out, int D) {
  extern __shared__ float smem[];
  const float* xr = x + blockIdx.x * D;
  float*       or_ = out + blockIdx.x * D;

  // Phase A: max reduction (reads x once).
  float mx = -FLT_MAX;
  for (int j = threadIdx.x; j < D; j += blockDim.x) mx = fmaxf(mx, xr[j]);
  mx = BlockReduceMax(mx, smem);  // smem[0] = row max after this

  // Phase B: exp(x-max) into out[], accumulate sum.
  // The __syncthreads inside BlockReduceMax ensures smem[0]=max is visible.
  float s = 0.f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    const float e = expf(xr[j] - smem[0]);
    or_[j] = e;
    s += e;
  }
  s = BlockReduceSum(s, smem);  // smem[0] = row sum after this

  // Normalization pass (reads out[] which was written by Phase B).
  const float inv_s = 1.f / smem[0];
  for (int j = threadIdx.x; j < D; j += blockDim.x) or_[j] *= inv_s;
}

}  // namespace

void LaunchSoftmax(const float* x, float* out, int N, int D,
                   SoftmaxVersion version, cudaStream_t stream) {
  if (N == 0 || D == 0) return;
  const int block = 256;
  const int smem  = (block / 32) * sizeof(float);
  const dim3 grid(N);
  if (version == SoftmaxVersion::Naive)
    SoftmaxNaiveKernel<<<grid, block, smem, stream>>>(x, out, D);
  else
    SoftmaxFusedKernel<<<grid, block, smem, stream>>>(x, out, D);
}

}  // namespace mgie
