// src/kernels/layernorm.cu
// Phase 3: LayerNorm — Version A (two-pass naive) + Version B (Welford).

#include "kernels/layernorm.hpp"
#include "common/errors.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace mgie {
namespace {

// ---- Version A: naive two-pass per row ----------------------------------------
// Pass 1 reduces mean and variance using shared memory + warp shuffle.
// Pass 2 normalizes and applies the affine transform.
// Two kernel launches; each block handles one row.

__device__ float WarpReduceSum(float val) {
  for (int off = 16; off > 0; off >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, off);
  return val;
}

// Shared-memory block reduction: returns the sum across all threads.
__device__ float BlockReduceSum(float val, float* smem) {
  const int lane = threadIdx.x % 32;
  const int wid  = threadIdx.x / 32;
  val = WarpReduceSum(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / 32)) ? smem[lane] : 0.f;
  if (wid == 0) val = WarpReduceSum(val);
  return val;
}

__global__ void LayerNormNaiveKernel(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     int H, float eps) {
  extern __shared__ float smem[];  // [0..nwarps-1]: warp sums; [nwarps]: broadcast mean/inv_std
  const int nwarps = blockDim.x / 32;
  const int row = blockIdx.x;
  const float* xr = x + row * H;
  float* or_ = out + row * H;

  // Pass 1a: mean — reduce sum, broadcast via smem[nwarps]
  float s = 0.f;
  for (int j = threadIdx.x; j < H; j += blockDim.x) s += xr[j];
  s = BlockReduceSum(s, smem);
  if (threadIdx.x == 0) smem[nwarps] = s / static_cast<float>(H);  // mean
  __syncthreads();
  const float mean = smem[nwarps];

  // Pass 1b: variance — reduce sum of (x-mean)^2, broadcast via smem[nwarps]
  float v = 0.f;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    float d = xr[j] - mean; v += d * d;
  }
  v = BlockReduceSum(v, smem);
  if (threadIdx.x == 0) smem[nwarps] = rsqrtf(v / static_cast<float>(H) + eps);
  __syncthreads();
  const float inv_std = smem[nwarps];

  for (int j = threadIdx.x; j < H; j += blockDim.x)
    or_[j] = gamma[j] * (xr[j] - mean) * inv_std + beta[j];
}

// ---- Version B: single-pass block-wide Welford --------------------------------
// Each thread accumulates a (count, mean, M2) partial over its slice via the
// serial Welford recurrence. Partials are combined with the Welford combine
// operator using warp shuffle then a small shared-memory reduction.

struct WelfordState { float mean, M2; int n; };

__device__ WelfordState WelfordCombine(WelfordState a, WelfordState b) {
  if (b.n == 0) return a;
  if (a.n == 0) return b;
  const int n  = a.n + b.n;
  const float d = b.mean - a.mean;
  return { a.mean + d * static_cast<float>(b.n) / static_cast<float>(n),
           a.M2 + b.M2 + d * d * static_cast<float>(a.n) * static_cast<float>(b.n)
                               / static_cast<float>(n),
           n };
}

__device__ WelfordState WarpReduceWelford(WelfordState w) {
  for (int off = 16; off > 0; off >>= 1) {
    WelfordState other;
    other.mean = __shfl_down_sync(0xffffffffu, w.mean, off);
    other.M2   = __shfl_down_sync(0xffffffffu, w.M2,   off);
    other.n    = __shfl_down_sync(0xffffffffu, w.n,    off);
    if (threadIdx.x % 32 + off < 32) w = WelfordCombine(w, other);
  }
  return w;
}

__global__ void LayerNormWelfordKernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       int H, float eps) {
  // smem: warp-leader WelfordState (3 floats per warp leader's mean/M2/n)
  extern __shared__ float smem[];
  float* smem_mean = smem;
  float* smem_m2   = smem + blockDim.x / 32;
  int*   smem_n    = reinterpret_cast<int*>(smem + 2 * (blockDim.x / 32));

  const int row  = blockIdx.x;
  const float* xr = x + row * H;
  float* or_ = out + row * H;

  // Per-thread Welford over the thread's slice of the row.
  WelfordState w{0.f, 0.f, 0};
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    ++w.n;
    float d = xr[j] - w.mean;
    w.mean += d / static_cast<float>(w.n);
    w.M2   += d * (xr[j] - w.mean);
  }

  // Warp-level combine.
  w = WarpReduceWelford(w);

  // Write warp leaders to shared memory.
  const int lane = threadIdx.x % 32;
  const int wid  = threadIdx.x / 32;
  if (lane == 0) { smem_mean[wid] = w.mean; smem_m2[wid] = w.M2; smem_n[wid] = w.n; }
  __syncthreads();

  // Thread 0 combines warp partials.
  if (threadIdx.x == 0) {
    WelfordState total{0.f, 0.f, 0};
    const int nwarps = blockDim.x / 32;
    for (int i = 0; i < nwarps; ++i) {
      WelfordState wi{smem_mean[i], smem_m2[i], smem_n[i]};
      total = WelfordCombine(total, wi);
    }
    smem_mean[0] = total.mean;
    smem_m2[0]   = total.M2 / static_cast<float>(total.n);  // variance
  }
  __syncthreads();

  const float mean    = smem_mean[0];
  const float inv_std = rsqrtf(smem_m2[0] + eps);

  for (int j = threadIdx.x; j < H; j += blockDim.x)
    or_[j] = gamma[j] * (xr[j] - mean) * inv_std + beta[j];
}

}  // namespace

void LaunchLayerNorm(const float* x, float* out,
                     const float* gamma, const float* beta,
                     int N, int H, float eps,
                     LayerNormVersion version, cudaStream_t stream) {
  if (N == 0 || H == 0) return;
  // 256 threads; each block handles one row; shared needs blockDim/32 slots.
  const int block = 256;
  const int nwarps = block / 32;
  // smem: for Welford, 2*nwarps floats + nwarps ints; for Naive, nwarps floats.
  const int smem_bytes = (version == LayerNormVersion::Welford)
      ? (2 * nwarps * sizeof(float) + nwarps * sizeof(int))
      : ((nwarps + 1) * sizeof(float));  // +1 slot for mean/inv_std broadcast
  const dim3 grid(N);
  if (version == LayerNormVersion::Naive)
    LayerNormNaiveKernel<<<grid, block, smem_bytes, stream>>>(x, out, gamma, beta, H, eps);
  else
    LayerNormWelfordKernel<<<grid, block, smem_bytes, stream>>>(x, out, gamma, beta, H, eps);
}

}  // namespace mgie
