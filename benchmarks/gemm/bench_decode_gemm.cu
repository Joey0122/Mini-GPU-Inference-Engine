// benchmarks/gemm/bench_decode_gemm.cu
// Phase 6b: RegTiled128Pad vs GEMV at M=1 decode shapes.
//
// BEFORE (RegTiled128Pad): launches the 128-tile kernel directly at M=1 to
// measure the pre-Phase-6b baseline (bypasses the new M=1 GEMV dispatch).
//
// AFTER (GEMV): calls LaunchGemm(M=1) which dispatches to the dedicated GEMV
// kernel through the new M==1 path.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "common/errors.hpp"
#include "cuda/memory.hpp"
#include "cuda/launch.hpp"
#include "kernels/gemm.hpp"

using mgie::DeviceBuffer;

// ---- BEFORE: RegTiled128Pad kernel at M=1 (bypasses dispatch) ---------------
// RegTiled128Pad tile constants (must match src/kernels/gemm.cu).
constexpr int kBM2 = 128, kBN2 = 128, kBK = 16, kTM2 = 8, kTN2 = 8;
constexpr int kTPB2 = (kBM2 / kTM2) * (kBN2 / kTN2); // 256

// Replicated kernel for benchmarking only — identical to GemmReg128PadKernel.
__global__ void Reg128PadKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
  __shared__ float sA[kBM2][kBK + 1];
  __shared__ float sB[kBK][kBN2 + 1];
  float acc[kTM2][kTN2] = {};
  float ra[kTM2], rb[kTN2];
  const int tr = threadIdx.y, tc = threadIdx.x;
  const int br = blockIdx.y * kBM2, bc = blockIdx.x * kBN2;
  const int tid = tr * (kBN2/kTN2) + tc;
  for (int t = 0; t < (K+kBK-1)/kBK; ++t) {
    for (int i=tid; i<kBM2*kBK; i+=kTPB2) {
      int r=i/kBK, c=i%kBK;
      sA[r][c] = (br+r<M && t*kBK+c<K) ? A[(br+r)*K + t*kBK+c] : 0.f;
    }
    for (int i=tid; i<kBK*kBN2; i+=kTPB2) {
      int r=i/kBN2, c=i%kBN2;
      sB[r][c] = (t*kBK+r<K && bc+c<N) ? B[(t*kBK+r)*N + bc+c] : 0.f;
    }
    __syncthreads();
    for (int k=0; k<kBK; ++k) {
      for (int m=0; m<kTM2; ++m) ra[m] = sA[tr*kTM2+m][k];
      for (int n=0; n<kTN2; ++n) rb[n] = sB[k][tc*kTN2+n];
      for (int m=0; m<kTM2; ++m)
        for (int n=0; n<kTN2; ++n) acc[m][n] += ra[m]*rb[n];
    }
    __syncthreads();
  }
  for (int m=0; m<kTM2; ++m) for (int n=0; n<kTN2; ++n)
    if (br+tr*kTM2+m<M && bc+tc*kTN2+n<N)
      C[(br+tr*kTM2+m)*N + bc+tc*kTN2+n] = acc[m][n];
}

void LaunchReg128PadDirect(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  dim3 block(kBN2/kTN2, kBM2/kTM2);
  dim3 grid(mgie::CeilDiv((unsigned)N, (unsigned)kBN2),
            mgie::CeilDiv((unsigned)M, (unsigned)kBM2));
  Reg128PadKernel<<<grid, block>>>(A, B, C, M, N, K);
}

template<typename F>
float Bench(F&& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t b, e;
  CUDA_CHECK(cudaEventCreate(&b)); CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(b));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, b, e));
  CUDA_CHECK(cudaEventDestroy(b)); CUDA_CHECK(cudaEventDestroy(e));
  return ms * 1000.f / iters;
}

struct Shape { const char* name; int N, K, count; };
const Shape shapes[] = {
  {"c_attn",   2304,  768, 12},
  {"c_proj",    768,  768, 12},
  {"mlp_fc",   3072,  768, 12},
  {"mlp_proj",  768, 3072, 12},
  {"lm_head", 50257,  768,  1},
};

int main() {
  printf("=== Phase 6b: Decode GEMM Before/After (RegTiled128Pad vs GEMV dispatch) ===\n\n");
  printf("%-12s %4s %12s %12s %7s\n", "kernel", "Nx", "Reg128Pad", "GEMV", "spdup");
  printf("%-12s %4s %12s %12s %7s\n", "--------", "---", "----------", "----------", "------");

  float tot_before = 0, tot_after = 0;

  for (auto& s : shapes) {
    int N = s.N, K = s.K;
    DeviceBuffer<float> dA(K), dB((size_t)K * N), dC(N);
    std::vector<float> hA(K, 1.f), hB(K * N, 1.f);
    dA.CopyFromHost(hA.data(), K);
    dB.CopyFromHost(hB.data(), K * N);

    auto Run = [&](auto&& fn) { return Bench(fn, 5, 20); };

    // BEFORE: RegTiled128Pad kernel directly at M=1 (bypasses dispatch)
    float tb = Run([&]() { LaunchReg128PadDirect(dA.data(), dB.data(), dC.data(), 1, N, K); });

    // AFTER: LaunchGemm(M=1) → dispatches to GEMV
    float ta = Run([&]() { mgie::LaunchGemm(dA.data(), dB.data(), dC.data(),
                                             1, N, K, mgie::GemmVersion::RegTiled128Pad); });

    printf("%-12s %4d %12.1f %12.1f %6.1fx\n", s.name, s.count, tb, ta, tb/ta);
    tot_before += s.count * tb;
    tot_after  += s.count * ta;
  }

  printf("%-12s %4s %12s %12s %7s\n", "--------", "---", "----------", "----------", "------");
  printf("%-12s %4s %12.0f %12.0f %6.1fx  us/step\n", "TOTAL", "49", tot_before, tot_after, tot_before/tot_after);

  // Projected decode step
  float base_total = 16171.f;
  float nongemm    = base_total - 14395.f;  // 1776 us from Phase 6a profile
  float latency_before = nongemm + tot_before;
  float latency_after  = nongemm + tot_after;

  printf("\n=== Projected Decode Step Latency ===\n");
  printf("Non-GEMM overhead:        %8.0f us\n", nongemm);
  printf("GEMM (RegTiled128Pad):    %8.0f us\n", tot_before);
  printf("GEMM (GEMV dispatch):     %8.0f us\n", tot_after);
  printf("\n");
  printf("Decode step (BEFORE):     %8.0f us  (%5.1f ms)\n", latency_before, latency_before/1000.f);
  printf("Decode step (AFTER):      %8.0f us  (%5.1f ms)\n", latency_after, latency_after/1000.f);
  printf("Overall speedup:          %.1fx\n", latency_before / latency_after);
  printf("GEMM-only speedup:        %.1fx\n", tot_before / tot_after);

  // Memory BW analysis
  printf("\n=== GEMV Memory BW Efficiency (via dispatch) ===\n");
  printf("%-12s %12s %12s %8s %8s\n", "kernel", "Bytes", "BW(GB/s)", "%%peak", "AI(F/B)");
  for (auto& s : shapes) {
    int N = s.N, K = s.K;
    double bytes = (K + (double)K * N + N) * 4;
    DeviceBuffer<float> dA(K), dB((size_t)K * N), dC(N);
    std::vector<float> hA(K, 1.f), hB(K * N, 1.f);
    dA.CopyFromHost(hA.data(), K);
    dB.CopyFromHost(hB.data(), K * N);
    float t = Bench([&]() {
      mgie::LaunchGemm(dA.data(), dB.data(), dC.data(), 1, N, K, mgie::GemmVersion::RegTiled128Pad);
    }, 5, 20);
    double bw = bytes / t / 1000.;
    double flops = 2.0 * 1.0 * N * K;
    printf("%-12s %12.0f %12.1f %7.1f%% %8.2f\n", s.name, bytes, bw, 100.*bw/256., flops/bytes);
  }

  printf("\nDone.\n");
  return 0;
}
