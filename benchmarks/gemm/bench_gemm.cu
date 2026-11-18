// benchmarks/gemm/bench_gemm.cu
//
// Phase 2 benchmark: GEMM — CPU, all GPU versions, cuBLAS baseline.
// Metric: GFLOPS = 2*M*N*K / time_s / 1e9 (arithmetic intensity is O(N)).
// PROJECT_SPEC gate: FP32 GEMM >= 50% of cuBLAS GFLOPS at 4096^3.

#include <benchmark/benchmark.h>
#include <cublas_v2.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/gemm.hpp"
#include "common/bench_harness.hpp"

namespace {

namespace mb = mgie::bench;

// Deterministic input.
std::vector<float> MakeMatrix(int rows, int cols, float fill = 0.5f) {
  std::vector<float> v(rows * cols);
  for (int i = 0; i < rows * cols; ++i)
    v[i] = fill * ((i % 7) - 3) / 3.0f;
  return v;
}

// ---- GPU versions ----------------------------------------------------------

void RunGpuBenchmark(benchmark::State& state, mgie::GemmVersion version,
                     const char* csv_name) {
  const int M = static_cast<int>(state.range(0));
  const int N = M, K = M;

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const auto hA = MakeMatrix(M, K);
  const auto hB = MakeMatrix(K, N, 0.3f);
  mgie::DeviceBuffer<float> dA(M*K), dB(K*N), dC(M*N);
  dA.CopyFromHost(hA.data(), M*K);
  dB.CopyFromHost(hB.data(), K*N);

  auto body = [&] { mgie::LaunchGemm(dA.data(), dB.data(), dC.data(), M, N, K, version); };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 50, 10);
  const double mean_ms = mgie::Mean(times);
  const std::size_t flops = 2ULL * M * N * K;

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = csv_name;
  r.input_shape    = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
  r.time_ms        = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops         = mb::GFlops(flops, mean_ms);
  r.bandwidth_gbs  = 0.0;  // GEMM is compute-bound at large N; BW not the headline

  state.counters["GFLOPS"] = r.gflops;
  mb::AppendCsv("gemm_results.csv", r, dev);
}

void BM_Gemm_Naive        (benchmark::State& s) { RunGpuBenchmark(s, mgie::GemmVersion::Naive,         "gemm_naive"); }
void BM_Gemm_Tiled        (benchmark::State& s) { RunGpuBenchmark(s, mgie::GemmVersion::TiledShared,   "gemm_tiled"); }
void BM_Gemm_RegTiled     (benchmark::State& s) { RunGpuBenchmark(s, mgie::GemmVersion::RegisterTiled, "gemm_register_tiled"); }
void BM_Gemm_DoubleBuf    (benchmark::State& s) { RunGpuBenchmark(s, mgie::GemmVersion::DoubleBuf,     "gemm_double_buffered"); }

// ---- cuBLAS baseline -------------------------------------------------------

void BM_Gemm_CuBLAS(benchmark::State& state) {
  const int M = static_cast<int>(state.range(0));
  const int N = M, K = M;

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const auto hA = MakeMatrix(M, K);
  const auto hB = MakeMatrix(K, N, 0.3f);
  mgie::DeviceBuffer<float> dA(M*K), dB(K*N), dC(M*N);
  dA.CopyFromHost(hA.data(), M*K);
  dB.CopyFromHost(hB.data(), K*N);

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    state.SkipWithError("cublasCreate failed");
    return;
  }
  if (cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    state.SkipWithError("could not enable strict FP32 cuBLAS math");
    return;
  }
  const float alpha = 1.0f, beta = 0.0f;
  // Row-major A*B via cuBLAS column-major: C^T = B^T * A^T
  auto body = [&] {
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, dB.data(), N, dA.data(), K, &beta, dC.data(), N)
        != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublasSgemm failed");
  };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 50, 10);
  const double mean_ms = mgie::Mean(times);
  const std::size_t flops = 2ULL * M * N * K;

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name   = "gemm_cublas_fp32";
  r.input_shape   = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
  r.time_ms       = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops        = mb::GFlops(flops, mean_ms);

  state.counters["GFLOPS"] = r.gflops;
  mb::AppendCsv("gemm_results.csv", r, dev);
  cublasDestroy(handle);
}

// ---- CPU baseline (small sizes only — skip 4096) --------------------------

void BM_Gemm_CPU(benchmark::State& state) {
  const int M = static_cast<int>(state.range(0));
  const int N = M, K = M;
  const auto hA = MakeMatrix(M, K);
  const auto hB = MakeMatrix(K, N, 0.3f);

  for (auto _ : state) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    // naive triple loop
    volatile double sink = 0.0;
    std::vector<double> C(M * N, 0.0);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
          C[i*N+j] += hA[i*K+k] * hB[k*N+j];
    for (auto v : C) sink += v;
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.SetIterationTime(ms * 1e-3);
    benchmark::DoNotOptimize(sink);
  }

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name = "gemm_cpu";
  r.input_shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
  r.time_ms     = 0.0;  // not collecting stddev for CPU
  r.gflops      = 0.0;
  mb::AppendCsv("gemm_results.csv", r, dev);
}

void BM_Gemm_Reg128        (benchmark::State& s) { RunGpuBenchmark(s, mgie::GemmVersion::RegTiled128,          "gemm_reg128"); }
void BM_Gemm_Reg128Pad     (benchmark::State& s) { RunGpuBenchmark(s, mgie::GemmVersion::RegTiled128Pad,       "gemm_reg128_pad"); }
void BM_Gemm_Reg128PadUnroll(benchmark::State& s){ RunGpuBenchmark(s, mgie::GemmVersion::RegTiled128PadUnroll, "gemm_reg128_pad_unroll"); }

// Sizes: 512, 1024, 4096 (4096 skipped for CPU/Naive at runtime via min_time).
BENCHMARK(BM_Gemm_CPU)          ->Arg(512)->Arg(1024)->UseManualTime();
BENCHMARK(BM_Gemm_Naive)        ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_Tiled)        ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_RegTiled)     ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_DoubleBuf)    ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_Reg128)       ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_Reg128Pad)    ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_Reg128PadUnroll)->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();
BENCHMARK(BM_Gemm_CuBLAS)       ->Arg(512)->Arg(1024)->Arg(4096)->UseManualTime();

}  // namespace
BENCHMARK_MAIN();
