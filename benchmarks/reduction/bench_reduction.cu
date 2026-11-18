// benchmarks/reduction/bench_reduction.cu
//
// Phase 1 benchmark: sum-reduction CPU baseline + three GPU versions.
//
// Compares, at three problem sizes (1M / 16M / 64M):
//   * reduction_cpu          - double-accumulate host loop (std::chrono timed)
//   * reduction_naive        - Version A (interleaved addressing)
//   * reduction_shared       - Version B (sequential addressing)
//   * reduction_warp_shuffle - Version C (__shfl_down_sync)
//
// Reduction reads each input element once (one pass) and writes a scalar, so it
// is memory-bound: bytes = n * sizeof(float). Achieved bandwidth vs the card's
// theoretical peak is the headline metric (PROJECT_SPEC: >=70% pass, >=80%
// target at large n).
//
// Methodology (docs/benchmarking.md): Google Benchmark drives repetition; CUDA
// events (timer.hpp) measure GPU time; bench_harness derives bandwidth/GFLOPS
// and appends the canonical CSV row. The ReductionWorkspace is allocated ONCE
// before the timed loop so cudaMalloc never enters the measurement.

#include <benchmark/benchmark.h>

#include <chrono>
#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/reduction.hpp"
#include "common/bench_harness.hpp"

namespace {

namespace mb = mgie::bench;

// Deterministic input so runs are comparable.
std::vector<float> MakeInput(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<float>((i % 17) + 1) * 0.0625f;
  }
  return v;
}

// ---- GPU versions ----------------------------------------------------------

void RunGpuBenchmark(benchmark::State& state, mgie::ReductionVersion version,
                     const char* csv_name) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    state.SkipWithError("no CUDA device available");
    return;
  }

  const std::vector<float> h_in = MakeInput(n);
  mgie::DeviceBuffer<float> d_in(n), d_out(1);
  d_in.CopyFromHost(h_in.data(), n);

  // Allocate workspace ONCE, before timing — excludes cudaMalloc from the loop.
  mgie::ReductionWorkspace ws;

  auto body = [&] {
    mgie::LaunchReduce(d_in.data(), d_out.data(), n, ws, version);
  };

  for (auto _ : state) {
    const std::vector<float> times = mgie::TimeKernel(body, /*iterations=*/50,
                                                      /*warmup=*/10);
    state.SetIterationTime(mgie::Mean(times) * 1.0e-3);  // seconds
  }

  const std::vector<float> times =
      mgie::TimeKernel(body, /*iterations=*/100, /*warmup=*/10);
  const double mean_ms = mgie::Mean(times);

  const std::size_t bytes = n * sizeof(float);  // one read pass
  const std::size_t flops = n > 0 ? n - 1 : 0;  // one add per pair

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult result;
  result.kernel_name = csv_name;
  result.input_shape = "n=" + std::to_string(n);
  result.time_ms = mean_ms;
  result.time_stddev_ms = mgie::StdDev(times);
  result.bandwidth_gbs = mb::BandwidthGBs(bytes, mean_ms);
  result.gflops = mb::GFlops(flops, mean_ms);

  state.counters["bandwidth_GBs"] = result.bandwidth_gbs;
  state.counters["GFLOPS"] = result.gflops;
  state.counters["pct_peak_bw"] =
      100.0 * result.bandwidth_gbs / dev.TheoreticalBandwidthGBs();

  mb::AppendCsv("reduction_results.csv", result, dev);
}

void BM_Reduction_Naive(benchmark::State& state) {
  RunGpuBenchmark(state, mgie::ReductionVersion::Naive, "reduction_naive");
}
void BM_Reduction_Shared(benchmark::State& state) {
  RunGpuBenchmark(state, mgie::ReductionVersion::SharedMem, "reduction_shared");
}
void BM_Reduction_WarpShuffle(benchmark::State& state) {
  RunGpuBenchmark(state, mgie::ReductionVersion::WarpShuffle,
                  "reduction_warp_shuffle");
}

// ---- CPU baseline ----------------------------------------------------------

void BM_Reduction_CPU(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  const std::vector<float> h_in = MakeInput(n);

  auto reduce = [&] {
    double acc = 0.0;
    for (float v : h_in) acc += static_cast<double>(v);
    return acc;
  };

  std::vector<double> times_ms;
  for (auto _ : state) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const double sum = reduce();
    const auto t1 = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(sum);
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    times_ms.push_back(ms);
    state.SetIterationTime(ms * 1.0e-3);
  }

  double mean_ms = 0.0;
  for (double t : times_ms) mean_ms += t;
  if (!times_ms.empty()) mean_ms /= static_cast<double>(times_ms.size());

  const std::size_t bytes = n * sizeof(float);
  const std::size_t flops = n > 0 ? n - 1 : 0;

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult result;
  result.kernel_name = "reduction_cpu";
  result.input_shape = "n=" + std::to_string(n);
  result.time_ms = mean_ms;
  result.bandwidth_gbs = mb::BandwidthGBs(bytes, mean_ms);
  result.gflops = mb::GFlops(flops, mean_ms);

  state.counters["bandwidth_GBs"] = result.bandwidth_gbs;
  state.counters["GFLOPS"] = result.gflops;

  mb::AppendCsv("reduction_results.csv", result, dev);
}

constexpr int kSize1M = 1 << 20;
constexpr int kSize16M = 1 << 24;
constexpr int kSize64M = 1 << 26;

BENCHMARK(BM_Reduction_CPU)
    ->Arg(kSize1M)->Arg(kSize16M)->Arg(kSize64M)->UseManualTime();
BENCHMARK(BM_Reduction_Naive)
    ->Arg(kSize1M)->Arg(kSize16M)->Arg(kSize64M)->UseManualTime();
BENCHMARK(BM_Reduction_Shared)
    ->Arg(kSize1M)->Arg(kSize16M)->Arg(kSize64M)->UseManualTime();
BENCHMARK(BM_Reduction_WarpShuffle)
    ->Arg(kSize1M)->Arg(kSize16M)->Arg(kSize64M)->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
