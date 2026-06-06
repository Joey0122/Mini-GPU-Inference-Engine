// benchmarks/reduction/bench_vector_add.cu
//
// Phase 0 benchmark for vector_add (the bandwidth-bound reference example).
//
// Lives under reduction/ because vector_add is the memory-bandwidth warm-up for
// the Phase 1 reduction work that this directory will hold; it establishes the
// "achieved vs. theoretical bandwidth" methodology every later benchmark reuses.
//
// vector_add moves 3 floats per element (2 read + 1 write) and does 1 add per
// element, so it is firmly memory-bound: we expect achieved bandwidth to
// approach peak and GFLOPS to be tiny. That contrast is the teaching point.
//
// Google Benchmark runs the outer loop; CUDA events (timer.hpp) measure GPU
// time; bench_harness derives bandwidth/GFLOPS and appends a CSV row.

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/vector_add.hpp"
#include "common/bench_harness.hpp"

namespace {

namespace mb = mgie::bench;

void BM_VectorAdd(benchmark::State& state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));

  // Skip cleanly if no GPU is present.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    state.SkipWithError("no CUDA device available");
    return;
  }

  std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f);
  mgie::DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.CopyFromHost(h_a.data(), n);
  d_b.CopyFromHost(h_b.data(), n);

  auto body = [&] {
    mgie::LaunchVectorAdd(d_a.data(), d_b.data(), d_c.data(), n);
  };

  // Warm up + time with CUDA events; let Google Benchmark drive repetitions.
  for (auto _ : state) {
    const std::vector<float> times = mgie::TimeKernel(body, /*iterations=*/50,
                                                      /*warmup=*/10);
    state.SetIterationTime(mgie::Mean(times) * 1.0e-3);  // seconds
  }

  const std::vector<float> times =
      mgie::TimeKernel(body, /*iterations=*/100, /*warmup=*/10);
  const double mean_ms = mgie::Mean(times);

  // 3 floats moved per element (2 loads + 1 store); 1 add per element.
  const std::size_t bytes = n * 3 * sizeof(float);
  const std::size_t flops = n;  // one add per element

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult result;
  result.kernel_name = "vector_add";
  result.input_shape = "n=" + std::to_string(n);
  result.time_ms = mean_ms;
  result.time_stddev_ms = mgie::StdDev(times);
  result.bandwidth_gbs = mb::BandwidthGBs(bytes, mean_ms);
  result.gflops = mb::GFlops(flops, mean_ms);

  state.counters["bandwidth_GBs"] = result.bandwidth_gbs;
  state.counters["GFLOPS"] = result.gflops;
  state.counters["pct_peak_bw"] =
      100.0 * result.bandwidth_gbs / dev.TheoreticalBandwidthGBs();

  mb::AppendCsv("vector_add_results.csv", result, dev);
}

BENCHMARK(BM_VectorAdd)
    ->Arg(1 << 16)
    ->Arg(1 << 20)
    ->Arg(1 << 24)
    ->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
