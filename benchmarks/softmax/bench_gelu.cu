// benchmarks/softmax/bench_gelu.cu
// Phase 3: NewGELU bandwidth benchmark.
// GELU is elementwise: 1 read + 1 write = 2*N*sizeof(float) bytes.

#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/gelu.hpp"
#include "common/bench_harness.hpp"

namespace {
namespace mb = mgie::bench;

void BM_GELU(benchmark::State& state) {
  const int N = static_cast<int>(state.range(0));
  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }
  std::vector<float> hx(N, 0.5f);
  mgie::DeviceBuffer<float> dx(N), dout(N);
  dx.CopyFromHost(hx.data(), N);

  auto body = [&] { mgie::LaunchGelu(dx.data(), dout.data(), N); };
  for (auto _ : state) {
    const auto t = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(t) * 1e-3);
  }
  const auto t = mgie::TimeKernel(body, 100, 10);
  const double ms = mgie::Mean(t);
  const std::size_t bytes = 2ULL * N * sizeof(float);

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = "gelu_newgelu";
  r.input_shape    = "n=" + std::to_string(N);
  r.time_ms        = ms;
  r.time_stddev_ms = mgie::StdDev(t);
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, ms);
  r.gflops         = mb::GFlops(static_cast<std::size_t>(N)*8, ms); // ~8 ops/elem

  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  state.counters["pct_peak_bw"]   = 100.0 * r.bandwidth_gbs / dev.TheoreticalBandwidthGBs();
  mb::AppendCsv("gelu_results.csv", r, dev);
}

// FFN-relevant sizes: 64*3072 and 256*3072
BENCHMARK(BM_GELU)->Arg(64*3072)->Arg(256*3072)->Arg(1<<20)->UseManualTime();
} // namespace
BENCHMARK_MAIN();
