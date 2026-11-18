// benchmarks/softmax/bench_softmax.cu
// Phase 3: Softmax bandwidth benchmark — Versions A (three-pass) and B (fused).

#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/softmax.hpp"
#include "common/bench_harness.hpp"

namespace {
namespace mb = mgie::bench;

void Run(benchmark::State& state, mgie::SoftmaxVersion ver, const char* name) {
  const int N = static_cast<int>(state.range(0));
  const int D = static_cast<int>(state.range(1));
  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }
  std::vector<float> hx(N*D);
  for (int i = 0; i < N*D; ++i) hx[i] = (i % 17) * 0.1f;
  mgie::DeviceBuffer<float> dx(N*D), dout(N*D);
  dx.CopyFromHost(hx.data(), N*D);

  auto body = [&] { mgie::LaunchSoftmax(dx.data(), dout.data(), N, D, ver); };
  for (auto _ : state) {
    const auto t = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(t) * 1e-3);
  }
  const auto t = mgie::TimeKernel(body, 100, 10);
  const double ms = mgie::Mean(t);
  // Naive: 3 reads + 1 write ≈ 4 passes; Fused: 2 reads + 1 write ≈ 3 passes
  const int passes = (ver == mgie::SoftmaxVersion::Naive) ? 4 : 3;
  const std::size_t bytes = static_cast<std::size_t>(passes) * N * D * sizeof(float);

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = name;
  r.input_shape    = std::to_string(N) + "x" + std::to_string(D);
  r.time_ms        = ms;
  r.time_stddev_ms = mgie::StdDev(t);
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, ms);
  r.gflops         = mb::GFlops(static_cast<std::size_t>(N)*D*5, ms);

  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  state.counters["pct_peak_bw"]   = 100.0 * r.bandwidth_gbs / dev.TheoreticalBandwidthGBs();
  mb::AppendCsv("softmax_results.csv", r, dev);
}

void BM_SM_Naive(benchmark::State& s) { Run(s, mgie::SoftmaxVersion::Naive, "softmax_naive"); }
void BM_SM_Fused(benchmark::State& s) { Run(s, mgie::SoftmaxVersion::Fused, "softmax_fused"); }

BENCHMARK(BM_SM_Naive)->Args({64,512})->Args({64,1024})->Args({64,2048})->Args({64,4096})->UseManualTime();
BENCHMARK(BM_SM_Fused)->Args({64,512})->Args({64,1024})->Args({64,2048})->Args({64,4096})->UseManualTime();
} // namespace
BENCHMARK_MAIN();
