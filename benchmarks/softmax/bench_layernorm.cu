// benchmarks/softmax/bench_layernorm.cu
// Phase 3: LayerNorm bandwidth benchmark — Versions A and B.
// Both kernels are memory-bound (read H floats, write H floats per row).

#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/layernorm.hpp"
#include "common/bench_harness.hpp"

namespace {
namespace mb = mgie::bench;

void Run(benchmark::State& state, mgie::LayerNormVersion ver, const char* name) {
  const int N = static_cast<int>(state.range(0));
  const int H = static_cast<int>(state.range(1));
  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }
  std::vector<float> hx(N*H, 0.5f), hg(H, 1.f), hb(H, 0.f);
  mgie::DeviceBuffer<float> dx(N*H), dout(N*H), dg(H), db(H);
  dx.CopyFromHost(hx.data(), N*H);
  dg.CopyFromHost(hg.data(), H);
  db.CopyFromHost(hb.data(), H);

  auto body = [&] {
    mgie::LaunchLayerNorm(dx.data(), dout.data(), dg.data(), db.data(), N, H, 1e-5f, ver);
  };
  for (auto _ : state) {
    const auto t = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(t) * 1e-3);
  }
  const auto t = mgie::TimeKernel(body, 100, 10);
  const double ms = mgie::Mean(t);
  // Read x once + write out once = 2*N*H*4 bytes
  const std::size_t bytes = 2ULL * N * H * sizeof(float);

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = name;
  r.input_shape    = std::to_string(N) + "x" + std::to_string(H);
  r.time_ms        = ms;
  r.time_stddev_ms = mgie::StdDev(t);
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, ms);
  r.gflops         = mb::GFlops(static_cast<std::size_t>(N)*H*5, ms); // 5 ops/elem

  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  state.counters["pct_peak_bw"]   = 100.0 * r.bandwidth_gbs / dev.TheoreticalBandwidthGBs();
  mb::AppendCsv("layernorm_results.csv", r, dev);
}

void BM_LN_Naive  (benchmark::State& s) { Run(s, mgie::LayerNormVersion::Naive,   "layernorm_naive"); }
void BM_LN_Welford(benchmark::State& s) { Run(s, mgie::LayerNormVersion::Welford, "layernorm_welford"); }

// Sizes: [N, H] — GPT-2 relevant shapes
BENCHMARK(BM_LN_Naive)  ->Args({1,768})->Args({64,768})->Args({256,768})->Args({64,3072})->UseManualTime();
BENCHMARK(BM_LN_Welford)->Args({1,768})->Args({64,768})->Args({256,768})->Args({64,3072})->UseManualTime();
} // namespace
BENCHMARK_MAIN();
