// benchmarks/bias_add/bench_bias_add.cu
//
// Phase 5 C5 benchmark: row-wise bias addition.
//
// Sweeps N and D across GPT-2 Small's typical bias shapes:
//   - N=1,  D=768   (ln_f bias)
//   - N=64, D=768   (short sequence, c_proj/mlp_proj)
//   - N=64, D=2304  (short sequence, c_attn)
//   - N=256, D=768  (medium sequence, c_proj/mlp_proj)
//   - N=256, D=3072 (medium sequence, mlp_fc)
//   - N=1024, D=768 (full context, c_proj/mlp_proj)
//
// The bias vector is small and cache-resident after the first row; the kernel
// is memory-bandwidth-bound on the x read + out write streams.
//
// Bandwidth note: the embedding benchmark showed that reported bandwidth can
// exceed physical DRAM bandwidth (~256 GB/s for RTX 4070 Laptop) when the
// working set fits in L2 cache. The BiasAdd working set is much smaller
// (bias is ~3 KB at D=768, each row is ~3 KB), so reported BW may also exceed
// DRAM limits. This is a useful-bandwidth metric, not DRAM bus throughput.

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "common/bench_harness.hpp"
#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/bias_add.hpp"

namespace {

namespace mb = mgie::bench;

std::vector<float> Fill(int n, float seed_scale) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = seed_scale * ((i % 13) - 6) / 6.0f;
  return v;
}

void BM_BiasAdd(benchmark::State& state) {
  const int N = static_cast<int>(state.range(0));
  const int D = static_cast<int>(state.range(1));

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const auto h_x = Fill(N * D, 0.5f);
  const auto h_bias = Fill(D, 0.3f);

  mgie::DeviceBuffer<float> dx(N * D), db(D), dout(N * D);
  dx.CopyFromHost(h_x.data(), N * D);
  db.CopyFromHost(h_bias.data(), D);

  auto body = [&] {
    mgie::LaunchBiasAdd(dx.data(), db.data(), dout.data(), N, D);
  };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 50, 10);
  const double mean_ms = mgie::Mean(times);

  // Bytes: read x[N*D] (coalesced), read bias[D] (cached after row 0),
  //        write out[N*D] (coalesced).
  const std::size_t bytes =
      2ULL * static_cast<std::size_t>(N) * D * sizeof(float)    // x read + out write
      + static_cast<std::size_t>(D) * sizeof(float);             // bias read (negligible)
  // FLOPs: one add per element = N*D.
  const std::size_t flops = static_cast<std::size_t>(N) * D;

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = "bias_add";
  r.input_shape    = "n" + std::to_string(N) + "_d" + std::to_string(D);
  r.time_ms        = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops         = mb::GFlops(flops, mean_ms);
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, mean_ms);

  state.counters["GFLOPS"] = r.gflops;
  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  mb::AppendCsv("bias_add_results.csv", r, dev);
}

BENCHMARK(BM_BiasAdd)
    ->Args({1, 768})
    ->Args({64, 768})
    ->Args({64, 2304})
    ->Args({256, 768})
    ->Args({256, 3072})
    ->Args({1024, 768})
    ->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
