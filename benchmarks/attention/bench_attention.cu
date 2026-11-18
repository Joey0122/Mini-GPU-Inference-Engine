// benchmarks/attention/bench_attention.cu
//
// Phase 4 benchmark: integrated causal multi-head attention (batch=1).
//
// Sweeps sequence length S over the quadratic regime at GPT-2 Small's head
// configuration (n_head=12, head_dim=64). Reports latency, GFLOPS, and the
// scores-buffer bandwidth. FLOP model per attention call:
//   QKᵀ : 2 * n_head * S² * head_dim   (mul+add per score element)
//   AV  : 2 * n_head * S² * head_dim   (mul+add per output accumulation)
//   total = 4 * n_head * S² * head_dim   (softmax FLOPs are negligible)
//
// At small S the path is launch/latency-bound; at large S the [n_head,S,S]
// scores traffic dominates (memory-bound) — the teaching contrast. cuBLAS/
// cuDNN are not used (forbidden as the implementation); PyTorch eager is the
// external comparison, recorded out-of-band in the Phase 4 report.

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/attention.hpp"
#include "common/bench_harness.hpp"

namespace {

namespace mb = mgie::bench;

constexpr int kNHead = 12;     // GPT-2 Small
constexpr int kHeadDim = 64;

std::vector<float> Fill(int n, float seed_scale) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = seed_scale * ((i % 13) - 6) / 6.0f;
  return v;
}

void BM_Attention(benchmark::State& state) {
  const int S = static_cast<int>(state.range(0));

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const int qn = kNHead * S * kHeadDim;
  const int sn = kNHead * S * S;
  const int on = S * kNHead * kHeadDim;
  const auto hq = Fill(qn, 0.5f), hk = Fill(qn, 0.3f), hv = Fill(qn, 0.4f);

  mgie::DeviceBuffer<float> dq(qn), dk(qn), dv(qn), dscores(sn), dweights(sn), dout(on);
  dq.CopyFromHost(hq.data(), qn);
  dk.CopyFromHost(hk.data(), qn);
  dv.CopyFromHost(hv.data(), qn);

  auto body = [&] {
    mgie::LaunchAttention(dq.data(), dk.data(), dv.data(), dout.data(),
                          dscores.data(), dweights.data(), kNHead, S, kHeadDim);
  };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 50, 10);
  const double mean_ms = mgie::Mean(times);

  const std::size_t flops = 4ULL * kNHead * static_cast<std::size_t>(S) * S * kHeadDim;
  // Dominant traffic: scores written (QKᵀ) + read (softmax) + weights written
  // (softmax) + read (AV) = 4 * n_head*S² floats; plus Q/K/V/out (O(S) terms).
  const std::size_t bytes = 4ULL * sn * sizeof(float) + 4ULL * qn * sizeof(float);

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = "attention_integrated";
  r.input_shape    = "h" + std::to_string(kNHead) + "_s" + std::to_string(S) +
                     "_d" + std::to_string(kHeadDim);
  r.time_ms        = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops         = mb::GFlops(flops, mean_ms);
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, mean_ms);

  state.counters["GFLOPS"] = r.gflops;
  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  state.counters["scores_MB"] = static_cast<double>(sn * sizeof(float)) / 1e6;
  mb::AppendCsv("attention_results.csv", r, dev);
}

BENCHMARK(BM_Attention)
    ->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
    ->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
