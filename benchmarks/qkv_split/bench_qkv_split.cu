// benchmarks/qkv_split/bench_qkv_split.cu
//
// Phase 5 C6 benchmark: c_attn output [S, 3*n_embd] → head-major Q, K, V.
//
// Sweeps S at GPT-2 Small's n_head=12, head_dim=64 (n_embd=768).
// The kernel is pure data rearrangement — no floating-point arithmetic besides
// the three load+store pairs per thread. This is a memory-bandwidth benchmark:
// it reads S*3*n_embd and writes 3*S*n_embd elements.

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "common/bench_harness.hpp"
#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/qkv_split.hpp"

namespace {

namespace mb = mgie::bench;

constexpr int kNHead   = 12;   // GPT-2 Small
constexpr int kHeadDim = 64;
constexpr int kNEmbd   = kNHead * kHeadDim;  // 768

std::vector<float> Fill(int n, float seed_scale) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = seed_scale * ((i % 13) - 6) / 6.0f;
  return v;
}

void BM_QkvSplit(benchmark::State& state) {
  const int S = static_cast<int>(state.range(0));

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const int n_head = kNHead;
  const int head_dim = kHeadDim;
  const int n_embd = kNEmbd;

  const int xn = S * 3 * n_embd;
  const int tn = n_head * S * head_dim;
  const auto hx = Fill(xn, 0.5f);

  mgie::DeviceBuffer<float> dx(xn), dq(tn), dk(tn), dv(tn);
  dx.CopyFromHost(hx.data(), xn);

  auto body = [&] {
    mgie::LaunchQkvSplit(dx.data(), dq.data(), dk.data(), dv.data(),
                          S, n_embd, n_head);
  };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 50, 10);
  const double mean_ms = mgie::Mean(times);

  // Bytes: read x[S*3*n_embd] + write q[tn] + k[tn] + v[tn]
  // = 3*S*n_embd*4 (read) + 3*S*n_embd*4 (write) = 6*S*n_embd*4
  const std::size_t bytes = 6ULL * static_cast<std::size_t>(S) * n_embd * sizeof(float);

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = "qkv_split";
  r.input_shape    = "h" + std::to_string(n_head) + "_s" + std::to_string(S) +
                     "_hd" + std::to_string(head_dim);
  r.time_ms        = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops         = 0.0;  // no FLOPs — pure data rearrangement
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, mean_ms);

  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  mb::AppendCsv("qkv_split_results.csv", r, dev);
}

BENCHMARK(BM_QkvSplit)
    ->Arg(1)->Arg(16)->Arg(64)->Arg(256)->Arg(512)->Arg(1024)
    ->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
