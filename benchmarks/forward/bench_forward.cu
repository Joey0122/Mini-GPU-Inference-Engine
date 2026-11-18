// benchmarks/forward/bench_forward.cu
//
// Phase 5 C8 benchmark: full GPT-2 forward pass latency.
//
// Measures wall-clock time for the complete inference pipeline:
//   embedding → 12 blocks → ln_f → lm_head
// Uses synthetic weights (not real GPT-2) to avoid 622 MB upload overhead
// in the benchmark loop. Latency is measured via CUDA events.
//
// Sweeps S at GPT-2 Small config; reports total latency (ms).

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "common/bench_harness.hpp"
#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "model/forward.hpp"
#include "model/gpt2_config.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"

namespace {

namespace mb = mgie::bench;

std::vector<float> Fill(int n, float seed_scale) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = seed_scale * ((i % 13) - 6) / 6.0f;
  return v;
}

std::vector<int> MakeTokenIds(int S) {
  std::vector<int> v(S);
  for (int i = 0; i < S; ++i) v[i] = i % 50257;
  return v;
}

mgie::Gpt2Weights MakeFakeWeights(const mgie::Gpt2Config& cfg) {
  const int D = cfg.n_embd, V = cfg.vocab_size, M = cfg.mlp_inner;
  auto w = Fill(D, 0.5f);  // reusable 1D filler

  auto make_buf = [](int n) {
    auto h = Fill(n, 0.5f);
    mgie::DeviceBuffer<float> buf(n);
    buf.CopyFromHost(h.data(), n);
    return buf;
  };

  mgie::Gpt2Weights gpt2w;
  gpt2w.config = cfg;
  gpt2w.wte   = make_buf(V * D);
  gpt2w.wte_T = make_buf(D * V);
  gpt2w.wpe   = make_buf(1024 * D);
  gpt2w.ln_f_w = make_buf(D);
  gpt2w.ln_f_b = make_buf(D);

  for (int i = 0; i < 12; ++i) {
    auto& b = gpt2w.blocks[i];
    b.ln1_w = make_buf(D);      b.ln1_b = make_buf(D);
    b.c_attn_w = make_buf(D * 3 * D);  b.c_attn_b = make_buf(3 * D);
    b.c_proj_w = make_buf(D * D);      b.c_proj_b = make_buf(D);
    b.ln2_w = make_buf(D);      b.ln2_b = make_buf(D);
    b.mlp_fc_w = make_buf(D * M);      b.mlp_fc_b = make_buf(M);
    b.mlp_proj_w = make_buf(M * D);    b.mlp_proj_b = make_buf(D);
  }
  return gpt2w;
}

void BM_Gpt2Forward(benchmark::State& state) {
  const int S = static_cast<int>(state.range(0));

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto weights = MakeFakeWeights(cfg);
  auto h_ids = MakeTokenIds(S);

  mgie::DeviceBuffer<int> d_ids(S);
  d_ids.CopyFromHost(h_ids.data(), S);

  auto workspace = mgie::Gpt2Workspace::Create(S, cfg);

  auto body = [&] {
    mgie::LaunchGpt2Forward(d_ids.data(), weights,
                            workspace.logits.data(),
                            workspace, cfg, S);
  };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 5, 3);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 20, 5);
  const double mean_ms = mgie::Mean(times);

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = "gpt2_forward";
  r.input_shape    = "s" + std::to_string(S);
  r.time_ms        = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops         = 0.0;   // composite op — meaningful GFLOPS needs per-op accounting
  r.bandwidth_gbs  = 0.0;

  state.counters["latency_ms"] = mean_ms;
  mb::AppendCsv("gpt2_forward_results.csv", r, dev);
}

BENCHMARK(BM_Gpt2Forward)
    ->Arg(1)->Arg(16)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
