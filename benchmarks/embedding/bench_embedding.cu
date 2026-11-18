// benchmarks/embedding/bench_embedding.cu
//
// Phase 5 C4 benchmark: token + positional embedding gather+add.
//
// Sweeps sequence length S at GPT-2 Small's embedding dimension (n_embd=768).
// The wte gather is a scattered read indexed by token_ids[s]; coalescing is
// poor by construction for this naive kernel. The benchmark teaches the
// latency-vs-S relationship: time should grow roughly linearly with S.

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/timer.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/embedding.hpp"
#include "common/bench_harness.hpp"

namespace {

namespace mb = mgie::bench;

constexpr int kVocabSize = 50257;  // GPT-2 Small
constexpr int kNEmbd    = 768;     // GPT-2 Small

std::vector<int> MakeTokenIds(int S, int vocab_size) {
  std::vector<int> v(S);
  for (int i = 0; i < S; ++i) v[i] = i % vocab_size;
  return v;
}

std::vector<float> Fill(int n, float seed_scale) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = seed_scale * ((i % 13) - 6) / 6.0f;
  return v;
}

void BM_Embedding(benchmark::State& state) {
  const int S = static_cast<int>(state.range(0));

  int dc = 0;
  if (cudaGetDeviceCount(&dc) != cudaSuccess || dc == 0) {
    state.SkipWithError("no CUDA device"); return;
  }

  const int vocab_size = kVocabSize;
  const int n_embd = kNEmbd;
  const auto h_ids = MakeTokenIds(S, vocab_size);
  const auto h_wte = Fill(vocab_size * n_embd, 0.5f);
  const auto h_wpe = Fill(1024 * n_embd, 0.3f);

  mgie::DeviceBuffer<int>   d_ids(S);
  mgie::DeviceBuffer<float> dwte(vocab_size * n_embd);
  mgie::DeviceBuffer<float> dwpe(1024 * n_embd);
  mgie::DeviceBuffer<float> dout(S * n_embd);
  d_ids.CopyFromHost(h_ids.data(), S);
  dwte.CopyFromHost(h_wte.data(), vocab_size * n_embd);
  dwpe.CopyFromHost(h_wpe.data(), 1024 * n_embd);

  auto body = [&] {
    mgie::LaunchEmbedding(d_ids.data(), dwte.data(), dwpe.data(), dout.data(),
                           S, n_embd, vocab_size);
  };

  for (auto _ : state) {
    const auto times = mgie::TimeKernel(body, 20, 5);
    state.SetIterationTime(mgie::Mean(times) * 1e-3);
  }

  const auto times = mgie::TimeKernel(body, 50, 10);
  const double mean_ms = mgie::Mean(times);

  // Bandwidth model: read wte[S*n_embd] (gather, scattered), read wpe[S*n_embd]
  // (coalesced), write out[S*n_embd] (coalesced). No FLOPS besides the add.
  const std::size_t bytes = 3ULL * static_cast<std::size_t>(S) * n_embd * sizeof(float);
  const std::size_t flops = 1ULL * static_cast<std::size_t>(S) * n_embd;  // one add per element

  const mgie::DeviceInfo dev = mgie::QueryDeviceInfo(0);
  mb::BenchResult r;
  r.kernel_name    = "embedding";
  r.input_shape    = "s" + std::to_string(S) + "_d" + std::to_string(n_embd);
  r.time_ms        = mean_ms;
  r.time_stddev_ms = mgie::StdDev(times);
  r.gflops         = mb::GFlops(flops, mean_ms);
  r.bandwidth_gbs  = mb::BandwidthGBs(bytes, mean_ms);

  state.counters["GFLOPS"] = r.gflops;
  state.counters["bandwidth_GBs"] = r.bandwidth_gbs;
  mb::AppendCsv("embedding_results.csv", r, dev);
}

BENCHMARK(BM_Embedding)
    ->Arg(1)->Arg(16)->Arg(64)->Arg(256)->Arg(1024)
    ->UseManualTime();

}  // namespace

BENCHMARK_MAIN();
