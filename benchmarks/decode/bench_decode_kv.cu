// benchmarks/decode/bench_decode_kv.cu
//
// Phase 6a: KV-cache decode benchmark.
// Measures prefill latency (full forward + cache population) and per-step
// decode latency for GPT-2 Small across varying prompt lengths and cache sizes.

#include <benchmark/benchmark.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common/errors.hpp"
#include "cuda/memory.hpp"
#include "model/decode.hpp"
#include "model/gpt2_config.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"
#include "model/kv_cache.hpp"

#ifndef MGIE_MODEL_WEIGHTS_DIR
#define MGIE_MODEL_WEIGHTS_DIR "data/model_weights"
#endif

namespace {

using mgie::DeviceBuffer;

bool WeightsExist() {
  std::ifstream f(std::string(MGIE_MODEL_WEIGHTS_DIR) + "/gpt2_manifest.txt");
  return f.good();
}

// Lazy-loaded weights — loaded once on first access.
struct LazyWeights {
  std::unique_ptr<mgie::Gpt2Weights> w;
  mgie::Gpt2Config cfg;

  void ensure() {
    if (!w && WeightsExist()) {
      cfg = mgie::DefaultGpt2SmallConfig();
      w.reset(new mgie::Gpt2Weights(
          mgie::LoadGpt2Weights(MGIE_MODEL_WEIGHTS_DIR, cfg)));
    }
  }
};

LazyWeights g_w;

void BM_DecodePrefill(benchmark::State& state) {
  g_w.ensure();
  if (!g_w.w) { state.SkipWithError("weights not found"); return; }

  const int S = state.range(0);
  auto ws = mgie::Gpt2Workspace::Create(S, g_w.cfg);
  auto kv = mgie::KVCache::Create(g_w.cfg);
  DeviceBuffer<int> d_ids(S);

  std::vector<int> ids(S, 0);
  d_ids.CopyFromHost(ids.data(), S);

  for (auto _ : state) {
    kv.num_positions = 0;
    mgie::LaunchGpt2Prefill(d_ids.data(), *g_w.w,
                             ws.logits.data(), ws, kv,
                             g_w.cfg, S);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  state.SetItemsProcessed(state.iterations() * S);
}

void BM_DecodeStep(benchmark::State& state) {
  g_w.ensure();
  if (!g_w.w) { state.SkipWithError("weights not found"); return; }

  const int S_cached = state.range(0);
  auto ws = mgie::Gpt2Workspace::Create(S_cached, g_w.cfg);
  auto kv = mgie::KVCache::Create(g_w.cfg);

  // Prefill S_cached positions.
  std::vector<int> ids(S_cached, 0);
  DeviceBuffer<int> d_ids(S_cached);
  d_ids.CopyFromHost(ids.data(), S_cached);
  mgie::LaunchGpt2Prefill(d_ids.data(), *g_w.w,
                           ws.logits.data(), ws, kv,
                           g_w.cfg, S_cached);
  CUDA_CHECK(cudaDeviceSynchronize());

  DeviceBuffer<float> d_row(g_w.cfg.vocab_size);

  for (auto _ : state) {
    mgie::LaunchGpt2DecodeStep(0, S_cached, *g_w.w,
                                d_row.data(), ws, kv,
                                g_w.cfg);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  state.SetItemsProcessed(state.iterations());
}

// Prefill: vary S
BENCHMARK(BM_DecodePrefill)
    ->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMillisecond);

// Decode step: vary cache size (max n_ctx-1 = 1023, as decode step at pos n_ctx
// would exceed wpe bounds)
BENCHMARK(BM_DecodeStep)
    ->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(512)->Arg(1023)
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
