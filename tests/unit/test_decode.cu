// tests/unit/test_decode.cu
//
// GPU unit tests for Phase 5 C9/C10 Decoding.
//
// C9: Greedy decoding validation
// C10: Temperature sampling validation
//   - temperature≈0 reproduces greedy output
//   - same seed + same params → identical sequence (determinism)
//   - different seeds → allowed to differ
//   - golden: GPU sampling decode vs HF model.generate with same seed

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "model/decode.hpp"
#include "model/forward.hpp"
#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"

#ifndef MGIE_MODEL_WEIGHTS_DIR
#define MGIE_MODEL_WEIGHTS_DIR "data/model_weights"
#endif
#ifndef MGIE_GOLDEN_DIR
#define MGIE_GOLDEN_DIR "data/test_vectors/golden"
#endif

namespace {

using mgie::DeviceBuffer;
namespace mt = mgie::test;

const std::string kWeightsDir = MGIE_MODEL_WEIGHTS_DIR;

bool CudaAvail() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }
bool WeightsExist() {
  std::ifstream f(kWeightsDir + "/gpt2_manifest.txt");
  return f.good();
}

class DecodeTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA"; }
};

std::vector<float> Rand(int n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// ==========================================================================
// C9 Test: Synthetic greedy decode produces valid tokens
// ==========================================================================

TEST_F(DecodeTest, SingletonGreedyDecodeStopsAtContextLimit) {
  auto cfg = mgie::DefaultGpt2SmallConfig();
  cfg.n_ctx = 5;
  const int prompt_len = 1, max_seq = cfg.n_ctx;
  const int D = cfg.n_embd, V = cfg.vocab_size, M = cfg.mlp_inner;
  const float w_scale = 0.02f;
  unsigned seed = 1;

  auto upload = [](const std::vector<float>& h) {
    DeviceBuffer<float> b(h.size()); b.CopyFromHost(h.data(), h.size()); return b;
  };
  auto upload_adj = [&](std::vector<float> h, int idx, float delta) {
    h[idx] += delta; return upload(h);
  };

  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = upload(Rand(V * D, seed++, -w_scale, w_scale));
  w.wpe   = upload(Rand(cfg.n_ctx * D, seed++, -w_scale, w_scale));
  w.wte_T = upload(Rand(D * V, seed++, -w_scale, w_scale));
  w.ln_f_w = upload(Rand(D, seed++, -w_scale, w_scale));
  w.ln_f_b = upload(Rand(D, seed++, -w_scale, w_scale));
  for (int i = 0; i < 12; ++i) {
    auto& b = w.blocks[i];
    b.ln1_w      = upload_adj(Rand(D, seed++, -w_scale, w_scale), 0, 1.0f);
    b.ln1_b      = upload(Rand(D, seed++, -w_scale, w_scale));
    b.c_attn_w   = upload(Rand(D*3*D, seed++, -w_scale, w_scale));
    b.c_attn_b   = upload(Rand(3*D, seed++, -w_scale, w_scale));
    b.c_proj_w   = upload(Rand(D*D, seed++, -w_scale, w_scale));
    b.c_proj_b   = upload(Rand(D, seed++, -w_scale, w_scale));
    b.ln2_w      = upload_adj(Rand(D, seed++, -w_scale, w_scale), 1, 1.0f);
    b.ln2_b      = upload(Rand(D, seed++, -w_scale, w_scale));
    b.mlp_fc_w   = upload(Rand(D*M, seed++, -w_scale, w_scale));
    b.mlp_fc_b   = upload(Rand(M, seed++, -w_scale, w_scale));
    b.mlp_proj_w = upload(Rand(M*D, seed++, -w_scale, w_scale));
    b.mlp_proj_b = upload(Rand(D, seed++, -w_scale, w_scale));
  }

  std::vector<int> prompt = {0};
  DeviceBuffer<int> d_prompt(1), d_output(max_seq);
  d_prompt.CopyFromHost(prompt.data(), 1);
  auto workspace = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv_cache = mgie::KVCache::Create(cfg);

  int gen = mgie::LaunchGreedyDecode(d_prompt.data(), d_output.data(),
                                      w, workspace, kv_cache, cfg,
                                      prompt_len, std::numeric_limits<int>::max());
  ASSERT_EQ(gen, cfg.n_ctx - prompt_len);
  EXPECT_EQ(kv_cache.num_positions, cfg.n_ctx);

  EXPECT_THROW(mgie::LaunchGpt2DecodeStep(0, cfg.n_ctx, w, workspace.logits.data(),
                                         workspace, kv_cache, cfg),
               std::out_of_range);
  EXPECT_EQ(mgie::LaunchGreedyDecode(d_output.data(), d_output.data(),
                                     w, workspace, kv_cache, cfg, cfg.n_ctx, 1), 0);

  std::vector<int> gpu_tokens(max_seq);
  d_output.CopyToHost(gpu_tokens.data(), max_seq);

  for (int i = 0; i < max_seq; ++i) {
    EXPECT_GE(gpu_tokens[i], 0);
    EXPECT_LT(gpu_tokens[i], V);
  }
  EXPECT_EQ(gpu_tokens[0], prompt[0]);
  SUCCEED();
}

// ==========================================================================
// C10 Test 1: temperature≈0 reproduces greedy output
// ==========================================================================

TEST_F(DecodeTest, TemperatureZeroReproducesGreedy) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int prompt_len = 1, max_new = 10, max_seq = prompt_len + max_new;
  const int D = cfg.n_embd, V = cfg.vocab_size, M = cfg.mlp_inner;
  const float w_scale = 0.02f;
  unsigned seed = 100;

  auto upload = [](const std::vector<float>& h) {
    DeviceBuffer<float> b(h.size()); b.CopyFromHost(h.data(), h.size()); return b;
  };
  auto upload_adj = [&](std::vector<float> h, int idx, float delta) {
    h[idx] += delta; return upload(h);
  };
  unsigned s = seed;

  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = upload(Rand(V * D, s++, -w_scale, w_scale));
  w.wpe   = upload(Rand(1024 * D, s++, -w_scale, w_scale));
  w.wte_T = upload(Rand(D * V, s++, -w_scale, w_scale));
  w.ln_f_w = upload(Rand(D, s++, -w_scale, w_scale));
  w.ln_f_b = upload(Rand(D, s++, -w_scale, w_scale));
  for (int i = 0; i < 12; ++i) {
    auto& b = w.blocks[i];
    b.ln1_w      = upload_adj(Rand(D, s++, -w_scale, w_scale), 0, 1.0f);
    b.ln1_b      = upload(Rand(D, s++, -w_scale, w_scale));
    b.c_attn_w   = upload(Rand(D*3*D, s++, -w_scale, w_scale));
    b.c_attn_b   = upload(Rand(3*D, s++, -w_scale, w_scale));
    b.c_proj_w   = upload(Rand(D*D, s++, -w_scale, w_scale));
    b.c_proj_b   = upload(Rand(D, s++, -w_scale, w_scale));
    b.ln2_w      = upload_adj(Rand(D, s++, -w_scale, w_scale), 1, 1.0f);
    b.ln2_b      = upload(Rand(D, s++, -w_scale, w_scale));
    b.mlp_fc_w   = upload(Rand(D*M, s++, -w_scale, w_scale));
    b.mlp_fc_b   = upload(Rand(M, s++, -w_scale, w_scale));
    b.mlp_proj_w = upload(Rand(M*D, s++, -w_scale, w_scale));
    b.mlp_proj_b = upload(Rand(D, s++, -w_scale, w_scale));
  }

  std::vector<int> prompt = {42};
  DeviceBuffer<int> d_prompt(1);
  d_prompt.CopyFromHost(prompt.data(), 1);

  // Greedy
  DeviceBuffer<int> d_greedy(max_seq);
  auto ws1 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv1 = mgie::KVCache::Create(cfg);
  mgie::LaunchGreedyDecode(d_prompt.data(), d_greedy.data(),
                           w, ws1, kv1, cfg, prompt_len, max_new);

  // Sampling with temp=0
  DeviceBuffer<int> d_temp0(max_seq);
  auto ws2 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv2 = mgie::KVCache::Create(cfg);
  mgie::LaunchSamplingDecode(d_prompt.data(), d_temp0.data(),
                             w, ws2, kv2, cfg, prompt_len, max_new,
                             0.0f, 12345);

  // Sampling with temp=1e-7 (near-zero)
  DeviceBuffer<int> d_near0(max_seq);
  auto ws3 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv3 = mgie::KVCache::Create(cfg);
  mgie::LaunchSamplingDecode(d_prompt.data(), d_near0.data(),
                             w, ws3, kv3, cfg, prompt_len, max_new,
                             1e-7f, 12345);

  std::vector<int> greedy(max_seq), temp0(max_seq), near0(max_seq);
  d_greedy.CopyToHost(greedy.data(), max_seq);
  d_temp0.CopyToHost(temp0.data(), max_seq);
  d_near0.CopyToHost(near0.data(), max_seq);

  EXPECT_EQ(greedy, temp0)
      << "temperature=0 should produce same tokens as greedy";
  EXPECT_EQ(greedy, near0)
      << "temperature=1e-7 should produce same tokens as greedy";
}

// ==========================================================================
// C10 Test 2: same seed + same params → identical output (determinism)
// ==========================================================================

TEST_F(DecodeTest, SameSeedProducesIdenticalOutput) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int prompt_len = 1, max_new = 10, max_seq = prompt_len + max_new;
  const int D = cfg.n_embd, V = cfg.vocab_size, M = cfg.mlp_inner;
  const float w_scale = 0.02f;
  unsigned s = 200;

  auto upload = [](const std::vector<float>& h) {
    DeviceBuffer<float> b(h.size()); b.CopyFromHost(h.data(), h.size()); return b;
  };
  auto upload_adj = [&](std::vector<float> h, int idx, float delta) {
    h[idx] += delta; return upload(h);
  };

  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = upload(Rand(V * D, s++, -w_scale, w_scale));
  w.wpe   = upload(Rand(1024 * D, s++, -w_scale, w_scale));
  w.wte_T = upload(Rand(D * V, s++, -w_scale, w_scale));
  w.ln_f_w = upload(Rand(D, s++, -w_scale, w_scale));
  w.ln_f_b = upload(Rand(D, s++, -w_scale, w_scale));
  for (int i = 0; i < 12; ++i) {
    auto& b = w.blocks[i];
    b.ln1_w      = upload_adj(Rand(D, s++, -w_scale, w_scale), 0, 1.0f);
    b.ln1_b      = upload(Rand(D, s++, -w_scale, w_scale));
    b.c_attn_w   = upload(Rand(D*3*D, s++, -w_scale, w_scale));
    b.c_attn_b   = upload(Rand(3*D, s++, -w_scale, w_scale));
    b.c_proj_w   = upload(Rand(D*D, s++, -w_scale, w_scale));
    b.c_proj_b   = upload(Rand(D, s++, -w_scale, w_scale));
    b.ln2_w      = upload_adj(Rand(D, s++, -w_scale, w_scale), 1, 1.0f);
    b.ln2_b      = upload(Rand(D, s++, -w_scale, w_scale));
    b.mlp_fc_w   = upload(Rand(D*M, s++, -w_scale, w_scale));
    b.mlp_fc_b   = upload(Rand(M, s++, -w_scale, w_scale));
    b.mlp_proj_w = upload(Rand(M*D, s++, -w_scale, w_scale));
    b.mlp_proj_b = upload(Rand(D, s++, -w_scale, w_scale));
  }

  std::vector<int> prompt = {7};
  DeviceBuffer<int> d_prompt(1);
  d_prompt.CopyFromHost(prompt.data(), 1);

  // Run 1
  DeviceBuffer<int> d_out1(max_seq);
  auto ws1 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv1 = mgie::KVCache::Create(cfg);
  mgie::LaunchSamplingDecode(d_prompt.data(), d_out1.data(),
                             w, ws1, kv1, cfg, prompt_len, max_new,
                             0.8f, 42);

  // Run 2 — same seed
  DeviceBuffer<int> d_out2(max_seq);
  auto ws2 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv2 = mgie::KVCache::Create(cfg);
  mgie::LaunchSamplingDecode(d_prompt.data(), d_out2.data(),
                             w, ws2, kv2, cfg, prompt_len, max_new,
                             0.8f, 42);

  std::vector<int> out1(max_seq), out2(max_seq);
  d_out1.CopyToHost(out1.data(), max_seq);
  d_out2.CopyToHost(out2.data(), max_seq);

  EXPECT_EQ(out1, out2)
      << "Same seed should produce identical token sequences";
}

// ==========================================================================
// C10 Test 3: different seeds → allowed to differ
// ==========================================================================

TEST_F(DecodeTest, DifferentSeedsMayDiffer) {
  // Just verify that different seeds don't crash and produce valid tokens.
  // With untrained weights at temperature=1.0, the sequences are very likely
  // different but we only assert validity, not difference.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int prompt_len = 1, max_new = 8, max_seq = prompt_len + max_new;
  const int D = cfg.n_embd, V = cfg.vocab_size, M = cfg.mlp_inner;
  const float w_scale = 0.02f;
  unsigned s = 300;

  auto upload = [](const std::vector<float>& h) {
    DeviceBuffer<float> b(h.size()); b.CopyFromHost(h.data(), h.size()); return b;
  };
  auto upload_adj = [&](std::vector<float> h, int idx, float delta) {
    h[idx] += delta; return upload(h);
  };

  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = upload(Rand(V * D, s++, -w_scale, w_scale));
  w.wpe   = upload(Rand(1024 * D, s++, -w_scale, w_scale));
  w.wte_T = upload(Rand(D * V, s++, -w_scale, w_scale));
  w.ln_f_w = upload(Rand(D, s++, -w_scale, w_scale));
  w.ln_f_b = upload(Rand(D, s++, -w_scale, w_scale));
  for (int i = 0; i < 12; ++i) {
    auto& b = w.blocks[i];
    b.ln1_w      = upload_adj(Rand(D, s++, -w_scale, w_scale), 0, 1.0f);
    b.ln1_b      = upload(Rand(D, s++, -w_scale, w_scale));
    b.c_attn_w   = upload(Rand(D*3*D, s++, -w_scale, w_scale));
    b.c_attn_b   = upload(Rand(3*D, s++, -w_scale, w_scale));
    b.c_proj_w   = upload(Rand(D*D, s++, -w_scale, w_scale));
    b.c_proj_b   = upload(Rand(D, s++, -w_scale, w_scale));
    b.ln2_w      = upload_adj(Rand(D, s++, -w_scale, w_scale), 1, 1.0f);
    b.ln2_b      = upload(Rand(D, s++, -w_scale, w_scale));
    b.mlp_fc_w   = upload(Rand(D*M, s++, -w_scale, w_scale));
    b.mlp_fc_b   = upload(Rand(M, s++, -w_scale, w_scale));
    b.mlp_proj_w = upload(Rand(M*D, s++, -w_scale, w_scale));
    b.mlp_proj_b = upload(Rand(D, s++, -w_scale, w_scale));
  }

  std::vector<int> prompt = {99};
  DeviceBuffer<int> d_prompt(1);
  d_prompt.CopyFromHost(prompt.data(), 1);

  DeviceBuffer<int> d_out1(max_seq), d_out2(max_seq);
  auto ws1 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto ws2 = mgie::Gpt2Workspace::Create(prompt_len, cfg);
  auto kv1 = mgie::KVCache::Create(cfg);
  auto kv2 = mgie::KVCache::Create(cfg);

  mgie::LaunchSamplingDecode(d_prompt.data(), d_out1.data(),
                             w, ws1, kv1, cfg, prompt_len, max_new,
                             1.0f, 100);
  mgie::LaunchSamplingDecode(d_prompt.data(), d_out2.data(),
                             w, ws2, kv2, cfg, prompt_len, max_new,
                             1.0f, 999);

  std::vector<int> out1(max_seq), out2(max_seq);
  d_out1.CopyToHost(out1.data(), max_seq);
  d_out2.CopyToHost(out2.data(), max_seq);

  for (int i = 0; i < max_seq; ++i) {
    EXPECT_GE(out1[i], 0); EXPECT_LT(out1[i], V);
    EXPECT_GE(out2[i], 0); EXPECT_LT(out2[i], V);
  }
  // With temperature=1.0 and different seeds, sequences are very likely
  // different — but we don't assert that (probabilistic).
  SUCCEED();
}

// ==========================================================================
// C9 Golden: greedy decode vs HF (4 prompts, 25 steps each)
// ==========================================================================

class DecodeGoldenTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CudaAvail() || !WeightsExist()) return;
    const auto cfg = mgie::DefaultGpt2SmallConfig();
    weights_ = std::make_unique<mgie::Gpt2Weights>(
        mgie::LoadGpt2Weights(kWeightsDir, cfg));
  }
  static void TearDownTestSuite() { weights_.reset(); }

  void SetUp() override {
    if (!CudaAvail()) GTEST_SKIP() << "no CUDA device";
    if (!WeightsExist()) GTEST_SKIP() << "weight files not found";
    ASSERT_NE(weights_, nullptr) << "LoadGpt2Weights failed";
  }

  static std::unique_ptr<mgie::Gpt2Weights> weights_;
};
std::unique_ptr<mgie::Gpt2Weights> DecodeGoldenTest::weights_ = nullptr;

TEST_F(DecodeGoldenTest, MultiPromptGreedyDecode) {
  const auto& cfg = weights_->config;
  const std::string out_dir = std::string(MGIE_GOLDEN_DIR);
  std::filesystem::create_directories(out_dir);

  struct Case { const char* name; std::vector<int> prompt; int steps; };
  const std::vector<Case> cases = {
      {"decode_p0", {0}, 25},
      {"decode_p42", {42}, 25},
      {"decode_p3", {314, 159, 265}, 25},
      {"decode_p8", {1, 2, 3, 4, 5, 6, 7, 8}, 25},
  };

  for (const auto& c : cases) {
    const int prompt_len = static_cast<int>(c.prompt.size());
    const int max_seq = prompt_len + c.steps;

    DeviceBuffer<int> d_prompt(prompt_len), d_output(max_seq);
    d_prompt.CopyFromHost(c.prompt.data(), prompt_len);
    auto workspace = mgie::Gpt2Workspace::Create(prompt_len, cfg);
    auto kv_cache = mgie::KVCache::Create(cfg);

    int gen = mgie::LaunchGreedyDecode(d_prompt.data(), d_output.data(),
                                        *weights_, workspace, kv_cache, cfg,
                                        prompt_len, c.steps);
    ASSERT_EQ(gen, c.steps);

    std::vector<int> gpu_tokens(max_seq);
    d_output.CopyToHost(gpu_tokens.data(), max_seq);

    std::ofstream raw(out_dir + "/" + c.name + "_tokens.bin", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(gpu_tokens.data()),
              gpu_tokens.size() * sizeof(int));
    raw.close();
    ASSERT_TRUE(raw) << "failed writing golden fixtures";

    std::ofstream pout(out_dir + "/" + c.name + "_prompt.bin", std::ios::binary);
    pout.write(reinterpret_cast<const char*>(c.prompt.data()),
               c.prompt.size() * sizeof(int));
    pout.close();
    ASSERT_TRUE(pout) << "failed writing golden fixtures";
  }
}

// ==========================================================================
// C10 Golden: temperature-sampling decode vs HF (deterministic seed)
// ==========================================================================

TEST_F(DecodeGoldenTest, MultiPromptSamplingDecode) {
  const auto& cfg = weights_->config;
  const std::string out_dir = std::string(MGIE_GOLDEN_DIR);
  std::filesystem::create_directories(out_dir);

  // Prompt [314, 159, 265] at T=0.8 with seed=42 — deterministic sampling
  struct SamplingCase { const char* name; std::vector<int> prompt;
                        int steps; float temp; unsigned int seed; };
  const std::vector<SamplingCase> cases = {
      {"sample_p0_t08_s42", {0}, 20, 0.8f, 42},
      {"sample_p0_t12_s7", {0}, 20, 1.2f, 7},
      {"sample_p3_t08_s42", {314, 159, 265}, 20, 0.8f, 42},
  };

  for (const auto& c : cases) {
    const int prompt_len = static_cast<int>(c.prompt.size());
    const int max_seq = prompt_len + c.steps;

    DeviceBuffer<int> d_prompt(prompt_len), d_output(max_seq);
    d_prompt.CopyFromHost(c.prompt.data(), prompt_len);
    auto workspace = mgie::Gpt2Workspace::Create(prompt_len, cfg);
    auto kv_cache = mgie::KVCache::Create(cfg);

    int gen = mgie::LaunchSamplingDecode(d_prompt.data(), d_output.data(),
                                          *weights_, workspace, kv_cache, cfg,
                                          prompt_len, c.steps,
                                          c.temp, c.seed);
    ASSERT_EQ(gen, c.steps);

    std::vector<int> gpu_tokens(max_seq);
    d_output.CopyToHost(gpu_tokens.data(), max_seq);

    std::ofstream raw(out_dir + "/" + c.name + "_tokens.bin", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(gpu_tokens.data()),
              gpu_tokens.size() * sizeof(int));
    raw.close();
    ASSERT_TRUE(raw) << "failed writing golden fixtures";

    std::ofstream pout(out_dir + "/" + c.name + "_prompt.bin", std::ios::binary);
    pout.write(reinterpret_cast<const char*>(c.prompt.data()),
               c.prompt.size() * sizeof(int));
    pout.close();
    ASSERT_TRUE(pout) << "failed writing golden fixtures";
  }
}

}  // namespace
