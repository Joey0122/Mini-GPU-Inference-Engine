// tests/unit/test_forward.cu
//
// GPU unit tests for the Phase 5 C8 Full Forward Pass.
//
// Three validation levels:
//   1. Synthetic CPU reference — S=1, scaled synthetic weights (chain debugging)
//   2. Real-weight golden test — S=4, compare logits against HuggingFace output
//      via file-based reference (scripts/golden_test.py generates the .npy)
//   3. Zero-length no-op

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "model/forward.hpp"
#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"

#include "common/npy_io.hpp"

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

class ForwardTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA"; }
};

std::vector<float> Rand(int n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// ---- Synthetic block helpers (shared with test_transformer_block.cu) ---------

struct HostBlockWeights {
  std::vector<float> ln1_w, ln1_b, c_attn_w, c_attn_b;
  std::vector<float> c_proj_w, c_proj_b, ln2_w, ln2_b;
  std::vector<float> mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b;
};

HostBlockWeights MakeHostBlockWeights(const mgie::Gpt2Config& cfg, unsigned seed) {
  const float w_scale = 0.02f;
  int D = cfg.n_embd, M = cfg.mlp_inner;
  HostBlockWeights w;
  w.ln1_w      = Rand(D, seed++, -w_scale, w_scale);       w.ln1_w[0] += 1.0f;
  w.ln1_b      = Rand(D, seed++, -w_scale, w_scale);
  w.c_attn_w   = Rand(D * 3 * D, seed++, -w_scale, w_scale);
  w.c_attn_b   = Rand(3 * D, seed++, -w_scale, w_scale);
  w.c_proj_w   = Rand(D * D, seed++, -w_scale, w_scale);
  w.c_proj_b   = Rand(D, seed++, -w_scale, w_scale);
  w.ln2_w      = Rand(D, seed++, -w_scale, w_scale);       w.ln2_w[1] += 1.0f;
  w.ln2_b      = Rand(D, seed++, -w_scale, w_scale);
  w.mlp_fc_w   = Rand(D * M, seed++, -w_scale, w_scale);
  w.mlp_fc_b   = Rand(M, seed++, -w_scale, w_scale);
  w.mlp_proj_w = Rand(M * D, seed++, -w_scale, w_scale);
  w.mlp_proj_b = Rand(D, seed++, -w_scale, w_scale);
  return w;
}

mgie::BlockWeights UploadBlockWeights(const HostBlockWeights& hw,
                                       const mgie::Gpt2Config& cfg) {
  int D = cfg.n_embd, M = cfg.mlp_inner;
  mgie::BlockWeights bw;
  bw.ln1_w      = DeviceBuffer<float>(D);       bw.ln1_w.CopyFromHost(hw.ln1_w.data(), D);
  bw.ln1_b      = DeviceBuffer<float>(D);       bw.ln1_b.CopyFromHost(hw.ln1_b.data(), D);
  bw.c_attn_w   = DeviceBuffer<float>(D*3*D);   bw.c_attn_w.CopyFromHost(hw.c_attn_w.data(), D*3*D);
  bw.c_attn_b   = DeviceBuffer<float>(3*D);     bw.c_attn_b.CopyFromHost(hw.c_attn_b.data(), 3*D);
  bw.c_proj_w   = DeviceBuffer<float>(D*D);     bw.c_proj_w.CopyFromHost(hw.c_proj_w.data(), D*D);
  bw.c_proj_b   = DeviceBuffer<float>(D);       bw.c_proj_b.CopyFromHost(hw.c_proj_b.data(), D);
  bw.ln2_w      = DeviceBuffer<float>(D);       bw.ln2_w.CopyFromHost(hw.ln2_w.data(), D);
  bw.ln2_b      = DeviceBuffer<float>(D);       bw.ln2_b.CopyFromHost(hw.ln2_b.data(), D);
  bw.mlp_fc_w   = DeviceBuffer<float>(D*M);     bw.mlp_fc_w.CopyFromHost(hw.mlp_fc_w.data(), D*M);
  bw.mlp_fc_b   = DeviceBuffer<float>(M);       bw.mlp_fc_b.CopyFromHost(hw.mlp_fc_b.data(), M);
  bw.mlp_proj_w = DeviceBuffer<float>(M*D);     bw.mlp_proj_w.CopyFromHost(hw.mlp_proj_w.data(), M*D);
  bw.mlp_proj_b = DeviceBuffer<float>(D);       bw.mlp_proj_b.CopyFromHost(hw.mlp_proj_b.data(), D);
  return bw;
}

// ==========================================================================
// Test 1: Synthetic CPU reference — S=1, 12 matching blocks
// ==========================================================================
// Uses the same convention as C7: all blocks share identical weights for
// compactness. The output logits from the GPU forward pass are diffed against
// CpuGpt2Forward which chains all CPU references.

TEST_F(ForwardTest, SyntheticS1MatchesCpu) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int S = 1, D = cfg.n_embd, V = cfg.vocab_size;
  const int H = cfg.n_head, hd = cfg.head_dim, M = cfg.mlp_inner;
  const float w_scale = 0.02f;

  // ---- Generate synthetic weights (host side) --------------------------------
  const auto wte = Rand(V * D, 1, -w_scale, w_scale);
  const auto wpe = Rand(1024 * D, 2, -w_scale, w_scale);
  const auto wte_T = Rand(D * V, 3, -w_scale, w_scale);
  const auto ln_f_w = Rand(D, 4, -w_scale, w_scale);
  const auto ln_f_b = Rand(D, 5, -w_scale, w_scale);

  // 12 identical blocks (keeps test data compact)
  std::vector<std::vector<float>> blk_ln1_w, blk_ln1_b;
  std::vector<std::vector<float>> blk_c_attn_w, blk_c_attn_b;
  std::vector<std::vector<float>> blk_c_proj_w, blk_c_proj_b;
  std::vector<std::vector<float>> blk_ln2_w, blk_ln2_b;
  std::vector<std::vector<float>> blk_mlp_fc_w, blk_mlp_fc_b;
  std::vector<std::vector<float>> blk_mlp_proj_w, blk_mlp_proj_b;
  for (int i = 0; i < 12; ++i) {
    blk_ln1_w.push_back(Rand(D, 10+i*100, -w_scale, w_scale));
    blk_ln1_w.back()[0] += 1.0f;
    blk_ln1_b.push_back(Rand(D, 11+i*100, -w_scale, w_scale));
    blk_c_attn_w.push_back(Rand(D * 3*D, 12+i*100, -w_scale, w_scale));
    blk_c_attn_b.push_back(Rand(3*D, 13+i*100, -w_scale, w_scale));
    blk_c_proj_w.push_back(Rand(D * D, 14+i*100, -w_scale, w_scale));
    blk_c_proj_b.push_back(Rand(D, 15+i*100, -w_scale, w_scale));
    blk_ln2_w.push_back(Rand(D, 16+i*100, -w_scale, w_scale));
    blk_ln2_w.back()[1] += 1.0f;
    blk_ln2_b.push_back(Rand(D, 17+i*100, -w_scale, w_scale));
    blk_mlp_fc_w.push_back(Rand(D * M, 18+i*100, -w_scale, w_scale));
    blk_mlp_fc_b.push_back(Rand(M, 19+i*100, -w_scale, w_scale));
    blk_mlp_proj_w.push_back(Rand(M * D, 20+i*100, -w_scale, w_scale));
    blk_mlp_proj_b.push_back(Rand(D, 21+i*100, -w_scale, w_scale));
  }

  // ---- CPU reference -------------------------------------------------------
  const std::vector<int> token_ids(S, 0);  // token 0
  const auto ref_logits = mt::CpuGpt2Forward(
      token_ids, wte, wpe,
      blk_ln1_w, blk_ln1_b, blk_c_attn_w, blk_c_attn_b,
      blk_c_proj_w, blk_c_proj_b, blk_ln2_w, blk_ln2_b,
      blk_mlp_fc_w, blk_mlp_fc_b, blk_mlp_proj_w, blk_mlp_proj_b,
      ln_f_w, ln_f_b, wte_T,
      S, H, hd, V, M, cfg.ln_eps);

  // ---- Upload weights and run GPU ------------------------------------------
  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = DeviceBuffer<float>(V*D);   w.wte.CopyFromHost(wte.data(), V*D);
  w.wte_T = DeviceBuffer<float>(D*V);   w.wte_T.CopyFromHost(wte_T.data(), D*V);
  w.wpe   = DeviceBuffer<float>(1024*D); w.wpe.CopyFromHost(wpe.data(), 1024*D);
  w.ln_f_w = DeviceBuffer<float>(D);    w.ln_f_w.CopyFromHost(ln_f_w.data(), D);
  w.ln_f_b = DeviceBuffer<float>(D);    w.ln_f_b.CopyFromHost(ln_f_b.data(), D);

  auto upload_blk = [&](int i) {
    auto& b = w.blocks[i];
    b.ln1_w      = DeviceBuffer<float>(D);     b.ln1_w.CopyFromHost(blk_ln1_w[i].data(), D);
    b.ln1_b      = DeviceBuffer<float>(D);     b.ln1_b.CopyFromHost(blk_ln1_b[i].data(), D);
    b.c_attn_w   = DeviceBuffer<float>(D*3*D); b.c_attn_w.CopyFromHost(blk_c_attn_w[i].data(), D*3*D);
    b.c_attn_b   = DeviceBuffer<float>(3*D);   b.c_attn_b.CopyFromHost(blk_c_attn_b[i].data(), 3*D);
    b.c_proj_w   = DeviceBuffer<float>(D*D);   b.c_proj_w.CopyFromHost(blk_c_proj_w[i].data(), D*D);
    b.c_proj_b   = DeviceBuffer<float>(D);     b.c_proj_b.CopyFromHost(blk_c_proj_b[i].data(), D);
    b.ln2_w      = DeviceBuffer<float>(D);     b.ln2_w.CopyFromHost(blk_ln2_w[i].data(), D);
    b.ln2_b      = DeviceBuffer<float>(D);     b.ln2_b.CopyFromHost(blk_ln2_b[i].data(), D);
    b.mlp_fc_w   = DeviceBuffer<float>(D*M);   b.mlp_fc_w.CopyFromHost(blk_mlp_fc_w[i].data(), D*M);
    b.mlp_fc_b   = DeviceBuffer<float>(M);     b.mlp_fc_b.CopyFromHost(blk_mlp_fc_b[i].data(), M);
    b.mlp_proj_w = DeviceBuffer<float>(M*D);   b.mlp_proj_w.CopyFromHost(blk_mlp_proj_w[i].data(), M*D);
    b.mlp_proj_b = DeviceBuffer<float>(D);     b.mlp_proj_b.CopyFromHost(blk_mlp_proj_b[i].data(), D);
  };
  for (int i = 0; i < 12; ++i) upload_blk(i);

  DeviceBuffer<int> d_ids(S);
  d_ids.CopyFromHost(token_ids.data(), S);
  auto workspace = mgie::Gpt2Workspace::Create(S, cfg);

  mgie::LaunchGpt2Forward(d_ids.data(), w, workspace.logits.data(),
                          workspace, cfg, S);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(S * V);
  workspace.logits.CopyToHost(gpu.data(), S * V);

  // Compound gate: 13 GEMMs (K=768) × GemmAbsTol(768) ≈ 4.8e-3, plus 12
  // LayerNorms and attention. Gate at 1e-2 for synthetic weights.
  const double err = mt::MaxAbsError(gpu, ref_logits);
  EXPECT_LT(err, 1e-2)
      << "Full forward S=1 max_abs_error=" << err;
}

// ==========================================================================
// Test 2: Real-weight golden test — multi-prompt argmax vs HF
// ==========================================================================
// The authoritative correctness gate. Loads real GPT-2 weights, runs forward
// on multiple prompts of varied lengths, saves logits for each, then a Python
// helper compares against HuggingFace.


class ForwardGoldenTest : public ::testing::Test {
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
std::unique_ptr<mgie::Gpt2Weights> ForwardGoldenTest::weights_ = nullptr;

TEST_F(ForwardGoldenTest, MultiPromptGoldenLogits) {
  const auto& cfg = weights_->config;
  const std::string out_dir = std::string(MGIE_GOLDEN_DIR);
  std::filesystem::create_directories(out_dir);

  // Multiple prompts of varied lengths for golden comparison
  struct Prompt { const char* name; std::vector<int> ids; };
  const std::vector<Prompt> prompts = {
      {"singleton",     {42}},
      {"short4",        {0, 1, 2, 3}},
      {"medium16",      {314, 159, 265, 358, 979, 323, 846, 264,
                         338, 327, 950, 288, 419, 716, 939, 937}},
      {"long32",        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                         31, 32}},
  };

  for (const auto& prompt : prompts) {
    const int S = static_cast<int>(prompt.ids.size());
    DeviceBuffer<int> d_ids(S);
    d_ids.CopyFromHost(prompt.ids.data(), S);

    auto workspace = mgie::Gpt2Workspace::Create(S, cfg);
    mgie::LaunchGpt2Forward(d_ids.data(), *weights_,
                            workspace.logits.data(),
                            workspace, cfg, S);
    CUDA_CHECK_LAST_SYNC();

    std::vector<float> gpu(S * cfg.vocab_size);
    workspace.logits.CopyToHost(gpu.data(), S * cfg.vocab_size);

    // Save
    std::ofstream raw(out_dir + "/" + prompt.name + "_logits.bin", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(gpu.data()),
              gpu.size() * sizeof(float));
    raw.close();
    ASSERT_TRUE(raw) << "failed writing golden fixtures";

    std::ofstream tids(out_dir + "/" + prompt.name + "_ids.bin", std::ios::binary);
    tids.write(reinterpret_cast<const char*>(prompt.ids.data()),
               prompt.ids.size() * sizeof(int));
    tids.close();
    ASSERT_TRUE(tids) << "failed writing golden fixtures";

    for (std::size_t i = 0; i < gpu.size(); ++i) {
      ASSERT_TRUE(std::isfinite(gpu[i]))
          << prompt.name << ": non-finite logit at element " << i;
    }

    float max_val = gpu[0], min_val = gpu[0];
    for (auto v : gpu) { max_val = std::max(max_val, v); min_val = std::min(min_val, v); }
    std::cout << "[Golden] " << prompt.name << " S=" << S
              << " logits range=[" << min_val << ", " << max_val << "]"
              << " saved to " << out_dir << std::endl;
  }
}

// ==========================================================================
// Test 3: Zero-length no-op
// ==========================================================================

TEST_F(ForwardTest, ZeroLengthIsNoop) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto workspace = mgie::Gpt2Workspace::Create(0, cfg);
  DeviceBuffer<int> d_empty(0);
  mgie::LaunchGpt2Forward(d_empty.data(), *std::make_unique<mgie::Gpt2Weights>(),
                          workspace.logits.data(), workspace, cfg, 0);
  SUCCEED();
}

// ==========================================================================
// Phase 6a: KV-cache forward variants
// ==========================================================================

TEST_F(ForwardTest, PrefillMatchesFullForward) {
  // LaunchGpt2Prefill must produce identical logits to LaunchGpt2Forward.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int S = 4, D = cfg.n_embd, V = cfg.vocab_size;
  const float w_scale = 0.02f;

  // Generate synthetic weights (compact: all non-zero, scaled).
  const auto wte = Rand(V * D, 50, -w_scale, w_scale);
  const auto wpe = Rand(1024 * D, 51, -w_scale, w_scale);
  const auto wte_T = Rand(D * V, 52, -w_scale, w_scale);
  auto ln_f_w = Rand(D, 53, -w_scale, w_scale); ln_f_w[0] += 1.0f;
  const auto ln_f_b = Rand(D, 54, -w_scale, w_scale);
  std::vector<HostBlockWeights> hblks;
  for (int i = 0; i < 12; ++i)
    hblks.push_back(MakeHostBlockWeights(cfg, 60 + i * 50));

  // Upload
  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = DeviceBuffer<float>(V*D);   w.wte.CopyFromHost(wte.data(), V*D);
  w.wpe   = DeviceBuffer<float>(1024*D); w.wpe.CopyFromHost(wpe.data(), 1024*D);
  w.wte_T = DeviceBuffer<float>(D*V);   w.wte_T.CopyFromHost(wte_T.data(), D*V);
  w.ln_f_w = DeviceBuffer<float>(D);    w.ln_f_w.CopyFromHost(ln_f_w.data(), D);
  w.ln_f_b = DeviceBuffer<float>(D);    w.ln_f_b.CopyFromHost(ln_f_b.data(), D);
  for (int i = 0; i < 12; ++i)
    w.blocks[i] = UploadBlockWeights(hblks[i], cfg);

  const std::vector<int> token_ids = {0, 1, 2, 3};
  DeviceBuffer<int> d_ids(S);
  d_ids.CopyFromHost(token_ids.data(), S);

  // Full forward
  auto ws_full = mgie::Gpt2Workspace::Create(S, cfg);
  mgie::LaunchGpt2Forward(d_ids.data(), w, ws_full.logits.data(),
                          ws_full, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> full_logits(S * V);
  ws_full.logits.CopyToHost(full_logits.data(), S * V);

  // Prefill
  auto ws_pref = mgie::Gpt2Workspace::Create(S, cfg);
  auto kv = mgie::KVCache::Create(cfg);
  mgie::LaunchGpt2Prefill(d_ids.data(), w, ws_pref.logits.data(),
                           ws_pref, kv, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> pref_logits(S * V);
  ws_pref.logits.CopyToHost(pref_logits.data(), S * V);

  // Must match.
  const float err = mt::MaxAbsError(pref_logits, full_logits);
  EXPECT_EQ(err, 0.0f)
      << "Prefill logits must be bitwise-identical to full forward";
  EXPECT_EQ(kv.num_positions, S);
}

TEST_F(ForwardTest, DecodeStepMatchesFullForward) {
  // Prefill S-1 positions, then decode step at position S-1.
  // Logits at position S-1 must match full forward at S. Gate: 1e-5.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int S = 4, D = cfg.n_embd, V = cfg.vocab_size;
  const float w_scale = 0.02f;

  const auto wte = Rand(V * D, 100, -w_scale, w_scale);
  const auto wpe = Rand(1024 * D, 101, -w_scale, w_scale);
  const auto wte_T = Rand(D * V, 102, -w_scale, w_scale);
  auto ln_f_w = Rand(D, 103, -w_scale, w_scale); ln_f_w[0] += 1.0f;
  const auto ln_f_b = Rand(D, 104, -w_scale, w_scale);
  std::vector<HostBlockWeights> hblks;
  for (int i = 0; i < 12; ++i)
    hblks.push_back(MakeHostBlockWeights(cfg, 110 + i * 50));

  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = DeviceBuffer<float>(V*D);   w.wte.CopyFromHost(wte.data(), V*D);
  w.wpe   = DeviceBuffer<float>(1024*D); w.wpe.CopyFromHost(wpe.data(), 1024*D);
  w.wte_T = DeviceBuffer<float>(D*V);   w.wte_T.CopyFromHost(wte_T.data(), D*V);
  w.ln_f_w = DeviceBuffer<float>(D);    w.ln_f_w.CopyFromHost(ln_f_w.data(), D);
  w.ln_f_b = DeviceBuffer<float>(D);    w.ln_f_b.CopyFromHost(ln_f_b.data(), D);
  for (int i = 0; i < 12; ++i)
    w.blocks[i] = UploadBlockWeights(hblks[i], cfg);

  const std::vector<int> all_ids = {0, 1, 2, 3};
  DeviceBuffer<int> d_all(S);
  d_all.CopyFromHost(all_ids.data(), S);

  // Full forward at S=4
  auto ws_full = mgie::Gpt2Workspace::Create(S, cfg);
  mgie::LaunchGpt2Forward(d_all.data(), w, ws_full.logits.data(),
                          ws_full, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> full_logits(S * V);
  ws_full.logits.CopyToHost(full_logits.data(), S * V);

  // Prefill S=3, then decode step for token[3] at position 3
  auto ws = mgie::Gpt2Workspace::Create(S, cfg);
  auto kv = mgie::KVCache::Create(cfg);

  DeviceBuffer<int> d_pref(3);
  d_pref.CopyFromHost(all_ids.data(), 3);
  mgie::LaunchGpt2Prefill(d_pref.data(), w, ws.logits.data(),
                           ws, kv, cfg, 3);
  CUDA_CHECK_LAST_SYNC();

  // Decode step: token 3 at position 3
  DeviceBuffer<float> d_logit_row(V);
  mgie::LaunchGpt2DecodeStep(3, 3, w, d_logit_row.data(),
                              ws, kv, cfg);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> decode_logit_row(V);
  d_logit_row.CopyToHost(decode_logit_row.data(), V);

  // Compare last row of full forward vs decode step output.
  float max_err = 0.f;
  for (int v = 0; v < V; ++v) {
    const float diff = std::fabs(decode_logit_row[v] -
        full_logits[static_cast<std::size_t>(3) * V + v]);
    max_err = std::max(max_err, diff);
  }
  EXPECT_LT(max_err, 1e-5f)
      << "Decode step logits diverge from full forward: max_err=" << max_err;
}

TEST_F(ForwardTest, MultiStepDecodeMatchesFullForward) {
  // Prefill S=1, then 4 decode steps. Compare logits at each step
  // against successive full forward calls. Gate: 1e-5.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const int P = 1, N = 4, D = cfg.n_embd, V = cfg.vocab_size;
  const float w_scale = 0.02f;

  const auto wte = Rand(V * D, 200, -w_scale, w_scale);
  const auto wpe = Rand(1024 * D, 201, -w_scale, w_scale);
  const auto wte_T = Rand(D * V, 202, -w_scale, w_scale);
  auto ln_f_w = Rand(D, 203, -w_scale, w_scale); ln_f_w[0] += 1.0f;
  const auto ln_f_b = Rand(D, 204, -w_scale, w_scale);
  std::vector<HostBlockWeights> hblks;
  for (int i = 0; i < 12; ++i)
    hblks.push_back(MakeHostBlockWeights(cfg, 210 + i * 50));

  mgie::Gpt2Weights w;
  w.config = cfg;
  w.wte   = DeviceBuffer<float>(V*D);   w.wte.CopyFromHost(wte.data(), V*D);
  w.wpe   = DeviceBuffer<float>(1024*D); w.wpe.CopyFromHost(wpe.data(), 1024*D);
  w.wte_T = DeviceBuffer<float>(D*V);   w.wte_T.CopyFromHost(wte_T.data(), D*V);
  w.ln_f_w = DeviceBuffer<float>(D);    w.ln_f_w.CopyFromHost(ln_f_w.data(), D);
  w.ln_f_b = DeviceBuffer<float>(D);    w.ln_f_b.CopyFromHost(ln_f_b.data(), D);
  for (int i = 0; i < 12; ++i)
    w.blocks[i] = UploadBlockWeights(hblks[i], cfg);

  // Token sequence: [0, 1,  2, 3, 4]  (P=1, N=4)
  const std::vector<int> tokens = {0, 1, 2, 3, 4};

  // Prefill + decode
  auto ws_kv = mgie::Gpt2Workspace::Create(P, cfg);
  auto kv = mgie::KVCache::Create(cfg);

  DeviceBuffer<int> d_pref(P);
  d_pref.CopyFromHost(tokens.data(), P);
  mgie::LaunchGpt2Prefill(d_pref.data(), w, ws_kv.logits.data(),
                           ws_kv, kv, cfg, P);
  CUDA_CHECK_LAST_SYNC();

  // Run decode steps and compare against full forward at each step.
  float worst_err = 0.f;
  DeviceBuffer<float> d_step_row(V);

  for (int step = 0; step < N; ++step) {
    const int pos = P + step;
    const int cur_S = pos + 1;

    // KV decode step
    mgie::LaunchGpt2DecodeStep(tokens[pos], pos, w, d_step_row.data(),
                                ws_kv, kv, cfg);
    CUDA_CHECK_LAST_SYNC();
    EXPECT_EQ(kv.num_positions, cur_S);
    std::vector<float> kv_logits(V);
    d_step_row.CopyToHost(kv_logits.data(), V);

    // Full forward at cur_S
    DeviceBuffer<int> d_full(cur_S);
    d_full.CopyFromHost(tokens.data(), cur_S);
    auto ws_full = mgie::Gpt2Workspace::Create(cur_S, cfg);
    mgie::LaunchGpt2Forward(d_full.data(), w, ws_full.logits.data(),
                            ws_full, cfg, cur_S);
    CUDA_CHECK_LAST_SYNC();
    std::vector<float> full_logits(cur_S * V);
    ws_full.logits.CopyToHost(full_logits.data(), cur_S * V);

    // Compare last row
    float step_err = 0.f;
    for (int v = 0; v < V; ++v)
      step_err = std::max(step_err,
          std::fabs(kv_logits[v] - full_logits[static_cast<size_t>(cur_S - 1) * V + v]));
    worst_err = std::max(worst_err, step_err);
  }

  EXPECT_LT(worst_err, 1e-5f)
      << "Multi-step KV decode logits diverge: worst_err=" << worst_err;
}

}  // namespace
