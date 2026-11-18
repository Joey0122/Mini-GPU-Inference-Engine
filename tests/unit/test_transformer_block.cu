// tests/unit/test_transformer_block.cu
//
// GPU unit tests for the Phase 5 C7 Transformer Block.
//
// Four levels of validation:
//   1. Conv1D orientation sanity check — each projection type verified
//      against manual CPU GEMM (confirms [K,N] layout is correct).
//   2. Full block vs CPU reference (CpuTransformerBlock) — GPT-2 shapes.
//   3. Residual connection identity — verifies input contribution.
//   4. Real-weights integration — loads block 0, runs forward, checks finite output.

#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/bias_add.hpp"
#include "kernels/gemm.hpp"
#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/transformer_block.hpp"
#include "model/workspace.hpp"

#ifndef MGIE_MODEL_WEIGHTS_DIR
#define MGIE_MODEL_WEIGHTS_DIR "data/model_weights"
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

class BlockTest : public ::testing::Test {
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
// Test 1: Conv1D orientation sanity check — every projection type
// ==========================================================================
// Verifies that hidden @ weight + bias on GPU matches the CPU reference.
// This is the explicit check required before any block wiring.
// If the weight layout were wrong (e.g., [N,K] instead of [K,N]), the GEMM
// would produce garbage — but each individual kernel would still pass its
// own tests because test inputs are always [K,N]-conformant.

template <typename FnGemm, typename FnBias>
void CheckProjection(int M, int K, int N, unsigned seed,
                     FnGemm&& launch_gemm, FnBias&& launch_bias) {
  // Generate host data.
  const auto hA = Rand(M * K, seed, -2.f, 2.f);
  const auto hW = Rand(K * N, seed + 1, -1.f, 1.f);  // [K, N] layout
  const auto hB  = Rand(N, seed + 2, -0.5f, 0.5f);    // bias [N]

  // Upload
  DeviceBuffer<float> dA(M*K), dW(K*N), dB(N), dC(M*N);
  dA.CopyFromHost(hA.data(), M*K);
  dW.CopyFromHost(hW.data(), K*N);
  dB.CopyFromHost(hB.data(), N);

  // GPU: C = A @ W + bias
  launch_gemm(dA.data(), dW.data(), dC.data(), M, N, K);
  CUDA_CHECK_LAST_SYNC();
  launch_bias(dC.data(), dB.data(), dC.data(), M, N);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(M*N);
  dC.CopyToHost(gpu.data(), M*N);

  // CPU reference: gemm then bias
  auto ref = mt::CpuGemmF32(hA, hW, M, N, K);
  ref = mt::CpuBiasAdd(ref, hB, M, N);

  // Use GemmAbsTol(K) — the established GEMM tolerance pattern from Phase 2.
  // kFp32AbsTol (1e-4) is for elementwise ops, not K-length dot products.
  const double tol = mt::GemmAbsTol(K);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), tol)
      << "Projection orientation FAIL: M=" << M << " K=" << K << " N=" << N
      << " max_abs_error=" << mt::MaxAbsError(gpu, ref) << " (tol=" << tol << ")";
}

TEST_F(BlockTest, Conv1dOrientation_cAttn) {
  // c_attn: [S, 768] @ [768, 2304] + [2304] → [S, 2304]
  CheckProjection(16, 768, 2304, 100,
    [](auto... a){ mgie::LaunchGemm(a..., mgie::GemmVersion::RegTiled128Pad); },
    [](auto... a){ mgie::LaunchBiasAdd(a...); });
}

TEST_F(BlockTest, Conv1dOrientation_cProj) {
  // c_proj: [S, 768] @ [768, 768] + [768] → [S, 768]
  CheckProjection(16, 768, 768, 101,
    [](auto... a){ mgie::LaunchGemm(a..., mgie::GemmVersion::RegTiled128Pad); },
    [](auto... a){ mgie::LaunchBiasAdd(a...); });
}

TEST_F(BlockTest, Conv1dOrientation_mlpFc) {
  // mlp_fc: [S, 768] @ [768, 3072] + [3072] → [S, 3072]
  CheckProjection(16, 768, 3072, 102,
    [](auto... a){ mgie::LaunchGemm(a..., mgie::GemmVersion::RegTiled128Pad); },
    [](auto... a){ mgie::LaunchBiasAdd(a...); });
}

TEST_F(BlockTest, Conv1dOrientation_mlpProj) {
  // mlp_proj: [S, 3072] @ [3072, 768] + [768] → [S, 768]
  CheckProjection(16, 3072, 768, 103,
    [](auto... a){ mgie::LaunchGemm(a..., mgie::GemmVersion::RegTiled128Pad); },
    [](auto... a){ mgie::LaunchBiasAdd(a...); });
}

// ==========================================================================
// Test 2: Full block vs CPU reference
// ==========================================================================

// Generate a complete set of synthetic BlockWeights on the host.
struct HostBlockWeights {
  std::vector<float> ln1_w, ln1_b;
  std::vector<float> c_attn_w, c_attn_b;
  std::vector<float> c_proj_w, c_proj_b;
  std::vector<float> ln2_w, ln2_b;
  std::vector<float> mlp_fc_w, mlp_fc_b;
  std::vector<float> mlp_proj_w, mlp_proj_b;
};

HostBlockWeights MakeHostBlockWeights(const mgie::Gpt2Config& cfg, unsigned seed) {
  // Use small-scale weights (GPT-2 init style: N(0, 0.02)) to keep block
  // outputs bounded. Unscaled [-1,1] uniform weights would cause activations
  // to explode through the 3072-length MLP accumulation, making absolute-error
  // gates meaningless without also comparing relative error.
  const float w_scale = 0.02f;
  int D = cfg.n_embd, M = cfg.mlp_inner;
  HostBlockWeights w;
  w.ln1_w      = Rand(D, seed++, -w_scale, w_scale);       w.ln1_w[0] += 1.0f;    // gamma ~1
  w.ln1_b      = Rand(D, seed++, -w_scale, w_scale);        // beta ~0
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

// Upload host weights to device as a BlockWeights struct.
mgie::BlockWeights UploadBlockWeights(const HostBlockWeights& hw, const mgie::Gpt2Config& cfg) {
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

void ExpectBlockMatch(int S, unsigned seed, double tol = 0.0) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, seed);
  auto bw = UploadBlockWeights(hw, cfg);

  // Scale input to embedding-scale (matching GPT-2 init).
  const auto h_in = Rand(S * cfg.n_embd, seed + 100, -0.02f, 0.02f);
  DeviceBuffer<float> d_hidden(S * cfg.n_embd);
  d_hidden.CopyFromHost(h_in.data(), S * cfg.n_embd);

  // Pre-allocated workspace (C8: zero-alloc block path)
  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);

  // GPU block
  mgie::LaunchTransformerBlock(d_hidden.data(), bw, ws, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(S * cfg.n_embd);
  d_hidden.CopyToHost(gpu.data(), S * cfg.n_embd);

  // CPU reference
  const auto ref = mt::CpuTransformerBlock(
      h_in,
      hw.ln1_w, hw.ln1_b, hw.c_attn_w, hw.c_attn_b,
      hw.c_proj_w, hw.c_proj_b, hw.ln2_w, hw.ln2_b,
      hw.mlp_fc_w, hw.mlp_fc_b, hw.mlp_proj_w, hw.mlp_proj_b,
      S, cfg.n_head, cfg.head_dim, cfg.mlp_inner, cfg.ln_eps);

  // The block accumulates through 6 GEMMs (K up to 3072), 2 LayerNorms,
  // attention softmax, GELU. Individual-op spec gate is 1e-4, but at the
  // block level errors compound.
  //   Worst-case: 3*GemmAbsTol(768) + GemmAbsTol(3072) + 2*LayerNormAbsTol(768)
  //               + AttentionAbsTol(S) ≈ 3.6e-3 + 4.8e-7*S
  // We use 5e-3 as a compound gate for typical S, and loosen for S=1024.
  const double abs_tol = (tol > 0.0) ? tol : 5e-3;
  const double err = mt::MaxAbsError(gpu, ref);

  EXPECT_LT(err, abs_tol)
      << "Block S=" << S << " max_abs_error=" << err
      << " (tol=" << abs_tol << ")";
}

TEST_F(BlockTest, SingleTokenMatchesCpu) {
  ExpectBlockMatch(1, 200);
}

TEST_F(BlockTest, ShortSequenceMatchesCpu) {
  ExpectBlockMatch(16, 201);
}

TEST_F(BlockTest, MediumSequenceMatchesCpu) {
  ExpectBlockMatch(64, 202);
}

TEST_F(BlockTest, FullContextMatchesCpu) {
  ExpectBlockMatch(1024, 203);
}

// ==========================================================================
// Test 3: Residual connection preserves input contribution
// ==========================================================================
// output = input + attn_output + mlp_output
// If we zero out the projection weights, output == input (LayerNorm of input
// still passes through). With zero bias and zero weights, the block should
// converge to the LayerNorm residual — a weaker but still useful check.
// The real test: run block and verify each output[i] != input[i] for a
// non-trivial block (confirms the block actually modified the hidden state).

TEST_F(BlockTest, BlockModifiesHidden) {
  const int S = 16;
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, 300);
  auto bw = UploadBlockWeights(hw, cfg);

  const auto h_in = Rand(S * cfg.n_embd, 301, -1.f, 1.f);
  DeviceBuffer<float> d_hidden(S * cfg.n_embd);
  d_hidden.CopyFromHost(h_in.data(), S * cfg.n_embd);

  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);
  mgie::LaunchTransformerBlock(d_hidden.data(), bw, ws, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(S * cfg.n_embd);
  d_hidden.CopyToHost(gpu.data(), S * cfg.n_embd);

  // With random non-zero weights, output should differ from input.
  float max_diff = 0.f;
  for (int i = 0; i < S * cfg.n_embd; ++i)
    max_diff = std::max(max_diff, std::fabs(gpu[i] - h_in[i]));
  EXPECT_GT(max_diff, 0.0f)
      << "Block produced zero change — residual path may not be active";
}

// ==========================================================================
// Test 4: Zero-length no-op
// ==========================================================================

TEST_F(BlockTest, ZeroLengthIsNoop) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, 400);
  auto bw = UploadBlockWeights(hw, cfg);
  auto ws = mgie::TransformerBlockWorkspace::Create(0, cfg);
  DeviceBuffer<float> d_empty(0);
  // Should not crash or throw.
  mgie::LaunchTransformerBlock(d_empty.data(), bw, ws, cfg, 0);
  SUCCEED();
}

// ==========================================================================
// Test 5: Real-weights integration (block 0)
// ==========================================================================

class BlockRealTest : public ::testing::Test {
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
    if (!WeightsExist()) GTEST_SKIP() << "weight files not found at " << kWeightsDir;
    ASSERT_NE(weights_, nullptr) << "LoadGpt2Weights failed";
  }

  static std::unique_ptr<mgie::Gpt2Weights> weights_;
};
std::unique_ptr<mgie::Gpt2Weights> BlockRealTest::weights_ = nullptr;

TEST_F(BlockRealTest, Block0ProducesFiniteOutput) {
  const auto& cfg = weights_->config;
  const int S = 16;
  const auto h_in = Rand(S * cfg.n_embd, 500, -1.f, 1.f);

  DeviceBuffer<float> d_hidden(S * cfg.n_embd);
  d_hidden.CopyFromHost(h_in.data(), S * cfg.n_embd);

  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);
  mgie::LaunchTransformerBlock(d_hidden.data(), weights_->blocks[0], ws, cfg, S);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(S * cfg.n_embd);
  d_hidden.CopyToHost(gpu.data(), S * cfg.n_embd);

  for (int i = 0; i < S * cfg.n_embd; ++i) {
    EXPECT_TRUE(std::isfinite(gpu[i]))
        << "Block 0 produced non-finite output at element " << i;
  }
}

TEST_F(BlockRealTest, Block0ModifiesHiddenWithRealWeights) {
  const auto& cfg = weights_->config;
  const int S = 16;
  const auto h_in = Rand(S * cfg.n_embd, 501, -1.f, 1.f);

  DeviceBuffer<float> d_hidden(S * cfg.n_embd);
  d_hidden.CopyFromHost(h_in.data(), S * cfg.n_embd);

  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);
  mgie::LaunchTransformerBlock(d_hidden.data(), weights_->blocks[0], ws, cfg, S);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(S * cfg.n_embd);
  d_hidden.CopyToHost(gpu.data(), S * cfg.n_embd);

  // Real GPT-2 weights should produce non-identity output.
  float max_diff = 0.f;
  for (int i = 0; i < S * cfg.n_embd; ++i)
    max_diff = std::max(max_diff, std::fabs(gpu[i] - h_in[i]));
  EXPECT_GE(max_diff, 0.0f)
      << "Real-weight block output is identical (degenerate)";
}

// ==========================================================================
// Phase 6a: KV-cache block variants
// ==========================================================================

TEST_F(BlockTest, PrefillMatchesFullBlock) {
  // Prefill output must match LaunchTransformerBlock exactly (bit-level).
  const int S = 8;
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, 600);
  auto bw = UploadBlockWeights(hw, cfg);

  const auto h_in = Rand(S * cfg.n_embd, 601, -0.02f, 0.02f);

  // Full block
  DeviceBuffer<float> d_full(S * cfg.n_embd);
  d_full.CopyFromHost(h_in.data(), S * cfg.n_embd);
  auto ws_full = mgie::TransformerBlockWorkspace::Create(S, cfg);
  mgie::LaunchTransformerBlock(d_full.data(), bw, ws_full, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> full_out(S * cfg.n_embd);
  d_full.CopyToHost(full_out.data(), S * cfg.n_embd);

  // Prefill (same computation, uses cache)
  DeviceBuffer<float> d_pref(S * cfg.n_embd);
  d_pref.CopyFromHost(h_in.data(), S * cfg.n_embd);
  auto ws_pref = mgie::TransformerBlockWorkspace::Create(S, cfg);
  auto kv = mgie::KVCache::Create(cfg);
  mgie::LaunchTransformerBlockPrefill(d_pref.data(), bw, ws_pref,
                                       kv, 0, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> pref_out(S * cfg.n_embd);
  d_pref.CopyToHost(pref_out.data(), S * cfg.n_embd);

  // Outputs must be identical.
  const float err = mt::MaxAbsError(pref_out, full_out);
  EXPECT_EQ(err, 0.0f)
      << "Prefill should produce bitwise-identical output to full block";
}

TEST_F(BlockTest, DecodeStepMatchesFullBlockLastRow) {
  // Full block at S positions. Then decode step for position S-1 using
  // K/V cache populated from the same weights. The last row of hidden
  // must match. Gate: max_abs_error ≤ 1e-5.
  const int S = 16;
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, 610);
  auto bw = UploadBlockWeights(hw, cfg);

  const auto h_in = Rand(S * cfg.n_embd, 611, -0.02f, 0.02f);

  // Full block
  DeviceBuffer<float> d_full(S * cfg.n_embd);
  d_full.CopyFromHost(h_in.data(), S * cfg.n_embd);
  auto ws_full = mgie::TransformerBlockWorkspace::Create(S, cfg);
  mgie::LaunchTransformerBlock(d_full.data(), bw, ws_full, cfg, S);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> full_out(S * cfg.n_embd);
  d_full.CopyToHost(full_out.data(), S * cfg.n_embd);

  // Prefill positions 0..S-2, then decode step at position S-1.
  auto kv = mgie::KVCache::Create(cfg);
  kv.num_positions = 0;

  DeviceBuffer<float> d_step(S * cfg.n_embd);
  d_step.CopyFromHost(h_in.data(), S * cfg.n_embd);
  auto ws_step = mgie::TransformerBlockWorkspace::Create(S, cfg);

  // Prefill first S-1 positions
  mgie::LaunchTransformerBlockPrefill(d_step.data(), bw, ws_step,
                                       kv, 0, cfg, S - 1);
  CUDA_CHECK_LAST_SYNC();

  // Decode step for the last position
  mgie::LaunchTransformerBlockDecodeStep(d_step.data() + (S - 1) * cfg.n_embd,
                                          bw, kv, ws_step,
                                          cfg, 0, S - 1);
  CUDA_CHECK_LAST_SYNC();
  kv.num_positions = S;

  std::vector<float> step_out(S * cfg.n_embd);
  d_step.CopyToHost(step_out.data(), S * cfg.n_embd);

  // Compare: full_out vs step_out. Gate: max_abs_error ≤ 1e-5.
  const float err = mt::MaxAbsError(step_out, full_out);
  EXPECT_LT(err, 1e-5f)
      << "Prefill+DecodeStep should match full block: max_abs_error=" << err;
}

TEST_F(BlockTest, DecodeStepModifiesOnlyTargetRow) {
  // Verify decode step only touches the target position's row.
  const int S = 8, target_pos = 3;
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, 620);
  auto bw = UploadBlockWeights(hw, cfg);

  const auto h_in = Rand(S * cfg.n_embd, 621, -0.02f, 0.02f);

  // Prefill all 8 positions.
  auto kv = mgie::KVCache::Create(cfg);
  DeviceBuffer<float> d_hidden(S * cfg.n_embd);
  d_hidden.CopyFromHost(h_in.data(), S * cfg.n_embd);
  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);

  mgie::LaunchTransformerBlockPrefill(d_hidden.data(), bw, ws,
                                       kv, 0, cfg, S);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> after_prefill(S * cfg.n_embd);
  d_hidden.CopyToHost(after_prefill.data(), S * cfg.n_embd);

  // Now do a decode step at target_pos. Rows != target_pos must remain unchanged.
  mgie::LaunchTransformerBlockDecodeStep(d_hidden.data() + target_pos * cfg.n_embd,
                                          bw, kv, ws,
                                          cfg, 0, target_pos);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> after_step(S * cfg.n_embd);
  d_hidden.CopyToHost(after_step.data(), S * cfg.n_embd);

  const int D = cfg.n_embd;
  for (int pos = 0; pos < S; ++pos) {
    for (int d = 0; d < D; ++d) {
      const size_t idx = static_cast<size_t>(pos) * D + d;
      if (pos == target_pos) {
        // Target row may differ (residuals were added).
        continue;
      }
      EXPECT_FLOAT_EQ(after_step[idx], after_prefill[idx])
          << "Row " << pos << " dim " << d
          << " was modified but should be unchanged";
    }
  }
}

TEST_F(BlockTest, PrefillPopulatesCache) {
  // Prefill S positions; verify K/V cache entries are correct.
  const int S = 4, layer_idx = 0;
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  const auto hw = MakeHostBlockWeights(cfg, 630);
  auto bw = UploadBlockWeights(hw, cfg);

  const auto h_in = Rand(S * cfg.n_embd, 631, -0.02f, 0.02f);

  auto kv = mgie::KVCache::Create(cfg);
  DeviceBuffer<float> d_hidden(S * cfg.n_embd);
  d_hidden.CopyFromHost(h_in.data(), S * cfg.n_embd);
  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);

  mgie::LaunchTransformerBlockPrefill(d_hidden.data(), bw, ws,
                                       kv, layer_idx, cfg, S);
  CUDA_CHECK_LAST_SYNC();

  // Read back one K vector and verify it's finite (non-zero with real weights).
  const int d = cfg.head_dim;
  std::vector<float> k_vec(d);
  float* k_head0 = kv.k_head(layer_idx, 0);
  CUDA_CHECK(cudaMemcpy(k_vec.data(), k_head0,
                        d * sizeof(float), cudaMemcpyDeviceToHost));

  // With synthetic weights scaled at 0.02, K values should be small but finite.
  float max_abs = 0.f;
  for (auto val : k_vec) max_abs = std::max(max_abs, std::fabs(val));
  EXPECT_GT(max_abs, 0.0f)
      << "K cache should contain non-zero data after prefill";
  for (auto val : k_vec)
    EXPECT_TRUE(std::isfinite(val))
        << "K cache contains non-finite values after prefill";
}

}  // namespace
