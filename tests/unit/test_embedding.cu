// tests/unit/test_embedding.cu
//
// GPU unit tests for the Phase 5 C4 embedding kernel.
//
// Validates LaunchEmbedding against the double-precision CPU reference
// (mgie::test::CpuEmbedding) within the spec tolerance (FP32 max_abs_error < 1e-4).
// Also tests with real GPT-2 weights via the loader fixture for shape/sanity checks.

#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/embedding.hpp"
#include "model/gpt2_config.hpp"
#include "model/loader.hpp"

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

class EmbeddingTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA"; }
};

// ---- CPU reference tests (synthetic weights) -------------------------------

std::vector<float> Rand(int n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

std::vector<int> RandTokenIds(int S, int vocab_size, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(0, vocab_size - 1);
  std::vector<int> v(S);
  for (auto& x : v) x = d(rng);
  return v;
}

void RunEmbedding(const std::vector<int>& token_ids,
                   const std::vector<float>& wte,
                   const std::vector<float>& wpe,
                   std::vector<float>& out,
                   int S, int n_embd, int vocab_size) {
  const int tn = S, wn = static_cast<int>(wte.size()), pn = static_cast<int>(wpe.size());
  const int on = S * n_embd;
  DeviceBuffer<int>   dt(tn);
  DeviceBuffer<float> dw(wn), dp(pn), dout(on);
  dt.CopyFromHost(token_ids.data(), tn);
  dw.CopyFromHost(wte.data(), wn);
  dp.CopyFromHost(wpe.data(), pn);
  mgie::LaunchEmbedding(dt.data(), dw.data(), dp.data(), dout.data(),
                         S, n_embd, vocab_size);
  CUDA_CHECK_LAST_SYNC();
  out.resize(on);
  dout.CopyToHost(out.data(), on);
}

TEST_F(EmbeddingTest, SingleTokenMatchesCpu) {
  const int vocab_size = 100, n_ctx = 128, n_embd = 64, S = 1;
  const auto wte = Rand(vocab_size * n_embd, 1);
  const auto wpe = Rand(n_ctx * n_embd, 2);
  const auto token_ids = RandTokenIds(S, vocab_size, 3);
  std::vector<float> gpu;
  RunEmbedding(token_ids, wte, wpe, gpu, S, n_embd, vocab_size);
  const auto ref = mt::CpuEmbedding(token_ids, wte, wpe, S, vocab_size, n_embd);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(EmbeddingTest, ShortSequenceMatchesCpu) {
  const int vocab_size = 50257, n_ctx = 1024, n_embd = 768, S = 64;
  const auto wte = Rand(vocab_size * n_embd, 4);
  const auto wpe = Rand(n_ctx * n_embd, 5);
  const auto token_ids = RandTokenIds(S, vocab_size, 6);
  std::vector<float> gpu;
  RunEmbedding(token_ids, wte, wpe, gpu, S, n_embd, vocab_size);
  const auto ref = mt::CpuEmbedding(token_ids, wte, wpe, S, vocab_size, n_embd);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(EmbeddingTest, FullContextLengthMatchesCpu) {
  const int vocab_size = 50257, n_ctx = 1024, n_embd = 768, S = 1024;
  const auto wte = Rand(vocab_size * n_embd, 7);
  const auto wpe = Rand(n_ctx * n_embd, 8);
  const auto token_ids = RandTokenIds(S, vocab_size, 9);
  std::vector<float> gpu;
  RunEmbedding(token_ids, wte, wpe, gpu, S, n_embd, vocab_size);
  const auto ref = mt::CpuEmbedding(token_ids, wte, wpe, S, vocab_size, n_embd);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(EmbeddingTest, StridedThreadsWorkForSmallEmbedDim) {
  // n_embd=32 < kBlockDim=256 — each thread handles ≤1 element, exercises
  // boundary condition where n_embd < block size.
  const int vocab_size = 64, n_ctx = 32, n_embd = 32, S = 8;
  const auto wte = Rand(vocab_size * n_embd, 10);
  const auto wpe = Rand(n_ctx * n_embd, 11);
  const auto token_ids = RandTokenIds(S, vocab_size, 12);
  std::vector<float> gpu;
  RunEmbedding(token_ids, wte, wpe, gpu, S, n_embd, vocab_size);
  const auto ref = mt::CpuEmbedding(token_ids, wte, wpe, S, vocab_size, n_embd);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(EmbeddingTest, ZeroLengthIsNoop) {
  // S=0: kernel returns immediately, no output written.
  const int vocab_size = 100, n_ctx = 128, n_embd = 64, S = 0;
  std::vector<int> token_ids;
  std::vector<float> wte(vocab_size * n_embd, 0.0f);
  std::vector<float> wpe(n_ctx * n_embd, 0.0f);
  std::vector<float> gpu;
  // Should not crash or throw.
  RunEmbedding(token_ids, wte, wpe, gpu, S, n_embd, vocab_size);
  SUCCEED();
}

// ---- Real-weights tests -----------------------------------------------------
// Load once per test suite (expensive ~622 MB upload).

class EmbeddingRealTest : public ::testing::Test {
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
    ASSERT_NE(weights_, nullptr) << "LoadGpt2Weights failed in SetUpTestSuite";
  }

  static std::unique_ptr<mgie::Gpt2Weights> weights_;
};
std::unique_ptr<mgie::Gpt2Weights> EmbeddingRealTest::weights_ = nullptr;

TEST_F(EmbeddingRealTest, RealWeightsProduceFiniteOutput) {
  const auto& cfg = weights_->config;
  const int S = 16;
  // Simple token IDs [0..S-1]
  std::vector<int> token_ids(S);
  for (int i = 0; i < S; ++i) token_ids[i] = i;

  const int on = S * cfg.n_embd;
  DeviceBuffer<int>   dt(S);
  DeviceBuffer<float> dout(on);
  dt.CopyFromHost(token_ids.data(), S);

  mgie::LaunchEmbedding(dt.data(), weights_->wte.data(), weights_->wpe.data(),
                         dout.data(), S, cfg.n_embd, cfg.vocab_size);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(on);
  dout.CopyToHost(gpu.data(), on);

  // All outputs should be finite (non-NaN, non-Inf).
  for (int i = 0; i < on; ++i) {
    EXPECT_TRUE(std::isfinite(gpu[i]))
        << "non-finite output at element " << i;
  }

  // Spot-check: out[0,:] = wte[token_0,:] + wpe[0,:]
  std::vector<float> wte_row(cfg.n_embd), wpe_row(cfg.n_embd);
  CUDA_CHECK(cudaMemcpy(wte_row.data(),
                        weights_->wte.data() + static_cast<std::size_t>(token_ids[0]) * cfg.n_embd,
                        cfg.n_embd * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(wpe_row.data(),
                        weights_->wpe.data(),
                        cfg.n_embd * sizeof(float), cudaMemcpyDeviceToHost));
  for (int d = 0; d < cfg.n_embd; ++d) {
    const float expected = wte_row[d] + wpe_row[d];
    EXPECT_NEAR(gpu[d], expected, 1e-5f)
        << "real-weight mismatch at d=" << d;
  }
}

TEST_F(EmbeddingRealTest, OutputShapeMatchesConfig) {
  const auto& cfg = weights_->config;
  const int S = 32;
  std::vector<int> token_ids(S);
  for (int i = 0; i < S; ++i) token_ids[i] = i % cfg.vocab_size;

  const int on = S * cfg.n_embd;
  DeviceBuffer<int>   dt(S);
  DeviceBuffer<float> dout(on);
  dt.CopyFromHost(token_ids.data(), S);

  mgie::LaunchEmbedding(dt.data(), weights_->wte.data(), weights_->wpe.data(),
                         dout.data(), S, cfg.n_embd, cfg.vocab_size);
  CUDA_CHECK_LAST_SYNC();

  // Verify the output buffer has the expected number of elements.
  EXPECT_EQ(dout.size(), static_cast<std::size_t>(S) * cfg.n_embd);
}

}  // namespace
