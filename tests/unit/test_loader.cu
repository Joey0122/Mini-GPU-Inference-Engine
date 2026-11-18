// tests/unit/test_loader.cu
//
// GPU unit tests for the Phase 5 weight loader.
//
// Tests gracefully skip when the weight files are absent. Run
//   python scripts/export_gpt2.py --synthetic
// to generate test weights (random values, correct GPT-2 Small shapes).
// For full inference validation use real weights (no --synthetic flag, requires
// internet access).

#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>

#include "common/errors.hpp"
#include "cuda/memory.hpp"
#include "model/gpt2_config.hpp"
#include "model/loader.hpp"

#ifndef MGIE_MODEL_WEIGHTS_DIR
#define MGIE_MODEL_WEIGHTS_DIR "data/model_weights"
#endif

namespace {

const std::string kWeightsDir = MGIE_MODEL_WEIGHTS_DIR;

bool WeightsExist() {
    std::ifstream f(kWeightsDir + "/gpt2_manifest.txt");
    return f.good();
}

bool CudaAvail() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

class LoaderTest : public ::testing::Test {
 protected:
    void SetUp() override {
        if (!CudaAvail()) GTEST_SKIP() << "no CUDA device";
        if (!WeightsExist()) {
            GTEST_SKIP() << "weight files not found at " << kWeightsDir
                         << " — run: python scripts/export_gpt2.py --synthetic";
        }
    }
};

// Load once via a test fixture so the 622 MB copy happens once per test run.
class LoaderFixture : public ::testing::Test {
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
std::unique_ptr<mgie::Gpt2Weights> LoaderFixture::weights_ = nullptr;

// ---- Shape tests -----------------------------------------------------------

TEST_F(LoaderFixture, EmbeddingShapes) {
    const auto& cfg = weights_->config;
    EXPECT_EQ(weights_->wte.size(),
              static_cast<std::size_t>(cfg.vocab_size) * cfg.n_embd);
    EXPECT_EQ(weights_->wte_T.size(),
              static_cast<std::size_t>(cfg.n_embd) * cfg.vocab_size);
    EXPECT_EQ(weights_->wpe.size(),
              static_cast<std::size_t>(cfg.n_ctx) * cfg.n_embd);
}

TEST_F(LoaderFixture, BlockShapes) {
    const auto& cfg = weights_->config;
    for (int i = 0; i < cfg.n_layer; ++i) {
        const auto& b = weights_->blocks[static_cast<std::size_t>(i)];
        EXPECT_EQ(b.ln1_w.size(), static_cast<std::size_t>(cfg.n_embd))
            << "block " << i << " ln1_w";
        EXPECT_EQ(b.c_attn_w.size(),
                  static_cast<std::size_t>(cfg.n_embd) * 3 * cfg.n_embd)
            << "block " << i << " c_attn_w";
        EXPECT_EQ(b.c_attn_b.size(), static_cast<std::size_t>(3 * cfg.n_embd))
            << "block " << i << " c_attn_b";
        EXPECT_EQ(b.c_proj_w.size(),
                  static_cast<std::size_t>(cfg.n_embd) * cfg.n_embd)
            << "block " << i << " c_proj_w";
        EXPECT_EQ(b.mlp_fc_w.size(),
                  static_cast<std::size_t>(cfg.n_embd) * cfg.mlp_inner)
            << "block " << i << " mlp_fc_w";
        EXPECT_EQ(b.mlp_proj_w.size(),
                  static_cast<std::size_t>(cfg.mlp_inner) * cfg.n_embd)
            << "block " << i << " mlp_proj_w";
    }
}

TEST_F(LoaderFixture, FinalLayerNormShapes) {
    const auto& cfg = weights_->config;
    EXPECT_EQ(weights_->ln_f_w.size(), static_cast<std::size_t>(cfg.n_embd));
    EXPECT_EQ(weights_->ln_f_b.size(), static_cast<std::size_t>(cfg.n_embd));
}

// ---- Transpose consistency -------------------------------------------------
// wte has shape [vocab_size, n_embd]. wte_T has shape [n_embd, vocab_size].
// wte[i, j] should equal wte_T[j, i], checked at a few non-trivial positions.
//
// Position table (i=row in wte, j=col in wte, → flat indices):
//   wte flat index    = i * n_embd + j
//   wte_T flat index  = j * vocab_size + i

TEST_F(LoaderFixture, WteTTransposeIsCorrect) {
    const auto& cfg = weights_->config;
    // (i, j) pairs: i < vocab_size, j < n_embd
    const int pairs[][2] = {{0, 0}, {0, 5}, {0, cfg.n_embd - 1},
                            {42, 100}, {cfg.vocab_size / 2, cfg.n_embd / 3}};
    for (const auto& p : pairs) {
        const int i = p[0], j = p[1];
        // wte[i, j] — flat index i*n_embd+j
        float wte_val = 0.f, wte_T_val = 0.f;
        CUDA_CHECK(cudaMemcpy(&wte_val,
                              weights_->wte.data() +
                                  static_cast<std::size_t>(i) * cfg.n_embd + j,
                              sizeof(float), cudaMemcpyDeviceToHost));
        // wte_T[j, i] — flat index j*vocab_size+i
        CUDA_CHECK(cudaMemcpy(&wte_T_val,
                              weights_->wte_T.data() +
                                  static_cast<std::size_t>(j) * cfg.vocab_size + i,
                              sizeof(float), cudaMemcpyDeviceToHost));
        EXPECT_FLOAT_EQ(wte_val, wte_T_val)
            << "wte[" << i << "," << j << "] != wte_T[" << j << "," << i << "]";
    }
}

// ---- Error-handling tests --------------------------------------------------

TEST_F(LoaderTest, ThrowsOnMissingManifest) {
    const auto cfg = mgie::DefaultGpt2SmallConfig();
    EXPECT_THROW(mgie::LoadGpt2Weights("/nonexistent/path", cfg),
                 std::runtime_error);
}

}  // namespace
