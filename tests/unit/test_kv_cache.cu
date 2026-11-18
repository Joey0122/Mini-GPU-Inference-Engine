// tests/unit/test_kv_cache.cu
//
// GPU unit tests for Phase 6a KV Cache data structure.
//
// Validates:
//   - Allocation, sizing, and head-pointer arithmetic
//   - Write-and-read-back bit-exactness
//   - Edge cases (zero dimensions)

#include <gtest/gtest.h>

#include <vector>

#include "common/errors.hpp"
#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"

namespace {

using mgie::KVCache;

bool CudaAvail() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }

class KVCacheTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA device"; }
};

TEST_F(KVCacheTest, Creation) {
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto cache = KVCache::Create(cfg);

  EXPECT_EQ(cache.n_layer(), 12);
  EXPECT_EQ(cache.n_head(), 12);
  EXPECT_EQ(cache.n_ctx(), 1024);
  EXPECT_EQ(cache.head_dim(), 64);
  EXPECT_EQ(cache.num_positions, 0);

  const std::size_t expected = 12ull * 12 * 1024 * 64;
  EXPECT_EQ(cache.k.size(), expected);
  EXPECT_EQ(cache.v.size(), expected);
  EXPECT_NE(cache.k.data(), nullptr);
  EXPECT_NE(cache.v.data(), nullptr);
  EXPECT_EQ(cache.k.bytes(), expected * sizeof(float));
  EXPECT_EQ(cache.v.bytes(), expected * sizeof(float));
}

TEST_F(KVCacheTest, HeadPointerArithmetic) {
  // Verify that head-pointers point to the correct offset and that
  // adjacent heads within a layer are contiguous.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto cache = KVCache::Create(cfg);

  const int n_head = cache.n_head();
  const int n_ctx = cache.n_ctx();
  const int d = cache.head_dim();
  const std::ptrdiff_t head_stride = static_cast<std::ptrdiff_t>(n_ctx) * d;

  // Layer 0, head 0 starts at the base.
  EXPECT_EQ(cache.k_head(0, 0), cache.k.data());
  EXPECT_EQ(cache.v_head(0, 0), cache.v.data());

  // Layer 0, head 1 is one head-stride past head 0.
  EXPECT_EQ(cache.k_head(0, 1) - cache.k_head(0, 0), head_stride);
  EXPECT_EQ(cache.v_head(0, 1) - cache.v_head(0, 0), head_stride);

  // Layer 1, head 0 starts after all heads of layer 0.
  const std::ptrdiff_t layer_stride = static_cast<std::ptrdiff_t>(n_head) * n_ctx * d;
  EXPECT_EQ(cache.k_head(1, 0) - cache.k_head(0, 0), layer_stride);
  EXPECT_EQ(cache.v_head(1, 0) - cache.v_head(0, 0), layer_stride);

  // Last valid head in last layer.
  const int L = cache.n_layer() - 1;
  const int H = n_head - 1;
  EXPECT_NE(cache.k_head(L, H), nullptr);
  EXPECT_NE(cache.v_head(L, H), nullptr);

  // const overloads
  const auto& ccache = cache;
  EXPECT_EQ(ccache.k_head(0, 0), cache.k.data());
  EXPECT_EQ(ccache.v_head(0, 0), cache.v.data());
}

TEST_F(KVCacheTest, WriteAndReadBack) {
  // Write known values into one position of one head of one layer,
  // then read back and verify bit-exact match.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto cache = KVCache::Create(cfg);

  const int layer = 3;
  const int head = 5;
  const int pos = 42;
  const int d = cache.head_dim();

  // Prepare host data for one key vector and one value vector.
  std::vector<float> hk(d), hv(d);
  for (int i = 0; i < d; ++i) {
    hk[i] = static_cast<float>(layer * 1000 + head * 100 + pos * 10 + i);
    hv[i] = static_cast<float>(layer * 1000 + head * 100 + pos * 10 + i + 50000);
  }

  // Write to device at the correct offset within the head's block.
  float* k_head = cache.k_head(layer, head);
  float* v_head = cache.v_head(layer, head);
  CUDA_CHECK(cudaMemcpy(k_head + static_cast<std::size_t>(pos) * d,
                        hk.data(), d * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v_head + static_cast<std::size_t>(pos) * d,
                        hv.data(), d * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Read back and verify.
  std::vector<float> rk(d), rv(d);
  CUDA_CHECK(cudaMemcpy(rk.data(),
                        k_head + static_cast<std::size_t>(pos) * d,
                        d * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rv.data(),
                        v_head + static_cast<std::size_t>(pos) * d,
                        d * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < d; ++i) {
    EXPECT_FLOAT_EQ(rk[i], hk[i])
        << "K cache readback mismatch at dim " << i;
    EXPECT_FLOAT_EQ(rv[i], hv[i])
        << "V cache readback mismatch at dim " << i;
  }
}

TEST_F(KVCacheTest, NonOverlappingHeads) {
  // Verify that writing to head 3 doesn't corrupt head 4's data.
  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto cache = KVCache::Create(cfg);

  const int layer = 0;
  const int pos = 0;
  const int d = cache.head_dim();

  // Write known patterns to head 3.
  std::vector<float> h3(d, 3.0f);
  CUDA_CHECK(cudaMemcpy(cache.k_head(layer, 3) + pos * d,
                        h3.data(), d * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Write known patterns to head 4.
  std::vector<float> h4(d, 4.0f);
  CUDA_CHECK(cudaMemcpy(cache.k_head(layer, 4) + pos * d,
                        h4.data(), d * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Read back head 3 — must still be 3.0, not 4.0.
  std::vector<float> r(d);
  CUDA_CHECK(cudaMemcpy(r.data(),
                        cache.k_head(layer, 3) + pos * d,
                        d * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < d; ++i) {
    EXPECT_FLOAT_EQ(r[i], 3.0f)
        << "Head 3 corrupted at dim " << i;
  }
}

TEST_F(KVCacheTest, ZeroConfigIsNoop) {
  // Should not crash — empty buffers.
  mgie::Gpt2Config cfg{};
  cfg.n_layer = 0;
  auto cache = KVCache::Create(cfg);
  EXPECT_EQ(cache.n_layer(), 0);
  EXPECT_EQ(cache.k.size(), 0);
  EXPECT_EQ(cache.v.size(), 0);
  // Head pointers of empty cache — data() may be nullptr, but the math is valid
  // as long as we don't dereference.
  SUCCEED();
}

}  // namespace
