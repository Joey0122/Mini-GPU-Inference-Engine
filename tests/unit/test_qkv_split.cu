// tests/unit/test_qkv_split.cu
//
// GPU unit tests for the Phase 5 C6 QKV Split kernel.
//
// Validates LaunchQkvSplit against:
//   1. CPU reference (CpuQkvSplit) — elementwise identity mapping
//   2. Deterministic layout test — every output element uniquely traceable
//   3. Round-trip test — split then inverse-reassemble → exact match
//   4. Head-major layout verification — confirmed compatible with LaunchAttention

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/qkv_split.hpp"

namespace {

using mgie::DeviceBuffer;
namespace mt = mgie::test;

bool CudaAvail() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }

class QkvSplitTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA"; }
};

std::vector<float> Rand(int n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

void RunQkvSplit(const std::vector<float>& x,
                 std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
                 int S, int n_head, int head_dim) {
  const int n_embd = n_head * head_dim;
  const int xn = S * 3 * n_embd;
  const int tn = n_head * S * head_dim;

  DeviceBuffer<float> dx(xn), dq(tn), dk(tn), dv(tn);
  dx.CopyFromHost(x.data(), xn);
  mgie::LaunchQkvSplit(dx.data(), dq.data(), dk.data(), dv.data(),
                        S, n_embd, n_head);
  CUDA_CHECK_LAST_SYNC();
  q.resize(tn); k.resize(tn); v.resize(tn);
  dq.CopyToHost(q.data(), tn);
  dk.CopyToHost(k.data(), tn);
  dv.CopyToHost(v.data(), tn);
}

// ==========================================================================
// Test 1: Deterministic layout test — every output element uniquely traceable
// ==========================================================================
// x[s, block*768 + h*64 + d] is assigned: block_base + s*offset_s + h*offset_h + d
// so the source (s, block, h, d) can be read back from the value.
// Block: 0=Q, 1=K, 2=V.  offset_s > n_head*offset_h + head_dim.

std::vector<float> MakeDeterministicX(int S, int n_head, int head_dim) {
  const int n_embd = n_head * head_dim;
  const int xn = S * 3 * n_embd;
  std::vector<float> x(xn);

  const float block_base[3] = {100000.0f, 200000.0f, 300000.0f};
  const float offset_s = 1000.0f;  // large enough: S=1024 max, 1024*1000 = 1,024,000 < block_base
  const float offset_h = 10.0f;    // n_head=12 max, 12*10=120 < offset_s
  const float offset_d = 1.0f;

  for (int s = 0; s < S; ++s) {
    const int src_row = s * 3 * n_embd;
    for (int h = 0; h < n_head; ++h) {
      const int h_off = h * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        for (int blk = 0; blk < 3; ++blk) {
          x[src_row + blk * n_embd + h_off + d] =
              block_base[blk] + s * offset_s + h * offset_h + d * offset_d;
        }
      }
    }
  }
  return x;
}

TEST_F(QkvSplitTest, DeterministicLayoutEveryElementTraceable) {
  const int S = 16, n_head = 12, head_dim = 64;
  const auto x = MakeDeterministicX(S, n_head, head_dim);

  std::vector<float> q, k, v;
  RunQkvSplit(x, q, k, v, S, n_head, head_dim);

  // Verify every element of q, k, v can be traced to the correct source.
  for (int h = 0; h < n_head; ++h) {
    for (int s = 0; s < S; ++s) {
      const std::size_t tgt_base = (static_cast<std::size_t>(h) * S + s) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        // Expected: block_base[0] + s*1000 + h*10 + d*1 for Q, etc.
        const float expected_q = 100000.0f + s * 1000.0f + h * 10.0f + d * 1.0f;
        const float expected_k = 200000.0f + s * 1000.0f + h * 10.0f + d * 1.0f;
        const float expected_v = 300000.0f + s * 1000.0f + h * 10.0f + d * 1.0f;
        EXPECT_EQ(q[tgt_base + d], expected_q)
            << "Q mismatch at h=" << h << " s=" << s << " d=" << d;
        EXPECT_EQ(k[tgt_base + d], expected_k)
            << "K mismatch at h=" << h << " s=" << s << " d=" << d;
        EXPECT_EQ(v[tgt_base + d], expected_v)
            << "V mismatch at h=" << h << " s=" << s << " d=" << d;
      }
    }
  }
}

// ==========================================================================
// Test 2: Round-trip — split then inverse reassemble → exact equality
// ==========================================================================

std::vector<float> ReassembleQkv(const std::vector<float>& q,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  int S, int n_head, int head_dim) {
  const int n_embd = n_head * head_dim;
  std::vector<float> x(static_cast<std::size_t>(S) * 3 * n_embd, 0.0f);

  for (int s = 0; s < S; ++s) {
    const int dst_row = s * 3 * n_embd;
    for (int h = 0; h < n_head; ++h) {
      const std::size_t tgt_base = (static_cast<std::size_t>(h) * S + s) * head_dim;
      const int h_off = h * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        x[dst_row + 0 * n_embd + h_off + d] = q[tgt_base + d];
        x[dst_row + 1 * n_embd + h_off + d] = k[tgt_base + d];
        x[dst_row + 2 * n_embd + h_off + d] = v[tgt_base + d];
      }
    }
  }
  return x;
}

TEST_F(QkvSplitTest, RoundTripPreservesExactIdentity) {
  const int S = 64, n_head = 12, head_dim = 64, n_embd = n_head * head_dim;
  const auto original_x = Rand(S * 3 * n_embd, 42);

  std::vector<float> q, k, v;
  RunQkvSplit(original_x, q, k, v, S, n_head, head_dim);

  const auto reassembled = ReassembleQkv(q, k, v, S, n_head, head_dim);

  // Pure rearrangement — every element must be bit-exact.
  const float max_err = mt::MaxAbsError(original_x, reassembled);
  EXPECT_EQ(max_err, 0.0f)
      << "round-trip max_abs_error = " << max_err << " (expected 0.0)";
}

// ==========================================================================
// Test 3: Random data vs CPU reference — GPT-2 shapes
// ==========================================================================

TEST_F(QkvSplitTest, Gpt2ShapesMatchCpu) {
  const int S = 128, n_head = 12, head_dim = 64, n_embd = n_head * head_dim;
  const auto x = Rand(S * 3 * n_embd, 7);

  std::vector<float> q, k, v;
  RunQkvSplit(x, q, k, v, S, n_head, head_dim);

  std::vector<float> ref_q, ref_k, ref_v;
  mt::CpuQkvSplit(x, ref_q, ref_k, ref_v, S, n_head, head_dim);

  EXPECT_LT(mt::MaxAbsError(q, ref_q), mt::kFp32AbsTol);
  EXPECT_LT(mt::MaxAbsError(k, ref_k), mt::kFp32AbsTol);
  EXPECT_LT(mt::MaxAbsError(v, ref_v), mt::kFp32AbsTol);
}

// ==========================================================================
// Test 4: Full context length
// ==========================================================================

TEST_F(QkvSplitTest, FullContextLengthMatchesCpu) {
  const int S = 1024, n_head = 12, head_dim = 64, n_embd = n_head * head_dim;
  const auto x = Rand(S * 3 * n_embd, 9);

  std::vector<float> q, k, v;
  RunQkvSplit(x, q, k, v, S, n_head, head_dim);

  std::vector<float> ref_q, ref_k, ref_v;
  mt::CpuQkvSplit(x, ref_q, ref_k, ref_v, S, n_head, head_dim);

  EXPECT_LT(mt::MaxAbsError(q, ref_q), mt::kFp32AbsTol);
  EXPECT_LT(mt::MaxAbsError(k, ref_k), mt::kFp32AbsTol);
  EXPECT_LT(mt::MaxAbsError(v, ref_v), mt::kFp32AbsTol);
}

// ==========================================================================
// Test 5: Head-major layout verification (compatible with LaunchAttention)
// ==========================================================================
// Confirms that element (h, s, d) is at offset (h*S + s)*head_dim + d.
// This is the exact formula used by AttentionScoresKernel line 38.

TEST_F(QkvSplitTest, HeadMajorLayoutMatchesAttentionFormula) {
  const int S = 4, n_head = 3, head_dim = 8, n_embd = n_head * head_dim;

  // Build x with known values: q[h,s,d] block = h*100 + s*10 + d
  // Placement in x: column h*head_dim + d within Q block = h*8 + d
  std::vector<float> x(static_cast<std::size_t>(S) * 3 * n_embd, -1.0f);
  for (int s = 0; s < S; ++s) {
    const int src_row = s * 3 * n_embd;
    for (int h = 0; h < n_head; ++h) {
      const int h_off = h * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        x[src_row + 0 * n_embd + h_off + d] = h * 100.0f + s * 10.0f + d;  // Q
        x[src_row + 1 * n_embd + h_off + d] = h * 100.0f + s * 10.0f + d + 500.0f;  // K
        x[src_row + 2 * n_embd + h_off + d] = h * 100.0f + s * 10.0f + d + 1000.0f; // V
      }
    }
  }

  std::vector<float> q, k, v;
  RunQkvSplit(x, q, k, v, S, n_head, head_dim);

  // Verify offset formula: q[(h * S + s) * head_dim + d] matches expected value.
  for (int h = 0; h < n_head; ++h) {
    for (int s = 0; s < S; ++s) {
      const std::size_t base = (static_cast<std::size_t>(h) * S + s) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        EXPECT_EQ(q[base + d], h * 100.0f + s * 10.0f + d)
            << "Q: wrong value at head=" << h << " s=" << s << " d=" << d
            << " (offset=" << base + d << ")";
        EXPECT_EQ(k[base + d], h * 100.0f + s * 10.0f + d + 500.0f)
            << "K: wrong value at head=" << h << " s=" << s << " d=" << d;
        EXPECT_EQ(v[base + d], h * 100.0f + s * 10.0f + d + 1000.0f)
            << "V: wrong value at head=" << h << " s=" << s << " d=" << d;
      }
    }
  }
}

// ==========================================================================
// Test 6: Zero-length no-op
// ==========================================================================

TEST_F(QkvSplitTest, ZeroLengthIsNoop) {
  const int S = 0, n_head = 12, head_dim = 64;
  std::vector<float> x;
  std::vector<float> q, k, v;
  RunQkvSplit(x, q, k, v, S, n_head, head_dim);
  SUCCEED();
}

// ---- Phase 6a: QKV Split to KV Cache ---------------------------------------

void RunQkvSplitToCache(const std::vector<float>& x,
                         std::vector<float>& q_out,
                         std::vector<float>& k_cache_out,
                         std::vector<float>& v_cache_out,
                         int pos_offset,
                         int S, int n_embd, int n_head, int n_ctx) {
  const int head_dim = n_embd / n_head;
  const size_t qn = static_cast<size_t>(n_head) * S * head_dim;
  const size_t kvn = static_cast<size_t>(n_head) * n_ctx * head_dim;
  DeviceBuffer<float> dx(x.size()), dq(qn), dk(kvn), dv(kvn);
  dx.CopyFromHost(x.data(), x.size());
  mgie::LaunchQkvSplitToCache(dx.data(), dq.data(), dk.data(), dv.data(),
                               pos_offset, S, n_embd, n_head, n_ctx);
  CUDA_CHECK_LAST_SYNC();
  q_out.resize(qn);
  k_cache_out.resize(kvn);
  v_cache_out.resize(kvn);
  dq.CopyToHost(q_out.data(), qn);
  dk.CopyToHost(k_cache_out.data(), kvn);
  dv.CopyToHost(v_cache_out.data(), kvn);
}

TEST_F(QkvSplitTest, CacheWriteQMatchesOriginalSplit) {
  // Q from QkvSplitToCache must match Q from regular QkvSplit.
  const int S = 8, n_head = 4, head_dim = 16, n_embd = n_head * head_dim;
  const int n_ctx = 32;

  const auto x = Rand(static_cast<size_t>(S) * 3 * n_embd, 100);
  std::vector<float> q_ref, k_ref, v_ref;
  RunQkvSplit(x, q_ref, k_ref, v_ref, S, n_head, head_dim);

  std::vector<float> q_cache, k_cache, v_cache;
  RunQkvSplitToCache(x, q_cache, k_cache, v_cache, 0, S, n_embd, n_head, n_ctx);

  // Q should be identical to the original split output.
  EXPECT_EQ(q_cache.size(), q_ref.size());
  EXPECT_EQ(mt::MaxAbsError(q_cache, q_ref), 0.0f)
      << "Q from cache-variant split should match original Q";
}

TEST_F(QkvSplitTest, CacheWriteKeysAtCorrectPositions) {
  // Prefill at pos_offset=0 with S=4: K/V for positions 0..3 should be
  // at the correct cache offsets.
  const int S = 4, n_head = 2, head_dim = 8, n_embd = n_head * head_dim;
  const int n_ctx = 16;

  // Build x with known, distinguishable values.
  auto x = Rand(static_cast<size_t>(S) * 3 * n_embd, 200);
  // Set x[0, n_embd + 0] = 42.0 as a marker in the K block at position 0.
  x[n_embd] = 42.0f;
  // Set x[1, n_embd + 1] = 99.0 in the K block at position 1.
  x[3 * n_embd + n_embd + 1] = 99.0f;

  std::vector<float> q_out, k_cache, v_cache;
  RunQkvSplitToCache(x, q_out, k_cache, v_cache, 0, S, n_embd, n_head, n_ctx);

  // Check marker values in K cache.
  // Position 0, head 0, dim 0.
  float* k_head0 = k_cache.data();
  float* v_head0 = v_cache.data();
  EXPECT_FLOAT_EQ(k_head0[0 * head_dim + 0], 42.0f)
      << "K marker at (h=0,pos=0,d=0) not found in cache";

  // Position 1, head 0, dim 1.
  EXPECT_FLOAT_EQ(k_head0[1 * head_dim + 1], 99.0f)
      << "K marker at (h=0,pos=1,d=1) not found in cache";
}

TEST_F(QkvSplitTest, CacheWriteWithPosOffset) {
  // Write S=2 rows at pos_offset=5. Verify K/V appear at cache positions 5..6.
  const int S = 2, n_head = 2, head_dim = 8, n_embd = n_head * head_dim;
  const int n_ctx = 16, pos_offset = 5;

  // Create input where each row/head/dim has a unique value.
  auto x = Rand(static_cast<size_t>(S) * 3 * n_embd, 300);

  std::vector<float> q_out, k_cache, v_cache;
  RunQkvSplitToCache(x, q_out, k_cache, v_cache, pos_offset, S, n_embd, n_head, n_ctx);

  // Positions 0..4 should be zero (never written).
  float* k_head0 = k_cache.data();
  for (int p = 0; p < pos_offset; ++p)
    for (int d = 0; d < head_dim; ++d)
      EXPECT_FLOAT_EQ(k_head0[static_cast<size_t>(p) * head_dim + d], 0.0f)
          << "K cache position " << p << " should be untouched (zero-initialized)";

  // Positions 5..6 should be non-zero (written).
  bool found_nonzero = false;
  for (int p = pos_offset; p < pos_offset + S; ++p) {
    for (int d = 0; d < head_dim; ++d) {
      if (std::fabs(k_head0[static_cast<size_t>(p) * head_dim + d]) > 1e-8f)
        found_nonzero = true;
    }
  }
  EXPECT_TRUE(found_nonzero) << "No non-zero values found in K cache written region";
}

TEST_F(QkvSplitTest, CacheWriteZeroLengthIsNoop) {
  mgie::LaunchQkvSplitToCache(nullptr, nullptr, nullptr, nullptr,
                               0, 0, 0, 0, 0);
  SUCCEED();
}

}  // namespace
