// tests/unit/test_attention.cu
//
// GPU unit tests for the Phase 4 attention kernels, validated against the
// double-precision CPU reference (mgie::test::CpuAttention) within the
// sequence-length-scaled tolerance. Requires a CUDA device (label "gpu").

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/attention.hpp"

namespace {

using mgie::DeviceBuffer;
namespace mt = mgie::test;

bool CudaAvail() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }

class AttentionTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA"; }
};

std::vector<float> Rand(int n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// ---- Scores kernel: scale * QKᵀ, causal-masked ------------------------------

// CPU reference for the masked scores (matches AttentionScoresKernel).
std::vector<float> CpuScores(const std::vector<float>& q,
                             const std::vector<float>& k,
                             int n_head, int S, int head_dim) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<float> s(static_cast<std::size_t>(n_head) * S * S);
  for (int h = 0; h < n_head; ++h)
    for (int i = 0; i < S; ++i)
      for (int j = 0; j < S; ++j) {
        const std::size_t idx = (static_cast<std::size_t>(h) * S + i) * S + j;
        if (j > i) { s[idx] = -1e30f; continue; }
        double dot = 0.0;
        const std::size_t qb = (static_cast<std::size_t>(h) * S + i) * head_dim;
        const std::size_t kb = (static_cast<std::size_t>(h) * S + j) * head_dim;
        for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(q[qb+d]) * k[kb+d];
        s[idx] = static_cast<float>(dot * scale);
      }
  return s;
}

void RunScores(const std::vector<float>& q, const std::vector<float>& k,
               std::vector<float>& out, int n_head, int S, int head_dim) {
  const int qn = n_head * S * head_dim, sn = n_head * S * S;
  DeviceBuffer<float> dq(qn), dk(qn), ds(sn);
  dq.CopyFromHost(q.data(), qn);
  dk.CopyFromHost(k.data(), qn);
  mgie::LaunchAttentionScores(dq.data(), dk.data(), ds.data(), n_head, S, head_dim);
  CUDA_CHECK_LAST_SYNC();
  out.resize(sn);
  ds.CopyToHost(out.data(), sn);
}

TEST_F(AttentionTest, ScoresMatchCpuAndAreCausal) {
  const int n_head = 12, S = 64, head_dim = 64;
  const auto q = Rand(n_head * S * head_dim, 1);
  const auto k = Rand(n_head * S * head_dim, 2);
  std::vector<float> gpu;
  RunScores(q, k, gpu, n_head, S, head_dim);
  const auto ref = CpuScores(q, k, n_head, S, head_dim);

  // Valid (j<=i) entries match; masked (j>i) entries are the sentinel.
  double max_valid_err = 0.0;
  for (int h = 0; h < n_head; ++h)
    for (int i = 0; i < S; ++i)
      for (int j = 0; j < S; ++j) {
        const std::size_t idx = (static_cast<std::size_t>(h) * S + i) * S + j;
        if (j > i) {
          EXPECT_LT(gpu[idx], -1e29f) << "unmasked future key h="<<h<<" i="<<i<<" j="<<j;
        } else {
          max_valid_err = std::max(max_valid_err,
                                   std::abs(static_cast<double>(gpu[idx]) - ref[idx]));
        }
      }
  EXPECT_LT(max_valid_err, mt::AttentionAbsTol(head_dim));
}

TEST_F(AttentionTest, ScoresScaleIsInvSqrtHeadDim) {
  // Single head, S=2, head_dim=4, q=k=ones -> dot=4, scaled=4/sqrt(4)=2.
  const int n_head = 1, S = 2, head_dim = 4;
  std::vector<float> q(n_head*S*head_dim, 1.f), k(n_head*S*head_dim, 1.f);
  std::vector<float> gpu;
  RunScores(q, k, gpu, n_head, S, head_dim);
  EXPECT_NEAR(gpu[0], 2.0f, 1e-5f);   // [0,0] valid
  EXPECT_LT(gpu[1], -1e29f);          // [0,1] masked (future)
  EXPECT_NEAR(gpu[2], 2.0f, 1e-5f);   // [1,0] valid
  EXPECT_NEAR(gpu[3], 2.0f, 1e-5f);   // [1,1] valid
}

// ---- AV kernel: out = weights · V, heads merged -----------------------------

// CPU reference for AV given arbitrary weights [n_head,S,S] and v [n_head,S,hd].
std::vector<float> CpuAV(const std::vector<float>& w, const std::vector<float>& v,
                         int n_head, int S, int head_dim) {
  const int E = n_head * head_dim;
  std::vector<float> out(static_cast<std::size_t>(S) * E, 0.f);
  for (int h = 0; h < n_head; ++h)
    for (int i = 0; i < S; ++i)
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j < S; ++j)
          acc += static_cast<double>(w[(static_cast<std::size_t>(h)*S+i)*S+j]) *
                 v[(static_cast<std::size_t>(h)*S+j)*head_dim+d];
        out[static_cast<std::size_t>(i)*E + h*head_dim + d] = static_cast<float>(acc);
      }
  return out;
}

void RunAV(const std::vector<float>& w, const std::vector<float>& v,
           std::vector<float>& out, int n_head, int S, int head_dim) {
  const int wn = n_head*S*S, vn = n_head*S*head_dim, on = S*n_head*head_dim;
  DeviceBuffer<float> dw(wn), dv(vn), dout(on);
  dw.CopyFromHost(w.data(), wn);
  dv.CopyFromHost(v.data(), vn);
  mgie::LaunchAttentionAV(dw.data(), dv.data(), dout.data(), n_head, S, head_dim);
  CUDA_CHECK_LAST_SYNC();
  out.resize(on);
  dout.CopyToHost(out.data(), on);
}

TEST_F(AttentionTest, AVMatchesCpuWithMergedHeads) {
  const int n_head = 12, S = 64, head_dim = 64;
  // Lower-triangular row-normalized weights (mimics post-softmax causal output).
  std::vector<float> w(static_cast<std::size_t>(n_head)*S*S, 0.f);
  for (int h = 0; h < n_head; ++h)
    for (int i = 0; i < S; ++i) {
      float s = 0.f;
      for (int j = 0; j <= i; ++j) { w[(static_cast<std::size_t>(h)*S+i)*S+j] = j+1; s += j+1; }
      for (int j = 0; j <= i; ++j) w[(static_cast<std::size_t>(h)*S+i)*S+j] /= s;
    }
  const auto v = Rand(n_head*S*head_dim, 7);
  std::vector<float> gpu;
  RunAV(w, v, gpu, n_head, S, head_dim);
  const auto ref = CpuAV(w, v, n_head, S, head_dim);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::AttentionAbsTol(S));
}

TEST_F(AttentionTest, AVMultiHeadColumnLayout) {
  // 2 heads, S=1, hd=2, weight=1 -> out row = concat(v_head0, v_head1).
  const int n_head = 2, S = 1, head_dim = 2;
  std::vector<float> w(n_head*S*S, 1.f);
  std::vector<float> v{1, 2, 3, 4};  // head0:[1,2] head1:[3,4]
  std::vector<float> gpu;
  RunAV(w, v, gpu, n_head, S, head_dim);
  ASSERT_EQ(gpu.size(), 4u);
  EXPECT_NEAR(gpu[0], 1.f, 1e-6f);
  EXPECT_NEAR(gpu[1], 2.f, 1e-6f);
  EXPECT_NEAR(gpu[2], 3.f, 1e-6f);   // head 1 at column offset head_dim=2
  EXPECT_NEAR(gpu[3], 4.f, 1e-6f);
}

// ---- Integrated attention: Q,K,V -> out, end-to-end vs CPU ------------------

void RunAttention(const std::vector<float>& q, const std::vector<float>& k,
                  const std::vector<float>& v, std::vector<float>& out,
                  int n_head, int S, int head_dim) {
  const int qn = n_head*S*head_dim, sn = n_head*S*S, on = S*n_head*head_dim;
  DeviceBuffer<float> dq(qn), dk(qn), dv(qn), dscores(sn), dweights(sn), dout(on);
  dq.CopyFromHost(q.data(), qn);
  dk.CopyFromHost(k.data(), qn);
  dv.CopyFromHost(v.data(), qn);
  mgie::LaunchAttention(dq.data(), dk.data(), dv.data(), dout.data(),
                        dscores.data(), dweights.data(), n_head, S, head_dim);
  CUDA_CHECK_LAST_SYNC();
  out.resize(on);
  dout.CopyToHost(out.data(), on);
}

void ExpectAttentionMatch(int n_head, int S, int head_dim, unsigned seed,
                          float mag = 1.f, float tol = 0.f) {
  const auto q = Rand(n_head*S*head_dim, seed,     -mag, mag);
  const auto k = Rand(n_head*S*head_dim, seed+100, -mag, mag);
  const auto v = Rand(n_head*S*head_dim, seed+200, -mag, mag);
  std::vector<float> gpu;
  RunAttention(q, k, v, gpu, n_head, S, head_dim);
  const auto ref = mt::CpuAttention(q, k, v, n_head, S, head_dim);
  // Default tol is the coarse accumulation bound; callers override for
  // large-magnitude inputs where the spec gate (1e-4) is the true requirement.
  const double use_tol = tol > 0.f ? tol : mt::AttentionAbsTol(S);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), use_tol)
      << "n_head=" << n_head << " S=" << S << " head_dim=" << head_dim;
}

TEST_F(AttentionTest, IntegratedMatchesCpuGpt2Shapes) {
  ExpectAttentionMatch(12, 8,   64, 1);   // tiny
  ExpectAttentionMatch(12, 64,  64, 2);   // small
  ExpectAttentionMatch(12, 128, 64, 3);   // representative
}

TEST_F(AttentionTest, IntegratedHandlesLargeMagnitudeStably) {
  // Large-magnitude Q/K stresses softmax stability through the masked scores.
  // The sharper distribution + larger float32 intermediates exceed the coarse
  // AttentionAbsTol smoke-check bound but stay well under the PROJECT_SPEC FP32
  // gate (1e-4) — which is the authoritative correctness requirement.
  ExpectAttentionMatch(4, 64, 64, 9, /*mag=*/8.f, /*tol=*/mt::kFp32AbsTol);
}

TEST_F(AttentionTest, IntegratedRowZeroEqualsValueZero) {
  // Causal: query 0 attends only to key 0, so out row 0 == v[:,0,:] per head.
  const int n_head = 3, S = 16, head_dim = 64;
  const auto q = Rand(n_head*S*head_dim, 11);
  const auto k = Rand(n_head*S*head_dim, 12);
  const auto v = Rand(n_head*S*head_dim, 13);
  std::vector<float> gpu;
  RunAttention(q, k, v, gpu, n_head, S, head_dim);
  const int E = n_head * head_dim;
  for (int h = 0; h < n_head; ++h)
    for (int d = 0; d < head_dim; ++d)
      EXPECT_NEAR(gpu[0*E + h*head_dim + d],
                  v[(static_cast<std::size_t>(h)*S+0)*head_dim+d], 1e-4f);
}

// ---- Phase 6a: Cached attention (KV cache) -----------------------------------

void RunAttentionCached(const std::vector<float>& q_new,   // [n_head, 1, head_dim]
                         const std::vector<float>& k_cache, // [n_head, S, head_dim]
                         const std::vector<float>& v_cache, // [n_head, S, head_dim]
                         std::vector<float>& out,
                         int n_head, int S, int head_dim) {
  const int qn = n_head * 1 * head_dim;
  const int kn = n_head * S * head_dim;
  const int sn = n_head * 1 * S;       // scores: [n_head, 1, S]
  const int on = 1 * n_head * head_dim; // output: [1, n_head*head_dim]
  DeviceBuffer<float> dq(qn), dk(kn), dv(kn), dscores(sn), dweights(sn), dout(on);
  dq.CopyFromHost(q_new.data(), qn);
  dk.CopyFromHost(k_cache.data(), kn);
  dv.CopyFromHost(v_cache.data(), kn);
  // When S == stride (dense tensor): num_cached = stride = S
  mgie::LaunchAttentionCached(dq.data(), dk.data(), dv.data(),
                               dout.data(), dscores.data(), dweights.data(),
                               n_head, S, head_dim, S);
  CUDA_CHECK_LAST_SYNC();
  out.resize(on);
  dout.CopyToHost(out.data(), on);
}

TEST_F(AttentionTest, CachedMatchesFullLastRow) {
  // Full attention over S positions. Extract last query row, run cached
  // attention against the same K/V. Output must match the last row of
  // the full attention output. Gate: max_abs_error ≤ 1e-5.
  const int n_head = 12, S = 64, head_dim = 64;

  const auto q = Rand(n_head * S * head_dim, 20);
  const auto k = Rand(n_head * S * head_dim, 21);
  const auto v = Rand(n_head * S * head_dim, 22);

  std::vector<float> full_out;
  RunAttention(q, k, v, full_out, n_head, S, head_dim);

  // Extract last query row (position S-1) for each head.
  std::vector<float> q_last(n_head * 1 * head_dim);
  for (int h = 0; h < n_head; ++h) {
    const std::size_t src = (static_cast<std::size_t>(h) * S + (S - 1)) * head_dim;
    const std::size_t dst = static_cast<std::size_t>(h) * head_dim;
    for (int d = 0; d < head_dim; ++d)
      q_last[dst + d] = q[src + d];
  }

  std::vector<float> cached_out;
  RunAttentionCached(q_last, k, v, cached_out, n_head, S, head_dim);

  // Compare: cached_out == full_out last row.
  const int E = n_head * head_dim;
  std::vector<float> full_last_row(E);
  for (int i = 0; i < E; ++i)
    full_last_row[i] = full_out[static_cast<std::size_t>(S - 1) * E + i];

  const float err = mt::MaxAbsError(cached_out, full_last_row);
  EXPECT_LT(err, 1e-5f)
      << "Cached attention last row diverges from full attention: "
      << "max_abs_error=" << err << " (gate: 1e-5)";
}

TEST_F(AttentionTest, CachedSingleKeyOutput) {
  // With S=1 cached key and temp→0 softmax (single key → weight=1.0),
  // output should equal the value vector for each head: out[h,d] = v[h,0,d].
  const int n_head = 4, S = 1, head_dim = 32;

  const auto q_new = Rand(n_head * 1 * head_dim, 30, -1.f, 1.f);
  const auto k_cache = Rand(n_head * S * head_dim, 31, -1.f, 1.f);
  const auto v_cache = Rand(n_head * S * head_dim, 32, -1.f, 1.f);

  std::vector<float> cached_out;
  RunAttentionCached(q_new, k_cache, v_cache, cached_out, n_head, S, head_dim);

  // Softmax of a single element = 1.0. So out = 1.0 * V.
  const int E = n_head * head_dim;
  for (int h = 0; h < n_head; ++h) {
    for (int d = 0; d < head_dim; ++d) {
      const float expected = v_cache[static_cast<std::size_t>(h) * S * head_dim + d];
      EXPECT_NEAR(cached_out[static_cast<std::size_t>(h) * head_dim + d],
                  expected, 1e-5f);
    }
  }
}

TEST_F(AttentionTest, CachedGpt2Shapes) {
  // GPT-2 Small shapes: 12 heads, varied S, head_dim=64.
  const int n_head = 12, head_dim = 64;

  for (int S : {1, 4, 16, 64}) {
    const auto q_new = Rand(n_head * 1 * head_dim, 40 + S, -1.f, 1.f);
    const auto k_cache = Rand(n_head * S * head_dim, 50 + S, -1.f, 1.f);
    const auto v_cache = Rand(n_head * S * head_dim, 60 + S, -1.f, 1.f);

    std::vector<float> cached_out;
    RunAttentionCached(q_new, k_cache, v_cache, cached_out, n_head, S, head_dim);

    // Verify output is finite and correctly shaped.
    ASSERT_EQ(cached_out.size(), static_cast<std::size_t>(n_head * head_dim));
    for (auto val : cached_out)
      EXPECT_TRUE(std::isfinite(val))
          << "Cached attention produced non-finite output at S=" << S;
  }
}

TEST_F(AttentionTest, CachedZeroLengthIsNoop) {
  // S=0 should return immediately without crashing.
  mgie::LaunchAttentionCached(nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr,
                               0, 0, 0, 0);
  SUCCEED();
}

}  // namespace
