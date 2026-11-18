// tests/reference/test_cpu_reference.cpp
//
// Host-only tests for CPU reference kernels and tolerance utilities.
// No GPU required — labeled "cpu" so CI can run these on GPU-less runners.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common/test_utils.hpp"

namespace {

namespace mt = mgie::test;

// ---- VectorAdd reference ---------------------------------------------------

TEST(CpuReference, VectorAddIsElementwiseSum) {
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> b{10.0f, 20.0f, 30.0f};
  const std::vector<float> expected{11.0f, 22.0f, 33.0f};
  EXPECT_TRUE(mt::AllClose(mt::CpuVectorAdd(a, b), expected));
}

// ---- Tolerance utilities ---------------------------------------------------

TEST(ToleranceUtils, MaxAbsErrorFindsLargestGap) {
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> b{1.0f, 2.5f, 3.0f};
  EXPECT_NEAR(mt::MaxAbsError(a, b), 0.5f, 1e-7f);
}

TEST(ToleranceUtils, AllCloseRespectsTolerance) {
  const std::vector<float> a{1.0f, 1.0f};
  const std::vector<float> b{1.0f, 1.0f + 5e-5f};
  EXPECT_TRUE(mt::AllClose(a, b, mt::kFp32AbsTol));   // 5e-5 < 1e-4
  EXPECT_FALSE(mt::AllClose(a, b, 1e-5f));            // 5e-5 > 1e-5
}

TEST(ToleranceUtils, AllCloseRejectsSizeMismatch) {
  const std::vector<float> a{1.0f, 2.0f};
  const std::vector<float> b{1.0f};
  EXPECT_FALSE(mt::AllClose(a, b));
}

// ---- CpuReduce reference ---------------------------------------------------

TEST(ReductionCpu, KnownSmall) {
  const std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_DOUBLE_EQ(mt::CpuReduce(a), 10.0);
}

TEST(ReductionCpu, AllZeros) {
  const std::vector<float> a(1024, 0.0f);
  EXPECT_DOUBLE_EQ(mt::CpuReduce(a), 0.0);
}

TEST(ReductionCpu, SingleElement) {
  const std::vector<float> a{3.14f};
  EXPECT_NEAR(mt::CpuReduce(a), 3.14, 1e-6);
}

TEST(ReductionCpu, LargeAllOnes) {
  const std::vector<float> a(1 << 20, 1.0f);  // 1M elements
  EXPECT_DOUBLE_EQ(mt::CpuReduce(a), static_cast<double>(1 << 20));
}

// ---- MaxRelError + ReductionRelTol -----------------------------------------

TEST(RelErrorUtils, MaxRelErrorZeroRef) {
  // When ref == 0, denominator clamps to eps (1e-8): result = |gpu| / eps.
  // MaxRelError(1e-10, 0.0) = 1e-10 / 1e-8 = 0.01
  EXPECT_NEAR(mt::MaxRelError(1e-10, 0.0), 0.01, 1e-10);
  // A perfect zero GPU result against a zero ref returns 0.
  EXPECT_DOUBLE_EQ(mt::MaxRelError(0.0, 0.0), 0.0);
}

TEST(RelErrorUtils, MaxRelErrorExact) {
  EXPECT_DOUBLE_EQ(mt::MaxRelError(5.0, 5.0), 0.0);
}

TEST(RelErrorUtils, ReductionRelTolScalesWithN) {
  // Tolerance must grow with n (log2 scaling).
  EXPECT_LT(mt::ReductionRelTol(1024), mt::ReductionRelTol(1 << 20));
  // Single element: near-exact.
  EXPECT_LE(mt::ReductionRelTol(1), 1e-6);
}

// ---- CpuGemm reference -----------------------------------------------------

TEST(GemmCpu, KnownSmall) {
  // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
  const std::vector<float> A{1, 2, 3, 4};
  const std::vector<float> B{5, 6, 7, 8};
  const auto C = mt::CpuGemm(A, B, 2, 2, 2);
  EXPECT_DOUBLE_EQ(C[0], 19.0);
  EXPECT_DOUBLE_EQ(C[1], 22.0);
  EXPECT_DOUBLE_EQ(C[2], 43.0);
  EXPECT_DOUBLE_EQ(C[3], 50.0);
}

TEST(GemmCpu, SquareIdentity) {
  // A x I = A for 4x4.
  const std::vector<float> A{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
  std::vector<float> I(16, 0.0f);
  for (int i = 0; i < 4; ++i) I[i * 4 + i] = 1.0f;
  const auto C = mt::CpuGemm(A, I, 4, 4, 4);
  for (int i = 0; i < 16; ++i) {
    EXPECT_NEAR(C[i], static_cast<double>(A[i]), 1e-12);
  }
}

TEST(GemmCpu, GemmRelTolScalesWithK) {
  EXPECT_LT(mt::GemmRelTol(64, 64, 64), mt::GemmRelTol(64, 64, 1024));
  EXPECT_NEAR(mt::GemmRelTol(1, 1, 512), 4.0 * 512 * 6e-8, 1e-20);
}

// ---- Phase 3 CPU references ------------------------------------------------

TEST(LayerNormCpu, KnownSmall) {
  // row=[2,2,2,2] mean=2 var=0 → output=0+beta (gamma=1,beta=0 → 0)
  const std::vector<float> x{2,2,2,2};
  const std::vector<float> g{1,1,1,1}, b{0,0,0,0};
  const auto out = mt::CpuLayerNorm(x, g, b, 1, 4);
  for (float v : out) EXPECT_NEAR(v, 0.f, 1e-5f);
}

TEST(LayerNormCpu, AffinePreserved) {
  // gamma=2, beta=1 should scale and shift normalised output
  const std::vector<float> x{1, 3};  // mean=2, var=1
  const std::vector<float> g{2, 2}, b{1, 1};
  const auto out = mt::CpuLayerNorm(x, g, b, 1, 2);
  EXPECT_NEAR(out[0], 2*(-1.f) + 1.f, 1e-4f);  // 2*(-1)+1 = -1
  EXPECT_NEAR(out[1], 2*( 1.f) + 1.f, 1e-4f);  // 2*(+1)+1 =  3
}

TEST(LayerNormCpu, ToleranceScalesWithH) {
  EXPECT_LT(mt::LayerNormAbsTol(64), mt::LayerNormAbsTol(768));
}

TEST(SoftmaxCpu, SumsToOne) {
  const std::vector<float> x{1, 2, 3};
  const auto out = mt::CpuSoftmax(x, 1, 3);
  float s = 0; for (float v : out) s += v;
  EXPECT_NEAR(s, 1.f, 1e-6f);
}

TEST(SoftmaxCpu, NumericalStability) {
  // Shifting all inputs by a constant must not change output.
  const std::vector<float> x{1000, 1001, 1002};
  const std::vector<float> xref{0, 1, 2};
  const auto a = mt::CpuSoftmax(x, 1, 3);
  const auto b = mt::CpuSoftmax(xref, 1, 3);
  EXPECT_NEAR(mt::MaxAbsError(a, b), 0.f, 1e-6f);
}

TEST(GeluCpu, ZeroIsZero)    { EXPECT_NEAR(mt::CpuGelu({0.f})[0], 0.f, 1e-6f); }
TEST(GeluCpu, KnownPositive) { EXPECT_NEAR(mt::CpuGelu({1.f})[0], 0.8411920f, 1e-4f); }
TEST(GeluCpu, KnownNegative) { EXPECT_NEAR(mt::CpuGelu({-1.f})[0], -0.1588080f, 1e-4f); }

// ---- Phase 4 CPU attention reference ---------------------------------------

TEST(AttentionCpu, SingleTokenIsValueProjection) {
  // S=1: only key is itself, softmax over one element = 1, so out = v.
  const std::vector<float> q{0.5f, -0.5f}, k{1.0f, 2.0f}, v{3.0f, 4.0f};
  const auto out = mt::CpuAttention(q, k, v, /*n_head=*/1, /*S=*/1, /*head_dim=*/2);
  EXPECT_NEAR(out[0], 3.0f, 1e-6f);
  EXPECT_NEAR(out[1], 4.0f, 1e-6f);
}

TEST(AttentionCpu, CausalRowZeroAttendsOnlyToSelf) {
  // S=2, head_dim=1. Row 0 may attend only to key 0 -> out[0] = v[0].
  const std::vector<float> q{1.0f, 1.0f};   // [S=2, d=1]
  const std::vector<float> k{0.0f, 5.0f};
  const std::vector<float> v{7.0f, 9.0f};
  const auto out = mt::CpuAttention(q, k, v, 1, 2, 1);
  EXPECT_NEAR(out[0], 7.0f, 1e-6f);          // row 0 = v[0] exactly (causal)
  // Row 1 attends to both; output is a convex combination in [7,9].
  EXPECT_GT(out[1], 7.0f);
  EXPECT_LT(out[1], 9.0f);
}

TEST(AttentionCpu, UniformScoresAverageValues) {
  // q=0 -> all scores 0 -> uniform softmax -> row i = mean of v[0..i].
  const std::vector<float> q{0.0f, 0.0f, 0.0f};   // [3,1]
  const std::vector<float> k{1.0f, 2.0f, 3.0f};
  const std::vector<float> v{2.0f, 4.0f, 6.0f};
  const auto out = mt::CpuAttention(q, k, v, 1, 3, 1);
  EXPECT_NEAR(out[0], 2.0f, 1e-5f);          // mean(2)
  EXPECT_NEAR(out[1], 3.0f, 1e-5f);          // mean(2,4)
  EXPECT_NEAR(out[2], 4.0f, 1e-5f);          // mean(2,4,6)
}

TEST(AttentionCpu, MultiHeadOutputLayoutIsConcatenated) {
  // 2 heads, S=1, head_dim=2 -> out is [1, 4] = concat of each head's v.
  const std::vector<float> q{0,0, 0,0};      // [n_head=2, S=1, d=2]
  const std::vector<float> k{1,1, 1,1};
  const std::vector<float> v{10,11, 20,21};
  const auto out = mt::CpuAttention(q, k, v, 2, 1, 2);
  ASSERT_EQ(out.size(), 4u);
  EXPECT_NEAR(out[0], 10.f, 1e-5f);          // head 0, d0
  EXPECT_NEAR(out[1], 11.f, 1e-5f);          // head 0, d1
  EXPECT_NEAR(out[2], 20.f, 1e-5f);          // head 1, d0 (column offset 2)
  EXPECT_NEAR(out[3], 21.f, 1e-5f);          // head 1, d1
}

TEST(AttentionCpu, ToleranceScalesWithSeqLen) {
  EXPECT_LT(mt::AttentionAbsTol(128), mt::AttentionAbsTol(1024));
}

}  // namespace
