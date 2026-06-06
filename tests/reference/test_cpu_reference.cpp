// tests/reference/test_cpu_reference.cpp
//
// Host-only tests for the CPU reference kernels and tolerance utilities.
//
// These need no GPU, so they run on GitHub-hosted CI runners (labeled "cpu").
// They guard the golden implementations and the AllClose/MaxAbsError helpers
// that every GPU test relies on — if the reference or the comparison logic is
// wrong, every downstream validation is meaningless.

#include <gtest/gtest.h>

#include <vector>

#include "common/test_utils.hpp"

namespace {

namespace mt = mgie::test;

TEST(CpuReference, VectorAddIsElementwiseSum) {
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> b{10.0f, 20.0f, 30.0f};
  const std::vector<float> expected{11.0f, 22.0f, 33.0f};
  EXPECT_TRUE(mt::AllClose(mt::CpuVectorAdd(a, b), expected));
}

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

}  // namespace
