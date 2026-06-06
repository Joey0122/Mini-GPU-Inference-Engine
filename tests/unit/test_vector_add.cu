// tests/unit/test_vector_add.cu
//
// GPU unit test for the Phase 0 vector_add kernel.
//
// Validates the kernel output against the CPU reference within the FP32
// tolerance from PROJECT_SPEC. This is the template every later kernel's unit
// test follows: allocate, launch via the public wrapper, copy back, diff
// against reference. Requires a CUDA device (labeled "gpu" in CMake).

#include <gtest/gtest.h>

#include <vector>

#include "common/errors.hpp"
#include "cuda/memory.hpp"
#include "kernels/vector_add.hpp"
#include "common/test_utils.hpp"

namespace {

using mgie::DeviceBuffer;
using mgie::LaunchVectorAdd;
namespace mt = mgie::test;

// Skip GPU tests gracefully when no device is present, so the binary can still
// run (and report SKIP) on a GPU-less machine.
bool CudaDeviceAvailable() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

class VectorAddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaDeviceAvailable()) {
      GTEST_SKIP() << "no CUDA device available";
    }
  }
};

TEST_F(VectorAddTest, MatchesCpuReference) {
  const std::size_t n = 1 << 16;  // 65,536 elements
  std::vector<float> h_a(n), h_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 97) * 0.5f;
    h_b[i] = static_cast<float>(i % 13) * 1.25f;
  }

  DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.CopyFromHost(h_a.data(), n);
  d_b.CopyFromHost(h_b.data(), n);

  LaunchVectorAdd(d_a.data(), d_b.data(), d_c.data(), n);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> h_c(n);
  d_c.CopyToHost(h_c.data(), n);

  const std::vector<float> expected = mt::CpuVectorAdd(h_a, h_b);
  EXPECT_LE(mt::MaxAbsError(h_c, expected), mt::kFp32AbsTol);
}

TEST_F(VectorAddTest, HandlesNonMultipleOfBlockSize) {
  const std::size_t n = 1000;  // not a multiple of 256
  std::vector<float> h_a(n, 1.5f), h_b(n, 2.5f);

  DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.CopyFromHost(h_a.data(), n);
  d_b.CopyFromHost(h_b.data(), n);

  LaunchVectorAdd(d_a.data(), d_b.data(), d_c.data(), n);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> h_c(n);
  d_c.CopyToHost(h_c.data(), n);

  for (float v : h_c) {
    EXPECT_NEAR(v, 4.0f, mt::kFp32AbsTol);
  }
}

}  // namespace
