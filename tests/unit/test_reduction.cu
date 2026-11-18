// tests/unit/test_reduction.cu
//
// GPU unit tests for the Phase 1 sum-reduction kernels.
//
// Validates each implemented ReductionVersion against the double-precision CPU
// reference (mt::CpuReduce) using the relative-error framework
// (mt::MaxRelError / mt::ReductionRelTol). Requires a CUDA device (labeled
// "gpu" in CMake).
//
// The version list `kVersions` grows as versions are implemented: Naive first,
// then SharedMem, then WarpShuffle. Every test runs over all listed versions so
// each is validated when it is added.

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/reduction.hpp"

namespace {

using mgie::DeviceBuffer;
using mgie::LaunchReduce;
using mgie::ReductionVersion;
using mgie::ReductionWorkspace;
namespace mt = mgie::test;

// Implemented versions under test. Append SharedMem / WarpShuffle as they land.
struct NamedVersion {
  ReductionVersion version;
  const char* name;
};
const std::vector<NamedVersion> kVersions = {
    {ReductionVersion::Naive, "Naive"},
    {ReductionVersion::SharedMem, "SharedMem"},
    {ReductionVersion::WarpShuffle, "WarpShuffle"},
};

bool CudaDeviceAvailable() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

class ReductionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaDeviceAvailable()) {
      GTEST_SKIP() << "no CUDA device available";
    }
  }
};

// Run one version over host input `h_in`, returning the device sum as double.
double RunReduce(ReductionVersion version, const std::vector<float>& h_in) {
  const std::size_t n = h_in.size();
  DeviceBuffer<float> d_in(n), d_out(1);
  d_in.CopyFromHost(h_in.data(), n);

  ReductionWorkspace ws;
  LaunchReduce(d_in.data(), d_out.data(), n, ws, version);
  CUDA_CHECK_LAST_SYNC();

  float h_out = 0.0f;
  d_out.CopyToHost(&h_out, 1);
  return static_cast<double>(h_out);
}

// Validate every implemented version against CpuReduce within ReductionRelTol.
void ExpectAllVersionsMatch(const std::vector<float>& h_in) {
  const double ref = mt::CpuReduce(h_in);
  const double tol = mt::ReductionRelTol(h_in.size());
  for (const auto& nv : kVersions) {
    const double gpu = RunReduce(nv.version, h_in);
    EXPECT_LE(mt::MaxRelError(gpu, ref), tol)
        << "version=" << nv.name << " n=" << h_in.size()
        << " gpu=" << gpu << " ref=" << ref;
  }
}

// Deterministic pseudo-random input in [0, 1).
std::vector<float> MakeInput(std::size_t n, unsigned seed = 1234u) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

// ---- Edge cases ------------------------------------------------------------

TEST_F(ReductionTest, SmallKnown) {
  const std::vector<float> h_in{1.0f, 2.0f, 3.0f, 4.0f};
  for (const auto& nv : kVersions) {
    EXPECT_NEAR(RunReduce(nv.version, h_in), 10.0, 1e-3) << nv.name;
  }
}

TEST_F(ReductionTest, SingleElement) {
  const std::vector<float> h_in{3.14159f};
  for (const auto& nv : kVersions) {
    EXPECT_NEAR(RunReduce(nv.version, h_in), 3.14159, 1e-4) << nv.name;
  }
}

TEST_F(ReductionTest, NonPowerOfTwo) {
  ExpectAllVersionsMatch(MakeInput(1001));
}

// ---- Scale sweep -----------------------------------------------------------

TEST_F(ReductionTest, Scale1K)  { ExpectAllVersionsMatch(MakeInput(1024)); }
TEST_F(ReductionTest, Scale64K) { ExpectAllVersionsMatch(MakeInput(65536)); }
TEST_F(ReductionTest, Scale1M)  { ExpectAllVersionsMatch(MakeInput(1u << 20)); }
TEST_F(ReductionTest, Scale16M) { ExpectAllVersionsMatch(MakeInput(1u << 24)); }

// ---- Cross-version agreement -----------------------------------------------

TEST_F(ReductionTest, AllVersionsAgree) {
  const std::vector<float> h_in = MakeInput(1u << 20);
  std::vector<double> results;
  for (const auto& nv : kVersions) {
    results.push_back(RunReduce(nv.version, h_in));
  }
  for (std::size_t i = 1; i < results.size(); ++i) {
    EXPECT_NEAR(results[i], results[0], 1.0)
        << "versions disagree: " << kVersions[i].name << " vs "
        << kVersions[0].name;
  }
}

}  // namespace
