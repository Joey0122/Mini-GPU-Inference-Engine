// tests/unit/test_gelu.cu
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/gelu.hpp"

namespace {
using mgie::DeviceBuffer; using mgie::LaunchGelu;
namespace mt = mgie::test;

bool CudaAvail() { int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0; }
class GeluTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP()<<"no CUDA"; }
};

std::vector<float> RunGelu(const std::vector<float>& x) {
  DeviceBuffer<float> dx(x.size()), dout(x.size());
  dx.CopyFromHost(x.data(),x.size());
  LaunchGelu(dx.data(),dout.data(),(int)x.size());
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> out(x.size());
  dout.CopyToHost(out.data(),x.size());
  return out;
}

std::vector<float> Rand(int n, unsigned s=42) {
  std::mt19937 rng(s); std::uniform_real_distribution<float> d(-3.f,3.f);
  std::vector<float> v(n); for (auto& x:v) x=d(rng); return v;
}

TEST_F(GeluTest, KnownValues) {
  auto out = RunGelu({0.f,1.f,-1.f});
  EXPECT_NEAR(out[0], 0.f,         1e-5f);
  EXPECT_NEAR(out[1], 0.8411920f,  1e-4f);
  EXPECT_NEAR(out[2],-0.1588080f,  1e-4f);
}

TEST_F(GeluTest, MatchesCpuReference) {
  auto x = Rand(1<<16);
  EXPECT_LE(mt::MaxAbsError(RunGelu(x), mt::CpuGelu(x)), 1e-4f);
}

TEST_F(GeluTest, LargeInput) {
  auto x = Rand(1<<20);
  EXPECT_LE(mt::MaxAbsError(RunGelu(x), mt::CpuGelu(x)), 1e-4f);
}

} // namespace
