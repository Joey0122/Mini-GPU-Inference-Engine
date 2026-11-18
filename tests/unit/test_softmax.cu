// tests/unit/test_softmax.cu
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/softmax.hpp"

namespace {
using mgie::DeviceBuffer; using mgie::LaunchSoftmax; using mgie::SoftmaxVersion;
namespace mt = mgie::test;
constexpr float kSmTol = 1e-5f;

bool CudaAvail() { int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0; }
class SoftmaxTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP()<<"no CUDA"; }
};

std::vector<float> RunSM(SoftmaxVersion v,
                          const std::vector<float>& x, int N, int D) {
  DeviceBuffer<float> dx(N*D), dout(N*D);
  dx.CopyFromHost(x.data(),N*D);
  LaunchSoftmax(dx.data(),dout.data(),N,D,v);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> out(N*D);
  dout.CopyToHost(out.data(),N*D);
  return out;
}

std::vector<float> Rand(int n, float lo=-2.f, float hi=2.f, unsigned s=42) {
  std::mt19937 rng(s); std::uniform_real_distribution<float> d(lo,hi);
  std::vector<float> v(n); for (auto& x:v) x=d(rng); return v;
}

const std::vector<std::pair<SoftmaxVersion,const char*>> kVersions = {
  {SoftmaxVersion::Naive,"Naive"},{SoftmaxVersion::Fused,"Fused"}
};

void ExpectMatch(SoftmaxVersion v, const std::vector<float>& x, int N, int D) {
  const auto ref = mt::CpuSoftmax(x,N,D);
  const auto gpu = RunSM(v,x,N,D);
  EXPECT_LE(mt::MaxAbsError(gpu,ref),kSmTol) << "N="<<N<<" D="<<D;
}

TEST_F(SoftmaxTest, SmallKnown) {
  std::vector<float> x{1,2,3};
  for (auto [v,_] : kVersions) {
    auto out = RunSM(v,x,1,3);
    float s=0; for (float f:out) s+=f;
    EXPECT_NEAR(s,1.f,1e-6f);
  }
}

TEST_F(SoftmaxTest, NumericalStability) {
  // Shifting all inputs by constant must not change output.
  std::vector<float> x{1000.f,1001.f,1002.f};
  std::vector<float> xref{0.f,1.f,2.f};
  for (auto [v,_] : kVersions) {
    EXPECT_LE(mt::MaxAbsError(RunSM(v,x,1,3),RunSM(v,xref,1,3)),1e-5f) << _;
  }
}

TEST_F(SoftmaxTest, Seq512)  { ExpectMatch(SoftmaxVersion::Fused,Rand(64*512), 64,512); }
TEST_F(SoftmaxTest, Seq1024) { ExpectMatch(SoftmaxVersion::Fused,Rand(64*1024),64,1024); }
TEST_F(SoftmaxTest, Seq2048) { ExpectMatch(SoftmaxVersion::Fused,Rand(32*2048),32,2048); }
TEST_F(SoftmaxTest, Seq4096) { ExpectMatch(SoftmaxVersion::Fused,Rand(16*4096),16,4096); }

TEST_F(SoftmaxTest, AllVersionsAgree) {
  auto x=Rand(64*512);
  EXPECT_LE(mt::MaxAbsError(RunSM(SoftmaxVersion::Naive,x,64,512),
                            RunSM(SoftmaxVersion::Fused,x,64,512)),1e-5f);
}

} // namespace
