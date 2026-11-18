// tests/unit/test_layernorm.cu
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/layernorm.hpp"

namespace {
using mgie::DeviceBuffer; using mgie::LaunchLayerNorm; using mgie::LayerNormVersion;
namespace mt = mgie::test;

bool CudaAvail() { int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0; }
class LayerNormTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP()<<"no CUDA"; }
};

std::vector<float> RunLN(LayerNormVersion v,
                          const std::vector<float>& x,
                          const std::vector<float>& g,
                          const std::vector<float>& b,
                          int N, int H, float eps=1e-5f) {
  DeviceBuffer<float> dx(N*H), dout(N*H), dg(H), db(H);
  dx.CopyFromHost(x.data(),N*H); dg.CopyFromHost(g.data(),H); db.CopyFromHost(b.data(),H);
  LaunchLayerNorm(dx.data(),dout.data(),dg.data(),db.data(),N,H,eps,v);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> out(N*H);
  dout.CopyToHost(out.data(),N*H);
  return out;
}

std::vector<float> Rand(int n, float lo=-1.f, float hi=1.f, unsigned s=42) {
  std::mt19937 rng(s); std::uniform_real_distribution<float> d(lo,hi);
  std::vector<float> v(n); for (auto& x:v) x=d(rng); return v;
}

void ExpectMatch(LayerNormVersion v,
                 const std::vector<float>& x,
                 const std::vector<float>& g,
                 const std::vector<float>& b,
                 int N, int H) {
  const auto ref = mt::CpuLayerNorm(x,g,b,N,H);
  const auto gpu = RunLN(v,x,g,b,N,H);
  const float tol = static_cast<float>(mt::LayerNormAbsTol(H)) * 10.f; // 10x safety
  EXPECT_LE(mt::MaxAbsError(gpu,ref), tol) << "version=" << (int)v << " N="<<N<<" H="<<H;
}

const std::vector<std::pair<LayerNormVersion,const char*>> kVersions = {
  {LayerNormVersion::Naive,"Naive"},{LayerNormVersion::Welford,"Welford"}
};

TEST_F(LayerNormTest, SmallKnown) {
  // uniform row → output ≈ 0 (gamma=1, beta=0)
  std::vector<float> x{2,2,2,2}, g{1,1,1,1}, b{0,0,0,0};
  for (auto [v,_] : kVersions) {
    auto out = RunLN(v,x,g,b,1,4);
    for (float val : out) EXPECT_NEAR(val,0.f,1e-4f);
  }
}

TEST_F(LayerNormTest, GPT2Hidden) {
  int N=64, H=768;
  auto x=Rand(N*H), g=Rand(H,0.5f,1.5f,1), b=Rand(H,-0.1f,0.1f,2);
  for (auto [v,_] : kVersions) ExpectMatch(v,x,g,b,N,H);
}

TEST_F(LayerNormTest, GPT2FFN) {
  int N=32, H=3072;
  auto x=Rand(N*H), g=Rand(H,0.5f,1.5f,1), b=Rand(H,-0.1f,0.1f,2);
  for (auto [v,_] : kVersions) ExpectMatch(v,x,g,b,N,H);
}

TEST_F(LayerNormTest, NearConstantRow) {
  int N=16, H=768;
  std::vector<float> x(N*H,1.f);
  // add tiny perturbations so variance > 0
  for (int i=0;i<N*H;++i) x[i]+=1e-3f*(i%7-3);
  auto g=std::vector<float>(H,1.f), b=std::vector<float>(H,0.f);
  for (auto [v,_] : kVersions) ExpectMatch(v,x,g,b,N,H);
}

TEST_F(LayerNormTest, AllVersionsAgree) {
  int N=32, H=768;
  auto x=Rand(N*H), g=Rand(H,0.5f,1.5f,1), b=Rand(H,-0.1f,0.1f,2);
  auto a = RunLN(LayerNormVersion::Naive,  x,g,b,N,H);
  auto c = RunLN(LayerNormVersion::Welford,x,g,b,N,H);
  EXPECT_LE(mt::MaxAbsError(a,c), 1e-4f);
}

} // namespace
