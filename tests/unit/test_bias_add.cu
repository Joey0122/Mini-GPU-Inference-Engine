// tests/unit/test_bias_add.cu
//
// GPU unit tests for the Phase 5 C5 BiasAdd kernel.
//
// Validates LaunchBiasAdd against the double-precision CPU reference
// (mgie::test::CpuBiasAdd) within the spec tolerance (FP32 max_abs_error < 1e-4).

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/bias_add.hpp"

namespace {

using mgie::DeviceBuffer;
namespace mt = mgie::test;

bool CudaAvail() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }

class BiasAddTest : public ::testing::Test {
  void SetUp() override { if (!CudaAvail()) GTEST_SKIP() << "no CUDA"; }
};

std::vector<float> Rand(int n, unsigned seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

void RunBiasAdd(const std::vector<float>& x,
                const std::vector<float>& bias,
                std::vector<float>& out,
                int N, int D) {
  const int xn = N * D;
  DeviceBuffer<float> dx(xn), db(D), dout(xn);
  dx.CopyFromHost(x.data(), xn);
  db.CopyFromHost(bias.data(), D);
  mgie::LaunchBiasAdd(dx.data(), db.data(), dout.data(), N, D);
  CUDA_CHECK_LAST_SYNC();
  out.resize(xn);
  dout.CopyToHost(out.data(), xn);
}

// ---- CPU reference tests (synthetic data) -----------------------------------

TEST_F(BiasAddTest, SingleRowMatchesCpu) {
  const int N = 1, D = 768;
  const auto x = Rand(N * D, 1);
  const auto bias = Rand(D, 2);
  std::vector<float> gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  const auto ref = mt::CpuBiasAdd(x, bias, N, D);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(BiasAddTest, Gpt2SequenceMatchesCpu) {
  // Typical GPT-2 projection bias: N=S=256, D=n_embd=768
  const int N = 256, D = 768;
  const auto x = Rand(N * D, 3);
  const auto bias = Rand(D, 4);
  std::vector<float> gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  const auto ref = mt::CpuBiasAdd(x, bias, N, D);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(BiasAddTest, CAttnShapeMatchesCpu) {
  // c_attn bias: N=S=256, D=3*n_embd=2304
  const int N = 256, D = 2304;
  const auto x = Rand(N * D, 5);
  const auto bias = Rand(D, 6);
  std::vector<float> gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  const auto ref = mt::CpuBiasAdd(x, bias, N, D);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(BiasAddTest, MlpFcShapeMatchesCpu) {
  // mlp_fc bias: N=S=256, D=4*n_embd=3072
  const int N = 256, D = 3072;
  const auto x = Rand(N * D, 7);
  const auto bias = Rand(D, 8);
  std::vector<float> gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  const auto ref = mt::CpuBiasAdd(x, bias, N, D);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(BiasAddTest, FullContextMatchesCpu) {
  // Full context length: N=S=1024, D=768
  const int N = 1024, D = 768;
  const auto x = Rand(N * D, 9);
  const auto bias = Rand(D, 10);
  std::vector<float> gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  const auto ref = mt::CpuBiasAdd(x, bias, N, D);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(BiasAddTest, StridedThreadsWorkForSmallDim) {
  // D=32 < kBlockDim=256 — exercises boundary condition where D < block size.
  const int N = 8, D = 32;
  const auto x = Rand(N * D, 11);
  const auto bias = Rand(D, 12);
  std::vector<float> gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  const auto ref = mt::CpuBiasAdd(x, bias, N, D);
  EXPECT_LT(mt::MaxAbsError(gpu, ref), mt::kFp32AbsTol);
}

TEST_F(BiasAddTest, ZeroLengthIsNoop) {
  // N=0: kernel returns immediately.
  const int N = 0, D = 768;
  std::vector<float> x, bias(D, 1.0f), gpu;
  RunBiasAdd(x, bias, gpu, N, D);
  SUCCEED();
}

}  // namespace
