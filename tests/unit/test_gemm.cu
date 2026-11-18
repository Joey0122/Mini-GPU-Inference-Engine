// tests/unit/test_gemm.cu
//
// GPU unit tests for Phase 2 GEMM kernels.
// Validates each GemmVersion against CpuGemm (double-precision oracle).
// Also cross-checks against cuBLAS for sanity at representative sizes.
// Requires a CUDA device (labeled "gpu").

#include <gtest/gtest.h>
#include <cublas_v2.h>

#include <random>
#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/gemm.hpp"

namespace {

using mgie::DeviceBuffer;
using mgie::GemmVersion;
using mgie::LaunchGemm;
using mgie::LaunchGemv;
namespace mt = mgie::test;

// Versions under test — grow this list as each version is implemented.
struct NamedVersion { GemmVersion version; const char* name; };
const std::vector<NamedVersion> kVersions = {
    {GemmVersion::Naive,                 "Naive"},
    {GemmVersion::TiledShared,           "TiledShared"},
    {GemmVersion::RegisterTiled,         "RegisterTiled"},
    {GemmVersion::DoubleBuf,             "DoubleBuf"},
    {GemmVersion::RegTiled128,           "RegTiled128"},
    {GemmVersion::RegTiled128Pad,        "RegTiled128Pad"},
    {GemmVersion::RegTiled128PadUnroll,  "RegTiled128PadUnroll"},
};

bool CudaDeviceAvailable() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

class GemmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaDeviceAvailable()) GTEST_SKIP() << "no CUDA device";
  }
};

// Run one version; return GPU result as flat float vector.
std::vector<float> RunGemm(GemmVersion version,
                            const std::vector<float>& hA,
                            const std::vector<float>& hB,
                            int M, int N, int K) {
  DeviceBuffer<float> dA(M * K), dB(K * N), dC(M * N);
  dA.CopyFromHost(hA.data(), M * K);
  dB.CopyFromHost(hB.data(), K * N);
  LaunchGemm(dA.data(), dB.data(), dC.data(), M, N, K, version);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> hC(M * N);
  dC.CopyToHost(hC.data(), M * N);
  return hC;
}

// Validate all current versions against CpuGemm; report max_abs and max_rel.
void ExpectAllVersionsMatch(const std::vector<float>& hA,
                             const std::vector<float>& hB,
                             int M, int N, int K) {
  const auto ref = mt::CpuGemm(hA, hB, M, N, K);
  const double tol = mt::GemmAbsTol(K);
  for (const auto& nv : kVersions) {
    const auto gpu = RunGemm(nv.version, hA, hB, M, N, K);
    const auto err = mt::GemmError(gpu, ref);
    EXPECT_LE(err.max_abs, tol)
        << nv.name << " M=" << M << " N=" << N << " K=" << K
        << " max_abs=" << err.max_abs << " max_rel=" << err.max_rel;
  }
}

std::vector<float> RandMatrix(int rows, int cols, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(rows * cols);
  for (auto& x : v) x = dist(rng);
  return v;
}

// ---- Correctness tests -----------------------------------------------------

TEST_F(GemmTest, SmallKnown) {
  // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
  const std::vector<float> A{1,2,3,4}, B{5,6,7,8};
  for (const auto& nv : kVersions) {
    const auto C = RunGemm(nv.version, A, B, 2, 2, 2);
    EXPECT_NEAR(C[0], 19.f, 1e-4f) << nv.name;
    EXPECT_NEAR(C[1], 22.f, 1e-4f) << nv.name;
    EXPECT_NEAR(C[2], 43.f, 1e-4f) << nv.name;
    EXPECT_NEAR(C[3], 50.f, 1e-4f) << nv.name;
  }
}

TEST_F(GemmTest, Square64)   { ExpectAllVersionsMatch(RandMatrix(64,64),   RandMatrix(64,64,1),   64,  64,  64); }
TEST_F(GemmTest, Square512)  { ExpectAllVersionsMatch(RandMatrix(512,512),  RandMatrix(512,512,1), 512, 512, 512); }
TEST_F(GemmTest, Square1024) { ExpectAllVersionsMatch(RandMatrix(1024,1024),RandMatrix(1024,1024,1),1024,1024,1024); }
TEST_F(GemmTest, NonSquareTall) { ExpectAllVersionsMatch(RandMatrix(512,512),RandMatrix(512,64,1), 512, 64, 512); }
TEST_F(GemmTest, NonSquareWide) { ExpectAllVersionsMatch(RandMatrix(64,512), RandMatrix(512,512,1),64, 512, 512); }
TEST_F(GemmTest, GPT2Proj)   { ExpectAllVersionsMatch(RandMatrix(768,768),  RandMatrix(768,768,1), 768, 768, 768); }

// LM-head GEMM boundary: N=50257 is not a multiple of 128. The last tile has
// only 81 valid columns (padding: 47). This test exercises the N-boundary path
// with a smaller representative N=257 (1 full tile of 128 + 1 partial of 129).
TEST_F(GemmTest, NonMultipleOf128N) {
  ExpectAllVersionsMatch(RandMatrix(64, 64, 30), RandMatrix(64, 257, 31), 64, 257, 64);
}

TEST_F(GemmTest, AllVersionsAgree) {
  const auto A = RandMatrix(512, 512);
  const auto B = RandMatrix(512, 512, 1);
  std::vector<std::vector<float>> results;
  for (const auto& nv : kVersions)
    results.push_back(RunGemm(nv.version, A, B, 512, 512, 512));
  for (std::size_t i = 1; i < results.size(); ++i) {
    const auto err = mt::GemmError(results[i],
        std::vector<double>(results[0].begin(), results[0].end()));
    EXPECT_LE(err.max_abs, 1.0) << kVersions[i].name << " vs " << kVersions[0].name;
  }
}

// ---- cuBLAS sanity checks --------------------------------------------------
// Secondary check: max_abs and max_rel vs cuBLAS (not the primary oracle).
// A loose absolute bound of 1.0 catches systematic errors; true validation
// is always against CpuGemm.

void CublasGemm(const float* dA, const float* dB, float* dC, int M, int N, int K) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f, beta = 0.0f;
  // cuBLAS is column-major; for row-major A*B we call B^T * A^T = C^T,
  // which is: cublasSgemm(N, M, K, B, N, A, K, C, N) with the standard trick.
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  cudaDeviceSynchronize();
  cublasDestroy(handle);
}

void ExpectCublasAgreement(int M, int N, int K, unsigned seed = 7) {
  const auto hA = RandMatrix(M, K, seed);
  const auto hB = RandMatrix(K, N, seed + 1);
  DeviceBuffer<float> dA(M*K), dB(K*N), dCref(M*N);
  dA.CopyFromHost(hA.data(), M*K);
  dB.CopyFromHost(hB.data(), K*N);
  CublasGemm(dA.data(), dB.data(), dCref.data(), M, N, K);
  std::vector<float> hCref(M*N);
  dCref.CopyToHost(hCref.data(), M*N);

  for (const auto& nv : kVersions) {
    const auto hCgpu = RunGemm(nv.version, hA, hB, M, N, K);
    const auto err = mt::GemmError(hCgpu,
        std::vector<double>(hCref.begin(), hCref.end()));
    EXPECT_LE(err.max_abs, 1.0)
        << nv.name << " vs cuBLAS M=" << M << " max_abs=" << err.max_abs
        << " max_rel=" << err.max_rel;
  }
}

TEST_F(GemmTest, CuBlasAgreement512) { ExpectCublasAgreement(512, 512, 512); }
TEST_F(GemmTest, CuBlasAgreement768) { ExpectCublasAgreement(768, 768, 768); }

// ---- Phase 6b: GEMV decode-shape correctness tests -------------------------
// Validate M=1 GEMV dispatch against CpuGemm for all 5 decode GEMM shapes.

struct DecodeShape { const char* name; int N, K; };
const DecodeShape kDecodeShapes[] = {
  {"c_attn",   2304,  768},
  {"c_proj",    768,  768},
  {"mlp_fc",   3072,  768},
  {"mlp_proj",  768, 3072},
  {"lm_head", 50257,  768},
};

TEST_F(GemmTest, GemvDecodeShapesMatchCpu) {
  for (const auto& s : kDecodeShapes) {
    const auto hA = RandMatrix(1, s.K);   // M=1 → vector of length K
    const auto hB = RandMatrix(s.K, s.N, static_cast<unsigned>(s.N));
    const auto ref = mt::CpuGemm(hA, hB, 1, s.N, s.K);
    // LaunchGemm with M=1 → dispatches to GEMV
    const auto gpu = RunGemm(GemmVersion::RegTiled128Pad, hA, hB, 1, s.N, s.K);
    const auto err = mt::GemmError(gpu, ref);
    const double tol = mt::GemmAbsTol(s.K);
    EXPECT_LE(err.max_abs, tol)
        << s.name << " N=" << s.N << " K=" << s.K
        << " max_abs=" << err.max_abs << " max_rel=" << err.max_rel;
  }
}

// M=1 dispatch ignores version: all versions produce identical output.
TEST_F(GemmTest, GemvDispatchIgnoresVersion) {
  const auto hA = RandMatrix(1, 768);
  const auto hB = RandMatrix(768, 3072, 1);
  std::vector<std::vector<float>> results;
  for (const auto& nv : kVersions)
    results.push_back(RunGemm(nv.version, hA, hB, 1, 3072, 768));
  for (std::size_t i = 1; i < results.size(); ++i) {
    const auto err = mt::GemmError(results[i],
        std::vector<double>(results[0].begin(), results[0].end()));
    EXPECT_LE(err.max_abs, 1e-6)
        << kVersions[i].name << " vs " << kVersions[0].name
        << " — all M=1 calls should dispatch to GEMV";
  }
}

// Direct LaunchGemv validation against CpuGemm.
TEST_F(GemmTest, LaunchGemvDirect) {
  for (const auto& s : kDecodeShapes) {
    const auto hA = RandMatrix(1, s.K);
    const auto hB = RandMatrix(s.K, s.N, static_cast<unsigned>(s.N + 1));
    const auto ref = mt::CpuGemm(hA, hB, 1, s.N, s.K);

    DeviceBuffer<float> dA(static_cast<std::size_t>(s.K)),
                         dB(static_cast<std::size_t>(s.K) * s.N),
                         dC(static_cast<std::size_t>(s.N));
    dA.CopyFromHost(hA.data(), s.K);
    dB.CopyFromHost(hB.data(), static_cast<std::size_t>(s.K) * s.N);
    LaunchGemv(dA.data(), dB.data(), dC.data(), s.N, s.K);
    CUDA_CHECK_LAST_SYNC();

    std::vector<float> hC(s.N);
    dC.CopyToHost(hC.data(), s.N);
    const auto err = mt::GemmError(hC, ref);
    const double tol = mt::GemmAbsTol(s.K);
    EXPECT_LE(err.max_abs, tol)
        << "LaunchGemv " << s.name << " N=" << s.N << " K=" << s.K
        << " max_abs=" << err.max_abs << " max_rel=" << err.max_rel;
  }
}

// GEMV-vs-cuBLAS SGEMV at decode shapes (sanity check).
// cuBLAS may be unavailable in some environments; skip on failure.
TEST_F(GemmTest, GemvCuBlasAgreement) {
  cublasHandle_t handle = nullptr;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
    GTEST_SKIP() << "cuBLAS not available (cublasCreate failed)";
    return;
  }

  for (const auto& s : kDecodeShapes) {
    const auto hA = RandMatrix(1, s.K, static_cast<unsigned>(s.N * 2));
    const auto hB = RandMatrix(s.K, s.N, static_cast<unsigned>(s.N * 2 + 1));

    // Our GEMV
    DeviceBuffer<float> dA(s.K), dB(static_cast<std::size_t>(s.K) * s.N),
                         dC_gemv(s.N);
    dA.CopyFromHost(hA.data(), s.K);
    dB.CopyFromHost(hB.data(), static_cast<std::size_t>(s.K) * s.N);
    LaunchGemv(dA.data(), dB.data(), dC_gemv.data(), s.N, s.K);
    CUDA_CHECK_LAST_SYNC();

    // cuBLAS SGEMV: y = alpha * op(A) * x + beta * y
    // B is K×N row-major → interpreted as N×K column-major (lda=N):
    //   A_col[n,k] at n + k*N = B_row[k,n]
    // Then c = A * a (no transpose) gives c[n] = sum_k B[k,n] * a[k].
    DeviceBuffer<float> dC_cublas(s.N);
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N,
                s.N, s.K, &alpha, dB.data(), s.N, dA.data(), 1, &beta,
                dC_cublas.data(), 1);
    cudaDeviceSynchronize();

    std::vector<float> hC_gemv(s.N), hC_cublas(s.N);
    dC_gemv.CopyToHost(hC_gemv.data(), s.N);
    dC_cublas.CopyToHost(hC_cublas.data(), s.N);

    const auto err = mt::GemmError(hC_gemv,
        std::vector<double>(hC_cublas.begin(), hC_cublas.end()));
    EXPECT_LE(err.max_abs, 1.0)
        << "GEMV vs cuBLAS " << s.name << " N=" << s.N << " K=" << s.K
        << " max_abs=" << err.max_abs << " max_rel=" << err.max_rel;
  }

  cublasDestroy(handle);
}

}  // namespace
