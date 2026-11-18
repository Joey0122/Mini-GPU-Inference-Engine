// src/kernels/gemm.cu

#include "kernels/gemm.hpp"
#include "cuda/launch.hpp"
#include "common/errors.hpp"

namespace mgie {
namespace {

// ---- Tile constants (original) ----------------------------------------------
constexpr int kTile = 16;
constexpr int kBK   = 16;
constexpr int kBM   = 64;
constexpr int kBN   = 64;
constexpr int kTM   = 4;
constexpr int kTN   = 4;
constexpr int kTPB  = (kBM / kTM) * (kBN / kTN); // 256

// ---- Tile constants (Step 1–3, larger tile) ---------------------------------
constexpr int kBM2  = 128;
constexpr int kBN2  = 128;
constexpr int kTM2  = 8;
constexpr int kTN2  = 8;
constexpr int kTPB2 = (kBM2 / kTM2) * (kBN2 / kTN2); // 256

// ---- Version A: naive -------------------------------------------------------
__global__ void GemmNaiveKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  if (row >= M || col >= N) return;
  float s = 0.f;
  for (int k = 0; k < K; ++k) s += A[row*K+k] * B[k*N+col];
  C[row*N+col] = s;
}

// ---- Version B: shared-memory tiling (TILE=16) ------------------------------
__global__ void GemmTiledKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];
  const int ty = threadIdx.y, tx = threadIdx.x;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;
  float s = 0.f;
  for (int t = 0; t < (K+kTile-1)/kTile; ++t) {
    sA[ty][tx] = (row<M && t*kTile+tx<K) ? A[row*K + t*kTile+tx] : 0.f;
    sB[ty][tx] = (t*kTile+ty<K && col<N) ? B[(t*kTile+ty)*N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < kTile; ++k) s += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  if (row<M && col<N) C[row*N+col] = s;
}

// ---- Version C: register tiling (TM=TN=4, BM=BN=64) ------------------------
__global__ void GemmRegTiledKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
  __shared__ float sA[kBM][kBK];
  __shared__ float sB[kBK][kBN];
  const int tr = threadIdx.y, tc = threadIdx.x;
  const int br = blockIdx.y * kBM, bc = blockIdx.x * kBN;
  const int tid = tr*(kBN/kTN) + tc;
  float acc[kTM][kTN] = {};
  float ra[kTM], rb[kTN];
  for (int t = 0; t < (K+kBK-1)/kBK; ++t) {
    for (int i=tid; i<kBM*kBK; i+=kTPB) {
      int r=i/kBK, c=i%kBK;
      sA[r][c] = (br+r<M && t*kBK+c<K) ? A[(br+r)*K + t*kBK+c] : 0.f;
    }
    for (int i=tid; i<kBK*kBN; i+=kTPB) {
      int r=i/kBN, c=i%kBN;
      sB[r][c] = (t*kBK+r<K && bc+c<N) ? B[(t*kBK+r)*N + bc+c] : 0.f;
    }
    __syncthreads();
    for (int k=0; k<kBK; ++k) {
      for (int m=0; m<kTM; ++m) ra[m] = sA[tr*kTM+m][k];
      for (int n=0; n<kTN; ++n) rb[n] = sB[k][tc*kTN+n];
      for (int m=0; m<kTM; ++m) for (int n=0; n<kTN; ++n) acc[m][n] += ra[m]*rb[n];
    }
    __syncthreads();
  }
  for (int m=0; m<kTM; ++m) for (int n=0; n<kTN; ++n)
    if (br+tr*kTM+m<M && bc+tc*kTN+n<N) C[(br+tr*kTM+m)*N + bc+tc*kTN+n] = acc[m][n];
}

// ---- Version D: double buffering (TM=TN=4, BM=BN=64) -----------------------
__global__ void GemmDoubleBufKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
  __shared__ float sA[2][kBM][kBK];
  __shared__ float sB[2][kBK][kBN];
  const int tr = threadIdx.y, tc = threadIdx.x;
  const int br = blockIdx.y * kBM, bc = blockIdx.x * kBN;
  const int tid = tr*(kBN/kTN) + tc;
  float acc[kTM][kTN] = {};
  float ra[kTM], rb[kTN];
  const int nT = (K+kBK-1)/kBK;

  auto load = [&](int t, int buf) {
    for (int i=tid; i<kBM*kBK; i+=kTPB) {
      int r=i/kBK, c=i%kBK;
      sA[buf][r][c] = (br+r<M && t*kBK+c<K) ? A[(br+r)*K + t*kBK+c] : 0.f;
    }
    for (int i=tid; i<kBK*kBN; i+=kTPB) {
      int r=i/kBN, c=i%kBN;
      sB[buf][r][c] = (t*kBK+r<K && bc+c<N) ? B[(t*kBK+r)*N + bc+c] : 0.f;
    }
  };
  if (nT == 0) return;
  load(0, 0); __syncthreads();
  for (int t=0; t<nT; ++t) {
    const int cur = t&1, nxt = 1-cur;
    if (t+1<nT) load(t+1, nxt);
    for (int k=0; k<kBK; ++k) {
      for (int m=0; m<kTM; ++m) ra[m] = sA[cur][tr*kTM+m][k];
      for (int n=0; n<kTN; ++n) rb[n] = sB[cur][k][tc*kTN+n];
      for (int m=0; m<kTM; ++m) for (int n=0; n<kTN; ++n) acc[m][n] += ra[m]*rb[n];
    }
    __syncthreads();
  }
  for (int m=0; m<kTM; ++m) for (int n=0; n<kTN; ++n)
    if (br+tr*kTM+m<M && bc+tc*kTN+n<N) C[(br+tr*kTM+m)*N + bc+tc*kTN+n] = acc[m][n];
}

// ---- Macro for the 128-tile compute body (avoids triplication) --------------
// Used by Steps 1, 2, 3; only the shared-memory declaration and unroll pragma
// differ between them, so the macro keeps the compute logic identical.
#define REG128_COMPUTE_BODY(sA_, sB_)                                   \
  float acc[kTM2][kTN2] = {};                                           \
  float ra[kTM2], rb[kTN2];                                             \
  const int tr = threadIdx.y, tc = threadIdx.x;                         \
  const int br = blockIdx.y * kBM2, bc = blockIdx.x * kBN2;             \
  const int tid = tr * (kBN2/kTN2) + tc;                                \
  for (int t = 0; t < (K+kBK-1)/kBK; ++t) {                            \
    for (int i=tid; i<kBM2*kBK; i+=kTPB2) {                            \
      int r=i/kBK, c=i%kBK;                                             \
      sA_[r][c] = (br+r<M && t*kBK+c<K) ? A[(br+r)*K + t*kBK+c] : 0.f;\
    }                                                                    \
    for (int i=tid; i<kBK*kBN2; i+=kTPB2) {                            \
      int r=i/kBN2, c=i%kBN2;                                           \
      sB_[r][c] = (t*kBK+r<K && bc+c<N) ? B[(t*kBK+r)*N + bc+c] : 0.f;\
    }                                                                    \
    __syncthreads();                                                     \
    INNER_K_LOOP(sA_, sB_)                                              \
    __syncthreads();                                                     \
  }                                                                      \
  for (int m=0; m<kTM2; ++m) for (int n=0; n<kTN2; ++n)                \
    if (br+tr*kTM2+m<M && bc+tc*kTN2+n<N)                               \
      C[(br+tr*kTM2+m)*N + bc+tc*kTN2+n] = acc[m][n];

// Steps 1 and 2 share the same inner loop (no unroll pragma).
#define INNER_K_LOOP_PLAIN(sA_, sB_)                                    \
  for (int k=0; k<kBK; ++k) {                                           \
    for (int m=0; m<kTM2; ++m) ra[m] = sA_[tr*kTM2+m][k];             \
    for (int n=0; n<kTN2; ++n) rb[n] = sB_[k][tc*kTN2+n];             \
    for (int m=0; m<kTM2; ++m)                                          \
      for (int n=0; n<kTN2; ++n) acc[m][n] += ra[m]*rb[n];             \
  }

// Step 3 adds #pragma unroll.
#define INNER_K_LOOP_UNROLL(sA_, sB_)                                   \
  _Pragma("unroll")                                                      \
  for (int k=0; k<kBK; ++k) {                                           \
    for (int m=0; m<kTM2; ++m) ra[m] = sA_[tr*kTM2+m][k];             \
    for (int n=0; n<kTN2; ++n) rb[n] = sB_[k][tc*kTN2+n];             \
    for (int m=0; m<kTM2; ++m)                                          \
      for (int n=0; n<kTN2; ++n) acc[m][n] += ra[m]*rb[n];             \
  }

// ---- Step 1: BM=BN=128, TM=TN=8, no padding, no unroll --------------------
#define INNER_K_LOOP INNER_K_LOOP_PLAIN
__global__ void GemmReg128Kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
  __shared__ float sA[kBM2][kBK];
  __shared__ float sB[kBK][kBN2];
  REG128_COMPUTE_BODY(sA, sB)
}
#undef INNER_K_LOOP

// ---- Step 2: + padding (sA trailing dim +1, sB trailing dim +1) ------------
#define INNER_K_LOOP INNER_K_LOOP_PLAIN
__global__ void GemmReg128PadKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
  __shared__ float sA[kBM2][kBK + 1];
  __shared__ float sB[kBK][kBN2 + 1];
  REG128_COMPUTE_BODY(sA, sB)
}
#undef INNER_K_LOOP

// ---- Step 3: + #pragma unroll on k-loop ------------------------------------
#define INNER_K_LOOP INNER_K_LOOP_UNROLL
__global__ void GemmReg128PadUnrollKernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
  __shared__ float sA[kBM2][kBK + 1];
  __shared__ float sB[kBK][kBN2 + 1];
  REG128_COMPUTE_BODY(sA, sB)
}
#undef INNER_K_LOOP

#undef REG128_COMPUTE_BODY
#undef INNER_K_LOOP_PLAIN
#undef INNER_K_LOOP_UNROLL

// ---- Dedicated GEMV for M=1 decode -------------------------------
//
// For M=1 the GEMM degenerates to a matrix-vector product c[n] = sum_k a[k]*B[k,n].
// RegTiled128Pad processes 128-row tiles where 127/128 rows produce zero output,
// achieving only 0.78% tile utilization. A dedicated GEMV kernel eliminates this
// waste by mapping one thread per output element.
//
// Grid: ceil(N / 256),  Block: 256 threads
// Dynamic shared memory: K * sizeof(float) to cache the input vector a[K].
// K ≤ 3072 for all per-layer GEMMs → ≤12 KB shared memory, well below 48 KB limit.
// K=768 for lm_head → 3 KB shared memory.

__global__ void GemvKernel(const float* __restrict__ a,   // [K]
                            const float* __restrict__ B,   // [K, N] row-major
                            float* __restrict__ c,          // [N]
                            int N, int K) {
  extern __shared__ float a_shared[];

  // Cooperative load of a into shared memory.
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    a_shared[k] = a[k];
  __syncthreads();

  int n = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += a_shared[k] * B[static_cast<long long>(k) * N + n];

  c[n] = acc;
}

}  // namespace

void LaunchGemv(const float* A, const float* B, float* C,
                int N, int K, cudaStream_t stream) {
  if (N == 0 || K == 0) return;
  const int block = 256;
  const int grid = CeilDiv(static_cast<unsigned int>(N),
                           static_cast<unsigned int>(block));
  const std::size_t smem = static_cast<std::size_t>(K) * sizeof(float);
  GemvKernel<<<grid, block, smem, stream>>>(A, B, C, N, K);
}

void LaunchGemm(const float* A, const float* B, float* C,
                int M, int N, int K,
                GemmVersion version, cudaStream_t stream) {
  if (M==0 || N==0 || K==0) return;

  // M=1 dispatch — route to dedicated GEMV kernel.
  // The tiled GEMM kernels are designed for M ≫ 1 and waste 127/128 rows
  // when M=1 (0.78% tile utilization). GEMV maps one thread per output element
  // and achieves near-peak memory bandwidth for all decode shapes.
  if (M == 1) {
    LaunchGemv(A, B, C, N, K, stream);
    return;
  }

  if (version == GemmVersion::Naive || version == GemmVersion::TiledShared) {
    const dim3 block(kTile, kTile);
    const dim3 grid(CeilDiv((unsigned)N,(unsigned)kTile),
                    CeilDiv((unsigned)M,(unsigned)kTile));
    if (version == GemmVersion::Naive)
      GemmNaiveKernel<<<grid,block,0,stream>>>(A,B,C,M,N,K);
    else
      GemmTiledKernel<<<grid,block,0,stream>>>(A,B,C,M,N,K);
  } else if (version == GemmVersion::RegisterTiled ||
             version == GemmVersion::DoubleBuf) {
    const dim3 block(kBN/kTN, kBM/kTM);
    const dim3 grid(CeilDiv((unsigned)N,(unsigned)kBN),
                    CeilDiv((unsigned)M,(unsigned)kBM));
    if (version == GemmVersion::RegisterTiled)
      GemmRegTiledKernel<<<grid,block,0,stream>>>(A,B,C,M,N,K);
    else
      GemmDoubleBufKernel<<<grid,block,0,stream>>>(A,B,C,M,N,K);
  } else {
    // Steps 1–3: 128-tile kernels, same 256-thread block.
    const dim3 block(kBN2/kTN2, kBM2/kTM2);
    const dim3 grid(CeilDiv((unsigned)N,(unsigned)kBN2),
                    CeilDiv((unsigned)M,(unsigned)kBM2));
    switch (version) {
      case GemmVersion::RegTiled128:
        GemmReg128Kernel<<<grid,block,0,stream>>>(A,B,C,M,N,K); break;
      case GemmVersion::RegTiled128Pad:
        GemmReg128PadKernel<<<grid,block,0,stream>>>(A,B,C,M,N,K); break;
      default:
        GemmReg128PadUnrollKernel<<<grid,block,0,stream>>>(A,B,C,M,N,K); break;
    }
  }
}

}  // namespace mgie
