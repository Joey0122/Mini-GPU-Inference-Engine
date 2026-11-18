// include/kernels/gemm.hpp
//
// Launch interface for the Phase 2 GEMM kernels.
//
// Computes C = A * B (row-major, single-precision) in four progressive
// versions, each kept for benchmarking (AGENT.md: do not delete versions):
//
//   Naive          - one thread per output element, no reuse
//   TiledShared    - TILE×TILE shared-memory tiling
//   RegisterTiled  - TM×TN register tile per thread, BM×BK shared strips
//   DoubleBuf      - register tiling + manual double-buffer prefetch
//
// All versions use the same public interface; callers never write <<<>>>.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace mgie {

enum class GemmVersion {
  Naive,            // one thread per C element
  TiledShared,      // TILE=16 shared-memory tiling
  RegisterTiled,    // TM=TN=4, BM=BN=64, BK=16 (baseline register tile)
  DoubleBuf,        // RegisterTiled + manual double-buffer prefetch
  RegTiled128,      // TM=TN=8, BM=BN=128, BK=16 (Step 1 optimisation)
  RegTiled128Pad,   // RegTiled128 + sA/sB padding (Step 2)
  RegTiled128PadUnroll, // RegTiled128Pad + #pragma unroll (Step 3)
};

// Computes C = A * B.
// A: M×K row-major device pointer
// B: K×N row-major device pointer
// C: M×N row-major device pointer (output, overwritten)
// All pointers must be valid device memory of the correct size.
// Asynchronous on `stream`; n=0 in any dimension is a no-op.
void LaunchGemm(const float* A, const float* B, float* C,
                int M, int N, int K,
                GemmVersion version = GemmVersion::DoubleBuf,
                cudaStream_t stream = nullptr);

// Phase 6b: Dedicated GEMV for M=1 decode GEMMs.
// Computes c[0:N] = a[0:K] * B[K,N]  (row-vector × matrix).
// A: [K] row-vector device pointer
// B: K×N row-major device pointer
// C: [N] output device pointer (overwritten)
// Asynchronous on `stream`; N=0 or K=0 is a no-op.
void LaunchGemv(const float* A, const float* B, float* C,
                int N, int K,
                cudaStream_t stream = nullptr);

}  // namespace mgie
