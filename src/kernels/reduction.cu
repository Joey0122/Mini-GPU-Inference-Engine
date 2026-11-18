// src/kernels/reduction.cu
//
// Phase 1 sum-reduction: ReductionWorkspace + two-pass LaunchReduce driver.
//
// See include/kernels/reduction.hpp for the design. Structure shared by all
// versions:
//   Pass 1: in[0..n)            -> partial[0..num_blocks)   (one sum per block)
//   Pass 2: partial[0..blocks)  -> out[0]                   (single block)
//
// Every version uses the SAME grid-stride global load (so each input element is
// read exactly once and the memory-traffic comparison is fair); they differ
// ONLY in the intra-block tree reduction. Versions are added incrementally and
// each is validated before the next (IMPLEMENTATION_RULES Rule 2).
//
// Implemented so far: Version A (Naive, interleaved addressing).

#include "kernels/reduction.hpp"

#include "common/errors.hpp"
#include "cuda/launch.hpp"

namespace mgie {

namespace {

// One block size for the whole reduction. 256 is warp-aligned and occupancy-
// friendly on Ada (CC 8.9); shared-mem footprint is kBlockSize floats.
constexpr int kBlockSize = 256;

// Cap on Pass-1 blocks. Pass 2 reduces the partial array in a SINGLE block of
// kBlockSize threads using the same grid-stride load, so the partial array must
// contain <= kBlockSize elements. A larger cap would require a third pass.
// 256 blocks × 256 threads = 65536 concurrent threads — enough to saturate
// the RTX 4070 Laptop (36 SMs) even at 64M elements, while keeping
// grid-stride work per thread at 64M/(256×256)=1024 elements, similar to
// 16M/(256×256)=256 at the same cap. The actual Pass-1 grid uses
// min(ceil(n/blockSize), kMaxPass1Blocks) blocks.
constexpr unsigned int kMaxPass1Blocks = static_cast<unsigned int>(kBlockSize);

// Grid-stride accumulate: each thread sums its strided slice of in[0..n) into a
// register. Shared by all versions — this is the fair, identical global-load
// path; only the in-block combine that follows differs between versions.
__device__ inline float GridStrideLoad(const float* __restrict__ in,
                                        std::size_t n) {
  float sum = 0.0f;
  std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (; idx < n; idx += stride) {
    sum += in[idx];
  }
  return sum;
}

// ---- Version A: naive interleaved addressing -------------------------------
// Mark Harris "reduction #1" baseline. The `tid % (2*s)` predicate makes
// consecutive threads in a warp diverge, and the strided shared-mem access
// causes bank conflicts. Both are the bottlenecks Versions B/C remove — kept
// intentionally so the benchmark shows why those optimizations matter.
__global__ void ReduceNaiveKernel(const float* __restrict__ in,
                                  float* __restrict__ out, std::size_t n) {
  __shared__ float smem[kBlockSize];
  const unsigned int tid = threadIdx.x;

  smem[tid] = GridStrideLoad(in, n);
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

// ---- Version B: sequential addressing --------------------------------------
// Mark Harris "reduction #3". Two fixes over Version A:
//   1. Sequential addressing (stride halves from blockDim/2): active threads
//      are contiguous, so warps stay converged and shared-mem access is
//      bank-conflict-free.
//   2. The grid-stride load already did the "first add on load" work, so each
//      thread enters the tree with a meaningful partial.
__global__ void ReduceSharedKernel(const float* __restrict__ in,
                                   float* __restrict__ out, std::size_t n) {
  __shared__ float smem[kBlockSize];
  const unsigned int tid = threadIdx.x;

  smem[tid] = GridStrideLoad(in, n);
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = smem[0];
}

// ---- Version C: warp shuffle -----------------------------------------------
// Intra-warp reduction via __shfl_down_sync — no shared-mem round trip for the
// 32-lane step. Each warp reduces to its lane 0, those per-warp sums go through
// one small shared array, and the first warp shuffle-reduces them to the block
// result. Assumes blockDim is a multiple of the warp size and <= 1024 (so the
// number of warps fits in a single warp's shuffle), both true for kBlockSize.
__global__ void ReduceWarpKernel(const float* __restrict__ in,
                                 float* __restrict__ out, std::size_t n) {
  constexpr unsigned int kWarp = 32;
  __shared__ float warp_sums[kBlockSize / kWarp];

  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid % kWarp;
  const unsigned int warp_id = tid / kWarp;
  const unsigned int num_warps = blockDim.x / kWarp;

  float val = GridStrideLoad(in, n);

  // Reduce within each (full) warp.
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }

  // Lane 0 of each warp publishes its partial.
  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  // First warp reduces the per-warp partials.
  if (warp_id == 0) {
    val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (lane == 0) out[blockIdx.x] = val;
  }
}

// Number of Pass-1 blocks for problem size n.
unsigned int Pass1Blocks(std::size_t n) {
  const unsigned int needed =
      CeilDiv(static_cast<unsigned int>(n), static_cast<unsigned int>(kBlockSize));
  return needed < kMaxPass1Blocks ? (needed == 0u ? 1u : needed)
                                  : kMaxPass1Blocks;
}

// Launch the selected version's kernel over `count` elements with `grid` blocks.
void LaunchVersion(ReductionVersion version, const float* in, float* out,
                   std::size_t count, dim3 grid, cudaStream_t stream) {
  const dim3 block(kBlockSize);
  switch (version) {
    case ReductionVersion::Naive:
      ReduceNaiveKernel<<<grid, block, 0, stream>>>(in, out, count);
      break;
    case ReductionVersion::SharedMem:
      ReduceSharedKernel<<<grid, block, 0, stream>>>(in, out, count);
      break;
    case ReductionVersion::WarpShuffle:
      ReduceWarpKernel<<<grid, block, 0, stream>>>(in, out, count);
      break;
    default:
      ReduceNaiveKernel<<<grid, block, 0, stream>>>(in, out, count);
      break;
  }
}

}  // namespace

// ---- ReductionWorkspace ----------------------------------------------------

ReductionWorkspace::~ReductionWorkspace() { Free(); }

ReductionWorkspace::ReductionWorkspace(ReductionWorkspace&& other) noexcept
    : partial_(other.partial_), capacity_(other.capacity_) {
  other.partial_ = nullptr;
  other.capacity_ = 0;
}

ReductionWorkspace& ReductionWorkspace::operator=(
    ReductionWorkspace&& other) noexcept {
  if (this != &other) {
    Free();
    partial_ = other.partial_;
    capacity_ = other.capacity_;
    other.partial_ = nullptr;
    other.capacity_ = 0;
  }
  return *this;
}

void ReductionWorkspace::Reserve(std::size_t num_blocks) {
  if (num_blocks <= capacity_) return;  // already big enough
  Free();
  CUDA_CHECK(cudaMalloc(&partial_, num_blocks * sizeof(float)));
  capacity_ = num_blocks;
}

void ReductionWorkspace::Free() {
  if (partial_ != nullptr) {
    cudaFree(partial_);  // best-effort; do not throw from destructor path
    partial_ = nullptr;
  }
  capacity_ = 0;
}

// ---- Two-pass driver -------------------------------------------------------

void LaunchReduce(const float* d_in, float* d_out, std::size_t n,
                  ReductionWorkspace& ws, ReductionVersion version,
                  cudaStream_t stream) {
  if (n == 0) return;

  const unsigned int blocks = Pass1Blocks(n);

  // Single block already produces the final scalar — skip the second pass.
  if (blocks == 1u) {
    LaunchVersion(version, d_in, d_out, n, dim3(1), stream);
    return;
  }

  ws.Reserve(blocks);

  // Pass 1: in -> partial[blocks].
  LaunchVersion(version, d_in, ws.partial(), n, dim3(blocks), stream);
  // Pass 2: partial[blocks] -> out[0] (single block, grid-stride over blocks).
  LaunchVersion(version, ws.partial(), d_out, blocks, dim3(1), stream);
}

}  // namespace mgie
