// include/kernels/reduction.hpp
//
// Launch interface for the Phase 1 sum-reduction kernel.
//
// Reduction is the first "real" GPU primitive (TASK_BREAKDOWN Phase 1). It
// computes the sum of an array of floats. Three versions are implemented and
// kept side-by-side for benchmarking (AGENT.md: do not delete previous
// versions), all sharing one two-pass structure so comparisons are meaningful:
//
//   Pass 1: input[0..n)        -> partial[0..num_blocks)   (one sum per block)
//   Pass 2: partial[0..blocks) -> out[0]                   (single block)
//
// The ONLY difference between versions is the intra-block reduction strategy:
//   * Naive       - shared-mem, interleaved addressing (Mark Harris baseline;
//                   exhibits warp divergence + bank conflicts).
//   * SharedMem   - shared-mem, sequential addressing + first-add-on-load
//                   (eliminates divergence and bank conflicts).
//   * WarpShuffle - __shfl_down_sync intra-warp reduce (no shared-mem round
//                   trip for the intra-warp step).
//
// The kernel layer exposes ONLY launch wrappers (ARCHITECTURE.md): callers in
// tests/benchmarks include this header and never write <<<...>>> themselves.

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace mgie {

// Selects the intra-block reduction strategy. All versions share the same
// two-pass driver and produce identical results within FP rounding.
enum class ReductionVersion {
  Naive,        // interleaved-address shared-mem (baseline)
  SharedMem,    // sequential-address shared-mem, first-add-on-load
  WarpShuffle,  // warp-shuffle intra-warp reduce
};

// Scratch space for the two-pass reduction's intermediate block partial sums.
//
// Allocated ONCE by the caller (e.g. at benchmark/test setup) and reused across
// launches so that no cudaMalloc occurs inside a timed region (AGENT.md: avoid
// cudaMalloc in hot paths; benchmark timing must exclude allocation overhead).
// Move-only RAII ownership; frees on destruction.
//
// Future kernels needing scratch (softmax, attention) follow this same
// caller-supplied-workspace convention.
class ReductionWorkspace {
 public:
  ReductionWorkspace() = default;
  ~ReductionWorkspace();

  ReductionWorkspace(const ReductionWorkspace&) = delete;
  ReductionWorkspace& operator=(const ReductionWorkspace&) = delete;
  ReductionWorkspace(ReductionWorkspace&& other) noexcept;
  ReductionWorkspace& operator=(ReductionWorkspace&& other) noexcept;

  // Ensure space for at least `num_blocks` partial sums. Reallocates only if
  // the current capacity is too small; a no-op when already large enough.
  void Reserve(std::size_t num_blocks);

  float* partial() { return partial_; }
  std::size_t capacity() const { return capacity_; }

 private:
  void Free();

  float* partial_ = nullptr;
  std::size_t capacity_ = 0;
};

// Reduces d_in[0..n) to a single float written to d_out[0] on the device.
//
// d_in, d_out are device pointers (d_out points to at least 1 float). `ws`
// supplies the inter-pass scratch buffer and must outlive the launch; it is
// sized internally via ws.Reserve(). Asynchronous w.r.t. the host on `stream`.
//
// n == 0 leaves d_out untouched (caller should treat the sum as 0).
void LaunchReduce(const float* d_in, float* d_out, std::size_t n,
                  ReductionWorkspace& ws,
                  ReductionVersion version = ReductionVersion::WarpShuffle,
                  cudaStream_t stream = nullptr);

}  // namespace mgie
