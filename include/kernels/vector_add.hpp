// include/kernels/vector_add.hpp
//
// Launch interface for the Phase 0 vector-add kernel.
//
// vector_add is the project's "hello, GPU": TASK_BREAKDOWN Phase 0 calls for a
// vector add kernel to prove the toolchain and exercise the benchmark harness.
// It is intentionally trivial and memory-bound (3 floats of traffic per FLOP),
// making it the canonical example of a bandwidth-limited kernel for Phase 0
// docs.
//
// The kernel layer exposes ONLY launch wrappers (ARCHITECTURE.md): callers in
// tests/benchmarks include this header and never write <<<...>>> themselves.
// This is the template every later kernel's public header follows.

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace mgie {

// Computes c[i] = a[i] + b[i] for i in [0, n) on the device.
// a, b, c are device pointers to at least `n` floats. Asynchronous with
// respect to the host; the launch is enqueued on `stream`.
//
// NOTE: this is the only kernel implemented in Phase 0. No production
// (reduction/GEMM/etc.) kernels exist yet by design.
void LaunchVectorAdd(const float* a, const float* b, float* c, std::size_t n,
                     cudaStream_t stream = nullptr);

}  // namespace mgie
