// include/kernels/vector_add.hpp
//
// Launch interface for elementwise vector addition.

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace mgie {

// Computes c[i] = a[i] + b[i] for i in [0, n) on the device.
// a, b, c are device pointers to at least `n` floats. Asynchronous with
// respect to the host; the launch is enqueued on `stream`.

void LaunchVectorAdd(const float* a, const float* b, float* c, std::size_t n,
                     cudaStream_t stream = nullptr);

}  // namespace mgie
