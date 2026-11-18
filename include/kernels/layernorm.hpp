// include/kernels/layernorm.hpp
#pragma once
#include <cuda_runtime.h>

namespace mgie {

enum class LayerNormVersion { Naive, Welford };

// Row-wise LayerNorm: out = gamma * (x - mean) / sqrt(var + eps) + beta
// x, out: [N, H] row-major device pointers
// gamma, beta: [H] device pointers
void LaunchLayerNorm(const float* x, float* out,
                     const float* gamma, const float* beta,
                     int N, int H, float eps,
                     LayerNormVersion version = LayerNormVersion::Welford,
                     cudaStream_t stream = nullptr);

}  // namespace mgie
