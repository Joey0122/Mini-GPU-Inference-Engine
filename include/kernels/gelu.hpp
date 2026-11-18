// include/kernels/gelu.hpp
#pragma once
#include <cuda_runtime.h>

namespace mgie {

// NewGELU (GPT-2): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// Matches F.gelu(approximate='tanh').
void LaunchGelu(const float* x, float* out, int N,
                cudaStream_t stream = nullptr);

}  // namespace mgie
