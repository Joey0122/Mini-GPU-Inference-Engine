// include/kernels/softmax.hpp
#pragma once
#include <cuda_runtime.h>

namespace mgie {

enum class SoftmaxVersion { Naive, Fused };

// Row-wise softmax over [N, D]: out[i,j] = exp(x[i,j]-max_i) / sum_i_exp
void LaunchSoftmax(const float* x, float* out, int N, int D,
                   SoftmaxVersion version = SoftmaxVersion::Fused,
                   cudaStream_t stream = nullptr);

}  // namespace mgie
