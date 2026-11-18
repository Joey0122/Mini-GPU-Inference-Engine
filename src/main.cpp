// src/main.cpp
//
// Phase 0 entry point / toolchain smoke test.
//
// Queries the GPU, runs vector_add, and verifies the result on the host.

#include <cstddef>
#include <vector>

#include "common/errors.hpp"
#include "common/logging.hpp"
#include "cuda/device_info.hpp"
#include "cuda/memory.hpp"
#include "kernels/vector_add.hpp"

int main() {
  using namespace mgie;

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    LOG_ERROR("no CUDA-capable device found");
    return 1;
  }

  const DeviceInfo info = QueryDeviceInfo(0);
  LOG_INFO("GPU: " << info.name);
  LOG_INFO("compute capability: " << info.compute_major << "."
                                  << info.compute_minor);
  LOG_INFO("global memory: "
           << (info.total_global_mem / (1024 * 1024)) << " MiB");
  LOG_INFO("SMs: " << info.multiprocessor_count
                   << ", warp size: " << info.warp_size);
  LOG_INFO("theoretical bandwidth: " << info.TheoreticalBandwidthGBs()
                                     << " GB/s");

  // Tiny vector_add smoke test: c = a + b, verified on host.
  const std::size_t n = 1 << 20;  // 1M elements
  std::vector<float> h_a(n, 1.0f);
  std::vector<float> h_b(n, 2.0f);
  std::vector<float> h_c(n, 0.0f);

  DeviceBuffer<float> d_a(n);
  DeviceBuffer<float> d_b(n);
  DeviceBuffer<float> d_c(n);
  d_a.CopyFromHost(h_a.data(), n);
  d_b.CopyFromHost(h_b.data(), n);

  LaunchVectorAdd(d_a.data(), d_b.data(), d_c.data(), n);
  CUDA_CHECK_LAST_SYNC();

  d_c.CopyToHost(h_c.data(), n);

  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (h_c[i] != 3.0f) {
      ok = false;
      LOG_ERROR("mismatch at " << i << ": got " << h_c[i] << " expected 3");
      break;
    }
  }

  if (ok) {
    LOG_INFO("vector_add smoke test PASSED (" << n << " elements)");
  }
  return ok ? 0 : 1;
}
