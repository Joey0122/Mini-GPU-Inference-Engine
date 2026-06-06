// include/cuda/device_info.hpp
//
// Query and report GPU device properties.
//
// Phase 0 needs to prove the toolchain sees the GPU and to record the
// reproducibility metadata required by Rule 6 (GPU model, compute capability,
// memory, bandwidth). This header exposes a small struct plus a query helper;
// main.cpp and the benchmark harness use it to stamp every result set.
//
// Future role: the runtime layer uses theoretical bandwidth here to compute
// "% of peak" for memory-bound kernels (reduction, transpose, softmax).

#pragma once

#include <cuda_runtime.h>

#include <string>

#include "common/errors.hpp"

namespace mgie {

struct DeviceInfo {
  int device_id = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_global_mem = 0;     // bytes
  int multiprocessor_count = 0;
  int max_threads_per_block = 0;
  int max_threads_per_sm = 0;
  int warp_size = 0;
  int memory_clock_khz = 0;             // peak memory clock
  int memory_bus_width_bits = 0;        // global memory bus width

  // Theoretical peak global-memory bandwidth in GB/s.
  // BW = 2 (DDR) * memClock(Hz) * busWidth(bytes). Result scaled to GB/s.
  double TheoreticalBandwidthGBs() const {
    const double clock_hz = static_cast<double>(memory_clock_khz) * 1.0e3;
    const double bus_bytes = static_cast<double>(memory_bus_width_bits) / 8.0;
    return 2.0 * clock_hz * bus_bytes / 1.0e9;
  }

  int ComputeCapability() const { return compute_major * 10 + compute_minor; }
};

// Query properties for the given device (default 0). Throws CudaError on
// failure via CUDA_CHECK.
inline DeviceInfo QueryDeviceInfo(int device_id = 0) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

  DeviceInfo info;
  info.device_id = device_id;
  info.name = prop.name;
  info.compute_major = prop.major;
  info.compute_minor = prop.minor;
  info.total_global_mem = prop.totalGlobalMem;
  info.multiprocessor_count = prop.multiProcessorCount;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info.warp_size = prop.warpSize;
  info.memory_clock_khz = prop.memoryClockRate;
  info.memory_bus_width_bits = prop.memoryBusWidth;
  return info;
}

}  // namespace mgie
