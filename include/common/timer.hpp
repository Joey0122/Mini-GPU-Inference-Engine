// include/common/timer.hpp
//
// GPU timing built on CUDA events.
//
// The benchmark harness reports execution
// time, bandwidth, and GFLOPS with warm-up and multi-iteration statistics.
// Wall-clock host timers cannot measure GPU kernel time accurately because
// launches are asynchronous; CUDA events recorded on a stream are the correct
// tool. This header provides:
//
//   * CudaEventTimer  - RAII wrapper around a start/stop event pair.
//   * TimeKernel(...)  - warm up, then run a callable N times, returning
//                        per-iteration milliseconds (mean + stddev computed
//                        by the benchmark harness in benchmarks/common).
//
// Host-only. Phases 1-6 reuse this for every kernel benchmark, so the timing
// methodology stays identical across the whole project.

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <vector>

#include "common/errors.hpp"

namespace mgie {

// RAII pair of CUDA events for timing a region on a stream.
class CudaEventTimer {
 public:
  CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~CudaEventTimer() {
    // Best-effort cleanup; destructors must not throw.
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;

  void Start(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  // Records the stop event, synchronizes on it, and returns elapsed ms.
  float StopAndElapsedMs(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

// Run `body` `warmup` times (untimed) then `iterations` times (each timed
// individually with CUDA events). Returns the per-iteration times in ms.
//
// `body` should enqueue exactly the work under test onto `stream` and must not
// itself synchronize. Synchronization happens inside the timer.
inline std::vector<float> TimeKernel(const std::function<void()>& body,
                                     int iterations = 100, int warmup = 10,
                                     cudaStream_t stream = nullptr) {
  for (int i = 0; i < warmup; ++i) {
    body();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> times;
  times.reserve(static_cast<std::size_t>(iterations));
  CudaEventTimer timer;
  for (int i = 0; i < iterations; ++i) {
    timer.Start(stream);
    body();
    times.push_back(timer.StopAndElapsedMs(stream));
  }
  return times;
}

// Simple mean/stddev helpers shared by the benchmark harness.
inline float Mean(const std::vector<float>& xs) {
  if (xs.empty()) return 0.0f;
  float sum = 0.0f;
  for (float x : xs) sum += x;
  return sum / static_cast<float>(xs.size());
}

inline float StdDev(const std::vector<float>& xs) {
  if (xs.size() < 2) return 0.0f;
  const float mean = Mean(xs);
  float acc = 0.0f;
  for (float x : xs) acc += (x - mean) * (x - mean);
  return std::sqrt(acc / static_cast<float>(xs.size() - 1));
}

}  // namespace mgie
