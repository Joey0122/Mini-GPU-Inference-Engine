// include/common/errors.hpp
//
// Centralized CUDA error handling.
//
// docs/architecture.md mandates that every CUDA runtime call be wrapped in
// CUDA_CHECK(...) and that the macro live here. This header is the single
// source of truth for that contract; nothing else in the codebase should
// inspect a raw cudaError_t.
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&ptr, bytes));
//   CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
//   CUDA_CHECK_LAST();  // check the error state after a kernel launch
//
// On failure the macro logs file:line, the failing expression, and the CUDA
// error string, then throws CudaError. Throwing (rather than abort()) lets
// GoogleTest report a clean failure instead of crashing the test binary.

#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace mgie {

// Exception type thrown by CUDA_CHECK on a non-success status.
class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t status, const char* expr, const char* file, int line)
      : std::runtime_error(BuildMessage(status, expr, file, line)),
        status_(status) {}

  cudaError_t status() const noexcept { return status_; }

 private:
  static std::string BuildMessage(cudaError_t status, const char* expr,
                                  const char* file, int line) {
    return std::string("CUDA error at ") + file + ":" + std::to_string(line) +
           "\n  call: " + expr +
           "\n  code: " + std::to_string(static_cast<int>(status)) + " (" +
           cudaGetErrorName(status) + ")" +
           "\n  what: " + cudaGetErrorString(status);
  }

  cudaError_t status_;
};

namespace detail {

inline void CheckCuda(cudaError_t status, const char* expr, const char* file,
                      int line) {
  if (status != cudaSuccess) {
    throw CudaError(status, expr, file, line);
  }
}

}  // namespace detail

}  // namespace mgie

// Wrap every CUDA runtime call. Evaluates the expression exactly once.
#define CUDA_CHECK(expr) \
  ::mgie::detail::CheckCuda((expr), #expr, __FILE__, __LINE__)

// Check the error captured by the most recent kernel launch. Use after a
// <<<...>>> launch, which does not itself return a cudaError_t.
#define CUDA_CHECK_LAST() \
  ::mgie::detail::CheckCuda(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)

// Launch check that also synchronizes — use in tests/debug where you want
// launch *and* execution errors surfaced at the call site. Avoid in hot paths.
#define CUDA_CHECK_LAST_SYNC()                                                \
  do {                                                                        \
    ::mgie::detail::CheckCuda(cudaGetLastError(), "kernel launch", __FILE__,  \
                              __LINE__);                                      \
    ::mgie::detail::CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", \
                              __FILE__, __LINE__);                            \
  } while (0)
