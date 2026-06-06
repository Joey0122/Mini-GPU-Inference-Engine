// tests/common/test_utils.hpp
//
// Shared test helpers: tolerance comparison and CPU reference implementations.
//
// IMPLEMENTATION_RULES Rule 8 requires diffing kernel output against a
// reference within a documented tolerance (FP32 < 1e-4, FP16 < 1e-2 per
// PROJECT_SPEC). These helpers centralize that comparison so every kernel test
// uses the same definition of "close enough", and provide the CPU reference
// kernels that GPU tests diff against (also used by the host-only CI tests).

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace mgie::test {

// Tolerances from PROJECT_SPEC.md.
inline constexpr float kFp32AbsTol = 1e-4f;
inline constexpr float kFp16AbsTol = 1e-2f;

// Maximum absolute elementwise difference between two equal-length buffers.
inline float MaxAbsError(const std::vector<float>& a,
                         const std::vector<float>& b) {
  float max_err = 0.0f;
  const std::size_t n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::fabs(a[i] - b[i]));
  }
  return max_err;
}

// True if every element matches within `tol`.
inline bool AllClose(const std::vector<float>& a, const std::vector<float>& b,
                     float tol = kFp32AbsTol) {
  if (a.size() != b.size()) return false;
  return MaxAbsError(a, b) <= tol;
}

// ---- CPU reference kernels -------------------------------------------------
// The golden implementations GPU kernels are validated against. Kept simple
// and obviously correct; performance is irrelevant here.

inline std::vector<float> CpuVectorAdd(const std::vector<float>& a,
                                       const std::vector<float>& b) {
  std::vector<float> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

}  // namespace mgie::test
