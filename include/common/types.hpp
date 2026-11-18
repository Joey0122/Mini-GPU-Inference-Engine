// include/common/types.hpp
//
// Core value types shared across every layer.
//
// The original design described a common Tensor abstraction so kernels operate
// through a shared structure instead of raw pointers threaded everywhere.
// This is the Phase 0 skeleton: enough to type-check launch wrappers and the
// benchmark/test harness. It deliberately does NOT own memory yet — allocation
// lives in the runtime layer (src/runtime/allocator.cu), arriving in a later
// phase. Tensor here is a non-owning view (data + shape + dtype + device).
//
// Future role: Phases 1-5 extend this with stride support, element-count and
// byte-size helpers, and an owning TensorStorage. The model layer passes
// Tensors to kernel launch wrappers; nothing downstream should see void* args.

#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace mgie {

// Element data type. FP32 is the Phase 0..2 workhorse; FP16 arrives with the
// Tensor Core GEMM path (Phase 2) and the half-precision tolerance in the spec.
enum class DType {
  kFloat32,
  kFloat16,
  kInt32,
};

// Where a tensor's data physically lives.
enum class DeviceType {
  kHost,
  kDevice,
};

// Size in bytes of one element of the given dtype.
inline std::size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return 4;
    case DType::kFloat16: return 2;
    case DType::kInt32:   return 4;
  }
  return 0;
}

// Non-owning tensor view. Memory ownership is documented at the allocation
// site (runtime layer).
struct Tensor {
  void* data = nullptr;
  std::vector<int> shape;
  DType dtype = DType::kFloat32;
  DeviceType device = DeviceType::kHost;

  // Total number of elements (product of dims). Empty shape -> 0 elements
  // by convention here (a scalar tensor is represented as shape {1}).
  std::size_t NumElements() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(),
                           static_cast<std::size_t>(1),
                           [](std::size_t acc, int dim) {
                             return acc * static_cast<std::size_t>(dim);
                           });
  }

  // Total byte footprint of the element data.
  std::size_t NumBytes() const { return NumElements() * DTypeSize(dtype); }
};

}  // namespace mgie
