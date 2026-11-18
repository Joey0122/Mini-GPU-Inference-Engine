// include/common/types.hpp
//
// Common tensor metadata and element types.

#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace mgie {

// Element data type.
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
// site.
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
