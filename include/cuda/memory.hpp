// include/cuda/memory.hpp
//
// Move-only CUDA device buffers with RAII allocation and ownership.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "common/errors.hpp"

namespace mgie {

// Owns a single device allocation. Move-only.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

  // Copy `count` elements from host into the buffer.
  void CopyFromHost(const T* host, std::size_t count) {
    CUDA_CHECK(cudaMemcpy(data_, host, count * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  // Copy `count` elements from the buffer back to host.
  void CopyToHost(T* host, std::size_t count) const {
    CUDA_CHECK(cudaMemcpy(host, data_, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  void Reset() {
    if (data_ != nullptr) {
      cudaFree(data_);  // best-effort; do not throw from destructor path
      data_ = nullptr;
    }
    count_ = 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace mgie
