// include/cuda/launch.hpp
//
// Reusable kernel-launch helpers.
//
// AGENT.md forbids hardcoded launch dimensions and duplicated launch code, and
// asks for reusable launch wrappers. This header centralizes the grid-size
// arithmetic every kernel needs. Phase 0 uses only CeilDiv / Grid1D; later
// phases add 2D tile grids for GEMM/transpose here so the pattern stays in one
// place.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>

namespace mgie {

// Default 1D block size. A multiple of the warp size (32); 256 is a safe,
// occupancy-friendly default on Ada (CC 8.9). Override per kernel when
// profiling justifies it (Rule 5).
inline constexpr int kDefaultBlockSize = 256;

// Ceiling division for computing grid dimensions from problem size.
inline constexpr unsigned int CeilDiv(unsigned int numerator,
                                      unsigned int denominator) {
  return (numerator + denominator - 1u) / denominator;
}

// Number of blocks needed to cover `n` elements at `block_size` threads each.
inline dim3 Grid1D(std::size_t n, int block_size = kDefaultBlockSize) {
  const unsigned int blocks =
      CeilDiv(static_cast<unsigned int>(n), static_cast<unsigned int>(block_size));
  return dim3(std::max(1u, blocks));
}

}  // namespace mgie
