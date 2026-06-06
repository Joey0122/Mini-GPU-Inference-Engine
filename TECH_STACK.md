# TECH_STACK.md

Target environment: **Windows 11**, **RTX 4070 Laptop** (Ada Lovelace,
compute capability **8.9**), 8 GB VRAM.

## Language
- C++17
- CUDA 12.x

## Host compiler
- Visual Studio 2022 (MSVC) — **required** as the `nvcc` host compiler on
  Windows. MinGW/GCC is not supported by `nvcc` on Windows.

## Build
- CMake (>= 3.24)
- Ninja generator

## Testing
- GoogleTest (fetched via CMake FetchContent)

## Benchmarking
- Google Benchmark (fetched via CMake FetchContent) — drives the iteration
  loop and statistics.
- Custom CUDA-event harness (`include/common/timer.hpp` +
  `benchmarks/common/bench_harness.hpp`) — supplies GPU execution time,
  achieved bandwidth, and GFLOPS, and exports the required CSV schema. Google
  Benchmark alone cannot measure GPU bandwidth/GFLOPS/occupancy.
- cuBLAS — baseline comparison only (never the primary implementation, per
  PROJECT_SPEC).

## Validation
- Python 3.10+ (3.11 recommended)
- PyTorch — reference outputs only; never linked into the engine
- numpy
- safetensors — GPT-2 Small weight loading (Phase 5)

## Profiling
- Nsight Compute (per-kernel)
- Nsight Systems (end-to-end timeline)

## Formatting / Linting
- clang-format
- clang-tidy

## Documentation
- Markdown

## CI
- GitHub Actions — build + host-only tests on GitHub-hosted runners (no GPU);
  GPU tests run locally or on a future self-hosted GPU runner.
