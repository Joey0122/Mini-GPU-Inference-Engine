# TECH_STACK.md

Target hardware: **RTX 4070 Laptop** (Ada Lovelace, compute capability **8.9**),
8 GB VRAM.

> **Migration note:** The original scaffolding was authored assuming Windows 11
> and the MSVC host compiler (Visual Studio 2022).  Active development has moved
> to **WSL2 Ubuntu** with GCC.  All build artifacts and CI targets have been
> verified against the Linux toolchain; no CMake changes were required.

---

## Language
- C++17
- CUDA 12.4

## Host compiler
- GCC 13 / g++-13 (WSL2 Ubuntu) — selected automatically by CMake when
  `CMAKE_CUDA_COMPILER=/usr/bin/nvcc`.

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
- Python 3.x (project `.venv` — see below)
- PyTorch 2.12.0+cu130 — reference outputs only; never linked into the engine
- numpy
- safetensors — GPT-2 Small weight loading (Phase 5)

## Python environment
- Location: `.venv/` inside the repository root
- Activation: `source .venv/bin/activate`
- Install: `pip install -r scripts/requirements.txt`
- The project `.venv` is the authoritative Python environment; do **not** rely
  on user-level or system-wide installations.

## Development OS
- WSL2 Ubuntu (Windows 11 host)

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
