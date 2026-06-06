# PROJECT_STATUS.md

# Mini GPU Inference Engine

Last Updated: 2026-06-06

---

# Project Goal

Build a CUDA-based GPT-2 Small (124M) inference engine from scratch.

Primary learning objectives:

* CUDA programming
* GPU memory hierarchy
* Kernel optimization
* Transformer inference
* Performance profiling
* Numerical validation

---

# Current Status

Current Phase:

Phase 0 (Environment & Infrastructure)

Status:

**COMPLETE**

Phase 1 (GPU Primitives / Reduction) is ready to begin.

---

# Repository State

Branch:

master

Working Tree:

clean

Commits:

2

---

# Current Environment

Development Environment:

WSL2 Ubuntu

GPU:

RTX 4070 Laptop (NVIDIA GeForce RTX 4070 Laptop GPU)

Compute Capability:

8.9 (Ada Lovelace)

CUDA Toolkit:

12.4.131

Host Compiler:

GCC 13.4.0 (g++-13) — nvcc 12.x maximum supported version

CMake:

4.2.3

Ninja:

1.13.2

Python:

3.14.4 (project .venv)

PyTorch:

2.12.0+cu130 (installed in .venv via scripts/requirements.txt)

CUDA Available:

True (torch.cuda.is_available() == True)

---

# Approved Architecture Changes

## Change 1

ARCHITECTURE.md: RMSNorm → LayerNorm

Reason: GPT-2 uses LayerNorm.

## Change 2

TASK_BREAKDOWN.md: RMSNorm and RoPE removed from required GPT-2 path.

Reason: GPT-2 does not use RMSNorm or RoPE. Retained as future Llama-family extensions.

---

# Completed Work

## Phase 0 — Environment & Infrastructure ✓

### Governance

* PROJECT_SPEC.md
* ARCHITECTURE.md
* TASK_BREAKDOWN.md
* IMPLEMENTATION_RULES.md
* AGENT.md
* TECH_STACK.md (updated for WSL2/GCC13)

### Repository Infrastructure

* Git repository initialized
* CMake build system (WSL2/GCC13/Ninja)
* Toolchain file: cmake/toolchains/wsl2-gcc13.cmake
* clang-format
* clang-tidy
* GitHub Actions workflow

### CUDA Infrastructure

* CUDA helper headers (errors, timer, launch, device_info, memory)
* Logging utilities
* Benchmark harness (CUDA event timing + CSV export)

### Phase 0 Kernel

vector_add — smoke test for CUDA toolchain and benchmark framework.

Files:

* include/kernels/vector_add.hpp
* src/kernels/vector_add.cu

### Testing Infrastructure

GoogleTest integration — 6/6 tests passing.

* VectorAddTest.MatchesCpuReference
* VectorAddTest.HandlesNonMultipleOfBlockSize
* CpuReference.VectorAddIsElementwiseSum
* ToleranceUtils.MaxAbsErrorFindsLargestGap
* ToleranceUtils.AllCloseRespectsTolerance
* ToleranceUtils.AllCloseRejectsSizeMismatch

### Benchmark Infrastructure

Google Benchmark integration — bench_vector_add produces CSV output.

Phase 0 benchmark results (RTX 4070 Laptop, sm_89):

| n | time (µs) | bandwidth (GB/s) | % peak |
|---|-----------|-----------------|--------|
| 65,536 | 11.0 | 61.4 | 24% |
| 1,048,576 | 11.2 | 1,103 | 431% |
| 16,777,216 | 890.6 | 225 | 88% |

Note: small-N bandwidth exceeds theoretical due to CUDA event overhead
dwarfing kernel time. Large-N (16M) at 88% of theoretical peak is the
meaningful number.

### Python Tooling

* scripts/device_query.py
* scripts/generate_reference.py
* scripts/plot_benchmarks.py
* scripts/requirements.txt (updated with cu130 index URL)
* .venv — project-local Python environment with torch 2.12.0+cu130

### Documentation

* docs/setup.md — updated for WSL2/GCC13

---

# Known Issues Resolved

## Issue 1 — Missing git repository

Repository was transferred without .git history.
Resolution: git init + initial commit on 2026-06-06.

## Issue 2 — Compiler ABI mismatch

Default g++ on this Ubuntu is GCC 15, which is unsupported by nvcc 12.4.
GoogleTest compiled with GCC 15 + CUDA device link with GCC 13 → linker
error (undefined __cxa_call_terminate).
Resolution: cmake/toolchains/wsl2-gcc13.cmake pins all compilers to GCC 13.

## Issue 3 — CUDA arch auto-detected as sm_52

CMAKE_CUDA_ARCHITECTURES was set after project() call, allowing CMake
auto-detection to override it with sm_52.
Resolution: moved the set() call before project() in CMakeLists.txt.

## Issue 4 — Project .venv had no torch

.venv was created with Python 3.14 but torch was not installed.
Resolution: pip install torch torchvision --index-url .../cu130

---

# Build Instructions

```bash
# Configure (must use toolchain file to lock GCC 13)
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/wsl2-gcc13.cmake

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure

# Benchmark
./build/benchmarks/bench_vector_add

# Python validation
source .venv/bin/activate
python3 scripts/device_query.py
```

---

# Phase Gate

Phase 0 exit criteria — all satisfied:

✓ Project builds successfully (sm_89, GCC 13, CUDA 12.4)

✓ Tests pass (6/6)

✓ Benchmark executes and produces CSV

✓ mgie_app smoke test passes on real GPU

✓ PyTorch available in project .venv

✓ Git repository initialized with clean commit history

---

# Recommended Next Action

Begin Phase 1: GPU Primitives.

First task: Reduction kernel (naive implementation).

Follow IMPLEMENTATION_RULES Rule 8: write CPU reference and test vectors before
implementing CUDA.
