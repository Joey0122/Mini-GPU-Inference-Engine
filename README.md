# Mini GPU Inference Engine

A from-scratch CUDA inference engine for **GPT-2 Small (124M)**, built to
demonstrate GPU systems engineering: every core kernel — reduction, GEMM,
softmax, LayerNorm, GELU, attention — is hand-written, benchmarked against
cuBLAS/PyTorch, profiled with Nsight, and documented.

> **Status:** Phase 0 (scaffolding). No production kernels implemented yet.
> The repository currently contains the build system, test/benchmark
> frameworks, and a single `vector_add` toolchain smoke-test kernel.

## Goals

- Run GPT-2 Small forward inference on GPU, batch size 1, arbitrary prompt
  length, greedy + temperature sampling.
- Hand-write all critical CUDA kernels (cuBLAS allowed only as a benchmark
  baseline, never as the implementation).
- Validate every kernel against a PyTorch/CPU reference (FP32 < 1e-4,
  FP16 < 1e-2) before any optimization.
- Profile and document the optimization journey with Nsight evidence.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the authoritative requirements and
[ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Target environment

| | |
|---|---|
| OS | Windows 11 |
| GPU | RTX 4070 Laptop (Ada, compute capability 8.9), 8 GB |
| CUDA | 12.x |
| Host compiler | Visual Studio 2022 (MSVC — required by `nvcc` on Windows) |
| Language | C++17 + CUDA |
| Build | CMake ≥ 3.24 + Ninja |
| Test | GoogleTest |
| Benchmark | Google Benchmark + CUDA-event harness |
| Validation | Python 3.10+, PyTorch (reference only) |

## Building

```powershell
# From a "x64 Native Tools Command Prompt for VS 2022" (so cl.exe + nvcc are on PATH)
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

### Run the Phase 0 smoke test

```powershell
build\mgie_app.exe            # prints device info + runs vector_add
```

### Tests

```powershell
ctest --test-dir build --output-on-failure          # all tests
ctest --test-dir build -L cpu --output-on-failure   # host-only (no GPU)
ctest --test-dir build -L gpu --output-on-failure   # GPU kernels
```

### Benchmarks

```powershell
build\bench_vector_add.exe                          # writes vector_add_results.csv
python scripts/plot_benchmarks.py vector_add_results.csv
```

> **Reproducible benchmarking:** lock GPU clocks before measuring
> (`nvidia-smi -lgc <min>,<max>`) — the laptop 4070 throttles under sustained
> load and will otherwise produce noisy numbers.

## Repository layout

See [ARCHITECTURE.md](ARCHITECTURE.md#repository-structure). In short:
`include/` + `src/` (common/cuda/kernels/runtime/model), `tests/`,
`benchmarks/`, `scripts/` (Python validation + plotting), `reports/`
(per-phase writeups), `docs/`.

## Roadmap

Phases 0–7 in [TASK_BREAKDOWN.md](TASK_BREAKDOWN.md): setup → primitives →
GEMM → transformer components → attention → GPT-2 inference → optimization →
publication. Phase gates are enforced — a phase is not done until it compiles,
tests pass, a benchmark exists, results are documented, and it's committed.
