# Toolchain Setup — WSL2 Ubuntu + CUDA 12.4 + GCC 13

This project targets WSL2 Ubuntu with an RTX 4070 Laptop (compute capability
8.9). Follow these steps once; afterward the CMake build in the root
[README](../README.md) works end-to-end.

> **Migration note:** The original scaffolding assumed Windows 11 / MSVC /
> Visual Studio 2022. The active development environment is now WSL2 Ubuntu.

> **Host compiler constraint:** nvcc 12.x only supports GCC ≤ 13 as the CUDA
> host compiler. GCC 15 (the Ubuntu system default) must NOT be used — it causes
> ABI mismatches at link time. Always build via the provided toolchain file.

## 1. CUDA Toolkit 12.x (Linux)

Verify your WSL2 install has nvcc:
```bash
nvcc --version      # should report release 12.x
nvidia-smi          # should list the RTX 4070 Laptop
```

If not present, install the CUDA Toolkit for Ubuntu from NVIDIA's package repos.

## 2. GCC 13

```bash
sudo apt install gcc-13 g++-13
```

Verify:
```bash
g++-13 --version
```

## 3. CMake ≥ 3.24 and Ninja

```bash
sudo apt install cmake ninja-build
cmake --version    # ≥ 3.24
ninja --version
```

## 4. Python + PyTorch (validation only)

PyTorch is used solely to generate reference outputs the C++ tests diff
against — it is never linked into the engine.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

## 5. (Optional) clang-format / clang-tidy

```bash
sudo apt install clang-format clang-tidy
```

## 6. (Optional) Nsight tools

Nsight Compute and Nsight Systems ship with the CUDA Toolkit installer. They
are required starting in Phase 1 for the profiling reports.

## Verifying the whole chain

```bash
source .venv/bin/activate
python3 scripts/device_query.py          # driver-level check (no build needed)

cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/wsl2-gcc13.cmake
cmake --build build

./build/mgie_app                          # device info + vector_add smoke test
ctest --test-dir build --output-on-failure
./build/benchmarks/bench_vector_add
```

If `mgie_app` prints the GPU name and "vector_add smoke test PASSED", the
Phase 0 toolchain is fully operational.
