# Toolchain Setup — Windows 11 + CUDA 12.x + Visual Studio 2022

This project targets Windows 11 with an RTX 4070 Laptop (compute capability
8.9). Follow these steps once; afterward the CMake build in the root
[README](../README.md) works end-to-end.

> Why MSVC and not MinGW/GCC: on Windows, `nvcc` only supports the Microsoft
> C++ compiler (`cl.exe`) as its host compiler. A MinGW g++ on PATH will not
> work as the CUDA host compiler — this is the most common Phase 0 setup
> failure.

## 1. Visual Studio 2022 (C++ toolchain)

Install **Visual Studio 2022** (Community is fine) with the
**"Desktop development with C++"** workload. This provides `cl.exe`, the
Windows SDK, and the "x64 Native Tools Command Prompt for VS 2022" that puts
the compiler on PATH.

Verify:
```powershell
# In "x64 Native Tools Command Prompt for VS 2022"
cl
```

## 2. CUDA Toolkit 12.x

Download and install the CUDA Toolkit 12.x for Windows from NVIDIA. Install it
**after** Visual Studio so the installer can register the MSBuild/VS
integration.

Verify:
```powershell
nvcc --version      # should report release 12.x
nvidia-smi          # should list the RTX 4070 Laptop and the driver version
```

## 3. CMake ≥ 3.24 and Ninja

Install CMake (≥ 3.24) and Ninja. Both are commonly bundled with the VS
installer's CMake component, or install standalone.

Verify:
```powershell
cmake --version
ninja --version
```

## 4. Python + PyTorch (validation only)

PyTorch is used solely to generate reference outputs the C++ tests diff
against — it is never linked into the engine.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r scripts/requirements.txt
```

A CPU-only PyTorch build is sufficient for reference generation; a CUDA build
only speeds it up.

## 5. (Optional) clang-format / clang-tidy

For local formatting/linting (also part of CI later). Install LLVM for Windows
to get `clang-format` and `clang-tidy`, or use the versions bundled with VS.

## 6. (Optional) Nsight tools

Nsight Compute and Nsight Systems ship with the CUDA Toolkit installer. They
are required starting in Phase 1 for the profiling reports — install them now
so they're ready.

## Verifying the whole chain

```powershell
python scripts/device_query.py          # driver-level check (no build needed)
cmake -S . -B build -G "Ninja" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
build\mgie_app.exe                       # device info + vector_add smoke test
ctest --test-dir build --output-on-failure
```

If `mgie_app.exe` prints the GPU name and "vector_add smoke test PASSED", the
Phase 0 toolchain is fully operational.
