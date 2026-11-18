# Linux and WSL2 toolchain setup

Install a Linux CUDA Toolkit with `nvcc` and configure NVIDIA driver access.
On WSL2, GPU access comes from the Windows host driver. Check both
`nvidia-smi` and `nvcc --version` before building.

The tested toolchain is CUDA 12.4, GCC 13, CMake 4.2 and Ninja. The minimum
CMake version is 3.24. On Ubuntu, install the host tools and a Python 3.14
interpreter with `venv`. Use a host compiler supported by your CUDA toolkit.
The included GCC 13 toolchain file selects `/usr/bin/g++-13`. Set
`CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability without the
decimal point, such as `89` for compute capability 8.9.

Use the [README commands](../README.md#reproduce) to install Python
dependencies, export the checkpoint and build the binaries. Run the
[validation commands](../README.md#validate) after generating the test vectors.
