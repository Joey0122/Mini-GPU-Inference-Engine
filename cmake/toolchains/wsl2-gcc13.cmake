# cmake/toolchains/wsl2-gcc13.cmake
#
# WSL2 Ubuntu toolchain: pins the host compiler to GCC 13, which is the
# maximum version supported by nvcc 12.x as a CUDA host compiler.
# GCC 15 is the system default on this Ubuntu install and must NOT be used
# because it causes ABI mismatches when linking CUDA device code.
#
# Usage:
#   cmake -S . -B build -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/wsl2-gcc13.cmake

set(CMAKE_C_COMPILER   /usr/bin/gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)
