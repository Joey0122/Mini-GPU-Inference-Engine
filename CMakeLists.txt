# CMakeLists.txt — root build configuration
#
# Mini GPU Inference Engine (CUDA from scratch), targeting GPT-2 Small inference.
# Phase 0: stand up a CMake project that compiles a CUDA target, links the
# common/runtime libraries, and wires tests + benchmarks behind options.
#
# Active environment: WSL2 Ubuntu, GCC 13, CUDA 12.4, RTX 4070 Laptop (sm_89).
# Host compiler MUST be gcc-13; nvcc 12.x does not support GCC > 13.
# Use: cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/wsl2-gcc13.cmake

cmake_minimum_required(VERSION 3.24)  # native CMAKE_CUDA_ARCHITECTURES, FetchContent

# Must be set BEFORE project() — CMake's CUDA language init reads this variable
# during the project() call. Setting it after has no effect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(mini_gpu_inference_engine
  VERSION 0.1.0
  DESCRIPTION "From-scratch CUDA inference engine for GPT-2 Small"
  LANGUAGES CXX CUDA)

# ---- Standards -------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile_commands.json for clang-tidy / editors.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type when not set by a multi-config generator.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options ---------------------------------------------------------------
option(MGIE_BUILD_TESTS "Build unit/integration tests (GoogleTest)" ON)
option(MGIE_BUILD_BENCHMARKS "Build benchmarks (Google Benchmark)" ON)
option(MGIE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ---- Shared compile flags via interface target -----------------------------
include(${CMAKE_SOURCE_DIR}/cmake/CompilerWarnings.cmake)

# ---- Core library ----------------------------------------------------------
# Hand-written CUDA kernels and GPT-2 model code.
add_library(mgie_core
  src/kernels/vector_add.cu
  src/kernels/reduction.cu
  src/kernels/gemm.cu
  src/kernels/layernorm.cu
  src/kernels/softmax.cu
  src/kernels/gelu.cu
  src/kernels/attention.cu
  src/kernels/embedding.cu
  src/kernels/bias_add.cu
  src/kernels/qkv_split.cu
  src/model/loader.cpp
  src/model/workspace.cu
  src/model/transformer_block.cu
  src/model/forward.cu
  src/model/decode.cu
  src/model/kv_cache.cu
)
target_include_directories(mgie_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mgie_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mgie_core PRIVATE mgie_warnings)

# ---- Executables -----------------------------------------------------------
add_executable(mgie_app src/main.cpp)
target_link_libraries(mgie_app PRIVATE mgie_core mgie_warnings)

add_executable(mgie_generate src/generate.cpp)
target_link_libraries(mgie_generate PRIVATE mgie_core mgie_warnings)

# ---- Tests -----------------------------------------------------------------
if(MGIE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Benchmarks ------------------------------------------------------------
if(MGIE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---- Configuration summary -------------------------------------------------
message(STATUS "==== mini_gpu_inference_engine configuration ====")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch       : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests           : ${MGIE_BUILD_TESTS}")
message(STATUS "  Benchmarks      : ${MGIE_BUILD_BENCHMARKS}")
message(STATUS "=================================================")
