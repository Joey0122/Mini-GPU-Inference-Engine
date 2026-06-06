# benchmarks/CMakeLists.txt
#
# Benchmark build. Fetches Google Benchmark (TECH_STACK) and defines benchmark
# executables. Google Benchmark drives the iteration/stat loop; the project's
# bench_harness (CUDA events) supplies GPU bandwidth/GFLOPS and CSV export per
# the ARCHITECTURE benchmark spec.
#
# Phase 0 ships one benchmark: vector_add (memory-bound reference example). Each
# later phase adds its kernels' benchmarks under the matching subdirectory
# (reduction/, gemm/, ...), all linking benchmarks/common.

include(${CMAKE_SOURCE_DIR}/cmake/FetchDependencies.cmake)
mgie_fetch_benchmark()

# Shared benchmark harness (header-only) exposed as an interface target.
add_library(mgie_bench_common INTERFACE)
target_include_directories(mgie_bench_common INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})

# ---- vector_add benchmark (Phase 0) ---------------------------------------
add_executable(bench_vector_add reduction/bench_vector_add.cu)
target_link_libraries(bench_vector_add PRIVATE
  mgie_core mgie_bench_common benchmark::benchmark)
