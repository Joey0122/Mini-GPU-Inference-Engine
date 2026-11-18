# benchmarks/CMakeLists.txt
#
# Benchmark build. Fetches Google Benchmark and defines benchmark
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

# ---- reduction benchmark (Phase 1) ----------------------------------------
# CPU baseline + three GPU versions (naive / shared / warp-shuffle).
add_executable(bench_reduction reduction/bench_reduction.cu)
target_link_libraries(bench_reduction PRIVATE
  mgie_core mgie_bench_common benchmark::benchmark)

# ---- gemm benchmark (Phase 2) ----------------------------------------
add_executable(bench_gemm gemm/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE
  mgie_core mgie_bench_common benchmark::benchmark CUDA::cublas)

# ---- decode GEMM comparison (Phase 6b) ----------------------------------
add_executable(bench_decode_gemm gemm/bench_decode_gemm.cu)
target_link_libraries(bench_decode_gemm PRIVATE
  mgie_core CUDA::cublas)

# ---- Phase 3 benchmarks ---------------------------------------------------
add_executable(bench_layernorm softmax/bench_layernorm.cu)
target_link_libraries(bench_layernorm PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

add_executable(bench_softmax softmax/bench_softmax.cu)
target_link_libraries(bench_softmax PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

add_executable(bench_gelu softmax/bench_gelu.cu)
target_link_libraries(bench_gelu PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- attention benchmark (Phase 4) ----------------------------------------
add_executable(bench_attention attention/bench_attention.cu)
target_link_libraries(bench_attention PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- embedding benchmark (Phase 5 C4) -------------------------------------
add_executable(bench_embedding embedding/bench_embedding.cu)
target_link_libraries(bench_embedding PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- bias_add benchmark (Phase 5 C5) --------------------------------------
add_executable(bench_bias_add bias_add/bench_bias_add.cu)
target_link_libraries(bench_bias_add PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- qkv_split benchmark (Phase 5 C6) ------------------------------------
add_executable(bench_qkv_split qkv_split/bench_qkv_split.cu)
target_link_libraries(bench_qkv_split PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- forward benchmark (Phase 5 C8) --------------------------------------
add_executable(bench_forward forward/bench_forward.cu)
target_link_libraries(bench_forward PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- KV-cache decode benchmark (Phase 6a) ---------------------------------
add_executable(bench_decode_kv decode/bench_decode_kv.cu)
target_link_libraries(bench_decode_kv PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- KV-cache decode profile (Phase 6a) -----------------------------------
add_executable(bench_decode_profile decode/bench_decode_profile.cu)
target_link_libraries(bench_decode_profile PRIVATE mgie_core mgie_bench_common benchmark::benchmark)

# ---- Kernel launch-overhead audit (Phase 6c readiness) ---------------------
add_executable(bench_launch_overhead decode/bench_launch_overhead.cu)
target_link_libraries(bench_launch_overhead PRIVATE mgie_core mgie_bench_common benchmark::benchmark)
