# tests/CMakeLists.txt
#
# Test build. Fetches GoogleTest and defines the test executables. The
# validation strategy checks each kernel
# against a CPU/PyTorch reference within a documented tolerance. Phase 0 wires
# the harness with the vector_add kernel as the first GPU unit test plus a
# host-only reference test that needs no GPU (so CI can run it — see ci.yml).

include(${CMAKE_SOURCE_DIR}/cmake/FetchDependencies.cmake)
mgie_fetch_googletest()

include(GoogleTest)

# Shared helpers for tests (tolerance comparisons, reference loading).
add_library(mgie_test_support INTERFACE)
target_include_directories(mgie_test_support INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})

# ---- GPU unit tests (require a CUDA device at runtime) ---------------------
add_executable(mgie_unit_tests
  unit/test_vector_add.cu
  unit/test_reduction.cu
  unit/test_gemm.cu
  unit/test_layernorm.cu
  unit/test_softmax.cu
  unit/test_gelu.cu
  unit/test_attention.cu
  unit/test_embedding.cu
  unit/test_bias_add.cu
  unit/test_qkv_split.cu
  unit/test_transformer_block.cu
  unit/test_forward.cu
  unit/test_decode.cu
  unit/test_loader.cu
  unit/test_kv_cache.cu
)
target_link_libraries(mgie_unit_tests PRIVATE
  mgie_core mgie_test_support gtest gtest_main CUDA::cublas)
target_compile_definitions(mgie_unit_tests PRIVATE
  MGIE_MODEL_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/data/model_weights"
  MGIE_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/data/test_vectors/golden")
# Label so CI can deselect GPU tests on GPU-less runners (ctest -L cpu).
gtest_discover_tests(mgie_unit_tests PROPERTIES LABELS "gpu")

# ---- Host-only reference tests (no GPU needed) -----------------------------
# Exercises the CPU reference + tolerance utilities that GPU tests diff against.
# Runnable on GitHub-hosted runners that have no GPU.
add_executable(mgie_reference_tests
  reference/test_cpu_reference.cpp
)
target_link_libraries(mgie_reference_tests PRIVATE
  mgie_test_support gtest gtest_main)
gtest_discover_tests(mgie_reference_tests PROPERTIES LABELS "cpu")

# ---- PyTorch validation runner (Phase 3) -----------------------------------
# Loads .npy vectors from scripts/generate_reference.py, runs the actual
# committed kernels, diffs against PyTorch references, emits
# tests/validation_report.json in the build tree. Gate: FP32 max_abs_error < 1e-4.
add_executable(mgie_validation_runner
  validation/validate_runner.cu
)
target_link_libraries(mgie_validation_runner PRIVATE
  mgie_core mgie_test_support gtest gtest_main)
# Read fixtures from the source tree and write the report into the build tree.
target_compile_definitions(mgie_validation_runner PRIVATE
  MGIE_TEST_VECTOR_DIR="${CMAKE_SOURCE_DIR}/data/test_vectors"
  MGIE_REPORT_DIR="${CMAKE_CURRENT_BINARY_DIR}")
gtest_discover_tests(mgie_validation_runner PROPERTIES LABELS "validation")
