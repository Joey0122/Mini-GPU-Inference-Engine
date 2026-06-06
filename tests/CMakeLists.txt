# tests/CMakeLists.txt
#
# Test build. Fetches GoogleTest and defines the test executables. The
# validation strategy (IMPLEMENTATION_RULES Rule 8) is: every kernel is checked
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
)
target_link_libraries(mgie_unit_tests PRIVATE
  mgie_core mgie_test_support gtest gtest_main)
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
