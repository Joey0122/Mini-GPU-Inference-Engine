# cmake/FetchDependencies.cmake
#
# Fetches third-party test/benchmark dependencies via FetchContent so the build
# is reproducible and requires no system-wide installs. GoogleTest and Google
# Benchmark are fetched at fixed tags.
#
# Called lazily by tests/ and benchmarks/ via the helper functions below so the
# fetch only happens when that subtree is enabled.

include(FetchContent)

# Pin versions for reproducibility. Bump deliberately, not implicitly.
set(MGIE_GOOGLETEST_TAG  "v1.15.2" CACHE STRING "GoogleTest git tag")
set(MGIE_BENCHMARK_TAG   "v1.9.0"  CACHE STRING "Google Benchmark git tag")

function(mgie_fetch_googletest)
  if(TARGET gtest_main)
    return()
  endif()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${MGIE_GOOGLETEST_TAG}
    GIT_SHALLOW    TRUE
  )
  # gtest_force_shared_crt: a GoogleTest CMake variable for the MSVC shared CRT.
  # It is a no-op on Linux/GCC but kept enabled for future Windows portability.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endfunction()

function(mgie_fetch_benchmark)
  if(TARGET benchmark::benchmark)
    return()
  endif()
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        ${MGIE_BENCHMARK_TAG}
    GIT_SHALLOW    TRUE
  )
  # Don't build benchmark's own tests, and don't fail if no matching libc++.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endfunction()
