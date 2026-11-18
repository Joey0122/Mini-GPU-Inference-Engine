# cmake/CompilerWarnings.cmake
#
# Defines an INTERFACE target `mgie_warnings` that carries the project's warning
# flags for both the C++ (GCC) and CUDA (nvcc) compilers. Centralizing warnings
# here keeps IMPLEMENTATION_RULES code-quality expectations consistent across
# every target and lets CI flip warnings-as-errors via MGIE_WARNINGS_AS_ERRORS
# without touching individual targets.
#
# Active environment: WSL2 Ubuntu, GCC 13 host compiler, nvcc 12.4.
# The MSVC warning block below is intentionally retained as a no-op on GCC;
# the $<CXX_COMPILER_ID:MSVC> generator expression evaluates false and adds no
# flags, preserving future portability without affecting the current build.

add_library(mgie_warnings INTERFACE)

# --- GCC / Clang (host C++ compiler) ---
set(MGIE_GCC_WARNINGS
  -Wall -Wextra -Wshadow -Wfloat-equal -Wold-style-cast
  -Wconversion -Wsuggest-override -pedantic
  -Wno-deprecated-declarations
)

# --- MSVC (inert on GCC; kept for future Windows portability) ---
set(MGIE_MSVC_WARNINGS /W4 /permissive-)

# --- nvcc device warnings ---
# Forward host warnings through -Xcompiler; enable ptxas verbosity later when
# profiling register/shared-mem usage (Phase 1+).
set(MGIE_CUDA_WARNINGS --Werror=cross-execution-space-call)

if(MGIE_WARNINGS_AS_ERRORS)
  list(APPEND MGIE_GCC_WARNINGS -Werror)
  list(APPEND MGIE_MSVC_WARNINGS /WX)
endif()

target_compile_options(mgie_warnings INTERFACE
  # Apply GCC/Clang warnings to host C++ TUs.
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:${MGIE_GCC_WARNINGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:${MGIE_GCC_WARNINGS}>
  # Apply MSVC warnings to host C++ TUs (inert on GCC — evaluates false).
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${MGIE_MSVC_WARNINGS}>
  # Apply CUDA warnings to device TUs.
  $<$<COMPILE_LANGUAGE:CUDA>:${MGIE_CUDA_WARNINGS}>
)
