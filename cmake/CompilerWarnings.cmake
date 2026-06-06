# cmake/CompilerWarnings.cmake
#
# Defines an INTERFACE target `mgie_warnings` that carries the project's warning
# flags for both the C++ (MSVC) and CUDA (nvcc) compilers. Centralizing warnings
# here keeps IMPLEMENTATION_RULES code-quality expectations consistent across
# every target and lets CI flip warnings-as-errors via MGIE_WARNINGS_AS_ERRORS
# without touching individual targets.

add_library(mgie_warnings INTERFACE)

# --- MSVC (host C++ compiler under Visual Studio 2022) ---
set(MGIE_MSVC_WARNINGS /W4 /permissive-)

# --- nvcc device warnings ---
# Forward host warnings through -Xcompiler; enable ptxas verbosity later when
# profiling register/shared-mem usage (Phase 1+).
set(MGIE_CUDA_WARNINGS --Werror=cross-execution-space-call)

if(MGIE_WARNINGS_AS_ERRORS)
  list(APPEND MGIE_MSVC_WARNINGS /WX)
endif()

target_compile_options(mgie_warnings INTERFACE
  # Apply MSVC warnings to host C++ TUs only.
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${MGIE_MSVC_WARNINGS}>
  # Apply CUDA warnings to device TUs; forward host flags through -Xcompiler.
  $<$<COMPILE_LANGUAGE:CUDA>:${MGIE_CUDA_WARNINGS}>
)
