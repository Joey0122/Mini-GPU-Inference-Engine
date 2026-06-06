# AGENT.md

# Agent Instructions

Mini GPU Inference Engine

Read this document before making any code changes.

This repository is intended to build a CUDA-based Transformer inference engine from scratch.

Your objective is not merely to produce working code.

Your objective is to produce:

* Correct code
* Measurable code
* Explainable code
* Reproducible code

All implementation decisions should follow the rules below.

---

# Mission

Build a CUDA inference engine capable of running GPT-2 Small (124M).

The project exists to demonstrate:

* CUDA programming
* GPU optimization
* Transformer inference
* Performance engineering

The final artifact should resemble a simplified production inference system.

---

# Non-Negotiable Priorities

Priority order:

1. Correctness
2. Testing
3. Profiling
4. Optimization
5. Features

Never violate this order.

---

# Scope Lock

Target model:

GPT-2 Small (124M)

Required:

* Batch size = 1
* Greedy decoding
* Temperature sampling

Not required:

* Training
* Distributed execution
* Multi-GPU
* Tensor parallelism
* Pipeline parallelism
* MoE
* vLLM features
* Speculative decoding
* Serving infrastructure

Do not expand project scope.

---

# Required Reading Order

Before implementing anything, read:

1. PROJECT_SPEC.md
2. ARCHITECTURE.md
3. TASK_BREAKDOWN.md
4. IMPLEMENTATION_RULES.md

If instructions conflict:

PROJECT_SPEC.md has highest authority.

---

# Development Workflow

For every task:

Step 1:
Understand requirements.

Step 2:
Create CPU reference implementation.

Step 3:
Create validation tests.

Step 4:
Implement CUDA version.

Step 5:
Verify correctness.

Step 6:
Benchmark.

Step 7:
Profile.

Step 8:
Optimize.

Never skip steps.

---

# Phase Gate Rules

You may NOT begin the next phase until the current phase satisfies all exit criteria.

Example:

Phase 1 cannot begin until Phase 0 is complete.

Phase 2 cannot begin until Phase 1 is complete.

If uncertain, stop and report status.

---

# Kernel Development Process

Every CUDA kernel must follow this sequence.

Version A:

Naive implementation.

Version B:

Shared memory optimization.

Version C:

Warp-level optimization.

Version D:

Advanced optimization.

Examples:

Reduction:

Naive
→ Shared Memory
→ Warp Shuffle

GEMM:

Naive
→ Tiled
→ Register Tiled
→ Double Buffered
→ Tensor Core

Maintain all versions for benchmarking.

Do not delete previous versions.

---

# Correctness Requirements

Every kernel requires:

1. CPU reference
2. Unit tests
3. PyTorch validation

Validation thresholds:

FP32:

max_abs_error < 1e-4

FP16:

max_abs_error < 1e-2

Do not optimize incorrect code.

---

# Benchmark Requirements

Every kernel must have a benchmark.

Benchmark output must include:

* kernel name
* input size
* execution time
* throughput
* bandwidth
* GFLOPS

Benchmark results must be saved.

---

# Profiling Requirements

Before proposing an optimization:

Collect evidence.

Use:

* Nsight Compute
* Nsight Systems

Required metrics:

* occupancy
* warp execution efficiency
* achieved bandwidth
* memory throughput
* SM utilization

Do not speculate.

Use measured data.

---

# Engineering Standards

Favor:

* simple designs
* clear abstractions
* deterministic behavior

Avoid:

* premature optimization
* unnecessary templates
* excessive metaprogramming
* hidden magic

Code should be understandable by a student reading it six months later.

---

# CUDA Rules

Always use:

CUDA_CHECK(...)

Never ignore:

cudaError_t

Avoid:

* hardcoded launch dimensions
* duplicated kernel launch code

Prefer reusable launch wrappers.

---

# Memory Rules

Minimize allocations.

Avoid:

cudaMalloc inside hot paths.

Prefer:

Initialization-time allocation.

Inference-time reuse.

Document ownership of all memory.

---

# Logging Rules

Use centralized logging.

Allowed levels:

INFO
WARN
ERROR
DEBUG

Avoid scattered printf statements.

Temporary debugging prints should be removed before completion.

---

# Testing Rules

For every feature:

Create tests first.

Required test categories:

Unit Tests

Integration Tests

Reference Tests

PyTorch Validation

New code without tests is incomplete.

---

# Documentation Rules

Every major task must update:

README

and

reports/

Documentation must explain:

1. What was built
2. Why it works
3. Benchmark results
4. Bottlenecks
5. Optimizations
6. Lessons learned

---

# Commit Rules

Create small commits.

Examples:

feat(reduction): naive implementation

feat(reduction): shared memory optimization

feat(gemm): tiled kernel

fix(softmax): numerical stability issue

Avoid large mixed-purpose commits.

---

# Forbidden Actions

Do NOT:

* copy CUTLASS kernels
* copy FlashAttention kernels
* copy TensorRT code
* use Triton kernels
* replace custom kernels with cuBLAS
* replace custom kernels with cuDNN

Reference implementations may be studied.

Implementations must be original.

---

# Allowed Dependencies

Allowed:

* CUDA Toolkit
* C++17
* CMake
* GoogleTest
* PyTorch (validation only)
* matplotlib (benchmark plotting)
* numpy

Use additional dependencies only when clearly justified.

---

# Reporting Format

At the end of each completed task, provide:

## Completed Work

What was implemented.

## Validation

Tests performed.

## Benchmark Results

Measured performance.

## Profiling Findings

Observed bottlenecks.

## Next Recommended Step

Exactly one next task.

---

# Failure Handling

If blocked:

Do not guess.

Do not fabricate.

Instead:

1. Explain the blocker.
2. Explain why it occurs.
3. Propose solutions.
4. Recommend the best option.

---

# Success Definition

Project success requires:

✓ GPT-2 Small runs end-to-end

✓ Text generation works

✓ Core kernels implemented manually

✓ Benchmarks collected

✓ Profiling reports collected

✓ Technical writeup completed

✓ Repository is portfolio-ready

Until all requirements are satisfied,
the project is not complete.

---

# Final Instruction

Act as a GPU systems engineer.

Do not act as a code generator.

Every code change should improve one of:

* correctness
* performance
* maintainability
* observability
* reproducibility

If a change does not improve one of these,
do not make it.
