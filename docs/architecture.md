# Architecture and engineering design

These notes record the original system design, including planned components.
The [project roadmap](roadmap.md) lists the components shipped in the FP32 release.

---

# Design Philosophy

This project follows five principles:

1. Correctness before performance
2. Simplicity before optimization
3. Measurement before conclusions
4. Reproducibility before claims
5. Documentation before polish

Every implementation decision should be justified by profiling data, benchmark results, or engineering constraints.

---

# High-Level System Architecture

```text
+------------------------------------------------+
|                    CLI Layer                   |
+------------------------------------------------+
                     |
                     v
+------------------------------------------------+
|               Inference Engine                 |
+------------------------------------------------+
| Tokenizer | Decoder | Sampler | KV Cache       |
+------------------------------------------------+
                     |
                     v
+------------------------------------------------+
|             Transformer Runtime                |
+------------------------------------------------+
| Embedding                                     |
| Attention                                     |
| MLP                                           |
| LayerNorm                                     |
+------------------------------------------------+
                     |
                     v
+------------------------------------------------+
|                CUDA Kernel Layer               |
+------------------------------------------------+
| GEMM | Softmax | Reduction | LayerNorm | GELU |
+------------------------------------------------+
                     |
                     v
+------------------------------------------------+
|                  CUDA Runtime                  |
+------------------------------------------------+
| Streams | Events | Memory Management          |
+------------------------------------------------+
```

---

# Planned Repository Structure

```text
mini-gpu-inference/

├── CMakeLists.txt

├── README.md

├── docs/
│
├── benchmarks/
│   ├── reduction/
│   ├── transpose/
│   ├── gemm/
│   ├── softmax/
│   └── attention/
│
├── data/
│   ├── test_vectors/
│   └── model_weights/
│
├── include/
│
│   ├── common/
│   │   ├── types.hpp
│   │   ├── errors.hpp
│   │   └── timer.hpp
│   │
│   ├── cuda/
│   │   ├── launch.hpp
│   │   ├── device_info.hpp
│   │   └── memory.hpp
│   │
│   ├── kernels/
│   │   ├── reduction.hpp
│   │   ├── scan.hpp
│   │   ├── transpose.hpp
│   │   ├── gemm.hpp
│   │   ├── softmax.hpp
│   │   ├── layernorm.hpp
│   │   └── attention.hpp
│   │
│   └── model/
│       ├── tensor.hpp
│       ├── transformer.hpp
│       ├── gpt2.hpp
│       └── loader.hpp
│
├── src/
│
│   ├── kernels/
│   │   ├── reduction.cu
│   │   ├── scan.cu
│   │   ├── transpose.cu
│   │   ├── gemm.cu
│   │   ├── softmax.cu
│   │   ├── layernorm.cu
│   │   └── attention.cu
│   │
│   ├── runtime/
│   │   ├── allocator.cu
│   │   ├── streams.cu
│   │   └── profiler.cu
│   │
│   ├── model/
│   │   ├── tensor.cpp
│   │   ├── transformer.cpp
│   │   ├── gpt2.cpp
│   │   └── loader.cpp
│   │
│   └── main.cpp
│
├── tests/
│
│   ├── unit/
│   ├── integration/
│   ├── reference/
│   └── pytorch/
│
├── scripts/
│   ├── generate_reference.py
│   ├── export_gpt2.py
│   └── benchmark.py
│
└── reports/
    ├── phase1/
    ├── phase2/
    ├── phase3/
    └── final/
```

---

# Layer Responsibilities

## Kernel Layer

Contains only CUDA kernels.

Allowed:

* CUDA kernels
* device functions
* launch wrappers

Forbidden:

* model logic
* tokenizer logic
* file IO

---

## Runtime Layer

Responsible for:

* memory allocation
* streams
* events
* profiling utilities

No model-specific code allowed.

---

## Model Layer

Responsible for:

* Transformer blocks
* GPT-2 architecture
* weight loading

No CUDA kernel implementation allowed.

Model code should call kernel interfaces.

---

# Tensor Abstraction

All kernels should operate through a common tensor structure.

Example:

```cpp
struct Tensor
{
    void* data;

    std::vector<int> shape;

    DType dtype;

    DeviceType device;
};
```

Avoid passing raw pointers throughout the entire codebase.

---

# Naming Conventions

Files:

```text
snake_case
```

Examples:

```text
layer_norm.cu
tensor_loader.cpp
```

---

Functions:

```cpp
PascalCase
```

Examples:

```cpp
RunSoftmax()
ComputeAttention()
LoadWeights()
```

---

Variables:

```cpp
snake_case
```

Examples:

```cpp
batch_size
hidden_dim
num_heads
```

---

Constants:

```cpp
UPPER_CASE
```

Examples:

```cpp
WARP_SIZE
MAX_THREADS
```

---

# Error Handling

Never ignore CUDA return values.

Every CUDA call must use:

```cpp
CUDA_CHECK(...)
```

Example:

```cpp
CUDA_CHECK(cudaMalloc(...));
```

Create a centralized macro:

```cpp
#define CUDA_CHECK(...)
```

inside:

```text
include/common/errors.hpp
```

---

# Logging

Use a single logging interface.

Levels:

```text
INFO
WARN
ERROR
DEBUG
```

No printf scattered across kernels.

---

# Testing Strategy

Three levels of testing are required.

---

## Unit Tests

Test individual kernels.

Examples:

* reduction
* softmax
* layernorm

Inputs should be small and deterministic.

---

## Reference Tests

Compare against CPU implementation.

Example:

```text
CUDA Reduction
vs
CPU Reduction
```

---

## PyTorch Validation

Compare against PyTorch outputs.

Required before optimization.

Example:

```text
CUDA Softmax
vs
torch.softmax()
```

---

# Benchmark Framework

Every benchmark should output:

```text
Kernel Name

Input Shape

Execution Time

Bandwidth

GFLOPS

Occupancy
```

Results should be exportable to:

```text
CSV
```

for plotting.

---

# Profiling Requirements

Every major kernel must include:

1. Nsight Compute report
2. Occupancy analysis
3. Memory throughput analysis
4. Bottleneck summary

Store reports under:

```text
reports/
```

---

# CI Requirements

Every commit should automatically run:

1. Build
2. Unit tests
3. Reference tests

Future workflow:

```text
GitHub Actions
```

Pipeline:

Build
→ Test
→ Validate
→ Merge

---

# Documentation Requirements

Every completed phase must produce:

## Summary

What was built

## Results

Benchmarks

## Problems

Observed bottlenecks

## Fixes

Implemented optimizations

## Lessons Learned

Engineering insights

---

# GPT-2 Scope Lock

To prevent scope creep:

Target Model:

GPT-2 Small (124M)

Supported:

* Batch size 1
* Greedy decoding
* Temperature sampling

Not Required:

* Training
* Distributed execution
* Multi-GPU
* Mixture of Experts
* Speculative decoding

Do not expand scope until GPT-2 inference works end-to-end.

---

# Definition of Done

A phase is complete only if:

✓ Code compiles

✓ Tests pass

✓ Benchmark exists

✓ Results documented

✓ Git commit created

✓ Report updated

If any item is missing,
the phase is not complete.
