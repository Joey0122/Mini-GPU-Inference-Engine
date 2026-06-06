# TASK_BREAKDOWN.md

# Agent Task Roadmap

Complete tasks sequentially.

Do not skip phases.

---

# Phase 0

Environment Setup

Tasks:

* Install CUDA
* Configure CMake
* Run deviceQuery
* Implement vector add kernel
* Create benchmark framework

Exit Criteria:

* CUDA compilation works
* Benchmark framework operational

---

# Phase 1

GPU Primitives

Tasks:

1. Reduction

   * Naive
   * Shared Memory
   * Warp Shuffle

2. Prefix Scan

3. Matrix Transpose

Exit Criteria:

* Unit tests pass
* Benchmarks generated
* Profiling screenshots collected

---

# Phase 2

GEMM

Tasks:

1. Naive GEMM

2. Shared Memory Tiling

3. Register Tiling

4. Double Buffering

5. Tensor Core WMMA Version

Exit Criteria:

* Correctness verified
* cuBLAS comparison produced
* Optimization report written

---

# Phase 3

Transformer Components

Tasks (required for GPT-2 Small):

1. LayerNorm

2. GELU

3. Softmax

Out of scope for GPT-2 Small (Llama-family only — stretch / future extension if scope is later expanded):

* RMSNorm
* RoPE

Exit Criteria:

* All required kernels validated against PyTorch

---

# Phase 4

Attention

Tasks:

1. QK^T

2. Softmax

3. Attention Output

4. Integrated Attention Kernel

Optional:

5. FlashAttention-style implementation

Exit Criteria:

* Attention outputs match PyTorch

---

# Phase 5

GPT-2 Inference

Tasks:

1. Weight Loader

2. Embedding Layer

3. Transformer Block

4. Full Forward Pass

5. Greedy Decoding

6. Temperature Sampling

Exit Criteria:

* GPT-2 generates coherent text

---

# Phase 6

Optimization

Tasks:

1. Bottleneck Analysis

2. Kernel Fusion

3. Memory Optimization

4. CUDA Graphs (optional)

5. KV Cache

Exit Criteria:

* Final benchmark report completed

---

# Phase 7

Publication

Tasks:

1. README

2. Architecture Diagram

3. Benchmark Figures

4. Profiling Figures

5. Technical Blog Post

6. Resume Project Description

Project Complete.
