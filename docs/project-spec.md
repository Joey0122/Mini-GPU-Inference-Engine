# Original project specification

The [project roadmap](roadmap.md) tracks which requirements the FP32 release implements.

## Project Objective

Build a CUDA inference engine capable of running GPT-2 124M for autoregressive text generation.

The purpose of this project is NOT merely to generate text.

The primary goal is to demonstrate understanding of:

* CUDA programming model
* GPU memory hierarchy
* Shared-memory optimization
* Warp-level programming
* Tensor Core programming
* Transformer inference
* Performance profiling
* Numerical validation

The final repository should resemble a simplified version of production inference systems such as:

* PyTorch Inference
* TensorRT
* vLLM
* llama.cpp

while keeping all critical CUDA kernels self-implemented.

---

# Success Criteria

The project is considered complete only if all requirements below are satisfied.

## Functional Requirements

The engine must:

1. Load GPT-2 124M weights
2. Run forward inference on GPU
3. Support batch size = 1
4. Support arbitrary prompt lengths
5. Generate text using greedy decoding
6. Generate text using temperature sampling

---

## CUDA Requirements

The following kernels MUST be written manually:

* Reduction
* Prefix Scan
* Transpose
* GEMM
* LayerNorm
* Softmax
* GELU
* Attention

Calling cuBLAS is allowed ONLY for benchmarking.

Calling cuBLAS as the primary implementation is forbidden.

---

## Validation Requirements

Every kernel must pass numerical validation against PyTorch.

Required tolerances:

FP32:
max_abs_error < 1e-4

FP16:
max_abs_error < 1e-2

No optimization work should begin until correctness is verified.

---

## Performance Requirements

Target performance goals:

Reduction:

> = 80% memory bandwidth efficiency

FP32 GEMM:

> = 50% cuBLAS performance

FP16 Tensor Core GEMM:

> = 30% cuBLAS performance

End-to-End GPT-2:
Measure tokens/sec against HuggingFace Transformers.

Matching Transformers performance is NOT required.

---

## Forbidden Shortcuts

Do NOT use:

* CUTLASS kernels
* Triton kernels
* FlashAttention implementation copies
* TensorRT
* vLLM
* Existing CUDA inference engines

Reference implementations may be studied but not copied.

---

# Deliverables

1. Source code
2. Unit tests
3. Benchmarks
4. Nsight profiling reports
5. Technical writeup
6. Final presentation material
