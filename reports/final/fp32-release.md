# GPT-2 Small FP32 release verification

## Environment

| Item | Tested configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB, sm_89 |
| Platform | WSL2 Ubuntu on Windows, NVIDIA driver 596.49 |
| Build | CUDA Toolkit 12.4.131, GCC 13.4.0, CMake 4.2.3, Ninja 1.13.2, Release, sm_89 |
| Python reference | Python 3.14.4, PyTorch 2.12.0+cpu, Transformers 5.10.2, four OpenMP threads |
| Checkpoint | [openai-community/gpt2 at 607a30d](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) |

## Generation and correctness

The [README](../../README.md) records the exact text command and its stdout.
The C++ executable loaded 149 tensors from the 652,148,736-byte FP32 export.
Python encoded and decoded the token IDs. The benchmark and tests used the
same release build of the CUDA engine.

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| CTest (27 CPU, 113 GPU, 1 PyTorch fixture runner) | 141 | 0 | 0 |
| Hugging Face golden comparison (4 logits, 4 greedy, 3 sampling) | 11 | 0 | 0 |

| Logits prompt | Positions with matching argmax | Max absolute error |
| --- | ---: | ---: |
| Singleton | 1/1 | 1.297e-4 |
| Short | 4/4 | 1.755e-4 |
| Medium | 16/16 | 2.594e-4 |
| Long | 32/32 | 2.594e-4 |

Each position compares all 50,257 FP32 logits against Hugging Face on CPU
with a strict maximum absolute error threshold of 5e-3. Greedy decoding
matched all 25 new tokens for each of the four prompts, 100/100 total. The
sampling cases checked the Hugging Face probability of each CUDA-selected
token on its actual trajectory. All 60 were above 1e-12.

The one-row workspace regression covered multiple cached steps from a one-token
prompt and compared logits to full recomputation. The boundary regression
filled a five-token context from a one-token prompt, capped a request for
`INT_MAX` new tokens, and rejected an out-of-range direct decode step.

The negative golden checks each returned exit status 1. The stricter
threshold failed all four logits cases, with all 53 argmax positions still
matching.

| Induced failure | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Remove one required greedy output | 10 | 1 | 0 |
| Supply an empty output directory | 0 | 11 | 0 |
| Lower the absolute error threshold to `1e-8` | 7 | 4 | 0 |

## 4096³ FP32 GEMM

```bash
./build/benchmarks/bench_gemm \
  --benchmark_filter='BM_Gemm_(Reg128Pad|CuBLAS)/4096' \
  --benchmark_min_time=0.3s
```

Square row-major matrices with M = N = K = 4096, preallocated device buffers
and FP32 inputs, accumulation, and output for both paths. The baseline uses
`cublasSgemm` with `CUBLAS_PEDANTIC_MATH` (strict FP32). The benchmark CSV
reports per-launch CUDA-event times from a final batch of 50 measurements
after 10 warmups. Throughput is `2MNK` divided by the mean device time.
GPU clocks were left at their default settings. Weight loading, transfers
and model generation are outside this measurement.

| Implementation | Mean (ms) | Std. dev. (ms) | TFLOP/s |
| --- | ---: | ---: | ---: |
| Register-tiled CUDA (`RegTiled128Pad`) | 21.333101 | 0.154741 | 6.442521 |
| cuBLAS SGEMM | 12.523138 | 0.136493 | 10.974801 |

The hand-written kernel reached **58.7%** of cuBLAS throughput on this shape
in the November 18, 2026 run. See the [Phase 2 report](../phase2/gemm_optimization.md)
for the original tiling experiments.
