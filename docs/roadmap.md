# Project status

The first public version runs GPT-2 Small (124M) FP32 inference and text
generation at batch size 1 with a 1,024-token context. The
[README](../README.md) has the demo and reproduction steps. The
[release report](../reports/final/fp32-release.md) records the release measurements.

| Phase | Implemented |
| --- | --- |
| 0–1 | CUDA build and test harness, vector addition, FP32 reduction |
| 2 | Tiled and register-tiled FP32 GEMM with cuBLAS benchmarking |
| 3–4 | LayerNorm, softmax, GELU, causal multi-head attention |
| 5 | Weight export and loading, complete GPT-2 forward pass, greedy decoding and temperature sampling |
| 6a–6b | KV-cache prefill and decode, hand-written GEMV dispatch for single-row projections |

At generation time, workspace rows are sized by the prompt. Each cached
step reuses the first hidden row and can attend to the full context.

## Remaining project specifications

- Prefix Scan and a standalone Transpose kernel.
- FP16 Tensor Core GEMM and numerical validation.
- End-to-end tokens/s comparison with Hugging Face Transformers.
- Further decode optimization and profiling.

[Earlier phase reports](../reports/README.md) document the designs and measurements
at each milestone.
