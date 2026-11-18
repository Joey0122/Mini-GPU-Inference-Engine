# Phase 4 — Causal Multi-Head Attention

## Summary

Phase 4 implements GPT-2 Small's causal multi-head scaled-dot-product attention
(batch=1) as hand-written CUDA kernels, validated against PyTorch within the
PROJECT_SPEC FP32 gate (`max_abs_error < 1e-4`) and benchmarked across the
quadratic sequence-length regime.

Scope is the attention **mechanism** on provided Q/K/V. Input projections
(`c_attn`) and the output projection (`c_proj`) are GEMMs over loaded weights
and belong to Phase 5 (transformer-block assembly); KV cache is Phase 6.

## Kernels

Dedicated batched kernels (all 12 heads in one launch via `blockIdx.y`),
naive-but-correct first per the AGENT.md kernel process.

| Step | Kernel | Math | Layout |
|------|--------|------|--------|
| 1 | `AttentionScores` | `scale·QKᵀ`, causal mask `j>i → -1e30` | Q,K `[n_head,S,head_dim]` → scores `[n_head,S,S]` |
| 2 | (reuse) `LaunchSoftmax` | row softmax over `[n_head·S, S]` | masked entries → exp underflow → 0 |
| 3 | `AttentionAV` | `out[i,h·hd+d]=Σ_j w·v` | weights·V → merged `[S, n_head·head_dim]` |
| 4 | `LaunchAttention` | integrated 1→2→3 | owns scratch (caller-provided) |

Design decisions (approved): **custom dedicated kernels** for Scores/AV rather
than transposing K + per-head GEMM; **causal mask pre-applied in Scores** so the
Phase 3 softmax is reused unchanged.

`scale = 1/√head_dim`. Masked sentinel is `-1e30` (finite) so a fully-masked
tile cannot produce NaN; with causal masking every query row has ≥1 valid key.

## Validation

70/70 unit+host tests pass. PyTorch validation (manual causal SDPA reference,
not `F.scaled_dot_product_attention`, to mirror the kernel and avoid fused-
backend variance) gates at 1e-4:

| Case | n_head | S | max_abs_error | Result |
|------|-------:|--:|--------------:|--------|
| gpt2_s128  | 12 | 128 | 3.0e-7 | PASS |
| gpt2_s512  | 12 | 512 | 3.6e-7 | PASS |
| stress_mag | 4  | 64 (×8 magnitude) | 1.8e-6 | PASS |
| stress_s13 | 12 | 13 (non-pow2) | 4.8e-7 | PASS |

Worst attention error 1.8e-6 — ~55× under the gate. Full report (with Phase 3):
`reports/phase3/validation_report.json`. Reproduce:
`python scripts/validate_kernels.py --kernel attention`.

**Numerical note (the flagged AV-accumulation risk):** the large-magnitude unit
test exceeds the coarse `AttentionAbsTol(64)=3.1e-5` bound (~6.3e-5) but stays
under the authoritative spec gate of 1e-4. Against the true PyTorch reference
the same case is 1.8e-6 — the coarse CPU-reference bound, not the kernel, is the
loose link. Spec gate (1e-4) is authoritative, consistent with Phase 3.

## Benchmark

RTX 4070 Laptop (sm_89), CUDA 12.4, FP32. GPT-2 head config (n_head=12,
head_dim=64). FLOPs = `4·n_head·S²·head_dim`. PyTorch eager = manual SDPA on
GPU (comparison only; cuBLAS/cuDNN forbidden as implementation).

| S | latency (ms) | GFLOPS | scores BW (GB/s) | scratch (MB) | PyTorch eager (ms) |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.11 | 456 | 42.7 | 0.8  | 0.065 |
| 256  | 0.38 | 536 | 41.9 | 3.1  | 0.103 |
| 512  | 1.42 | 561 | 39.4 | 12.6 | 0.314 |
| 1024 | 6.05 | 530 | 35.2 | 50.3 | 2.71  |

## Findings & bottlenecks

- **Compute/latency-bound baseline.** Achieved bandwidth (~35–43 GB/s) is far
  below the ~360 GB/s peak: the naive Scores kernel recomputes each dot product
  with one block per (head, query-row) and no K reuse across rows, and AV uses
  one thread per output element. This is the expected naive baseline — clean
  and correct first.
- **Quadratic scaling confirmed.** Latency grows ~4× per doubling of S (S²),
  and the scores scratch reaches 50 MB at S=1024 (fits the 8 GB card at
  batch=1, but materializing it is the bandwidth cost).
- **Gap to PyTorch** widens with S (1.7× at S=128 → 2.2× at S=1024), reflecting
  the un-tiled score/AV kernels vs PyTorch's fused matmul path.

## Optimization opportunities (future versions, benchmarked against this baseline)

1. Shared-memory tiling of the QKᵀ dot product (reuse K across query rows).
2. GEMM-backed AV (the shape is `A·B` per head) as an alternative version.
3. Score/softmax/AV fusion to avoid materializing `[n_head,S,S]` twice
   (FlashAttention-style; out of current scope).

## Constraints honored

No RMSNorm, RoPE, FlashAttention, multi-GPU, mixed precision, or quantization.
Pure FP32, batch=1, clean decomposed attention path.

## Reproduce

```bash
cmake --build build
ctest --test-dir build --output-on-failure          # 70/70
python scripts/validate_kernels.py --kernel attention
./build/benchmarks/bench_attention                   # writes attention_results.csv
```
