# Phase 3 — PyTorch Validation Report

## Summary

The Phase 3 transformer-component kernels (LayerNorm, Softmax, GELU) are
validated against PyTorch references through a file-based pipeline that runs the
**actual committed CUDA kernels** and diffs their output against PyTorch within
the PROJECT_SPEC FP32 tolerance (`max_abs_error < 1e-4`).

This closes the gap left by the earlier CPU-only validation: PROJECT_SPEC and
IMPLEMENTATION_RULES Rule 8 require validation against PyTorch specifically.

## Pipeline

```
scripts/generate_reference.py --kernel <name>   # PyTorch refs -> data/test_vectors/<name>/*.npy
tests/validation/validate_runner.cu             # runs real kernels, diffs, gates at 1e-4
scripts/validate_kernels.py --kernel all         # orchestrates + prints + gates (CI exit code)
reports/phase3/validation_report.json            # machine-readable results
```

- **Reference math** matches the kernels exactly: `F.layer_norm` (biased
  variance ÷H), `torch.softmax(dim=-1)`, `F.gelu(approximate='tanh')` (GPT-2
  NewGELU).
- **Both kernel versions** are validated where they exist (LayerNorm
  Naive/Welford, Softmax Naive/Fused).
- **Representative + stress shapes**: GPT-2 sizes plus a non-warp-multiple
  LayerNorm width (13×769), large-magnitude softmax inputs (×8) to exercise the
  max-subtraction stability path, and wide-range GELU inputs (×6) for
  saturation.

## Results (gate = 1e-4)

| Kernel | Version | Shape | max_abs_error | Result |
|--------|---------|-------|--------------:|--------|
| LayerNorm | naive   | 256×768  | 9.5e-7 | PASS |
| LayerNorm | welford | 256×768  | 9.5e-7 | PASS |
| LayerNorm | naive   | 32×3072  | 9.5e-7 | PASS |
| LayerNorm | welford | 32×3072  | 9.5e-7 | PASS |
| LayerNorm | naive   | 13×769   | 9.5e-7 | PASS |
| LayerNorm | welford | 13×769   | 9.5e-7 | PASS |
| Softmax   | naive   | 64×4096  | 1.9e-9 | PASS |
| Softmax   | fused   | 64×4096  | 1.9e-9 | PASS |
| Softmax   | naive   | 64×4096 (×8) | 6.0e-7 | PASS |
| Softmax   | fused   | 64×4096 (×8) | 6.0e-7 | PASS |
| GELU      | newgelu | 64×3072  | 2.4e-7 | PASS |
| GELU      | newgelu | 32×1024 (×6) | 2.4e-7 | PASS |

12/12 cases pass; worst error 9.5e-7 — ~105× under the gate.

## Tolerance reconciliation

The authoritative gate is PROJECT_SPEC's FP32 `1e-4` absolute. To justify it
empirically, the float32-vs-float64 error floor of each PyTorch reference was
measured:

| Kernel | Shape | max_abs (fp32 vs fp64) |
|--------|-------|-----------------------:|
| LayerNorm | 256×768 | 8.1e-7 |
| LayerNorm | 13×769  | 6.4e-7 |
| Softmax   | 64×4096 (×5) | 1.6e-6 |
| GELU      | 64×3072 | 2.9e-7 |

The intrinsic float32 error floor (~1e-6) is two orders of magnitude under the
gate, so `1e-4` is both spec-compliant and comfortably achievable. The
`LayerNormAbsTol(H) = 4·H·1.2e-7` formula in `test_utils.hpp` (≈3.7e-4 at
H=768) is a worst-case adversarial accumulation bound retained as a coarse
CPU-reference smoke check; it is intentionally **not** the spec gate.

## Deferred

50257-vocabulary logits softmax is deferred to Phase 5 (a sampling concern, not
a Phase 3 transformer-component requirement). The committed softmax kernel is
generic `[N, D]`, so no kernel change is needed when that case is added.

## Reproduce

```bash
cmake --build build
python scripts/validate_kernels.py --kernel all
```
