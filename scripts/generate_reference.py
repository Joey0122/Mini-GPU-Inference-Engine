#!/usr/bin/env python3
"""scripts/generate_reference.py — PyTorch reference generator.

Writes kernel inputs and PyTorch reference outputs to data/test_vectors/ as
.npy files. The C++ validation runner (tests/validation/validate_runner.cu)
loads these, runs the actual committed CUDA kernel, and diffs against the
reference within the spec tolerance (FP32 max_abs_error < 1e-4). This decouples
reference generation (PyTorch) from kernel validation (CUDA); the file-based
vectors are reused in Phases 4-5.

Layout:
    data/test_vectors/vector_add_*.npy           (Phase 0, flat)
    data/test_vectors/<kernel>/<case>/*.npy       (Phase 3+)

References match the committed kernels exactly:
    layernorm -> F.layer_norm  (biased variance / H, NOT torch.var)
    softmax   -> torch.softmax(dim=-1)
    gelu      -> F.gelu(approximate='tanh')  (GPT-2 NewGELU)

Usage:
    python scripts/generate_reference.py --kernel layernorm
    python scripts/generate_reference.py --kernel all
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

TEST_VECTOR_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "test_vectors"

# Shared eps; matches LaunchLayerNorm's default and the kernel.
EPS = 1e-5

# (case_name, N, H, seed)
LAYERNORM_CASES = [("gpt2", 256, 768, 0), ("ffn", 32, 3072, 1), ("stress", 13, 769, 2)]
# (case_name, N, D, seed, scale) — scale stresses the max-subtraction stability path.
SOFTMAX_CASES = [("attn", 64, 4096, 10, 1.0), ("stress", 64, 4096, 11, 8.0)]
# (case_name, N, D, seed, scale) — scale stresses GELU saturation.
GELU_CASES = [("mlp", 64, 3072, 20, 1.0), ("stress", 32, 1024, 21, 6.0)]
# (case_name, n_head, S, head_dim, seed, scale). GPT-2 Small: n_head=12,
# head_dim=64, n_ctx=1024. scale stresses softmax stability through the mask.
ATTENTION_CASES = [
    ("gpt2_s128", 12, 128, 64, 30, 1.0),
    ("gpt2_s512", 12, 512, 64, 31, 1.0),
    ("stress_mag", 4, 64, 64, 32, 8.0),
    ("stress_s13", 12, 13, 64, 33, 1.0),
]
# (case_name, S, seed). GPT-2 Small vocab=50257, n_embd=768, n_ctx=1024.
EMBEDDING_CASES = [
    ("s1",     1,    40),
    ("s64",   64,   41),
    ("s256", 256,   42),
    ("stress_s1k", 1024, 43),
]
# (case_name, N, D, seed). Covers GPT-2 bias shapes: c_proj (768), c_attn (2304),
# mlp_fc (3072), and ln_f (N=1).
BIAS_ADD_CASES = [
    ("single_row",    1,  768, 50),
    ("gpt2_proj",    64,  768, 51),
    ("c_attn",       64, 2304, 52),
    ("mlp_fc",       64, 3072, 53),
    ("full_ctx",   1024,  768, 54),
]
# (case_name, S, n_head, head_dim, seed). GPT-2 Small: n_head=12, head_dim=64,
# n_embd = n_head * head_dim = 768. x [S, 2304] → q/k/v [12, S, 64].
QKV_SPLIT_CASES = [
    ("single_token",    1,  12, 64, 60),
    ("short_seq",      64,  12, 64, 61),
    ("medium_seq",    256,  12, 64, 62),
    ("full_ctx",      1024, 12, 64, 63),
    ("nontrivial",     17,   4, 32, 64),  # non-GPT2 shapes: exercises generic path
]
# (case_name, S, seed). GPT-2 Small block: n_head=12, head_dim=64, n_embd=768,
# mlp_inner=3072, ln_eps=1e-5. Each case saves hidden_in + all 14 weight tensors
# + expected hidden_out.
BLOCK_CASES = [
    ("single_token",  1, 70),
    ("short_seq",    64, 71),
]
# (case_name, S, seed). Full forward pass. S=1 to keep the synthetic case
# focused on a single logit row. The authoritative validation is the golden test with
# real GPT-2 weights (scripts/golden_test.py), not this synthetic reference.
FORWARD_CASES = [
    ("s1",  1, 80),
]


def _save(case_dir: pathlib.Path, **arrays) -> None:
    """Write each named array as a C-contiguous float32 .npy in case_dir."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(case_dir / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))


def gen_vector_add(n: int = 1 << 16, seed: int = 0) -> None:
    """Phase 0 numpy reference (no torch) — proves the save/load plumbing."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    TEST_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TEST_VECTOR_DIR / "vector_add_a.npy", a)
    np.save(TEST_VECTOR_DIR / "vector_add_b.npy", b)
    np.save(TEST_VECTOR_DIR / "vector_add_ref.npy", a + b)
    print(f"wrote vector_add reference (n={n})")


def gen_layernorm() -> None:
    import torch
    import torch.nn.functional as F
    for name, N, H, seed in LAYERNORM_CASES:
        torch.manual_seed(seed)
        x = torch.randn(N, H)
        gamma = torch.rand(H) + 0.5            # ~[0.5, 1.5]
        beta = (torch.rand(H) - 0.5) * 0.2     # ~[-0.1, 0.1]
        ref = F.layer_norm(x, (H,), gamma, beta, EPS)  # biased var / H
        _save(TEST_VECTOR_DIR / "layernorm" / name,
              x=x.numpy(), gamma=gamma.numpy(), beta=beta.numpy(), ref=ref.numpy())
        print(f"wrote layernorm/{name} ({N}x{H})")


def gen_softmax() -> None:
    import torch
    for name, N, D, seed, scale in SOFTMAX_CASES:
        torch.manual_seed(seed)
        x = torch.randn(N, D) * scale
        ref = torch.softmax(x, dim=-1)
        _save(TEST_VECTOR_DIR / "softmax" / name, x=x.numpy(), ref=ref.numpy())
        print(f"wrote softmax/{name} ({N}x{D}, scale={scale})")


def gen_gelu() -> None:
    import torch
    import torch.nn.functional as F
    for name, N, D, seed, scale in GELU_CASES:
        torch.manual_seed(seed)
        x = torch.randn(N, D) * scale
        ref = F.gelu(x, approximate="tanh")    # GPT-2 NewGELU
        _save(TEST_VECTOR_DIR / "gelu" / name, x=x.numpy(), ref=ref.numpy())
        print(f"wrote gelu/{name} ({N}x{D}, scale={scale})")


def gen_attention() -> None:
    import math
    import torch
    import torch.nn.functional as F
    for name, n_head, S, head_dim, seed, scale in ATTENTION_CASES:
        torch.manual_seed(seed)
        # [n_head, S, head_dim] per-head-contiguous, matching the kernel layout.
        q = torch.randn(n_head, S, head_dim) * scale
        k = torch.randn(n_head, S, head_dim) * scale
        v = torch.randn(n_head, S, head_dim)
        # Manual SDPA (NOT F.scaled_dot_product_attention) so the reference
        # mirrors the kernel math exactly and avoids backend-fused variance.
        s = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)   # [n_head, S, S]
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(causal, float("-inf"))
        w = torch.softmax(s, dim=-1)                          # [n_head, S, S]
        o = w @ v                                             # [n_head, S, head_dim]
        # Merge heads -> [S, n_head*head_dim] (out[i, h*head_dim+d]).
        ref = o.permute(1, 0, 2).reshape(S, n_head * head_dim)
        _save(TEST_VECTOR_DIR / "attention" / name,
              q=q.numpy(), k=k.numpy(), v=v.numpy(), ref=ref.numpy())
        print(f"wrote attention/{name} (n_head={n_head} S={S} hd={head_dim} scale={scale})")


def gen_embedding() -> None:
    import torch
    import torch.nn.functional as F
    for name, S, seed in EMBEDDING_CASES:
        torch.manual_seed(seed)
        token_ids = torch.randint(0, 50257, (S,))
        wte = torch.randn(50257, 768)
        wpe = torch.randn(1024, 768)
        ref = F.embedding(token_ids, wte) + wpe[:S]
        _save(TEST_VECTOR_DIR / "embedding" / name,
              token_ids=token_ids.numpy().astype(np.int32),
              wte=wte.numpy(), wpe=wpe.numpy(), ref=ref.numpy())
        print(f"wrote embedding/{name} (S={S})")


def gen_bias_add() -> None:
    import torch
    for name, N, D, seed in BIAS_ADD_CASES:
        torch.manual_seed(seed)
        x = torch.randn(N, D)
        bias = torch.randn(D)
        ref = x + bias
        _save(TEST_VECTOR_DIR / "bias_add" / name,
              x=x.numpy(), bias=bias.numpy(), ref=ref.numpy())
        print(f"wrote bias_add/{name} (N={N} D={D})")


def gen_qkv_split() -> None:
    """Simulate c_attn output [S, 3*n_embd] and split into head-major Q/K/V.

    Mirrors the exact HuggingFace Conv1D c_attn output layout:
        x = hidden @ c_attn_w + c_attn_b         # [S, 3*n_embd]
        q_raw, k_raw, v_raw = x.split(n_embd, dim=1)  # [S, n_embd] each
        q = q_raw.view(S, n_head, head_dim).permute(1, 0, 2).contiguous()
        ...

    The permute(1,0,2) followed by .contiguous() produces the head-major layout
    that LaunchAttention expects: q[(h * S + s) * head_dim + d].
    """
    import torch
    for name, S, n_head, head_dim, seed in QKV_SPLIT_CASES:
        torch.manual_seed(seed)
        n_embd = n_head * head_dim
        x = torch.randn(S, 3 * n_embd)           # [S, 3*n_embd]
        # Exact HF semantics: split → view → permute → contiguous (data copy)
        q_raw, k_raw, v_raw = x.split(n_embd, dim=1)   # each [S, n_embd]
        q = q_raw.view(S, n_head, head_dim).permute(1, 0, 2).contiguous()  # [n_head, S, head_dim]
        k = k_raw.view(S, n_head, head_dim).permute(1, 0, 2).contiguous()
        v = v_raw.view(S, n_head, head_dim).permute(1, 0, 2).contiguous()
        _save(TEST_VECTOR_DIR / "qkv_split" / name,
              x=x.numpy(), q=q.numpy(), k=k.numpy(), v=v.numpy())
        print(f"wrote qkv_split/{name} (S={S} n_head={n_head} head_dim={head_dim})")


def gen_transformer_block() -> None:
    """Manual PyTorch GPT-2 Pre-LN transformer block — one block end-to-end.

    Mirrors the exact sequence in LaunchTransformerBlock:
        ln1 → c_attn(GEMM+ bias) → QKV split → attn → c_proj(GEMM+ bias) → +residual
        ln2 → mlp_fc(GEMM+ bias) → GELU → mlp_proj(GEMM+ bias) → +residual

    All operations match the GPU kernels:
      - F.layer_norm (biased var, eps=1e-5) = LaunchLayerNorm(Naive)
      - x @ w + b = LaunchGemm + LaunchBiasAdd
      - split→view→permute = LaunchQkvSplit
      - manual SDPA = LaunchAttention
      - F.gelu(approximate='tanh') = LaunchGelu

    Weights are generated randomly (not loaded from HF — that's a separate
    real-weights integration test). This validates the block wiring, not the
    weight values.
    """
    import math
    import torch
    import torch.nn.functional as F
    for name, S, seed in BLOCK_CASES:
        torch.manual_seed(seed)
        D = 768          # n_embd
        H = 12           # n_head
        hd = 64          # head_dim
        M = 3072         # mlp_inner
        eps = 1e-5

        # Generate synthetic weights with scaled variance to prevent exploding
        # activations. Untrained N(0,1) weights would blow up to 10^4 range
        # through the 3072-length MLP accumulation, making absolute-error gates
        # meaningless. Xavier-like scaling keeps outputs in [-10,10] range.
        # (This is a validation-only concern — real GPT-2 weights are already scaled.)
        hidden_in = torch.randn(S, D) * 0.02   # embedding-scale input
        ln1_w = torch.randn(D) * 0.02 + 1.0   # gamma ~1
        ln1_b = torch.randn(D) * 0.02          # beta ~0 (small)
        c_attn_w = torch.randn(D, 3 * D) * 0.02       # [K=768, N=2304]
        c_attn_b = torch.randn(3 * D) * 0.02
        c_proj_w = torch.randn(D, D) * 0.02            # [K=768, N=768]
        c_proj_b = torch.randn(D) * 0.02
        ln2_w = torch.randn(D) * 0.02 + 1.0
        ln2_b = torch.randn(D) * 0.02
        mlp_fc_w = torch.randn(D, M) * 0.02            # [K=768, N=3072]
        mlp_fc_b = torch.randn(M) * 0.02
        mlp_proj_w = torch.randn(M, D) * 0.02          # [K=3072, N=768]
        mlp_proj_b = torch.randn(D) * 0.02

        # --- Attention path ---
        ln1_out = F.layer_norm(hidden_in, (D,), ln1_w, ln1_b, eps)
        c_attn_out = ln1_out @ c_attn_w + c_attn_b                # [S, 2304]
        q_raw, k_raw, v_raw = c_attn_out.split(D, dim=1)          # each [S, 768]
        q = q_raw.view(S, H, hd).permute(1, 0, 2).contiguous()    # [12, S, 64]
        k = k_raw.view(S, H, hd).permute(1, 0, 2).contiguous()
        v = v_raw.view(S, H, hd).permute(1, 0, 2).contiguous()
        # Manual SDPA (not F.scaled_dot_product_attention — matches kernel exactly)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)        # [12, S, S]
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        o = weights @ v                                            # [12, S, 64]
        attn_out = o.permute(1, 0, 2).reshape(S, D)               # [S, 768]
        c_proj_out = attn_out @ c_proj_w + c_proj_b
        hidden = hidden_in + c_proj_out                            # residual 1

        # --- MLP path ---
        ln2_out = F.layer_norm(hidden, (D,), ln2_w, ln2_b, eps)
        fc_out = ln2_out @ mlp_fc_w + mlp_fc_b                    # [S, 3072]
        gelu_out = F.gelu(fc_out, approximate='tanh')              # GPT-2 NewGELU
        mlp_proj_out = gelu_out @ mlp_proj_w + mlp_proj_b         # [S, 768]
        hidden_out = hidden + mlp_proj_out                         # residual 2

        _save(TEST_VECTOR_DIR / "transformer_block" / name,
              hidden_in=hidden_in.numpy(),
              ln1_w=ln1_w.numpy(), ln1_b=ln1_b.numpy(),
              c_attn_w=c_attn_w.numpy(), c_attn_b=c_attn_b.numpy(),
              c_proj_w=c_proj_w.numpy(), c_proj_b=c_proj_b.numpy(),
              ln2_w=ln2_w.numpy(), ln2_b=ln2_b.numpy(),
              mlp_fc_w=mlp_fc_w.numpy(), mlp_fc_b=mlp_fc_b.numpy(),
              mlp_proj_w=mlp_proj_w.numpy(), mlp_proj_b=mlp_proj_b.numpy(),
              hidden_out=hidden_out.numpy())
        print(f"wrote transformer_block/{name} (S={S})")


def gen_gpt2_forward() -> None:
    """Minimal full forward pass reference: S=1 only (~170 KB .npy).

    Chains embedding → 12 blocks → ln_f → lm_head. Uses scaled synthetic
    weights (same convention as gen_transformer_block) to keep activations
    bounded.

    This is for component-level debugging only. The authoritative validation
    is scripts/golden_test.py (GPU logits vs HuggingFace with real weights).
    """
    import math
    import torch
    import torch.nn.functional as F
    for name, S, seed in FORWARD_CASES:
        torch.manual_seed(seed)
        D = 768; H = 12; hd = 64; M = 3072; V = 50257
        eps = 1e-5
        w_scale = 0.02

        token_ids = torch.zeros(S, dtype=torch.long)
        wte = torch.randn(V, D) * w_scale
        wpe = torch.randn(1024, D) * w_scale
        wte_T = torch.randn(D, V) * w_scale
        ln_f_w = torch.randn(D) * w_scale + 1.0
        ln_f_b = torch.randn(D) * w_scale

        # 1. Embedding
        hidden = F.embedding(token_ids, wte) + wpe[:S]

        # 2. 12 blocks with per-block distinct weights
        all_weights = {}
        for i in range(12):
            p = f"blk{i}"
            ln1_w     = torch.randn(D) * w_scale + 1.0
            ln1_b     = torch.randn(D) * w_scale
            c_attn_w  = torch.randn(D, 3*D) * w_scale
            c_attn_b  = torch.randn(3*D) * w_scale
            c_proj_w  = torch.randn(D, D) * w_scale
            c_proj_b  = torch.randn(D) * w_scale
            ln2_w     = torch.randn(D) * w_scale + 1.0
            ln2_b     = torch.randn(D) * w_scale
            mlp_fc_w  = torch.randn(D, M) * w_scale
            mlp_fc_b  = torch.randn(M) * w_scale
            mlp_proj_w = torch.randn(M, D) * w_scale
            mlp_proj_b = torch.randn(D) * w_scale

            ln1_out = F.layer_norm(hidden, (D,), ln1_w, ln1_b, eps)
            c_attn = ln1_out @ c_attn_w + c_attn_b
            qr, kr, vr = c_attn.split(D, dim=1)
            q = qr.view(S, H, hd).permute(1, 0, 2).contiguous()
            k = kr.view(S, H, hd).permute(1, 0, 2).contiguous()
            v = vr.view(S, H, hd).permute(1, 0, 2).contiguous()
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
            causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
            sc = sc.masked_fill(causal, float("-inf"))
            w = torch.softmax(sc, dim=-1)
            o = w @ v
            attn_out = o.permute(1, 0, 2).reshape(S, D)
            proj = attn_out @ c_proj_w + c_proj_b
            hidden = hidden + proj

            ln2_out = F.layer_norm(hidden, (D,), ln2_w, ln2_b, eps)
            fc = ln2_out @ mlp_fc_w + mlp_fc_b
            gelu = F.gelu(fc, approximate="tanh")
            mlp = gelu @ mlp_proj_w + mlp_proj_b
            hidden = hidden + mlp

            all_weights.update({
                f"{p}.ln1_w": ln1_w, f"{p}.ln1_b": ln1_b,
                f"{p}.c_attn_w": c_attn_w, f"{p}.c_attn_b": c_attn_b,
                f"{p}.c_proj_w": c_proj_w, f"{p}.c_proj_b": c_proj_b,
                f"{p}.ln2_w": ln2_w, f"{p}.ln2_b": ln2_b,
                f"{p}.mlp_fc_w": mlp_fc_w, f"{p}.mlp_fc_b": mlp_fc_b,
                f"{p}.mlp_proj_w": mlp_proj_w, f"{p}.mlp_proj_b": mlp_proj_b,
            })

        # 3. Final LayerNorm
        hidden = F.layer_norm(hidden, (D,), ln_f_w, ln_f_b, eps)
        # 4. LM head
        logits = hidden @ wte_T

        _save(TEST_VECTOR_DIR / "forward" / name,
              token_ids=token_ids.numpy().astype(np.int32),
              wte=wte.numpy(), wpe=wpe.numpy(), wte_T=wte_T.numpy(),
              ln_f_w=ln_f_w.numpy(), ln_f_b=ln_f_b.numpy(),
              logits=logits.numpy(), **{k: v.numpy() for k, v in all_weights.items()})
        print(f"wrote forward/{name} (S={S})")


# Registry: kernel name -> generator thunk taking parsed args.
GENERATORS = {
    "vector_add": lambda a: gen_vector_add(a.n, a.seed),
    "layernorm": lambda a: gen_layernorm(),
    "softmax": lambda a: gen_softmax(),
    "gelu": lambda a: gen_gelu(),
    "attention": lambda a: gen_attention(),
    "embedding": lambda a: gen_embedding(),
    "bias_add": lambda a: gen_bias_add(),
    "qkv_split": lambda a: gen_qkv_split(),
    "transformer_block": lambda a: gen_transformer_block(),
    "forward": lambda a: gen_gpt2_forward(),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate kernel references.")
    parser.add_argument("--kernel", required=True, choices=sorted(GENERATORS) + ["all"])
    parser.add_argument("--n", type=int, default=1 << 16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.kernel == "all":
        for k in GENERATORS:
            GENERATORS[k](args)
    else:
        GENERATORS[args.kernel](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
