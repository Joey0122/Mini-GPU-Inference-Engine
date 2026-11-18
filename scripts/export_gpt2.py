#!/usr/bin/env python3
"""scripts/export_gpt2.py — export HuggingFace GPT-2 Small weights to a flat
binary format readable by the C++ weight loader.

Output (in data/model_weights/):
    gpt2_weights.bin      flat FP32 tensors, all concatenated
    gpt2_manifest.txt     one line per tensor: name offset_bytes ndim d0 [d1]

Weight layout notes:
    * HuggingFace GPT-2 uses Conv1D for attention/MLP projections. Conv1D stores
      weights as [in_features, out_features], which is exactly the [K, N]
      layout our LaunchGemm(A, B, C, M, N, K) expects for B. No transpose is
      needed for any projection weight.
    * wte (token embeddings) is [vocab_size, n_embd]. The C++ loader also needs
      wte_T = [n_embd, vocab_size] for the tied LM-head GEMM
      (logits = hidden @ wte_T). wte_T is computed here on the host (a simple
      Python .T) and written as a separate tensor. This costs one extra copy of
      wte (~147 MB) as a deliberate Phase 5 simplicity tradeoff.
    * All tensors are converted to float32, contiguous C-order before writing.

Usage:
    source .venv/bin/activate
    python scripts/export_gpt2.py [--model openai-community/gpt2] [--out data/model_weights]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from tempfile import TemporaryDirectory

import numpy as np

from gpt2_checkpoint import MODEL_ID, REVISION


def export(model_name: str, out_dir: pathlib.Path) -> None:
    # Lazy import: transformers is only needed here.
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("ERROR: 'transformers' is not installed. Run:\n"
              "  pip install -r scripts/requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading '{model_name}' from HuggingFace (downloads once, then cached)...")
    kwargs = {"revision": REVISION} if model_name == MODEL_ID else {}
    model = GPT2LMHeadModel.from_pretrained(model_name, use_safetensors=True, **kwargs)
    model.eval()
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    cfg = model.config

    # Verify GPT-2 Small constants match what the C++ engine expects.
    assert cfg.n_layer == 12,     f"unexpected n_layer={cfg.n_layer}"
    assert cfg.n_head == 12,      f"unexpected n_head={cfg.n_head}"
    assert cfg.n_embd == 768,     f"unexpected n_embd={cfg.n_embd}"
    assert cfg.n_positions == 1024
    assert cfg.vocab_size == 50257

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "gpt2_weights.bin"
    manifest_path = out_dir / "gpt2_manifest.txt"

    # Build the ordered tensor list. Order is fixed; both the manifest and the
    # binary are written in this order so the C++ loader can seek by offset.
    tensors: list[tuple[str, np.ndarray]] = []

    # --- embeddings ---
    wte = np.ascontiguousarray(sd["transformer.wte.weight"], dtype=np.float32)
    wte_T = np.ascontiguousarray(wte.T, dtype=np.float32)  # host-side transpose
    tensors.append(("wte",   wte))    # [vocab_size=50257, n_embd=768]
    tensors.append(("wte_T", wte_T))  # [n_embd=768, vocab_size=50257]
    tensors.append(("wpe",   np.ascontiguousarray(
        sd["transformer.wpe.weight"], dtype=np.float32)))  # [n_ctx=1024, n_embd=768]

    # --- transformer blocks ---
    for i in range(cfg.n_layer):
        p = f"h{i}"
        hf = f"transformer.h.{i}"
        # Layer-norm 1
        tensors.append((f"{p}.ln1.weight", np.ascontiguousarray(
            sd[f"{hf}.ln_1.weight"], dtype=np.float32)))   # [768]
        tensors.append((f"{p}.ln1.bias",   np.ascontiguousarray(
            sd[f"{hf}.ln_1.bias"],   dtype=np.float32)))   # [768]
        # Combined QKV projection — Conv1D: [n_embd=768, 3*n_embd=2304], no transpose needed.
        tensors.append((f"{p}.c_attn.weight", np.ascontiguousarray(
            sd[f"{hf}.attn.c_attn.weight"], dtype=np.float32)))  # [768, 2304]
        tensors.append((f"{p}.c_attn.bias",   np.ascontiguousarray(
            sd[f"{hf}.attn.c_attn.bias"],   dtype=np.float32)))  # [2304]
        # Attention output projection — Conv1D: [n_embd=768, n_embd=768]
        tensors.append((f"{p}.c_proj.weight", np.ascontiguousarray(
            sd[f"{hf}.attn.c_proj.weight"], dtype=np.float32)))  # [768, 768]
        tensors.append((f"{p}.c_proj.bias",   np.ascontiguousarray(
            sd[f"{hf}.attn.c_proj.bias"],   dtype=np.float32)))  # [768]
        # Layer-norm 2
        tensors.append((f"{p}.ln2.weight", np.ascontiguousarray(
            sd[f"{hf}.ln_2.weight"], dtype=np.float32)))   # [768]
        tensors.append((f"{p}.ln2.bias",   np.ascontiguousarray(
            sd[f"{hf}.ln_2.bias"],   dtype=np.float32)))   # [768]
        # MLP first linear — Conv1D: [n_embd=768, mlp_inner=3072]
        tensors.append((f"{p}.mlp_fc.weight", np.ascontiguousarray(
            sd[f"{hf}.mlp.c_fc.weight"], dtype=np.float32)))    # [768, 3072]
        tensors.append((f"{p}.mlp_fc.bias",   np.ascontiguousarray(
            sd[f"{hf}.mlp.c_fc.bias"],   dtype=np.float32)))    # [3072]
        # MLP second linear — Conv1D: [mlp_inner=3072, n_embd=768]
        tensors.append((f"{p}.mlp_proj.weight", np.ascontiguousarray(
            sd[f"{hf}.mlp.c_proj.weight"], dtype=np.float32)))  # [3072, 768]
        tensors.append((f"{p}.mlp_proj.bias",   np.ascontiguousarray(
            sd[f"{hf}.mlp.c_proj.bias"],   dtype=np.float32)))  # [768]

    # --- final layer-norm ---
    tensors.append(("ln_f.weight", np.ascontiguousarray(
        sd["transformer.ln_f.weight"], dtype=np.float32)))  # [768]
    tensors.append(("ln_f.bias",   np.ascontiguousarray(
        sd["transformer.ln_f.bias"],   dtype=np.float32)))  # [768]

    # Self-consistency check: wte_T must be the exact transpose of wte.
    _verify_transpose(wte, wte_T, "wte_T")

    # Verify tied LM head: lm_head.weight should be identical to wte.
    lm_head = sd.get("lm_head.weight")
    if lm_head is not None:
        assert np.array_equal(lm_head, wte), \
            "lm_head.weight is NOT tied to wte — unexpected for GPT-2 Small"
        print("Verified: lm_head.weight is tied to wte (identical).")

    # --- write binary and manifest ---
    total_bytes = 0
    manifest_lines = [
        "# GPT-2 Small weight manifest",
        "# fmt: name offset_bytes ndim d0 [d1]",
        "#",
    ]

    print(f"Writing {len(tensors)} tensors to {bin_path} ...")
    with TemporaryDirectory(dir=out_dir) as temp_dir:
        temp_bin = pathlib.Path(temp_dir) / bin_path.name
        temp_manifest = pathlib.Path(temp_dir) / manifest_path.name
        with temp_bin.open("wb") as f:
            for name, arr in tensors:
                assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
                shape_str = " ".join(str(d) for d in arr.shape)
                manifest_lines.append(f"{name} {total_bytes} {arr.ndim} {shape_str}")
                f.write(memoryview(arr).cast("B"))
                total_bytes += arr.nbytes
        temp_manifest.write_text("\n".join(manifest_lines) + "\n")
        temp_bin.replace(bin_path)
        temp_manifest.replace(manifest_path)
    print(f"Wrote manifest to {manifest_path}")
    print(f"Total binary size: {total_bytes / 1e6:.1f} MB")

    print("Export complete.")


def _verify_transpose(a: np.ndarray, a_T: np.ndarray, name: str) -> None:
    """Spot-check that a_T == a.T using a few non-trivial index pairs."""
    assert a_T.shape == a.T.shape, f"{name} shape mismatch: {a_T.shape} vs {a.T.shape}"
    r, c = a.shape
    # (i, j) must be valid indices for a: 0 <= i < r, 0 <= j < c.
    # Then a_T[j, i] should equal a[i, j].
    checks = [(0, 0), (0, 5), (0, c - 1), (r // 2, c // 3)]
    for (i, j) in checks:
        # a[i, j] should equal a_T[j, i]
        val_orig = float(a[i, j])
        val_T    = float(a_T[j, i])
        assert val_orig == val_T, (
            f"{name}: a[{i},{j}]={val_orig} != a_T[{j},{i}]={val_T}"
        )
    print(f"Verified: {name} is the correct transpose of source "
          f"({len(checks)} spot-checks passed).")


def export_synthetic(out_dir: pathlib.Path, seed: int = 42) -> None:
    """Generate random GPT-2 Small–shaped weights for loader testing.

    The values are random but the shapes, names, and binary format are
    identical to the real export. This lets the C++ loader be tested fully
    (shape validation, round-trip, wte_T consistency) without network access.
    NOT suitable for actual inference.
    """
    rng = np.random.default_rng(seed)
    n_layer, n_embd, n_head, n_ctx, vocab, mlp = 12, 768, 12, 1024, 50257, 3072

    def rand(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    wte = rand(vocab, n_embd)
    wte_T = np.ascontiguousarray(wte.T, dtype=np.float32)

    tensors: list[tuple[str, np.ndarray]] = [
        ("wte",   wte),
        ("wte_T", wte_T),
        ("wpe",   rand(n_ctx, n_embd)),
    ]
    for i in range(n_layer):
        p = f"h{i}"
        tensors += [
            (f"{p}.ln1.weight",     rand(n_embd)),
            (f"{p}.ln1.bias",       rand(n_embd)),
            (f"{p}.c_attn.weight",  rand(n_embd, 3 * n_embd)),
            (f"{p}.c_attn.bias",    rand(3 * n_embd)),
            (f"{p}.c_proj.weight",  rand(n_embd, n_embd)),
            (f"{p}.c_proj.bias",    rand(n_embd)),
            (f"{p}.ln2.weight",     rand(n_embd)),
            (f"{p}.ln2.bias",       rand(n_embd)),
            (f"{p}.mlp_fc.weight",  rand(n_embd, mlp)),
            (f"{p}.mlp_fc.bias",    rand(mlp)),
            (f"{p}.mlp_proj.weight", rand(mlp, n_embd)),
            (f"{p}.mlp_proj.bias",  rand(n_embd)),
        ]
    tensors += [("ln_f.weight", rand(n_embd)), ("ln_f.bias", rand(n_embd))]

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "gpt2_weights.bin"
    manifest_path = out_dir / "gpt2_manifest.txt"

    total_bytes = 0
    lines = [
        "# GPT-2 Small weight manifest (SYNTHETIC — random values, not real GPT-2)",
        "# fmt: name offset_bytes ndim d0 [d1]", "#",
    ]
    print(f"Writing synthetic weights to {bin_path} ...")
    with open(bin_path, "wb") as f:
        for name, arr in tensors:
            assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
            lines.append(f"{name} {total_bytes} {arr.ndim} " +
                         " ".join(str(d) for d in arr.shape))
            f.write(arr.tobytes())
            total_bytes += arr.nbytes
    manifest_path.write_text("\n".join(lines) + "\n")
    _verify_transpose(wte, wte_T, "wte_T")
    print(f"Synthetic export complete. {len(tensors)} tensors, "
          f"{total_bytes / 1e6:.1f} MB.")
    print("NOTE: these are random values — NOT real GPT-2 weights. "
          "For inference, run without --synthetic on a machine with internet access.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GPT-2 weights to flat binary.")
    parser.add_argument("--model", default=MODEL_ID,
                        help=f"Hugging Face model ID or local directory (default: {MODEL_ID})")
    parser.add_argument("--out", default="data/model_weights",
                        help="output directory (default: data/model_weights)")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate random weights with correct shapes (no internet "
                             "needed; for loader testing only, NOT real GPT-2 weights)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for --synthetic mode (default: 42)")
    args = parser.parse_args()
    if args.synthetic:
        export_synthetic(pathlib.Path(args.out), args.seed)
    else:
        export(args.model, pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
