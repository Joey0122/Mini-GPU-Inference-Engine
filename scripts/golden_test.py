#!/usr/bin/env python3
"""Compare the CTest CUDA outputs with Hugging Face GPT-2 Small on CPU."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from gpt2_checkpoint import MODEL_ID, REVISION

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "data" / "test_vectors" / "golden"

LOGIT_PROMPTS = [
    ("singleton", [42]),
    ("short4", [0, 1, 2, 3]),
    ("medium16", [314, 159, 265, 358, 979, 323, 846, 264,
                  338, 327, 950, 288, 419, 716, 939, 937]),
    ("long32", list(range(1, 33))),
]

DECODE_CASES = [
    ("decode_p0", [0], 25),
    ("decode_p42", [42], 25),
    ("decode_p3", [314, 159, 265], 25),
    ("decode_p8", [1, 2, 3, 4, 5, 6, 7, 8], 25),
]

SAMPLE_CASES = [
    ("sample_p0_t08_s42", [0], 20, 0.8),
    ("sample_p0_t12_s7", [0], 20, 1.2),
    ("sample_p3_t08_s42", [314, 159, 265], 20, 0.8),
]


def read_output(path: Path, dtype, count: int) -> np.ndarray:
    size = path.stat().st_size
    expected = count * np.dtype(dtype).itemsize
    if size != expected:
        raise ValueError(f"{path.name}: got {size} bytes, expected {expected}")
    values = np.fromfile(path, dtype=dtype)
    if values.size != count:
        raise ValueError(f"{path.name}: read {values.size} values, expected {count}")
    return values


def check_prompt(path: Path, prompt: list[int]) -> None:
    if not np.array_equal(read_output(path, "<i4", len(prompt)), prompt):
        raise ValueError(f"{path.name}: prompt IDs differ")


def read_continuation(directory: Path, name: str, prompt: list[int],
                      steps: int, vocab_size: int) -> np.ndarray:
    check_prompt(directory / f"{name}_prompt.bin", prompt)
    tokens = read_output(directory / f"{name}_tokens.bin", "<i4", len(prompt) + steps)
    if not np.array_equal(tokens[:len(prompt)], prompt):
        raise ValueError("output prefix differs from the prompt")
    new_tokens = tokens[len(prompt):]
    if np.any((new_tokens < 0) | (new_tokens >= vocab_size)):
        raise ValueError("generated token ID outside vocabulary")
    return new_tokens


def run_logit_golden(model, directory: Path, tolerance: float) -> int:
    passed = total_matches = 0
    worst_error = 0.0
    vocab_size = model.config.vocab_size
    for name, prompt in LOGIT_PROMPTS:
        try:
            check_prompt(directory / f"{name}_ids.bin", prompt)
            gpu = read_output(directory / f"{name}_logits.bin", "<f4",
                              len(prompt) * vocab_size).reshape(len(prompt), vocab_size)
            hf = model(torch.tensor([prompt])).logits[0].float().numpy()
            if not np.isfinite(gpu).all() or not np.isfinite(hf).all():
                raise ValueError("non-finite logits")
        except (OSError, ValueError) as exc:
            print(f"[{name}] FAIL {exc}")
            continue

        error = float(np.max(np.abs(gpu - hf)))
        matches = int(np.count_nonzero(gpu.argmax(axis=-1) == hf.argmax(axis=-1)))
        worst_error = max(worst_error, error)
        total_matches += matches
        ok = error < tolerance and matches == len(prompt)
        passed += ok
        print(f"[{name}] {'PASS' if ok else 'FAIL'} max_abs_error={error:.3e} "
              f"(must be < {tolerance:.3e}), argmax={matches}/{len(prompt)}")

    print(f"Logits: worst max_abs_error={worst_error:.3e}, "
          f"argmax={total_matches}/{sum(len(prompt) for _, prompt in LOGIT_PROMPTS)}")
    return passed


def run_decode_golden(model, directory: Path) -> int:
    passed = total_matches = 0
    for name, prompt, steps in DECODE_CASES:
        try:
            gpu = read_continuation(directory, name, prompt, steps, model.config.vocab_size)
            input_ids = torch.tensor([prompt])
            hf = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                max_new_tokens=steps, do_sample=False,
                                pad_token_id=model.config.eos_token_id)
            hf = hf[0, len(prompt):].numpy()
            if len(hf) != steps:
                raise ValueError(f"Hugging Face produced {len(hf)} of {steps} new tokens")
        except (OSError, ValueError) as exc:
            print(f"[{name}] FAIL {exc}")
            continue

        matches = int(np.count_nonzero(gpu == hf))
        total_matches += matches
        ok = matches == steps
        passed += ok
        print(f"[{name}] {'PASS' if ok else 'FAIL'} new tokens={matches}/{steps}")

    print(f"Greedy decode: {total_matches}/{sum(steps for _, _, steps in DECODE_CASES)} "
          "new tokens matched (prompt excluded)")
    return passed


def run_sample_golden(model, directory: Path) -> int:
    """Check HF probability on the actual CUDA sampling trajectory."""
    passed = checked = total_failed = 0
    for name, prompt, steps, temperature in SAMPLE_CASES:
        try:
            new_tokens = read_continuation(
                directory, name, prompt, steps, model.config.vocab_size)
        except (OSError, ValueError) as exc:
            print(f"[{name}] FAIL {exc}")
            continue

        # Causal logits at each position predict its successor.
        ids = torch.tensor([prompt + new_tokens.tolist()])
        logits = model(ids).logits[0, len(prompt) - 1:-1].float()
        probs = torch.softmax(logits / temperature, dim=-1)
        selected = probs[torch.arange(steps), torch.tensor(new_tokens, dtype=torch.long)]
        failures = int((~torch.isfinite(selected) | (selected <= 1e-12)).sum())
        ok = failures == 0
        passed += ok
        checked += steps
        total_failed += failures
        print(f"[{name}] {'PASS' if ok else 'FAIL'} implausible samples={failures}/{steps}")

    print(f"Sampling: {total_failed}/{checked} checked samples had HF probability <= 1e-12 "
          "or a non-finite probability")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=5e-3,
                        help="exclusive max absolute logit error threshold (default: 5e-3)")
    parser.add_argument("--outputs-dir", type=Path, default=GOLDEN_DIR,
                        help="directory containing the binary outputs from CTest")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive and finite")

    print(f"Loading {MODEL_ID} on CPU")
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, revision=REVISION, use_safetensors=True, dtype=torch.float32).eval()
    with torch.inference_mode():
        passed = run_logit_golden(model, args.outputs_dir, args.tolerance)
        passed += run_decode_golden(model, args.outputs_dir)
        passed += run_sample_golden(model, args.outputs_dir)

    required = len(LOGIT_PROMPTS) + len(DECODE_CASES) + len(SAMPLE_CASES)
    failed = required - passed
    print(f"Golden cases: {passed} passed, {failed} failed, 0 skipped")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
