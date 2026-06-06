#!/usr/bin/env python3
"""scripts/generate_reference.py — PyTorch reference generator (scaffold).

IMPLEMENTATION_RULES Rule 8 / AGENT.md require that every kernel be validated
against a PyTorch reference. The workflow is: this script computes a reference
output on CPU/GPU with PyTorch and writes it (plus inputs) to data/test_vectors/
as .npy files; the C++ tests load those tensors and diff their kernel output
against them within the spec tolerance (FP32 < 1e-4, FP16 < 1e-2).

Phase 0 only needs the vector_add reference to prove the harness end-to-end.
Each later phase registers its kernel's reference generator here (softmax,
layernorm, gemm, attention, ...).

Usage:
    python scripts/generate_reference.py --kernel vector_add --n 65536
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

# torch is required only when actually generating references; import lazily so
# `--help` and the vector_add numpy path work before torch is installed.
TEST_VECTOR_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "test_vectors"


def gen_vector_add(n: int, seed: int = 0) -> None:
    """Generate inputs + reference output for vector_add.

    vector_add needs no PyTorch (it is a plain elementwise sum), so it uses
    numpy. It exists to prove the save/load reference plumbing the GPU tests
    depend on.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = a + b

    TEST_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TEST_VECTOR_DIR / "vector_add_a.npy", a)
    np.save(TEST_VECTOR_DIR / "vector_add_b.npy", b)
    np.save(TEST_VECTOR_DIR / "vector_add_ref.npy", c)
    print(f"wrote vector_add reference (n={n}) to {TEST_VECTOR_DIR}")


# Registry: kernel name -> generator. Later phases append entries here.
GENERATORS = {
    "vector_add": lambda args: gen_vector_add(args.n, args.seed),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate kernel references.")
    parser.add_argument("--kernel", required=True, choices=sorted(GENERATORS))
    parser.add_argument("--n", type=int, default=1 << 16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    GENERATORS[args.kernel](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
