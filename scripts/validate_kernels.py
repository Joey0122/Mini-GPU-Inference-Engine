#!/usr/bin/env python3
"""scripts/validate_kernels.py — unified Phase 3 validation driver.

Orchestrates the file-based PyTorch validation pipeline end-to-end:
    1. generate references  (scripts/generate_reference.py --kernel <name>)
    2. run the CUDA kernel validation runner (ctest -L validation), which loads
       the .npy vectors, executes the actual committed kernels, diffs against
       the PyTorch references, and writes reports/phase3/validation_report.json
    3. parse + pretty-print the report, gating on FP32 max_abs_error < 1e-4

Exit code is non-zero if any case exceeds the tolerance, so CI can gate on it.

Usage:
    python scripts/validate_kernels.py --kernel all
    python scripts/validate_kernels.py --kernel layernorm
    python scripts/validate_kernels.py --kernel all --skip-generate   # reuse vectors
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "phase3" / "validation_report.json"
GATE = 1e-4
# The validation runner emits one combined report for all kernels. Listed by
# phase so --kernel all covers everything implemented so far.
PHASE3 = ["layernorm", "softmax", "gelu"]
PHASE4 = ["attention"]
PHASE5 = ["embedding", "bias_add", "qkv_split", "transformer_block", "forward"]
ALL_KERNELS = PHASE3 + PHASE4 + PHASE5


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def generate(kernels: list[str]) -> int:
    for k in kernels:
        rc = run([sys.executable, "scripts/generate_reference.py", "--kernel", k])
        if rc != 0:
            print(f"reference generation failed for {k}", file=sys.stderr)
            return rc
    return 0


def run_runner() -> int:
    # The runner is a labeled ctest; it writes validation_report.json. We do not
    # build here (the runner must already be built); surface a clear error if not.
    return run(["ctest", "--test-dir", "build", "-L", "validation",
                "--output-on-failure"])


def report(kernels: list[str]) -> int:
    if not REPORT.exists():
        print(f"no report at {REPORT}", file=sys.stderr)
        return 1
    data = json.loads(REPORT.read_text())
    rows = [r for r in data["results"] if r["kernel"] in kernels]
    if not rows:
        print("no matching results in report", file=sys.stderr)
        return 1

    width = max(len(r["kernel"]) for r in rows)
    worst = 0.0
    failed = 0
    print(f"\n{'kernel':<{width}}  {'version':<8} {'shape':<10} "
          f"{'max_abs_err':>12}  result")
    print("-" * (width + 46))
    for r in rows:
        worst = max(worst, r["max_abs_error"])
        if not r["passed"]:
            failed += 1
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['kernel']:<{width}}  {r['version']:<8} {r['shape']:<10} "
              f"{r['max_abs_error']:>12.3e}  {status}")
    print("-" * (width + 46))
    print(f"gate = {GATE:.0e}   worst max_abs_error = {worst:.3e}   "
          f"{len(rows)} cases, {failed} failed\n")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Validate CUDA kernels vs PyTorch.")
    p.add_argument("--kernel", default="all", choices=ALL_KERNELS + ["all"])
    p.add_argument("--skip-generate", action="store_true",
                   help="reuse existing .npy vectors")
    args = p.parse_args()
    kernels = ALL_KERNELS if args.kernel == "all" else [args.kernel]

    if not args.skip_generate:
        if generate(kernels) != 0:
            return 1
    if run_runner() != 0:
        # ctest non-zero already prints failing cases; still emit the table.
        report(kernels)
        return 1
    return report(kernels)


if __name__ == "__main__":
    raise SystemExit(main())
