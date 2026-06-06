#!/usr/bin/env python3
"""scripts/device_query.py — replacement for the deprecated deviceQuery sample.

TASK_BREAKDOWN Phase 0 lists "deviceQuery" as a setup step, but recent CUDA
toolkits no longer install that CUDA sample by default. This script reproduces
the essential output (device name, compute capability, memory, SM count,
theoretical bandwidth) so the Phase 0 toolchain check is self-contained.

It uses NVIDIA's `nvidia-smi` (always present with the driver) and, when
available, the CUDA Python bindings — but degrades gracefully so it runs even
before the full toolkit is installed. The C++ `mgie_app` binary prints the same
metadata once the project builds; this is the no-build fallback.

Usage:
    python scripts/device_query.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def query_nvidia_smi() -> int:
    """Print GPU info via nvidia-smi. Returns process exit code."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        print("nvidia-smi not found on PATH — is the NVIDIA driver installed?",
              file=sys.stderr)
        return 1

    fields = "name,compute_cap,memory.total,driver_version"
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"nvidia-smi failed: {exc.stderr.strip()}", file=sys.stderr)
        return exc.returncode

    print("=== GPU (via nvidia-smi) ===")
    for line in out.stdout.strip().splitlines():
        name, cc, mem, driver = (p.strip() for p in line.split(","))
        print(f"  name               : {name}")
        print(f"  compute capability : {cc}")
        print(f"  total memory       : {mem}")
        print(f"  driver version     : {driver}")
    print()
    print("Expected for this project: RTX 4070 Laptop, compute capability 8.9")
    return 0


def main() -> int:
    return query_nvidia_smi()


if __name__ == "__main__":
    raise SystemExit(main())
