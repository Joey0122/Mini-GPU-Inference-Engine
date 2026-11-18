#!/usr/bin/env python3
"""scripts/plot_benchmarks.py — turn benchmark CSVs into plots (scaffold).

docs/architecture.md / the Phase deliverables call for graphs showing each
optimization step's speedup (e.g. GFLOPS vs. matrix size, kernel vs. cuBLAS).
The C++ benchmarks emit CSV rows via benchmarks/common/bench_harness.hpp; this
script reads one or more of those CSVs and renders the standard plots into
benchmarks/<kernel>/ or reports/.

Phase 0 only has vector_add bandwidth data, so the default plot is achieved
bandwidth vs. input size. Plotting deps (matplotlib/pandas) are optional and
imported lazily so this file is committable before they're installed.

Usage:
    python scripts/plot_benchmarks.py vector_add_results.csv --metric bandwidth_gbs
"""
from __future__ import annotations

import argparse
import csv
import pathlib


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV results.")
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--metric", default="bandwidth_gbs",
                        help="CSV column to plot on the Y axis")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"no such CSV: {args.csv} (run a benchmark first)")
        return 1

    rows = load_csv(args.csv)
    if not rows:
        print("CSV is empty")
        return 1

    # Lazy import so the script is importable without plotting deps installed.
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433 (intentional lazy import)
    except ImportError:
        print("matplotlib not installed; printing data instead:\n")
        for r in rows:
            print(f"  {r['input_shape']:>16}  {args.metric}={r.get(args.metric)}")
        return 0

    xs = [r["input_shape"] for r in rows]
    ys = [float(r[args.metric]) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("input shape")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{rows[0]['kernel']} — {args.metric}")
    ax.grid(True)

    out = args.out or args.csv.with_suffix(f".{args.metric}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
