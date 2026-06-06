# .claude/commands/benchmark.md

Benchmark a kernel or subsystem.

Arguments:
$ARGUMENTS

---

# Instructions

Target:

$ARGUMENTS

Collect benchmark data.

Requirements:

1. Run multiple iterations.
2. Warm up GPU first.
3. Record averages.
4. Record variance.

Report:

# Benchmark Report

## Hardware

GPU
CUDA Version
Driver

## Workload

Input Sizes

## Results

Latency

Bandwidth

GFLOPS

Tokens/sec (if applicable)

## Comparison

Baseline

Current

Improvement

## Observations

Identify whether performance is:

* compute bound
* memory bound
* launch bound

Do not propose optimizations yet.

Only report measurements.
