# Benchmarking & Profiling Methodology

This is the project-wide methodology that keeps performance numbers honest and
reproducible. Every phase reuses it
unchanged.

## Timing: CUDA events, not wall clock

Kernel launches are asynchronous, so a host timer measures launch overhead, not
kernel time. We time with **CUDA events** recorded on the stream
([`include/common/timer.hpp`](../include/common/timer.hpp)):

1. Warm up (default 10 iterations, untimed) to reach steady-state clocks and
   prime caches.
2. Time each of N iterations (default 100) individually with an event pair.
3. Report **mean ± stddev**.

Google Benchmark drives the outer repetition loop and CLI; the CUDA-event
harness supplies the GPU-accurate per-iteration time via `SetIterationTime` +
`UseManualTime()`.

## Required metrics

Every benchmark emits: **kernel name, input shape, execution time, bandwidth,
GFLOPS, occupancy**. The first five come from the runtime harness
([`benchmarks/common/bench_harness.hpp`](../benchmarks/common/bench_harness.hpp)).
**Occupancy is reported as `n/a` by the runtime harness** and filled in from
Nsight Compute (see below) — we do not fabricate it.

- **Bandwidth (GB/s)** = bytes moved / time. Compare against the card's
  theoretical peak (`DeviceInfo::TheoreticalBandwidthGBs()`, ~256 GB/s on the
  4070 Laptop) to classify memory-bound kernels.
- **GFLOPS** = floating-point ops / time. Use for compute-bound kernels (GEMM).

## CSV schema

`bench_harness` appends rows with this header:

```
kernel,input_shape,time_ms,time_stddev_ms,bandwidth_gbs,gflops,occupancy,gpu,compute_capability
```

Plot with [`scripts/plot_benchmarks.py`](../scripts/plot_benchmarks.py). Curated
result CSVs + plots belong under `reports/phaseN/`.

## Reproducibility (Rule 6)

Record for every benchmark run: GPU model, CUDA version, driver version, clock
settings, and the input/matrix/batch sizes. GPU + compute capability are
stamped into each CSV row automatically; record the rest in the phase report.

**Lock clocks before measuring.** The RTX 4070 Laptop throttles under sustained
load, so unlocked numbers are not reproducible:

```bash
nvidia-smi -lgc <min_mhz>,<max_mhz>   # lock graphics clock
# ... run benchmarks ...
nvidia-smi -rgc                        # reset when done
```

## Profiling workflow (Nsight)

Profile each major kernel to measure the effect of an optimization.

- **Nsight Compute** (`ncu`) — per-kernel: occupancy, warp execution
  efficiency, achieved bandwidth, memory throughput, SM utilization, L2 hit
  rate, register/shared-mem usage.
- **Nsight Systems** (`nsys`) — end-to-end timeline: launch overhead, stream
  overlap, host/device gaps.

Store `.ncu-rep` / `.nsys-rep` exports and the bottleneck summary under
`reports/phaseN/`.
