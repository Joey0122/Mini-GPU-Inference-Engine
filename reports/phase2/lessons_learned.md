# Phase 2 — Lessons Learned

**Phase:** 2 — GEMM
**Date:** 2026-06-06

---

## 1. Correctness Before Optimization — Enforcement Revealed a Tolerance Bug

The relative-error tolerance formula (`GemmRelTol = 4*K*6e-8`) looked correct
on paper but failed in practice: for random inputs in `[-1,1]`, many output
elements are near zero, and relative error is undefined (or explodes) there.
Two iterations were required to arrive at the correct tolerance:
`GemmAbsTol(K) = 4 * K * 1.2e-7` — absolute error scaled linearly with K,
matching the O(K·ε_float) theoretical bound for a K-length dot product.

**Lesson:** For kernels whose output is a sum/dot-product rather than an
element-wise operation, absolute error is the correct primary metric.
Relative error is only meaningful when the reference magnitude is bounded away
from zero. Future kernels (Softmax, LayerNorm) must have their tolerance
strategy reviewed in light of this.

---

## 2. The Arithmetic-Intensity Bottleneck Was the Dominant Factor

The design note predicted that the 2:1 FMA:load ratio at TM=TN=4 was the
primary bottleneck. Step 1 (tile enlargement to TM=TN=8) delivered +35% in
isolation, validating this hypothesis experimentally. The root cause was not
occupancy, register pressure, or memory bandwidth — it was insufficient
compute work per shared-memory access.

**Lesson:** For compute-bound kernels, profile the FMA:load ratio before
assuming occupancy is the bottleneck. High occupancy with low arithmetic
intensity is a common sub-optimal configuration.

---

## 3. Padding Has Measurable Impact (+8.8%)

Adding one float of padding per row of the shared arrays (`sA[128][17]`,
`sB[16][129]`) delivered an additional +8.8% after the tile enlargement.
This confirmed shared-memory bank conflicts in the cooperative load pattern.
The cost (576 bytes = 3.4% smem overhead) was negligible.

**Lesson:** Always pad shared-memory arrays whose row stride is a power of two.
The +1 padding idiom is low-risk and consistently beneficial. Apply by default
in future kernels that use 2D shared-memory with stride-1 access.

---

## 4. Loop Unrolling Was Counter-Productive at BK=16

`#pragma unroll` on the fixed-count BK=16 loop showed −2.5% rather than the
expected gain. Under `-O3`, the compiler already partially handles the loop.
Full unrolling at 128 registers/thread appears to degrade PTX instruction
scheduling for this particular FMA/load sequence.

**Lesson:** Do not assume `#pragma unroll` is always beneficial on short
fixed-count loops. Measure first. For future kernels, use unrolling only when
Nsight Compute confirms a warp stall bottleneck that unrolling would address.

---

## 5. Register Spilling Was Not a Problem at 128 Regs/Thread

The design note expressed uncertainty about whether the 128 registers/thread
configuration would cause spilling. ptxas confirmed 0 spill stores and 0 spill
loads. Ada (sm_89) has 65,536 registers/SM, which provides ample headroom even
at high register usage.

**Lesson:** On modern Ada hardware, register spilling is less of a concern than
on older architectures. Theoretical occupancy analysis (33%) should not be
treated as a hard barrier to performance — the actual throughput depends more on
arithmetic intensity than on warp count.

---

## 6. Double Buffering Provided Negligible Gain Over Register Tiling

Version D (manual ping-pong double buffer) delivered only +1.4% over Version C.
The bottleneck was the inner k-loop shared→register loads, not the
global→shared loads. Double buffering eliminates the global→shared latency but
cannot hide the inner-loop shared→register accesses.

**Lesson:** Double buffering is most effective when global memory latency is the
bottleneck. When the bottleneck is in the compute loop itself (shared-memory
reads in the inner k-loop), prefetching the next outer tile provides no benefit.
For future kernels, identify the latency stage being hidden before implementing
double buffering.

---

## 7. GemmAbsTol as a Pattern for Future Kernels

The `GemmAbsTol(K)` formula generalizes:
- For reduction: `ReductionRelTol(n) = 4 * log2(n) * eps`
- For GEMM: `GemmAbsTol(K) = 4 * K * eps_float`

Both follow the same structure: multiply the theoretical worst-case rounding
error bound by a small safety constant (4×). This pattern should guide tolerance
design for LayerNorm, Softmax, and Attention in subsequent phases.
