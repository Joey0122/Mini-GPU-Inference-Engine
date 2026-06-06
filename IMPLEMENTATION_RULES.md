# IMPLEMENTATION_RULES.md

# Agent Execution Rules

Follow these rules throughout the project.

---

## Rule 1: Correctness Before Optimization

Never optimize code that has not passed validation.

Order:

Correctness
→ Testing
→ Profiling
→ Optimization

---

## Rule 2: One Kernel At A Time

Never implement multiple major kernels simultaneously.

Only proceed when:

* Current kernel compiles
* Current kernel passes tests
* Benchmark exists

---

## Rule 3: Benchmark Everything

Every CUDA kernel must have:

* benchmark executable
* timing results
* comparison against baseline

Required outputs:

* latency
* throughput
* achieved bandwidth
* achieved FLOPS

---

## Rule 4: Maintain Documentation

Every optimization step must include:

### Previous Version

Description

### Optimization

Description

### Result

Measured improvement

### Explanation

Why improvement occurred

---

## Rule 5: Profile Before Making Claims

Performance claims without profiler evidence are invalid.

Use:

* Nsight Compute
* Nsight Systems

for every major optimization.

---

## Rule 6: Preserve Reproducibility

Every benchmark must record:

* GPU model
* CUDA version
* Driver version
* Clock settings
* Matrix sizes
* Batch sizes

---

## Rule 7: Small Commits

Each completed task should correspond to a separate git commit.

Examples:

feat(reduction): naive reduction

feat(reduction): shared memory optimization

feat(gemm): tiled implementation

feat(attention): scaled dot-product attention

---

## Rule 8: Write Tests First

Before implementing a kernel:

1. Create CPU reference implementation
2. Create PyTorch reference implementation
3. Generate test vectors
4. Define numerical tolerance

Only then implement CUDA.
