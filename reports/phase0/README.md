# Phase 0 Report — Environment Setup & Scaffolding

> Per docs/architecture.md, every phase produces a report with Summary / Results /
> Problems / Fixes / Lessons. This report covers Phase 0 (scaffolding). It will
> be finalized once the toolchain is installed and the build/test/benchmark
> have actually run on the target GPU.

## Summary — What was built

The full repository skeleton from docs/architecture.md, a CMake build targeting
Windows 11 / CUDA 12.x / VS 2022 / sm_89, the GoogleTest and Google Benchmark
frameworks, a CUDA-event benchmark harness emitting the required CSV columns,
Python validation/plotting scripts, GitHub Actions CI (build + host-only
tests), and lint/format configs. The only kernel implemented is `vector_add`,
the Phase 0 toolchain smoke test.

No production kernels (reduction, GEMM, softmax, etc.) and no GPT-2 code were
implemented — those belong to Phase 1+.

## Results — Benchmarks

_Pending toolchain installation._ Once CUDA + MSVC are installed:

- `mgie_app.exe` prints device info and runs the `vector_add` smoke test.
- `bench_vector_add.exe` writes `vector_add_results.csv` with achieved
  bandwidth vs. theoretical peak (~256 GB/s) at n = 2^16, 2^20, 2^24.

Expected qualitative result: `vector_add` is memory-bound (3 floats moved per
1 add), so achieved bandwidth should dominate the story and GFLOPS should be
tiny — the reference example for "memory-bound" in Phase 0 docs.

## Problems — Observed during scaffolding

1. **No CUDA Toolkit / no MSVC on the dev machine yet.** `nvcc` and `cl.exe`
   are both absent; only MinGW g++ is present, which `nvcc` cannot use as a
   host compiler on Windows. Phase 0 code can be authored but not yet compiled.
2. **No PyTorch installed.** Blocks the Rule 8 reference-generation step.
3. **CI cannot run GPU tests.** GitHub-hosted runners have no NVIDIA GPU.
4. **Governing-doc inconsistencies** (RMSNorm/RoPE listed despite GPT-2 scope;
   diagram vs. file-tree mismatch) — see the project review.

## Fixes — Decisions made

1. Documented the exact install path in [`docs/setup.md`](../../docs/setup.md);
   provided `scripts/device_query.py` as a no-build driver check.
2. Pinned `scripts/requirements.txt`; PyTorch installs into a venv, CPU build
   sufficient.
3. Split CI honestly: `build` (compiles CUDA on a CUDA+MSVC runner) +
   `host-tests` (`ctest -L cpu`, no GPU). A `gpu-tests` job is stubbed and
   disabled until a self-hosted GPU runner exists. We do not pretend CI
   validates kernels on hardware it lacks.
4. Proposed (not yet applied) governing-doc diffs to align with the GPT-2
   Small goal; awaiting approval.

## Lessons Learned

- On Windows, the `nvcc`↔MSVC dependency is the single biggest setup hazard;
  surfacing it in Phase 0 avoids a Phase 1 stall.
- GPU-less CI is a real constraint, not a footnote: the test suite was
  partitioned (`cpu`/`gpu` labels) from day one so compilation is always
  verifiable in CI even when execution is not.
- The benchmark harness must own bandwidth/GFLOPS math; Google Benchmark alone
  cannot produce GPU metrics. Establishing this in Phase 0 keeps every later
  benchmark consistent.

## Definition of Done — Phase 0 checklist

- [x] Directory tree matches docs/architecture.md
- [x] CMake configuration authored (CUDA target, tests, benchmarks)
- [x] GoogleTest integrated; unit + reference tests authored
- [x] Google Benchmark integrated; CUDA-event harness + vector_add benchmark
- [x] Documentation scaffolding (README, docs/, this report)
- [x] GitHub Actions workflow (build + host tests)
- [x] clang-format + clang-tidy configs
- [x] git repository initialized
- [ ] **CUDA compilation verified** (blocked: install CUDA + MSVC)
- [ ] **vector_add unit test passes on GPU** (blocked: same)
- [ ] **Benchmark CSV produced on target GPU** (blocked: same)
- [ ] Initial commit(s) created

The authoring tasks are complete; the three validation items are blocked only
on toolchain installation, which is a local admin action.
