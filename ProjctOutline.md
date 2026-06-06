# Project Outline — Mini GPU Inference Engine (CUDA from Scratch)

## TL;DR
Build a from-scratch CUDA inference engine that can run a small Transformer-style
language model (e.g. GPT-2 124M or TinyLlama-style architecture). You write every
kernel yourself — matmul, softmax, layernorm, attention — then benchmark against
cuBLAS / PyTorch and write up the optimization journey.

This is the most CV-relevant CUDA project a junior CS student can do right now:
it touches every core GPU concept (memory hierarchy, coalescing, tiling, warps,
tensor cores) and lands on a topic — LLM inference — that recruiters care about.

---

## Why this project (vs. the usual options)
| Option | Pros | Cons |
|---|---|---|
| **Mini inference engine (this)** | High industry relevance, deep CUDA coverage, clear "good vs. great" benchmarks | Math-heavier; needs patience for debugging numerics |
| Real-time N-body / fluid sim | Visually impressive, classic GPGPU showcase | Less directly transferable to ML/infra jobs |
| Image filters / convolution suite | Easy ramp-up | Caps out shallow — you'll outgrow it fast |
| Ray tracer | Fun, visual | Spends more time on graphics than CUDA optimization |

If you find the math frustrating, fall back to the **N-body sim with OpenGL interop**
as a Plan B — it covers ~60% of the same CUDA concepts.

---

## Hardware fit check (RTX 4070 Laptop, 8GB VRAM)
- **Architecture:** Ada Lovelace (compute capability 8.9) — has 3rd-gen RT cores and
  4th-gen tensor cores. You can experiment with FP16/BF16 and even FP8.
- **VRAM (8GB):** Plenty for GPT-2 124M (≈500MB FP16 weights) or TinyLlama-1.1B in
  INT8 (~1.2GB). Stay away from 7B+ models unquantized.
- **Bandwidth:** ~256 GB/s — perfect for learning to identify memory-bound vs.
  compute-bound kernels (you'll see both).
- **Caveat:** Laptop 4070s throttle under sustained load. Lock the clock with
  `nvidia-smi -lgc` when benchmarking so your numbers are reproducible.

---

## Prerequisites
- **Required:** C++ basics (pointers, templates at a surface level), comfort with
  command-line builds.
- **Helpful:** Linear algebra (matmul, dot product), basic neural net concepts
  (what a layer is, what softmax does). You do *not* need to know how to train
  models — you only need to run inference.
- **Toolchain to install on Day 1:**
  - CUDA Toolkit 12.x
  - Nsight Compute + Nsight Systems (profilers — non-negotiable)
  - CMake, a C++17 compiler
  - Python + PyTorch (only for generating reference outputs to diff against)

---

## Phased Roadmap (≈8–10 weeks at 6–8 hrs/week)

### Phase 0 — Setup & "Hello, GPU" (3–4 days)
**Goal:** Prove your toolchain works and you understand the launch model.
- Install CUDA toolkit, verify with `nvidia-smi` and `deviceQuery` sample.
- Write a vector add kernel. Time it. Compare host vs. device.
- Understand grid/block/thread hierarchy by deliberately picking bad block sizes
  and watching the perf cliff.
- **Deliverable:** A repo with a working `cmake` build and one benchmarked kernel.

### Phase 1 — GPGPU Primitives (1.5 weeks)
**Goal:** Build the building blocks every later kernel will reuse.
- **Reduction** (sum/max over an array): naive → shared memory → warp shuffle.
  Aim to beat 80% of cuBLAS `cublasIsamax` bandwidth.
- **Scan / prefix-sum** (Blelloch or Hillis-Steele).
- **Transpose** with shared memory + padding to avoid bank conflicts. Profile
  with Nsight Compute and *see* the conflicts disappear.
- **Deliverable:** Each primitive benchmarked at multiple sizes, written up as a
  short README with a graph showing each optimization step's speedup.

### Phase 2 — Tiled GEMM (2 weeks — this is the hard one)
**Goal:** Write a matrix multiply that gets within 50% of cuBLAS on FP32, then 30%.
- Naive global-memory version → tiled shared-memory version → register tiling →
  double-buffering / async copies (`cp.async` on Ampere+).
- Once FP32 works, swap to FP16 and use **Tensor Cores via WMMA** (`mma.sync`).
  This is the single most resume-worthy kernel you'll write.
- **Deliverable:** Plot of GFLOPS vs. matrix size, your kernel vs. cuBLAS.
  Write up *why* each optimization helped — that explanation is the actual asset.

### Phase 3 — Transformer Building Blocks (1.5 weeks)
**Goal:** Implement the layers a Transformer needs, reusing your primitives.
- **LayerNorm / RMSNorm** (reduction + elementwise).
- **Softmax** with the numerically-stable max-subtraction trick. Fused single-pass
  online softmax is the stretch version (this is what FlashAttention uses).
- **GELU / SiLU activations** (trivial — useful warm-up).
- **Rotary positional embeddings** (RoPE) if you target a Llama-style model.
- Validate every kernel by diffing outputs against a PyTorch reference to
  `<1e-4` absolute error (FP32) or `<1e-2` (FP16). Don't skip this — silent
  numerical bugs in attention will haunt you for days.

### Phase 4 — Attention (1.5 weeks)
**Goal:** Implement scaled dot-product attention end-to-end.
- Version 1: three separate kernels (Q·Kᵀ → softmax → ·V). Works, but allocates
  the full N×N attention matrix.
- Version 2 (stretch): **fused FlashAttention-style kernel** — tile across the
  sequence dimension, compute softmax online, never materialize the full matrix.
  Even a partial implementation is genuinely impressive.

### Phase 5 — End-to-End Model (1.5 weeks)
**Goal:** Load a real pretrained model and generate text.
- Pick a model — GPT-2 small (124M) is the easiest entry; TinyLlama-1.1B is more
  modern but trickier (RoPE, SwiGLU, grouped-query attention).
- Write a weight loader (parse the HuggingFace safetensors file).
- Wire up the forward pass using your kernels.
- Implement greedy decoding, then temperature sampling.
- **Deliverable:** A CLI that takes a prompt and streams tokens. The moment your
  GPU spits out coherent English from kernels you wrote yourself is the payoff.

### Phase 6 — Optimization & Writeup (1 week)
- Profile with Nsight Compute. Find the bottleneck. Fix it. Repeat.
- Benchmark tokens/sec vs. `llama.cpp` and HuggingFace `transformers`.
- Write a blog-post-style README documenting the optimization journey, with
  before/after Nsight screenshots. **This writeup is what gets you interviews** —
  more than the code itself.

---

## Stretch Goals (pick 1–2 if you have time)
- **INT8 / INT4 quantization** — fit a bigger model in 8GB.
- **KV-cache** so generation isn't O(n²) per token. (Almost mandatory if you
  want usable inference speed — promote this from "stretch" if you have time.)
- **Multi-stream execution** to overlap kernel launches.
- **CUDA Graphs** to amortize launch overhead in the decode loop.
- **Speculative decoding** with a draft model.

---

## What Success Looks Like
By the end you will have, in a public repo:
1. A working LLM inference engine written entirely in CUDA.
2. Benchmarks comparing your kernels to cuBLAS and your end-to-end throughput
   to a reference implementation.
3. A writeup explaining *every* optimization decision with profiler evidence.

The single most valuable artifact is **the writeup**. A junior who can explain
*why* their tiled GEMM hits 70% of cuBLAS is more hireable than one who can write
a faster kernel but can't articulate the reasoning.

---

## Tools & Learning Resources
- **Books / refs**
  - *Programming Massively Parallel Processors* (Kirk & Hwu) — the canonical text
  - NVIDIA CUDA C++ Programming Guide (skim, then reference)
  - CUTLASS source code — read, don't copy
- **Walkthroughs to learn from (don't just clone)**
  - Simon Boehm's "How to Optimize a CUDA Matmul Kernel" blog post
  - Andrej Karpathy's `llm.c` — exactly the project shape you're building toward
  - The original FlashAttention paper (Dao et al.)
- **Profilers**
  - Nsight Compute (per-kernel)
  - Nsight Systems (end-to-end timeline)

---

## Common Pitfalls (forewarned is forearmed)
- **Skipping the profiler.** Without Nsight, you're guessing. Learn it in Phase 1.
- **Not diffing against PyTorch.** Attention is full of off-by-one errors that
  produce plausible-looking but wrong output. Build the diffing harness early.
- **Optimizing before measuring.** Get correctness first, then profile, *then*
  optimize the actual bottleneck — not the one you assume exists.
- **Laptop thermals.** Run benchmarks at locked clocks. Don't chase a 3% speedup
  that's actually a thermal artifact.
- **Scope creep.** GPT-2 124M end-to-end is *plenty*. Don't pivot to a 7B model
  mid-project.

---

## First Week — Concrete Starter Tasks
1. Install CUDA Toolkit + Nsight tools. Run `deviceQuery`. Screenshot it.
2. `git init` a repo. Set up CMake with a single CUDA target.
3. Write `vec_add.cu` — vector addition, host vs. device timing.
4. Read chapters 1–3 of *PMPP* alongside writing the code.
5. Write a 200-word README stating the project goal — this is your commitment device.

Start there. Don't plan Phase 5 in detail until you've finished Phase 1.