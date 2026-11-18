# GPT-2 Small in hand-written CUDA

FP32 inference for GPT-2 Small (124M), batch size 1. C++/CUDA runs the full
forward pass, KV-cache decoding, and token selection. Python handles weight
export and turns text into token IDs and back. cuBLAS is used in benchmarking
and tests.

## Generate text

```bash
python scripts/generate.py --max-new-tokens 24 "In a quiet corner of the library, she found"
```

Actual output from the CUDA engine

```text
In a quiet corner of the library, she found a book.

"I'm sorry, but I don't know what you're talking about," she said.
```

## Results

| Check | Result |
| --- | --- |
| FP32 logits vs Hugging Face on CPU | Maximum absolute error **2.594e-4** (limit **5e-3**) and **53/53** matching argmax positions across four prompts |
| Greedy decoding vs Hugging Face | **100/100** matching newly generated tokens across four prompts, excluding the prompt tokens |
| 4096 × 4096 × 4096 FP32 GEMM | **6.44 TFLOP/s** for the register-tiled kernel vs **10.97 TFLOP/s** for cuBLAS SGEMM (**58.7%**) |

Measured on an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, sm_89), CUDA 12.4.
GEMM uses strict FP32 for both paths and reports the mean of 50 per-launch
CUDA-event timings after 10 warmups, with matrices already on the GPU.
See the [release report](reports/final/fp32-release.md) for the benchmark
command and full validation results.

## Reproduce

Tested on WSL2 Ubuntu with an RTX 4070 Laptop GPU, CUDA Toolkit 12.4, GCC 13,
CMake 4.2 and Python 3.14. You need a Linux or WSL2 NVIDIA CUDA environment
with a compatible host compiler, CMake 3.24+, Ninja, and Python with `venv`.
Set `CMAKE_CUDA_ARCHITECTURES` for your GPU (`89` for compute capability 8.9).
The included toolchain file expects GCC 13 in `/usr/bin`. See
[toolchain setup](docs/setup.md).

```bash
git clone https://github.com/Joey0122/Mini-GPU-Inference-Engine.git
cd Mini-GPU-Inference-Engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/requirements.txt
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4
python scripts/export_gpt2.py
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/wsl2-gcc13.cmake \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 8
```

Run the [generation command above](#generate-text) from the repository root.
The export downloads the pinned public [GPT-2 checkpoint](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)
(548 MB) and writes 652 MB of FP32 weights to `data/model_weights/`. The
first generation also downloads the tokenizer. The total budget is 1,024
tokens including the prompt, and the CLI uses greedy decoding.

## Validate

```bash
python scripts/generate_reference.py --kernel all
rm -f data/test_vectors/golden/*.bin
ctest --test-dir build --output-on-failure
python scripts/golden_test.py
```

Allow 5 GB of free disk space for the full setup and validation, including
about 1.4 GiB for generated test vectors. The golden check expects the
CUDA output files written by CTest. Run from a fresh checkout for release
validation. Every golden case is required, and a missing output or failed
comparison produces a nonzero exit code.

## Further work

[Project roadmap](docs/roadmap.md) tracks FP16 Tensor Core GEMM, Prefix Scan,
standalone Transpose, and end-to-end tokens/s comparisons.
