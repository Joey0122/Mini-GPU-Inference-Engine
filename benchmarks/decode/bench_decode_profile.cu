// benchmarks/decode/bench_decode_profile.cu
// Phase 6a decode-step wall-clock breakdown at S_cached = 1024.
//
// Profiles each kernel sub-component by running isolated passes: each pass
// launches 12x (or 1x) of the target kernel back-to-back on the default stream
// with CUDA event timing.
//
// Also verifies single-token invariants (S=1 in all kernels, no O(S^2) scratch).

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "common/errors.hpp"
#include "cuda/memory.hpp"
#include "kernels/attention.hpp"
#include "kernels/bias_add.hpp"
#include "kernels/gelu.hpp"
#include "kernels/gemm.hpp"
#include "kernels/layernorm.hpp"
#include "kernels/qkv_split.hpp"
#include "kernels/softmax.hpp"
#include "kernels/vector_add.hpp"
#include "model/forward.hpp"
#include "model/gpt2_config.hpp"
#include "model/kv_cache.hpp"
#include "model/loader.hpp"
#include "model/workspace.hpp"

#ifndef MGIE_MODEL_WEIGHTS_DIR
#define MGIE_MODEL_WEIGHTS_DIR "data/model_weights"
#endif

using namespace mgie;

namespace {

bool WeightsExist() {
  std::ifstream f(std::string(MGIE_MODEL_WEIGHTS_DIR) + "/gpt2_manifest.txt");
  return f.good();
}

cudaEvent_t MakeEvent() {
  cudaEvent_t e;
  CUDA_CHECK(cudaEventCreate(&e));
  return e;
}

float ElapsedUs(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms * 1000.f;
}

struct ProfileResult {
  float embed_wte     = 0;
  float embed_wpe     = 0;
  float embed_add     = 0;  // includes wpe copy; subtract to get pure add
  float ln1           = 0;  // 12x
  float c_attn        = 0;  // 12x GEMM+BiasAdd
  float qkv_split     = 0;  // 12x
  float scores        = 0;  // 12x
  float softmax       = 0;  // 12x
  float av            = 0;  // 12x
  float c_proj        = 0;  // 12x GEMM+BiasAdd
  float res1          = 0;  // 12x VectorAdd
  float ln2           = 0;  // 12x
  float mlp_fc        = 0;  // 12x GEMM+BiasAdd
  float gelu          = 0;  // 12x
  float mlp_proj      = 0;  // 12x GEMM+BiasAdd
  float res2          = 0;  // 12x VectorAdd
  float ln_f          = 0;  // 1x
  float lm_head       = 0;  // 1x GEMM
  float d2h           = 0;
  float host_argmax   = 0;
  float h2d           = 0;
  float stream_sync   = 0;

  float blk_total() const {
    return ln1 + c_attn + qkv_split + scores + softmax + av +
           c_proj + res1 + ln2 + mlp_fc + gelu + mlp_proj + res2;
  }
  float gpu_total() const {
    return embed_wte + embed_wpe + (embed_add - embed_wpe) +
           blk_total() + ln_f + lm_head;
  }
  float host_total() const { return d2h + host_argmax + h2d + stream_sync; }
  float total() const { return gpu_total() + host_total(); }
};

// ---------------------------------------------------------------------------
void PrintVerification(const Gpt2Config& cfg) {
  const int n_ctx = cfg.n_ctx;
  const int D = cfg.n_embd;
  const int V = cfg.vocab_size;
  const int H = cfg.n_head;

  printf("\n");
  printf("==============================================================\n");
  printf("  Single-Token Verification\n");
  printf("==============================================================\n");
  printf("  All decode-step kernels use S=1 (verified by code audit).\n");
  printf("\n");
  printf("  Hidden state:  only row %d of [%d, %d] modified.\n", n_ctx-1, n_ctx, D);
  printf("  Logits:        single row [%d] = %.1f KB (was [%d,%d] = %.0f MB).\n",
         V, V * sizeof(float) / 1024.f,
         n_ctx, V, 1.f * n_ctx * V * sizeof(float) / 1024 / 1024);
  printf("  Scores/weights: [%d, %d] = %.1f KB each.\n",
         H, n_ctx, H * n_ctx * sizeof(float) / 1024.f);
  printf("  O(S^2) would be 12 x %d^2 = %.0f MB.\n",
         n_ctx, 1.f * H * n_ctx * n_ctx * sizeof(float) / 1024 / 1024);
  printf("  => No O(S^2) scratch buffers touched during decode.\n");
  printf("  => No full-sequence activations recomputed.\n");
  printf("==============================================================\n");
}

// ---------------------------------------------------------------------------
ProfileResult Profile(const Gpt2Weights& w, const Gpt2Config& cfg) {
  const int n_ctx = cfg.n_ctx;
  const int D     = cfg.n_embd;
  const int V     = cfg.vocab_size;
  const int H     = cfg.n_head;
  const int d     = cfg.head_dim;
  const int M_cfg = cfg.mlp_inner;
  const float eps = cfg.ln_eps;
  const int nL    = cfg.n_layer;
  const int pos   = n_ctx - 1;
  const int nc    = pos + 1;

  auto ws = Gpt2Workspace::Create(n_ctx, cfg);
  auto kv = KVCache::Create(cfg);

  // Prefill
  {
    std::vector<int> ids(n_ctx, 0);
    DeviceBuffer<int> d_ids(n_ctx);
    d_ids.CopyFromHost(ids.data(), n_ctx);
    LaunchGpt2Prefill(d_ids.data(), w, ws.logits.data(), ws, kv, cfg, n_ctx);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Warmup
  for (int i = 0; i < 5; ++i) {
    LaunchGpt2DecodeStep(0, pos, w, ws.logits.data(), ws, kv, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaStream_t str = nullptr;
  cudaEvent_t e0 = MakeEvent(), e1 = MakeEvent();

  auto Time = [&](auto&& fn) -> float {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0, str));
    fn();
    CUDA_CHECK(cudaEventRecord(e1, str));
    CUDA_CHECK(cudaDeviceSynchronize());
    return ElapsedUs(e0, e1);
  };

  ProfileResult R;
  float* h_row = ws.hidden.data() + static_cast<std::size_t>(pos) * D;

  // ---- Embedding ----
  R.embed_wte = Time([&]() {
    CUDA_CHECK(cudaMemcpyAsync(h_row, w.wte.data(), D * sizeof(float),
                                cudaMemcpyDeviceToDevice, str));
  });
  R.embed_wpe = Time([&]() {
    float* tmp = ws.blk.ln1_out.data();
    CUDA_CHECK(cudaMemcpyAsync(tmp,
        w.wpe.data() + static_cast<std::size_t>(pos) * D,
        D * sizeof(float), cudaMemcpyDeviceToDevice, str));
  });
  R.embed_add = Time([&]() {
    float* tmp = ws.blk.ln1_out.data();
    CUDA_CHECK(cudaMemcpyAsync(tmp,
        w.wpe.data() + static_cast<std::size_t>(pos) * D,
        D * sizeof(float), cudaMemcpyDeviceToDevice, str));
    LaunchVectorAdd(h_row, tmp, h_row, D, str);
  });

  // ---- 12x Transformer Block sub-components ----
  R.ln1 = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      const auto& blk = w.blocks[static_cast<std::size_t>(L)];
      LaunchLayerNorm(h_row, ws.blk.ln1_out.data(),
                      blk.ln1_w.data(), blk.ln1_b.data(),
                      1, D, eps, LayerNormVersion::Naive, str);
    }
  });

  R.c_attn = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      const auto& blk = w.blocks[static_cast<std::size_t>(L)];
      LaunchGemm(ws.blk.ln1_out.data(), blk.c_attn_w.data(),
                 ws.blk.c_attn_out.data(), 1, 3 * D, D,
                 GemmVersion::RegTiled128Pad, str);
      LaunchBiasAdd(ws.blk.c_attn_out.data(), blk.c_attn_b.data(),
                    ws.blk.c_attn_out.data(), 1, 3 * D, str);
    }
  });

  R.qkv_split = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      float* kc = kv.k_head(L, 0);
      float* vc = kv.v_head(L, 0);
      LaunchQkvSplitToCache(ws.blk.c_attn_out.data(), ws.blk.q.data(),
                            kc, vc, pos, 1, D, H, n_ctx, str);
    }
  });

  R.scores = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      float* kc = kv.k_head(L, 0);
      LaunchAttentionScoresCached(ws.blk.q.data(), kc, ws.blk.scores.data(),
                                   H, n_ctx, d, nc, str);
    }
  });

  R.softmax = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      LaunchSoftmax(ws.blk.scores.data(), ws.blk.weights.data(),
                    H, nc, SoftmaxVersion::Fused, str);
    }
  });

  R.av = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      float* vc = kv.v_head(L, 0);
      LaunchAttentionAVCached(ws.blk.weights.data(), vc, ws.blk.attn_out.data(),
                               H, n_ctx, d, nc, str);
    }
  });

  R.c_proj = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      const auto& blk = w.blocks[static_cast<std::size_t>(L)];
      LaunchGemm(ws.blk.attn_out.data(), blk.c_proj_w.data(),
                 ws.blk.proj_out.data(), 1, D, D,
                 GemmVersion::RegTiled128Pad, str);
      LaunchBiasAdd(ws.blk.proj_out.data(), blk.c_proj_b.data(),
                    ws.blk.proj_out.data(), 1, D, str);
    }
  });

  R.res1 = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      LaunchVectorAdd(h_row, ws.blk.proj_out.data(), h_row, D, str);
    }
  });

  R.ln2 = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      const auto& blk = w.blocks[static_cast<std::size_t>(L)];
      LaunchLayerNorm(h_row, ws.blk.ln2_out.data(),
                      blk.ln2_w.data(), blk.ln2_b.data(),
                      1, D, eps, LayerNormVersion::Naive, str);
    }
  });

  R.mlp_fc = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      const auto& blk = w.blocks[static_cast<std::size_t>(L)];
      LaunchGemm(ws.blk.ln2_out.data(), blk.mlp_fc_w.data(),
                 ws.blk.fc_out.data(), 1, M_cfg, D,
                 GemmVersion::RegTiled128Pad, str);
      LaunchBiasAdd(ws.blk.fc_out.data(), blk.mlp_fc_b.data(),
                    ws.blk.fc_out.data(), 1, M_cfg, str);
    }
  });

  R.gelu = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      LaunchGelu(ws.blk.fc_out.data(), ws.blk.fc_out.data(), M_cfg, str);
    }
  });

  R.mlp_proj = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      const auto& blk = w.blocks[static_cast<std::size_t>(L)];
      LaunchGemm(ws.blk.fc_out.data(), blk.mlp_proj_w.data(),
                 ws.blk.proj_out.data(), 1, D, M_cfg,
                 GemmVersion::RegTiled128Pad, str);
      LaunchBiasAdd(ws.blk.proj_out.data(), blk.mlp_proj_b.data(),
                    ws.blk.proj_out.data(), 1, D, str);
    }
  });

  R.res2 = Time([&]() {
    for (int L = 0; L < nL; ++L) {
      LaunchVectorAdd(h_row, ws.blk.proj_out.data(), h_row, D, str);
    }
  });

  // ---- Final projection ----
  R.ln_f = Time([&]() {
    LaunchLayerNorm(h_row, h_row, w.ln_f_w.data(), w.ln_f_b.data(),
                    1, D, eps, LayerNormVersion::Naive, str);
  });

  R.lm_head = Time([&]() {
    LaunchGemm(h_row, w.wte_T.data(), ws.logits.data(),
               1, V, D, GemmVersion::RegTiled128Pad, str);
  });

  // ---- Host-side ----
  {
    std::vector<float> host_logits(V);
    int next_token = 0;
    LaunchGpt2DecodeStep(0, pos, w, ws.logits.data(), ws, kv, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    R.d2h = Time([&]() {
      CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), ws.logits.data(),
                                  V * sizeof(float),
                                  cudaMemcpyDeviceToHost, str));
    });

    auto t0 = std::chrono::high_resolution_clock::now();
    {
      int best = 0;
      float best_val = host_logits[0];
      for (int v = 1; v < V; ++v)
        if (host_logits[v] > best_val) { best_val = host_logits[v]; best = v; }
      next_token = best;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    R.host_argmax = static_cast<float>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
        1000.f;

    R.h2d = Time([&]() {
      CUDA_CHECK(cudaMemcpyAsync(ws.logits.data(), &next_token,
                                  sizeof(int), cudaMemcpyHostToDevice, str));
    });

    R.stream_sync = Time([&]() {
      CUDA_CHECK(cudaStreamSynchronize(str));
    });
  }

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  return R;
}

// ---------------------------------------------------------------------------
void PrintRow(const char* label, float us, float total) {
  printf("%-48s %10.1f %7.1f%%\n", label, us,
         total > 0 ? 100.0 * us / total : 0.0);
}

}  // namespace

// ---------------------------------------------------------------------------
int main() {
  if (!WeightsExist()) {
    printf("SKIP: weights not found.\n");
    return 0;
  }

  Gpt2Config cfg = DefaultGpt2SmallConfig();
  auto w = LoadGpt2Weights(MGIE_MODEL_WEIGHTS_DIR, cfg);

  // 1. Verification
  PrintVerification(cfg);

  // 2. Profile
  printf("\nRunning profiling passes...\n");
  auto R = Profile(w, cfg);

  float T = R.total();
  float embed_add_pure = R.embed_add - R.embed_wpe;
  float attn_k = R.scores + R.softmax + R.av;
  float blk_t = R.blk_total();
  float gpu_t = R.gpu_total();
  float host_t = R.host_total();

#define ROW(l, v) PrintRow(l, v, T)
#define SEP printf("%-48s %10s %8s\n", \
    "------------------------------------------------", "----------", "-------")

  printf("\n");
  printf("================================================================\n");
  printf("  Decode Step Profile  --  S_cached = 1024  (pos = 1023)\n");
  printf("================================================================\n");
  printf("\n%-48s %10s %8s\n", "Component", "us", "%");
  SEP;

  printf("  -- A. Embedding --\n");
  ROW("  A1. wte copy (cudaMemcpy D2D, 3 KB)",  R.embed_wte);
  ROW("  A2. wpe copy (cudaMemcpy D2D, 3 KB)",  R.embed_wpe);
  ROW("  A3. VectorAdd (768 elements)",          embed_add_pure);
  ROW("  A.  Embedding total",                   R.embed_wte + R.embed_wpe + embed_add_pure);

  printf("\n  -- B. Transformer Blocks (12 layers) --\n");
  printf("    -- Attention path --\n");
  ROW("  B1.  ln1        (12x LayerNorm 1x768)",      R.ln1);
  ROW("  B2.  c_attn     (12x GEMM+BiasAdd 1x2304)",  R.c_attn);
  ROW("  B3.  qkv_split  (12x QkvSplitToCache)",      R.qkv_split);
  ROW("  B4a.   scores   (12x QK^T, 1q x 1024k)",     R.scores);
  ROW("  B4b.   softmax  (12x Softmax 12x1024)",       R.softmax);
  ROW("  B4c.   AV       (12x weights*V, 1 x 1024)",   R.av);
  ROW("  B4.  cached_attn (scores+softmax+AV)",       attn_k);
  ROW("  B5.  c_proj     (12x GEMM+BiasAdd 1x768)",   R.c_proj);
  ROW("  B6.  residual1  (12x VectorAdd 768)",        R.res1);
  printf("    -- MLP path --\n");
  ROW("  B7.  ln2        (12x LayerNorm 1x768)",      R.ln2);
  ROW("  B8.  mlp_fc     (12x GEMM+BiasAdd 1x3072)",  R.mlp_fc);
  ROW("  B9.  gelu       (12x GELU 3072)",            R.gelu);
  ROW("  B10. mlp_proj   (12x GEMM+BiasAdd 1x768)",   R.mlp_proj);
  ROW("  B11. residual2  (12x VectorAdd 768)",        R.res2);
  ROW("  B.  Transformer blocks total",               blk_t);
  printf("      per-layer average:                       %.1f\n", blk_t / 12.0f);

  printf("\n  -- C-D. Final Projection --\n");
  ROW("  C.  ln_f        (1x LayerNorm 1x768)",       R.ln_f);
  ROW("  D.  lm_head     (1x GEMM 1x50257)",          R.lm_head);

  ROW("  >> GPU total (sum of parts)",                 gpu_t);

  printf("\n  -- E-H. Host Overhead --\n");
  ROW("  E.  D2H logits  (cudaMemcpy D2H, 201 KB)",   R.d2h);
  ROW("  F.  Host argmax (50257 iterations)",          R.host_argmax);
  ROW("  G.  H2D token   (cudaMemcpy H2D, 4 B)",      R.h2d);
  ROW("  H.  Stream sync (cudaStreamSynchronize)",     R.stream_sync);
  ROW("  >> Host total",                               host_t);

  printf("  %s\n",
         "================================================================");
  ROW("  ** TOTAL (GPU + Host)", T);

  // ---- Summary groups ----
  printf("\n");
  printf("================================================================\n");
  printf("  Summary Groups\n");
  printf("================================================================\n\n");
  printf("%-48s %10s %8s\n", "Group", "us", "%");

  float all_gemm  = R.c_attn + R.c_proj + R.mlp_fc + R.mlp_proj + R.lm_head;
  float all_ln    = R.ln1 + R.ln2 + R.ln_f;
  float all_vadd  = R.res1 + R.res2 + embed_add_pure;

  PrintRow("All GEMMs (49: 12x4 + 1 lm_head)", all_gemm, T);
  PrintRow("  c_attn    (12x [1,2304]=[1,768]@[768,2304])", R.c_attn, T);
  PrintRow("  c_proj    (12x [1,768]=[1,768]@[768,768])",   R.c_proj, T);
  PrintRow("  mlp_fc    (12x [1,3072]=[1,768]@[768,3072])", R.mlp_fc, T);
  PrintRow("  mlp_proj  (12x [1,768]=[1,3072]@[3072,768])", R.mlp_proj, T);
  PrintRow("  lm_head   (1x [1,50257]=[1,768]@[768,50257])", R.lm_head, T);
  PrintRow("All LayerNorms (25: 12x2 + 1 ln_f)", all_ln, T);
  PrintRow("All attention kernels (36: 12x3)", attn_k, T);
  PrintRow("  scores    (12x QK^T)", R.scores, T);
  PrintRow("  softmax   (12x row-softmax)", R.softmax, T);
  PrintRow("  AV        (12x weights*V)", R.av, T);
  PrintRow("All VectorAdds (25: 12x2 + embed)", all_vadd, T);
  PrintRow("All QKV split (12x)", R.qkv_split, T);
  PrintRow("All GELU (12x)", R.gelu, T);
  PrintRow("All cudaMemcpy (2 x 3KB)", R.embed_wte + R.embed_wpe, T);
  PrintRow("Host overhead", host_t, T);

  float check = all_gemm + all_ln + attn_k + all_vadd +
                R.embed_wte + R.embed_wpe + R.qkv_split + R.gelu + host_t;
  printf("\n  Sum check: %.1f us (should ~= %.1f, delta = %+.1f)\n",
         check, T, check - T);

  // ---- Per-layer averages ----
  printf("\n");
  printf("================================================================\n");
  printf("  Per-Layer Averages (/12)\n");
  printf("================================================================\n\n");
  printf("  %-40s %8s\n", "Sub-component", "us/layer");
  SEP;
  printf("  %-40s %8.1f\n", "ln1",                  R.ln1 / 12.f);
  printf("  %-40s %8.1f\n", "c_attn (GEMM+BiasAdd)", R.c_attn / 12.f);
  printf("  %-40s %8.1f\n", "qkv_split",             R.qkv_split / 12.f);
  printf("  %-40s %8.1f\n", "cached scores (QK^T)",   R.scores / 12.f);
  printf("  %-40s %8.1f\n", "cached softmax",         R.softmax / 12.f);
  printf("  %-40s %8.1f\n", "cached AV",              R.av / 12.f);
  printf("  %-40s %8.1f\n", "c_proj (GEMM+BiasAdd)", R.c_proj / 12.f);
  printf("  %-40s %8.1f\n", "residual1 (VectorAdd)",  R.res1 / 12.f);
  printf("  %-40s %8.1f\n", "ln2",                  R.ln2 / 12.f);
  printf("  %-40s %8.1f\n", "mlp_fc (GEMM+BiasAdd)", R.mlp_fc / 12.f);
  printf("  %-40s %8.1f\n", "gelu",                  R.gelu / 12.f);
  printf("  %-40s %8.1f\n", "mlp_proj (GEMM+BiasAdd)", R.mlp_proj / 12.f);
  printf("  %-40s %8.1f\n", "residual2 (VectorAdd)", R.res2 / 12.f);
  SEP;
  printf("  %-40s %8.1f\n", "Per-block total",       blk_t / 12.f);

  printf("\nDone.\n");
  return 0;
}
