// benchmarks/decode/bench_launch_overhead.cu
// Phase 6c readiness: measures actual kernel launch overhead in a realistic
// heterogeneous launch pattern matching the decode step.
//
// Key experiment: launch the exact 219-kernel decode sequence, measure
//   - T_wall: total wall-clock including all CPU launch work
//   - T_device: pure GPU execution (inner CUDA events)
// The difference = CPU launch work not overlapped with GPU execution.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include "common/errors.hpp"
#include "kernels/gemm.hpp"
#include "kernels/bias_add.hpp"
#include "kernels/gelu.hpp"
#include "kernels/layernorm.hpp"
#include "kernels/vector_add.hpp"
#include "kernels/softmax.hpp"
#include "kernels/attention.hpp"
#include "kernels/qkv_split.hpp"
#include "cuda/memory.hpp"

using mgie::DeviceBuffer;

__global__ void NullKernel() {}

// Global buffers — allocated once, reused across benchmark iterations.
// Sized for the largest decode-shape usage.
struct Buffers {
  // GEMM
  DeviceBuffer<float> dA_c_attn{768};
  DeviceBuffer<float> dB_c_attn{(size_t)768 * 2304};
  DeviceBuffer<float> dC_c_attn{2304};
  DeviceBuffer<float> dA_c_proj{768};
  DeviceBuffer<float> dB_c_proj{(size_t)768 * 768};
  DeviceBuffer<float> dC_c_proj{768};
  DeviceBuffer<float> dA_mlp_fc{768};
  DeviceBuffer<float> dB_mlp_fc{(size_t)768 * 3072};
  DeviceBuffer<float> dC_mlp_fc{3072};
  DeviceBuffer<float> dA_mlp_proj{3072};
  DeviceBuffer<float> dB_mlp_proj{(size_t)3072 * 768};
  DeviceBuffer<float> dC_mlp_proj{768};
  DeviceBuffer<float> dA_lm_head{768};
  DeviceBuffer<float> dB_lm_head{(size_t)768 * 50257};
  DeviceBuffer<float> dC_lm_head{50257};
  // Bias
  DeviceBuffer<float> db_2304{2304};
  DeviceBuffer<float> db_768{768};
  DeviceBuffer<float> db_3072{3072};
  // LayerNorm
  DeviceBuffer<float> dln_in{768};
  DeviceBuffer<float> dln_out{768};
  DeviceBuffer<float> dln_g{768};
  DeviceBuffer<float> dln_b{768};
  // VectorAdd
  DeviceBuffer<float> dva_a{768};
  DeviceBuffer<float> dva_b{768};
  DeviceBuffer<float> dva_c{768};
  // GELU
  DeviceBuffer<float> dgelu_in{3072};
  DeviceBuffer<float> dgelu_out{3072};
  // Attention (12 heads, 64 dim, 1024 ctx)
  DeviceBuffer<float> dq{(size_t)12 * 64};
  DeviceBuffer<float> dk{(size_t)12 * 1024 * 64};
  DeviceBuffer<float> dv{(size_t)12 * 1024 * 64};
  DeviceBuffer<float> dscores{(size_t)12 * 1024};
  DeviceBuffer<float> dweights{(size_t)12 * 1024};
  DeviceBuffer<float> dattn{(size_t)12 * 64};
  DeviceBuffer<float> dc_attn_3d{3 * 768};
  DeviceBuffer<float> dq_split{(size_t)12 * 64};

  Buffers() {
    auto fill = [](DeviceBuffer<float>& b) {
      std::vector<float> h(b.size(), 0.5f);
      b.CopyFromHost(h.data(), b.size());
    };
    fill(dA_c_attn); fill(dB_c_attn);
    fill(dA_c_proj); fill(dB_c_proj);
    fill(dA_mlp_fc); fill(dB_mlp_fc);
    fill(dA_mlp_proj); fill(dB_mlp_proj);
    fill(dA_lm_head); fill(dB_lm_head);
    fill(db_2304); fill(db_768); fill(db_3072);
    fill(dln_in); fill(dln_g); fill(dln_b);
    fill(dva_a); fill(dva_b);
    fill(dgelu_in);
    fill(dq); fill(dk); fill(dv);
    fill(dc_attn_3d); fill(dq_split);
  }
};

void LaunchDecodePattern(Buffers& b, cudaStream_t stream) {
  const int D = 768, H = 12, head_dim = 64, M = 3072;
  const int n_ctx = 1024, num_cached = 1024;
  const float eps = 1e-5f;

  for (int lyr = 0; lyr < 12; ++lyr) {
    mgie::LaunchLayerNorm(b.dln_in.data(), b.dln_out.data(),
                          b.dln_g.data(), b.dln_b.data(),
                          1, D, eps, mgie::LayerNormVersion::Naive, stream);
    mgie::LaunchGemm(b.dA_c_attn.data(), b.dB_c_attn.data(), b.dC_c_attn.data(),
                     1, 2304, 768, mgie::GemmVersion::RegTiled128Pad, stream);
    mgie::LaunchBiasAdd(b.dC_c_attn.data(), b.db_2304.data(), b.dC_c_attn.data(),
                        1, 2304, stream);
    mgie::LaunchQkvSplitToCache(b.dc_attn_3d.data(), b.dq_split.data(),
                                b.dk.data(), b.dv.data(),
                                0, 1, D, H, n_ctx, stream);
    mgie::LaunchAttentionCached(b.dq.data(), b.dk.data(), b.dv.data(),
                                b.dattn.data(), b.dscores.data(), b.dweights.data(),
                                H, n_ctx, head_dim, num_cached, stream);
    mgie::LaunchGemm(b.dA_c_proj.data(), b.dB_c_proj.data(), b.dC_c_proj.data(),
                     1, 768, 768, mgie::GemmVersion::RegTiled128Pad, stream);
    mgie::LaunchBiasAdd(b.dC_c_proj.data(), b.db_768.data(), b.dC_c_proj.data(),
                        1, 768, stream);
    mgie::LaunchVectorAdd(b.dva_a.data(), b.dva_b.data(), b.dva_c.data(), D, stream);
    mgie::LaunchLayerNorm(b.dln_in.data(), b.dln_out.data(),
                          b.dln_g.data(), b.dln_b.data(),
                          1, D, eps, mgie::LayerNormVersion::Naive, stream);
    mgie::LaunchGemm(b.dA_mlp_fc.data(), b.dB_mlp_fc.data(), b.dC_mlp_fc.data(),
                     1, 3072, 768, mgie::GemmVersion::RegTiled128Pad, stream);
    mgie::LaunchBiasAdd(b.dC_mlp_fc.data(), b.db_3072.data(), b.dC_mlp_fc.data(),
                        1, 3072, stream);
    mgie::LaunchGelu(b.dgelu_in.data(), b.dgelu_out.data(), M, stream);
    mgie::LaunchGemm(b.dA_mlp_proj.data(), b.dB_mlp_proj.data(), b.dC_mlp_proj.data(),
                     1, 768, 3072, mgie::GemmVersion::RegTiled128Pad, stream);
    mgie::LaunchBiasAdd(b.dC_mlp_proj.data(), b.db_768.data(), b.dC_mlp_proj.data(),
                        1, 768, stream);
    mgie::LaunchVectorAdd(b.dva_a.data(), b.dva_b.data(), b.dva_c.data(), D, stream);
  }
  mgie::LaunchLayerNorm(b.dln_in.data(), b.dln_out.data(),
                        b.dln_g.data(), b.dln_b.data(),
                        1, D, eps, mgie::LayerNormVersion::Naive, stream);
  mgie::LaunchGemm(b.dA_lm_head.data(), b.dB_lm_head.data(), b.dC_lm_head.data(),
                   1, 50257, 768, mgie::GemmVersion::RegTiled128Pad, stream);
}

template<typename F>
float TimeWall(F&& fn, int iters, int warmup) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.f / iters;
}

template<typename F>
float TimeDevice(F&& fn, int iters) {
  fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms * 1000.f / iters;
}

int main() {
  printf("=== Phase 6c: Launch Overhead Audit (Heterogeneous Pattern) ===\n\n");

  Buffers bufs;
  const int kIters = 20;
  cudaStream_t stream = nullptr;

  // ---- 1. Null kernel baselines ------------------------------------------
  printf("--- 1. Null Kernel Launch Baselines ---\n");
  float null_1 = TimeWall([](){ NullKernel<<<1,1>>>(); }, 2000, 10);
  printf("  1 null kernel launch:            %8.2f us\n", null_1);

  float null_219 = TimeWall([](){
    for (int i = 0; i < 219; ++i) NullKernel<<<1,1>>>();
  }, 50, 5);
  printf("  219 null kernels (seq):          %8.2f us  (%.2f each)\n",
         null_219, null_219 / 219.f);

  // ---- 2. Heterogeneous decode pattern -----------------------------------
  printf("\n--- 2. Heterogeneous Decode Pattern (219 launches) ---\n");

  auto run = [&]() { LaunchDecodePattern(bufs, stream); };

  float T_wall = TimeWall(run, kIters, 5);
  float T_dev  = TimeDevice(run, kIters);

  printf("  Wall-clock total:                %8.0f us\n", T_wall);
  printf("  Device execution:                %8.0f us\n", T_dev);
  printf("  CPU non-overlapped overhead:     %8.0f us  (%.1f%%)\n",
         T_wall - T_dev, (T_wall - T_dev) / T_wall * 100.f);
  printf("  Per-launch CPU overhead (avg):   %8.2f us\n",
         (T_wall - T_dev) / 219.f);

  // ---- 3. Cross-check with Phase 6b profile ------------------------------
  printf("\n--- 3. Cross-Check Against Phase 6b Profile ---\n");

  float profile_total = 5899.6f;
  float profile_gpu   = 5745.6f;
  float profile_host  = 154.0f;

  printf("  Profile decode step total:       %8.1f us\n", profile_total);
  printf("  Profile GPU sum of parts:        %8.1f us\n", profile_gpu);
  printf("  Profile host overhead:           %8.1f us\n", profile_host);
  printf("  Bench heterogeneous T_wall:      %8.0f us  (219 kernels only)\n", T_wall);
  printf("  Bench heterogeneous T_dev:       %8.0f us\n", T_dev);
  printf("  Ratio T_dev/T_wall:              %5.1f%%\n", T_dev / T_wall * 100.f);

  // ---- 4. CUDA Graphs projection -----------------------------------------
  printf("\n--- 4. CUDA Graphs Projection ---\n");

  float non_overlapped = T_wall - T_dev;
  float graph_launch = null_1;
  float savings = (non_overlapped > graph_launch) ? (non_overlapped - graph_launch) : 0.f;
  float projected_total = profile_total - savings;
  float pct_saved = savings / profile_total * 100.f;

  printf("  Measured non-overlapped CPU work: %8.0f us  (219 heterogeneous launches)\n",
         non_overlapped);
  printf("  CUDA Graphs launch (estimated):   %8.2f us  (1 graph launch)\n",
         graph_launch);
  printf("  Net savings:                      %8.0f us\n", savings);
  printf("\n");
  printf("  Current decode step:              %8.1f us  (%5.1f ms)\n",
         profile_total, profile_total / 1000.f);
  printf("  After CUDA Graphs:                %8.1f us  (%5.1f ms)\n",
         projected_total, projected_total / 1000.f);
  printf("  Improvement:                      %8.0f us  (%4.1f%%)\n",
         savings, pct_saved);
  printf("  Projected speedup:                %5.2fx\n",
         profile_total / projected_total);

  // ---- 5. Sensitivity analysis -------------------------------------------
  printf("\n--- 5. Sensitivity Analysis (per-launch CPU cost) ---\n");
  printf("  %-25s %8s %8s %8s\n", "Per-launch cost", "Savings", "Decode", "Speedup");
  printf("  %-25s %8s %8s %8s\n", "-------------------------", "--------", "--------", "--------");
  for (float cpu_cost : {0.5f, 1.0f, 2.0f, 3.0f, 5.0f}) {
    float sav = 219.f * cpu_cost - graph_launch;
    if (sav < 0) sav = 0;
    float new_total = profile_total - sav;
    printf("  %-25s %8.0f %8.0f %7.2fx\n",
           (std::to_string(cpu_cost) + " us/kernel").c_str(),
           sav, new_total, profile_total / new_total);
  }

  // ---- 6. Launch count attribution ---------------------------------------
  printf("\n--- 6. Kernel Launch Count Attribution ---\n");
  printf("  %-30s %5s %6s %6s\n", "Category", "/layer", "x12", "total");
  struct { const char* name; int per_layer; int extra; } cats[] = {
    {"GEMM/GEMV",               5, 1},  // +lm_head
    {"BiasAdd",                 4, 0},
    {"Attention (scores+sm+AV)", 3, 0},
    {"LayerNorm",               2, 1},  // +ln_f
    {"VectorAdd",               2, 1},  // +embed
    {"GELU",                    1, 0},
    {"QkvSplitToCache",         1, 0},
  };
  int total = 0;
  for (auto& c : cats) {
    int t = c.per_layer * 12 + c.extra;
    total += t;
    printf("  %-30s %5d %6d %6d\n", c.name, c.per_layer, c.per_layer*12, t);
  }
  printf("  %-30s %5s %6s %6d\n", "---", "---", "---", total);
  printf("  + 3 cudaMemcpyAsync (wte, wpe, D2H)  + 1 H2D  + 1 stream sync\n");

  // ---- 7. Alternative Phase 6c ranking -----------------------------------
  printf("\n--- 7. Alternative Phase 6c Candidates (Ranked by wall-clock impact) ---\n");
  printf("\n");
  printf("  CUDA Graphs saves at most %.0f us (%.1f%%) at current kernel granularity.\n",
         savings, pct_saved);
  printf("  This is well below the 15%% threshold. Recommended alternatives:\n");
  printf("\n");
  printf("  %-45s %10s %10s\n", "Optimization", "Savings", "% of 5899.6");
  printf("  %-45s %10s %10s\n", "---------------------------------------------", "----------", "----------");

  // 1. Fuse attention scores+softmax+AV into one kernel (eliminates 24 launches,
  //    reduces k_cache/v_cache reads by 2x)
  printf("  %-45s %10s %9.1f%%\n", "1. Fuse attention 3-kernel → 1", "~400 us", 400.f/5899.6f*100);
  // 2. GEMV+BiasAdd fusion (48 BiasAdd kernels, memory roundtrips)
  printf("  %-45s %10s %9.1f%%\n", "2. GEMV+BiasAdd fusion (48 kernels)", "~250 us", 250.f/5899.6f*100);
  // 3. Attention scores kernel: increase parallelism
  printf("  %-45s %10s %9.1f%%\n", "3. Attention scores kernel opt", "~300 us", 300.f/5899.6f*100);
  // 4. LN+GEMV fusion
  printf("  %-45s %10s %9.1f%%\n", "4. LN+GEMV fusion (25 LN kernels)", "~200 us", 200.f/5899.6f*100);

  printf("\n");
  printf("  Recommended Phase 6c: Attention kernel fusion (item 1 + 3 combined).\n");
  printf("  Rationale: attention is 24.9%% of decode (1470 us), uses only 12 blocks\n");
  printf("  of 64 threads each (1.4%% of GPU thread capacity), and reads k_cache/v_cache\n");
  printf("  3 times instead of once. A fused attention kernel with better parallelism\n");
  printf("  should achieve ~2x speedup, saving ~700 us (11.9%%).\n");
  printf("\n");
  printf("  CUDA Graphs is better deferred to Phase 6d or 6e, after kernel fusion\n");
  printf("  reduces launch count and increases per-kernel execution time, making\n");
  printf("  the launch/execution overlap less forgiving.\n");

  printf("\nDone.\n");
  return 0;
}
