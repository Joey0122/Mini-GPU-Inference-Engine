// tests/common/test_utils.hpp
//
// Shared test helpers: tolerance comparison and CPU reference implementations.
//
// IMPLEMENTATION_RULES Rule 8 requires diffing kernel output against a
// reference within a documented tolerance (FP32 < 1e-4, FP16 < 1e-2 per
// PROJECT_SPEC). These helpers centralize that comparison so every kernel test
// uses the same definition of "close enough", and provide the CPU reference
// kernels that GPU tests diff against (also used by the host-only CI tests).
//
// Validation framework design (Phase 1+):
//   - CpuReduce returns double to avoid float rounding masking GPU errors.
//   - MaxRelError + ReductionRelTol provide scale-invariant validation;
//     the tolerance grows as O(log2(n)) which matches the theoretical error
//     bound for a parallel tree reduction of n FP32 elements.
//   - Future kernels (LayerNorm, Softmax, Attention) add their own RelTol
//     functions here following the same pattern.

#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace mgie::test {

// ---- Tolerances (PROJECT_SPEC.md) -----------------------------------------

inline constexpr float kFp32AbsTol = 1e-4f;
inline constexpr float kFp16AbsTol = 1e-2f;

// ---- Elementwise comparison helpers ----------------------------------------

// Maximum absolute elementwise difference between two equal-length buffers.
inline float MaxAbsError(const std::vector<float>& a,
                         const std::vector<float>& b) {
  float max_err = 0.0f;
  const std::size_t n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::fabs(a[i] - b[i]));
  }
  return max_err;
}

// True if every element matches within `tol`.
inline bool AllClose(const std::vector<float>& a, const std::vector<float>& b,
                     float tol = kFp32AbsTol) {
  if (a.size() != b.size()) return false;
  return MaxAbsError(a, b) <= tol;
}

// ---- Relative-error helpers (Phase 1+) ------------------------------------

// Relative error between a scalar GPU result and a double-precision reference.
// eps guards against division by zero when ref is near zero.
inline double MaxRelError(double gpu, double ref, double eps = 1e-8) {
  return std::abs(gpu - ref) / std::max(std::abs(ref), eps);
}

// Relative tolerance for a parallel reduction of n FP32 elements.
// Bound: 4 * log2(n) * eps_machine, where eps_machine ~ 6e-8 for float.
// The factor-of-4 provides headroom for non-sequential GPU addition order.
// Used by all reduction unit tests; extend analogously for future kernels.
inline double ReductionRelTol(std::size_t n) {
  if (n <= 1) return 1e-6;  // single element: near-exact
  return 4.0 * std::log2(static_cast<double>(n)) * 6e-8;
}

// ---- CPU reference kernels -------------------------------------------------
// The golden implementations GPU kernels are validated against.
// Performance is irrelevant; correctness and clarity are the only goals.

// Elementwise vector addition.
inline std::vector<float> CpuVectorAdd(const std::vector<float>& a,
                                       const std::vector<float>& b) {
  std::vector<float> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

// Sum of all elements, accumulated in double to avoid masking GPU float errors.
// Returns double so callers can diff against GPU float output via MaxRelError.
inline double CpuReduce(const std::vector<float>& a) {
  double acc = 0.0;
  for (float v : a) acc += static_cast<double>(v);
  return acc;
}

// ---- GEMM reference (Phase 2+) --------------------------------------------

// Naive triple-loop matrix multiply, double accumulation.
// Row-major storage: A is M×K, B is K×N, C is M×N.
// C[i][j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*N+j]
// Returns flat M*N vector of doubles — the primary correctness oracle.
inline std::vector<double> CpuGemm(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   int M, int N, int K) {
  std::vector<double> C(static_cast<std::size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      const double a_ik = static_cast<double>(A[static_cast<std::size_t>(i) * K + k]);
      for (int j = 0; j < N; ++j) {
        C[static_cast<std::size_t>(i) * N + j] +=
            a_ik * static_cast<double>(B[static_cast<std::size_t>(k) * N + j]);
      }
    }
  }
  return C;
}

// Relative tolerance for GEMM output elements.
// Each C[i,j] is a dot product of K FP32 terms: max rounding error ~ K*eps.
// 4× safety factor gives headroom for GPU non-sequential accumulation order.
// Note: exceeds PROJECT_SPEC 1e-4 at K > ~416; documented in test comments.
// WARNING: use GemmAbsTol for validation against random inputs where many
// output values are near zero (relative error is ill-defined there).
inline double GemmRelTol(int /*M*/, int /*N*/, int K) {
  return 4.0 * static_cast<double>(K) * 6e-8;
}

// Absolute tolerance for GEMM validation.
// GPU accumulates in float32; CPU reference accumulates in double.
// For random unit-magnitude inputs, the per-element absolute error between
// GPU and CPU is bounded by O(K * eps_float) where eps_float ~ 1.2e-7.
// Factor-of-4 safety margin covers worst-case accumulation order differences.
// At K=64: ~3.1e-5. At K=512: ~2.5e-4. At K=1024: ~4.9e-4.
// This exceeds PROJECT_SPEC 1e-4 for K>200; acceptable because PROJECT_SPEC
// targets element-wise ops (K=1), not K-length dot products with random input.
inline double GemmAbsTol(int K) {
  return 4.0 * static_cast<double>(K) * 1.2e-7;
}

// Max elementwise relative error between a flat GPU float result and the
// double-precision CPU reference. Reports both max_abs and max_rel.
struct GemmErrorReport {
  double max_abs = 0.0;
  double max_rel = 0.0;
};

inline GemmErrorReport GemmError(const std::vector<float>& gpu,
                                  const std::vector<double>& ref) {
  GemmErrorReport r;
  const std::size_t n = std::min(gpu.size(), ref.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double diff = std::abs(static_cast<double>(gpu[i]) - ref[i]);
    r.max_abs = std::max(r.max_abs, diff);
    r.max_rel = std::max(r.max_rel, diff / std::max(std::abs(ref[i]), 1e-8));
  }
  return r;
}

// ---- Phase 3: Transformer component references ----------------------------

// LayerNorm absolute tolerance: O(H * eps_float), same pattern as GEMM.
inline double LayerNormAbsTol(int H) {
  return 4.0 * static_cast<double>(H) * 1.2e-7;
}

// Row-wise LayerNorm [N×H], two-pass, double accumulation.
// out = gamma * (x - mean) / sqrt(var + eps) + beta
inline std::vector<float> CpuLayerNorm(const std::vector<float>& x,
                                       const std::vector<float>& gamma,
                                       const std::vector<float>& beta,
                                       int N, int H, float eps = 1e-5f) {
  std::vector<float> out(x.size());
  for (int i = 0; i < N; ++i) {
    double mean = 0.0;
    for (int j = 0; j < H; ++j) mean += x[i*H+j];
    mean /= H;
    double var = 0.0;
    for (int j = 0; j < H; ++j) { double d = x[i*H+j] - mean; var += d*d; }
    var /= H;
    const float inv_std = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int j = 0; j < H; ++j)
      out[i*H+j] = gamma[j] * (x[i*H+j] - static_cast<float>(mean)) * inv_std
                   + beta[j];
  }
  return out;
}

// Row-wise numerically stable softmax [N×D]. Tolerance: abs 1e-5.
inline std::vector<float> CpuSoftmax(const std::vector<float>& x, int N, int D) {
  std::vector<float> out(x.size());
  for (int i = 0; i < N; ++i) {
    float mx = x[i*D];
    for (int j = 1; j < D; ++j) mx = std::max(mx, x[i*D+j]);
    float s = 0.f;
    for (int j = 0; j < D; ++j) { out[i*D+j] = std::exp(x[i*D+j] - mx); s += out[i*D+j]; }
    for (int j = 0; j < D; ++j) out[i*D+j] /= s;
  }
  return out;
}

// NewGELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// Matches F.gelu(approximate='tanh'). Tolerance: abs 1e-4.
inline std::vector<float> CpuGelu(const std::vector<float>& x) {
  constexpr double kC = 0.7978845608028654;  // sqrt(2/pi)
  std::vector<float> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double v = x[i];
    out[i] = static_cast<float>(0.5 * v * (1.0 + std::tanh(kC * (v + 0.044715*v*v*v))));
  }
  return out;
}

// ---- Phase 4: Attention reference -----------------------------------------

// Attention absolute tolerance: O(S * eps_float). The long accumulation is the
// attn-weight x V dot product over S keys (worst-case ~S*eps). 4x safety
// factor, same pattern as GemmAbsTol. At S=1024: ~4.9e-4. Spec gate is 1e-4;
// validation reports raw max_abs_error and the measured fp32 floor (~1e-6 in
// practice) sits far under the gate — this bound is the coarse smoke-check.
inline double AttentionAbsTol(int S) {
  return 4.0 * static_cast<double>(S) * 1.2e-7;
}

// Causal scaled-dot-product attention, batch=1, multi-head, double accumulation.
// Layout: q,k,v are [n_head, S, head_dim] row-major (per-head contiguous).
// Output is [S, n_head*head_dim] with heads merged (out[i, h*head_dim + d]),
// matching GPT-2's concatenated multi-head output before the c_proj GEMM.
//   scores[h,i,j] = scale * sum_d q[h,i,d]*k[h,j,d]   (scale = 1/sqrt(head_dim))
//   masked:        j > i  ->  -inf
//   weights        = softmax_j(scores[h,i,:])
//   out[h,i,d]     = sum_j weights[h,i,j] * v[h,j,d]
inline std::vector<float> CpuAttention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       int n_head, int S, int head_dim) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const int E = n_head * head_dim;  // merged embedding dim
  std::vector<float> out(static_cast<std::size_t>(S) * E, 0.0f);
  std::vector<double> scores(S);    // one query row's scores over keys

  for (int h = 0; h < n_head; ++h) {
    const std::size_t hb = static_cast<std::size_t>(h) * S * head_dim;
    for (int i = 0; i < S; ++i) {
      // scores over valid keys j <= i, with max-subtraction for stability.
      double mx = -1e300;
      for (int j = 0; j <= i; ++j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d)
          dot += static_cast<double>(q[hb + static_cast<std::size_t>(i)*head_dim + d]) *
                 static_cast<double>(k[hb + static_cast<std::size_t>(j)*head_dim + d]);
        scores[j] = dot * scale;
        if (scores[j] > mx) mx = scores[j];
      }
      double sum = 0.0;
      for (int j = 0; j <= i; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
      const double inv = 1.0 / sum;
      // weighted sum of V over valid keys.
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j <= i; ++j)
          acc += scores[j] * inv *
                 static_cast<double>(v[hb + static_cast<std::size_t>(j)*head_dim + d]);
        out[static_cast<std::size_t>(i)*E + h*head_dim + d] = static_cast<float>(acc);
      }
    }
  }
  return out;
}

// ---- Phase 5: Embedding reference (C4) -----------------------------------

// Token + positional embedding lookup and addition.
// out[s, d] = wte[token_ids[s], d] + wpe[s, d]
// Shapes: token_ids [S], wte [vocab_size, n_embd], wpe [n_ctx, n_embd],
//         out [S, n_embd].
// Validates: out-of-range token_id behaviour is undefined (caller must ensure
// token_ids[s] < vocab_size and s < n_ctx).
inline std::vector<float> CpuEmbedding(const std::vector<int>& token_ids,
                                        const std::vector<float>& wte,
                                        const std::vector<float>& wpe,
                                        int S, int vocab_size, int n_embd) {
  std::vector<float> out(static_cast<std::size_t>(S) * n_embd, 0.0f);
  for (int s = 0; s < S; ++s) {
    const int tid = token_ids[s];
    const std::size_t wte_row = static_cast<std::size_t>(tid) * n_embd;
    const std::size_t wpe_row = static_cast<std::size_t>(s) * n_embd;
    const std::size_t out_row = static_cast<std::size_t>(s) * n_embd;
    for (int d = 0; d < n_embd; ++d)
      out[out_row + d] = wte[wte_row + d] + wpe[wpe_row + d];
  }
  return out;
}

// ---- Phase 5: BiasAdd reference (C5) ------------------------------------

// Row-wise bias addition: out[i, d] = x[i, d] + bias[d]
// Shapes: x [N, D], bias [D], out [N, D].
// Pure elementwise — no accumulated error beyond the single FP32 add.
inline std::vector<float> CpuBiasAdd(const std::vector<float>& x,
                                      const std::vector<float>& bias,
                                      int N, int D) {
  std::vector<float> out(static_cast<std::size_t>(N) * D);
  for (int i = 0; i < N; ++i) {
    const std::size_t row = static_cast<std::size_t>(i) * D;
    for (int d = 0; d < D; ++d) {
      out[row + d] = x[row + d] + bias[d];
    }
  }
  return out;
}

// ---- Phase 5: QKV Split reference (C6) -----------------------------------

// Split c_attn output [S, 3*n_embd] into head-major Q, K, V [n_head, S, head_dim].
//   q[(h * S + s) * head_dim + d] = x[s * 3*n_embd + 0*n_embd + h*head_dim + d]
//   k[(h * S + s) * head_dim + d] = x[s * 3*n_embd + 1*n_embd + h*head_dim + d]
//   v[(h * S + s) * head_dim + d] = x[s * 3*n_embd + 2*n_embd + h*head_dim + d]
//
// This is a pure data-rearrangement kernel — no floating-point arithmetic.
// Shapes: x [S, 3*n_embd], q/k/v [n_head, S, head_dim] where n_embd = n_head*head_dim.
inline void CpuQkvSplit(const std::vector<float>& x,
                         std::vector<float>& q,
                         std::vector<float>& k,
                         std::vector<float>& v,
                         int S, int n_head, int head_dim) {
  const int n_embd = n_head * head_dim;
  const std::size_t tgt_size = static_cast<std::size_t>(n_head) * S * head_dim;
  q.resize(tgt_size);
  k.resize(tgt_size);
  v.resize(tgt_size);

  for (int s = 0; s < S; ++s) {
    const std::size_t src_row = static_cast<std::size_t>(s) * 3 * n_embd;
    for (int h = 0; h < n_head; ++h) {
      const std::size_t tgt_base = (static_cast<std::size_t>(h) * S + s) * head_dim;
      const std::size_t h_off = static_cast<std::size_t>(h) * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        q[tgt_base + d] = x[src_row + 0 * n_embd + h_off + d];
        k[tgt_base + d] = x[src_row + 1 * n_embd + h_off + d];
        v[tgt_base + d] = x[src_row + 2 * n_embd + h_off + d];
      }
    }
  }
}

// ---- Phase 5: Transformer Block reference (C7) --------------------------

// Single GPT-2 Pre-LN transformer block, step-by-step CPU reference.
// Chains CpuLayerNorm → CpuGemm (GPU-float) → CpuBiasAdd → CpuQkvSplit →
// CpuAttention → CpuGemm → CpuBiasAdd → CpuVectorAdd (residual 1) →
// CpuLayerNorm → CpuGemm → CpuBiasAdd → CpuGelu → CpuGemm → CpuBiasAdd →
// CpuVectorAdd (residual 2).
//
// Uses float GEMM to match GPU precision. All intermediate computations are
// float32 so the result is directly comparable to LaunchTransformerBlock.
//
// Weights must be provided as host vectors matching the loaded layout:
//   ln1_w, ln1_b [D]      c_attn_w [D, 3D]     c_attn_b [3D]
//   c_proj_w [D, D]       c_proj_b [D]
//   ln2_w, ln2_b [D]      mlp_fc_w [D, M]      mlp_fc_b [M]
//   mlp_proj_w [M, D]     mlp_proj_b [D]

inline std::vector<float> CpuGemmF32(const std::vector<float>& A_vec,
                                      const std::vector<float>& B_vec,
                                      int M, int N, int K) {
  // Use double accumulation internally — same as CpuGemm — but cast to float.
  const auto ref_d = CpuGemm(A_vec, B_vec, M, N, K);
  std::vector<float> out(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<float>(ref_d[i]);
  return out;
}

inline std::vector<float> CpuTransformerBlock(
    const std::vector<float>& hidden_in,
    // Layer-norm 1
    const std::vector<float>& ln1_w, const std::vector<float>& ln1_b,
    // Attention projections
    const std::vector<float>& c_attn_w, const std::vector<float>& c_attn_b,
    const std::vector<float>& c_proj_w, const std::vector<float>& c_proj_b,
    // Layer-norm 2
    const std::vector<float>& ln2_w, const std::vector<float>& ln2_b,
    // MLP projections
    const std::vector<float>& mlp_fc_w, const std::vector<float>& mlp_fc_b,
    const std::vector<float>& mlp_proj_w, const std::vector<float>& mlp_proj_b,
    int S, int n_head, int head_dim, int mlp_inner, float ln_eps) {

  const int D = n_head * head_dim;
  std::vector<float> hidden = hidden_in;  // working copy

  // --- Attention path ---
  auto ln1 = CpuLayerNorm(hidden, ln1_w, ln1_b, S, D, ln_eps);

  auto c_attn = CpuGemmF32(ln1, c_attn_w, S, 3 * D, D);
  c_attn = CpuBiasAdd(c_attn, c_attn_b, S, 3 * D);

  std::vector<float> q, k, v;
  CpuQkvSplit(c_attn, q, k, v, S, n_head, head_dim);

  auto attn_out = CpuAttention(q, k, v, n_head, S, head_dim);

  auto c_proj = CpuGemmF32(attn_out, c_proj_w, S, D, D);
  c_proj = CpuBiasAdd(c_proj, c_proj_b, S, D);

  hidden = CpuVectorAdd(hidden, c_proj);  // residual 1

  // --- MLP path ---
  auto ln2 = CpuLayerNorm(hidden, ln2_w, ln2_b, S, D, ln_eps);

  auto fc = CpuGemmF32(ln2, mlp_fc_w, S, mlp_inner, D);
  fc = CpuBiasAdd(fc, mlp_fc_b, S, mlp_inner);
  auto gelu_out = CpuGelu(fc);

  auto mlp_proj = CpuGemmF32(gelu_out, mlp_proj_w, S, D, mlp_inner);
  mlp_proj = CpuBiasAdd(mlp_proj, mlp_proj_b, S, D);

  hidden = CpuVectorAdd(hidden, mlp_proj);  // residual 2

  return hidden;
}

// ---- Phase 5: Full Forward Pass reference (C8) ------------------------

inline std::vector<float> CpuGpt2Forward(
    const std::vector<int>& token_ids,
    // Embedding
    const std::vector<float>& wte, const std::vector<float>& wpe,
    // 12 blocks — each as 14 individual vectors
    const std::vector<std::vector<float>>& blk_ln1_w,
    const std::vector<std::vector<float>>& blk_ln1_b,
    const std::vector<std::vector<float>>& blk_c_attn_w,
    const std::vector<std::vector<float>>& blk_c_attn_b,
    const std::vector<std::vector<float>>& blk_c_proj_w,
    const std::vector<std::vector<float>>& blk_c_proj_b,
    const std::vector<std::vector<float>>& blk_ln2_w,
    const std::vector<std::vector<float>>& blk_ln2_b,
    const std::vector<std::vector<float>>& blk_mlp_fc_w,
    const std::vector<std::vector<float>>& blk_mlp_fc_b,
    const std::vector<std::vector<float>>& blk_mlp_proj_w,
    const std::vector<std::vector<float>>& blk_mlp_proj_b,
    // Final LayerNorm
    const std::vector<float>& ln_f_w, const std::vector<float>& ln_f_b,
    // LM head
    const std::vector<float>& wte_T,
    int S, int n_head, int head_dim, int vocab_size, int mlp_inner, float ln_eps) {

  const int D = n_head * head_dim;

  // 1. Embedding
  auto hidden = CpuEmbedding(token_ids, wte, wpe, S, vocab_size, D);

  // 2. 12 transformer blocks
  for (int i = 0; i < 12; ++i) {
    hidden = CpuTransformerBlock(
        hidden,
        blk_ln1_w[i], blk_ln1_b[i],
        blk_c_attn_w[i], blk_c_attn_b[i],
        blk_c_proj_w[i], blk_c_proj_b[i],
        blk_ln2_w[i], blk_ln2_b[i],
        blk_mlp_fc_w[i], blk_mlp_fc_b[i],
        blk_mlp_proj_w[i], blk_mlp_proj_b[i],
        S, n_head, head_dim, mlp_inner, ln_eps);
  }

  // 3. Final LayerNorm
  hidden = CpuLayerNorm(hidden, ln_f_w, ln_f_b, S, D, ln_eps);

  // 4. LM head: logits = hidden @ wte_T
  auto logits = CpuGemmF32(hidden, wte_T, S, vocab_size, D);

  return logits;
}

}  // namespace mgie::test
