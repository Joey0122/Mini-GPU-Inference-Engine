// src/kernels/attention.cu
// Phase 4: causal multi-head scaled-dot-product attention (batch=1).
//
// Dedicated batched kernels; all heads handled in one launch via blockIdx.z.
// Naive-but-correct first (AGENT.md kernel process): one block per (head, query
// row); fp32 accumulation. Optimization versions (fused, shared-mem tiling,
// GEMM-backed AV) are future work benchmarked against this baseline.

#include "kernels/attention.hpp"

#include "common/errors.hpp"
#include "kernels/softmax.hpp"

#include <float.h>

namespace mgie {
namespace {

// Masked-out score sentinel: large negative but finite, so softmax's
// exp(score - max) underflows to 0 without producing NaN/Inf even if an entire
// tile were masked. With causal masking row i always has >=1 valid key (j=i).
constexpr float kNegInf = -1e30f;

// ---- Step 1: scores = scale * (Q Kᵀ), causal-masked ---------------------------
// Grid:  (S, n_head)         blockIdx.x = query row i, blockIdx.y = head h
// Block: head_dim threads    each thread loads one q[i,:] component into smem,
//                            then the block cooperatively dots against each k[j].
// scores[h,i,j] for j in [0,S): valid j<=i get scale*dot; j>i get kNegInf.
__global__ void AttentionScoresKernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       float* __restrict__ scores,
                                       int S, int head_dim, float scale) {
  extern __shared__ float q_row[];  // [head_dim] — this query row
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int t = threadIdx.x;

  const float* qh = q + (static_cast<long long>(h) * S + i) * head_dim;
  const float* kh = k + static_cast<long long>(h) * S * head_dim;
  float* srow = scores + (static_cast<long long>(h) * S + i) * S;

  // Stage q[i,:] into shared memory (head_dim threads, one element each).
  for (int d = t; d < head_dim; d += blockDim.x) q_row[d] = qh[d];
  __syncthreads();

  // Each thread computes scores for a strided set of keys j.
  for (int j = t; j < S; j += blockDim.x) {
    if (j > i) {                       // causal: future keys masked
      srow[j] = kNegInf;
      continue;
    }
    const float* kj = kh + static_cast<long long>(j) * head_dim;
    float dot = 0.f;
    for (int d = 0; d < head_dim; ++d) dot += q_row[d] * kj[d];
    srow[j] = dot * scale;
  }
}

}  // namespace

void LaunchAttentionScores(const float* q, const float* k, float* scores,
                           int n_head, int S, int head_dim,
                           cudaStream_t stream) {
  if (n_head == 0 || S == 0 || head_dim == 0) return;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  // block = head_dim threads (>=32 for GPT-2's 64); smem holds the query row.
  const int block = head_dim < 32 ? 32 : head_dim;
  const dim3 grid(S, n_head);
  const int smem = head_dim * sizeof(float);
  AttentionScoresKernel<<<grid, block, smem, stream>>>(q, k, scores, S, head_dim,
                                                       scale);
}

// ---- Step 3: out = weights · V, heads merged into [S, n_head*head_dim] --------
// Grid:  (S, n_head)         blockIdx.x = query row i, blockIdx.y = head h
// Block: head_dim threads    thread d accumulates out[h,i,d] = sum_j w[i,j]*v[j,d].
// Only j<=i contribute (weights are 0 above the diagonal after softmax of the
// masked scores), but we sum the full row — masked weights are exactly 0, so
// the result is identical and the loop stays branch-free.
namespace {

__global__ void AttentionAVKernel(const float* __restrict__ weights,
                                   const float* __restrict__ v,
                                   float* __restrict__ out,
                                   int S, int head_dim, int E) {
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  if (d >= head_dim) return;

  const float* wrow = weights + (static_cast<long long>(h) * S + i) * S;
  const float* vh   = v + static_cast<long long>(h) * S * head_dim;

  float acc = 0.f;
  for (int j = 0; j <= i; ++j) acc += wrow[j] * vh[static_cast<long long>(j) * head_dim + d];

  // Merge: head h occupies columns [h*head_dim, (h+1)*head_dim) of row i.
  out[static_cast<long long>(i) * E + h * head_dim + d] = acc;
}

}  // namespace

void LaunchAttentionAV(const float* weights, const float* v, float* out,
                       int n_head, int S, int head_dim, cudaStream_t stream) {
  if (n_head == 0 || S == 0 || head_dim == 0) return;
  const int block = head_dim < 32 ? 32 : head_dim;
  const dim3 grid(S, n_head);
  const int E = n_head * head_dim;
  AttentionAVKernel<<<grid, block, 0, stream>>>(weights, v, out, S, head_dim, E);
}

// ---- Step 4: integrated attention --------------------------------------------
// Orchestrates Scores -> Softmax (reused Phase 3 kernel) -> AV. The softmax runs
// over [n_head*S, S] rows; masked entries (kNegInf) become exp(-large)=0, so
// each row normalizes over exactly its valid (causal) keys.
void LaunchAttention(const float* q, const float* k, const float* v, float* out,
                     float* scores, float* weights,
                     int n_head, int S, int head_dim, cudaStream_t stream) {
  if (n_head == 0 || S == 0 || head_dim == 0) return;
  LaunchAttentionScores(q, k, scores, n_head, S, head_dim, stream);
  LaunchSoftmax(scores, weights, n_head * S, S, SoftmaxVersion::Fused, stream);
  LaunchAttentionAV(weights, v, out, n_head, S, head_dim, stream);
}

	// =========================================================================
	// Phase 6a: KV-cache attention kernels (single-step decode)
	// =========================================================================
	//
	// These kernels compute attention for a SINGLE query position against
	// all previously cached keys/values.  No causal mask is needed — all
	// cached positions [0..num_cached) are valid by construction.

	namespace {

	// Scaled Q·K^T for a single query against num_cached keys.
	// Grid: (1, n_head).  blockIdx.y = head.
	// q    : [n_head, 1, head_dim]     — implicit S=1 (head stride = head_dim)
	// k    : [n_head, n_ctx, head_dim] — stride = n_ctx positions per head
	// scores: [n_head, num_cached]
	__global__ void AttentionScoresKernelCached(const float* __restrict__ q,
	                                              const float* __restrict__ k,
	                                              float* __restrict__ scores,
	                                              int stride, int num_cached,
	                                              int head_dim, float scale) {
	  extern __shared__ float q_row[];
	  const int h = blockIdx.y;
	  const int t = threadIdx.x;

	  // q: [n_head, 1, head_dim] — single query row per head.
	  const float* qh = q + static_cast<long long>(h) * head_dim;
	  // k: [n_head, stride, head_dim].
	  const float* kh = k + static_cast<long long>(h) * stride * head_dim;
	  // scores: [n_head, num_cached].
	  float* srow = scores + static_cast<long long>(h) * num_cached;

	  for (int d = t; d < head_dim; d += blockDim.x) q_row[d] = qh[d];
	  __syncthreads();

	  // Dot against num_cached keys — NO causal mask.
	  for (int j = t; j < num_cached; j += blockDim.x) {
	    const float* kj = kh + static_cast<long long>(j) * head_dim;
	    float dot = 0.f;
	    for (int d = 0; d < head_dim; ++d) dot += q_row[d] * kj[d];
	    srow[j] = dot * scale;
	  }
	}

	// Unmasked AV: out[h, 0, d] = sum_j weights[h, j] * v_cache[h, j, d]
	// Grid: (1, n_head).  blockIdx.y = head.
	__global__ void AttentionAVKernelCached(const float* __restrict__ weights,
	                                         const float* __restrict__ v,
	                                         float* __restrict__ out,
	                                         int stride, int num_cached,
	                                         int head_dim, int E) {
	  const int h = blockIdx.y;
	  const int d = threadIdx.x;
	  if (d >= head_dim) return;

	  // weights: [n_head, num_cached].
	  const float* wrow = weights + static_cast<long long>(h) * num_cached;
	  // v: [n_head, stride, head_dim].
	  const float* vh = v + static_cast<long long>(h) * stride * head_dim;

	  float acc = 0.f;
	  for (int j = 0; j < num_cached; ++j)
	    acc += wrow[j] * vh[static_cast<long long>(j) * head_dim + d];

	  out[static_cast<long long>(h) * head_dim + d] = acc;
	}

	}  // namespace

	void LaunchAttentionCached(const float* q_new,
	                           const float* k_cache,
	                           const float* v_cache,
	                           float* out,
	                           float* scores,
	                           float* weights,
	                           int n_head, int n_ctx, int head_dim,
	                           int num_cached,
	                           cudaStream_t stream) {
	  if (n_head == 0 || num_cached == 0 || head_dim == 0) return;

	  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
	  const int block = head_dim < 32 ? 32 : head_dim;
	  const dim3 grid(1, n_head);
	  const int smem = head_dim * sizeof(float);

	  // 1. Scores: Q_new × K_cache^T, unmasked
	  AttentionScoresKernelCached<<<grid, block, smem, stream>>>(
	      q_new, k_cache, scores, n_ctx, num_cached, head_dim, scale);

	  // 2. Softmax: row-wise over [n_head, num_cached] (reuses Phase 3 kernel)
	  LaunchSoftmax(scores, weights, n_head, num_cached, SoftmaxVersion::Fused, stream);

	  // 3. AV: weights × V_cache
	  const int E = n_head * head_dim;
	  AttentionAVKernelCached<<<grid, block, 0, stream>>>(
	      weights, v_cache, out, n_ctx, num_cached, head_dim, E);
	}

	// ---- Phase 6a profiling wrappers ---------------------------------------
	// Thin wrappers that expose individual cached-attention sub-kernels for
	// fine-grained wall-clock profiling.  Grid/block config matches internal
	// calls in LaunchAttentionCached exactly.

	void LaunchAttentionScoresCached(const float* q_new,
	                                  const float* k_cache,
	                                  float* scores,
	                                  int n_head, int n_ctx, int head_dim,
	                                  int num_cached,
	                                  cudaStream_t stream) {
	  if (n_head == 0 || num_cached == 0 || head_dim == 0) return;

	  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
	  const int block = head_dim < 32 ? 32 : head_dim;
	  const dim3 grid(1, n_head);
	  const int smem = head_dim * sizeof(float);

	  AttentionScoresKernelCached<<<grid, block, smem, stream>>>(
	      q_new, k_cache, scores, n_ctx, num_cached, head_dim, scale);
	}

	void LaunchAttentionAVCached(const float* weights,
	                              const float* v_cache,
	                              float* out,
	                              int n_head, int n_ctx, int head_dim,
	                              int num_cached,
	                              cudaStream_t stream) {
	  if (n_head == 0 || num_cached == 0 || head_dim == 0) return;

	  const int block = head_dim < 32 ? 32 : head_dim;
	  const dim3 grid(1, n_head);
	  const int E = n_head * head_dim;

	  AttentionAVKernelCached<<<grid, block, 0, stream>>>(
	      weights, v_cache, out, n_ctx, num_cached, head_dim, E);
	}

	}  // namespace mgie
