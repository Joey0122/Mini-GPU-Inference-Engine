// include/model/workspace.hpp
//
// Pre-allocated workspace for GPT-2 forward pass.
//
// TransformerBlockWorkspace holds all
// the intermediate buffers needed by one block; it is reused across all 12
// layers (the blocks run sequentially, so one set of buffers is enough).
//
// Gpt2Workspace adds the residual stream and logits output on top.
//
// Ownership: the caller creates Gpt2Workspace once and passes it into
// LaunchGpt2Forward. No allocations happen inside the forward pass.
//
// S is the maximum number of rows in a full forward or prefill call.
// A cached decode step reuses row 0 and can attend to the entire context.

#pragma once

#include "cuda/memory.hpp"
#include "model/gpt2_config.hpp"

namespace mgie {

// Scratch buffers for one transformer block. Sized for a single S and reused
// across all 12 layers. The block processes sequentially so no double-buffering
// is needed.
struct TransformerBlockWorkspace {
  // Attention path
  DeviceBuffer<float> ln1_out;      // [S, D]
  DeviceBuffer<float> c_attn_out;   // [S, 3*D]
  DeviceBuffer<float> q;            // [n_head, S, head_dim]
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> scores;       // n_head * max(S * S, n_ctx) elements
  DeviceBuffer<float> weights;      // n_head * max(S * S, n_ctx) elements
  DeviceBuffer<float> attn_out;     // [S, D]
  DeviceBuffer<float> proj_out;     // [S, D] — reused (c_proj → mlp_proj)

  // MLP path
  DeviceBuffer<float> ln2_out;      // [S, D]
  DeviceBuffer<float> fc_out;       // [S, mlp_inner]

  // Allocate all buffers for a given (S, cfg). Move-only semantics let the
  // caller pass this by reference into every block call.
  static TransformerBlockWorkspace Create(int S, const Gpt2Config& cfg);
};

// Top-level workspace for the full forward pass.
struct Gpt2Workspace {
  DeviceBuffer<float> hidden;       // [S, D] — residual stream
  DeviceBuffer<float> logits;       // [S, vocab_size] — output
  TransformerBlockWorkspace blk;    // per-block scratch (shared across layers)

  static Gpt2Workspace Create(int S, const Gpt2Config& cfg);
};

}  // namespace mgie
