// src/model/workspace.cu
// C8: workspace allocation factories.

#include "model/workspace.hpp"

#include <algorithm>

namespace mgie {

TransformerBlockWorkspace TransformerBlockWorkspace::Create(
    int S, const Gpt2Config& cfg) {
  const int D = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim;
  const int M = cfg.mlp_inner;

  TransformerBlockWorkspace ws;
  ws.ln1_out    = DeviceBuffer<float>(static_cast<std::size_t>(S) * D);
  ws.c_attn_out = DeviceBuffer<float>(static_cast<std::size_t>(S) * 3 * D);
  ws.q          = DeviceBuffer<float>(static_cast<std::size_t>(H) * S * d);
  ws.k          = DeviceBuffer<float>(static_cast<std::size_t>(H) * S * d);
  ws.v          = DeviceBuffer<float>(static_cast<std::size_t>(H) * S * d);
  const auto attention_width = S > 0
      ? std::max(static_cast<std::size_t>(S) * S, static_cast<std::size_t>(cfg.n_ctx))
      : 0;
  ws.scores     = DeviceBuffer<float>(H * attention_width);
  ws.weights    = DeviceBuffer<float>(H * attention_width);
  ws.attn_out   = DeviceBuffer<float>(static_cast<std::size_t>(S) * D);
  ws.proj_out   = DeviceBuffer<float>(static_cast<std::size_t>(S) * D);
  ws.ln2_out    = DeviceBuffer<float>(static_cast<std::size_t>(S) * D);
  ws.fc_out     = DeviceBuffer<float>(static_cast<std::size_t>(S) * M);
  return ws;
}

Gpt2Workspace Gpt2Workspace::Create(int S, const Gpt2Config& cfg) {
  Gpt2Workspace ws;
  ws.hidden = DeviceBuffer<float>(static_cast<std::size_t>(S) * cfg.n_embd);
  ws.logits = DeviceBuffer<float>(static_cast<std::size_t>(S) * cfg.vocab_size);
  ws.blk    = TransformerBlockWorkspace::Create(S, cfg);
  return ws;
}

}  // namespace mgie
