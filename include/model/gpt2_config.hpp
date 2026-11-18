// include/model/gpt2_config.hpp
//
// GPT-2 Small (124M) architectural constants.
//
// All Phase 5 code takes a Gpt2Config by const-ref so the constants are in one
// place and easy to audit. DefaultGpt2SmallConfig() is the only configuration
// this project targets (AGENT.md scope-lock: GPT-2 Small, batch=1).
//
// Naming follows the HuggingFace GPT-2 config keys so the export script and C++
// loader can use the same names without translation.

#pragma once

namespace mgie {

struct Gpt2Config {
    int n_layer;      // transformer blocks (depth)
    int n_head;       // attention heads per block
    int n_embd;       // embedding / hidden dimension
    int head_dim;     // n_embd / n_head
    int n_ctx;        // maximum sequence length (context window)
    int vocab_size;   // vocabulary size
    int mlp_inner;    // MLP first-layer width (= 4 * n_embd for GPT-2)
    float ln_eps;     // layer-norm epsilon
};

// GPT-2 Small (124M parameters, as released by OpenAI and available on
// HuggingFace as 'gpt2'). Head count 12, hidden dim 768, 12 layers.
inline Gpt2Config DefaultGpt2SmallConfig() {
    return {
        12,     // n_layer
        12,     // n_head
        768,    // n_embd
        64,     // head_dim
        1024,   // n_ctx
        50257,  // vocab_size
        3072,   // mlp_inner
        1e-5f,  // ln_eps
    };
}

}  // namespace mgie
