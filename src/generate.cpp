#include <algorithm>
#include <charconv>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "common/logging.hpp"
#include "cuda/memory.hpp"
#include "model/decode.hpp"

namespace {

int ParseInt(std::string_view text, const char* label) {
  int value = 0;
  const auto end = text.data() + text.size();
  const auto result = std::from_chars(text.data(), end, value);
  if (result.ec != std::errc{} || result.ptr != end)
    throw std::invalid_argument(std::string("invalid ") + label);
  return value;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: mgie_generate WEIGHTS_DIR MAX_NEW_TOKENS TOKEN_ID...\n";
    return 2;
  }

  try {
    const auto cfg = mgie::DefaultGpt2SmallConfig();
    const int prompt_len = argc - 3;
    const int requested = ParseInt(argv[2], "generation length");
    if (requested <= 0 || prompt_len >= cfg.n_ctx)
      throw std::invalid_argument("need a positive generation length and a prompt shorter than 1024 tokens");

    std::vector<int> prompt;
    prompt.reserve(prompt_len);
    for (int i = 3; i < argc; ++i) {
      const int token = ParseInt(argv[i], "token ID");
      if (token < 0 || token >= cfg.vocab_size)
        throw std::invalid_argument("token ID outside the GPT-2 vocabulary");
      prompt.push_back(token);
    }

    // Keep stdout available for the token protocol.
    mgie::Logger::SetLevel(mgie::LogLevel::kError);
    const auto weights = mgie::LoadGpt2Weights(argv[1], cfg);
    auto workspace = mgie::Gpt2Workspace::Create(prompt_len, cfg);
    auto cache = mgie::KVCache::Create(cfg);

    mgie::DeviceBuffer<int> d_prompt(prompt_len);
    mgie::DeviceBuffer<int> d_output(prompt_len + std::min(requested, cfg.n_ctx - prompt_len));
    d_prompt.CopyFromHost(prompt.data(), prompt.size());
    const int generated = mgie::LaunchGreedyDecode(d_prompt.data(), d_output.data(),
                                                  weights, workspace, cache, cfg,
                                                  prompt_len, requested);

    std::vector<int> output(prompt_len + generated);
    d_output.CopyToHost(output.data(), output.size());
    for (int token : output) std::cout << token << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mgie_generate: " << error.what() << '\n';
    return 1;
  }
}
