// src/model/loader.cpp
// GPT-2 weight loader — manifest + flat binary → DeviceBuffers.

#include "model/loader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "common/errors.hpp"
#include "common/logging.hpp"

namespace mgie {
namespace {

// One parsed manifest entry.
struct ManifestEntry {
    std::string name;
    std::size_t offset;       // byte offset in the .bin file
    std::vector<int> shape;
    std::size_t numel() const {
        std::size_t n = 1;
        for (int d : shape) n *= static_cast<std::size_t>(d);
        return n;
    }
};

// Parse gpt2_manifest.txt. Each non-comment line:
//   name offset_bytes ndim d0 [d1]
std::vector<ManifestEntry> ParseManifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open manifest: " + path);
    std::vector<ManifestEntry> entries;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        ManifestEntry e;
        ss >> e.name >> e.offset;
        int ndim = 0;
        ss >> ndim;
        e.shape.resize(ndim);
        for (int i = 0; i < ndim; ++i) ss >> e.shape[i];
        if (ss.fail()) throw std::runtime_error("malformed manifest line: " + line);
        entries.push_back(std::move(e));
    }
    return entries;
}

// Read `numel` float32 values from `bin` at byte `offset` into `buf`.
void ReadTensor(std::ifstream& bin, std::size_t offset,
                std::size_t numel, std::vector<float>& buf) {
    buf.resize(numel);
    bin.seekg(static_cast<std::streamoff>(offset));
    if (!bin) throw std::runtime_error("seek failed at offset " +
                                       std::to_string(offset));
    bin.read(reinterpret_cast<char*>(buf.data()),
             static_cast<std::streamsize>(numel * sizeof(float)));
    if (static_cast<std::size_t>(bin.gcount()) != numel * sizeof(float)) {
        throw std::runtime_error("truncated read at offset " +
                                 std::to_string(offset));
    }
}

// Allocate a DeviceBuffer, copy host data, return it.
DeviceBuffer<float> Upload(const std::vector<float>& data) {
    DeviceBuffer<float> buf(data.size());
    buf.CopyFromHost(data.data(), data.size());
    return buf;
}

// Validate that `shape` matches `expected`. Throws if not.
void CheckShape(const std::string& name,
                const std::vector<int>& shape,
                const std::vector<int>& expected) {
    if (shape != expected) {
        std::string s, e;
        for (int d : shape)    s += std::to_string(d) + " ";
        for (int d : expected) e += std::to_string(d) + " ";
        throw std::runtime_error(
            "shape mismatch for '" + name + "': got [" + s + "], expected [" + e + "]");
    }
}

}  // namespace

Gpt2Weights LoadGpt2Weights(const std::string& weights_dir, const Gpt2Config& cfg) {
    const std::string manifest_path = weights_dir + "/gpt2_manifest.txt";
    const std::string bin_path      = weights_dir + "/gpt2_weights.bin";

    const auto entries = ParseManifest(manifest_path);
    if (entries.empty()) throw std::runtime_error("empty manifest: " + manifest_path);

    // Build name → entry lookup.
    std::unordered_map<std::string, const ManifestEntry*> lookup;
    for (const auto& e : entries) lookup[e.name] = &e;

    std::ifstream bin(bin_path, std::ios::binary);
    if (!bin) throw std::runtime_error("cannot open weights: " + bin_path);

    Gpt2Weights w;
    w.config = cfg;
    std::vector<float> buf;  // reused scratch; avoids per-tensor heap alloc

    // Helper: find entry by name, validate shape, upload to device.
    auto load = [&](const std::string& name,
                    const std::vector<int>& expected) -> DeviceBuffer<float> {
        auto it = lookup.find(name);
        if (it == lookup.end())
            throw std::runtime_error("tensor not found in manifest: " + name);
        CheckShape(name, it->second->shape, expected);
        ReadTensor(bin, it->second->offset, it->second->numel(), buf);
        return Upload(buf);
    };

    LOG_INFO("Loading GPT-2 weights from " << weights_dir
             << " (" << entries.size() << " tensors)");

    // --- embeddings ---
    w.wte   = load("wte",   {cfg.vocab_size, cfg.n_embd});
    w.wte_T = load("wte_T", {cfg.n_embd,     cfg.vocab_size});
    w.wpe   = load("wpe",   {cfg.n_ctx,       cfg.n_embd});

    // --- transformer blocks ---
    for (int i = 0; i < cfg.n_layer; ++i) {
        auto& blk = w.blocks[static_cast<std::size_t>(i)];
        const std::string p = "h" + std::to_string(i);
        blk.ln1_w      = load(p + ".ln1.weight",     {cfg.n_embd});
        blk.ln1_b      = load(p + ".ln1.bias",       {cfg.n_embd});
        blk.c_attn_w   = load(p + ".c_attn.weight",  {cfg.n_embd, 3 * cfg.n_embd});
        blk.c_attn_b   = load(p + ".c_attn.bias",    {3 * cfg.n_embd});
        blk.c_proj_w   = load(p + ".c_proj.weight",  {cfg.n_embd, cfg.n_embd});
        blk.c_proj_b   = load(p + ".c_proj.bias",    {cfg.n_embd});
        blk.ln2_w      = load(p + ".ln2.weight",     {cfg.n_embd});
        blk.ln2_b      = load(p + ".ln2.bias",       {cfg.n_embd});
        blk.mlp_fc_w   = load(p + ".mlp_fc.weight",  {cfg.n_embd,   cfg.mlp_inner});
        blk.mlp_fc_b   = load(p + ".mlp_fc.bias",    {cfg.mlp_inner});
        blk.mlp_proj_w = load(p + ".mlp_proj.weight",{cfg.mlp_inner, cfg.n_embd});
        blk.mlp_proj_b = load(p + ".mlp_proj.bias",  {cfg.n_embd});
    }

    // --- final layer-norm ---
    w.ln_f_w = load("ln_f.weight", {cfg.n_embd});
    w.ln_f_b = load("ln_f.bias",   {cfg.n_embd});

    LOG_INFO("Weights loaded successfully ("
             << (entries.size()) << " tensors validated).");
    return w;
}

}  // namespace mgie
