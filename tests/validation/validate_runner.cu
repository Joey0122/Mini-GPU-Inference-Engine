// tests/validation/validate_runner.cu
//
// CUDA kernel validation against PyTorch references (Phase 3).
//
// Loads the .npy test vectors written by scripts/generate_reference.py, runs
// the ACTUAL committed kernels (LaunchLayerNorm / LaunchSoftmax / LaunchGelu,
// every version), diffs the GPU output against the PyTorch reference, and
// gates on PROJECT_SPEC's FP32 tolerance (max_abs_error < 1e-4). Emits
// reports/phase3/validation_report.json with the raw max_abs_error per case so
// the real margin (~1e-6, measured) stays visible.
//
// This is the deliverable that satisfies "validate against PyTorch": it proves
// the committed CUDA kernels, not just the references, within the gate.
//
// Run as a GTest (ctest -L validation). The test vectors are resolved relative
// to the source tree via MGIE_TEST_VECTOR_DIR (set by CMake).

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "common/errors.hpp"
#include "common/npy_io.hpp"
#include "common/test_utils.hpp"
#include "cuda/memory.hpp"
#include "kernels/attention.hpp"
#include "kernels/bias_add.hpp"
#include "kernels/embedding.hpp"
#include "kernels/qkv_split.hpp"
#include "model/forward.hpp"
#include "model/gpt2_config.hpp"
#include "model/loader.hpp"
#include "model/transformer_block.hpp"
#include "model/workspace.hpp"
#include "kernels/gelu.hpp"
#include "kernels/layernorm.hpp"
#include "kernels/softmax.hpp"

namespace {

using mgie::DeviceBuffer;
namespace mt = mgie::test;

#ifndef MGIE_TEST_VECTOR_DIR
#define MGIE_TEST_VECTOR_DIR "data/test_vectors"
#endif
#ifndef MGIE_REPORT_DIR
#define MGIE_REPORT_DIR "reports/phase3"
#endif

// Spec gate (docs/project-spec.md): FP32 max_abs_error < 1e-4.
constexpr float kGate = 1e-4f;

bool CudaAvail() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::string VecDir() { return std::string(MGIE_TEST_VECTOR_DIR); }

// One validation outcome row, accumulated into the JSON report.
struct ReportRow {
  std::string kernel, version, shape;
  double max_abs_error = 0.0;
  bool passed = false;
};

// Global report accumulator + writer. JSON is emitted once at process exit so a
// single run of the validation suite produces one validation_report.json.
std::vector<ReportRow>& Rows() {
  static std::vector<ReportRow> rows;
  return rows;
}

void WriteReport() {
  const std::string path = std::string(MGIE_REPORT_DIR) + "/validation_report.json";
  std::ofstream out(path);
  if (!out) return;
  out << "{\n  \"tolerance\": " << kGate << ",\n  \"results\": [\n";
  const auto& rows = Rows();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    out << "    {\"kernel\": \"" << r.kernel << "\", \"version\": \"" << r.version
        << "\", \"shape\": \"" << r.shape << "\", \"max_abs_error\": "
        << r.max_abs_error << ", \"passed\": " << (r.passed ? "true" : "false")
        << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
  }
  out << "  ]\n}\n";
}

void Record(const ReportRow& r) {
  Rows().push_back(r);
  WriteReport();  // rewrite after each case so a crash still leaves partial data
}

class ValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvail()) GTEST_SKIP() << "no CUDA device available";
  }
};

// ---- LayerNorm: validate both Naive and Welford against PyTorch -------------

void ValidateLayerNorm(const std::string& case_name,
                       mgie::LayerNormVersion version, const char* vlabel) {
  const std::string d = VecDir() + "/layernorm/" + case_name + "/";
  const auto x = mt::LoadNpy(d + "x.npy");
  const auto gamma = mt::LoadNpy(d + "gamma.npy");
  const auto beta = mt::LoadNpy(d + "beta.npy");
  const auto ref = mt::LoadNpy(d + "ref.npy");
  const int N = x.shape.at(0), H = x.shape.at(1);

  DeviceBuffer<float> dx(N * H), dout(N * H), dg(H), db(H);
  dx.CopyFromHost(x.data.data(), N * H);
  dg.CopyFromHost(gamma.data.data(), H);
  db.CopyFromHost(beta.data.data(), H);
  mgie::LaunchLayerNorm(dx.data(), dout.data(), dg.data(), db.data(), N, H,
                        1e-5f, version);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(N * H);
  dout.CopyToHost(gpu.data(), N * H);

  const double err = mt::MaxAbsError(gpu, ref.data);
  const bool ok = err < kGate;
  Record({"layernorm", vlabel,
          std::to_string(N) + "x" + std::to_string(H), err, ok});
  EXPECT_LT(err, kGate) << "layernorm " << vlabel << " " << case_name;
}

void ValidateSoftmax(const std::string& case_name,
                     mgie::SoftmaxVersion version, const char* vlabel) {
  const std::string d = VecDir() + "/softmax/" + case_name + "/";
  const auto x = mt::LoadNpy(d + "x.npy");
  const auto ref = mt::LoadNpy(d + "ref.npy");
  const int N = x.shape.at(0), D = x.shape.at(1);

  DeviceBuffer<float> dx(N * D), dout(N * D);
  dx.CopyFromHost(x.data.data(), N * D);
  mgie::LaunchSoftmax(dx.data(), dout.data(), N, D, version);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(N * D);
  dout.CopyToHost(gpu.data(), N * D);

  const double err = mt::MaxAbsError(gpu, ref.data);
  const bool ok = err < kGate;
  Record({"softmax", vlabel,
          std::to_string(N) + "x" + std::to_string(D), err, ok});
  EXPECT_LT(err, kGate) << "softmax " << vlabel << " " << case_name;
}

// ---- GELU: validate NewGELU (tanh) against PyTorch --------------------------

void ValidateGelu(const std::string& case_name) {
  const std::string d = VecDir() + "/gelu/" + case_name + "/";
  const auto x = mt::LoadNpy(d + "x.npy");
  const auto ref = mt::LoadNpy(d + "ref.npy");
  const int n = static_cast<int>(x.numel());

  DeviceBuffer<float> dx(n), dout(n);
  dx.CopyFromHost(x.data.data(), n);
  mgie::LaunchGelu(dx.data(), dout.data(), n);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(n);
  dout.CopyToHost(gpu.data(), n);

  const double err = mt::MaxAbsError(gpu, ref.data);
  const bool ok = err < kGate;
  const int N = x.shape.at(0), D = x.shape.at(1);
  Record({"gelu", "newgelu",
          std::to_string(N) + "x" + std::to_string(D), err, ok});
  EXPECT_LT(err, kGate) << "gelu " << case_name;
}

// ---- Attention: integrated LaunchAttention vs PyTorch SDPA ------------------

void ValidateAttention(const std::string& case_name) {
  const std::string d = VecDir() + "/attention/" + case_name + "/";
  const auto q = mt::LoadNpy(d + "q.npy");   // [n_head, S, head_dim]
  const auto k = mt::LoadNpy(d + "k.npy");
  const auto v = mt::LoadNpy(d + "v.npy");
  const auto ref = mt::LoadNpy(d + "ref.npy");  // [S, n_head*head_dim]
  const int n_head = q.shape.at(0), S = q.shape.at(1), head_dim = q.shape.at(2);

  const int qn = n_head * S * head_dim, sn = n_head * S * S;
  const int on = S * n_head * head_dim;
  DeviceBuffer<float> dq(qn), dk(qn), dv(qn), dscores(sn), dweights(sn), dout(on);
  dq.CopyFromHost(q.data.data(), qn);
  dk.CopyFromHost(k.data.data(), qn);
  dv.CopyFromHost(v.data.data(), qn);
  mgie::LaunchAttention(dq.data(), dk.data(), dv.data(), dout.data(),
                        dscores.data(), dweights.data(), n_head, S, head_dim);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(on);
  dout.CopyToHost(gpu.data(), on);

  const double err = mt::MaxAbsError(gpu, ref.data);
  const bool ok = err < kGate;
  Record({"attention", "integrated",
          "h" + std::to_string(n_head) + "_s" + std::to_string(S), err, ok});
  EXPECT_LT(err, kGate) << "attention " << case_name;
}

// ---- Embedding: validate LaunchEmbedding vs PyTorch F.embedding ----------------

void ValidateEmbedding(const std::string& case_name) {
  const std::string d = VecDir() + "/embedding/" + case_name + "/";
  const auto token_ids_npy = mt::LoadNpy(d + "token_ids.npy");
  const auto wte  = mt::LoadNpy(d + "wte.npy");   // [vocab_size=50257, n_embd=768]
  const auto wpe  = mt::LoadNpy(d + "wpe.npy");   // [n_ctx=1024, n_embd=768]
  const auto ref  = mt::LoadNpy(d + "ref.npy");   // [S, n_embd]
  const int S = static_cast<int>(token_ids_npy.numel());
  const int vocab_size = wte.shape.at(0);
  const int n_embd = wte.shape.at(1);

  // Convert float token_ids to int32 (np.save doesn't preserve dtype for ints
  // written via astype, so we read back as float and cast).
  std::vector<int> token_ids(S);
  for (int i = 0; i < S; ++i)
    token_ids[i] = static_cast<int>(token_ids_npy.data[i]);

  DeviceBuffer<int>   dt(S);
  DeviceBuffer<float> dw(vocab_size * n_embd), dp(1024 * n_embd), dout(S * n_embd);
  dt.CopyFromHost(token_ids.data(), S);
  dw.CopyFromHost(wte.data.data(), vocab_size * n_embd);
  dp.CopyFromHost(wpe.data.data(), 1024 * n_embd);
  mgie::LaunchEmbedding(dt.data(), dw.data(), dp.data(), dout.data(),
                         S, n_embd, vocab_size);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(S * n_embd);
  dout.CopyToHost(gpu.data(), S * n_embd);

  const double err = mt::MaxAbsError(gpu, ref.data);
  const bool ok = err < kGate;
  Record({"embedding", "naive",
          "s" + std::to_string(S) + "_d" + std::to_string(n_embd), err, ok});
  EXPECT_LT(err, kGate) << "embedding " << case_name;
}

// ---- BiasAdd: validate LaunchBiasAdd vs PyTorch x+bias ------------------------

void ValidateBiasAdd(const std::string& case_name) {
  const std::string d = VecDir() + "/bias_add/" + case_name + "/";
  const auto x    = mt::LoadNpy(d + "x.npy");     // [N, D]
  const auto bias = mt::LoadNpy(d + "bias.npy");   // [D]
  const auto ref  = mt::LoadNpy(d + "ref.npy");    // [N, D]
  const int N = x.shape.at(0), D = x.shape.at(1);

  DeviceBuffer<float> dx(N * D), db(D), dout(N * D);
  dx.CopyFromHost(x.data.data(), N * D);
  db.CopyFromHost(bias.data.data(), D);
  mgie::LaunchBiasAdd(dx.data(), db.data(), dout.data(), N, D);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> gpu(N * D);
  dout.CopyToHost(gpu.data(), N * D);

  const double err = mt::MaxAbsError(gpu, ref.data);
  const bool ok = err < kGate;
  Record({"bias_add", "naive",
          "n" + std::to_string(N) + "_d" + std::to_string(D), err, ok});
  EXPECT_LT(err, kGate) << "bias_add " << case_name;
}

// ---- QKV Split: validate LaunchQkvSplit vs PyTorch split→view→permute ----------

void ValidateQkvSplit(const std::string& case_name) {
  const std::string d = VecDir() + "/qkv_split/" + case_name + "/";
  const auto x  = mt::LoadNpy(d + "x.npy");   // [S, 3*n_embd]
  const auto qr = mt::LoadNpy(d + "q.npy");   // [n_head, S, head_dim]
  const auto kr = mt::LoadNpy(d + "k.npy");
  const auto vr = mt::LoadNpy(d + "v.npy");
  const int S = x.shape.at(0), n_embd3 = x.shape.at(1);
  const int n_embd = n_embd3 / 3;
  const int n_head = qr.shape.at(0), head_dim = qr.shape.at(2);
  const int tn = n_head * S * head_dim;

  DeviceBuffer<float> dx(S * 3 * n_embd), dq(tn), dk(tn), dv(tn);
  dx.CopyFromHost(x.data.data(), S * 3 * n_embd);
  mgie::LaunchQkvSplit(dx.data(), dq.data(), dk.data(), dv.data(),
                        S, n_embd, n_head);
  CUDA_CHECK_LAST_SYNC();
  std::vector<float> q(tn), k(tn), v(tn);
  dq.CopyToHost(q.data(), tn);
  dk.CopyToHost(k.data(), tn);
  dv.CopyToHost(v.data(), tn);

  double err = mt::MaxAbsError(q, qr.data);
  Record({"qkv_split", "q_out", "s" + std::to_string(S) + "_h" +
          std::to_string(n_head) + "_hd" + std::to_string(head_dim), err, err < kGate});
  EXPECT_LT(err, kGate) << "qkv_split Q " << case_name;

  err = mt::MaxAbsError(k, kr.data);
  Record({"qkv_split", "k_out", "s" + std::to_string(S), err, err < kGate});
  EXPECT_LT(err, kGate) << "qkv_split K " << case_name;

  err = mt::MaxAbsError(v, vr.data);
  Record({"qkv_split", "v_out", "s" + std::to_string(S), err, err < kGate});
  EXPECT_LT(err, kGate) << "qkv_split V " << case_name;
}

// ---- Transformer Block: validate LaunchTransformerBlock vs PyTorch ---------

void ValidateTransformerBlock(const std::string& case_name) {
  const std::string d = VecDir() + "/transformer_block/" + case_name + "/";
  const auto hidden_in  = mt::LoadNpy(d + "hidden_in.npy");   // [S, D]
  const auto ln1_w      = mt::LoadNpy(d + "ln1_w.npy");      // [D]
  const auto ln1_b      = mt::LoadNpy(d + "ln1_b.npy");      // [D]
  const auto c_attn_w   = mt::LoadNpy(d + "c_attn_w.npy");   // [D, 3*D]
  const auto c_attn_b   = mt::LoadNpy(d + "c_attn_b.npy");   // [3*D]
  const auto c_proj_w   = mt::LoadNpy(d + "c_proj_w.npy");   // [D, D]
  const auto c_proj_b   = mt::LoadNpy(d + "c_proj_b.npy");   // [D]
  const auto ln2_w      = mt::LoadNpy(d + "ln2_w.npy");      // [D]
  const auto ln2_b      = mt::LoadNpy(d + "ln2_b.npy");      // [D]
  const auto mlp_fc_w   = mt::LoadNpy(d + "mlp_fc_w.npy");   // [D, M]
  const auto mlp_fc_b   = mt::LoadNpy(d + "mlp_fc_b.npy");   // [M]
  const auto mlp_proj_w = mt::LoadNpy(d + "mlp_proj_w.npy"); // [M, D]
  const auto mlp_proj_b = mt::LoadNpy(d + "mlp_proj_b.npy"); // [D]
  const auto ref        = mt::LoadNpy(d + "hidden_out.npy");  // [S, D]

  const int S = hidden_in.shape.at(0);
  const int D = hidden_in.shape.at(1);
  const int M = mlp_fc_b.numel();  // mlp_inner

  // Upload weights to device
  DeviceBuffer<float> d_hidden(S * D);
  d_hidden.CopyFromHost(hidden_in.data.data(), S * D);

  mgie::BlockWeights bw;
  bw.ln1_w      = DeviceBuffer<float>(D);        bw.ln1_w.CopyFromHost(ln1_w.data.data(), D);
  bw.ln1_b      = DeviceBuffer<float>(D);        bw.ln1_b.CopyFromHost(ln1_b.data.data(), D);
  bw.c_attn_w   = DeviceBuffer<float>(D*3*D);    bw.c_attn_w.CopyFromHost(c_attn_w.data.data(), D*3*D);
  bw.c_attn_b   = DeviceBuffer<float>(3*D);      bw.c_attn_b.CopyFromHost(c_attn_b.data.data(), 3*D);
  bw.c_proj_w   = DeviceBuffer<float>(D*D);      bw.c_proj_w.CopyFromHost(c_proj_w.data.data(), D*D);
  bw.c_proj_b   = DeviceBuffer<float>(D);        bw.c_proj_b.CopyFromHost(c_proj_b.data.data(), D);
  bw.ln2_w      = DeviceBuffer<float>(D);        bw.ln2_w.CopyFromHost(ln2_w.data.data(), D);
  bw.ln2_b      = DeviceBuffer<float>(D);        bw.ln2_b.CopyFromHost(ln2_b.data.data(), D);
  bw.mlp_fc_w   = DeviceBuffer<float>(D*M);      bw.mlp_fc_w.CopyFromHost(mlp_fc_w.data.data(), D*M);
  bw.mlp_fc_b   = DeviceBuffer<float>(M);        bw.mlp_fc_b.CopyFromHost(mlp_fc_b.data.data(), M);
  bw.mlp_proj_w = DeviceBuffer<float>(M*D);      bw.mlp_proj_w.CopyFromHost(mlp_proj_w.data.data(), M*D);
  bw.mlp_proj_b = DeviceBuffer<float>(D);        bw.mlp_proj_b.CopyFromHost(mlp_proj_b.data.data(), D);

  const auto cfg = mgie::DefaultGpt2SmallConfig();
  auto ws = mgie::TransformerBlockWorkspace::Create(S, cfg);
  mgie::LaunchTransformerBlock(d_hidden.data(), bw, ws, cfg, S);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(S * D);
  d_hidden.CopyToHost(gpu.data(), S * D);

  const double err = mt::MaxAbsError(gpu, ref.data);
  // Block-level tolerance: the block chains 6 GEMMs (K up to 3072), 2 LayerNorms,
  // attention softmax, GELU. Individual-op gate is 1e-4; compound gate is ~5e-3.
  const double kBlockGate = 5e-3;
  const bool ok = err < kBlockGate;
  Record({"transformer_block", "full_block",
          "s" + std::to_string(S), err, ok});
  EXPECT_LT(err, kBlockGate) << "transformer_block " << case_name;
}

// ---- Forward Pass: validate LaunchGpt2Forward vs PyTorch (S=1 only) -----------

void ValidateGpt2Forward(const std::string& case_name) {
  const std::string d = VecDir() + "/forward/" + case_name + "/";
  const auto token_ids_npy = mt::LoadNpy(d + "token_ids.npy");
  const auto wte    = mt::LoadNpy(d + "wte.npy");      // [V, D]
  const auto wpe    = mt::LoadNpy(d + "wpe.npy");      // [1024, D]
  const auto wte_T  = mt::LoadNpy(d + "wte_T.npy");    // [D, V]
  const auto ln_f_w = mt::LoadNpy(d + "ln_f_w.npy");   // [D]
  const auto ln_f_b = mt::LoadNpy(d + "ln_f_b.npy");   // [D]
  const auto ref    = mt::LoadNpy(d + "logits.npy");    // [S, V]

  const int S = token_ids_npy.numel();
  const int D = wte.shape.at(1);
  const int V = wte_T.shape.at(1);

  std::vector<int> token_ids(S);
  for (int i = 0; i < S; ++i)
    token_ids[i] = static_cast<int>(token_ids_npy.data[i]);

  // Upload embedding + lm_head weights
  auto load_buf = [](const std::vector<float>& h) {
    DeviceBuffer<float> buf(h.size());
    buf.CopyFromHost(h.data(), h.size());
    return buf;
  };

  mgie::Gpt2Weights w;
  w.config = mgie::DefaultGpt2SmallConfig();
  w.wte   = load_buf(wte.data);
  w.wpe   = load_buf(wpe.data);
  w.wte_T = load_buf(wte_T.data);
  w.ln_f_w = load_buf(ln_f_w.data);
  w.ln_f_b = load_buf(ln_f_b.data);

  // Upload 12 blocks
  for (int i = 0; i < 12; ++i) {
    const std::string p = "blk" + std::to_string(i) + ".";
    auto& b = w.blocks[i];
    b.ln1_w      = load_buf(mt::LoadNpy(d + p + "ln1_w.npy").data);
    b.ln1_b      = load_buf(mt::LoadNpy(d + p + "ln1_b.npy").data);
    b.c_attn_w   = load_buf(mt::LoadNpy(d + p + "c_attn_w.npy").data);
    b.c_attn_b   = load_buf(mt::LoadNpy(d + p + "c_attn_b.npy").data);
    b.c_proj_w   = load_buf(mt::LoadNpy(d + p + "c_proj_w.npy").data);
    b.c_proj_b   = load_buf(mt::LoadNpy(d + p + "c_proj_b.npy").data);
    b.ln2_w      = load_buf(mt::LoadNpy(d + p + "ln2_w.npy").data);
    b.ln2_b      = load_buf(mt::LoadNpy(d + p + "ln2_b.npy").data);
    b.mlp_fc_w   = load_buf(mt::LoadNpy(d + p + "mlp_fc_w.npy").data);
    b.mlp_fc_b   = load_buf(mt::LoadNpy(d + p + "mlp_fc_b.npy").data);
    b.mlp_proj_w = load_buf(mt::LoadNpy(d + p + "mlp_proj_w.npy").data);
    b.mlp_proj_b = load_buf(mt::LoadNpy(d + p + "mlp_proj_b.npy").data);
  }

  DeviceBuffer<int> d_ids(S);
  d_ids.CopyFromHost(token_ids.data(), S);
  auto workspace = mgie::Gpt2Workspace::Create(S, w.config);

  mgie::LaunchGpt2Forward(d_ids.data(), w, workspace.logits.data(),
                          workspace, w.config, S);
  CUDA_CHECK_LAST_SYNC();

  std::vector<float> gpu(S * V);
  workspace.logits.CopyToHost(gpu.data(), S * V);

  const double err = mt::MaxAbsError(gpu, ref.data);
  // Compound gate: 13 GEMMs through 768-dim K, plus LayerNorms and attention.
  const double kForwardGate = 1e-2;
  const bool ok = err < kForwardGate;
  Record({"gpt2_forward", "full_forward", "s" + std::to_string(S), err, ok});
  EXPECT_LT(err, kForwardGate) << "gpt2_forward " << case_name;
}

TEST_F(ValidationTest, AllKernels) {
  // Single test = single process, so one complete validation_report.json is
  // written (gtest_discover_tests runs each TEST in its own process; splitting
  // by kernel would let the last process overwrite the others' report rows).
  for (const char* c : {"gpt2", "ffn", "stress"}) {
    ValidateLayerNorm(c, mgie::LayerNormVersion::Naive, "naive");
    ValidateLayerNorm(c, mgie::LayerNormVersion::Welford, "welford");
  }
  for (const char* c : {"attn", "stress"}) {
    ValidateSoftmax(c, mgie::SoftmaxVersion::Naive, "naive");
    ValidateSoftmax(c, mgie::SoftmaxVersion::Fused, "fused");
  }
  for (const char* c : {"mlp", "stress"}) ValidateGelu(c);
  for (const char* c : {"gpt2_s128", "gpt2_s512", "stress_mag", "stress_s13"})
    ValidateAttention(c);
  for (const char* c : {"s1", "s64", "s256", "stress_s1k"})
    ValidateEmbedding(c);
  for (const char* c : {"single_row", "gpt2_proj", "c_attn", "mlp_fc", "full_ctx"})
    ValidateBiasAdd(c);
  for (const char* c : {"single_token", "short_seq", "medium_seq", "full_ctx",
                         "nontrivial"})
    ValidateQkvSplit(c);
  for (const char* c : {"single_token", "short_seq"})
    ValidateTransformerBlock(c);
  for (const char* c : {"s1"})
    ValidateGpt2Forward(c);
}

}  // namespace
