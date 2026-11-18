// tests/common/npy_io.hpp
//
// Minimal NumPy .npy reader for the validation pipeline.
//
// The PyTorch validation pipeline (scripts/generate_reference.py) writes kernel
// inputs + references as .npy files; the CUDA validation runner
// (tests/validation/validate_runner.cu) loads them, runs the actual committed
// kernel, and diffs against the reference within the spec tolerance.
//
// Deliberately minimal (AGENT.md: justify new deps, prefer simple designs):
//   * .npy format v1.0 only (the version numpy.save writes by default)
//   * dtype '<f4' (little-endian float32) only — the pipeline saves float32
//   * C-contiguous, non-fortran-order only
// Anything else is rejected loudly with a std::runtime_error. No third-party
// dependency is introduced.

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mgie::test {

// One loaded .npy array: flat row-major float data plus its shape.
struct NpyArray {
  std::vector<float> data;
  std::vector<int> shape;

  std::size_t numel() const {
    std::size_t n = 1;
    for (int d : shape) n *= static_cast<std::size_t>(d);
    return n;
  }
};

namespace detail {

// Extract the value of a Python-dict-literal key like 'descr': '<f4' or
// 'shape': (256, 768,) from the .npy header text. Returns the raw substring
// after the key up to the next comma/brace (caller interprets it).
inline std::string HeaderField(const std::string& header, const std::string& key) {
  const std::string needle = "'" + key + "':";
  const std::size_t k = header.find(needle);
  if (k == std::string::npos) {
    throw std::runtime_error("npy header missing key: " + key);
  }
  return header.substr(k + needle.size());
}

}  // namespace detail

// Load an uncompressed .npy v1.0 float32 C-order array from `path`.
// Throws std::runtime_error on any unsupported format.
inline NpyArray LoadNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open .npy: " + path);

  char magic[6];
  f.read(magic, 6);
  if (f.gcount() != 6 || magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
    throw std::runtime_error("not a .npy file: " + path);
  }

  std::uint8_t major = 0, minor = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);
  if (major != 1) {
    throw std::runtime_error("only .npy v1.x supported: " + path);
  }

  std::uint16_t header_len = 0;  // little-endian
  f.read(reinterpret_cast<char*>(&header_len), 2);
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  // dtype must be little-endian float32 ('<f4' or '|f4'/'=f4' on LE hosts).
  const std::string descr = detail::HeaderField(header, "descr");
  if (descr.find("f4") == std::string::npos ||
      descr.find(">f4") != std::string::npos) {
    throw std::runtime_error("npy dtype must be little-endian float32: " + path);
  }

  // Reject Fortran (column-major) order.
  const std::string fortran = detail::HeaderField(header, "fortran_order");
  if (fortran.find("True") != std::string::npos) {
    throw std::runtime_error("npy fortran_order unsupported: " + path);
  }

  // Parse shape tuple, e.g. "(256, 768,)".
  const std::string shape_field = detail::HeaderField(header, "shape");
  const std::size_t lp = shape_field.find('(');
  const std::size_t rp = shape_field.find(')', lp);
  if (lp == std::string::npos || rp == std::string::npos) {
    throw std::runtime_error("npy shape unparseable: " + path);
  }
  NpyArray arr;
  std::string dims = shape_field.substr(lp + 1, rp - lp - 1);
  std::string num;
  for (char c : dims) {
    if (c == ',' || c == ' ') {
      if (!num.empty()) { arr.shape.push_back(std::stoi(num)); num.clear(); }
    } else {
      num += c;
    }
  }
  if (!num.empty()) arr.shape.push_back(std::stoi(num));

  arr.data.resize(arr.numel());
  f.read(reinterpret_cast<char*>(arr.data.data()),
         static_cast<std::streamsize>(arr.numel() * sizeof(float)));
  if (static_cast<std::size_t>(f.gcount()) != arr.numel() * sizeof(float)) {
    throw std::runtime_error("npy truncated payload: " + path);
  }
  return arr;
}

}  // namespace mgie::test
