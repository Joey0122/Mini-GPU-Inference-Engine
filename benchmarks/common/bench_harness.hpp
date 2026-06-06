// benchmarks/common/bench_harness.hpp
//
// Project benchmark harness — the columns ARCHITECTURE.md requires.
//
// ARCHITECTURE.md specifies every benchmark must emit: kernel name, input
// shape, execution time, bandwidth, GFLOPS, occupancy, exportable to CSV.
// Google Benchmark (per TECH_STACK) drives the iteration loop and statistics,
// but it has no concept of GPU bandwidth/GFLOPS/occupancy — so this harness
// computes those from CUDA-event timings and writes the canonical CSV row.
//
// Division of labor:
//   * Google Benchmark    -> registration, repetition, CPU-side stats/CLI.
//   * common/timer.hpp     -> accurate per-iteration GPU time (CUDA events).
//   * this header          -> derive bandwidth/GFLOPS, format the CSV row.
//
// Occupancy is reported as "n/a" from this harness (it requires Nsight Compute
// or the occupancy API per kernel); the /profile workflow fills it in. This
// keeps the runtime harness honest rather than fabricating a number.
//
// Future role: every Phase 1-6 kernel benchmark links this and appends to a
// shared CSV under benchmarks/<kernel>/, which scripts/benchmark.py plots.

#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

#include "cuda/device_info.hpp"

namespace mgie::bench {

// One benchmarked configuration's results, in the ARCHITECTURE column order.
struct BenchResult {
  std::string kernel_name;
  std::string input_shape;     // e.g. "n=1048576" or "1024x1024"
  double time_ms = 0.0;        // mean per-iteration time
  double time_stddev_ms = 0.0;
  double bandwidth_gbs = 0.0;  // achieved GB/s
  double gflops = 0.0;         // achieved GFLOP/s
  std::string occupancy = "n/a";  // filled by Nsight (see /profile)
};

// Achieved bandwidth in GB/s given total bytes moved and time in ms.
inline double BandwidthGBs(std::size_t bytes, double time_ms) {
  if (time_ms <= 0.0) return 0.0;
  return static_cast<double>(bytes) / (time_ms * 1.0e-3) / 1.0e9;
}

// Achieved GFLOP/s given total floating-point ops and time in ms.
inline double GFlops(std::size_t flops, double time_ms) {
  if (time_ms <= 0.0) return 0.0;
  return static_cast<double>(flops) / (time_ms * 1.0e-3) / 1.0e9;
}

// CSV header matching the ARCHITECTURE-required columns plus reproducibility
// metadata (Rule 6: GPU, CUDA/driver implied by build, clocks recorded
// out-of-band by scripts/benchmark.py).
inline std::string CsvHeader() {
  return "kernel,input_shape,time_ms,time_stddev_ms,bandwidth_gbs,gflops,"
         "occupancy,gpu,compute_capability";
}

// Format one result as a CSV row, stamped with device identity.
inline std::string CsvRow(const BenchResult& r, const DeviceInfo& dev) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << r.kernel_name << ',' << r.input_shape << ',' << r.time_ms << ','
      << r.time_stddev_ms << ',' << r.bandwidth_gbs << ',' << r.gflops << ','
      << r.occupancy << ',' << '"' << dev.name << '"' << ','
      << dev.compute_major << '.' << dev.compute_minor;
  return oss.str();
}

// Append a result to a CSV file, writing the header if the file is new/empty.
// Thread-safe across benchmark callbacks within one process.
inline void AppendCsv(const std::string& path, const BenchResult& r,
                      const DeviceInfo& dev) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);

  bool needs_header = true;
  {
    std::ifstream probe(path);
    needs_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream out(path, std::ios::app);
  if (needs_header) out << CsvHeader() << '\n';
  out << CsvRow(r, dev) << '\n';
}

}  // namespace mgie::bench
