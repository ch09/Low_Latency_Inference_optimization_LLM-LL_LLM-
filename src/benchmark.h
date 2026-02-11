#pragma once
// =============================================================================
// benchmark.h — Metrics Collection for LLM Inference Benchmarking
// =============================================================================
// Provides timing, throughput, and memory measurement utilities.

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "sysinfo.h"

namespace ll_llm {

// -------------------------------------------------------------------------
// High-resolution timer
// -------------------------------------------------------------------------
class Timer {
public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  void stop() { end_ = std::chrono::high_resolution_clock::now(); }

  double elapsed_ms() const {
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
    return static_cast<double>(duration.count()) / 1000.0;
  }

  double elapsed_s() const { return elapsed_ms() / 1000.0; }

private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point end_;
};

// -------------------------------------------------------------------------
// Benchmark result for a single run
// -------------------------------------------------------------------------
struct BenchmarkResult {
  std::string model_name;
  std::string prompt_type;  // "short", "long", "reasoning"
  std::string quantization; // "Q4_K_M", "Q8_0", etc.
  int thread_count = 0;

  double model_load_time_ms = 0.0;
  double first_token_ms = 0.0;
  double total_generation_ms = 0.0;
  int tokens_generated = 0;
  double tokens_per_second = 0.0;
  double peak_ram_mb = 0.0;
  double ram_before_mb = 0.0;
  double ram_after_mb = 0.0;

  std::string generated_text;
};

// -------------------------------------------------------------------------
// Print benchmark result
// -------------------------------------------------------------------------
inline void print_result(const BenchmarkResult &r) {
  std::cout << "\n";
  std::cout << "=== Benchmark Results ===\n";
  std::cout << "  Model:           " << r.model_name << "\n";
  std::cout << "  Quantization:    " << r.quantization << "\n";
  std::cout << "  Prompt Type:     " << r.prompt_type << "\n";
  std::cout << "  Threads:         " << r.thread_count << "\n";
  std::cout << "  --------------------------\n";
  std::cout << "  Model Load:      " << std::fixed << std::setprecision(1)
            << r.model_load_time_ms << " ms\n";
  std::cout << "  First Token:     " << std::fixed << std::setprecision(1)
            << r.first_token_ms << " ms\n";
  std::cout << "  Tokens Generated:" << r.tokens_generated << "\n";
  std::cout << "  Tokens/sec:      " << std::fixed << std::setprecision(2)
            << r.tokens_per_second << "\n";
  std::cout << "  Peak RAM:        " << std::fixed << std::setprecision(1)
            << r.peak_ram_mb << " MB\n";
  std::cout << "  RAM (before):    " << std::fixed << std::setprecision(1)
            << r.ram_before_mb << " MB\n";
  std::cout << "  RAM (after):     " << std::fixed << std::setprecision(1)
            << r.ram_after_mb << " MB\n";
  std::cout << "=========================\n";
  std::cout << "\n";
}

// -------------------------------------------------------------------------
// Save result as JSON
// -------------------------------------------------------------------------
inline bool save_result_json(const BenchmarkResult &r,
                             const std::string &filepath) {
  std::ofstream out(filepath);
  if (!out.is_open()) {
    std::cerr << "ERROR: Could not open " << filepath << " for writing.\n";
    return false;
  }

  out << "{\n";
  out << "  \"model\": \"" << r.model_name << "\",\n";
  out << "  \"quantization\": \"" << r.quantization << "\",\n";
  out << "  \"prompt_type\": \"" << r.prompt_type << "\",\n";
  out << "  \"threads\": " << r.thread_count << ",\n";
  out << "  \"model_load_ms\": " << std::fixed << std::setprecision(2)
      << r.model_load_time_ms << ",\n";
  out << "  \"first_token_ms\": " << std::fixed << std::setprecision(2)
      << r.first_token_ms << ",\n";
  out << "  \"tokens_generated\": " << r.tokens_generated << ",\n";
  out << "  \"tokens_per_second\": " << std::fixed << std::setprecision(2)
      << r.tokens_per_second << ",\n";
  out << "  \"peak_ram_mb\": " << std::fixed << std::setprecision(1)
      << r.peak_ram_mb << ",\n";
  out << "  \"ram_before_mb\": " << std::fixed << std::setprecision(1)
      << r.ram_before_mb << ",\n";
  out << "  \"ram_after_mb\": " << std::fixed << std::setprecision(1)
      << r.ram_after_mb << "\n";
  out << "}\n";

  out.close();
  std::cout << "Results saved to: " << filepath << "\n";
  return true;
}

// -------------------------------------------------------------------------
// Generate a timestamped filename
// -------------------------------------------------------------------------
inline std::string make_result_filename(const std::string &prefix) {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf = {};
#ifdef _WIN32
  localtime_s(&tm_buf, &time);
#else
  localtime_r(&time, &tm_buf);
#endif
  std::ostringstream oss;
  oss << prefix << "_" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".json";
  return oss.str();
}

} // namespace ll_llm
