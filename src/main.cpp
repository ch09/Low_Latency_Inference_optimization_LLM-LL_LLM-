// =============================================================================
// main.cpp — LL_LLM Baseline Inference Runner
// =============================================================================
// Entry point for LLM inference benchmarking.
// Uses llama.cpp as the inference backend.
//
// Usage:
//   ll_llm --model <path.gguf> --prompt <text|@file> [--threads N]
//   [--max-tokens N]
//
// =============================================================================

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "benchmark.h"
#include "llama.h"
#include "sysinfo.h"


namespace fs = std::filesystem;

// =============================================================================
// CLI Argument Parsing
// =============================================================================
struct CliArgs {
  std::string model_path;
  std::string prompt;
  std::string prompt_type = "custom"; // "short", "long", "reasoning", "custom"
  std::string output_dir = "results";
  int threads = 0; // 0 = auto-detect
  int max_tokens = 128;
  int seed = 42;
  bool show_help = false;
  bool show_sysinfo = false;
};

void print_usage(const char *prog) {
  std::cout << "\n";
  std::cout << "LL_LLM — LLM Inference Benchmark\n";
  std::cout << "Usage: " << prog << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --model <path>       Path to GGUF model file (required)\n";
  std::cout
      << "  --prompt <text>      Prompt text or @filepath to load from file\n";
  std::cout << "  --prompt-type <type> Label: short, long, reasoning, custom "
               "(default: custom)\n";
  std::cout << "  --threads <N>        Number of threads (default: auto)\n";
  std::cout << "  --max-tokens <N>     Max tokens to generate (default: 128)\n";
  std::cout << "  --seed <N>           Random seed (default: 42)\n";
  std::cout << "  --output-dir <dir>   Output directory for results (default: "
               "results)\n";
  std::cout << "  --sysinfo            Print system info and exit\n";
  std::cout << "  --help               Show this help\n";
  std::cout << "\n";
}

CliArgs parse_args(int argc, char **argv) {
  CliArgs args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      args.show_help = true;
    } else if (arg == "--sysinfo") {
      args.show_sysinfo = true;
    } else if (arg == "--model" && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      args.prompt = argv[++i];
    } else if (arg == "--prompt-type" && i + 1 < argc) {
      args.prompt_type = argv[++i];
    } else if (arg == "--threads" && i + 1 < argc) {
      args.threads = std::stoi(argv[++i]);
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      args.max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      args.seed = std::stoi(argv[++i]);
    } else if (arg == "--output-dir" && i + 1 < argc) {
      args.output_dir = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
    }
  }

  return args;
}

// =============================================================================
// Load prompt from file if prefixed with @
// =============================================================================
std::string load_prompt(const std::string &prompt_arg) {
  if (!prompt_arg.empty() && prompt_arg[0] == '@') {
    std::string filepath = prompt_arg.substr(1);
    std::ifstream file(filepath);
    if (!file.is_open()) {
      std::cerr << "ERROR: Could not open prompt file: " << filepath << "\n";
      return "";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
  }
  return prompt_arg;
}

// =============================================================================
// Extract quantization type from model filename
// =============================================================================
std::string guess_quantization(const std::string &model_path) {
  std::string filename = fs::path(model_path).stem().string();
  std::transform(filename.begin(), filename.end(), filename.begin(), ::toupper);

  // Common quantization suffixes
  std::vector<std::string> quant_types = {
      "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
      "Q5_0", "Q5_1",   "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16",    "F32"};

  for (const auto &qt : quant_types) {
    if (filename.find(qt) != std::string::npos) {
      return qt;
    }
  }
  return "UNKNOWN";
}

// =============================================================================
// Main Inference Loop
// =============================================================================
int main(int argc, char **argv) {
  CliArgs args = parse_args(argc, argv);

  if (args.show_help) {
    print_usage(argv[0]);
    return 0;
  }

  // --- System Info ---
  auto sys = ll_llm::collect_system_info();
  ll_llm::print_system_info(sys);

  if (args.show_sysinfo) {
    return 0;
  }

  // --- Validate arguments ---
  if (args.model_path.empty()) {
    std::cerr << "ERROR: --model is required.\n";
    print_usage(argv[0]);
    return 1;
  }

  if (!fs::exists(args.model_path)) {
    std::cerr << "ERROR: Model file not found: " << args.model_path << "\n";
    return 1;
  }

  std::string prompt_text = load_prompt(args.prompt);
  if (prompt_text.empty()) {
    prompt_text = "Hello, how are you?";
    std::cout << "No prompt provided, using default: \"" << prompt_text
              << "\"\n";
  }

  int n_threads = args.threads > 0 ? args.threads : sys.physical_cores;
  std::cout << "Using " << n_threads << " threads.\n\n";

  // --- Prepare result ---
  ll_llm::BenchmarkResult result;
  result.model_name = fs::path(args.model_path).filename().string();
  result.quantization = guess_quantization(args.model_path);
  result.prompt_type = args.prompt_type;
  result.thread_count = n_threads;
  result.ram_before_mb = ll_llm::get_current_rss_mb();

  // =====================================================================
  // STEP 1: Load Model
  // =====================================================================
  std::cout << "Loading model: " << args.model_path << "\n";

  ll_llm::Timer load_timer;
  load_timer.start();

  // Initialize llama backend
  llama_backend_init();

  // Load model
  llama_model_params model_params = llama_model_default_params();
  llama_model *model =
      llama_model_load_from_file(args.model_path.c_str(), model_params);

  if (!model) {
    std::cerr << "ERROR: Failed to load model.\n";
    llama_backend_free();
    return 1;
  }

  load_timer.stop();
  result.model_load_time_ms = load_timer.elapsed_ms();
  std::cout << "Model loaded in " << result.model_load_time_ms << " ms\n";

  // =====================================================================
  // STEP 2: Create Context
  // =====================================================================
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_threads = n_threads;
  ctx_params.n_threads_batch = n_threads;
  ctx_params.seed = static_cast<uint32_t>(args.seed);

  llama_context *ctx = llama_init_from_model(model, ctx_params);
  if (!ctx) {
    std::cerr << "ERROR: Failed to create context.\n";
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  // =====================================================================
  // STEP 3: Tokenize Prompt
  // =====================================================================
  const llama_vocab *vocab = llama_model_get_vocab(model);
  const int max_prompt_tokens = 1024;
  std::vector<llama_token> prompt_tokens(max_prompt_tokens);

  int n_prompt_tokens = llama_tokenize(
      vocab, prompt_text.c_str(), static_cast<int32_t>(prompt_text.length()),
      prompt_tokens.data(), max_prompt_tokens,
      true, // add_special (BOS)
      true  // parse_special
  );

  if (n_prompt_tokens < 0) {
    std::cerr << "ERROR: Tokenization failed.\n";
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  prompt_tokens.resize(n_prompt_tokens);
  std::cout << "Prompt tokens: " << n_prompt_tokens << "\n";

  // =====================================================================
  // STEP 4: Prefill (Process prompt)
  // =====================================================================
  ll_llm::Timer first_token_timer;
  first_token_timer.start();

  llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
  for (int i = 0; i < n_prompt_tokens; ++i) {
    llama_batch_add(batch, prompt_tokens[i], i, {0},
                    (i == n_prompt_tokens - 1));
  }

  if (llama_decode(ctx, batch) != 0) {
    std::cerr << "ERROR: Failed to decode prompt.\n";
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  // =====================================================================
  // STEP 5: Generate Tokens (Decode phase)
  // =====================================================================
  ll_llm::Timer gen_timer;
  gen_timer.start();

  std::string generated_text;
  int tokens_generated = 0;
  llama_token new_token_id;

  // Greedy sampling (temperature = 0)
  for (int i = 0; i < args.max_tokens; ++i) {
    float *logits = llama_get_logits_ith(ctx, -1);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Greedy: find argmax
    new_token_id = 0;
    float max_logit = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
      if (logits[v] > max_logit) {
        max_logit = logits[v];
        new_token_id = v;
      }
    }

    // Record first token time
    if (i == 0) {
      first_token_timer.stop();
      result.first_token_ms = first_token_timer.elapsed_ms();
    }

    // Check for EOS
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }

    // Convert token to text
    char buf[256] = {};
    int n =
        llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (n > 0) {
      std::string piece(buf, n);
      generated_text += piece;
      std::cout << piece << std::flush;
    }

    ++tokens_generated;

    // Prepare next batch
    llama_batch_clear(batch);
    llama_batch_add(batch, new_token_id, n_prompt_tokens + i, {0}, true);

    if (llama_decode(ctx, batch) != 0) {
      std::cerr << "\nERROR: Decode failed at token " << i << "\n";
      break;
    }
  }

  gen_timer.stop();
  std::cout << "\n";

  // =====================================================================
  // STEP 6: Record Results
  // =====================================================================
  result.total_generation_ms = gen_timer.elapsed_ms();
  result.tokens_generated = tokens_generated;
  result.tokens_per_second =
      (tokens_generated > 0)
          ? (tokens_generated / (result.total_generation_ms / 1000.0))
          : 0.0;
  result.peak_ram_mb = ll_llm::get_peak_rss_mb();
  result.ram_after_mb = ll_llm::get_current_rss_mb();
  result.generated_text = generated_text;

  ll_llm::print_result(result);

  // --- Save JSON ---
  fs::create_directories(args.output_dir);
  std::string result_file =
      args.output_dir + "/" + ll_llm::make_result_filename("baseline");
  ll_llm::save_result_json(result, result_file);

  // =====================================================================
  // Cleanup
  // =====================================================================
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();

  std::cout << "Done.\n";
  return 0;
}
