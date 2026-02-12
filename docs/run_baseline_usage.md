# run_baseline.py — Usage Guide

## What It Does

`run_baseline.py` is the **automated benchmark runner** for the LL_LLM project. It establishes a **performance baseline** by:

1. **Building** the C++ inference harness (CMake + MSVC Release build)
2. **Validating** that the specified GGUF model file exists
3. **Running** the inference binary against **3 prompt types** — `short`, `long`, and `reasoning`
4. **Collecting** results as JSON files in the output directory

This baseline is the reference point before any optimization work begins. Every future change is measured against these numbers.

---

## Prerequisites

- **Python 3.10+**
- **CMake 3.20+** (must be on `PATH`)
- **Visual Studio 2022** (or MSVC Build Tools)
- A **GGUF model** downloaded into the `models/` directory

---

## Usage

Run from the **project root**:

```bash
python scripts/run_baseline.py --model-path models/your_model.gguf
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-path` | ✅ Yes | — | Relative path to the GGUF model file |
| `--threads` | No | `0` (auto) | Number of CPU threads for inference |
| `--max-tokens` | No | `128` | Maximum tokens to generate per prompt |
| `--output-dir` | No | `results` | Directory where JSON results are saved |
| `--skip-build` | No | `false` | Skip the CMake build step |

### Examples

```bash
# Basic run with TinyLlama
python scripts/run_baseline.py --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Use 4 threads and generate up to 256 tokens
python scripts/run_baseline.py --model-path models/mistral-7b-instruct-v0.2.Q4_K_M.gguf --threads 4 --max-tokens 256

# Skip build (if already compiled)
python scripts/run_baseline.py --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --skip-build

# Save results to a custom directory
python scripts/run_baseline.py --model-path models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --output-dir results/baseline_v1
```

---

## What the Script Runs (Step by Step)

### Step 1 — Build

Runs CMake to configure and compile the project in **Release** mode:

```
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Skipped if `--skip-build` is passed.

### Step 2 — Model Check

Verifies the model file exists at the given path. If not found, prints a download recommendation:

> **Recommended**: [TinyLlama 1.1B Q4_K_M](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (~670 MB)

### Step 3 — Benchmark Execution

Runs `build/Release/ll_llm.exe` once for each prompt type:

| Prompt Type | File | Purpose |
|-------------|------|---------|
| `short` | `benchmarks/prompts/short.txt` | Measures first-token latency on simple queries |
| `long` | `benchmarks/prompts/long.txt` | Tests sustained throughput on longer context |
| `reasoning` | `benchmarks/prompts/reasoning.txt` | Evaluates performance on complex reasoning tasks |

### Step 4 — Summary

Lists the most recent JSON result files in the output directory.

---

## Output

Results are saved as **JSON files** in the output directory (default: `results/`). Each file contains:

- First token latency (ms)
- Tokens per second (tok/s)
- Peak RAM usage (MB)
- Model load time (ms)
- Prompt type and full configuration
