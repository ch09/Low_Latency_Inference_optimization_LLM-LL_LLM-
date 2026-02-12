# Phase 1 — Progress Log

> **Date**: February 12, 2026
> **Status**: ✅ First successful inference run achieved

---

## Step 1: Script Conversion (PowerShell → Python)

The original benchmark runner `scripts/run_baseline.ps1` was converted to `scripts/run_baseline.py` for better cross-platform support.

| Aspect | PowerShell | Python |
| ------ | ---------- | ------ |
| Arguments | `param()` block | `argparse` (standard CLI) |
| File paths | String manipulation | `pathlib.Path` (cross-platform) |
| Subprocesses | `Invoke-Expression` | `subprocess.run` (safer) |
| Colors | `-ForegroundColor` | ANSI escape codes |

Usage:

```bash
python scripts/run_baseline.py --model-path models/your_model.gguf --threads 4
```

---

## Step 2: Documentation Created

Two new docs were added:

- `docs/run_baseline_usage.md` — How to use the benchmark runner (arguments, examples, step-by-step)
- `docs/measurement_apis.md` — Deep-dive into every measurement API (chrono, PSAPI, CPUID, etc.)

---

## Step 3: Build Fixes (llama.cpp API Changes)

The project fetches llama.cpp via CMake `FetchContent`. The latest version introduced **breaking API changes**. Three errors were fixed:

### Error 1: `seed` removed from `llama_context_params`

```diff
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_threads = n_threads;
  ctx_params.n_threads_batch = n_threads;
- ctx_params.seed = static_cast<uint32_t>(args.seed);
+ // seed is no longer part of llama_context_params
+ // For greedy decoding (temperature=0), seed has no effect
```

**Why**: The llama.cpp project moved seed handling into the sampler chain API. Since we use greedy decoding (argmax), seed is irrelevant.

### Error 2: `llama_batch_add()` removed

```diff
- llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
- for (int i = 0; i < n_prompt_tokens; ++i) {
-     llama_batch_add(batch, prompt_tokens[i], i, {0},
-                     (i == n_prompt_tokens - 1));
- }
+ // Use llama_batch_get_one for simple single-sequence prompt prefill
+ llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
```

**Why**: `llama_batch_add()` was a helper that populated the batch struct field-by-field. It was removed. `llama_batch_get_one()` is the new recommended API for single-sequence use — it wraps a token array into a batch without allocating memory.

### Error 3: `llama_batch_clear()` removed

```diff
- llama_batch_clear(batch);
- llama_batch_add(batch, new_token_id, n_prompt_tokens + i, {0}, true);
+ // Single token decode
+ batch = llama_batch_get_one(&new_token_id, 1);
```

**Why**: Same reason as above — `llama_batch_get_one()` replaces the clear+add pattern for single tokens.

---

## Step 4: DLL Resolution

After building, the executable failed silently (exit code 1, no output). Root cause:

```text
build/
├── Release/
│   └── ll_llm.exe          ← executable here
└── bin/Release/
    ├── ggml.dll             ← DLLs built here
    ├── ggml-base.dll
    ├── ggml-cpu.dll
    └── llama.dll
```

**Fix**: Copied all 4 DLLs from `build/bin/Release/` to `build/Release/` alongside the executable.

> [!WARNING]
> After any clean rebuild, you must copy the DLLs again. A future CMake improvement should add a `POST_BUILD` copy step to automate this.

---

## Step 5: First Successful Run

```bash
ll_llm.exe --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "Hello" --threads 4 --max-tokens 16
```

### System Detected

| Property | Value |
| -------- | ----- |
| CPU | Intel Core Ultra 9 275HX |
| Logical Cores | 24 |
| RAM | 64,980 MB (~64 GB) |
| AVX2 | ✅ YES |
| AVX-512 | ❌ NO |

### Baseline Results (TinyLlama 1.1B, Q4_K_M, 4 threads)

| Metric | Value | Status |
| ------ | ----- | ------ |
| Model Load Time | 405.8 ms | — |
| First Token Latency | 54.5 ms | ✅ Under 100ms target |
| Tokens/sec | 39.65 tok/s | — |
| Peak RAM | 1,116.9 MB | — |
| RAM Before | 5.5 MB | — |
| RAM After | 1,116.9 MB | — |

Results saved to `results/baseline_20260212_111756.json`.

---

## Next Steps

- [ ] Run full benchmark with all 3 prompt types (short/long/reasoning)
- [ ] Download Mistral 7B Q4_K_M for primary benchmarking
- [ ] Add CMake `POST_BUILD` step to auto-copy DLLs
- [ ] Begin Phase 2: Profiling with Visual Studio Performance Profiler
