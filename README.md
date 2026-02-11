# LL_LLM — C++ LLM Inference Optimization

> A systematic, profiling-driven study of Large Language Model inference optimization on CPU-only hardware.

---

## Objective

This project investigates **how fast we can run LLM inference on a CPU-only laptop** through systematic profiling and optimization of every layer in the inference pipeline.

### Target

| Parameter | Value |
|-----------|-------|
| **Hardware** | CPU-only (Intel Core Ultra, ~64 GB RAM) |
| **OS** | Windows 11 |
| **Compiler** | MSVC (Visual Studio 2022) |
| **Model Class** | 7B parameter models (GGUF format) |
| **Latency Target** | < 100ms first token |
| **Scope** | Inference only (no training) |

### Non-Goals

- GPU / CUDA acceleration (future work)
- Training or fine-tuning
- Distributed inference
- Mobile or embedded targets

---

## Metrics

Every optimization is measured against these metrics. No blind optimization — **data drives decisions**.

| Metric | Unit | Measurement Method |
|--------|------|--------------------|
| **First Token Latency** | ms | Wall-clock time from prompt submission to first generated token |
| **Tokens per Second** | tok/s | Total generated tokens ÷ total generation time |
| **Peak RAM Usage** | MB | `GetProcessMemoryInfo()` peak working set |
| **Model Load Time** | ms | Wall-clock time to load GGUF file into memory |
| **Accuracy Score** | % | Exact-match on deterministic test suite vs FP16 baseline |

---

## Project Phases

### Phase 0 — Define Objective (Done)

Define target hardware, metrics, and project structure.

### Phase 1 — Baseline System (Week 1)

- Integrate llama.cpp as inference backend
- Run 7B Q4_K_M model with temperature=0
- Record baseline metrics across 3 prompt types (short / long / reasoning)

### Phase 2 — Profiling & Bottleneck Analysis (Week 2)

- Profile with Visual Studio Performance Profiler
- Identify: MatMul hotspots, KV cache costs, threading inefficiency
- Document findings in `docs/profiling_notes.md`

### Phase 3 — Quantization Study (Week 3)

- Systematic comparison: FP16, Q8, Q4, Q2
- Measure latency, RAM, and accuracy for each
- Create comparison table and graphs

### Phase 4 — Advanced Optimizations (Week 4-5)

Pick 2-3 from:

- **A) KV Cache Optimization** — pooling, compression, eviction
- **B) Threading Optimization** — pinning, custom pools, scaling curves
- **C) Speculative Decoding** — draft model + target model verification
- **D) Custom MatMul** — SIMD (AVX2), cache-aware tiling

### Phase 5 — Evaluation Framework (Week 6)

- Automated benchmark suite with fixed seeds
- Performance graphs (latency vs quantization, throughput vs threads, etc.)

### Phase 6 — Portfolio Features (Optional)

- Real-time streaming CLI (< 100ms first token)
- Adaptive quantization engine
- Auto-tuning hardware detection

---

## Building

### Prerequisites

- CMake 3.20+
- Visual Studio 2022 (or MSVC Build Tools)
- Git

### Build

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Run

```powershell
# Download a model first (see Models section below)
.\build\Release\ll_llm.exe --model models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "What is the capital of France?" --threads 4
```

---

## Models

Models are **not included** in the repository. Download GGUF models from [Hugging Face](https://huggingface.co/) and place them in the `models/` directory.

### Recommended Models

| Model | Size | Use Case |
|-------|------|----------|
| TinyLlama 1.1B Q4_K_M | ~670 MB | Quick iteration, smoke tests |
| Mistral 7B Instruct v0.2 Q4_K_M | ~4.4 GB | Primary benchmark model |
| Mistral 7B Instruct v0.2 Q8_0 | ~7.7 GB | Higher accuracy baseline |

---

## Project Structure

```
LL_LLM/
├── README.md                  # This file
├── CMakeLists.txt             # Build system
├── .gitignore                 # Ignored files
├── src/
│   ├── main.cpp               # Entry point & CLI
│   ├── benchmark.h            # Metrics collection
│   └── sysinfo.h              # Hardware detection
├── benchmarks/
│   └── prompts/               # Test prompts
│       ├── short.txt
│       ├── long.txt
│       └── reasoning.txt
├── results/                   # Benchmark output (gitignored)
├── models/                    # GGUF models (gitignored)
├── scripts/
│   └── run_baseline.ps1       # Automated benchmark runner
└── docs/
    ├── project_documentation.md  # Full technical documentation
    └── profiling_notes.md        # Phase 2 findings
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Walid Chebbi — Systems & Performance Engineering
