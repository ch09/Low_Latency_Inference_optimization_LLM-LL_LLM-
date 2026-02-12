# How Measurements Are Done — APIs & Techniques

This document explains **how every metric** in the LL_LLM benchmark is measured, which **system APIs** are called, and where in the code each measurement happens.

---

## Overview

| Metric | Unit | API / Technique | Source File |
|--------|------|-----------------|-------------|
| Model Load Time | ms | `std::chrono::high_resolution_clock` | `benchmark.h` |
| First Token Latency | ms | `std::chrono::high_resolution_clock` | `benchmark.h` |
| Tokens per Second | tok/s | Computed: `tokens ÷ elapsed_s` | `main.cpp` |
| Peak RAM (Working Set) | MB | `GetProcessMemoryInfo()` (Win) / `getrusage()` (Linux) | `sysinfo.h` |
| Current RAM | MB | `GetProcessMemoryInfo()` (Win) / `/proc/self/statm` (Linux) | `sysinfo.h` |
| CPU Detection | — | `__cpuid` / `__cpuidex` intrinsics | `sysinfo.h` |
| Total System RAM | MB | `GlobalMemoryStatusEx()` (Win) / `sysconf()` (Linux) | `sysinfo.h` |

---

## Time Measurements

All time measurements use C++ `<chrono>` high-resolution clock, implemented in the `Timer` class (`benchmark.h`).

### How the Timer Works

```cpp
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;

    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop()  { end_   = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() {
        // Microsecond precision, converted to milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
        return duration.count() / 1000.0;
    }
};
```

**Why `high_resolution_clock`?** It provides the finest granularity available on the platform — typically **nanosecond resolution** on modern CPUs. This is essential because first-token latency can be just a few milliseconds.

### What Gets Timed

| Timer | Starts | Stops | Measures |
|-------|--------|-------|----------|
| `load_timer` | Before `llama_model_load_from_file()` | After model is loaded into memory | **Model Load Time** — how long to read and parse the GGUF file |
| `first_token_timer` | Before prompt prefill (`llama_decode`) | After the first generated token's logits are sampled | **First Token Latency** — prompt processing + first decode step |
| `gen_timer` | Before the generation loop | After the last token is generated or EOS | **Total Generation Time** — used to compute tokens/sec |

### Tokens per Second Calculation

```cpp
result.tokens_per_second = tokens_generated / (total_generation_ms / 1000.0);
```

This is raw **decode throughput** — it measures only the token generation loop, excluding prompt prefill.

---

## Memory Measurements

### Peak RAM — `get_peak_rss_mb()`

Measures the **highest amount of physical memory** the process ever used.

#### Windows

```cpp
#include <psapi.h>

PROCESS_MEMORY_COUNTERS pmc = {};
GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
double peak_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
```

| API | Header | What It Returns |
|-----|--------|-----------------|
| `GetCurrentProcess()` | `<windows.h>` | Handle to the current process |
| `GetProcessMemoryInfo()` | `<psapi.h>` | Fills `PROCESS_MEMORY_COUNTERS` struct |
| `pmc.PeakWorkingSetSize` | — | Peak physical memory in **bytes** |

> **Working Set** = the set of physical RAM pages currently mapped to the process. The *peak* is the all-time maximum.

#### Linux

```cpp
#include <sys/resource.h>

struct rusage usage = {};
getrusage(RUSAGE_SELF, &usage);
double peak_mb = usage.ru_maxrss / 1024.0;  // Linux reports in KB
```

| API | Header | What It Returns |
|-----|--------|-----------------|
| `getrusage()` | `<sys/resource.h>` | Resource usage stats for the process |
| `ru_maxrss` | — | Peak resident set size in **kilobytes** |

---

### Current RAM — `get_current_rss_mb()`

Measures the **current** physical memory usage (not the peak).

#### Windows

```cpp
pmc.WorkingSetSize / (1024.0 * 1024.0);  // Same API, different field
```

#### Linux

```cpp
// Reads /proc/self/statm — a virtual file exposing process memory stats
FILE* f = fopen("/proc/self/statm", "r");
fscanf(f, "%*s%ld", &rss);  // Second field = resident pages
double mb = rss * sysconf(_SC_PAGE_SIZE) / (1024.0 * 1024.0);
```

### When Memory Is Sampled

```
┌─────────────────────────────────────────────────────────────┐
│                    Timeline                                  │
├──────────┬────────────────────────────────┬──────────────────┤
│ ram_before│    Model load + Inference      │ ram_after        │
│          │                                │ peak_ram         │
└──────────┴────────────────────────────────┴──────────────────┘
     ▲                                            ▲
     │                                            │
  get_current_rss_mb()                    get_current_rss_mb()
  (before loading model)                  get_peak_rss_mb()
                                          (after generation)
```

- **`ram_before_mb`** — snapshot *before* the model is loaded (process baseline)
- **`ram_after_mb`** — snapshot *after* generation completes
- **`peak_ram_mb`** — the all-time peak during the entire run (includes model loading spike)

---

## CPU Detection

### CPU Brand String — `get_cpu_name()`

Uses the `CPUID` instruction (leaves `0x80000002`–`0x80000004`) to read the CPU's marketing name string (e.g. `"Intel(R) Core(TM) Ultra 7 155H"`).

```cpp
// Windows: MSVC intrinsic
__cpuid(cpui.data(), 0x80000002);  // First 16 chars of brand string
__cpuid(cpui.data(), 0x80000003);  // Next 16 chars
__cpuid(cpui.data(), 0x80000004);  // Last 16 chars

// Linux: GCC/Clang intrinsic
__cpuid(0x80000002, eax, ebx, ecx, edx);
```

### SIMD Feature Detection

Uses `CPUID` leaf 7 to check specific CPU feature flags.

| Feature | Bit Checked | Register | Formula |
|---------|-------------|----------|---------|
| **AVX2** | Bit 5 | EBX | `(ebx & (1 << 5)) != 0` |
| **AVX-512F** | Bit 16 | EBX | `(ebx & (1 << 16)) != 0` |

```cpp
// Windows
__cpuidex(cpui.data(), 7, 0);   // Leaf 7, sub-leaf 0

// Linux
__cpuid_count(7, 0, eax, ebx, ecx, edx);
```

> **Why this matters**: AVX2 enables 256-bit SIMD operations for matrix multiplication. AVX-512 doubles that to 512-bit. Knowing which is available helps us understand the theoretical throughput ceiling.

### Total System RAM — `get_total_ram_mb()`

#### Windows

```cpp
#include <windows.h>

MEMORYSTATUSEX mem = {};
mem.dwLength = sizeof(mem);
GlobalMemoryStatusEx(&mem);
uint64_t total_mb = mem.ullTotalPhys / (1024 * 1024);
```

#### Linux

```cpp
#include <unistd.h>

long pages     = sysconf(_SC_PHYS_PAGES);  // Total physical pages
long page_size = sysconf(_SC_PAGE_SIZE);    // Bytes per page (usually 4096)
uint64_t total = pages * page_size / (1024 * 1024);
```

### Core Count

```cpp
#include <thread>

int cores = std::thread::hardware_concurrency();  // Returns logical core count
```

> This is a C++ standard library call — no OS-specific API needed. Returns the number of **logical** cores (includes hyper-threading).

---

## Timestamp Generation

Result files are named with a timestamp to avoid overwriting previous runs:

```cpp
auto now  = std::chrono::system_clock::now();
auto time = std::chrono::system_clock::to_time_t(now);

// Platform-safe localtime conversion
#ifdef _WIN32
    localtime_s(&tm_buf, &time);    // Thread-safe (MSVC)
#else
    localtime_r(&time, &tm_buf);    // Thread-safe (POSIX)
#endif

// Format: baseline_20260212_103000.json
std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
```

> **Why not `localtime()`?** The standard `localtime()` returns a pointer to a shared static buffer — it's **not thread-safe**. Both `localtime_s` (Windows) and `localtime_r` (POSIX) write to a caller-provided buffer instead.

---

## Summary: API Quick Reference

### Windows APIs

| API | Header | Purpose |
|-----|--------|---------|
| `GetProcessMemoryInfo()` | `<psapi.h>` | Current and peak process memory |
| `GlobalMemoryStatusEx()` | `<windows.h>` | Total system physical RAM |
| `__cpuid()` | `<intrin.h>` | CPU brand string |
| `__cpuidex()` | `<intrin.h>` | CPU feature flags (AVX2, AVX-512) |
| `localtime_s()` | `<ctime>` | Thread-safe timestamp conversion |

### Linux APIs

| API | Header | Purpose |
|-----|--------|---------|
| `getrusage()` | `<sys/resource.h>` | Peak process memory (RSS) |
| `sysconf()` | `<unistd.h>` | Total physical pages, page size |
| `/proc/self/statm` | (filesystem) | Current process memory |
| `__cpuid()` | `<cpuid.h>` | CPU brand string |
| `__cpuid_count()` | `<cpuid.h>` | CPU feature flags (AVX2, AVX-512) |
| `localtime_r()` | `<ctime>` | Thread-safe timestamp conversion |

### C++ Standard Library

| API | Header | Purpose |
|-----|--------|---------|
| `std::chrono::high_resolution_clock` | `<chrono>` | All time measurements |
| `std::chrono::system_clock` | `<chrono>` | Timestamp for filenames |
| `std::thread::hardware_concurrency()` | `<thread>` | Logical core count |
