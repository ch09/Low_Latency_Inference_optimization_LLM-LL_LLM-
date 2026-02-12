# Compilation Process — How `ll_llm.exe` Is Built

> This document explains every step of the build process, from source code to executable.

---

## Overview

The project uses **CMake** as its build system and **MSVC** (Visual Studio Build Tools) as the compiler. The key complexity is that the main dependency — **llama.cpp** — is fetched and compiled from source as part of our build.

```mermaid
flowchart LR
    subgraph Input["Source Files"]
        CMAKE["CMakeLists.txt"]
        MAIN["main.cpp"]
        BENCH["benchmark.h"]
        SYS["sysinfo.h"]
    end

    subgraph Fetch["Dependency Fetch"]
        GIT["llama.cpp\n(from GitHub)"]
    end

    subgraph Build["Compilation"]
        MSVC["MSVC Compiler\n(cl.exe)"]
    end

    subgraph Output["Build Artifacts"]
        EXE["ll_llm.exe"]
        DLL1["llama.dll"]
        DLL2["ggml.dll"]
        DLL3["ggml-base.dll"]
        DLL4["ggml-cpu.dll"]
    end

    CMAKE --> Fetch
    CMAKE --> Build
    GIT --> Build
    Input --> Build
    Build --> Output

    style Input fill:#0f3460,stroke:#00d4ff,color:#fff
    style Fetch fill:#533483,stroke:#fff,color:#fff
    style Build fill:#e94560,stroke:#fff,color:#fff
    style Output fill:#16213e,stroke:#00d4ff,color:#fff
```

---

## Step 1: CMake Configure

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

This step reads `CMakeLists.txt` and generates the Visual Studio build files (`.sln`, `.vcxproj`). Here's what happens:

```mermaid
flowchart TB
    CMAKE["cmake -B build"] --> PARSE["Parse CMakeLists.txt"]

    PARSE --> STD["Set C++17 Standard"]
    PARSE --> FLAGS["Set Compiler Flags\n/W4 /O2 /EHsc /arch:AVX2"]
    PARSE --> FETCH["FetchContent: Clone llama.cpp"]
    PARSE --> TARGETS["Define Build Targets"]

    FETCH --> CLONE["git clone --shallow\nggerganov/llama.cpp\n→ build/_deps/llama_cpp-src/"]

    CLONE --> DISABLE["Disable GPU Backends\nCUDA=OFF, Vulkan=OFF\nMetal=OFF, OpenCL=OFF"]

    DISABLE --> AVAILABLE["FetchContent_MakeAvailable\n→ llama.cpp's own CMakeLists.txt\n  is processed recursively"]

    TARGETS --> EXEC["add_executable(ll_llm)\n→ src/main.cpp"]

    EXEC --> INCLUDES["target_include_directories\n• src/\n• llama.cpp/include/\n• llama.cpp/ggml/include/"]

    EXEC --> LINK["target_link_libraries\n• llama (inference engine)\n• ggml (tensor math)\n• psapi (Windows memory API)"]

    AVAILABLE --> GEN["Generate .sln + .vcxproj files"]
    LINK --> GEN

    style CMAKE fill:#e94560,stroke:#fff,color:#fff
    style FETCH fill:#533483,stroke:#fff,color:#fff
    style CLONE fill:#533483,stroke:#fff,color:#fff
```

### What FetchContent Does

`FetchContent` is CMake's built-in dependency manager. On first configuration it:

1. Clones `llama.cpp` from GitHub into `build/_deps/llama_cpp-src/`
2. Processes llama.cpp's own `CMakeLists.txt` as if it were part of our project
3. Makes all llama.cpp targets (`llama`, `ggml`, `ggml-base`, `ggml-cpu`) available for linking

On subsequent builds, the clone is cached — it won't re-download unless you delete the `build/` directory.

### Compiler Flags Explained

| Flag | Purpose |
| ---- | ------- |
| `/W4` | Warning level 4 — catch most potential bugs at compile time |
| `/O2` | Full optimization — enables inlining, loop unrolling, SIMD auto-vectorization |
| `/EHsc` | C++ exception handling — required for `try`/`catch` |
| `/arch:AVX2` | Enable AVX2 SIMD instructions — process 8 floats per instruction |

---

## Step 2: Compilation

```bash
cmake --build build --config Release
```

This step compiles every `.cpp` file into object files (`.obj`), then links them into executables and DLLs.

```mermaid
flowchart TB
    subgraph OurCode["Our Source Files"]
        MAIN["src/main.cpp"]
        BENCH["src/benchmark.h\n(included by main.cpp)"]
        SYS["src/sysinfo.h\n(included by main.cpp)"]
    end

    subgraph LlamaCode["llama.cpp Source Files (~50+ files)"]
        LLAMA_SRC["llama.cpp\nllama-context.cpp\nllama-model.cpp\nllama-sampling.cpp\n..."]
        GGML_SRC["ggml.c\nggml-cpu.c\nggml-backend.c\nggml-alloc.c\n..."]
    end

    subgraph Compile["MSVC Compiler (cl.exe)"]
        direction TB
        C1["Preprocess\n(#include expansion,\n#define substitution)"]
        C2["Compile to .obj\n(C++ → machine code)"]
        C3["each .cpp → one .obj file"]
        C1 --> C2 --> C3
    end

    subgraph Link["MSVC Linker (link.exe)"]
        direction TB
        L1["Resolve symbols\nacross all .obj files"]
        L2["Link system libraries\n(psapi.lib, kernel32.lib)"]
        L3["Generate outputs"]
        L1 --> L2 --> L3
    end

    OurCode --> Compile
    LlamaCode --> Compile
    Compile --> Link

    subgraph Outputs["Build Outputs"]
        EXE["build/Release/\nll_llm.exe"]
        DLL["build/bin/Release/\nllama.dll\nggml.dll\nggml-base.dll\nggml-cpu.dll"]
    end

    Link --> Outputs

    style Compile fill:#e94560,stroke:#fff,color:#fff
    style Link fill:#533483,stroke:#fff,color:#fff
    style Outputs fill:#0f3460,stroke:#00d4ff,color:#fff
```

### What Gets Compiled

| Component | Source Files | Output | Role |
| --------- | ----------- | ------ | ---- |
| **ll_llm** | `main.cpp` (includes `benchmark.h`, `sysinfo.h`) | `ll_llm.exe` | Our benchmark harness |
| **llama** | `llama.cpp`, `llama-context.cpp`, `llama-model.cpp`, ... | `llama.dll` | Model loading, tokenization, inference |
| **ggml** | `ggml.c`, `ggml-alloc.c`, `ggml-backend.c`, ... | `ggml.dll` | Core tensor operations |
| **ggml-base** | Backend base abstractions | `ggml-base.dll` | Backend management |
| **ggml-cpu** | `ggml-cpu.c` (AVX2/SSE kernels) | `ggml-cpu.dll` | CPU-optimized SIMD matrix math |

### Preprocessing: How Headers Work

`main.cpp` is the only `.cpp` file we write, but it `#include`s two headers:

```mermaid
flowchart LR
    MAIN["main.cpp"] -->|"#include"| BENCH["benchmark.h\n• Timer class\n• BenchmarkResult struct\n• JSON output"]
    MAIN -->|"#include"| SYS["sysinfo.h\n• CPU name detection\n• Core count\n• RAM size\n• AVX2/AVX-512 check"]
    MAIN -->|"#include"| LLAMA_H["llama.h\n(from llama.cpp)\n• llama_model_load\n• llama_decode\n• llama_batch_get_one"]

    style MAIN fill:#e94560,stroke:#fff,color:#fff
    style BENCH fill:#0f3460,stroke:#00d4ff,color:#fff
    style SYS fill:#0f3460,stroke:#00d4ff,color:#fff
    style LLAMA_H fill:#533483,stroke:#fff,color:#fff
```

The preprocessor pastes the contents of each header directly into `main.cpp` before compilation. The result is one large "translation unit" that the compiler converts to `main.obj`.

---

## Step 3: Linking

The linker combines all compiled object files and resolves cross-references:

```mermaid
flowchart LR
    subgraph Objects["Object Files"]
        OBJ1["main.obj"]
        OBJ2["llama.obj"]
        OBJ3["ggml.obj"]
    end

    subgraph SystemLibs["System Libraries"]
        PSAPI["psapi.lib\n(GetProcessMemoryInfo)"]
        KERNEL["kernel32.lib\n(Windows APIs)"]
        CRT["MSVC CRT\n(printf, malloc, ...)"]
    end

    LINKER["link.exe\n(Linker)"]

    Objects --> LINKER
    SystemLibs --> LINKER
    LINKER --> EXE["ll_llm.exe"]
    LINKER --> DLL["llama.dll + ggml*.dll"]

    style LINKER fill:#e94560,stroke:#fff,color:#fff
    style EXE fill:#0f3460,stroke:#00d4ff,color:#fff
```

### Why DLLs?

llama.cpp builds as **shared libraries** (DLLs) by default. This means:

- `ll_llm.exe` contains only our code (~40 KB)
- `llama.dll` + `ggml*.dll` contain the heavy inference engine (~15 MB total)
- At runtime, Windows loads the DLLs into the same process

> [!IMPORTANT]
> The DLLs are built to `build/bin/Release/` while the exe goes to `build/Release/`. You must copy the DLLs next to the exe for it to run. Without them, the exe silently exits with code 1.

---

## Step 4: Runtime — What Happens When You Run `ll_llm.exe`

```mermaid
sequenceDiagram
    participant OS as Windows OS
    participant EXE as ll_llm.exe
    participant DLL as llama.dll + ggml*.dll
    participant PSAPI as psapi.dll

    OS->>EXE: Load executable into memory

    Note over OS: Dynamic Linking Phase
    OS->>DLL: Load llama.dll, ggml.dll,<br/>ggml-base.dll, ggml-cpu.dll
    OS->>PSAPI: Load psapi.dll
    OS->>EXE: Resolve all imported symbols

    Note over EXE: Execution Begins
    EXE->>EXE: Parse CLI arguments
    EXE->>EXE: Detect CPU (sysinfo.h)
    EXE->>DLL: llama_backend_init()
    EXE->>DLL: llama_model_load_from_file()

    Note over DLL: Model Loading<br/>mmap() the GGUF file<br/>~400ms for 670 MB

    EXE->>DLL: llama_tokenize(prompt)
    EXE->>DLL: llama_decode(batch)

    Note over DLL: Inference<br/>Matrix multiplications<br/>using AVX2 SIMD

    EXE->>PSAPI: GetProcessMemoryInfo()
    EXE->>EXE: Print results + save JSON
```

---

## Complete Build Command Sequence

```bash
# 1. Configure (first time only — downloads llama.cpp)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Compile + Link
cmake --build build --config Release

# 3. Copy DLLs (required after clean build)
cp build/bin/Release/*.dll build/Release/

# 4. Run
./build/Release/ll_llm.exe --model models/your_model.gguf --threads 4
```

---

## Directory Layout After Build

```text
LL_LLM/
├── CMakeLists.txt                          ← Build configuration
├── src/
│   ├── main.cpp                            ← Our code (entry point)
│   ├── benchmark.h                         ← Timer + metrics
│   └── sysinfo.h                           ← Hardware detection
├── build/
│   ├── Release/
│   │   ├── ll_llm.exe                      ← Our executable
│   │   ├── llama.dll                       ← (copied from bin/)
│   │   ├── ggml.dll                        ← (copied from bin/)
│   │   ├── ggml-base.dll                   ← (copied from bin/)
│   │   └── ggml-cpu.dll                    ← (copied from bin/)
│   ├── bin/Release/
│   │   ├── llama.dll                       ← Original DLL location
│   │   ├── ggml.dll
│   │   ├── ggml-base.dll
│   │   └── ggml-cpu.dll
│   └── _deps/
│       └── llama_cpp-src/                  ← Fetched llama.cpp source
│           ├── include/llama.h
│           ├── ggml/include/ggml.h
│           └── src/...                     ← ~50+ source files
└── models/
    └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```
