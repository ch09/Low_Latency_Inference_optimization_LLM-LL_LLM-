#pragma once
// =============================================================================
// sysinfo.h — Hardware Detection Utilities
// =============================================================================
// Detects CPU, RAM, and SIMD capabilities for auto-tuning and reporting.

#include <string>
#include <cstdint>
#include <sstream>
#include <iostream>
#include <thread>
#include <array>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <psapi.h>
    #include <intrin.h>
#else
    #include <sys/resource.h>
    #include <cpuid.h>
    #include <unistd.h>
#endif

namespace ll_llm {

struct SystemInfo {
    std::string cpu_name;
    int physical_cores   = 0;
    int logical_cores    = 0;
    uint64_t total_ram_mb = 0;
    bool has_avx2        = false;
    bool has_avx512      = false;
};

// -------------------------------------------------------------------------
// Get CPU brand string
// -------------------------------------------------------------------------
inline std::string get_cpu_name() {
    std::array<int, 4> cpui = {};
    char brand[0x40] = {};

#ifdef _WIN32
    __cpuid(cpui.data(), 0x80000000);
#else
    __cpuid(0x80000000, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif

    unsigned int max_ext = static_cast<unsigned int>(cpui[0]);
    if (max_ext >= 0x80000004) {
        for (unsigned int i = 0x80000002; i <= 0x80000004; ++i) {
#ifdef _WIN32
            __cpuid(cpui.data(), static_cast<int>(i));
#else
            __cpuid(i, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif
            std::memcpy(brand + (i - 0x80000002) * 16, cpui.data(), sizeof(cpui));
        }
    }
    return std::string(brand);
}

// -------------------------------------------------------------------------
// Detect SIMD capabilities
// -------------------------------------------------------------------------
inline bool detect_avx2() {
    std::array<int, 4> cpui = {};
#ifdef _WIN32
    __cpuidex(cpui.data(), 7, 0);
#else
    __cpuid_count(7, 0, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif
    return (cpui[1] & (1 << 5)) != 0; // AVX2 is bit 5 of EBX
}

inline bool detect_avx512() {
    std::array<int, 4> cpui = {};
#ifdef _WIN32
    __cpuidex(cpui.data(), 7, 0);
#else
    __cpuid_count(7, 0, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif
    return (cpui[1] & (1 << 16)) != 0; // AVX-512F is bit 16 of EBX
}

// -------------------------------------------------------------------------
// Get total system RAM
// -------------------------------------------------------------------------
inline uint64_t get_total_ram_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX mem = {};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        return mem.ullTotalPhys / (1024 * 1024);
    }
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<uint64_t>(pages) * page_size / (1024 * 1024);
#endif
}

// -------------------------------------------------------------------------
// Get current process peak RSS (Working Set) in MB
// -------------------------------------------------------------------------
inline double get_peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc = {};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage = {};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_maxrss) / 1024.0; // Linux: KB → MB
#endif
}

// -------------------------------------------------------------------------
// Get current RSS in MB (not peak)
// -------------------------------------------------------------------------
inline double get_current_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc = {};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    // Read from /proc/self/statm on Linux
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        long rss = 0;
        if (fscanf(f, "%*s%ld", &rss) == 1) {
            fclose(f);
            return static_cast<double>(rss) * sysconf(_SC_PAGE_SIZE) / (1024.0 * 1024.0);
        }
        fclose(f);
    }
    return 0.0;
#endif
}

// -------------------------------------------------------------------------
// Collect all system info
// -------------------------------------------------------------------------
inline SystemInfo collect_system_info() {
    SystemInfo info;
    info.cpu_name      = get_cpu_name();
    info.physical_cores = static_cast<int>(std::thread::hardware_concurrency());
    info.logical_cores  = info.physical_cores; // Approximation; refined later
    info.total_ram_mb   = get_total_ram_mb();
    info.has_avx2       = detect_avx2();
    info.has_avx512     = detect_avx512();
    return info;
}

// -------------------------------------------------------------------------
// Print system info
// -------------------------------------------------------------------------
inline void print_system_info(const SystemInfo& info) {
    std::cout << "\n";
    std::cout << "=== System Information ===\n";
    std::cout << "  CPU:        " << info.cpu_name << "\n";
    std::cout << "  Cores:      " << info.physical_cores << " (logical)\n";
    std::cout << "  RAM:        " << info.total_ram_mb << " MB\n";
    std::cout << "  AVX2:       " << (info.has_avx2   ? "YES" : "NO") << "\n";
    std::cout << "  AVX-512:    " << (info.has_avx512  ? "YES" : "NO") << "\n";
    std::cout << "==========================\n";
    std::cout << "\n";
}

} // namespace ll_llm
