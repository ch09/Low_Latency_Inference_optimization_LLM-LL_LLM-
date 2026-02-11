# Profiling Notes — Phase 2

> This document will be filled during Phase 2 with profiling data from Visual Studio Performance Profiler.

## Tools

- **Visual Studio 2022 Performance Profiler**
  - CPU Usage (sampling)
  - Memory Usage
  - Concurrency Visualizer

## Methodology

1. Build in **Release** mode with debug info (`/Zi /O2`)
2. Run with a fixed prompt and fixed seed for reproducibility
3. Capture a 30-second profile during full inference
4. Focus on:
   - Top 10 hottest functions
   - Memory allocation patterns
   - Thread utilization

## Findings

_To be completed during Phase 2._

### CPU Hotspots

| Function | % CPU Time | Module | Notes |
|----------|-----------|--------|-------|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Memory Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Peak Working Set | _TBD_ | |
| KV Cache Size | _TBD_ | |
| Model Weight Memory | _TBD_ | |

### Thread Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Active Threads | _TBD_ | |
| Core Utilization | _TBD_ | |
| Lock Contention | _TBD_ | |

## Conclusions & Next Steps

_To be completed during Phase 2._
