#!/usr/bin/env python3
"""
=============================================================================
run_baseline.py — Automated Baseline Benchmark Runner
=============================================================================
Builds the project, runs benchmarks with all 3 prompt types, and saves results.

Usage:
    python scripts/run_baseline.py --model-path models/your_model.gguf [--threads 4]

=============================================================================
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── ANSI color helpers ──────────────────────────────────────────────────────

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
GRAY    = "\033[90m"
WHITE   = "\033[97m"
RESET   = "\033[0m"


def info(msg: str)    -> None: print(f"{CYAN}{msg}{RESET}")
def success(msg: str) -> None: print(f"{GREEN}{msg}{RESET}")
def warn(msg: str)    -> None: print(f"{YELLOW}{msg}{RESET}")
def error(msg: str)   -> None: print(f"{RED}{msg}{RESET}")
def dim(msg: str)     -> None: print(f"{GRAY}{msg}{RESET}")


def run(cmd: list[str], cwd: str | Path | None = None) -> int:
    """Run a command and return the exit code."""
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main() -> None:
    # ── Argument parsing ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="LL_LLM — Automated Baseline Benchmark Runner"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Relative path to the GGUF model (e.g. models/your_model.gguf)"
    )
    parser.add_argument("--threads",     type=int, default=0,     help="Number of threads (0 = auto)")
    parser.add_argument("--max-tokens",  type=int, default=128,   help="Maximum tokens to generate")
    parser.add_argument("--output-dir",  default="results",       help="Directory for benchmark results")
    parser.add_argument("--skip-build",  action="store_true",     help="Skip the CMake build step")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    print()
    info("============================================")
    info("  LL_LLM — Baseline Benchmark Runner")
    info("============================================")
    print()

    # ── Step 1: Build (unless skipped) ──────────────────────────────────
    if not args.skip_build:
        warn("[1/4] Building project...")
        build_dir = project_root / "build"

        if not build_dir.exists():
            rc = run(
                ["cmake", "-B", "build", "-G", "Visual Studio 17 2022", "-A", "x64"],
                cwd=project_root,
            )
            if rc != 0:
                error("ERROR: CMake configure failed!")
                sys.exit(1)

        rc = run(
            ["cmake", "--build", "build", "--config", "Release"],
            cwd=project_root,
        )
        if rc != 0:
            error("ERROR: Build failed!")
            sys.exit(1)

        success("[1/4] Build complete.")
    else:
        dim("[1/4] Skipping build (--skip-build).")

    # ── Step 2: Check model exists ──────────────────────────────────────
    warn("[2/4] Checking model...")
    full_model_path = project_root / args.model_path

    if not full_model_path.exists():
        error(f"ERROR: Model not found: {full_model_path}")
        print()
        warn("Download a GGUF model and place it in the models/ directory.")
        warn("Recommended: TinyLlama 1.1B Q4_K_M (~670 MB)")
        info("  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print()
        sys.exit(1)

    success(f"[2/4] Model found: {full_model_path}")

    # ── Step 3: Run benchmarks ──────────────────────────────────────────
    warn("[3/4] Running benchmarks...")

    executable = project_root / "build" / "Release" / "ll_llm.exe"
    prompts_dir = project_root / "benchmarks" / "prompts"
    full_output_dir = project_root / args.output_dir

    full_output_dir.mkdir(parents=True, exist_ok=True)

    prompt_types = ["short", "long", "reasoning"]

    for ptype in prompt_types:
        prompt_file = prompts_dir / f"{ptype}.txt"

        if not prompt_file.exists():
            warn(f"WARNING: Prompt file not found: {prompt_file}, skipping.")
            continue

        print()
        info(f"--- Running: {ptype} prompt ---")

        cmd = [
            str(executable),
            "--model",       str(full_model_path),
            "--prompt",      f"@{prompt_file}",
            "--prompt-type", ptype,
            "--max-tokens",  str(args.max_tokens),
            "--output-dir",  str(full_output_dir),
        ]
        if args.threads > 0:
            cmd.extend(["--threads", str(args.threads)])

        dim(f"Command: {' '.join(cmd)}")
        rc = run(cmd)

        if rc != 0:
            warn(f"WARNING: Benchmark failed for {ptype} prompt.")

        print()

    # ── Step 4: Summary ─────────────────────────────────────────────────
    success("[4/4] Benchmark complete!")
    print()
    info(f"Results saved to: {full_output_dir}")
    print()

    json_files = sorted(
        full_output_dir.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:5]

    if json_files:
        warn("Recent result files:")
        for f in json_files:
            print(f"  {WHITE}{f.name}{RESET}")

    print()
    success("Done.")


if __name__ == "__main__":
    main()
