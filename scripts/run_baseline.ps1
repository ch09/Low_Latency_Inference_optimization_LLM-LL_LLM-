# =============================================================================
# run_baseline.ps1 — Automated Baseline Benchmark Runner
# =============================================================================
# Builds the project, runs benchmarks with all 3 prompt types, and saves results.
#
# Usage:
#   .\scripts\run_baseline.ps1 -ModelPath "models\your_model.gguf" [-Threads 4]
#
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [int]$Threads = 0,

    [int]$MaxTokens = 128,

    [string]$OutputDir = "results",

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  LL_LLM — Baseline Benchmark Runner" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------
# Step 1: Build (unless skipped)
# ---------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "[1/4] Building project..." -ForegroundColor Yellow
    Push-Location $ProjectRoot

    if (-not (Test-Path "build")) {
        cmake -B build -G "Visual Studio 17 2022" -A x64
    }
    cmake --build build --config Release

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed!" -ForegroundColor Red
        Pop-Location
        exit 1
    }

    Pop-Location
    Write-Host "[1/4] Build complete." -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping build (--SkipBuild)." -ForegroundColor DarkGray
}

# ---------------------------------------------------------
# Step 2: Check model exists
# ---------------------------------------------------------
Write-Host "[2/4] Checking model..." -ForegroundColor Yellow

$FullModelPath = Join-Path $ProjectRoot $ModelPath
if (-not (Test-Path $FullModelPath)) {
    Write-Host "ERROR: Model not found: $FullModelPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download a GGUF model and place it in the models/ directory." -ForegroundColor Yellow
    Write-Host "Recommended: TinyLlama 1.1B Q4_K_M (~670 MB)" -ForegroundColor Yellow
    Write-Host "  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host "[2/4] Model found: $FullModelPath" -ForegroundColor Green

# ---------------------------------------------------------
# Step 3: Run benchmarks
# ---------------------------------------------------------
Write-Host "[3/4] Running benchmarks..." -ForegroundColor Yellow

$Executable = Join-Path $ProjectRoot "build\Release\ll_llm.exe"
$PromptsDir = Join-Path $ProjectRoot "benchmarks\prompts"
$FullOutputDir = Join-Path $ProjectRoot $OutputDir

# Create output directory
New-Item -ItemType Directory -Force -Path $FullOutputDir | Out-Null

$PromptTypes = @("short", "long", "reasoning")
$ThreadArg = if ($Threads -gt 0) { "--threads $Threads" } else { "" }

foreach ($type in $PromptTypes) {
    $promptFile = Join-Path $PromptsDir "$type.txt"

    if (-not (Test-Path $promptFile)) {
        Write-Host "WARNING: Prompt file not found: $promptFile, skipping." -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "--- Running: $type prompt ---" -ForegroundColor Cyan

    $cmd = "$Executable --model `"$FullModelPath`" --prompt `"@$promptFile`" --prompt-type $type --max-tokens $MaxTokens --output-dir `"$FullOutputDir`" $ThreadArg"

    Write-Host "Command: $cmd" -ForegroundColor DarkGray
    Invoke-Expression $cmd

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Benchmark failed for $type prompt." -ForegroundColor Yellow
    }

    Write-Host ""
}

# ---------------------------------------------------------
# Step 4: Summary
# ---------------------------------------------------------
Write-Host "[4/4] Benchmark complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to: $FullOutputDir" -ForegroundColor Cyan
Write-Host ""

$resultFiles = Get-ChildItem -Path $FullOutputDir -Filter "*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
if ($resultFiles.Count -gt 0) {
    Write-Host "Recent result files:" -ForegroundColor Yellow
    foreach ($f in $resultFiles) {
        Write-Host "  $($f.Name)" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
