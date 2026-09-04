<#
.SYNOPSIS
    Synapse AI-Native Programming Language - Windows One-Liner Demo Quickstart
.DESCRIPTION
    Executes instant Synapse showcase demos (NanoGPT, Tensor Matmul, Tour, DataLoader)
    without requiring manual installation.
.EXAMPLE
    irm https://get.synapse-lang.org/demo.ps1 | iex
.EXAMPLE
    .\scripts\demo.ps1 -Preset matmul
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false, Position = 0)]
    [ValidateSet("nanogpt", "matmul", "tour", "dataloader", "all")]
    [string]$Preset = "nanogpt",

    [Parameter(Mandatory = $false)]
    [string]$SourceDir = "",

    [Parameter(Mandatory = $false)]
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"

# Enable UTF-8 output encoding for console
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # Ignore if console encoding cannot be changed
}

Write-Host "======================================================" -ForegroundColor DarkGray
Write-Host "  Synapse AI-Native Language - Instant Demo Quickstart " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor DarkGray

# 1. Detect Python runtime
$pythonCmd = $null
foreach ($cmd in @("python", "py", "python3")) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        $pythonCmd = $found.Source
        break
    }
}

if (-not $pythonCmd) {
    Write-Host "[!] Python runtime not found in PATH." -ForegroundColor Yellow
    # Fallback to precompiled standalone edge binary if available locally
    $localExe = Join-Path (Split-Path -Parent $PSScriptRoot) "examples\edge_nanogpt\nanogpt.exe"
    if (Test-Path $localExe) {
        Write-Host "[*] Executing standalone C99 binary directly: $localExe" -ForegroundColor Green
        & $localExe
        exit 0
    }
    Write-Error "Error: Python 3.10+ is required for the full Synapse demo experience. Please install Python."
    exit 1
}

# 2. Determine repository or package root
if (-not $SourceDir) {
    if ($PSScriptRoot -and (Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) "synapse"))) {
        $SourceDir = Split-Path -Parent $PSScriptRoot
    } elseif (Test-Path ".\synapse") {
        $SourceDir = (Get-Item ".").FullName
    } elseif (Test-Path "$env:LOCALAPPDATA\synapse\lib") {
        $SourceDir = "$env:LOCALAPPDATA\synapse"
    }
}

# 3. Configure PYTHONPATH
$prevPythonPath = $env:PYTHONPATH
if ($SourceDir) {
    if (Test-Path (Join-Path $SourceDir "lib\synapse")) {
        $env:PYTHONPATH = "$SourceDir\lib;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = "$SourceDir;$env:PYTHONPATH"
    }
}

# 4. Dispatch demo presets
$cliArgs = @("-m", "synapse.cli", "demo")
if ($NonInteractive) {
    $cliArgs += "--non-interactive"
}

if ($Preset -eq "all") {
    $presetsToRun = @("nanogpt", "matmul", "tour", "dataloader")
} else {
    $presetsToRun = @($Preset)
}

foreach ($p in $presetsToRun) {
    Write-Host "`n[*] Running Synapse Showcase Preset: [$p]..." -ForegroundColor DarkGray
    $runArgs = $cliArgs + @("--preset=$p")
    & $pythonCmd @runArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Preset '$p' exited with code $LASTEXITCODE" -ForegroundColor Red
    }
}

# Restore environment
$env:PYTHONPATH = $prevPythonPath

Write-Host "======================================================" -ForegroundColor DarkGray
Write-Host "  Synapse Quickstart Demo Finished!                   " -ForegroundColor Green
Write-Host "  Install Full CLI: irm https://get.synapse-lang.org/install.ps1 | iex" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor DarkGray
