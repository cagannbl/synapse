<#
.SYNOPSIS
    Synapse AI-Native Programming Language - Windows One-Click Installer
.DESCRIPTION
    Installs Synapse CLI and libraries into %LOCALAPPDATA%\synapse,
    configures User PATH environment variable atomically, and verifies installation.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$SourceDir = "",

    [Parameter(Mandatory = $false)]
    [string]$InstallDir = "",

    [Parameter(Mandatory = $false)]
    [string]$BinaryUrl = "",

    [Parameter(Mandatory = $false)]
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor DarkGray
Write-Host "  Synapse AI-Native Language - Windows Installer      " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor DarkGray

# 1. Python runtime check
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Error "Hata: Python 3.10+ kurulu bulunamadi. Lutfen Python yukleyip tekrar deneyin."
        exit 1
    }
}
Write-Host "[1/5] Python calisma ortami tespit edildi: $($pythonCmd.Source)" -ForegroundColor Green

# 2. Determine paths
if (-not $SourceDir) {
    $SourceDir = Split-Path -Parent $PSScriptRoot
}
$synapsePackageSource = Join-Path $SourceDir "synapse"
if (-not (Test-Path $synapsePackageSource)) {
    Write-Error "Hata: Synapse paket kaynak dizini bulunamadi: $synapsePackageSource"
    exit 1
}

if (-not $InstallDir) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "synapse"
}
$BinDir = Join-Path $InstallDir "bin"
$LibDir = Join-Path $InstallDir "lib"
$DestPackageDir = Join-Path $LibDir "synapse"

Write-Host "[2/5] Kurulum hedefleri hazirlaniyor: $InstallDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
New-Item -ItemType Directory -Path $LibDir -Force | Out-Null

# 3. Copy/Sync Synapse library files & Standalone Binary
Write-Host "[3/5] Synapse kutuphaneleri ve binary dosyalari hazirlaniyor..." -ForegroundColor Cyan

# Standalone binary kurulumu (eger mevcutsa veya URL verilmisse)
$standaloneExeSource = Join-Path $SourceDir "dist\synapse.exe"
$destExe = Join-Path $BinDir "synapse.exe"
if ($BinaryUrl) {
    Write-Host "    Standalone binary indiriliyor: $BinaryUrl" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $BinaryUrl -OutFile $destExe -UseBasicParsing
} elseif (Test-Path $standaloneExeSource) {
    Write-Host "    Hazir standalone binary kopyalaniyor ($standaloneExeSource -> $destExe)..." -ForegroundColor Green
    Copy-Item -Path $standaloneExeSource -Destination $destExe -Force
}

if (Test-Path $DestPackageDir) {
    Remove-Item -Path $DestPackageDir -Recurse -Force | Out-Null
}
Copy-Item -Path $synapsePackageSource -Destination $DestPackageDir -Recurse -Force

# Create executable wrappers in bin
$cmdWrapper = @"
@echo off
set "SYNAPSE_HOME=%~dp0.."
set "PYTHONPATH=%SYNAPSE_HOME%\lib;%PYTHONPATH%"
python -m synapse.cli %*
"@
Set-Content -Path (Join-Path $BinDir "synapse.cmd") -Value $cmdWrapper -Encoding ASCII -Force
Set-Content -Path (Join-Path $BinDir "synapse.bat") -Value $cmdWrapper -Encoding ASCII -Force

$ps1Wrapper = @"
`$SynapseHome = Split-Path -Parent `$PSScriptRoot
`$env:PYTHONPATH = "`$SynapseHome\lib;`$env:PYTHONPATH"
python -m synapse.cli @args
"@
Set-Content -Path (Join-Path $BinDir "synapse.ps1") -Value $ps1Wrapper -Encoding UTF8 -Force

$shWrapper = @"
#!/usr/bin/env sh
DIR="`$(cd "`$(dirname "`$0")/.." && pwd)"
export PYTHONPATH="`$DIR/lib:`${PYTHONPATH}"
exec python -m synapse.cli "`$@"
"@
Set-Content -Path (Join-Path $BinDir "synapse") -Value $shWrapper -Encoding ASCII -Force

# 4. Atomic Windows User PATH configuration
Write-Host "[4/5] Windows PATH ortami yapilandiriliyor..." -ForegroundColor Cyan
$currentUserPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)
$currentSegments = if ([string]::IsNullOrWhiteSpace($currentUserPath)) {
    @()
} else {
    $currentUserPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

$alreadyInPath = $false
foreach ($seg in $currentSegments) {
    if ($seg.TrimEnd('\/').Equals($BinDir.TrimEnd('\/'), [System.StringComparison]::OrdinalIgnoreCase)) {
        $alreadyInPath = $true
        break
    }
}

if (-not $alreadyInPath) {
    $updatedSegments = $currentSegments + $BinDir
    $newPathValue = $updatedSegments -join ';'
    [Environment]::SetEnvironmentVariable("Path", $newPathValue, [EnvironmentVariableTarget]::User)
    Write-Host "  -> %LOCALAPPDATA%\synapse\bin kullanici PATH'ine basariyla eklendi." -ForegroundColor Green
} else {
    Write-Host "  -> %LOCALAPPDATA%\synapse\bin zaten kullanici PATH degiskeninde mevcut." -ForegroundColor DarkGray
}

# Update current session PATH
$sessionSegments = $env:PATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
$inSession = $false
foreach ($seg in $sessionSegments) {
    if ($seg.TrimEnd('\/').Equals($BinDir.TrimEnd('\/'), [System.StringComparison]::OrdinalIgnoreCase)) {
        $inSession = $true
        break
    }
}
if (-not $inSession) {
    $env:PATH = "$BinDir;$env:PATH"
}

# 5. Version verification and CLI test
if (-not $SkipTest) {
    Write-Host "[5/5] Sürüm dogrulamasi ve calisma testi yapiliyor..." -ForegroundColor Cyan
    $testLauncher = Join-Path $BinDir "synapse.cmd"
    $versionOutput = & $testLauncher --version 2>&1
    if ($LASTEXITCODE -ne 0 -or ($versionOutput -notmatch "Synapse")) {
        Write-Error "Hata: 'synapse --version' testi basarisiz oldu: $versionOutput"
        exit 1
    }
    Write-Host "  -> Dogrulama basarili: $versionOutput" -ForegroundColor Green
} else {
    Write-Host "[5/5] Sürüm dogrulamasi atlandi (-SkipTest)." -ForegroundColor DarkGray
}

Write-Host "`n======================================================" -ForegroundColor DarkGray
Write-Host "  Synapse kurulumu basariyla tamamlandi!               " -ForegroundColor Green
Write-Host "  Komut: synapse --version                            " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor DarkGray
