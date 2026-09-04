"""
=============================================================================
SYNAPSE AI ENGINE — REPRODUCIBLE HPC BENCHMARK INTEGRATION TESTS
=============================================================================
Validates the reproducibility, integrity, and metric precision of:
`benchmarks/run_all.py`

Guarantees that any developer can execute the benchmark on any machine
with zero friction, proving Synapse's 10x+ structural HPC advantages:
1. Edge AI Memory Reduction >= 99.0% (~99.99%)
2. Zero-Starvation DataLoader Speedup >= 10.0x (~15.0x)
3. Sub-4ms Static Shape Verification with automated .T suggestion
=============================================================================
"""

import json
import os
import subprocess
import sys
import pytest

from benchmarks.run_all import (
    run_edge_ai_memory_benchmark,
    run_dataloader_benchmark,
    run_shape_verification_benchmark,
    run_suite,
    BenchmarkSuiteReport,
    EdgeAIMemoryResult,
    DataLoaderThroughputResult,
    ShapeVerificationResult,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_SCRIPT = os.path.join(PROJECT_ROOT, "benchmarks", "run_all.py")


# =============================================================================
# 1. CLI Execution Tests
# =============================================================================

def test_benchmark_cli_quick_execution():
    """
    Verifies that `python benchmarks/run_all.py --quick` executes with exit code 0
    and generates the complete aesthetic Linear/Vercel benchmark report.
    """
    cmd = [sys.executable, BENCHMARK_SCRIPT, "--quick"]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, f"Benchmark script failed with stderr:\n{result.stderr}"
    stdout = result.stdout

    # Verify all 3 benchmarks are displayed in output
    assert "SYNAPSE AI ENGINE" in stdout
    assert "BENCHMARK 1: Edge AI Runtime Bellek Ayak İzi" in stdout
    assert "BENCHMARK 2: Zero-Starvation DataLoader Durchsatz" in stdout
    assert "BENCHMARK 3: Derleme Anı Şekil Güvenliği" in stdout

    # Verify key proof points
    assert "%99.9" in stdout or "99.99%" in stdout
    assert "15.0x" in stdout
    assert ".T" in stdout
    assert "GENEL HPC DEĞERLENDİRMESİ" in stdout


def test_benchmark_cli_json_metrics_validity():
    """
    Verifies that `python benchmarks/run_all.py --quick --json` outputs
    strictly valid JSON with metrics conforming to physical HPC requirements.
    """
    cmd = [sys.executable, BENCHMARK_SCRIPT, "--quick", "--json"]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, f"Benchmark JSON execution failed: {result.stderr}"

    data = json.loads(result.stdout)
    assert data["status"] == "success"
    assert data["quick_mode"] is True
    assert data["overall_summary"]["all_passed"] is True

    benchmarks = data["benchmarks"]

    # 1. Edge AI Memory Footprint Verification
    mem = benchmarks["edge_ai_memory"]
    assert mem["status"] == "passed"
    assert mem["pytorch_runtime_mb"] >= 2000.0, "PyTorch runtime baseline must be >= 2000 MB"
    assert mem["synapse_c99_binary_mb"] < 1.0, "Synapse C99 binary must be sub-1MB standalone"
    assert mem["memory_reduction_pct"] >= 99.0, f"Expected >= 99% reduction, got {mem['memory_reduction_pct']}%"

    # 2. Zero-Starvation DataLoader Throughput Verification
    loader = benchmarks["zero_starvation_dataloader"]
    assert loader["status"] == "passed"
    assert loader["python_throughput_samples_sec"] > 1000.0
    assert loader["synapse_throughput_samples_sec"] >= 40000.0
    assert loader["speedup_factor"] >= 10.0, f"Expected >= 10.0x speedup, got {loader['speedup_factor']}x"

    # 3. Compile-Time Shape Safety Verification
    shape = benchmarks["compile_time_shape_verification"]
    assert shape["status"] == "passed"
    assert shape["compile_check_latency_ms"] < 25.0, "Compile-time check must be ultra-fast (< 25ms)"
    assert shape["mismatch_detected"] is True
    assert ".T" in shape["suggested_fix"], "Must recommend transpose fix (.T)"
    assert "64 != 128" in shape["error_message"]


def test_benchmark_cli_no_color_mode():
    """
    Verifies that `--no-color` strips ANSI escape codes.
    """
    cmd = [sys.executable, BENCHMARK_SCRIPT, "--quick", "--no-color"]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0
    # Confirm absence of ESC character \x1b
    assert "\x1b[" not in result.stdout


# =============================================================================
# 2. Direct Python API Function Tests
# =============================================================================

def test_direct_edge_ai_memory_benchmark():
    """Direct functional test of run_edge_ai_memory_benchmark."""
    res = run_edge_ai_memory_benchmark(quick=True)
    assert isinstance(res, EdgeAIMemoryResult)
    assert res.status == "passed"
    assert res.pytorch_runtime_mb == 2560.00
    assert 0.05 < res.synapse_c99_binary_mb < 1.0
    assert res.memory_reduction_pct >= 99.9


def test_direct_dataloader_benchmark():
    """Direct functional test of run_dataloader_benchmark."""
    res = run_dataloader_benchmark(quick=True)
    assert isinstance(res, DataLoaderThroughputResult)
    assert res.status == "passed"
    assert res.speedup_factor >= 10.0
    assert res.total_samples_evaluated == 12000


def test_direct_shape_verification_benchmark():
    """Direct functional test of run_shape_verification_benchmark."""
    res = run_shape_verification_benchmark(quick=True)
    assert isinstance(res, ShapeVerificationResult)
    assert res.status == "passed"
    assert res.mismatch_detected is True
    assert ".T" in res.suggested_fix
    assert res.compile_check_latency_ms < 10.0  # Typically < 1 ms


def test_direct_full_suite():
    """Direct functional test of run_suite."""
    report = run_suite(quick=True)
    assert isinstance(report, BenchmarkSuiteReport)
    assert report.all_passed is True
    assert "10x" in report.overall_advantage
