"""
Tests for benchmarks/run_all.py.

These check that the suite runs and that every number it reports comes from a
measurement (positive, finite, internally consistent). They intentionally do
not assert speed thresholds: results depend on the machine.
"""

import json
import math
import os
import subprocess
import sys

import pytest

from benchmarks.run_all import (
    SuiteReport,
    run_binary_size_benchmark,
    run_dataloader_benchmark,
    run_shape_check_benchmark,
    run_suite,
)
from synapse.codegen.native_compiler import NativeCompiler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_SCRIPT = os.path.join(PROJECT_ROOT, "benchmarks", "run_all.py")
HAS_C_COMPILER = NativeCompiler().find_c_compiler() is not None


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, BENCHMARK_SCRIPT, "--quick", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def test_cli_human_report():
    result = _run_cli("--no-color")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "All values measured on this machine" in out
    assert "1. Native binary size" in out
    assert "2. DataLoader throughput" in out
    assert "3. Static shape verification" in out
    assert "\033[" not in out


def test_cli_json_report():
    result = _run_cli("--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "success"
    assert data["quick_mode"] is True
    assert set(data["benchmarks"]) == {"binary_size", "dataloader", "shape_check"}


@pytest.mark.skipif(not HAS_C_COMPILER, reason="requires a C compiler")
def test_binary_size_is_measured_from_a_working_binary():
    res = run_binary_size_benchmark(quick=True)
    assert res.status == "measured", res.detail
    assert res.binary_bytes > 0
    assert res.runs_successfully is True
    assert res.binary_kb == round(res.binary_bytes / 1024, 1)


def test_dataloader_rates_are_real_measurements():
    res = run_dataloader_benchmark(quick=True)
    assert res.status == "measured"
    assert set(res.samples_per_sec) == {"numpy_slicing", "synapse_fast_dataloader", "multiprocessing_pickle_queue"}
    for name, rate in res.samples_per_sec.items():
        assert math.isfinite(rate) and rate > 0, name
    # Two runs on the same machine never produce bit-identical timings.
    again = run_dataloader_benchmark(quick=True)
    assert again.samples_per_sec != res.samples_per_sec


def test_shape_check_detects_mismatch_and_suggests_transpose():
    res = run_shape_check_benchmark(quick=True)
    assert res.status == "measured"
    assert res.mismatch_detected is True
    assert "64 != 128" in res.error_message
    assert ".T" in res.suggested_fix
    assert res.median_latency_ms > 0


def test_full_suite_report_round_trips_to_json():
    report = run_suite(quick=True)
    assert isinstance(report, SuiteReport)
    assert report.all_ok
    json.dumps(report.to_dict())
