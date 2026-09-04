"""
Tests for Python Compatibility Mode, Migration CLI, and Dynamic Stubs
======================================================================
Verifies:
1. DynamicStubGenerator reflection on standard modules (math, json).
2. Synapse stub header generation (.syn format).
3. LSP completion integration for dynamic Python stubs.
4. CLI `synapse migrate <file.py> -o <out.syn>`.
5. CLI `synapse migrate <file.py> --diff`.
6. Direct Python execution via CLI `synapse run <file.py>`.
7. Direct Python execution via CLI `synapse run --compat=python <file.py>`.
8. CLI `synapse kernel install`.
"""

import json
import os
import subprocess
import sys
import tempfile
import pytest

from synapse.interop.stubs import DynamicStubGenerator
from synapse.lsp.features import get_completions


# =============================================================================
# 1. Dynamic Stubs & LSP Reflection
# =============================================================================

def test_dynamic_stub_generator_inspection():
    stubs = DynamicStubGenerator.get_module_stubs("math")
    assert stubs["module"] == "math"
    assert "sqrt" in stubs["members"]
    assert "sin" in stubs["members"]
    assert "pi" in stubs["members"]

    sqrt_item = stubs["members"]["sqrt"]
    assert sqrt_item["kind"] == "function"
    assert "sqrt" in sqrt_item["synapse_decl"]


def test_dynamic_stub_header_generation():
    header = DynamicStubGenerator.generate_synapse_stubs("math")
    assert "// Synapse Type Stubs for Python module 'math'" in header
    assert "fn sqrt" in header
    assert "const pi" in header


def test_dynamic_stub_lsp_completions():
    completions = DynamicStubGenerator.get_completions_for_lsp("math")
    labels = [c["label"] for c in completions]
    assert "sqrt" in labels
    assert "sin" in labels
    assert "cos" in labels


def test_lsp_features_integration_with_python_dot():
    # When user types `py.math.` in IDE
    items = get_completions(trigger_char=".", line_prefix="py.math.")
    labels = [item["label"] for item in items]
    assert "sqrt" in labels
    assert "sin" in labels


# =============================================================================
# 2. CLI Migration (`synapse migrate`)
# =============================================================================

def test_cli_migrate_file(tmp_path):
    py_code = """
import numpy as np

def add_tensors(a, b):
    return a + b

X = np.zeros((32, 64))
Y = np.ones((32, 64))
Z = np.dot(X, Y.T)
"""
    py_file = tmp_path / "model.py"
    py_file.write_text(py_code, encoding="utf-8")

    out_syn = tmp_path / "model.syn"
    cmd = [sys.executable, "-m", "synapse.cli", "migrate", str(py_file), "-o", str(out_syn)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert out_syn.exists()
    syn_content = out_syn.read_text(encoding="utf-8")
    assert "fn add_tensors" in syn_content
    assert "zeros(" in syn_content
    assert "@" in syn_content


def test_cli_migrate_diff(tmp_path):
    py_code = """
def multiply(x, y):
    return x * y
"""
    py_file = tmp_path / "calc.py"
    py_file.write_text(py_code, encoding="utf-8")

    cmd = [sys.executable, "-m", "synapse.cli", "migrate", str(py_file), "--diff"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "---" in res.stdout
    assert "+++" in res.stdout
    assert "-def multiply" in res.stdout
    assert "+fn multiply" in res.stdout


# =============================================================================
# 3. Direct Python Execution (`synapse run --compat=python` & `.py` files)
# =============================================================================

def test_cli_run_direct_python_file(tmp_path):
    py_code = """
def compute(val):
    return val * 2

result = compute(21)
print("Computed:", result)
"""
    py_file = tmp_path / "script.py"
    py_file.write_text(py_code, encoding="utf-8")

    # Run directly without flags, just .py extension
    cmd = [sys.executable, "-m", "synapse.cli", "run", str(py_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "Computed: 42" in res.stdout


def test_cli_run_python_compat_flag(tmp_path):
    py_code = """
def greet(name):
    return "Hello " + name

msg = greet("Synapse")
print(msg)
"""
    py_file = tmp_path / "legacy.txt"
    py_file.write_text(py_code, encoding="utf-8")

    # Pass --compat=python
    cmd = [sys.executable, "-m", "synapse.cli", "run", "--compat=python", str(py_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "Hello Synapse" in res.stdout


# =============================================================================
# 4. CLI Kernel Installation (`synapse kernel install`)
# =============================================================================

def test_cli_kernel_install(tmp_path):
    kernel_dir = tmp_path / "test_kernel_dir"
    cmd = [sys.executable, "-m", "synapse.cli", "kernel", "install", "--dir", str(kernel_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "[Kernel]" in res.stdout
    assert (kernel_dir / "kernel.json").exists()
