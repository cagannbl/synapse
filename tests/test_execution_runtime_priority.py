"""
Tests for Synapse C99 AOT Priority and VM Execution Harmonization
"""

import os
import sys
import subprocess
import pytest
from synapse.codegen.native_compiler import NativeCompiler


def run_cli(*args):
    """Helper to invoke synapse CLI."""
    cmd = [sys.executable, "-m", "synapse.cli"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_cli_run_aot_priority_and_cache(tmp_path):
    """synapse run should compile and execute via C99 AOT by default."""
    compiler = NativeCompiler()
    c_info = compiler.find_c_compiler()
    if not c_info:
        pytest.skip("No native C compiler found in test environment")

    syn_file = str(tmp_path / "aot_test.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("let a = 10\nlet b = 20\nlet c = a + b\nprint(c)\n")

    # First run: compiles to .syn_cache/bin/
    res1 = run_cli("run", syn_file)
    assert res1.returncode == 0
    assert "30" in res1.stdout

    cache_bin_dir = os.path.join(str(tmp_path), ".syn_cache", "bin")
    assert os.path.isdir(cache_bin_dir)
    cached_exes = [f for f in os.listdir(cache_bin_dir) if f.endswith(".exe") or "." not in f]
    assert len(cached_exes) >= 1

    # Second run: runs from cached binary
    res2 = run_cli("run", syn_file)
    assert res2.returncode == 0
    assert "30" in res2.stdout


def test_cli_run_forced_vm_flag(tmp_path):
    """synapse run --vm should bypass C99 AOT and run on bytecode VM."""
    syn_file = str(tmp_path / "vm_test.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("let msg = \"Hello Bytecode VM\"\nprint(msg)\n")

    res = run_cli("run", syn_file, "--vm")
    assert res.returncode == 0
    assert "Hello Bytecode VM" in res.stdout

    # Verify no .syn_cache/bin/ was created
    cache_bin_dir = os.path.join(str(tmp_path), ".syn_cache", "bin")
    assert not os.path.exists(cache_bin_dir)


def test_cli_run_tensor_aot(tmp_path):
    """synapse run should execute tensor operations with C99 runtime."""
    compiler = NativeCompiler()
    if not compiler.find_c_compiler():
        pytest.skip("No native C compiler found in test environment")

    syn_file = str(tmp_path / "tensor_aot.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("let x = tensor([1.0, 2.0, 3.0])\nlet y = x + 1.0\nprint(y)\n")

    res = run_cli("run", syn_file)
    assert res.returncode == 0
    assert "[2" in res.stdout or "tensor" in res.stdout.lower() or "2.0" in res.stdout
