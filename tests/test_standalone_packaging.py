"""
Tests for Synapse Standalone Binary Packaging and Embedded Runtime System
"""

import os
import sys
import shutil
import tempfile
import subprocess
import pytest

from synapse.codegen.embedded_runtime import (
    get_embedded_runtime_content,
    ensure_runtime_extracted,
)
from synapse.codegen.native_compiler import NativeCompiler


def test_embedded_runtime_content_validity():
    """Embedded runtime should decompress cleanly and contain valid C headers."""
    h_code, c_code = get_embedded_runtime_content()
    assert len(h_code) > 1000
    assert len(c_code) > 10000
    assert "syn_arena_t" in h_code
    assert "syn_tensor_create" in h_code
    assert "syn_arena_scope_enter" in h_code
    assert "SYN_THREAD_LOCAL" in h_code
    assert "syn_tensor_escape" in c_code


def test_embedded_runtime_extraction_to_temp():
    """ensure_runtime_extracted should create physical files in target directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_dir, c_path, h_path = ensure_runtime_extracted(target_dir=tmpdir)
        assert os.path.isdir(extracted_dir)
        assert os.path.isfile(c_path)
        assert os.path.isfile(h_path)
        with open(h_path, "r", encoding="utf-8") as f:
            assert "syn_tensor_t" in f.read()
        with open(c_path, "r", encoding="utf-8") as f:
            assert "syn_arena_create" in f.read()


def test_native_compiler_embedded_fallback():
    """NativeCompiler should find/extract runtime even if initialized without disk files."""
    compiler = NativeCompiler()
    assert os.path.isfile(compiler.runtime_c)
    assert os.path.isfile(compiler.runtime_h)
    assert os.path.isdir(compiler.runtime_dir)


def test_build_standalone_dry_run():
    """build_standalone.py script should execute successfully in dry-run mode."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "build_standalone.py")
    res = subprocess.run(
        [sys.executable, script_path, "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Dry-run complete" in res.stdout
    assert "Standalone Binary Builder" in res.stdout
