"""
Tests for Synapse Zero-Crash Static Shape Invariant & Symbolic Solver Engine
=============================================================================
Phase 2: Verifies compile-time static contracts that eliminate runtime shape crashes
in deep learning pipelines:
1. SymbolicDimension unification and resolution.
2. SymbolicShapeSolver constraint solving and dimension binding.
3. Multi-Head Attention (MHA) 4D tensor geometry: Q @ K.T -> Attn @ V.
4. Automatic Transpose (.T) recommendation for inner-dimension mismatches.
5. Reshape element count conservation (e.g. [32, 512] -> [32, 8, 64]).
6. Detection of element count violations during reshape ([32, 512] -> [32, 8, 63]).
7. Deep learning multi-layer perceptron (MLP) forward-pass contracts.
8. CLI integration: `synapse verify-shapes` (stdout PASS/FAIL and --json output).
"""

import json
import os
import subprocess
import sys
import tempfile
import pytest

from synapse.analyzer.shape_guard import (
    SymbolicDimension,
    SymbolicShapeSolver,
    TensorShapeGuard,
    verify_shapes_in_file,
)
from synapse.analyzer.shape_checker import CompileTimeShapeMismatchError


# =============================================================================
# 1. Symbolic Dimension Representation & Unification
# =============================================================================

def test_symbolic_dimension_basics():
    d_sym = SymbolicDimension.from_val("B")
    assert not d_sym.is_concrete()
    assert d_sym.name == "B"
    assert str(d_sym) == "B"

    d_num = SymbolicDimension.from_val(128)
    assert d_num.is_concrete()
    assert d_num.value == 128
    assert str(d_num) == "128"

    d_str_num = SymbolicDimension.from_val("256")
    assert d_str_num.is_concrete()
    assert d_str_num.value == 256

    assert d_num == 128
    assert d_num == "128"
    assert d_sym == "B"
    assert d_sym == SymbolicDimension("B")


def test_symbolic_solver_bindings():
    solver = SymbolicShapeSolver()
    solver.bind("D", 512)
    solver.bind("H", 8)
    solver.bind("K", "D")

    assert solver.resolve("D") == 512
    assert solver.resolve("H") == 8
    assert solver.resolve("K") == 512
    assert solver.resolve("Unknown") == "Unknown"

    assert solver.are_compatible("D", 512)
    assert solver.are_compatible(512, "D")
    assert solver.are_compatible("B", "B")
    assert solver.are_compatible("-1", 128)  # Wildcard compatibility


def test_symbolic_solver_matmul_validation():
    solver = SymbolicShapeSolver()

    # 2D MatMul: [B, 128] @ [128, 64] -> [B, 64]
    valid, out_shape, err = solver.check_matmul(("B", 128), (128, 64))
    assert valid
    assert out_shape == ("B", 64)
    assert err == ""

    # Batched 3D MatMul: [B, S, 512] @ [B, 512, 256] -> [B, S, 256]
    valid, out_shape, err = solver.check_matmul(("B", "S", 512), ("B", 512, 256))
    assert valid
    assert out_shape == ("B", "S", 256)

    # Incompatible inner dimension
    valid, out_shape, err = solver.check_matmul(("B", 128), (256, 64))
    assert not valid
    assert "inner dimension mismatch" in err.lower()


# =============================================================================
# 2. Transformer Multi-Head Attention (MHA) Geometry Verification
# =============================================================================

def test_transformer_mha_geometry_valid():
    """
    Verifies that the entire 4D Multi-Head Attention forward equation:
    logits = (Q @ K.T)
    out = logits @ V
    statically passes without shape violations.
    """
    syn_code = """
let B: int = 16
let H: int = 8
let S: int = 128
let Dk: int = 64
let Dv: int = 64

let Q: Tensor[16, 8, 128, 64] = zeros([16, 8, 128, 64])
let K: Tensor[16, 8, 128, 64] = zeros([16, 8, 128, 64])
let V: Tensor[16, 8, 128, 64] = zeros([16, 8, 128, 64])

let K_T = K.T
let attn_weights = Q @ K_T
let context = attn_weights @ V
"""
    guard = TensorShapeGuard(source_text=syn_code, filename="transformer_mha.syn")
    errors = guard.verify_source()

    assert len(errors) == 0, f"Expected 0 errors in valid MHA geometry, got: {[e.message for e in errors]}"
    assert guard.env["K_T"] == (16, 8, 64, 128)
    assert guard.env["attn_weights"] == (16, 8, 128, 128)
    assert guard.env["context"] == (16, 8, 128, 64)


def test_transformer_mha_missing_transpose_suggested_fix():
    """
    Verifies that forgetting to transpose K (writing Q @ K instead of Q @ K.T)
    is caught at compile-time and suggests transposing K.
    """
    syn_code = """
let Q: Tensor[16, 8, 128, 64] = zeros([16, 8, 128, 64])
let K: Tensor[16, 8, 128, 64] = zeros([16, 8, 128, 64])

let broken_attn = Q @ K
"""
    guard = TensorShapeGuard(source_text=syn_code, filename="broken_mha.syn")
    errors = guard.verify_source()

    assert len(errors) == 1
    err = errors[0]
    assert "inner dimension mismatch" in err.message.lower()
    assert err.suggested_fix is not None
    assert ".T" in err.suggested_fix


# =============================================================================
# 3. Reshape Element Conservation
# =============================================================================

def test_reshape_element_conservation_valid():
    """32 * 512 == 32 * 8 * 64 (16,384 elements)."""
    syn_code = """
let x: Tensor[32, 512] = zeros([32, 512])
let reshaped = x.reshape([32, 8, 64])
"""
    guard = TensorShapeGuard(source_text=syn_code, filename="valid_reshape.syn")
    errors = guard.verify_source()
    assert len(errors) == 0
    assert guard.env["reshaped"] == (32, 8, 64)


def test_reshape_element_conservation_invalid():
    """32 * 512 (16,384) != 32 * 8 * 63 (16,128) -> Compile-time error."""
    syn_code = """
let x: Tensor[32, 512] = zeros([32, 512])
let bad_reshape = x.reshape([32, 8, 63])
"""
    guard = TensorShapeGuard(source_text=syn_code, filename="invalid_reshape.syn")
    errors = guard.verify_source()

    assert len(errors) == 1
    err = errors[0]
    assert "reshape element count mismatch" in err.message.lower()
    assert err.line >= 1
    assert "63" in str(err.actual)


# =============================================================================
# 4. Multi-Layer Perceptron (MLP) Deep Learning Pipeline
# =============================================================================

def test_mlp_pipeline_statically_safe():
    """Multi-layer forward contract: Input -> Linear1 -> Linear2 -> Output."""
    syn_code = """
fn forward(x: Tensor[32, 128]):
    let W1: Tensor[128, 256] = zeros([128, 256])
    let W2: Tensor[256, 10] = zeros([256, 10])

    let h1 = x @ W1
    let logits = h1 @ W2
    return logits
"""
    guard = TensorShapeGuard(source_text=syn_code, filename="mlp.syn")
    errors = guard.verify_source()
    assert len(errors) == 0


def test_mlp_pipeline_dimension_mismatch():
    """Passing [32, 128] into [256, 64] without alignment."""
    syn_code = """
fn forward(x: Tensor[32, 128]):
    let W_bad: Tensor[256, 64] = zeros([256, 64])
    let out = x @ W_bad
"""
    guard = TensorShapeGuard(source_text=syn_code, filename="bad_mlp.syn")
    errors = guard.verify_source()
    assert len(errors) == 1
    assert "inner dimension mismatch" in errors[0].message.lower()


# =============================================================================
# 5. CLI Command Integration (`synapse verify-shapes`)
# =============================================================================

def test_cli_verify_shapes_pass(tmp_path):
    code = """
let A: Tensor[64, 128] = zeros([64, 128])
let B: Tensor[128, 256] = zeros([128, 256])
let C = A @ B
"""
    syn_file = tmp_path / "valid_model.syn"
    syn_file.write_text(code, encoding="utf-8")

    cmd = [sys.executable, "-m", "synapse.cli", "verify-shapes", str(syn_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "[Shape Guard] PASS" in res.stdout


def test_cli_verify_shapes_fail(tmp_path):
    code = """
let A: Tensor[64, 128] = zeros([64, 128])
let B: Tensor[256, 64] = zeros([256, 64])
let C = A @ B
"""
    syn_file = tmp_path / "invalid_model.syn"
    syn_file.write_text(code, encoding="utf-8")

    cmd = [sys.executable, "-m", "synapse.cli", "verify-shapes", str(syn_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode != 0
    assert "[Shape Guard] FAILED" in res.stderr
    assert "inner dimension mismatch" in res.stderr.lower()


def test_cli_verify_shapes_json_output(tmp_path):
    code = """
let x: Tensor[32, 512] = zeros([32, 512])
let bad = x.reshape([32, 8, 63])
"""
    syn_file = tmp_path / "bad_reshape.syn"
    syn_file.write_text(code, encoding="utf-8")

    cmd = [sys.executable, "-m", "synapse.cli", "verify-shapes", "--json", str(syn_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode != 0
    data = json.loads(res.stdout)
    assert data["valid"] is False
    assert len(data["errors"]) >= 1
    assert "reshape element count mismatch" in data["errors"][0]["message"].lower()


def test_verify_shapes_in_file_helper(tmp_path):
    code = """
let M1: Tensor[10, 20] = zeros([10, 20])
let M2: Tensor[20, 30] = zeros([20, 30])
let M3 = M1 @ M2
"""
    syn_file = tmp_path / "helper_test.syn"
    syn_file.write_text(code, encoding="utf-8")

    is_valid, errs = verify_shapes_in_file(str(syn_file))
    assert is_valid
    assert len(errs) == 0
