"""
Tests for Synapse Compile-Time Static Tensor Shape Checker & Type Theorist
==========================================================================
Verifies:
1. Valid matrix multiplications ([32, 64] @ [64, 128])
2. Invalid matrix multiplications ([32, 64] @ [128, 64] -> CompileTimeShapeMismatchError)
3. Transpose-corrected matrix multiplication ([32, 64] @ [128, 64].T)
4. Symbolic dimension matching ([B, S, D] x [D, H]) and symbolic mismatches
5. Broadcast addition ([32, 1] + [1, 64] -> [32, 64])
6. Incompatible broadcast errors ([32, 10] + [32, 20])
7. Visual error pointer, line, and column accuracy
8. Integration with TypeChecker and synapse.parser.ast re-export
"""

import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast import (
    Program, TensorType, VarDeclStmt, BinaryExpr, MemberExpr
)
import synapse.parser.ast as ast_module
import synapse.parser.ast_nodes as ast_nodes_module
from synapse.analyzer.shape_checker import (
    CompileTimeShapeMismatchError,
    StaticShapeChecker,
    check_shapes,
    verify_shapes,
)
from synapse.analyzer.type_checker import (
    TypeChecker,
    check_source as check_typed_source,
)


def parse_code(code: str) -> Program:
    tokens = Lexer(code).tokenize()
    return Parser(tokens).parse()


# =============================================================================
# 1. synapse.parser.ast Re-Export Verification
# =============================================================================

def test_ast_module_reexport():
    """Confirms synapse.parser.ast re-exports all nodes from ast_nodes."""
    assert hasattr(ast_module, "Program")
    assert hasattr(ast_module, "TensorType")
    assert hasattr(ast_module, "VarDeclStmt")
    assert hasattr(ast_module, "BinaryExpr")
    assert hasattr(ast_module, "MemberExpr")
    assert ast_module.Program is ast_nodes_module.Program
    assert ast_module.TensorType is ast_nodes_module.TensorType


# =============================================================================
# 2. Valid Matrix Multiplications ([32, 64] @ [64, 128])
# =============================================================================

def test_valid_matrix_multiplication():
    code = """
let A: Tensor[32, 64] = zeros([32, 64])
let B: Tensor[64, 128] = zeros([64, 128])
let C = A @ B
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0, f"Expected 0 errors, got: {[r.message for r in reports]}"
    assert checker.get_var_shape("C") == (32, 128)


def test_valid_batched_matrix_multiplication():
    code = """
let A: Tensor[16, 32, 64] = zeros([16, 32, 64])
let B: Tensor[16, 64, 128] = zeros([16, 64, 128])
let C = A @ B
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0
    assert checker.get_var_shape("C") == (16, 32, 128)


def test_valid_vector_matrix_multiplication():
    code = """
let v: Tensor[64] = zeros([64])
let M: Tensor[64, 128] = zeros([64, 128])
let out = v @ M
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0
    assert checker.get_var_shape("out") == (128,)


# =============================================================================
# 3. Invalid Matrix Multiplications ([32, 64] @ [128, 64])
# =============================================================================

def test_invalid_matrix_multiplication_raises_error():
    code = """let A: Tensor[32, 64] = zeros([32, 64])
let B: Tensor[128, 64] = zeros([128, 64])
let C = A @ B"""

    ast = parse_code(code)
    checker = StaticShapeChecker(source=code, raise_on_error=True)

    with pytest.raises(CompileTimeShapeMismatchError) as exc_info:
        checker.check(ast)

    err = exc_info.value
    assert err.expected == 64
    assert err.actual == 128
    assert "Inner dimensions must match: 64 != 128" in err.message
    assert err.line == 3
    assert err.column == 11
    assert "let C = A @ B" in err.source_line
    assert "^" in err.pointer


def test_invalid_matrix_multiplication_report_mode():
    code = """let A: Tensor[32, 64] = zeros([32, 64])
let B: Tensor[128, 64] = zeros([128, 64])
let C = A @ B"""

    reports = check_shapes(code, raise_on_error=False)
    assert len(reports) == 1
    rep = reports[0]
    assert rep.error_type == "CompileTimeShapeMismatchError"
    assert rep.line == 3
    assert rep.column == 11
    assert "64 != 128" in rep.message
    assert rep.suggested_fix is not None
    assert "B.T" in rep.suggested_fix


# =============================================================================
# 4. Transpose Corrected Matrix Multiplication ([32, 64] @ [128, 64].T)
# =============================================================================

def test_transpose_corrected_matrix_multiplication():
    code = """
let A: Tensor[32, 64] = zeros([32, 64])
let B: Tensor[128, 64] = zeros([128, 64])
let C = A @ B.T
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0, f"Expected 0 errors, got: {[r.message for r in reports]}"
    assert checker.get_var_shape("C") == (32, 128)


def test_transpose_left_corrected_matrix_multiplication():
    code = """
let A: Tensor[64, 32] = zeros([64, 32])
let B: Tensor[64, 128] = zeros([64, 128])
let C = A.T @ B
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0
    assert checker.get_var_shape("C") == (32, 128)


def test_transpose_1d_tensor_rejected():
    code = """
let v: Tensor[64] = zeros([64])
let invalid = v.T
"""
    with pytest.raises(CompileTimeShapeMismatchError) as exc_info:
        verify_shapes(code)

    assert "Transposition (.T) requires at least 2 dimensions" in exc_info.value.message


# =============================================================================
# 5. Symbolic Dimension Matching ([B, S, D] x [D, H])
# =============================================================================

def test_symbolic_dimension_matching():
    code = """
let X: Tensor[B, S, D] = zeros([B, S, D])
let W: Tensor[D, H] = zeros([D, H])
let Y = X @ W
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0, f"Expected 0 errors, got: {[r.message for r in reports]}"
    assert checker.get_var_shape("Y") == ("B", "S", "H")


def test_symbolic_dimension_function_contract():
    code = """
fn forward(x: Tensor[B, S, D], w: Tensor[D, H]) -> Tensor[B, S, H]:
    let out = x @ w
    return out
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0


def test_symbolic_dimension_mismatch_raises_error():
    code = """
let X: Tensor[B, S, D] = zeros([B, S, D])
let W: Tensor[K, H] = zeros([K, H])
let Y = X @ W
"""
    with pytest.raises(CompileTimeShapeMismatchError) as exc_info:
        verify_shapes(code)

    err = exc_info.value
    assert "Inner dimensions must match: D != K" in err.message
    assert err.expected == "D"
    assert err.actual == "K"


# =============================================================================
# 6. Broadcast Addition ([32, 1] + [1, 64] -> [32, 64])
# =============================================================================

def test_broadcast_addition():
    code = """
let A: Tensor[32, 1] = zeros([32, 1])
let B: Tensor[1, 64] = zeros([1, 64])
let C = A + B
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0, f"Expected 0 errors, got: {[r.message for r in reports]}"
    assert checker.get_var_shape("C") == (32, 64)


def test_broadcast_multiplication_and_subtraction():
    code = """
let X: Tensor[16, 1, 32] = zeros([16, 1, 32])
let Y: Tensor[1, 10, 32] = zeros([1, 10, 32])
let Z = X * Y
let W = Z - X
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0
    assert checker.get_var_shape("Z") == (16, 10, 32)
    assert checker.get_var_shape("W") == (16, 10, 32)


def test_broadcast_with_scalar():
    code = """
let A: Tensor[32, 64] = zeros([32, 64])
let B = A + 1.5
"""
    ast = parse_code(code)
    checker = StaticShapeChecker(source=code)
    reports = checker.check(ast)
    assert len(reports) == 0
    assert checker.get_var_shape("B") == (32, 64)


# =============================================================================
# 7. Incompatible Broadcast Error ([32, 10] + [32, 20])
# =============================================================================

def test_incompatible_broadcast_error():
    code = """let A: Tensor[32, 10] = zeros([32, 10])
let B: Tensor[32, 20] = zeros([32, 20])
let C = A + B"""

    with pytest.raises(CompileTimeShapeMismatchError) as exc_info:
        verify_shapes(code)

    err = exc_info.value
    assert "Shapes are not broadcast-compatible" in err.message
    assert err.line == 3
    assert err.column == 11
    assert err.actual == (32, 20)


def test_incompatible_broadcast_report_mode():
    code = """
let A: Tensor[32, 10] = zeros([32, 10])
let B: Tensor[32, 20] = zeros([32, 20])
let C = A * B
"""
    reports = check_shapes(code, raise_on_error=False)
    assert len(reports) == 1
    assert "Shapes are not broadcast-compatible" in reports[0].message
    assert reports[0].error_type == "CompileTimeShapeMismatchError"


# =============================================================================
# 8. Visual Error Pointer, Line and Column Accuracy
# =============================================================================

def test_visual_error_pointer_and_location():
    code = """let X: Tensor[32, 64] = zeros([32, 64])
let Y: Tensor[128, 64] = zeros([128, 64])
let Z = X @ Y"""

    try:
        verify_shapes(code)
        pytest.fail("Should have raised CompileTimeShapeMismatchError")
    except CompileTimeShapeMismatchError as err:
        assert err.line == 3
        assert err.column == 11
        assert err.source_line == "let Z = X @ Y"
        assert err.pointer == "          ^"
        assert err.ascii_pointer == "          ^"
        assert "\033[93m" in err.ansi_pointer

        # Validate formatted representation
        err_str = str(err)
        assert "CompileTimeShapeMismatchError" in err_str
        assert "Line 3, Column 11" in err_str
        assert "let Z = X @ Y" in err_str
        assert "^" in err_str
        assert "Expected shape / dimension: 64" in err_str
        assert "Actual shape / dimension:   128" in err_str
        assert "Suggested fix: let Z = X @ Y.T" in err_str


# =============================================================================
# 9. Type Contract Violation (Declared Shape vs Inferred Value Shape)
# =============================================================================

def test_declared_shape_contract_violation():
    code = """let A: Tensor[32, 64] = zeros([32, 128])"""

    with pytest.raises(CompileTimeShapeMismatchError) as exc_info:
        verify_shapes(code)

    err = exc_info.value
    assert "Type contract violation: Variable 'A' declared with contract" in err.message
    assert err.expected == (32, 64)
    assert err.actual == (32, 128)


# =============================================================================
# 10. Unified TypeChecker Integration
# =============================================================================

def test_unified_type_checker_valid():
    code = """
enum Mode: Training, Inference
let current_mode: Mode = Mode.Training

fn forward(x: Tensor[B, 64], w: Tensor[64, 128]) -> Tensor[B, 128]:
    return x @ w

let inp: Tensor[32, 64] = zeros([32, 64])
let weights: Tensor[64, 128] = zeros([64, 128])
let logits = forward(inp, weights)
"""
    result = check_typed_source(code)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_unified_type_checker_catches_shape_mismatch():
    code = """
let A: Tensor[32, 64] = zeros([32, 64])
let B: Tensor[128, 64] = zeros([128, 64])
let C = A @ B
"""
    result = check_typed_source(code)
    assert result.is_valid is False
    assert any("64 != 128" in e.message for e in result.errors)
