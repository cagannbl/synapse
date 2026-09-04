"""
Tests for Synapse AI-Tolerant Parsing, Self-Healing Diff, and Python Drift Normalization
"""

import os
import tempfile
from synapse.core.diagnostics import fix_ai_drift, diagnose_code, generate_unified_diff
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def test_fix_ai_drift_converts_def_to_fn():
    code = """def add(a, b):
    return a + b
"""
    fixed, diff, changes = fix_ai_drift(code)
    assert "fn add(a, b):" in fixed
    assert "def add" not in fixed
    assert len(changes) == 1
    assert "Converted 'def' to 'fn'" in changes[0]
    assert "--- a/source.syn" in diff
    assert "+++ b/source.syn" in diff


def test_fix_ai_drift_adds_let_declarations():
    code = """x = 10
y = 20
let total = x + y
"""
    fixed, diff, changes = fix_ai_drift(code)
    assert "let x = 10" in fixed
    assert "let y = 20" in fixed
    assert "let total = x + y" in fixed


def test_fix_ai_drift_removes_numpy_and_torch_imports():
    code = """import numpy as np
import torch
let A = np.array([[1.0, 2.0], [3.0, 4.0]])
let B = np.zeros([2, 2])
"""
    fixed, diff, changes = fix_ai_drift(code)
    assert "import numpy" not in fixed
    assert "import torch" not in fixed
    assert "let A = tensor([[1.0, 2.0], [3.0, 4.0]])" in fixed
    assert "let B = zeros([2, 2])" in fixed


def test_fix_ai_drift_appends_missing_block_colons():
    code = """fn compute(x)
    return x * 2
"""
    fixed, diff, changes = fix_ai_drift(code)
    assert "fn compute(x):" in fixed


def test_diagnose_code_includes_diff_on_syntax_error():
    # Kodda '!' var (Synapse'de '!=' olmalı)
    code = """let x = 10
if x ! 5:
    print(x)
"""
    report = diagnose_code(code, filepath="test_err.syn")
    assert report.status == "error"
    assert report.error_type == "LexerError"
    assert report.suggested_fix is not None
    assert report.ai_prompt_hint is not None


def test_ai_tolerant_execution_flow():
    """Python drift içeren bir kod parçasının fix_ai_drift sonrasında başarıyla VM'de çalışması."""
    drift_code = """import numpy as np
def square_and_add(a, b):
    val = (a + b) ** 2
    return val

let res = square_and_add(2.0, 3.0)
print("Result:", res)
"""
    fixed_code, _, _ = fix_ai_drift(drift_code)

    tokens = Lexer(fixed_code).tokenize()
    ast = Parser(tokens).parse()
    code_obj = Compiler(name="test_flow").compile(ast)
    vm = VirtualMachine()
    vm.set_custom_print()
    vm.execute(code_obj)

    assert any("Result: 25.0" in line for line in vm.output_buffer)


def test_tensor_type_ast_nodes():
    from synapse.parser.ast_nodes import TensorType, ShapeAnnotation, TypeAnnotation, VarDeclStmt, Param

    tt = TensorType((32, 64))
    assert tt.to_shape_tuple() == (32, 64)
    assert str(tt) == "Tensor[32, 64]"
    assert tt.is_static() is True
    assert tt.is_symbolic() is False

    tt_sym = TensorType(("B", 128))
    assert tt_sym.to_shape_tuple() == ("B", 128)
    assert tt_sym.is_symbolic() is True
    assert tt_sym.matches_shape((32, 128)) is True
    assert tt_sym.matches_shape((32, 64)) is False

    tt_dtype = TensorType((32, 64), dtype="float16")
    assert "dtype='float16'" in str(tt_dtype)

    parsed_tt = TensorType.from_string("Tensor[64, 128]")
    assert parsed_tt is not None
    assert parsed_tt.to_shape_tuple() == (64, 128)

    type_annot = TypeAnnotation("Tensor[16, 32]")
    assert type_annot.tensor_type is not None
    assert type_annot.to_shape_tuple() == (16, 32)

    var_decl = VarDeclStmt(name="W", type_annot="Tensor[32, 64]")
    assert var_decl.tensor_type is not None
    assert var_decl.to_shape_tuple() == (32, 64)

    param = Param(name="x", type_annot="Tensor[32, 64]")
    assert param.tensor_type is not None
    assert param.to_shape_tuple() == (32, 64)


def test_static_tensor_contract_violation():
    # Değişken Tensor[32, 64] olarak tanımlanmış fakat zeros([10, 20]) atanmış
    code = "let A: Tensor[32, 64] = zeros([10, 20])"
    report = diagnose_code(code, filepath="contract_test.syn")
    assert report.status == "error"
    assert report.error_type == "TypeContractViolationError"
    assert "Variable 'A'" in report.message
    assert report.details["expected_shape"] == [32, 64]
    assert report.details["actual_shape"] == [10, 20]


def test_static_tensor_contract_valid():
    code = "let A: Tensor[32, 64] = zeros([32, 64])"
    report = diagnose_code(code, filepath="contract_valid.syn")
    assert report.status == "ok"


def test_static_tensor_matmul_mismatch_and_suggested_fix():
    code = """let A = zeros([32, 64])
let B = zeros([128, 64])
let C = A @ B
"""
    report = diagnose_code(code, filepath="matmul_test.syn")
    assert report.status == "error"
    assert report.error_type == "TensorShapeMismatchError"
    assert "Cannot multiply tensor of shape (32, 64)" in report.message
    assert report.suggested_fix == "let C = A @ B.T"
    assert "let C = A @ B.T" in report.auto_fixed_code
    assert report.diff is not None
    assert "+let C = A @ B.T" in report.diff


def test_static_tensor_addition_broadcasting():
    # Uyumsuz yayınlama şekilleri: (32, 64) ve (32, 48)
    incompat_code = """let A = zeros([32, 64])
let B = zeros([32, 48])
let C = A + B
"""
    report = diagnose_code(incompat_code, filepath="add_test.syn")
    assert report.status == "error"
    assert report.error_type == "TensorShapeMismatchError"
    assert "broadcast-compatible" in report.message

    # Uyumlu yayınlama: (32, 64) ve (1, 64)
    compat_code = """let A = zeros([32, 64])
let B = zeros([1, 64])
let C = A + B
"""
    report_ok = diagnose_code(compat_code, filepath="add_compat.syn")
    assert report_ok.status == "ok"


def test_static_transposition_dimension_check():
    # 1D tensör transpoze edilemez
    code = """let A = zeros([32])
let B = A.T
"""
    report = diagnose_code(code, filepath="transpose_test.syn")
    assert report.status == "error"
    assert report.error_type == "TensorShapeMismatchError"
    assert "requires at least 2 dimensions" in report.message


def test_fix_ai_drift_transposition_healing():
    drift_code = """import torch
let X = zeros([32, 64])
let W = zeros([128, 64])
let Y = X @ W
"""
    fixed, diff, changes = fix_ai_drift(drift_code)
    assert "import torch" not in fixed
    assert "let Y = X @ W.T" in fixed
    assert any("transposed right operand 'W' to 'W.T'" in c for c in changes)
    assert "+let Y = X @ W.T" in diff


def test_heal_and_execute_full_flow():
    from synapse.core.diagnostics import heal_and_execute

    # Python sözdizimi sapmaları + ters matris çarpımı içeren LLM kodu
    ai_code = """import numpy as np
def predict(x, w):
    val = x @ w
    return val

let x = ones([2, 4])
let w = ones([8, 4])
let out = predict(x, w.T)
print("Healed execution success!")
"""
    vm = VirtualMachine()
    vm.set_custom_print()
    result, report = heal_and_execute(ai_code, vm=vm, filename="ai_healed.syn")

    assert report.status == "ok"
    assert report.diff is not None
    assert report.auto_fixed_code is not None
    assert any("Healed execution success!" in line for line in vm.output_buffer)

