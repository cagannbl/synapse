import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import (
    Program, VarDeclStmt, AssignStmt, FunctionDef, BinaryExpr, PipeExpr,
    TensorLiteralExpr, PromptDef, IfStmt, WhileStmt, ForStmt, ImportStmt, CallExpr
)


def parse_code(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def test_parse_variable_decl():
    ast = parse_code("let lr = 0.001\nconst BATCH = 32")
    assert len(ast.statements) == 2
    assert isinstance(ast.statements[0], VarDeclStmt)
    assert ast.statements[0].name == "lr"
    assert ast.statements[0].is_const is False

    assert isinstance(ast.statements[1], VarDeclStmt)
    assert ast.statements[1].name == "BATCH"
    assert ast.statements[1].is_const is True


def test_parse_matrix_multiply_and_pipe():
    ast = parse_code("let result = (X @ W + B) |> relu")
    assert len(ast.statements) == 1
    decl = ast.statements[0]
    assert isinstance(decl, VarDeclStmt)
    assert isinstance(decl.value, PipeExpr)

    # Sol taraf (X @ W + B)
    left_bin = decl.value.left
    assert isinstance(left_bin, BinaryExpr)
    assert left_bin.op == "+"
    assert isinstance(left_bin.left, BinaryExpr)
    assert left_bin.left.op == "@"


def test_parse_function_def():
    source = """
fn train_step(w, x, y):
    let pred = x @ w
    let loss = (pred - y) * 2.0
    return loss
"""
    ast = parse_code(source)
    assert len(ast.statements) == 1
    fn = ast.statements[0]
    assert isinstance(fn, FunctionDef)
    assert fn.name == "train_step"
    assert len(fn.params) == 3
    assert len(fn.body) == 3


def test_parse_tensor_literal():
    source = "let A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=true)"
    ast = parse_code(source)
    assert len(ast.statements) == 1
    decl = ast.statements[0]
    assert isinstance(decl.value, TensorLiteralExpr)
    assert "requires_grad" in decl.value.kwargs


def test_parse_prompt_definition():
    source = """
prompt summarize(doc_text: str) -> str:
    system: "Summarize the text in 3 bullet points."
    user: doc_text
    temperature: 0.2
"""
    ast = parse_code(source)
    assert len(ast.statements) == 1
    p = ast.statements[0]
    assert isinstance(p, PromptDef)
    assert p.name == "summarize"
    assert len(p.params) == 1
    assert "system" in p.fields
    assert "user" in p.fields
    assert "temperature" in p.fields


def test_parse_control_flow():
    source = """
if x > 0:
    let y = 1
elif x == 0:
    let y = 0
else:
    let y = -1
"""
    ast = parse_code(source)
    assert len(ast.statements) == 1
    if_stmt = ast.statements[0]
    assert isinstance(if_stmt, IfStmt)
    assert len(if_stmt.then_branch) == 1
    assert len(if_stmt.elif_branches) == 1
    assert len(if_stmt.else_branch) == 1


def test_parse_python_import():
    source = "import py.numpy as np\nimport py.torch"
    ast = parse_code(source)
    assert len(ast.statements) == 2
    imp1 = ast.statements[0]
    assert isinstance(imp1, ImportStmt)
    assert imp1.is_python is True
    assert imp1.module_path == ["numpy"]
    assert imp1.alias == "np"

    imp2 = ast.statements[1]
    assert imp2.is_python is True
    assert imp2.module_path == ["torch"]
    assert imp2.alias is None
