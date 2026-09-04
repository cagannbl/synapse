import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import BinaryExpr, IdentifierExpr, ExprStmt


def parse_expr(source: str):
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    prog = parser.parse()
    assert len(prog.statements) == 1
    assert isinstance(prog.statements[0], ExprStmt)
    return prog.statements[0].expr


def test_matmul_star_left_associative():
    # A * B @ C must be parsed as: ((A * B) @ C)
    expr = parse_expr("A * B @ C\n")
    assert isinstance(expr, BinaryExpr)
    assert expr.op == "@"
    assert isinstance(expr.right, IdentifierExpr)
    assert expr.right.name == "C"
    assert isinstance(expr.left, BinaryExpr)
    assert expr.left.op == "*"
    assert expr.left.left.name == "A"
    assert expr.left.right.name == "B"


def test_star_matmul_left_associative():
    # A @ B * C must be parsed as: ((A @ B) * C)
    expr = parse_expr("A @ B * C\n")
    assert isinstance(expr, BinaryExpr)
    assert expr.op == "*"
    assert isinstance(expr.right, IdentifierExpr)
    assert expr.right.name == "C"
    assert isinstance(expr.left, BinaryExpr)
    assert expr.left.op == "@"
    assert expr.left.left.name == "A"
    assert expr.left.right.name == "B"


def test_additive_and_matmul_precedence():
    # A + B @ C * D must be parsed as: A + ((B @ C) * D)
    expr = parse_expr("A + B @ C * D\n")
    assert isinstance(expr, BinaryExpr)
    assert expr.op == "+"
    assert expr.left.name == "A"
    assert isinstance(expr.right, BinaryExpr)
    assert expr.right.op == "*"
    assert isinstance(expr.right.left, BinaryExpr)
    assert expr.right.left.op == "@"
    assert expr.right.left.left.name == "B"
    assert expr.right.left.right.name == "C"
    assert expr.right.right.name == "D"
