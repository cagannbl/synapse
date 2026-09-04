import pytest
from synapse.lexer.token import TokenType, Token
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import (
    Program, VarDeclStmt, MatchStmt, MatchCase, TryExpr, CallExpr,
    MemberExpr, IdentifierExpr, LiteralExpr, AgentDef, PromptDef,
    ToolDef, TensorLiteralExpr, ReturnStmt
)


def parse_code(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def test_soft_keywords_variable_declarations():
    # 1. let tensor = 100;
    # 2. let agent = "agent_007";
    # 3. let tool = true;
    # 4. let prompt = "translate";
    # 5. let grad = 0.01;
    source = """
let tensor = 100;
let agent = "agent_007";
let tool = true;
let prompt = "translate";
let grad = 0.01;
"""
    ast = parse_code(source)
    assert len(ast.statements) == 5

    s0 = ast.statements[0]
    assert isinstance(s0, VarDeclStmt)
    assert s0.name == "tensor"
    assert isinstance(s0.value, LiteralExpr)
    assert s0.value.value == 100

    s1 = ast.statements[1]
    assert isinstance(s1, VarDeclStmt)
    assert s1.name == "agent"
    assert isinstance(s1.value, LiteralExpr)
    assert s1.value.value == "agent_007"

    s2 = ast.statements[2]
    assert isinstance(s2, VarDeclStmt)
    assert s2.name == "tool"
    assert isinstance(s2.value, LiteralExpr)
    assert s2.value.value is True

    s3 = ast.statements[3]
    assert isinstance(s3, VarDeclStmt)
    assert s3.name == "prompt"
    assert s3.value.value == "translate"

    s4 = ast.statements[4]
    assert isinstance(s4, VarDeclStmt)
    assert s4.name == "grad"
    assert s4.value.value == 0.01


def test_pattern_matching_match_stmt():
    source = """
match opt:
    case Option.Some(x):
        return x
    case Option.None:
        return 0
"""
    ast = parse_code(source)
    assert len(ast.statements) == 1
    m = ast.statements[0]
    assert isinstance(m, MatchStmt)
    assert isinstance(m.subject, IdentifierExpr)
    assert m.subject.name == "opt"
    assert len(m.cases) == 2

    case0 = m.cases[0]
    assert isinstance(case0, MatchCase)
    assert isinstance(case0.pattern, CallExpr)
    assert isinstance(case0.pattern.callee, MemberExpr)
    assert case0.pattern.callee.member == "Some"
    assert len(case0.pattern.args) == 1
    assert case0.pattern.args[0].name == "x"
    assert len(case0.body) == 1
    assert isinstance(case0.body[0], ReturnStmt)
    assert case0.body[0].value.name == "x"

    case1 = m.cases[1]
    assert isinstance(case1, MatchCase)
    assert isinstance(case1.pattern, MemberExpr)
    assert case1.pattern.member == "None"
    assert len(case1.body) == 1
    assert isinstance(case1.body[0], ReturnStmt)
    assert case1.body[0].value.value == 0


def test_try_expr_question_mark_operator():
    source = "let val = get_res()?"
    ast = parse_code(source)
    assert len(ast.statements) == 1
    stmt = ast.statements[0]
    assert isinstance(stmt, VarDeclStmt)
    assert stmt.name == "val"
    assert isinstance(stmt.value, TryExpr)
    assert isinstance(stmt.value.expr, CallExpr)
    assert isinstance(stmt.value.expr.callee, IdentifierExpr)
    assert stmt.value.expr.callee.name == "get_res"


def test_try_expr_chaining():
    source = "let item = fetch_data()?.process()?"
    ast = parse_code(source)
    assert len(ast.statements) == 1
    stmt = ast.statements[0]
    assert isinstance(stmt.value, TryExpr)
    call = stmt.value.expr
    assert isinstance(call, CallExpr)
    assert isinstance(call.callee, MemberExpr)
    assert call.callee.member == "process"
    inner_try = call.callee.target
    assert isinstance(inner_try, TryExpr)
    assert isinstance(inner_try.expr, CallExpr)
    assert inner_try.expr.callee.name == "fetch_data"


def test_agent_block_definition():
    # Agent with indentation block
    src1 = """
agent ResearchBot:
    model: "claude-3-7-sonnet"
    memory: true
"""
    ast1 = parse_code(src1)
    assert len(ast1.statements) == 1
    a1 = ast1.statements[0]
    assert isinstance(a1, AgentDef)
    assert a1.name == "ResearchBot"
    assert "model" in a1.fields
    assert "memory" in a1.fields

    # Agent with braced block
    src2 = """
agent Assistant {
    model: "gpt-4o",
    instructions: "Help the user"
}
"""
    ast2 = parse_code(src2)
    assert len(ast2.statements) == 1
    a2 = ast2.statements[0]
    assert isinstance(a2, AgentDef)
    assert a2.name == "Assistant"
    assert "model" in a2.fields
    assert "instructions" in a2.fields


def test_prompt_and_tool_blocks():
    src = """
prompt classify(text):
    system: "Classify sentiment"
    user: text

tool search(query: str) -> str:
    let q = query
    return q
"""
    ast = parse_code(src)
    assert len(ast.statements) == 2
    p = ast.statements[0]
    assert isinstance(p, PromptDef)
    assert p.name == "classify"

    t = ast.statements[1]
    assert isinstance(t, ToolDef)
    assert t.name == "search"
    assert len(t.body) == 2


def test_tensor_constructor_and_type_annotations():
    src = """
let t1: Tensor = tensor([1, 2, 3]);
let t2: tensor = tensor([[1, 2], [3, 4]], requires_grad=true);
let t3: tensor[2, 3] = tensor([[1, 2, 3], [4, 5, 6]]);
"""
    ast = parse_code(src)
    assert len(ast.statements) == 3

    assert isinstance(ast.statements[0].value, TensorLiteralExpr)
    assert isinstance(ast.statements[1].value, TensorLiteralExpr)
    assert "requires_grad" in ast.statements[1].value.kwargs
    assert isinstance(ast.statements[2].value, TensorLiteralExpr)

    assert ast.statements[0].tensor_type is not None
    assert ast.statements[1].tensor_type is not None
    assert ast.statements[2].tensor_type is not None
    assert ast.statements[2].tensor_type.dims == (2, 3)


def test_match_case_with_guard_and_literals():
    src = """
match status:
    case 200:
        return "OK"
    case 404:
        return "Not Found"
    case code if code >= 500:
        return "Server Error"
"""
    ast = parse_code(src)
    assert len(ast.statements) == 1
    m = ast.statements[0]
    assert isinstance(m, MatchStmt)
    assert len(m.cases) == 3
    assert isinstance(m.cases[0].pattern, LiteralExpr)
    assert m.cases[0].pattern.value == 200
    assert m.cases[2].guard is not None
