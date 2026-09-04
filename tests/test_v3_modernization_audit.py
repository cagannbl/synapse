"""
Regression and Modernization Audit Tests for Synapse (v3.0.0)
============================================================
Verifies the modernization of the 4 areas scored < 7.5:
1. Phase 1: C Runtime Thread-Local Arena & syn_graph_free autograd cleanup
2. Phase 2: MSVC ISO C99 Unnested TryExpr (?) without GNU statement expressions
3. Phase 3: Operator Precedence (@ in multiplicative) & Generalized Prompt Fields
4. Phase 4: Strict Null-Safety (Rejection of 'none' assignment to non-optional types)
"""

import os
import subprocess
import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import Program, VarDeclStmt, BinaryExpr, PromptDef
from synapse.core.type_checker import check_source, TypeContractViolationError
from synapse.codegen.c_emitter import CEmitter
from synapse.codegen.native_compiler import NativeCompiler


# =============================================================================
# Phase 1: C Runtime Thread-Local Arena & syn_graph_free
# =============================================================================

def test_runtime_header_has_syn_thread_local():
    """Confirms synapse_runtime.h declares SYN_THREAD_LOCAL and syn_graph_free."""
    runtime_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime"))
    header_path = os.path.join(runtime_dir, "synapse_runtime.h")
    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SYN_THREAD_LOCAL" in content
    assert "void syn_graph_free(syn_tensor_t* root);" in content


def test_runtime_c_has_syn_graph_free_and_thread_local():
    """Confirms synapse_runtime.c defines g_syn_active_arena as SYN_THREAD_LOCAL and implements syn_graph_free."""
    runtime_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime"))
    c_path = os.path.join(runtime_dir, "synapse_runtime.c")
    with open(c_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SYN_THREAD_LOCAL syn_arena_t* g_syn_active_arena = NULL;" in content
    assert "void syn_graph_free(syn_tensor_t* root) {" in content


def test_compile_and_run_autograd_graph_free(tmp_path):
    """Compile and run C code testing syn_graph_free cleans up intermediate nodes cleanly."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available")

    c_source = """
#include <stdio.h>
#include <assert.h>
#include "synapse_runtime.h"

int main() {
    int shape[2] = {2, 2};
    double d1[4] = {1.0, 2.0, 3.0, 4.0};
    double d2[4] = {2.0, 0.0, 1.0, 2.0};

    syn_tensor_t* a = syn_tensor_create(d1, shape, 2, 1);
    syn_tensor_t* b = syn_tensor_create(d2, shape, 2, 0);

    /* Build intermediate computation graph: c = a @ b, loss = sum(c) */
    syn_tensor_t* c = syn_matmul(a, b);
    syn_tensor_t* loss = syn_sum(c);

    syn_backward(loss);

    assert(a->grad != NULL);
    assert(loss->grad != NULL);

    /* Free intermediate nodes using syn_graph_free */
    syn_graph_free(loss);

    syn_tensor_free(a);
    syn_tensor_free(b);
    printf("GRAPH_FREE_SUCCESS\\n");
    return 0;
}
"""
    test_c = tmp_path / "test_graph_free.c"
    test_exe = tmp_path / ("test_graph_free.exe" if os.name == "nt" else "test_graph_free")

    with open(test_c, "w", encoding="utf-8") as f:
        f.write(c_source)

    ok, res = nc.build_executable(str(test_c), str(test_exe))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "GRAPH_FREE_SUCCESS" in proc.stdout


# =============================================================================
# Phase 2: MSVC ISO C99 Unnested TryExpr (?)
# =============================================================================

def test_c_emitter_try_expr_produces_no_gnu_statement_expressions():
    """Verify CEmitter produces unnested ISO C99 code for TryExpr without '({' GNU extensions."""
    source = """
fn parse_number(flag: int) -> Result[int, str]:
    if flag > 0:
        return Ok(100)
    return Err("bad flag")

fn run_pipeline(flag: int) -> Result[int, str]:
    let val: int = parse_number(flag)?
    return Ok(val + 5)
"""
    emitter = CEmitter()
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    c_code = emitter.emit(ast)

    # Must NOT contain GNU statement expression pattern "({"
    assert "({\n" not in c_code
    assert "({ " not in c_code
    assert "SYN_RES_ERR" in c_code
    assert "val = _try_res" in c_code or "_try_res_1.as.ok" in c_code


# =============================================================================
# Phase 3: Operator Precedence & Generalized Prompt Fields
# =============================================================================

def test_operator_precedence_multiplicative_matmul():
    """Confirms @ operator has left-associative multiplicative precedence matching PEP 465."""
    source = "let res = A * B @ C"
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    stmt = ast.statements[0]
    assert isinstance(stmt, VarDeclStmt)
    expr = stmt.value
    assert isinstance(expr, BinaryExpr)
    # Left-associative: (A * B) @ C
    assert expr.op == "@"
    assert isinstance(expr.left, BinaryExpr)
    assert expr.left.op == "*"


def test_generalized_prompt_field_parsing():
    """Verifies prompt blocks accept arbitrary model fields (e.g. top_p, reasoning_effort)."""
    source = """
prompt reasoning_eval(query: str) -> str:
    system: "You are a reasoner."
    user: query
    temperature: 0.2
    top_p: 0.95
    reasoning_effort: 0.8
"""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    assert len(ast.statements) == 1
    prompt_stmt = ast.statements[0]
    assert isinstance(prompt_stmt, PromptDef)
    assert "top_p" in prompt_stmt.fields
    assert "reasoning_effort" in prompt_stmt.fields


# =============================================================================
# Phase 4: Strict Null-Safety
# =============================================================================

def test_strict_null_safety_rejects_none_on_primitive():
    """Confirms assigning 'none' to a non-optional primitive type is strictly rejected."""
    source = "let x: int = none"
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert "Cannot assign 'none' to non-optional type 'int'" in res.errors[0].message
    assert "Option[int]" in res.errors[0].message


def test_strict_null_safety_rejects_reassigning_none_to_primitive():
    """Confirms reassigning 'none' to an existing typed variable is strictly rejected."""
    source = """
let x: int = 42
x = none
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Cannot assign 'none' to non-optional type 'int'" in err.message for err in res.errors)


def test_strict_null_safety_permits_none_on_option():
    """Confirms assigning 'none' or None to an Option[T] is valid."""
    source = """
let o1: Option[int] = none
let o2: Option[str] = None
"""
    res = check_source(source)
    assert res.is_valid is True
    assert len(res.errors) == 0
