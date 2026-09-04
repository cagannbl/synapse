"""
Synapse Native Model Context Protocol (MCP) Server
Allows AI coding assistants (Claude Desktop, Cursor, Antigravity, custom agents)
to validate, execute, fix, and transpile Synapse code over stdio.
"""

import sys
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

from mcp.server.fastmcp import FastMCP

from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError
from synapse.core.diagnostics import diagnose_code, fix_ai_drift, _check_static_tensor_shapes
from synapse.codegen.native_compiler import NativeCompiler


SYSTEM_PROMPT_DENSE = """You are an expert compiler and engineer for Synapse (.syn), an AI-native programming language.
Rules:
1. Python-like indentation (4 spaces). NO semicolons (;). NO braces ({}).
2. Variables: Always declare with 'let' or 'const'. Example: 'let x = 10', 'const MAX = 100'.
3. Functions: Always declare with 'fn', NEVER 'def'. Example: 'fn add(a, b): return a + b'.
4. Tensors: Native primitives, ZERO external imports (NEVER import numpy or torch).
   - Creation: tensor([[1.0, 2.0], [3.0, 4.0]]), zeros([2, 2]), randn([3, 3])
   - Matmul: Matrix multiplication MUST use '@' (e.g., let C = A @ B).
   - Autograd: let w = tensor(0.5, requires_grad=true); loss.backward(); w.grad
5. Deep Learning: Sequential([Linear(in, out), ReLU()]), Adam(params, lr=0.01), MSELoss()
6. AI Prompts & Agents:
   prompt summarize(text: str) -> str:
       system: "Summarize concisely."
       user: text
   agent ResearchBot:
       model: "gpt-4o"
       instructions: "Research topics."
7. Pipeline: Data chaining using '|>'. Example: 'data |> clean |> predict'
8. Python FFI: 'import py.math as math'
"""


def create_mcp_server() -> FastMCP:
    server = FastMCP(
        name="synapse-mcp",
        instructions="Synapse AI Language Server providing validation, sandbox execution, AI drift repair, and C transpilation."
    )

    @server.tool()
    def validate_synapse_code(code: str) -> str:
        """
        Validates Synapse source code syntax, AST, and tensor shapes.
        Returns a structured JSON report with error location, suggested fix, AI prompt hint, and unified diff if invalid.
        """
        report = diagnose_code(code, filepath="<mcp_snippet>")
        return report.to_json(indent=2)

    @server.tool()
    def execute_synapse(code: str) -> str:
        """
        Executes Synapse code in a sandboxed virtual machine and returns captured stdout and returned values.
        If an error occurs, returns machine-readable error diagnostics with a suggested fix and diff.
        """
        buf = io.StringIO()
        err_buf = io.StringIO()

        try:
            tokens = Lexer(code).tokenize()
            ast = Parser(tokens).parse()
            compiled = Compiler(name="<mcp_sandbox>").compile(ast)
            vm = VirtualMachine()

            with redirect_stdout(buf), redirect_stderr(err_buf):
                result = vm.execute(compiled)

            return json.dumps({
                "status": "success",
                "stdout": buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "result": str(result) if result is not None else None
            }, indent=2)

        except Exception as e:
            report = diagnose_code(code, filepath="<mcp_sandbox>", error=e)
            return json.dumps({
                "status": "error",
                "stdout": buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "diagnostic": report.to_dict()
            }, indent=2)

    @server.tool()
    def fix_synapse_code(code: str) -> str:
        """
        Automatically cleans and repairs Python drift in Synapse code (e.g. converting 'def' to 'fn',
        adding missing 'let' declarations, removing redundant numpy/torch imports, and fixing colons).
        Returns the fixed code, a unified diff patch, and a list of modifications.
        """
        fixed_code, diff, changes = fix_ai_drift(code)
        # Doğrula
        report = diagnose_code(fixed_code, filepath="<repaired>")
        return json.dumps({
            "fixed_code": fixed_code,
            "diff": diff,
            "changes_applied": changes,
            "validation_after_fix": report.status,
            "diagnostic_message": report.message
        }, indent=2)

    @server.tool()
    def transpile_to_c(code: str) -> str:
        """
        Transpiles high-level Synapse code into pure, zero-dependency C99 code with standalone tensor runtime.
        """
        try:
            nc = NativeCompiler()
            c_code = nc.transpile(code)
            return json.dumps({
                "status": "success",
                "c_code": c_code
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            }, indent=2)

    @server.tool()
    def inspect_tensor_shapes(code: str) -> str:
        """
        Performs static shape analysis on tensor operations (@, +, -, *, /) in Synapse code.
        Catches matrix dimension mismatches before execution.
        """
        try:
            tokens = Lexer(code).tokenize()
            ast = Parser(tokens).parse()
            report = _check_static_tensor_shapes(ast, code, "<inspect>")
            if report is not None:
                return json.dumps({
                    "status": "shape_mismatch",
                    "report": report.to_dict()
                }, indent=2)
            return json.dumps({
                "status": "ok",
                "message": "All static tensor dimensions align correctly."
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "parse_failed",
                "error": str(e)
            }, indent=2)

    @server.tool()
    def get_ai_system_prompt() -> str:
        """
        Returns the ultra-dense token-optimized system prompt (<200 tokens) for LLMs to generate 100% valid Synapse code.
        """
        return SYSTEM_PROMPT_DENSE.strip()

    return server


def main():
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
