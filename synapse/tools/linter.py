"""
Synapse Linter and Static Analysis Tool.
Enforces Synapse idioms, detects AI/Python syntax drift, finds dead/unreachable code,
and warns about missing tensor shape contracts.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.parser.ast_nodes import (
    ASTNode, Program, Stmt, Expr,
    VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, BreakStmt, ContinueStmt, PassStmt,
    IfStmt, WhileStmt, ForStmt, FunctionDef, ToolDef, AgentDef, PromptDef, EnumDeclStmt, ImportStmt,
    LiteralExpr, IdentifierExpr, BinaryExpr, UnaryExpr, CallExpr, MemberExpr, IndexExpr,
    PipeExpr, ListLiteralExpr, DictLiteralExpr, TensorLiteralExpr,
    TensorType, TypeAnnotation
)
from synapse.core.diagnostics import fix_ai_drift


@dataclass
class LintDiagnostic:
    """Represents a single linter diagnostic finding."""
    code: str
    message: str
    line: int
    column: int
    severity: str = "warning"  # "warning", "error", "info"
    filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "filename": self.filename,
        }

    def __str__(self) -> str:
        loc = f"{self.filename}:{self.line}:{self.column}" if self.filename else f"{self.line}:{self.column}"
        return f"{loc}: [{self.code}] {self.severity.upper()}: {self.message}"


class _Scope:
    """Tracks declared and referenced variables within a lexical scope."""
    def __init__(self, parent: Optional["_Scope"] = None, is_function: bool = False):
        self.parent = parent
        self.is_function = is_function
        # name -> VarDeclStmt (or ASTNode)
        self.declared_vars: dict[str, ASTNode] = {}
        # set of names read/referenced
        self.read_vars: set[str] = set()
        # all known names (including params, loop targets, function names)
        self.known_symbols: set[str] = set()

    def declare(self, name: str, node: ASTNode, is_let_const: bool = True):
        self.known_symbols.add(name)
        if is_let_const:
            self.declared_vars[name] = node

    def is_known(self, name: str) -> bool:
        if name in self.known_symbols:
            return True
        if self.parent:
            return self.parent.is_known(name)
        return False

    def mark_read(self, name: str):
        self.read_vars.add(name)
        # Propagate upward if declared in enclosing scopes
        curr = self.parent
        while curr is not None:
            if name in curr.declared_vars or name in curr.known_symbols:
                curr.read_vars.add(name)
            curr = curr.parent


class SynapseLinter:
    """
    Synapse Static Code Analyzer and Linter.

    Rules:
    - SYN001: Tanımlanıp kullanılmayan değişkenler (Unused variable)
    - SYN002: Ulaşılamaz kod (Unreachable code after return/break/continue)
    - SYN003: Python sözdizimi sapması (def yerine fn, eksik let)
    - SYN004: Eksik tensör şekil bildirimi uyarısı (Missing tensor shape contract)
    """

    BUILTINS = {
        "print", "len", "range", "tensor", "zeros", "ones", "randn", "eye",
        "arange", "full", "assert", "assert_eq", "Some", "None", "Result",
        "Ok", "Err", "Linear", "Sequential", "ReLU", "Sigmoid", "Tanh", "Dropout",
        "BatchNorm", "Conv2d", "MaxPool2d", "MSELoss", "CrossEntropyLoss",
        "Adam", "SGD", "RMSprop", "Module", "int", "float", "str", "bool", "list", "dict"
    }

    def lint_file(self, filepath: str) -> list[LintDiagnostic]:
        """Reads and lints a file from disk."""
        if not os.path.isfile(filepath):
            return [
                LintDiagnostic(
                    code="SYN999",
                    message=f"File not found: '{filepath}'",
                    line=1,
                    column=1,
                    severity="error",
                    filename=filepath
                )
            ]
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        return self.lint_source(source, filename=filepath)

    def lint_source(self, source: str, filename: str = "") -> list[LintDiagnostic]:
        """Lints Synapse source code string and returns a list of diagnostics."""
        diagnostics: list[LintDiagnostic] = []

        # 1. Pre-pass: Check for Python syntax divergence (SYN003) at source level
        has_python_divergence = self._check_source_python_drift(source, filename, diagnostics)

        # 2. Parse AST (use normalized source if Python syntax drift was detected to allow remaining checks)
        ast: Optional[Program] = None
        try:
            tokens = Lexer(source).tokenize()
            ast = Parser(tokens).parse()
        except (LexerError, ParseError) as err:
            if has_python_divergence:
                # Try parsing normalized code so SYN001/SYN002/SYN004 can still be evaluated
                try:
                    fixed_code, _, _ = fix_ai_drift(source, filename=filename or "source.syn")
                    norm_tokens = Lexer(fixed_code).tokenize()
                    ast = Parser(norm_tokens).parse()
                except Exception:
                    ast = None
            else:
                diagnostics.append(
                    LintDiagnostic(
                        code="SYN999",
                        message=str(err),
                        line=getattr(err, "line", 1),
                        column=getattr(err, "column", 1),
                        severity="error",
                        filename=filename
                    )
                )

        # 3. If AST is available, perform AST-level inspections
        if ast:
            self._check_ast_rules(ast, filename, diagnostics)

        # 4. Sort and deduplicate diagnostics by (line, column, code)
        seen = set()
        unique_diagnostics: list[LintDiagnostic] = []
        diagnostics.sort(key=lambda d: (d.line, d.column, d.code, d.message))
        for d in diagnostics:
            key = (d.code, d.line, d.column, d.message)
            if key not in seen:
                seen.add(key)
                unique_diagnostics.append(d)

        return unique_diagnostics

    @classmethod
    def format_report(cls, diagnostics: list[LintDiagnostic], as_json: bool = False) -> str:
        """Formats diagnostics as human-readable report or JSON."""
        if as_json:
            data = [d.to_dict() for d in diagnostics]
            return json.dumps(data, indent=2, ensure_ascii=False)

        if not diagnostics:
            return "No lint issues found."

        lines = []
        errors = 0
        warnings = 0
        infos = 0

        for d in diagnostics:
            if d.severity == "error":
                errors += 1
            elif d.severity == "warning":
                warnings += 1
            else:
                infos += 1
            lines.append(str(d))

        summary = f"\nFound {len(diagnostics)} issue{'s' if len(diagnostics) != 1 else ''} ({errors} error{'s' if errors != 1 else ''}, {warnings} warning{'s' if warnings != 1 else ''})."
        lines.append(summary)
        return "\n".join(lines)

    # =========================================================================
    # Rule SYN003: Python Syntax Divergence (def yerine fn, eksik let)
    # =========================================================================

    def _check_source_python_drift(self, source: str, filename: str, diagnostics: list[LintDiagnostic]) -> bool:
        """Scans raw source lines for Python-specific patterns."""
        has_drift = False
        lines = source.splitlines()

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Check 1: 'def func_name(' -> 'fn' should be used
            def_match = re.match(r"^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if def_match:
                has_drift = True
                col = len(def_match.group(1)) + 1
                fn_name = def_match.group(2)
                diagnostics.append(
                    LintDiagnostic(
                        code="SYN003",
                        message=f"Python syntax divergence: use 'fn' instead of 'def' for function '{fn_name}'",
                        line=idx,
                        column=col,
                        severity="error",
                        filename=filename
                    )
                )

            # Check 2: 'import numpy', 'import torch', etc.
            import_match = re.match(r"^\s*(import\s+(numpy|torch|tensorflow|scipy|sklearn)|from\s+(numpy|torch|tensorflow|scipy|sklearn)\s+import)", line)
            if import_match:
                has_drift = True
                col = len(line) - len(line.lstrip()) + 1
                diagnostics.append(
                    LintDiagnostic(
                        code="SYN003",
                        message="Python syntax divergence: external deep learning imports are unnecessary in Synapse",
                        line=idx,
                        column=col,
                        severity="warning",
                        filename=filename
                    )
                )

        return has_drift

    # =========================================================================
    # AST Rules: SYN001 (Unused), SYN002 (Unreachable), SYN003 (Missing let), SYN004 (Tensor Shape)
    # =========================================================================

    def _check_ast_rules(self, program: Program, filename: str, diagnostics: list[LintDiagnostic]):
        """Traverses the AST to check for unused variables, unreachable code, missing let, and shapes."""
        # 1. Unreachable code analysis (SYN002)
        self._check_unreachable_in_stmts(program.statements, filename, diagnostics)

        # 2. Scope & variable usage analysis (SYN001, SYN003, SYN004)
        module_scope = _Scope(parent=None, is_function=False)
        for sym in self.BUILTINS:
            module_scope.known_symbols.add(sym)

        self._analyze_stmts_in_scope(program.statements, module_scope, filename, diagnostics)

        # Check module-level unused variables
        self._finalize_scope(module_scope, filename, diagnostics)

    def _check_unreachable_in_stmts(self, statements: list[Stmt], filename: str, diagnostics: list[LintDiagnostic]):
        """Recursively inspects statement lists for unreachable statements after return/break/continue."""
        unreachable = False
        term_keyword = ""

        for stmt in statements:
            if unreachable:
                diagnostics.append(
                    LintDiagnostic(
                        code="SYN002",
                        message=f"Unreachable code detected after '{term_keyword}'",
                        line=stmt.line,
                        column=stmt.column,
                        severity="warning",
                        filename=filename
                    )
                )
            else:
                if isinstance(stmt, ReturnStmt):
                    unreachable = True
                    term_keyword = "return"
                elif isinstance(stmt, BreakStmt):
                    unreachable = True
                    term_keyword = "break"
                elif isinstance(stmt, ContinueStmt):
                    unreachable = True
                    term_keyword = "continue"

            # Check nested blocks
            if isinstance(stmt, FunctionDef):
                self._check_unreachable_in_stmts(stmt.body, filename, diagnostics)
            elif isinstance(stmt, ToolDef):
                self._check_unreachable_in_stmts(stmt.body, filename, diagnostics)
            elif isinstance(stmt, WhileStmt):
                self._check_unreachable_in_stmts(stmt.body, filename, diagnostics)
            elif isinstance(stmt, ForStmt):
                self._check_unreachable_in_stmts(stmt.body, filename, diagnostics)
            elif isinstance(stmt, IfStmt):
                self._check_unreachable_in_stmts(stmt.then_branch, filename, diagnostics)
                for _, branch in stmt.elif_branches:
                    self._check_unreachable_in_stmts(branch, filename, diagnostics)
                if stmt.else_branch:
                    self._check_unreachable_in_stmts(stmt.else_branch, filename, diagnostics)

    def _analyze_stmts_in_scope(
        self,
        statements: list[Stmt],
        scope: _Scope,
        filename: str,
        diagnostics: list[LintDiagnostic]
    ):
        """Sequential walk through statements to track declarations, assignments, and expressions."""
        for stmt in statements:
            if isinstance(stmt, VarDeclStmt):
                # Check SYN004: Missing tensor shape annotation
                self._check_tensor_shape_decl(stmt, filename, diagnostics)

                # Right-hand value is read
                self._visit_expr(stmt.value, scope)
                # Register declaration in scope
                scope.declare(stmt.name, stmt, is_let_const=True)

            elif isinstance(stmt, AssignStmt):
                # Right-hand value is read
                self._visit_expr(stmt.value, scope)

                # Target inspection
                if isinstance(stmt.target, IdentifierExpr):
                    var_name = stmt.target.name
                    if stmt.op != "=":
                        # Reassignment like += reads the variable
                        scope.mark_read(var_name)
                    else:
                        # Direct assignment =
                        if not scope.is_known(var_name):
                            # SYN003: Missing let or const keyword
                            diagnostics.append(
                                LintDiagnostic(
                                    code="SYN003",
                                    message=f"Python syntax divergence: missing 'let' or 'const' keyword for variable '{var_name}'",
                                    line=stmt.line,
                                    column=stmt.column,
                                    severity="error",
                                    filename=filename
                                )
                            )
                            # Register it so we don't duplicate warning
                            scope.declare(var_name, stmt, is_let_const=False)
                elif isinstance(stmt.target, (MemberExpr, IndexExpr)):
                    # Assigning to attribute or index (e.g. obj.x = 1, arr[0] = 1) reads target
                    self._visit_expr(stmt.target, scope)

            elif isinstance(stmt, ExprStmt):
                self._visit_expr(stmt.expr, scope)

            elif isinstance(stmt, ReturnStmt):
                if stmt.value:
                    self._visit_expr(stmt.value, scope)

            elif isinstance(stmt, FunctionDef):
                # Function name is known in current scope
                scope.known_symbols.add(stmt.name)

                # Check SYN004 for params or return type
                self._check_tensor_shape_function(stmt, filename, diagnostics)

                # Create function scope
                fn_scope = _Scope(parent=scope, is_function=True)
                for param in stmt.params:
                    fn_scope.declare(param.name, param, is_let_const=False)

                self._analyze_stmts_in_scope(stmt.body, fn_scope, filename, diagnostics)
                self._finalize_scope(fn_scope, filename, diagnostics)

            elif isinstance(stmt, ToolDef):
                scope.known_symbols.add(stmt.name)
                tool_scope = _Scope(parent=scope, is_function=True)
                for param in stmt.params:
                    tool_scope.declare(param.name, param, is_let_const=False)
                self._analyze_stmts_in_scope(stmt.body, tool_scope, filename, diagnostics)
                self._finalize_scope(tool_scope, filename, diagnostics)

            elif isinstance(stmt, IfStmt):
                self._visit_expr(stmt.condition, scope)
                self._analyze_stmts_in_scope(stmt.then_branch, scope, filename, diagnostics)
                for cond, branch in stmt.elif_branches:
                    self._visit_expr(cond, scope)
                    self._analyze_stmts_in_scope(branch, scope, filename, diagnostics)
                if stmt.else_branch:
                    self._analyze_stmts_in_scope(stmt.else_branch, scope, filename, diagnostics)

            elif isinstance(stmt, WhileStmt):
                self._visit_expr(stmt.condition, scope)
                self._analyze_stmts_in_scope(stmt.body, scope, filename, diagnostics)

            elif isinstance(stmt, ForStmt):
                self._visit_expr(stmt.iterable, scope)
                scope.declare(stmt.target, stmt, is_let_const=False)
                self._analyze_stmts_in_scope(stmt.body, scope, filename, diagnostics)

            elif isinstance(stmt, ImportStmt):
                if stmt.alias:
                    scope.known_symbols.add(stmt.alias)
                elif stmt.module_path:
                    scope.known_symbols.add(stmt.module_path[-1])

            elif isinstance(stmt, EnumDeclStmt):
                scope.known_symbols.add(stmt.name)

            elif isinstance(stmt, (PromptDef, AgentDef)):
                scope.known_symbols.add(stmt.name)
                for f_expr in stmt.fields.values():
                    self._visit_expr(f_expr, scope)

    def _visit_expr(self, expr: Expr, scope: _Scope):
        """Visits expression nodes to track variable reads."""
        if expr is None:
            return

        if isinstance(expr, IdentifierExpr):
            scope.mark_read(expr.name)

        elif isinstance(expr, BinaryExpr):
            self._visit_expr(expr.left, scope)
            self._visit_expr(expr.right, scope)

        elif isinstance(expr, UnaryExpr):
            self._visit_expr(expr.operand, scope)

        elif isinstance(expr, CallExpr):
            self._visit_expr(expr.callee, scope)
            for arg in expr.args:
                self._visit_expr(arg, scope)
            for kw_val in expr.kwargs.values():
                self._visit_expr(kw_val, scope)

        elif isinstance(expr, MemberExpr):
            self._visit_expr(expr.target, scope)

        elif isinstance(expr, IndexExpr):
            self._visit_expr(expr.target, scope)
            self._visit_expr(expr.index, scope)

        elif isinstance(expr, PipeExpr):
            self._visit_expr(expr.left, scope)
            self._visit_expr(expr.right, scope)

        elif isinstance(expr, ListLiteralExpr):
            for elem in expr.elements:
                self._visit_expr(elem, scope)

        elif isinstance(expr, DictLiteralExpr):
            for k, v in expr.entries:
                self._visit_expr(k, scope)
                self._visit_expr(v, scope)

        elif isinstance(expr, TensorLiteralExpr):
            self._visit_expr(expr.data, scope)
            for kw_val in expr.kwargs.values():
                self._visit_expr(kw_val, scope)

    def _finalize_scope(self, scope: _Scope, filename: str, diagnostics: list[LintDiagnostic]):
        """Checks for declared variables that were never read (SYN001)."""
        for name, node in scope.declared_vars.items():
            # Leading underscore ignores unused check (e.g. '_temp', '_')
            if name.startswith("_"):
                continue
            if name not in scope.read_vars:
                diagnostics.append(
                    LintDiagnostic(
                        code="SYN001",
                        message=f"Variable '{name}' is declared but never used",
                        line=node.line,
                        column=node.column,
                        severity="warning",
                        filename=filename
                    )
                )

    # =========================================================================
    # Rule SYN004: Missing Tensor Shape Annotation
    # =========================================================================

    def _is_tensor_producing_expr(self, expr: Expr) -> bool:
        """Determines whether an expression creates/initializes a Tensor."""
        if isinstance(expr, TensorLiteralExpr):
            return True
        if isinstance(expr, CallExpr):
            if isinstance(expr.callee, IdentifierExpr):
                return expr.callee.name in {"tensor", "zeros", "ones", "randn", "eye", "arange", "full"}
        return False

    def _check_tensor_shape_decl(self, stmt: VarDeclStmt, filename: str, diagnostics: list[LintDiagnostic]):
        """Warns if a tensor variable declaration is missing shape dimensions."""
        is_tensor_expr = self._is_tensor_producing_expr(stmt.value)
        has_shape = (stmt.tensor_type is not None and len(stmt.tensor_type.dims) > 0)

        # Check raw type annotation if string was passed
        if not has_shape and stmt.type_annot:
            annot_str = str(stmt.type_annot).strip()
            if annot_str.startswith("Tensor[") and "]" in annot_str:
                inner = annot_str[7:annot_str.find("]")].strip()
                if inner:
                    has_shape = True

        if not has_shape and (is_tensor_expr or (stmt.type_annot and str(stmt.type_annot).strip() == "Tensor")):
            if stmt.type_annot:
                msg = f"Missing tensor shape dimensions for '{stmt.name}' (e.g. 'let {stmt.name}: Tensor[...] = ...')"
            else:
                msg = f"Missing tensor shape annotation for '{stmt.name}' (e.g. 'let {stmt.name}: Tensor[...] = ...')"

            diagnostics.append(
                LintDiagnostic(
                    code="SYN004",
                    message=msg,
                    line=stmt.line,
                    column=stmt.column,
                    severity="warning",
                    filename=filename
                )
            )

    def _check_tensor_shape_function(self, stmt: FunctionDef, filename: str, diagnostics: list[LintDiagnostic]):
        """Checks for missing tensor shape contracts in function signature."""
        if stmt.return_type:
            ret_str = str(stmt.return_type).strip()
            if ret_str == "Tensor":
                diagnostics.append(
                    LintDiagnostic(
                        code="SYN004",
                        message=f"Missing tensor shape dimensions for return type of function '{stmt.name}' (consider '-> Tensor[...]')",
                        line=stmt.line,
                        column=stmt.column,
                        severity="warning",
                        filename=filename
                    )
                )

        for p in stmt.params:
            if p.type_annot:
                p_str = str(p.type_annot).strip()
                if p_str == "Tensor":
                    diagnostics.append(
                        LintDiagnostic(
                            code="SYN004",
                            message=f"Missing tensor shape dimensions for parameter '{p.name}' in function '{stmt.name}' (consider 'Tensor[...]')",
                            line=p.line,
                            column=p.column,
                            severity="warning",
                            filename=filename
                        )
                    )
