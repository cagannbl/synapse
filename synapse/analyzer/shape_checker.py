"""
Synapse Compile-Time Static Tensor Shape Checker & Type Theorist
================================================================
Provides static analysis for tensor shape contracts, symbolic dimension
verification, matrix multiplication compatibility, broadcasting rules,
and transposition algebra during the compilation phase.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Union

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast import (
    ASTNode, Program, Stmt, Expr, LiteralExpr, IdentifierExpr,
    BinaryExpr, UnaryExpr, CallExpr, MemberExpr, ListLiteralExpr,
    TensorLiteralExpr, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, FunctionDef, Param, TensorType
)
from synapse.core.diagnostics import DiagnosticReport


class CompileTimeShapeMismatchError(Exception):
    """
    Raised at compile-time when tensor shape constraints, contracts,
    or operations (MatMul, broadcasting, transpose) are violated.
    Includes exact line, column, expected vs actual shapes, and an ASCII/ANSI visual pointer.
    """

    def __init__(
        self,
        message: str,
        line: int = 1,
        column: int = 1,
        expected: Any = None,
        actual: Any = None,
        source_line: str = "",
        pointer: str = "",
        suggested_fix: Optional[str] = None,
    ):
        self.message = message
        self.line = max(1, line)
        self.column = max(1, column)
        self.expected = expected
        self.actual = actual
        self.source_line = source_line
        self.suggested_fix = suggested_fix

        col_idx = max(0, self.column - 1)
        self.ascii_pointer = pointer if pointer else " " * col_idx + "^"
        self.ansi_pointer = f"\033[93m{self.ascii_pointer}\033[0m"
        self.pointer = self.ascii_pointer

        formatted = self._format_error()
        super().__init__(formatted)

    def _format_error(self) -> str:
        lines = [
            f"CompileTimeShapeMismatchError: {self.message}",
            f"  --> Line {self.line}, Column {self.column}",
        ]
        if self.source_line:
            col_idx = max(0, self.column - 1)
            lines.append("    |")
            lines.append(f"{self.line:3d} | {self.source_line}")
            lines.append(f"    | {' ' * col_idx}^")
        if self.expected is not None:
            lines.append(f"  Expected shape / dimension: {self.expected}")
        if self.actual is not None:
            lines.append(f"  Actual shape / dimension:   {self.actual}")
        if self.suggested_fix:
            lines.append(f"  Suggested fix: {self.suggested_fix}")
        return "\n".join(lines)

    def to_diagnostic_report(self, filepath: str = "<source>") -> DiagnosticReport:
        return DiagnosticReport(
            status="error",
            error_type="CompileTimeShapeMismatchError",
            message=self.message,
            line=self.line,
            column=self.column,
            source_line=self.source_line,
            pointer=self.pointer,
            suggested_fix=self.suggested_fix,
            ai_prompt_hint="Tensor shape mismatch at compile time. Verify matrix dimensions, transpose (.T), or broadcasting rules.",
            details={
                "filepath": filepath,
                "expected": str(self.expected) if self.expected is not None else None,
                "actual": str(self.actual) if self.actual is not None else None,
            },
        )


def _extract_source_line(source: str, line_no: int) -> str:
    lines = source.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def _generate_pointer(col_no: int) -> str:
    col_idx = max(0, col_no - 1)
    return (" " * col_idx) + "^"


def _infer_list_literal_shape(elem: Any) -> Optional[tuple[int, ...]]:
    if isinstance(elem, ListLiteralExpr):
        if not elem.elements:
            return (0,)
        sub_shapes = [_infer_list_literal_shape(sub) for sub in elem.elements]
        if any(s is None for s in sub_shapes):
            return None
        first = sub_shapes[0]
        for s in sub_shapes[1:]:
            if s != first:
                return None
        return (len(elem.elements),) + first
    elif isinstance(elem, LiteralExpr) and isinstance(elem.value, (int, float)):
        return ()
    return None


class StaticShapeChecker:
    """
    Compile-time static tensor shape verifier and type theorist.
    Performs static checking over the AST:
    - Verifies tensor declarations and symbolic contracts (Tensor[32, 64], Tensor[B, S, D])
    - Enforces MatMul inner dimension constraints: [..., M, K] x [..., K, N] -> [..., M, N]
    - Verifies broadcasting rules (+, -, *, /)
    - Validates transposition algebra (.T swaps last two dimensions)
    - Detects inverted dimensions and provides intelligent suggested fixes
    """

    def __init__(
        self,
        source: str = "",
        filepath: str = "<source>",
        raise_on_error: bool = False,
    ):
        self.source = source
        self.filepath = filepath
        self.raise_on_error = raise_on_error

        # Scope stack: var_name -> shape tuple
        self.scopes: list[dict[str, tuple[Union[int, str], ...]]] = [{}]
        # Contract stack: var_name -> TensorType
        self.contracts: list[dict[str, TensorType]] = [{}]
        # Function registry: name -> (param_shapes, return_shape)
        self.functions: dict[
            str,
            tuple[
                list[tuple[str, Optional[tuple[Union[int, str], ...]]]],
                Optional[tuple[Union[int, str], ...]],
            ],
        ] = {}
        # Current function return shape if in function body
        self.current_fn_name: Optional[str] = None
        self.current_fn_return_shape: Optional[tuple[Union[int, str], ...]] = None

        # Symbolic dimension bindings (e.g. 'B' -> 16)
        self.symbol_bindings: dict[str, Union[int, str]] = {}

        # Diagnostics collected
        self.diagnostics: list[DiagnosticReport] = []

    # -------------------------------------------------------------------------
    # Scope Management
    # -------------------------------------------------------------------------
    def push_scope(self):
        self.scopes.append({})
        self.contracts.append({})

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
            self.contracts.pop()

    def set_var_shape(
        self,
        name: str,
        shape: tuple[Union[int, str], ...],
        contract: Optional[TensorType] = None,
    ):
        self.scopes[-1][name] = shape
        if contract is not None:
            self.contracts[-1][name] = contract

    def get_var_shape(self, name: str) -> Optional[tuple[Union[int, str], ...]]:
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        return None

    def get_var_contract(self, name: str) -> Optional[TensorType]:
        for c in reversed(self.contracts):
            if name in c:
                return c[name]
        return None

    # -------------------------------------------------------------------------
    # Error Handling & Diagnostics
    # -------------------------------------------------------------------------
    def _handle_error(
        self,
        message: str,
        node: ASTNode,
        expected: Any = None,
        actual: Any = None,
        suggested_fix: Optional[str] = None,
    ):
        line = getattr(node, "line", 1) or 1
        col = getattr(node, "column", 1) or 1
        source_line = _extract_source_line(self.source, line)

        # Refine column for operators if needed
        if col <= 1 and source_line:
            if isinstance(node, BinaryExpr):
                op = node.op
                if op in source_line:
                    col = source_line.find(op) + 1

        pointer = _generate_pointer(col)

        err = CompileTimeShapeMismatchError(
            message=message,
            line=line,
            column=col,
            expected=expected,
            actual=actual,
            source_line=source_line,
            pointer=pointer,
            suggested_fix=suggested_fix,
        )

        report = DiagnosticReport(
            status="error",
            error_type="CompileTimeShapeMismatchError",
            message=message,
            line=line,
            column=col,
            source_line=source_line,
            pointer=pointer,
            suggested_fix=suggested_fix,
            ai_prompt_hint="Tensor shape mismatch at compile time. Verify matrix dimensions, transpose (.T), or broadcasting rules.",
            details={
                "filepath": self.filepath,
                "expected": str(expected) if expected is not None else None,
                "actual": str(actual) if actual is not None else None,
            },
        )
        self.diagnostics.append(report)

        if self.raise_on_error:
            raise err

    # -------------------------------------------------------------------------
    # Dimension & Broadcasting Utilities
    # -------------------------------------------------------------------------
    def _dim_equals(self, d1: Union[int, str], d2: Union[int, str]) -> bool:
        """Compares two dimensions (concrete integer or symbolic identifier)."""
        if d1 == d2:
            return True
        b1 = self.symbol_bindings.get(d1, d1) if isinstance(d1, str) else d1
        b2 = self.symbol_bindings.get(d2, d2) if isinstance(d2, str) else d2
        if b1 == b2:
            return True
        return False

    def _broadcast_shapes_tuple(
        self,
        s1: tuple[Union[int, str], ...],
        s2: tuple[Union[int, str], ...],
    ) -> Optional[tuple[Union[int, str], ...]]:
        """
        Broadcasting logic following NumPy/PyTorch standards with symbolic dimension support.
        Returns the unified broadcasted shape, or None if incompatible.
        """
        res: list[Union[int, str]] = []
        r1 = list(reversed(s1))
        r2 = list(reversed(s2))
        max_len = max(len(r1), len(r2))

        for i in range(max_len):
            d1 = r1[i] if i < len(r1) else 1
            d2 = r2[i] if i < len(r2) else 1

            if d1 == d2:
                res.append(d1)
            elif d1 == 1:
                res.append(d2)
            elif d2 == 1:
                res.append(d1)
            elif isinstance(d1, str) and isinstance(d2, str):
                b1 = self.symbol_bindings.get(d1, d1)
                b2 = self.symbol_bindings.get(d2, d2)
                if b1 == b2:
                    res.append(b1)
                elif b1 == 1:
                    res.append(b2)
                elif b2 == 1:
                    res.append(b1)
                else:
                    return None
            elif isinstance(d1, str) and isinstance(d2, int):
                b1 = self.symbol_bindings.get(d1)
                if b1 == d2:
                    res.append(d2)
                elif b1 == 1:
                    res.append(d2)
                elif d2 == 1:
                    res.append(d1)
                else:
                    return None
            elif isinstance(d2, str) and isinstance(d1, int):
                b2 = self.symbol_bindings.get(d2)
                if b2 == d1:
                    res.append(d1)
                elif b2 == 1:
                    res.append(d1)
                elif d1 == 1:
                    res.append(d2)
                else:
                    return None
            else:
                return None

        return tuple(reversed(res))

    def _suggest_matmul_fix(
        self,
        s1: tuple[Union[int, str], ...],
        s2: tuple[Union[int, str], ...],
        node: ASTNode,
    ) -> Optional[str]:
        """Provides an intelligent suggested fix when transposition aligns inner dimensions."""
        line = getattr(node, "line", 1) or 1
        source_line = _extract_source_line(self.source, line)
        if not source_line or "@" not in source_line:
            return None

        if len(s2) >= 2 and self._dim_equals(s1[-1], s2[-1]):
            # Suggest transposing right operand
            return re.sub(r"(@\s*[A-Za-z_][A-Za-z0-9_]*)(\b(?!\.T))", r"\1.T", source_line).strip()
        elif len(s1) >= 2 and self._dim_equals(s1[-2], s2[-2]):
            # Suggest transposing left operand
            return re.sub(r"([A-Za-z_][A-Za-z0-9_]*)(\s*@)(?!\.T)", r"\1.T\2", source_line).strip()

        return None

    # -------------------------------------------------------------------------
    # Core Shape Checking Operations
    # -------------------------------------------------------------------------
    def _check_matmul(
        self,
        s1: Optional[tuple[Union[int, str], ...]],
        s2: Optional[tuple[Union[int, str], ...]],
        node: ASTNode,
    ) -> Optional[tuple[Union[int, str], ...]]:
        """
        Enforces matrix multiplication rule:
        [..., M, K1] x [..., K2, N] -> [..., M, N] where K1 == K2
        Also supports 1D vector products.
        """
        if s1 is None or s2 is None:
            return None

        # 1D x 1D dot product: [K] @ [K] -> scalar ()
        if len(s1) == 1 and len(s2) == 1:
            k1, k2 = s1[0], s2[0]
            if not self._dim_equals(k1, k2):
                self._handle_error(
                    message=(
                        f"Cannot multiply 1D tensor of shape {s1} with 1D tensor of shape {s2}. "
                        f"Inner dimensions must match: {k1} != {k2}."
                    ),
                    node=node,
                    expected=k1,
                    actual=k2,
                )
                return None
            return ()

        # 1D x 2D+: [K] @ [..., K, N] -> [..., N]
        if len(s1) == 1 and len(s2) >= 2:
            k1 = s1[0]
            k2 = s2[-2]
            if not self._dim_equals(k1, k2):
                self._handle_error(
                    message=(
                        f"Cannot multiply 1D tensor of shape {s1} with tensor of shape {s2}. "
                        f"Inner dimensions must match: {k1} != {k2}."
                    ),
                    node=node,
                    expected=k1,
                    actual=k2,
                    suggested_fix=self._suggest_matmul_fix(s1, s2, node),
                )
                return None
            return s2[:-2] + (s2[-1],)

        # 2D+ x 1D: [..., M, K] @ [K] -> [..., M]
        if len(s1) >= 2 and len(s2) == 1:
            k1 = s1[-1]
            k2 = s2[0]
            if not self._dim_equals(k1, k2):
                self._handle_error(
                    message=(
                        f"Cannot multiply tensor of shape {s1} with 1D tensor of shape {s2}. "
                        f"Inner dimensions must match: {k1} != {k2}."
                    ),
                    node=node,
                    expected=k1,
                    actual=k2,
                    suggested_fix=self._suggest_matmul_fix(s1, s2, node),
                )
                return None
            return s1[:-1]

        # 2D+ x 2D+: [..., M, K1] x [..., K2, N] -> [..., M, N]
        if len(s1) >= 2 and len(s2) >= 2:
            k1 = s1[-1]
            k2 = s2[-2]
            if not self._dim_equals(k1, k2):
                self._handle_error(
                    message=(
                        f"Cannot multiply tensor of shape {s1} with tensor of shape {s2}. "
                        f"Inner dimensions must match: {k1} != {k2}."
                    ),
                    node=node,
                    expected=k1,
                    actual=k2,
                    suggested_fix=self._suggest_matmul_fix(s1, s2, node),
                )
                return None

            # Batch dimension broadcasting
            batch1 = s1[:-2]
            batch2 = s2[:-2]
            broadcasted_batch = self._broadcast_shapes_tuple(batch1, batch2)
            if broadcasted_batch is None:
                self._handle_error(
                    message=(
                        f"Cannot multiply tensor of shape {s1} with tensor of shape {s2}. "
                        f"Batch dimensions {batch1} and {batch2} are not broadcast-compatible."
                    ),
                    node=node,
                    expected=f"Broadcast-compatible batch dims with {batch1}",
                    actual=batch2,
                )
                return None

            return broadcasted_batch + (s1[-2], s2[-1])

        return None

    def _check_broadcasting(
        self,
        s1: Optional[tuple[Union[int, str], ...]],
        s2: Optional[tuple[Union[int, str], ...]],
        op: str,
        node: ASTNode,
    ) -> Optional[tuple[Union[int, str], ...]]:
        """Verifies broadcasting rules for element-wise operations."""
        if s1 is None or s2 is None:
            return None
        if s1 == ():
            return s2
        if s2 == ():
            return s1

        broadcasted = self._broadcast_shapes_tuple(s1, s2)
        if broadcasted is None:
            self._handle_error(
                message=(
                    f"Cannot perform elementwise '{op}' on tensor of shape {s1} "
                    f"and tensor of shape {s2}. Shapes are not broadcast-compatible."
                ),
                node=node,
                expected=f"Broadcast-compatible shape with {s1}",
                actual=s2,
            )
            return None
        return broadcasted

    # -------------------------------------------------------------------------
    # Shape Inference Engine
    # -------------------------------------------------------------------------
    def infer_shape(self, expr: Optional[Expr]) -> Optional[tuple[Union[int, str], ...]]:
        if expr is None:
            return None

        # 1. Identifier
        if isinstance(expr, IdentifierExpr):
            return self.get_var_shape(expr.name)

        # 2. Scalar Literal
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, (int, float, bool)) or expr.value is None:
                return ()
            return None

        # 3. List Literal: [[1.0, 2.0], [3.0, 4.0]]
        if isinstance(expr, ListLiteralExpr):
            return _infer_list_literal_shape(expr)

        # 4. Tensor Literal: tensor([[1.0, 2.0]])
        if isinstance(expr, TensorLiteralExpr):
            return self.infer_shape(expr.data)

        # 5. Member Expression: tensor.T or tensor.shape
        if isinstance(expr, MemberExpr):
            if expr.member == "T":
                target_shape = self.infer_shape(expr.target)
                if target_shape is not None:
                    if len(target_shape) < 2:
                        self._handle_error(
                            message=(
                                f"Cannot transpose tensor of shape {target_shape}. "
                                f"Transposition (.T) requires at least 2 dimensions."
                            ),
                            node=expr,
                            expected="At least 2 dimensions",
                            actual=len(target_shape),
                        )
                        return target_shape
                    # Swap last two dimensions: [..., M, N] -> [..., N, M]
                    return target_shape[:-2] + (target_shape[-1], target_shape[-2])
            return None

        # 6. Call Expression
        if isinstance(expr, CallExpr):
            callee_name = getattr(expr.callee, "name", "")

            # Native tensor constructor: tensor(data)
            if callee_name == "tensor" and expr.args:
                return self.infer_shape(expr.args[0])

            # Built-in shape allocation functions: zeros, ones, randn, empty
            if callee_name in ("zeros", "ones", "randn", "empty") and expr.args:
                first_arg = expr.args[0]
                if isinstance(first_arg, ListLiteralExpr):
                    dims: list[Union[int, str]] = []
                    for el in first_arg.elements:
                        if isinstance(el, LiteralExpr) and isinstance(el.value, int):
                            dims.append(el.value)
                        elif isinstance(el, LiteralExpr) and isinstance(el.value, str):
                            dims.append(el.value)
                        elif isinstance(el, IdentifierExpr):
                            dims.append(el.name)
                        else:
                            return None
                    return tuple(dims)
                elif all(
                    (isinstance(a, LiteralExpr) and isinstance(a.value, (int, str)))
                    or isinstance(a, IdentifierExpr)
                    for a in expr.args
                ):
                    dims = []
                    for a in expr.args:
                        if isinstance(a, LiteralExpr):
                            dims.append(a.value)
                        elif isinstance(a, IdentifierExpr):
                            dims.append(a.name)
                    return tuple(dims)

            # Functional matmul: matmul(A, B)
            if callee_name in ("matmul", "dot") and len(expr.args) >= 2:
                s1 = self.infer_shape(expr.args[0])
                s2 = self.infer_shape(expr.args[1])
                return self._check_matmul(s1, s2, expr)

            # Method calls: A.matmul(B), A.transpose(), A.reshape(...)
            if isinstance(expr.callee, MemberExpr):
                member = expr.callee.member
                target = expr.callee.target
                if member in ("matmul", "dot") and expr.args:
                    s1 = self.infer_shape(target)
                    s2 = self.infer_shape(expr.args[0])
                    return self._check_matmul(s1, s2, expr)
                elif member in ("transpose", "t"):
                    s = self.infer_shape(target)
                    if s is not None and len(s) >= 2:
                        return s[:-2] + (s[-1], s[-2])
                    return s
                elif member == "reshape" and expr.args:
                    first = expr.args[0]
                    if isinstance(first, ListLiteralExpr):
                        return tuple(
                            e.value
                            for e in first.elements
                            if isinstance(e, LiteralExpr) and isinstance(e.value, int)
                        )
                    elif all(isinstance(a, LiteralExpr) and isinstance(a.value, int) for a in expr.args):
                        return tuple(a.value for a in expr.args)

            # User-defined function call
            if callee_name and callee_name in self.functions:
                param_shapes, ret_shape = self.functions[callee_name]
                if ret_shape is not None:
                    # Unify argument shapes with parameter symbolic dimensions
                    bindings: dict[str, Union[int, str]] = {}
                    for i, (p_name, p_shape) in enumerate(param_shapes):
                        if i < len(expr.args) and p_shape is not None:
                            arg_shape = self.infer_shape(expr.args[i])
                            if arg_shape is not None:
                                if len(arg_shape) != len(p_shape):
                                    self._handle_error(
                                        message=(
                                            f"Argument shape mismatch for parameter '{p_name}' in function '{callee_name}': "
                                            f"expected {p_shape}, got {arg_shape}."
                                        ),
                                        node=expr.args[i],
                                        expected=p_shape,
                                        actual=arg_shape,
                                    )
                                else:
                                    for p_d, a_d in zip(p_shape, arg_shape):
                                        if isinstance(p_d, str):
                                            if p_d in bindings and bindings[p_d] != a_d:
                                                self._handle_error(
                                                    message=(
                                                        f"Symbolic dimension '{p_d}' conflict in call to '{callee_name}': "
                                                        f"bound to {bindings[p_d]} but encountered {a_d}."
                                                    ),
                                                    node=expr.args[i],
                                                    expected=bindings[p_d],
                                                    actual=a_d,
                                                )
                                            bindings[p_d] = a_d
                                        elif isinstance(p_d, int) and p_d != a_d:
                                            self._handle_error(
                                                message=(
                                                    f"Argument shape mismatch for parameter '{p_name}' in function '{callee_name}': "
                                                    f"expected {p_shape}, got {arg_shape}."
                                                ),
                                                node=expr.args[i],
                                                expected=p_shape,
                                                actual=arg_shape,
                                            )

                    # Substitute bindings into return shape
                    substituted_ret: list[Union[int, str]] = []
                    for rd in ret_shape:
                        if isinstance(rd, str) and rd in bindings:
                            substituted_ret.append(bindings[rd])
                        else:
                            substituted_ret.append(rd)
                    return tuple(substituted_ret)

            return None

        # 7. Binary Expression: MatMul (@) or Elementwise (+, -, *, /, etc.)
        if isinstance(expr, BinaryExpr):
            if expr.op == "@":
                s1 = self.infer_shape(expr.left)
                s2 = self.infer_shape(expr.right)
                return self._check_matmul(s1, s2, expr)

            if expr.op in ("+", "-", "*", "/", "//", "%", "**"):
                s1 = self.infer_shape(expr.left)
                s2 = self.infer_shape(expr.right)
                return self._check_broadcasting(s1, s2, expr.op, expr)

        # 8. Unary Expression: -A, +A
        if isinstance(expr, UnaryExpr):
            if expr.op in ("-", "+"):
                return self.infer_shape(expr.operand)

        return None

    # -------------------------------------------------------------------------
    # AST Traversal & Verification
    # -------------------------------------------------------------------------
    def visit(self, node: ASTNode):
        if node is None:
            return

        if isinstance(node, Program):
            # Pass 1: Register function signatures
            for stmt in node.statements:
                if isinstance(stmt, FunctionDef):
                    self._register_function_signature(stmt)

            # Pass 2: Visit statements
            for stmt in node.statements:
                self.visit(stmt)

        elif isinstance(node, VarDeclStmt):
            declared_tt = getattr(node, "tensor_type", None)
            if declared_tt is None and node.type_annot:
                declared_tt = TensorType.from_string(
                    node.type_annot if isinstance(node.type_annot, str) else str(node.type_annot)
                )

            # Evaluate RHS shape
            inferred_shape = self.infer_shape(node.value)

            if declared_tt is not None:
                declared_shape = declared_tt.to_shape_tuple()

                # Verify inferred shape conforms to declared contract
                if inferred_shape is not None:
                    if len(inferred_shape) != len(declared_shape):
                        self._handle_error(
                            message=(
                                f"Type contract violation: Variable '{node.name}' declared with contract {declared_tt} "
                                f"(shape {declared_shape}) but assigned expression of shape {inferred_shape}."
                            ),
                            node=node,
                            expected=declared_shape,
                            actual=inferred_shape,
                        )
                    else:
                        for exp_d, act_d in zip(declared_shape, inferred_shape):
                            if isinstance(exp_d, int) and isinstance(act_d, int) and exp_d != act_d:
                                self._handle_error(
                                    message=(
                                        f"Type contract violation: Variable '{node.name}' declared with contract {declared_tt} "
                                        f"(shape {declared_shape}) but assigned expression of shape {inferred_shape}."
                                    ),
                                    node=node,
                                    expected=declared_shape,
                                    actual=inferred_shape,
                                )
                                break
                            elif isinstance(exp_d, str) and isinstance(act_d, str) and exp_d != act_d:
                                self._handle_error(
                                    message=(
                                        f"Type contract violation: Variable '{node.name}' declared with contract {declared_tt} "
                                        f"(shape {declared_shape}) but assigned expression of shape {inferred_shape}."
                                    ),
                                    node=node,
                                    expected=declared_shape,
                                    actual=inferred_shape,
                                )
                                break

                self.set_var_shape(node.name, declared_shape, contract=declared_tt)
            elif inferred_shape is not None:
                self.set_var_shape(node.name, inferred_shape)

        elif isinstance(node, AssignStmt):
            inferred_shape = self.infer_shape(node.value)
            if isinstance(node.target, IdentifierExpr):
                name = node.target.name
                contract = self.get_var_contract(name)
                if contract is not None:
                    contract_shape = contract.to_shape_tuple()
                    if inferred_shape is not None:
                        if len(inferred_shape) != len(contract_shape):
                            self._handle_error(
                                message=(
                                    f"Type contract violation: Reassignment to variable '{name}' with contract {contract} "
                                    f"(shape {contract_shape}) with incompatible shape {inferred_shape}."
                                ),
                                node=node,
                                expected=contract_shape,
                                actual=inferred_shape,
                            )
                        else:
                            for exp_d, act_d in zip(contract_shape, inferred_shape):
                                if isinstance(exp_d, int) and isinstance(act_d, int) and exp_d != act_d:
                                    self._handle_error(
                                        message=(
                                            f"Type contract violation: Reassignment to variable '{name}' with contract {contract} "
                                            f"(shape {contract_shape}) with incompatible shape {inferred_shape}."
                                        ),
                                        node=node,
                                        expected=contract_shape,
                                        actual=inferred_shape,
                                    )
                                    break
                elif inferred_shape is not None:
                    self.set_var_shape(name, inferred_shape)
            else:
                self.infer_shape(node.value)

        elif isinstance(node, FunctionDef):
            prev_fn_name = self.current_fn_name
            prev_fn_ret = self.current_fn_return_shape

            self.current_fn_name = node.name
            sig = self.functions.get(node.name)
            self.current_fn_return_shape = sig[1] if sig else None

            self.push_scope()
            for p in node.params:
                p_tt = getattr(p, "tensor_type", None)
                if p_tt is None and p.type_annot:
                    p_tt = TensorType.from_string(
                        p.type_annot if isinstance(p.type_annot, str) else str(p.type_annot)
                    )
                if p_tt is not None:
                    self.set_var_shape(p.name, p_tt.to_shape_tuple(), contract=p_tt)

            for s in node.body:
                self.visit(s)

            self.pop_scope()
            self.current_fn_name = prev_fn_name
            self.current_fn_return_shape = prev_fn_ret

        elif isinstance(node, ReturnStmt):
            if node.value is not None:
                val_shape = self.infer_shape(node.value)
                if self.current_fn_return_shape is not None and val_shape is not None:
                    if len(val_shape) != len(self.current_fn_return_shape):
                        self._handle_error(
                            message=(
                                f"Return shape mismatch in function '{self.current_fn_name}': "
                                f"declared return shape {self.current_fn_return_shape} "
                                f"but returned expression of shape {val_shape}."
                            ),
                            node=node,
                            expected=self.current_fn_return_shape,
                            actual=val_shape,
                        )
                    else:
                        for exp_d, act_d in zip(self.current_fn_return_shape, val_shape):
                            if not self._dim_equals(exp_d, act_d):
                                self._handle_error(
                                    message=(
                                        f"Return shape mismatch in function '{self.current_fn_name}': "
                                        f"declared return shape {self.current_fn_return_shape} "
                                        f"but returned expression of shape {val_shape}."
                                    ),
                                    node=node,
                                    expected=self.current_fn_return_shape,
                                    actual=val_shape,
                                )
                                break

        elif isinstance(node, IfStmt):
            self.infer_shape(node.condition)
            self.push_scope()
            for s in node.then_branch:
                self.visit(s)
            self.pop_scope()

            for cond, branch in node.elif_branches:
                self.infer_shape(cond)
                self.push_scope()
                for s in branch:
                    self.visit(s)
                self.pop_scope()

            if node.else_branch:
                self.push_scope()
                for s in node.else_branch:
                    self.visit(s)
                self.pop_scope()

        elif isinstance(node, WhileStmt):
            self.infer_shape(node.condition)
            self.push_scope()
            for s in node.body:
                self.visit(s)
            self.pop_scope()

        elif isinstance(node, ForStmt):
            self.infer_shape(node.iterable)
            self.push_scope()
            for s in node.body:
                self.visit(s)
            self.pop_scope()

        elif isinstance(node, ExprStmt):
            self.infer_shape(node.expr)

    def _register_function_signature(self, stmt: FunctionDef):
        param_shapes: list[tuple[str, Optional[tuple[Union[int, str], ...]]]] = []
        for p in stmt.params:
            p_tt = getattr(p, "tensor_type", None)
            if p_tt is None and p.type_annot:
                p_tt = TensorType.from_string(
                    p.type_annot if isinstance(p.type_annot, str) else str(p.type_annot)
                )
            shape = p_tt.to_shape_tuple() if p_tt else None
            param_shapes.append((p.name, shape))

        ret_shape = None
        if getattr(stmt, "return_tensor_type", None):
            ret_shape = stmt.return_tensor_type.to_shape_tuple()
        elif stmt.return_type:
            r_tt = TensorType.from_string(
                stmt.return_type if isinstance(stmt.return_type, str) else str(stmt.return_type)
            )
            if r_tt:
                ret_shape = r_tt.to_shape_tuple()

        self.functions[stmt.name] = (param_shapes, ret_shape)

    # -------------------------------------------------------------------------
    # Public Entry Points
    # -------------------------------------------------------------------------
    def check(
        self,
        ast: Program,
        raise_on_error: Optional[bool] = None,
    ) -> list[DiagnosticReport]:
        """
        Statically checks all tensor shapes and contracts in the given AST program.
        Returns a list of DiagnosticReport objects, or raises CompileTimeShapeMismatchError
        if raise_on_error is True.
        """
        prev_raise = self.raise_on_error
        if raise_on_error is not None:
            self.raise_on_error = raise_on_error

        try:
            self.diagnostics.clear()
            self.visit(ast)
            return list(self.diagnostics)
        finally:
            self.raise_on_error = prev_raise

    def verify(self, ast: Program) -> None:
        """
        Verifies all tensor shapes in the AST.
        Raises CompileTimeShapeMismatchError immediately upon encountering any shape mismatch.
        """
        self.check(ast, raise_on_error=True)


# =============================================================================
# Helper Functions
# =============================================================================

def check_shapes(
    source: str,
    filepath: str = "<source>",
    raise_on_error: bool = False,
) -> list[DiagnosticReport]:
    """
    Tokenizes, parses, and runs compile-time static shape analysis on Synapse source code.
    Returns a list of DiagnosticReport objects.
    """
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    checker = StaticShapeChecker(
        source=source,
        filepath=filepath,
        raise_on_error=raise_on_error,
    )
    return checker.check(ast)


def verify_shapes(source: str, filepath: str = "<source>") -> None:
    """
    Runs compile-time static shape analysis and raises CompileTimeShapeMismatchError
    on any incompatibility.
    """
    check_shapes(source, filepath=filepath, raise_on_error=True)
