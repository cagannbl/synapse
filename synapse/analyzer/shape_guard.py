"""
Synapse Symbolic Tensor Shape Guard & Zero-Crash Contract Verifier
==================================================================
Phase 2: Symbolic Dimension Equations, Transformer Geometry, and
Compile-Time Zero-Crash Training Invariants.

Verifies tensor shapes, matrix multiplications, transpose alignments,
and reshaping element conservation across deep learning models before
weights are allocated or GPU training loops are executed.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast import (
    Program, Stmt, Expr, LiteralExpr, IdentifierExpr,
    BinaryExpr, UnaryExpr, CallExpr, MemberExpr, ListLiteralExpr,
    TensorLiteralExpr, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, FunctionDef, Param, TensorType
)
from synapse.analyzer.shape_checker import CompileTimeShapeMismatchError


@dataclass
class SymbolicDimension:
    """Represents a symbolic or concrete tensor dimension (e.g. B, S, D, or 128)."""
    name: str
    value: Optional[int] = None
    factors: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if str(self.name).isdigit():
            self.value = int(self.name)
            self.factors = {}
        elif not self.factors:
            self.factors = {self.name: 1}

    @classmethod
    def from_val(cls, val: Union[int, str, SymbolicDimension]) -> SymbolicDimension:
        if isinstance(val, SymbolicDimension):
            return val
        s = str(val).strip()
        if s.isdigit():
            return cls(name=s, value=int(s))
        return cls(name=s)

    def is_concrete(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        return str(self.value) if self.value is not None else self.name

    def __str__(self) -> str:
        return str(self.value) if self.value is not None else self.name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self.value == other
        if isinstance(other, str):
            if self.value is not None:
                return str(self.value) == other
            return self.name == other
        if isinstance(other, SymbolicDimension):
            if self.value is not None and other.value is not None:
                return self.value == other.value
            return self.name == other.name or self.factors == other.factors
        return False


class SymbolicShapeSolver:
    """
    Solves and unifies symbolic dimension constraints across tensor operations.
    Handles dimension substitutions (e.g., D = H * K).
    """

    def __init__(self):
        self.bindings: Dict[str, Union[int, str]] = {}
        self.equations: List[Tuple[str, str]] = []

    def bind(self, name: str, val: Union[int, str]):
        self.bindings[name] = val

    def resolve(self, dim: Union[int, str, SymbolicDimension]) -> Union[int, str]:
        s_dim = SymbolicDimension.from_val(dim)
        if s_dim.is_concrete():
            return s_dim.value  # type: ignore
        name = s_dim.name
        while name in self.bindings:
            bound = self.bindings[name]
            if isinstance(bound, int):
                return bound
            if bound == name:
                break
            name = str(bound)
        return name

    def are_compatible(
        self,
        d1: Union[int, str, SymbolicDimension],
        d2: Union[int, str, SymbolicDimension],
    ) -> bool:
        """Determines if two dimensions match concretely or symbolically."""
        r1 = self.resolve(d1)
        r2 = self.resolve(d2)

        # Direct equality
        if r1 == r2:
            return True

        # Wildcard / Batch flexible dimensions
        if r1 in ("-1", "*", "None") or r2 in ("-1", "*", "None"):
            return True

        # If both are concrete integers, must be equal
        if isinstance(r1, int) and isinstance(r2, int):
            return r1 == r2

        # Bind unknown symbol to known value or symbol
        if isinstance(r1, str) and not r1.isdigit():
            self.bindings[r1] = r2
            return True
        if isinstance(r2, str) and not r2.isdigit():
            self.bindings[r2] = r1
            return True

        return False

    def check_matmul(
        self,
        shape_a: Tuple[Union[int, str], ...],
        shape_b: Tuple[Union[int, str], ...],
    ) -> Tuple[bool, Optional[Tuple[Union[int, str], ...]], str]:
        """
        Validates MatMul [..., M, K] @ [..., K, N] -> [..., M, N].
        Returns (is_valid, result_shape, error_message).
        """
        if len(shape_a) < 2 or len(shape_b) < 2:
            return False, None, f"MatMul requires at least 2D tensors, got {shape_a} and {shape_b}"

        k_left = shape_a[-1]
        k_right = shape_b[-2]

        if not self.are_compatible(k_left, k_right):
            err = (
                f"MatMul inner dimension mismatch: left inner '{k_left}' "
                f"does not match right inner '{k_right}' "
                f"(shapes: {shape_a} @ {shape_b})"
            )
            return False, None, err

        # Batch dimensions check
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]
        if batch_a and batch_b and batch_a != batch_b:
            # Check broadcasting compatibility
            if not all(self.are_compatible(ba, bb) for ba, bb in zip(batch_a, batch_b)):
                return False, None, f"Batch dimension mismatch: {batch_a} vs {batch_b}"

        out_batch = batch_a if len(batch_a) >= len(batch_b) else batch_b
        out_shape = out_batch + (shape_a[-2], shape_b[-1])
        return True, out_shape, ""

    def check_reshape_conservation(
        self,
        src_shape: Tuple[Union[int, str], ...],
        dst_shape: Tuple[Union[int, str], ...],
    ) -> Tuple[bool, str]:
        """Validates that total element count is conserved during reshape."""
        src_concrete = [self.resolve(d) for d in src_shape]
        dst_concrete = [self.resolve(d) for d in dst_shape]

        # If all are integers, compute product
        if all(isinstance(x, int) for x in src_concrete) and all(isinstance(x, int) for x in dst_concrete):
            prod_src = 1
            for x in src_concrete:
                prod_src *= int(x)
            prod_dst = 1
            for x in dst_concrete:
                prod_dst *= int(x)
            if prod_src != prod_dst:
                return False, f"Reshape element count mismatch: {src_shape} ({prod_src}) vs {dst_shape} ({prod_dst})"

        return True, ""


class TensorShapeGuard:
    """
    AST Shape Invariant Analyzer for deep learning architectures.
    Statically discovers hidden shape bugs, MHA projection mismatches,
    and missing transpose operators.
    """

    def __init__(self, source_text: str = "", filename: str = "<memory>"):
        self.source_text = source_text
        self.filename = filename
        self.solver = SymbolicShapeSolver()
        self.env: Dict[str, Tuple[Union[int, str], ...]] = {}
        self.source_lines = source_text.splitlines()

    def _get_line_text(self, line: int) -> str:
        idx = line - 1
        if 0 <= idx < len(self.source_lines):
            return self.source_lines[idx]
        return ""

    def verify_ast(self, program: Program) -> List[CompileTimeShapeMismatchError]:
        """Traverses AST and records all compile-time shape mismatch errors."""
        errors: List[CompileTimeShapeMismatchError] = []

        for stmt in getattr(program, "statements", []):
            try:
                self._check_stmt(stmt, errors)
            except CompileTimeShapeMismatchError as e:
                errors.append(e)

        return errors

    def _check_stmt(self, stmt: Stmt, errors: List[CompileTimeShapeMismatchError]):
        if isinstance(stmt, VarDeclStmt):
            shape = None
            if getattr(stmt, "tensor_type", None):
                shape = stmt.tensor_type.to_shape_tuple()
            elif getattr(stmt, "type_annot", None):
                t_str = str(stmt.type_annot)
                if "Tensor[" in t_str:
                    t_inst = TensorType.from_string(t_str)
                    shape = t_inst.to_shape_tuple()

            if getattr(stmt, "value", None):
                val_shape = self._infer_expr_shape(stmt.value, errors)
                if shape and val_shape:
                    if not all(self.solver.are_compatible(s, v) for s, v in zip(shape, val_shape)):
                        err = CompileTimeShapeMismatchError(
                            message=f"Declared shape {shape} does not match assigned shape {val_shape}",
                            line=getattr(stmt, "line", 1),
                            column=getattr(stmt, "column", 1),
                            expected=shape,
                            actual=val_shape,
                            source_line=self._get_line_text(getattr(stmt, "line", 1)),
                            suggested_fix=f"let {stmt.name}: Tensor{list(val_shape)}",
                        )
                        errors.append(err)
                shape = shape or val_shape

            if shape:
                self.env[stmt.name] = shape

        elif isinstance(stmt, AssignStmt):
            val_shape = self._infer_expr_shape(stmt.value, errors)
            target_name = stmt.target.name if isinstance(stmt.target, IdentifierExpr) else (stmt.target if isinstance(stmt.target, str) else None)
            if target_name and target_name in self.env and val_shape:
                existing_shape = self.env[target_name]
                if not all(self.solver.are_compatible(e, v) for e, v in zip(existing_shape, val_shape)):
                    err = CompileTimeShapeMismatchError(
                        message=f"Variable '{target_name}' expected shape {existing_shape}, got {val_shape}",
                        line=getattr(stmt, "line", 1),
                        column=getattr(stmt, "column", 1),
                        expected=existing_shape,
                        actual=val_shape,
                        source_line=self._get_line_text(getattr(stmt, "line", 1)),
                    )
                    errors.append(err)
            elif target_name and val_shape:
                self.env[target_name] = val_shape

        elif isinstance(stmt, FunctionDef):
            # Record parameter tensor shapes in function scope
            local_env = dict(self.env)
            for p in stmt.params:
                p_shape = getattr(p, "tensor_type", None)
                if p_shape:
                    self.env[p.name] = p_shape.to_shape_tuple()
                elif p.type_annot and "Tensor[" in str(p.type_annot):
                    self.env[p.name] = TensorType.from_string(str(p.type_annot)).to_shape_tuple()

            for s in stmt.body:
                try:
                    self._check_stmt(s, errors)
                except CompileTimeShapeMismatchError as e:
                    errors.append(e)
            self.env = local_env

        elif isinstance(stmt, ExprStmt):
            self._infer_expr_shape(stmt.expr, errors)

        elif isinstance(stmt, IfStmt):
            for s in getattr(stmt, "then_branch", []):
                try:
                    self._check_stmt(s, errors)
                except CompileTimeShapeMismatchError as e:
                    errors.append(e)
            for s in (getattr(stmt, "else_branch", []) or []):
                try:
                    self._check_stmt(s, errors)
                except CompileTimeShapeMismatchError as e:
                    errors.append(e)

        elif isinstance(stmt, WhileStmt):
            for s in getattr(stmt, "body", []):
                try:
                    self._check_stmt(s, errors)
                except CompileTimeShapeMismatchError as e:
                    errors.append(e)

        elif isinstance(stmt, ForStmt):
            for s in getattr(stmt, "body", []):
                try:
                    self._check_stmt(s, errors)
                except CompileTimeShapeMismatchError as e:
                    errors.append(e)

        elif isinstance(stmt, ReturnStmt):
            if getattr(stmt, "value", None):
                self._infer_expr_shape(stmt.value, errors)

    def _infer_expr_shape(
        self, expr: Expr, errors: List[CompileTimeShapeMismatchError]
    ) -> Optional[Tuple[Union[int, str], ...]]:
        if expr is None:
            return None

        if isinstance(expr, IdentifierExpr):
            return self.env.get(expr.name)

        if isinstance(expr, CallExpr):
            callee_name = None
            if isinstance(expr.callee, IdentifierExpr):
                callee_name = expr.callee.name
            elif isinstance(expr.callee, MemberExpr):
                callee_name = expr.callee.member

            # Tensor shape initializers
            if callee_name in ("zeros", "ones", "randn", "rand", "empty", "Tensor"):
                if expr.args:
                    first_arg = expr.args[0]
                    if isinstance(first_arg, ListLiteralExpr):
                        dims = []
                        for el in first_arg.elements:
                            if isinstance(el, LiteralExpr) and isinstance(el.value, (int, str)):
                                dims.append(int(el.value) if str(el.value).isdigit() else str(el.value))
                            elif isinstance(el, IdentifierExpr):
                                dims.append(el.name)
                        return tuple(dims)
                    elif all(isinstance(a, (LiteralExpr, IdentifierExpr)) for a in expr.args):
                        dims = []
                        for a in expr.args:
                            if isinstance(a, LiteralExpr):
                                dims.append(int(a.value) if str(a.value).isdigit() else str(a.value))
                            elif isinstance(a, IdentifierExpr):
                                dims.append(a.name)
                        return tuple(dims)

            # Method calls: target.reshape(...) or target.transpose(...)
            if isinstance(expr.callee, MemberExpr):
                target_shape = self._infer_expr_shape(expr.callee.target, errors)
                member = expr.callee.member

                if member == "reshape" and expr.args:
                    dst_shape = None
                    first_arg = expr.args[0]
                    if isinstance(first_arg, ListLiteralExpr):
                        dst_dims = []
                        for el in first_arg.elements:
                            if isinstance(el, LiteralExpr) and isinstance(el.value, (int, str)):
                                dst_dims.append(int(el.value) if str(el.value).isdigit() else str(el.value))
                            elif isinstance(el, IdentifierExpr):
                                dst_dims.append(el.name)
                        dst_shape = tuple(dst_dims)
                    elif all(isinstance(a, (LiteralExpr, IdentifierExpr)) for a in expr.args):
                        dst_dims = []
                        for a in expr.args:
                            if isinstance(a, LiteralExpr):
                                dst_dims.append(int(a.value) if str(a.value).isdigit() else str(a.value))
                            elif isinstance(a, IdentifierExpr):
                                dst_dims.append(a.name)
                        dst_shape = tuple(dst_dims)

                    if target_shape and dst_shape:
                        is_conserved, err_msg = self.solver.check_reshape_conservation(target_shape, dst_shape)
                        if not is_conserved:
                            err = CompileTimeShapeMismatchError(
                                message=err_msg,
                                line=getattr(expr, "line", 1),
                                column=getattr(expr, "column", 1),
                                expected=f"Compatible element count with {target_shape}",
                                actual=f"Target shape {dst_shape}",
                                source_line=self._get_line_text(getattr(expr, "line", 1)),
                                suggested_fix="Ensure product of dimensions equals original element count",
                            )
                            errors.append(err)
                    return dst_shape or target_shape

                if member in ("transpose", "t"):
                    if target_shape and len(target_shape) >= 2:
                        return target_shape[:-2] + (target_shape[-1], target_shape[-2])
                    return target_shape

                if member in ("matmul", "dot") and expr.args:
                    arg_shape = self._infer_expr_shape(expr.args[0], errors)
                    if target_shape and arg_shape:
                        is_valid, out_shape, err_msg = self.solver.check_matmul(target_shape, arg_shape)
                        if not is_valid:
                            errors.append(
                                CompileTimeShapeMismatchError(
                                    message=err_msg,
                                    line=getattr(expr, "line", 1),
                                    column=getattr(expr, "column", 1),
                                    expected=f"Matching inner dimension '{target_shape[-1]}'",
                                    actual=f"Inner dimension '{arg_shape[-2]}'",
                                    source_line=self._get_line_text(getattr(expr, "line", 1)),
                                )
                            )
                            return None
                        return out_shape

            # Standalone functions: reshape(x, [shape]), transpose(x), matmul(a, b)
            if callee_name == "reshape" and len(expr.args) >= 2:
                target_shape = self._infer_expr_shape(expr.args[0], errors)
                shape_arg = expr.args[1]
                dst_shape = None
                if isinstance(shape_arg, ListLiteralExpr):
                    dst_dims = [
                        (int(el.value) if str(el.value).isdigit() else str(el.value))
                        for el in shape_arg.elements
                        if isinstance(el, (LiteralExpr, IdentifierExpr))
                    ]
                    dst_shape = tuple(dst_dims)
                if target_shape and dst_shape:
                    is_conserved, err_msg = self.solver.check_reshape_conservation(target_shape, dst_shape)
                    if not is_conserved:
                        errors.append(
                            CompileTimeShapeMismatchError(
                                message=err_msg,
                                line=getattr(expr, "line", 1),
                                column=getattr(expr, "column", 1),
                                expected=f"Compatible element count with {target_shape}",
                                actual=f"Target shape {dst_shape}",
                                source_line=self._get_line_text(getattr(expr, "line", 1)),
                            )
                        )
                return dst_shape or target_shape

            if callee_name == "matmul" and len(expr.args) >= 2:
                s1 = self._infer_expr_shape(expr.args[0], errors)
                s2 = self._infer_expr_shape(expr.args[1], errors)
                if s1 and s2:
                    is_valid, out_shape, err_msg = self.solver.check_matmul(s1, s2)
                    if not is_valid:
                        line_no = getattr(expr, "line", 1) or 1
                        source_line = self._get_line_text(line_no)
                        suggested = "Transpose tensor with .T" if len(s2) >= 2 else None
                        diff = None
                        if "@" in source_line:
                            fixed_line = re.sub(r"(@\s*[A-Za-z_][A-Za-z0-9_]*)(\b(?!\.T))", r"\1.T", source_line)
                            if fixed_line != source_line:
                                from synapse.core.diagnostics import generate_unified_diff
                                orig_lines = self.source_text.splitlines(keepends=True)
                                if 1 <= line_no <= len(orig_lines):
                                    has_nl = orig_lines[line_no - 1].endswith("\n")
                                    mod_lines = list(orig_lines)
                                    mod_lines[line_no - 1] = fixed_line + ("\n" if has_nl else "")
                                    diff = generate_unified_diff(self.source_text, "".join(mod_lines), filename=self.filename)
                        errors.append(
                            CompileTimeShapeMismatchError(
                                message=err_msg,
                                line=line_no,
                                column=getattr(expr, "column", 1),
                                expected=str(s1[-1]),
                                actual=str(s2[-2]),
                                source_line=source_line,
                                suggested_fix=suggested,
                                diff=diff,
                                code="SYN-E202",
                            )
                        )
                        return None
                    return out_shape

        if isinstance(expr, MemberExpr):
            # Check for transpose .T
            if expr.member in ("T", "transpose"):
                base_shape = self._infer_expr_shape(expr.target, errors)
                if base_shape and len(base_shape) >= 2:
                    # Swap last two dimensions
                    return base_shape[:-2] + (base_shape[-1], base_shape[-2])
                return base_shape

        if isinstance(expr, BinaryExpr):
            left_shape = self._infer_expr_shape(expr.left, errors)
            right_shape = self._infer_expr_shape(expr.right, errors)

            # MatMul operator (@)
            if expr.op == "@":
                if left_shape and right_shape:
                    is_valid, out_shape, err_msg = self.solver.check_matmul(left_shape, right_shape)
                    if not is_valid:
                        # Check if transposing right solves it
                        suggested = "Transpose tensor with .T"
                        diff = None
                        line_no = getattr(expr, "line", 1) or 1
                        source_line = self._get_line_text(line_no)
                        if "@" in source_line:
                            fixed_line = re.sub(r"(@\s*[A-Za-z_][A-Za-z0-9_]*)(\b(?!\.T))", r"\1.T", source_line)
                            if fixed_line != source_line:
                                from synapse.core.diagnostics import generate_unified_diff
                                orig_lines = self.source_text.splitlines(keepends=True)
                                if 1 <= line_no <= len(orig_lines):
                                    has_nl = orig_lines[line_no - 1].endswith("\n")
                                    mod_lines = list(orig_lines)
                                    mod_lines[line_no - 1] = fixed_line + ("\n" if has_nl else "")
                                    diff = generate_unified_diff(self.source_text, "".join(mod_lines), filename=self.filename)

                        err = CompileTimeShapeMismatchError(
                            message=err_msg,
                            line=line_no,
                            column=getattr(expr, "column", 1),
                            expected=str(left_shape[-1]),
                            actual=str(right_shape[-2]),
                            source_line=source_line,
                            suggested_fix=suggested,
                            diff=diff,
                            code="SYN-E202",
                        )
                        errors.append(err)
                        return None
                    return out_shape

            # Element-wise operations (+, -, *)
            if expr.op in ("+", "-", "*", "/"):
                if left_shape and right_shape:
                    if left_shape == right_shape:
                        return left_shape
                    # Broadcast
                    return left_shape if len(left_shape) >= len(right_shape) else right_shape

        return None

    def verify_source(self) -> List[CompileTimeShapeMismatchError]:
        """Parses source_text and verifies shape invariants."""
        tokens = Lexer(self.source_text).tokenize()
        ast = Parser(tokens).parse()
        return self.verify_ast(ast)


def verify_shapes_in_file(filepath: str) -> Tuple[bool, List[CompileTimeShapeMismatchError]]:
    """Convenience helper to verify tensor shape contracts in a .syn file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    guard = TensorShapeGuard(source_text=content, filename=filepath)
    errs = guard.verify_source()
    return len(errs) == 0, errs
