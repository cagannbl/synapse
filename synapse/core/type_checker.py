"""
Synapse Static Type Checker & Contract Enforcement System
=========================================================
Implements compile-time static type analysis and contract verification for Synapse:
- Variable declarations and reassignments type consistency
- Function parameter types and return type contracts
- Static and symbolic TensorType contracts against declared shapes and inferred operations
- Algebraic Data Types: Enums (declaration & variant validation), Option, Result
- Generic Types: List[T], Dict[K, V], Option[T], Result[T, E]
- Union Types: Type1 | Type2
- Safe unwrapping and panic prevention for Option / Result
- Structured diagnostics with line pointers and CLI handler
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.parser.ast_nodes import (
    ASTNode, Program, Stmt, Expr, LiteralExpr, IdentifierExpr,
    BinaryExpr, UnaryExpr, PipeExpr, CallExpr, MemberExpr, IndexExpr,
    ListLiteralExpr, DictLiteralExpr, TensorLiteralExpr,
    VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, PassStmt,
    BreakStmt, ContinueStmt, IfStmt, WhileStmt, ForStmt, FunctionDef,
    ImportStmt, PromptDef, AgentDef, ToolDef, Param,
    TensorType, ShapeAnnotation, TypeAnnotation,
    EnumDeclStmt, GenericType, UnionType,
    Some, NoneOption, Option, Ok, Err, Result,
    MatchStmt, MatchCase, TryExpr,
)
from synapse.core.diagnostics import (
    DiagnosticReport, TypeContractViolationError, generate_unified_diff,
    _extract_source_line, _generate_pointer, _infer_expr_shape, _infer_list_shape
)


class NonExhaustiveMatchError(TypeError):
    """Raised at compile-time when a pattern matching construct fails exhaustiveness checking."""

    def __init__(self, message: str, missing_variants: Optional[list[str]] = None):
        super().__init__(message)
        self.message = message
        self.missing_variants = missing_variants or []


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EnumVariantType:
    """Represents a specific variant of an enum (e.g. Color.Red)."""
    enum_name: str
    variant: str

    def __str__(self) -> str:
        return f"{self.enum_name}.{self.variant}"

    def __repr__(self) -> str:
        return f"EnumVariantType({self.enum_name}.{self.variant})"


@dataclass
class FunctionSignature:
    """Represents the static signature of a function."""
    name: str
    params: list[tuple[str, Optional[Any]]]  # (param_name, type)
    return_type: Optional[Any]
    line: int = 0
    column: int = 0


@dataclass
class TypeCheckResult:
    """Structured result returned by the static type checker."""
    errors: list[DiagnosticReport] = field(default_factory=list)
    is_valid: bool = True
    warnings: list[DiagnosticReport] = field(default_factory=list)

    def __post_init__(self):
        if self.errors:
            self.is_valid = False


# =============================================================================
# Type System Utilities
# =============================================================================

def normalize_type_annotation(annot: Any) -> Any:
    """Normalizes various type representations into a canonical format."""
    if annot is None:
        return None
    if isinstance(annot, GenericType):
        if annot.base == "Tensor":
            if annot.tensor_type is not None:
                return annot.tensor_type
            tt = TensorType.from_string(annot.raw)
            if tt is not None:
                return tt
            return "Tensor"
        return annot
    if isinstance(annot, (UnionType, TensorType, EnumVariantType)):
        return annot
    if isinstance(annot, TypeAnnotation):
        if annot.tensor_type is not None:
            return annot.tensor_type
        return normalize_type_str(annot.raw)
    if isinstance(annot, str):
        return normalize_type_str(annot)
    return annot


def normalize_type_str(raw: str) -> Any:
    """Parses a type string into a structured type representation."""
    raw = raw.strip()
    if not raw:
        return None

    if raw in ("none", "None"):
        return "None"
    if raw in ("int", "float", "str", "bool", "Any"):
        return raw

    # Tensor type
    if raw.startswith("Tensor[") or raw == "Tensor":
        tt = TensorType.from_string(raw)
        if tt is not None:
            return tt
        return "Tensor"

    # Union type: A | B
    if "|" in raw:
        parts = [normalize_type_str(p.strip()) for p in raw.split("|") if p.strip()]
        return UnionType(types=parts, raw=raw)

    # Generic type: Base[Arg1, Arg2]
    if "[" in raw and raw.endswith("]"):
        base = raw[:raw.index("[")].strip()
        inner = raw[raw.index("[") + 1:-1].strip()
        # Split args respecting nested brackets
        args: list[Any] = []
        depth = 0
        current: list[str] = []
        for ch in inner:
            if ch == "[":
                depth += 1
                current.append(ch)
            elif ch == "]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                args.append(normalize_type_str("".join(current).strip()))
                current = []
            else:
                current.append(ch)
        if current:
            args.append(normalize_type_str("".join(current).strip()))
        return GenericType(base=base, type_args=args, raw=raw)

    return raw


def are_types_compatible(actual: Any, expected: Any, enums: Optional[dict[str, set[str]]] = None) -> bool:
    """
    Determines if `actual` type can be safely assigned or passed to `expected` type contract.
    Handles primitives, generics (Option, Result, List, Dict), unions, enums, and tensors.
    """
    if enums is None:
        enums = {}

    act = normalize_type_annotation(actual)
    exp = normalize_type_annotation(expected)

    # Untyped / Any is universally compatible
    if exp is None or exp == "Any" or act is None or act == "Any":
        return True

    # Expected is Union: actual must match at least one variant
    if isinstance(exp, UnionType):
        return any(are_types_compatible(act, variant, enums) for variant in exp.types)

    # Actual is Union: all variants must be compatible with expected
    if isinstance(act, UnionType):
        return all(are_types_compatible(variant, exp, enums) for variant in act.types)

    # TensorType checking
    if isinstance(exp, TensorType):
        if isinstance(act, TensorType):
            if not exp.dims:  # Tensor without shape constraint
                return True
            if not act.dims:
                return True
            return exp.matches_shape(act.dims)
        elif act == "Tensor":
            return True
        return False

    if exp == "Tensor":
        return isinstance(act, TensorType) or act == "Tensor"

    # GenericType checking
    if isinstance(exp, GenericType):
        # Option[T]
        if exp.base == "Option":
            # Option can be None
            if act in ("None", "none"):
                return True
            if isinstance(act, GenericType) and act.base == "Option":
                if not act.type_args or act.type_args[0] == "Any":
                    return True
                if not exp.type_args or exp.type_args[0] == "Any":
                    return True
                return are_types_compatible(act.type_args[0], exp.type_args[0], enums)
            # Cannot directly assign unwrapped T to Option[T]
            return False

        # Result[T, E]
        if exp.base == "Result":
            if isinstance(act, GenericType) and act.base == "Result":
                exp_t = exp.type_args[0] if len(exp.type_args) > 0 else "Any"
                exp_e = exp.type_args[1] if len(exp.type_args) > 1 else "Any"
                act_t = act.type_args[0] if len(act.type_args) > 0 else "Any"
                act_e = act.type_args[1] if len(act.type_args) > 1 else "Any"
                return (
                    are_types_compatible(act_t, exp_t, enums) and
                    are_types_compatible(act_e, exp_e, enums)
                )
            return False

        # List[T]
        if exp.base == "List":
            if isinstance(act, GenericType) and act.base == "List":
                if not act.type_args or act.type_args[0] == "Any":
                    return True  # empty list
                if not exp.type_args or exp.type_args[0] == "Any":
                    return True
                return are_types_compatible(act.type_args[0], exp.type_args[0], enums)
            return False

        # Dict[K, V]
        if exp.base == "Dict":
            if isinstance(act, GenericType) and act.base == "Dict":
                if not act.type_args or act.type_args[0] == "Any":
                    return True
                exp_k = exp.type_args[0] if len(exp.type_args) > 0 else "Any"
                exp_v = exp.type_args[1] if len(exp.type_args) > 1 else "Any"
                act_k = act.type_args[0] if len(act.type_args) > 0 else "Any"
                act_v = act.type_args[1] if len(act.type_args) > 1 else "Any"
                return (
                    are_types_compatible(act_k, exp_k, enums) and
                    are_types_compatible(act_v, exp_v, enums)
                )
            return False

        # Generic matching by base and arguments
        if isinstance(act, GenericType) and act.base == exp.base:
            if len(act.type_args) == len(exp.type_args):
                return all(
                    are_types_compatible(a, e, enums)
                    for a, e in zip(act.type_args, exp.type_args)
                )
        return False

    # Enum checking
    if isinstance(exp, str) and exp in enums:
        if isinstance(act, EnumVariantType) and act.enum_name == exp:
            return True
        if isinstance(act, str) and act == exp:
            return True
        return False

    if isinstance(act, EnumVariantType):
        if isinstance(exp, EnumVariantType):
            return act.enum_name == exp.enum_name and act.variant == exp.variant
        if isinstance(exp, str):
            return exp == act.enum_name or exp == f"{act.enum_name}.{act.variant}"
        return False

    # Primitives & Equality
    if exp == act:
        return True

    # Numeric widening: int -> float
    if exp == "float" and act == "int":
        return True

    # None checking
    if exp in ("None", "none") and act in ("None", "none"):
        return True

    return False


# =============================================================================
# Static Type Checker Visitor
# =============================================================================

class StaticTypeChecker:
    """
    AST Visitor analyzing statements and expressions before runtime execution.
    Enforces type contracts, shape contracts, ADT variants, and safe unwrapping.
    """

    def __init__(self, source: str = "", filepath: str = "<source>"):
        self.source = source
        self.filepath = filepath
        self.errors: list[DiagnosticReport] = []
        self.warnings: list[DiagnosticReport] = []

        # Scopes stack for lexical scoping: var_name -> type
        self.scopes: list[dict[str, Any]] = [{}]
        # Tensor shapes stack: var_name -> tuple of dims
        self.tensor_shapes: list[dict[str, tuple[Union[int, str], ...]]] = [{}]
        # Known value state: var_name -> "None" | "Some" | "Err" | "Ok" | literal
        self.known_states: list[dict[str, Any]] = [{}]
        # Constant variables set
        self.const_vars: set[str] = set()

        # Enums: enum_name -> set of variants
        self.enums: dict[str, set[str]] = {}
        # Functions: fn_name -> FunctionSignature
        self.functions: dict[str, FunctionSignature] = {}

        # Current function being checked
        self.current_fn: Optional[FunctionSignature] = None

    # -------------------------------------------------------------------------
    # Scope Management
    # -------------------------------------------------------------------------
    def push_scope(self):
        self.scopes.append({})
        self.tensor_shapes.append({})
        self.known_states.append({})

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
            self.tensor_shapes.pop()
            self.known_states.pop()

    def set_var(
        self,
        name: str,
        type_val: Any,
        shape: Optional[tuple[Union[int, str], ...]] = None,
        known_state: Any = None,
        is_const: bool = False,
    ):
        norm_t = normalize_type_annotation(type_val)
        self.scopes[-1][name] = norm_t
        if shape is not None:
            self.tensor_shapes[-1][name] = shape
        if known_state is not None:
            self.known_states[-1][name] = known_state
        if is_const:
            self.const_vars.add(name)

    def get_var_type(self, name: str) -> Optional[Any]:
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        return None

    def get_var_shape(self, name: str) -> Optional[tuple[Union[int, str], ...]]:
        for s in reversed(self.tensor_shapes):
            if name in s:
                return s[name]
        return None

    def get_var_known_state(self, name: str) -> Any:
        for s in reversed(self.known_states):
            if name in s:
                return s[name]
        return None

    def update_var_state(self, name: str, state: Any):
        for s in reversed(self.known_states):
            if name in s:
                s[name] = state
                return
        self.known_states[-1][name] = state

    # -------------------------------------------------------------------------
    # Diagnostic Reporting
    # -------------------------------------------------------------------------
    def report_error(
        self,
        message: str,
        node: ASTNode,
        error_type: str = "TypeContractViolationError",
        suggested_fix: Optional[str] = None,
        hint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        line = getattr(node, "line", 1) or 1
        col = getattr(node, "column", 1) or 1
        source_line = _extract_source_line(self.source, line)
        pointer = _generate_pointer(col)

        report = DiagnosticReport(
            status="error",
            error_type=error_type,
            message=message,
            line=line,
            column=col,
            source_line=source_line,
            pointer=pointer,
            suggested_fix=suggested_fix,
            ai_prompt_hint=hint,
            details=details or {},
        )
        self.errors.append(report)

    # -------------------------------------------------------------------------
    # Main Check Entry Point
    # -------------------------------------------------------------------------
    def check_program(self, program: Program) -> TypeCheckResult:
        # Pass 1: Register all Enums and Function definitions
        for stmt in program.statements:
            if isinstance(stmt, EnumDeclStmt):
                self._register_enum(stmt)
            elif isinstance(stmt, FunctionDef):
                self._register_function_signature(stmt)

        # Pass 2: Visit each statement
        for stmt in program.statements:
            self.visit_stmt(stmt)

        return TypeCheckResult(errors=self.errors, is_valid=len(self.errors) == 0, warnings=self.warnings)

    def _register_enum(self, stmt: EnumDeclStmt):
        if stmt.name in self.enums:
            self.report_error(
                f"Duplicate definition of enum '{stmt.name}'.",
                node=stmt,
            )
            return
        self.enums[stmt.name] = set(stmt.variants)

    def _register_function_signature(self, stmt: FunctionDef):
        params: list[tuple[str, Optional[Any]]] = []
        for p in stmt.params:
            norm_t = normalize_type_annotation(p.type_annot)
            params.append((p.name, norm_t))
        ret_t = normalize_type_annotation(stmt.return_type)
        sig = FunctionSignature(
            name=stmt.name,
            params=params,
            return_type=ret_t,
            line=stmt.line,
            column=stmt.column,
        )
        self.functions[stmt.name] = sig

    # -------------------------------------------------------------------------
    # Statement Visitors
    # -------------------------------------------------------------------------
    def visit_stmt(self, stmt: Stmt):
        if isinstance(stmt, VarDeclStmt):
            self.visit_var_decl(stmt)
        elif isinstance(stmt, AssignStmt):
            self.visit_assign(stmt)
        elif isinstance(stmt, FunctionDef):
            self.visit_function_def(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.visit_return(stmt)
        elif isinstance(stmt, IfStmt):
            self.visit_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self.visit_while(stmt)
        elif isinstance(stmt, ForStmt):
            self.visit_for(stmt)
        elif isinstance(stmt, ExprStmt):
            self.infer_expr_type(stmt.expr)
        elif isinstance(stmt, MatchStmt):
            return self.visit_match_stmt(stmt)
        elif isinstance(stmt, EnumDeclStmt):
            pass  # Registered in Pass 1
        elif isinstance(stmt, (PassStmt, BreakStmt, ContinueStmt, ImportStmt, PromptDef, AgentDef, ToolDef)):
            pass

    def visit_var_decl(self, stmt: VarDeclStmt):
        declared_type = normalize_type_annotation(stmt.type_annot)
        inferred_type = self.infer_expr_type(stmt.value)

        # 1. Type compatibility verification
        if declared_type is not None and inferred_type is not None:
            is_tensor_decl = (
                isinstance(declared_type, TensorType) or declared_type == "Tensor" or
                (isinstance(declared_type, GenericType) and declared_type.base == "Tensor")
            )
            if is_tensor_decl:
                is_tensor_val = isinstance(inferred_type, TensorType) or inferred_type == "Tensor"
                if not is_tensor_val:
                    self.report_error(
                        f"Type contract violation: Variable '{stmt.name}' declared as '{stmt.type_annot}' "
                        f"but assigned non-tensor expression of type '{inferred_type}'.",
                        node=stmt,
                        details={
                            "variable": stmt.name,
                            "declared_type": str(stmt.type_annot),
                            "assigned_type": str(inferred_type),
                        },
                    )
            else:
                if not are_types_compatible(inferred_type, declared_type, self.enums):
                    hint = None
                    if inferred_type in ("None", "none"):
                        hint = f"Cannot assign 'none' to non-optional type '{stmt.type_annot}'. Use 'Option[{stmt.type_annot}]' or 'Union[{stmt.type_annot}, None]' instead."
                        msg = (
                            f"Type contract violation: Cannot assign 'none' to non-optional type '{stmt.type_annot}'. "
                            f"Use 'Option[{stmt.type_annot}]' or 'Union[{stmt.type_annot}, None]' instead."
                        )
                    elif isinstance(declared_type, GenericType) and declared_type.base == "Option":
                        if not (isinstance(inferred_type, GenericType) and inferred_type.base == "Option"):
                            hint = f"Did you mean 'Some({self._expr_repr(stmt.value)})'?"
                        msg = (
                            f"Type contract violation: Variable '{stmt.name}' declared as '{stmt.type_annot}' "
                            f"but assigned expression of type '{inferred_type}'."
                        )
                    else:
                        msg = (
                            f"Type contract violation: Variable '{stmt.name}' declared as '{stmt.type_annot}' "
                            f"but assigned expression of type '{inferred_type}'."
                        )
                    if hint and inferred_type not in ("None", "none"):
                        msg += f" {hint}"

                    self.report_error(
                        msg,
                        node=stmt,
                        hint=hint,
                        details={
                            "variable": stmt.name,
                            "declared_type": str(stmt.type_annot),
                            "assigned_type": str(inferred_type),
                        },
                    )

        # 2. TensorType shape contract verification
        if stmt.tensor_type is not None:
            inferred_shape = self.infer_expr_shape(stmt.value)
            if inferred_shape is not None:
                if not stmt.tensor_type.matches_shape(inferred_shape):
                    self.report_error(
                        f"Type contract violation: Variable '{stmt.name}' declared with contract {stmt.tensor_type} "
                        f"(shape {stmt.tensor_type.dims}) but assigned expression of shape {inferred_shape}.",
                        node=stmt,
                        details={
                            "variable": stmt.name,
                            "contract": str(stmt.tensor_type),
                            "expected_shape": list(stmt.tensor_type.dims),
                            "actual_shape": list(inferred_shape),
                        },
                    )

        # 3. Known Option / Result state tracking
        known_state = self._extract_known_state(stmt.value)
        var_shape = stmt.to_shape_tuple() or self.infer_expr_shape(stmt.value)

        self.set_var(
            stmt.name,
            declared_type or inferred_type,
            shape=var_shape,
            known_state=known_state,
            is_const=stmt.is_const,
        )

    def visit_assign(self, stmt: AssignStmt):
        if isinstance(stmt.target, IdentifierExpr):
            name = stmt.target.name

            # Check const violation
            if name in self.const_vars:
                self.report_error(
                    f"Cannot reassign to constant variable '{name}'.",
                    node=stmt,
                )
                return

            existing_type = self.get_var_type(name)
            inferred_type = self.infer_expr_type(stmt.value)

            if existing_type is not None and inferred_type is not None:
                if not are_types_compatible(inferred_type, existing_type, self.enums):
                    hint = None
                    if inferred_type in ("None", "none"):
                        hint = f"Cannot assign 'none' to non-optional type '{existing_type}'. Use 'Option[{existing_type}]' or 'Union[{existing_type}, None]' instead."
                        msg = (
                            f"Type contract violation: Cannot assign 'none' to non-optional type '{existing_type}'. "
                            f"Use 'Option[{existing_type}]' or 'Union[{existing_type}, None]' instead."
                        )
                    else:
                        msg = (
                            f"Type contract violation: Reassignment to variable '{name}' with declared type '{existing_type}' "
                            f"with incompatible expression of type '{inferred_type}'."
                        )
                    if hint and inferred_type not in ("None", "none"):
                        msg += f" {hint}"
                    self.report_error(
                        msg,
                        node=stmt,
                        hint=hint,
                        details={
                            "variable": name,
                            "expected_type": str(existing_type),
                            "assigned_type": str(inferred_type),
                        },
                    )

            # Check tensor shape contract reassignment
            existing_shape = self.get_var_shape(name)
            if existing_shape is not None:
                val_shape = self.infer_expr_shape(stmt.value)
                if val_shape is not None and val_shape != existing_shape:
                    self.report_error(
                        f"Type contract violation: Reassignment to tensor '{name}' with shape {existing_shape} "
                        f"with incompatible shape {val_shape}.",
                        node=stmt,
                        details={
                            "variable": name,
                            "expected_shape": list(existing_shape),
                            "actual_shape": list(val_shape),
                        },
                    )

            known_state = self._extract_known_state(stmt.value)
            self.update_var_state(name, known_state)
        else:
            # Reassignment to indexed or member expression
            self.infer_expr_type(stmt.target)
            self.infer_expr_type(stmt.value)

    def visit_function_def(self, stmt: FunctionDef):
        sig = self.functions.get(stmt.name)
        self.push_scope()
        self.current_fn = sig

        for p in stmt.params:
            norm_t = normalize_type_annotation(p.type_annot)
            self.set_var(p.name, norm_t, shape=p.to_shape_tuple())

        for s in stmt.body:
            self.visit_stmt(s)

        self.current_fn = None
        self.pop_scope()

    def visit_return(self, stmt: ReturnStmt):
        val_type = self.infer_expr_type(stmt.value) if stmt.value else "None"
        if self.current_fn and self.current_fn.return_type is not None:
            decl_ret = self.current_fn.return_type
            if not are_types_compatible(val_type, decl_ret, self.enums):
                self.report_error(
                    f"Return type mismatch in function '{self.current_fn.name}': "
                    f"declared return type '{decl_ret}' but returned expression of type '{val_type}'.",
                    node=stmt,
                    details={
                        "function": self.current_fn.name,
                        "declared_return": str(decl_ret),
                        "actual_return": str(val_type),
                    },
                )

    def visit_if(self, stmt: IfStmt):
        self.infer_expr_type(stmt.condition)
        self.push_scope()
        for s in stmt.then_branch:
            self.visit_stmt(s)
        self.pop_scope()

        for cond, branch in stmt.elif_branches:
            self.infer_expr_type(cond)
            self.push_scope()
            for s in branch:
                self.visit_stmt(s)
            self.pop_scope()

        if stmt.else_branch:
            self.push_scope()
            for s in stmt.else_branch:
                self.visit_stmt(s)
            self.pop_scope()

    def visit_while(self, stmt: WhileStmt):
        self.infer_expr_type(stmt.condition)
        self.push_scope()
        for s in stmt.body:
            self.visit_stmt(s)
        self.pop_scope()

    def visit_for(self, stmt: ForStmt):
        iter_type = self.infer_expr_type(stmt.iterable)
        elem_type = "Any"
        if isinstance(iter_type, GenericType) and iter_type.base == "List" and iter_type.type_args:
            elem_type = iter_type.type_args[0]
        self.push_scope()
        self.set_var(stmt.target, elem_type)
        for s in stmt.body:
            self.visit_stmt(s)
        self.pop_scope()

    # -------------------------------------------------------------------------
    # Pattern Matching & Exhaustiveness Checker (MatchStmt)
    # -------------------------------------------------------------------------
    def visit_match_stmt(self, stmt: MatchStmt) -> Any:
        """
        Performs static type checking and exhaustiveness validation for MatchStmt.
        Enforces that Option, Result, and Enum subjects have all variants handled,
        or provide a catch-all/wildcard pattern. Extracts inner pattern variable types
        and reconciles branch return types.
        """
        # 1. Determine subject type
        if isinstance(stmt.subject, EnumDeclStmt):
            subject_type = stmt.subject
        else:
            subject_type = self.infer_expr_type(stmt.subject)

        norm_subj = normalize_type_annotation(subject_type)
        if isinstance(norm_subj, str):
            norm_subj = normalize_type_str(norm_subj)

        is_option = False
        is_result = False
        is_enum = False
        enum_name = ""
        enum_variants: list[str] = []

        if isinstance(norm_subj, GenericType):
            if norm_subj.base == "Option":
                is_option = True
            elif norm_subj.base == "Result":
                is_result = True
        elif isinstance(norm_subj, str):
            if norm_subj == "Option" or norm_subj.startswith("Option["):
                is_option = True
            elif norm_subj == "Result" or norm_subj.startswith("Result["):
                is_result = True
            elif norm_subj.startswith("EnumMeta:"):
                clean = norm_subj.split(":", 1)[1]
                if clean in self.enums:
                    is_enum = True
                    enum_name = clean
                    enum_variants = sorted(list(self.enums[clean]))
            elif norm_subj.startswith("enum "):
                clean = norm_subj.split(" ", 1)[1]
                if clean in self.enums:
                    is_enum = True
                    enum_name = clean
                    enum_variants = sorted(list(self.enums[clean]))
            elif norm_subj in self.enums:
                is_enum = True
                enum_name = norm_subj
                enum_variants = sorted(list(self.enums[norm_subj]))
        elif isinstance(norm_subj, EnumVariantType):
            is_enum = True
            enum_name = norm_subj.enum_name
            enum_variants = sorted(list(self.enums.get(enum_name, [])))
        elif isinstance(stmt.subject, EnumDeclStmt):
            is_enum = True
            enum_name = stmt.subject.name
            enum_variants = sorted(list(stmt.subject.variants))
        elif isinstance(norm_subj, EnumDeclStmt):
            is_enum = True
            enum_name = norm_subj.name
            enum_variants = sorted(list(norm_subj.variants))

        # Check if subject is an IdentifierExpr referring to an enum
        if not is_enum and not is_option and not is_result:
            if isinstance(stmt.subject, IdentifierExpr):
                var_t = self.get_var_type(stmt.subject.name)
                if var_t and isinstance(var_t, str) and var_t in self.enums:
                    is_enum = True
                    enum_name = var_t
                    enum_variants = sorted(list(self.enums[var_t]))
                elif stmt.subject.name in self.enums:
                    is_enum = True
                    enum_name = stmt.subject.name
                    enum_variants = sorted(list(self.enums[stmt.subject.name]))

        # 2. Normalize cases into (pattern, body, guard)
        norm_cases: list[tuple[Any, list[Any], Optional[Expr]]] = []
        for c in stmt.cases:
            if isinstance(c, tuple):
                p = c[0]
                b = c[1] if isinstance(c[1], list) else [c[1]]
                g = c[2] if len(c) > 2 else None
                norm_cases.append((p, b, g))
            else:
                p = getattr(c, "pattern", None)
                b = getattr(c, "body", [])
                if not isinstance(b, list):
                    b = [b]
                g = getattr(c, "guard", None)
                norm_cases.append((p, b, g))

        # 3. Classify patterns and check exhaustiveness
        covered_variants: set[str] = set()
        has_wildcard = False
        classified_cases: list[tuple[str, str, list[str], list[Any], Optional[Expr]]] = []

        for p, b, g in norm_cases:
            p_kind, v_name, bound_vars = self._classify_pattern(
                p, is_option=is_option, is_result=is_result, is_enum=is_enum, enum_variants=enum_variants
            )
            classified_cases.append((p_kind, v_name, bound_vars, b, g))

            if g is None:
                if p_kind in ("WILDCARD", "VARIABLE"):
                    has_wildcard = True
                elif p_kind == "OPTION_SOME":
                    covered_variants.add("Some")
                elif p_kind == "OPTION_NONE":
                    covered_variants.add("None")
                elif p_kind == "RESULT_OK":
                    covered_variants.add("Ok")
                elif p_kind == "RESULT_ERR":
                    covered_variants.add("Err")
                elif p_kind == "ENUM_VARIANT":
                    covered_variants.add(v_name)

        # Validate exhaustiveness
        if is_option:
            expected = {"Some", "None"}
            if not has_wildcard and not expected.issubset(covered_variants):
                missing = [v for v in ["Some", "None"] if v not in covered_variants]
                msg = f"Non-exhaustive pattern match on Option: missing variant(s): {', '.join(missing)}."
                self.report_error(msg, node=stmt, error_type="NonExhaustiveMatchError")
                raise NonExhaustiveMatchError(msg, missing_variants=missing)

        elif is_result:
            expected = {"Ok", "Err"}
            if not has_wildcard and not expected.issubset(covered_variants):
                missing = [v for v in ["Ok", "Err"] if v not in covered_variants]
                msg = f"Non-exhaustive pattern match on Result: missing variant(s): {', '.join(missing)}."
                self.report_error(msg, node=stmt, error_type="NonExhaustiveMatchError")
                raise NonExhaustiveMatchError(msg, missing_variants=missing)

        elif is_enum:
            expected = set(enum_variants)
            if not has_wildcard and not expected.issubset(covered_variants):
                missing = [v for v in enum_variants if v not in covered_variants]
                msg = f"Non-exhaustive pattern match on enum '{enum_name}': missing variant(s): {', '.join(missing)}."
                self.report_error(msg, node=stmt, error_type="NonExhaustiveMatchError")
                raise NonExhaustiveMatchError(msg, missing_variants=missing)

        # 4. Scope analysis, variable binding & case branch body checking
        branch_types: list[Any] = []

        for p_kind, v_name, bound_vars, body, guard in classified_cases:
            self.push_scope()
            try:
                # Bind inner pattern variables
                if is_option:
                    inner_t = "Any"
                    if isinstance(norm_subj, GenericType) and norm_subj.type_args:
                        inner_t = norm_subj.type_args[0]
                        if isinstance(inner_t, TypeAnnotation):
                            inner_t = inner_t.raw
                    if p_kind == "OPTION_SOME" and bound_vars:
                        self.set_var(bound_vars[0], inner_t)
                    elif p_kind == "VARIABLE" and bound_vars:
                        self.set_var(bound_vars[0], norm_subj)

                elif is_result:
                    ok_t = "Any"
                    err_t = "Any"
                    if isinstance(norm_subj, GenericType):
                        if len(norm_subj.type_args) > 0:
                            ok_t = norm_subj.type_args[0]
                            if isinstance(ok_t, TypeAnnotation):
                                ok_t = ok_t.raw
                        if len(norm_subj.type_args) > 1:
                            err_t = norm_subj.type_args[1]
                            if isinstance(err_t, TypeAnnotation):
                                err_t = err_t.raw
                    if p_kind == "RESULT_OK" and bound_vars:
                        self.set_var(bound_vars[0], ok_t)
                    elif p_kind == "RESULT_ERR" and bound_vars:
                        self.set_var(bound_vars[0], err_t)
                    elif p_kind == "VARIABLE" and bound_vars:
                        self.set_var(bound_vars[0], norm_subj)

                elif is_enum:
                    if p_kind == "VARIABLE" and bound_vars:
                        self.set_var(bound_vars[0], enum_name or norm_subj)

                else:
                    if p_kind == "VARIABLE" and bound_vars:
                        self.set_var(bound_vars[0], norm_subj or "Any")

                if guard is not None:
                    self.infer_expr_type(guard)

                branch_t = self._check_case_body(body)
                branch_types.append(branch_t)
            finally:
                self.pop_scope()

        # 5. Reconcile branch return types
        reconciled_type = None
        for bt in branch_types:
            if bt is None:
                continue
            if reconciled_type is None:
                reconciled_type = bt
            else:
                merged = self._reconcile_match_types(reconciled_type, bt)
                if merged is None:
                    msg = (
                        f"Type mismatch: Pattern match cases do not reconcile to a common return type: "
                        f"incompatible types '{reconciled_type}' and '{bt}'."
                    )
                    self.report_error(msg, node=stmt, error_type="TypeMismatchError")
                    raise TypeError(msg)
                reconciled_type = merged

        if reconciled_type is None:
            reconciled_type = "None"

        return reconciled_type

    def check_match_stmt(self, stmt: MatchStmt) -> Any:
        return self.visit_match_stmt(stmt)

    def check_match(self, stmt: MatchStmt) -> Any:
        return self.visit_match_stmt(stmt)

    def _classify_pattern(
        self,
        pattern: Any,
        is_option: bool,
        is_result: bool,
        is_enum: bool,
        enum_variants: list[str],
    ) -> tuple[str, str, list[str]]:
        """
        Classifies pattern into (pattern_kind, variant_name, bound_var_names).
        Supported kinds:
          'WILDCARD', 'VARIABLE', 'OPTION_SOME', 'OPTION_NONE',
          'RESULT_OK', 'RESULT_ERR', 'ENUM_VARIANT', 'OTHER'
        """
        # 1. Direct ADT instance objects
        if isinstance(pattern, Some):
            v_val = pattern.value
            b_vars = [v_val.name if isinstance(v_val, IdentifierExpr) else str(v_val)] if v_val is not None else []
            return ("OPTION_SOME", "Some", b_vars)
        if isinstance(pattern, (NoneOption.__class__, type(None))):
            return ("OPTION_NONE", "None", [])
        if isinstance(pattern, Ok):
            v_val = pattern.value
            b_vars = [v_val.name if isinstance(v_val, IdentifierExpr) else str(v_val)] if v_val is not None else []
            return ("RESULT_OK", "Ok", b_vars)
        if isinstance(pattern, Err):
            v_val = pattern.error
            b_vars = [v_val.name if isinstance(v_val, IdentifierExpr) else str(v_val)] if v_val is not None else []
            return ("RESULT_ERR", "Err", b_vars)

        # 2. String representation
        if isinstance(pattern, str):
            p_str = pattern.strip()
            if p_str == "_":
                return ("WILDCARD", "_", [])
            if p_str in ("None", "Option.None", "None_"):
                return ("OPTION_NONE", "None", [])
            if p_str.startswith("Some") or p_str.startswith("Option.Some"):
                b_vars = []
                if "(" in p_str and p_str.endswith(")"):
                    inner = p_str[p_str.index("(") + 1:-1].strip()
                    if inner and inner != "_":
                        b_vars.append(inner)
                return ("OPTION_SOME", "Some", b_vars)
            if p_str.startswith("Ok") or p_str.startswith("Result.Ok"):
                b_vars = []
                if "(" in p_str and p_str.endswith(")"):
                    inner = p_str[p_str.index("(") + 1:-1].strip()
                    if inner and inner != "_":
                        b_vars.append(inner)
                return ("RESULT_OK", "Ok", b_vars)
            if p_str.startswith("Err") or p_str.startswith("Result.Err"):
                b_vars = []
                if "(" in p_str and p_str.endswith(")"):
                    inner = p_str[p_str.index("(") + 1:-1].strip()
                    if inner and inner != "_":
                        b_vars.append(inner)
                return ("RESULT_ERR", "Err", b_vars)
            if is_enum:
                v_name = p_str.split(".")[-1]
                if v_name in enum_variants:
                    return ("ENUM_VARIANT", v_name, [])
            if p_str.isidentifier() and p_str not in ("None", "Some", "Ok", "Err"):
                return ("VARIABLE", p_str, [p_str])
            return ("OTHER", p_str, [])

        # 3. IdentifierExpr
        if isinstance(pattern, IdentifierExpr):
            name = pattern.name
            if name == "_":
                return ("WILDCARD", "_", [])
            if name in ("None", "None_", "none"):
                return ("OPTION_NONE", "None", [])
            if name == "Some":
                return ("OPTION_SOME", "Some", [])
            if name == "Ok":
                return ("RESULT_OK", "Ok", [])
            if name == "Err":
                return ("RESULT_ERR", "Err", [])
            if is_enum and name in enum_variants:
                return ("ENUM_VARIANT", name, [])
            return ("VARIABLE", name, [name])

        # 4. MemberExpr
        if isinstance(pattern, MemberExpr):
            member = pattern.member
            if member in ("None", "None_"):
                return ("OPTION_NONE", "None", [])
            if member == "Some":
                return ("OPTION_SOME", "Some", [])
            if member == "Ok":
                return ("RESULT_OK", "Ok", [])
            if member == "Err":
                return ("RESULT_ERR", "Err", [])
            if is_enum and member in enum_variants:
                return ("ENUM_VARIANT", member, [])
            return ("OTHER", member, [])

        # 5. CallExpr
        if isinstance(pattern, CallExpr):
            callee_name = ""
            member_name = ""
            if isinstance(pattern.callee, IdentifierExpr):
                callee_name = pattern.callee.name
            elif isinstance(pattern.callee, MemberExpr):
                callee_name = f"{getattr(pattern.callee.target, 'name', '')}.{pattern.callee.member}"
                member_name = pattern.callee.member

            bound_vars = []
            for arg in pattern.args:
                if isinstance(arg, IdentifierExpr):
                    if arg.name != "_":
                        bound_vars.append(arg.name)
                elif isinstance(arg, str):
                    if arg != "_":
                        bound_vars.append(arg)

            if callee_name in ("Some", "Option.Some") or member_name == "Some":
                return ("OPTION_SOME", "Some", bound_vars)
            if callee_name in ("None", "Option.None") or member_name in ("None", "None_"):
                return ("OPTION_NONE", "None", bound_vars)
            if callee_name in ("Ok", "Result.Ok") or member_name == "Ok":
                return ("RESULT_OK", "Ok", bound_vars)
            if callee_name in ("Err", "Result.Err") or member_name == "Err":
                return ("RESULT_ERR", "Err", bound_vars)
            return ("OTHER", callee_name, bound_vars)

        # 6. LiteralExpr
        if isinstance(pattern, LiteralExpr):
            if pattern.value is None:
                return ("OPTION_NONE", "None", [])
            if pattern.value == "_":
                return ("WILDCARD", "_", [])
            return ("OTHER", str(pattern.value), [])

        return ("OTHER", str(pattern), [])

    def _check_case_body(self, body: list[Any]) -> Any:
        """Type checks all statements in a match case body and infers the branch type."""
        case_ret_type = None
        has_return = False

        for stmt in body:
            if isinstance(stmt, ReturnStmt):
                has_return = True
                ret_t = self.infer_expr_type(stmt.value) if stmt.value else "None"
                case_ret_type = ret_t
                if self.current_fn and self.current_fn.return_type is not None:
                    decl_ret = self.current_fn.return_type
                    if not are_types_compatible(ret_t, decl_ret, self.enums):
                        self.report_error(
                            f"Return type mismatch in match case: expected '{decl_ret}', got '{ret_t}'.",
                            node=stmt,
                        )
            elif isinstance(stmt, ExprStmt):
                expr_t = self.infer_expr_type(stmt.expr)
                if not has_return and case_ret_type is None:
                    case_ret_type = expr_t
            elif isinstance(stmt, Expr):
                expr_t = self.infer_expr_type(stmt)
                if not has_return and case_ret_type is None:
                    case_ret_type = expr_t
            elif isinstance(stmt, Stmt):
                self.visit_stmt(stmt)

        return case_ret_type if case_ret_type is not None else "None"

    def _reconcile_match_types(self, t1: Any, t2: Any) -> Optional[Any]:
        """Reconciles two match branch return types into a unified common type."""
        norm1 = normalize_type_annotation(t1)
        norm2 = normalize_type_annotation(t2)
        if isinstance(norm1, TypeAnnotation):
            norm1 = norm1.raw
        if isinstance(norm2, TypeAnnotation):
            norm2 = norm2.raw

        if norm1 == norm2:
            return norm1
        if norm1 in (None, "Any"):
            return norm2
        if norm2 in (None, "Any"):
            return norm1
        if are_types_compatible(norm1, norm2, self.enums):
            return norm2
        if are_types_compatible(norm2, norm1, self.enums):
            return norm1
        return None

    # -------------------------------------------------------------------------
    # '?' TryExpr Error Propagation Type Checker
    # -------------------------------------------------------------------------
    def _infer_try_expr(self, expr: TryExpr) -> Any:
        """
        Infers and verifies 'TryExpr' ('?'):
        1. Ensures expr is of type Result[T, E] or Option[T].
           Otherwise raises TypeError("? operator can only be applied to Result or Option types").
        2. Ensures enclosing function returns Result[..., E] or Option[...].
           Otherwise raises TypeError.
        3. Resolves and returns inner type T.
        """
        target_type = self.infer_expr_type(expr.expr)
        norm_type = normalize_type_annotation(target_type)
        if isinstance(norm_type, str):
            norm_type = normalize_type_str(norm_type)

        is_result = False
        is_option = False

        if isinstance(norm_type, GenericType):
            if norm_type.base == "Result":
                is_result = True
            elif norm_type.base == "Option":
                is_option = True
        elif isinstance(norm_type, str):
            if norm_type == "Result" or norm_type.startswith("Result["):
                is_result = True
            elif norm_type == "Option" or norm_type.startswith("Option["):
                is_option = True

        if not is_result and not is_option:
            raise TypeError("? operator can only be applied to Result or Option types")

        # Function context check
        if self.current_fn is not None:
            decl_ret = self.current_fn.return_type
            if decl_ret is not None and decl_ret != "Any":
                norm_fn_ret = normalize_type_annotation(decl_ret)
                if isinstance(norm_fn_ret, str):
                    norm_fn_ret = normalize_type_str(norm_fn_ret)

                if is_result:
                    fn_is_result = (
                        (isinstance(norm_fn_ret, GenericType) and norm_fn_ret.base == "Result")
                        or (isinstance(norm_fn_ret, str) and (norm_fn_ret == "Result" or norm_fn_ret.startswith("Result[")))
                    )
                    if not fn_is_result:
                        raise TypeError(
                            f"Type contract violation: '?' operator used on Result expression inside function "
                            f"'{self.current_fn.name}' which declares non-Result return type '{decl_ret}'."
                        )
                    if isinstance(norm_fn_ret, GenericType) and len(norm_fn_ret.type_args) > 1:
                        fn_e = norm_fn_ret.type_args[1]
                        if isinstance(norm_type, GenericType) and len(norm_type.type_args) > 1:
                            expr_e = norm_type.type_args[1]
                            if not are_types_compatible(expr_e, fn_e, self.enums):
                                raise TypeError(
                                    f"Type mismatch: '?' operator propagates error of type '{expr_e}', "
                                    f"which is incompatible with function '{self.current_fn.name}' error return type '{fn_e}'."
                                )
                elif is_option:
                    fn_is_option = (
                        (isinstance(norm_fn_ret, GenericType) and norm_fn_ret.base == "Option")
                        or (isinstance(norm_fn_ret, str) and (norm_fn_ret == "Option" or norm_fn_ret.startswith("Option[")))
                    )
                    if not fn_is_option:
                        raise TypeError(
                            f"Type contract violation: '?' operator used on Option expression inside function "
                            f"'{self.current_fn.name}' which declares non-Option return type '{decl_ret}'."
                        )

        # Resolved type is inner T
        resolved_t: Any = "Any"
        if isinstance(norm_type, GenericType) and norm_type.type_args:
            resolved_t = norm_type.type_args[0]
            if isinstance(resolved_t, TypeAnnotation):
                resolved_t = resolved_t.raw

        return resolved_t

    def infer_try_expr(self, expr: TryExpr) -> Any:
        return self._infer_try_expr(expr)

    # -------------------------------------------------------------------------
    # Expression Type Inference & Validation
    # -------------------------------------------------------------------------
    def infer_expr_type(self, expr: Optional[Expr]) -> Any:
        if expr is None:
            return None

        # 0. TryExpr and MatchStmt
        if isinstance(expr, TryExpr):
            return self._infer_try_expr(expr)
        if isinstance(expr, MatchStmt):
            return self.visit_match_stmt(expr)

        # 1. Literal Expressions
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, int):
                return "int"
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, str):
                return "str"
            if expr.value is None:
                return "None"
            return "Any"

        # 2. Identifier Expressions
        if isinstance(expr, IdentifierExpr):
            name = expr.name
            if name in ("True", "False", "true", "false"):
                return "bool"
            if name in ("None", "none"):
                return "None"
            if name in self.enums:
                return f"EnumMeta:{name}"
            var_t = self.get_var_type(name)
            return var_t

        # 3. List Literals
        if isinstance(expr, ListLiteralExpr):
            if not expr.elements:
                return GenericType(base="List", type_args=[TypeAnnotation("Any")])
            elem_types = [self.infer_expr_type(e) for e in expr.elements]
            first_t = elem_types[0]
            if all(t == first_t for t in elem_types):
                return GenericType(base="List", type_args=[TypeAnnotation(str(first_t))])
            unique_types = list(dict.fromkeys(str(t) for t in elem_types))
            union = UnionType(types=[TypeAnnotation(u) for u in unique_types])
            return GenericType(base="List", type_args=[union])

        # 4. Dict Literals
        if isinstance(expr, DictLiteralExpr):
            if not expr.entries:
                return GenericType(base="Dict", type_args=[TypeAnnotation("Any"), TypeAnnotation("Any")])
            key_types = [self.infer_expr_type(k) for k, _ in expr.entries]
            val_types = [self.infer_expr_type(v) for _, v in expr.entries]
            k_t = key_types[0] if len(key_types) > 0 else "Any"
            v_t = val_types[0] if len(val_types) > 0 else "Any"
            return GenericType(base="Dict", type_args=[TypeAnnotation(str(k_t)), TypeAnnotation(str(v_t))])

        # 5. Tensor Literals
        if isinstance(expr, TensorLiteralExpr):
            shape = self.infer_expr_shape(expr) or ()
            return TensorType(dims=shape, dtype="float32")

        # 6. Member Expressions (Enum variants, properties, methods)
        if isinstance(expr, MemberExpr):
            target_name = getattr(expr.target, "name", None)
            # Enum variant access: Color.Red
            if target_name and target_name in self.enums:
                valid_variants = self.enums[target_name]
                if expr.member not in valid_variants:
                    self.report_error(
                        f"Invalid variant '{expr.member}' for enum '{target_name}'. "
                        f"Valid variants: {', '.join(sorted(valid_variants))}.",
                        node=expr,
                    )
                    return None
                return EnumVariantType(enum_name=target_name, variant=expr.member)

            # Tensor transpose property: t.T
            target_type = self.infer_expr_type(expr.target)
            if expr.member == "T" and isinstance(target_type, TensorType):
                shape = target_type.dims
                if len(shape) >= 2:
                    return TensorType(dims=shape[:-2] + (shape[-1], shape[-2]), dtype=target_type.dtype)
                return target_type

            return "Any"

        # 7. Call Expressions (Functions, Constructors, Unwrapping)
        if isinstance(expr, CallExpr):
            return self._infer_call_type(expr)

        # 8. Binary Expressions
        if isinstance(expr, BinaryExpr):
            return self._infer_binary_type(expr)

        # 9. Unary Expressions
        if isinstance(expr, UnaryExpr):
            op_t = self.infer_expr_type(expr.operand)
            if expr.op == "not":
                return "bool"
            if expr.op in ("-", "+"):
                return op_t
            return "Any"

        # 10. Pipeline Expressions: x |> f
        if isinstance(expr, PipeExpr):
            left_t = self.infer_expr_type(expr.left)
            # Right side is typically a function or call
            if isinstance(expr.right, IdentifierExpr):
                fn_name = expr.right.name
                if fn_name in self.functions:
                    fn_sig = self.functions[fn_name]
                    if fn_sig.params and fn_sig.params[0][1] is not None:
                        expected_param = fn_sig.params[0][1]
                        if not are_types_compatible(left_t, expected_param, self.enums):
                            self.report_error(
                                f"Pipeline type mismatch: function '{fn_name}' expects '{expected_param}' "
                                f"as first argument, but received '{left_t}'.",
                                node=expr,
                            )
                    return fn_sig.return_type or "Any"
            return self.infer_expr_type(expr.right)

        # 11. Index Expressions: arr[idx]
        if isinstance(expr, IndexExpr):
            target_t = self.infer_expr_type(expr.target)
            if isinstance(target_t, GenericType):
                if target_t.base == "List" and target_t.type_args:
                    return target_t.type_args[0]
                if target_t.base == "Dict" and len(target_t.type_args) > 1:
                    return target_t.type_args[1]
            return "Any"

        return "Any"

    def _infer_call_type(self, expr: CallExpr) -> Any:
        callee_name = getattr(expr.callee, "name", None)

        # Option / Result Constructors
        if callee_name == "Some":
            arg_t = self.infer_expr_type(expr.args[0]) if expr.args else "Any"
            return GenericType(base="Option", type_args=[TypeAnnotation(str(arg_t))])

        if callee_name == "Ok":
            arg_t = self.infer_expr_type(expr.args[0]) if expr.args else "Any"
            return GenericType(base="Result", type_args=[TypeAnnotation(str(arg_t)), TypeAnnotation("Any")])

        if callee_name == "Err":
            arg_t = self.infer_expr_type(expr.args[0]) if expr.args else "Any"
            return GenericType(base="Result", type_args=[TypeAnnotation("Any"), TypeAnnotation(str(arg_t))])

        if callee_name == "Option":
            arg_t = self.infer_expr_type(expr.args[0]) if expr.args else "Any"
            return GenericType(base="Option", type_args=[TypeAnnotation(str(arg_t))])

        # Method calls: target.method(...)
        if isinstance(expr.callee, MemberExpr):
            target = expr.callee.target
            method = expr.callee.member
            target_t = self.infer_expr_type(target)

            # Option / Result safe unwrapping checks
            if method == "unwrap":
                # Check for known None or Err panic
                if isinstance(target, IdentifierExpr):
                    known = self.get_var_known_state(target.name)
                    if known == "None":
                        self.report_error(
                            f"Unsafe unwrap: Variable '{target.name}' is known to be None. "
                            f"Calling unwrap() will panic at runtime.",
                            node=expr,
                        )
                    elif known == "Err":
                        self.report_error(
                            f"Unsafe unwrap: Variable '{target.name}' is known to be Err. "
                            f"Calling unwrap() will panic at runtime.",
                            node=expr,
                        )

                if isinstance(target_t, GenericType) and target_t.base in ("Option", "Result"):
                    return target_t.type_args[0] if target_t.type_args else "Any"
                elif target_t is not None and target_t != "Any":
                    self.report_error(
                        f"Cannot call 'unwrap()' on type '{target_t}': only Option and Result types can be unwrapped.",
                        node=expr,
                    )
                    return "Any"

            elif method == "unwrap_or":
                if isinstance(target_t, GenericType) and target_t.base in ("Option", "Result"):
                    inner_t = target_t.type_args[0] if target_t.type_args else "Any"
                    if expr.args:
                        default_t = self.infer_expr_type(expr.args[0])
                        if not are_types_compatible(default_t, inner_t, self.enums):
                            self.report_error(
                                f"Type contract violation in 'unwrap_or': default value of type '{default_t}' "
                                f"does not match expected inner type '{inner_t}'.",
                                node=expr.args[0],
                            )
                    return inner_t

            elif method in ("is_some", "is_none", "is_ok", "is_err"):
                return "bool"

            elif method == "unwrap_err":
                if isinstance(target_t, GenericType) and target_t.base == "Result":
                    return target_t.type_args[1] if len(target_t.type_args) > 1 else "Any"
                return "Any"

        # Tensor creation functions
        if callee_name in ("tensor", "zeros", "ones", "randn", "empty"):
            shape = self.infer_expr_shape(expr) or ()
            return TensorType(dims=shape, dtype="float32")

        # User defined functions
        if callee_name and callee_name in self.functions:
            fn_sig = self.functions[callee_name]
            # Verify parameter types against arguments
            for i, (param_name, param_type) in enumerate(fn_sig.params):
                if i < len(expr.args) and param_type is not None:
                    arg_expr = expr.args[i]
                    arg_t = self.infer_expr_type(arg_expr)
                    if not are_types_compatible(arg_t, param_type, self.enums):
                        self.report_error(
                            f"Argument type mismatch for parameter '{param_name}' of function '{callee_name}': "
                            f"expected '{param_type}', got '{arg_t}'.",
                            node=arg_expr,
                            details={
                                "function": callee_name,
                                "parameter": param_name,
                                "expected_type": str(param_type),
                                "actual_type": str(arg_t),
                            },
                        )
            return fn_sig.return_type or "Any"

        return "Any"

    def _infer_binary_type(self, expr: BinaryExpr) -> Any:
        left_t = self.infer_expr_type(expr.left)
        right_t = self.infer_expr_type(expr.right)

        # Safe Unwrapping Enforcement: Reject arithmetic directly on Option/Result
        if isinstance(left_t, GenericType) and left_t.base in ("Option", "Result"):
            if expr.op in ("+", "-", "*", "/", "//", "%", "**", "@", "<", "<=", ">", ">="):
                self.report_error(
                    f"Cannot perform binary operation '{expr.op}' directly on '{left_t}'. "
                    f"Option/Result values must be safely unwrapped before operation.",
                    node=expr.left,
                )
        if isinstance(right_t, GenericType) and right_t.base in ("Option", "Result"):
            if expr.op in ("+", "-", "*", "/", "//", "%", "**", "@", "<", "<=", ">", ">="):
                self.report_error(
                    f"Cannot perform binary operation '{expr.op}' directly on '{right_t}'. "
                    f"Option/Result values must be safely unwrapped before operation.",
                    node=expr.right,
                )

        # Tensor Matrix Multiplication (@) shape verification
        if expr.op == "@":
            s1 = self.infer_expr_shape(expr.left)
            s2 = self.infer_expr_shape(expr.right)
            if s1 is not None and s2 is not None and len(s1) >= 2 and len(s2) >= 2:
                k1 = s1[-1]
                k2 = s2[-2]
                if isinstance(k1, int) and isinstance(k2, int) and k1 != k2:
                    self.report_error(
                        f"Matrix multiplication shape mismatch: cannot multiply shapes {s1} and {s2}. "
                        f"Inner dimensions ({k1} vs {k2}) do not match.",
                        node=expr,
                    )
                out_shape = s1[:-1] + (s2[-1],)
                return TensorType(dims=out_shape, dtype="float32")
            return TensorType(dims=(), dtype="float32")

        # Comparisons
        if expr.op in ("==", "!=", "<", "<=", ">", ">="):
            return "bool"

        # Logical
        if expr.op in ("and", "or"):
            return "bool"

        # Bitwise OR
        if expr.op == "|":
            if left_t == "int" and right_t == "int":
                return "int"
            return "int"

        # Arithmetic
        if expr.op == "/":
            return "float"

        if expr.op in ("+", "-", "*", "//", "%", "**"):
            if left_t == "str" and right_t == "str" and expr.op == "+":
                return "str"
            if left_t == "float" or right_t == "float":
                return "float"
            if left_t == "int" and right_t == "int":
                return "int"

        return "Any"

    # -------------------------------------------------------------------------
    # Shape Inference Integration
    # -------------------------------------------------------------------------
    def infer_expr_shape(self, expr: Any) -> Optional[tuple[Union[int, str], ...]]:
        current_env: dict[str, tuple[Union[int, str], ...]] = {}
        for s in self.tensor_shapes:
            current_env.update(s)
        return _infer_expr_shape(expr, current_env)

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------
    def _extract_known_state(self, expr: Expr) -> Any:
        if isinstance(expr, LiteralExpr) and expr.value is None:
            return "None"
        if isinstance(expr, IdentifierExpr) and expr.name in ("None", "none"):
            return "None"
        if isinstance(expr, CallExpr):
            callee_name = getattr(expr.callee, "name", None)
            if callee_name == "Some":
                return "Some"
            if callee_name == "Ok":
                return "Ok"
            if callee_name == "Err":
                return "Err"
        return None

    def _expr_repr(self, expr: Expr) -> str:
        if isinstance(expr, LiteralExpr):
            return repr(expr.value)
        if isinstance(expr, IdentifierExpr):
            return expr.name
        return "..."


# =============================================================================
# Public API Functions
# =============================================================================

def check_source(source: str, filepath: str = "<source>") -> TypeCheckResult:
    """
    Analyzes Synapse source code and returns a structured TypeCheckResult.
    Catches lexer errors, syntax errors, and performs comprehensive static type checking.
    """
    # 1. Lexer Phase
    try:
        tokens = Lexer(source).tokenize()
    except LexerError as e:
        source_line = _extract_source_line(source, e.line)
        pointer = _generate_pointer(e.column)
        report = DiagnosticReport(
            status="error",
            error_type="LexerError",
            message=str(e),
            line=e.line,
            column=e.column,
            source_line=source_line,
            pointer=pointer,
        )
        return TypeCheckResult(errors=[report], is_valid=False)

    # 2. Parser Phase
    try:
        ast = Parser(tokens).parse()
    except ParseError as e:
        source_line = _extract_source_line(source, e.token.line)
        pointer = _generate_pointer(e.token.column)
        report = DiagnosticReport(
            status="error",
            error_type="ParseError",
            message=str(e),
            line=e.token.line,
            column=e.token.column,
            source_line=source_line,
            pointer=pointer,
        )
        return TypeCheckResult(errors=[report], is_valid=False)

    # 3. Static Type Checking Phase
    checker = StaticTypeChecker(source=source, filepath=filepath)
    return checker.check_program(ast)


def handle_check_cli(filepath: str, verbose: bool = False) -> int:
    """
    CLI handler for static type checking: reads the file, runs `check_source`,
    prints formatted diagnostics with line pointers and colors, and returns exit code 0 or 1.
    """
    if not os.path.exists(filepath):
        print(f"\033[91mError: File '{filepath}' not found.\033[0m", file=sys.stderr)
        return 1

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"\033[91mError reading file '{filepath}': {e}\033[0m", file=sys.stderr)
        return 1

    result = check_source(source, filepath=filepath)

    if result.is_valid:
        print(f"\033[92m[PASS] [SYNAPSE TYPE CHECK] All type contracts satisfied for '{filepath}'.\033[0m")
        if verbose:
            lines_count = len(source.splitlines())
            print(f"  Verified {lines_count} lines. 0 errors, {len(result.warnings)} warnings.")
        return 0
    else:
        print(f"\033[91m[FAIL] [SYNAPSE TYPE CHECK] Found {len(result.errors)} type error(s) in '{filepath}':\033[0m")
        print("=" * 70)
        for idx, err in enumerate(result.errors, start=1):
            print(f"\033[91m[{idx}] {err.error_type} at {filepath}:{err.line}:{err.column}\033[0m")
            if err.source_line:
                print(f"    {err.source_line}")
                print(f"    \033[93m{err.pointer}\033[0m")
            print(f"    \033[1mMessage:\033[0m {err.message}")
            if err.suggested_fix:
                print(f"    \033[96mSuggested Fix:\033[0m {err.suggested_fix}")
            if err.ai_prompt_hint:
                print(f"    \033[95mAI Hint:\033[0m {err.ai_prompt_hint}")
            if err.diff:
                print(f"    \033[90mDiff:\n{err.diff}\033[0m")
            print("-" * 70)
        return 1
