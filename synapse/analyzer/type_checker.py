"""
Synapse Unified Type & Shape Checker
====================================
Bridges synapse.core.type_checker with StaticShapeChecker to provide
an end-to-end compile-time static type and tensor shape analysis system.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Union

from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.parser.ast import Program
from synapse.core.diagnostics import (
    DiagnosticReport,
    TypeContractViolationError,
    _extract_source_line,
    _generate_pointer,
)
from synapse.core.type_checker import (
    StaticTypeChecker,
    TypeCheckResult,
    FunctionSignature,
    EnumVariantType,
    NonExhaustiveMatchError,
    MatchStmt,
    MatchCase,
    TryExpr,
    normalize_type_annotation,
    normalize_type_str,
    are_types_compatible,
    handle_check_cli as core_handle_check_cli,
)
from synapse.analyzer.shape_checker import (
    CompileTimeShapeMismatchError,
    StaticShapeChecker,
    check_shapes,
    verify_shapes,
)



class TypeChecker(StaticTypeChecker):
    """
    Unified Compile-Time Static Type and Tensor Shape Checker.
    Inherits from StaticTypeChecker and deeply integrates StaticShapeChecker
    for compile-time tensor algebra and symbolic shape verification.
    """

    def __init__(
        self,
        source: str = "",
        filepath: str = "<source>",
        raise_on_error: bool = False,
        raise_on_shape_error: bool = False,
    ):
        super().__init__(source=source, filepath=filepath)
        self.raise_on_error = raise_on_error
        self.raise_on_shape_error = raise_on_shape_error or raise_on_error
        self.shape_checker = StaticShapeChecker(
            source=source,
            filepath=filepath,
            raise_on_error=self.raise_on_shape_error,
        )

    def check_program(self, program: Program) -> TypeCheckResult:
        """
        Executes unified type and shape verification on the AST.
        Runs Pass 1 & 2 type checks, then executes static shape analysis.
        """
        # 1. Run core static type checking
        type_result = super().check_program(program)

        # 2. Run static shape analysis
        shape_diagnostics = self.shape_checker.check(
            program,
            raise_on_error=self.raise_on_shape_error,
        )

        # Merge diagnostics, deduplicating any overlapping reports
        combined_errors: list[DiagnosticReport] = list(type_result.errors)
        existing_signatures = {
            (e.line, e.column, e.message) for e in combined_errors
        }

        for report in shape_diagnostics:
            sig = (report.line, report.column, report.message)
            if sig not in existing_signatures:
                combined_errors.append(report)
                existing_signatures.add(sig)

        is_valid = len(combined_errors) == 0
        return TypeCheckResult(
            errors=combined_errors,
            is_valid=is_valid,
            warnings=type_result.warnings,
        )

    def check_match(self, stmt: MatchStmt) -> Any:
        """Type checks and verifies exhaustiveness for a pattern matching statement."""
        return self.visit_match_stmt(stmt)

    def check_match_stmt(self, stmt: MatchStmt) -> Any:
        """Alias for check_match."""
        return self.visit_match_stmt(stmt)

    def infer_try(self, expr: TryExpr) -> Any:
        """Infers resolved type for TryExpr ('?') and validates error propagation."""
        return self._infer_try_expr(expr)

    def verify(self, program: Program) -> None:
        """
        Verifies types and shapes, raising NonExhaustiveMatchError, TypeContractViolationError,
        or CompileTimeShapeMismatchError on any violation.
        """
        result = self.check_program(program)
        if not result.is_valid and result.errors:
            first_err = result.errors[0]
            if first_err.error_type == "NonExhaustiveMatchError":
                raise NonExhaustiveMatchError(first_err.message)
            elif first_err.error_type == "CompileTimeShapeMismatchError":
                raise CompileTimeShapeMismatchError(
                    message=first_err.message,
                    line=first_err.line,
                    column=first_err.column,
                    source_line=first_err.source_line,
                    pointer=first_err.pointer,
                    suggested_fix=first_err.suggested_fix,
                )
            else:
                raise TypeContractViolationError(
                    message=first_err.message,
                    line=first_err.line,
                    column=first_err.column,
                )



def check_source(
    source: str,
    filepath: str = "<source>",
    raise_on_error: bool = False,
    raise_on_shape_error: bool = False,
) -> TypeCheckResult:
    """
    Analyzes Synapse source code and returns a unified TypeCheckResult
    incorporating both general type checking and static tensor shape checking.
    """
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

    checker = TypeChecker(
        source=source,
        filepath=filepath,
        raise_on_error=raise_on_error,
        raise_on_shape_error=raise_on_shape_error,
    )
    return checker.check_program(ast)


check_types = check_source


def handle_check_cli(filepath: str, verbose: bool = False) -> int:
    """CLI handler for unified type & shape checking."""
    return core_handle_check_cli(filepath, verbose=verbose)
