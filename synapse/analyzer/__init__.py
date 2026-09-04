"""
Synapse Analyzer Package
========================
Provides compile-time static analysis, tensor shape verification,
symbolic dimension checking, and type theory enforcement.
"""

from synapse.analyzer.shape_checker import (
    CompileTimeShapeMismatchError,
    StaticShapeChecker,
    check_shapes,
    verify_shapes,
)
from synapse.analyzer.type_checker import (
    TypeChecker,
    TypeCheckResult,
    check_source,
    check_types,
    handle_check_cli,
)

from synapse.analyzer.shape_guard import (
    SymbolicDimension,
    SymbolicShapeSolver,
    TensorShapeGuard,
    verify_shapes_in_file,
)

__all__ = [
    "CompileTimeShapeMismatchError",
    "StaticShapeChecker",
    "TypeChecker",
    "TypeCheckResult",
    "check_shapes",
    "verify_shapes",
    "check_source",
    "check_types",
    "handle_check_cli",
    "SymbolicDimension",
    "SymbolicShapeSolver",
    "TensorShapeGuard",
    "verify_shapes_in_file",
]
