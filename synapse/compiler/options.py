"""
Synapse Compiler Options and Profile Management.

Provides configuration profiles for compiler determinism and isolation:
- CompilationProfile: STANDALONE (pure Synapse, no Python interop) vs HYBRID (Python interop allowed)
- CompilationMode: STRICT (strict syntax checks) vs TOLERANT (auto-healing syntax drift)
- CompilerOptions: Dataclass capturing compiler settings
- StandaloneViolationError / E_INTEROP_STANDALONE_VIOLATION: Raised when Python interop is used in STANDALONE profile
- check_profile_compliance: AST scanner to enforce profile constraints
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, List, Union
from synapse.parser.ast_nodes import (
    Program, Stmt, ImportStmt, FunctionDef, ToolDef, IfStmt, WhileStmt, ForStmt
)


class CompilationProfile(str, Enum):
    """Execution and compilation profile for Synapse code."""
    STANDALONE = "standalone"
    HYBRID = "hybrid"

    @classmethod
    def from_string(cls, val: Union[str, "CompilationProfile"]) -> "CompilationProfile":
        if isinstance(val, CompilationProfile):
            return val
        s = str(val).lower().strip()
        if s in ("standalone", "stand-alone", "std"):
            return cls.STANDALONE
        elif s in ("hybrid", "hyb"):
            return cls.HYBRID
        raise ValueError(f"Unknown CompilationProfile: '{val}'. Expected 'standalone' or 'hybrid'.")


class CompilationMode(str, Enum):
    """Compilation tolerance mode for Synapse code."""
    STRICT = "strict"
    TOLERANT = "tolerant"

    @classmethod
    def from_string(cls, val: Union[str, "CompilationMode"]) -> "CompilationMode":
        if isinstance(val, CompilationMode):
            return val
        s = str(val).lower().strip()
        if s in ("strict", "str"):
            return cls.STRICT
        elif s in ("tolerant", "tol", "ai_tolerant"):
            return cls.TOLERANT
        raise ValueError(f"Unknown CompilationMode: '{val}'. Expected 'strict' or 'tolerant'.")


class StandaloneViolationError(Exception):
    """
    Raised when Python interop (e.g. 'import py.*') is detected in a STANDALONE profile.
    Error Code: E_INTEROP_STANDALONE_VIOLATION
    """
    code: str = "E_INTEROP_STANDALONE_VIOLATION"

    def __init__(
        self,
        message: str,
        module: Optional[str] = None,
        line: int = 1,
        column: int = 1,
    ):
        super().__init__(message)
        self.message = message
        self.module = module
        self.line = line
        self.column = column
        self.error_code = self.code

    def __str__(self) -> str:
        loc = f" at line {self.line}:{self.column}" if self.line else ""
        return f"[{self.code}]{loc} {self.message}"


# Aliases for specification compliance
E_INTEROP_STANDALONE_VIOLATION = StandaloneViolationError


@dataclass
class CompilerOptions:
    """Configuration options for Synapse compiler and execution pipelines."""
    profile: CompilationProfile = CompilationProfile.STANDALONE
    mode: CompilationMode = CompilationMode.STRICT
    optimization_level: int = 0
    target: str = "native"
    enable_cache: bool = True

    def __post_init__(self):
        if isinstance(self.profile, str):
            self.profile = CompilationProfile.from_string(self.profile)
        if isinstance(self.mode, str):
            self.mode = CompilationMode.from_string(self.mode)


def _find_all_import_stmts(node: Any) -> List[ImportStmt]:
    """Recursively traverses the AST and gathers all ImportStmt instances."""
    results: List[ImportStmt] = []
    if node is None:
        return results

    if isinstance(node, ImportStmt):
        results.append(node)
        return results

    if isinstance(node, Program):
        for stmt in getattr(node, "statements", []):
            results.extend(_find_all_import_stmts(stmt))
    elif isinstance(node, (FunctionDef, ToolDef)):
        for stmt in getattr(node, "body", []):
            results.extend(_find_all_import_stmts(stmt))
    elif isinstance(node, IfStmt):
        for stmt in getattr(node, "then_branch", []):
            results.extend(_find_all_import_stmts(stmt))
        for _, elif_body in getattr(node, "elif_branches", []):
            for stmt in elif_body:
                results.extend(_find_all_import_stmts(stmt))
        if getattr(node, "else_branch", None):
            for stmt in node.else_branch:
                results.extend(_find_all_import_stmts(stmt))
    elif isinstance(node, (WhileStmt, ForStmt)):
        for stmt in getattr(node, "body", []):
            results.extend(_find_all_import_stmts(stmt))
    elif isinstance(node, list):
        for item in node:
            results.extend(_find_all_import_stmts(item))
    elif hasattr(node, "__dict__"):
        for key, val in node.__dict__.items():
            if isinstance(val, (Stmt, list)):
                results.extend(_find_all_import_stmts(val))

    return results


def check_profile_compliance(
    ast: Any,
    profile: Union[CompilationProfile, str] = CompilationProfile.STANDALONE,
) -> bool:
    """
    Scans the given AST to ensure compliance with the specified CompilationProfile.

    If profile is STANDALONE and any Python interop import (e.g. `import py.*` or `is_python=True`)
    is encountered, raises StandaloneViolationError.
    If profile is HYBRID, full Python interop is permitted.

    Returns True if compliant.
    """
    if isinstance(profile, str):
        profile = CompilationProfile.from_string(profile)

    if profile == CompilationProfile.HYBRID:
        return True

    # Standalone mode: Scan AST for python interop imports
    imports = _find_all_import_stmts(ast)
    for imp in imports:
        is_python = getattr(imp, "is_python", False)
        mod_parts = getattr(imp, "module_path", []) or []
        mod_name = ".".join(mod_parts)
        if is_python or (mod_parts and mod_parts[0] == "py"):
            display_path = f"py.{mod_name}" if is_python and not mod_name.startswith("py.") else mod_name
            msg = (
                f"Python interop ('import {display_path}') is prohibited in STANDALONE profile. "
                f"Specify '--profile=hybrid' to enable Python ecosystem interoperability."
            )
            raise StandaloneViolationError(
                message=msg,
                module=mod_name,
                line=getattr(imp, "line", 1),
                column=getattr(imp, "column", 1),
            )

    return True


def heal_and_execute(*args, **kwargs):
    """
    Backward-compatible re-export of heal_and_execute from synapse.core.diagnostics.
    """
    from synapse.core.diagnostics import heal_and_execute as _heal
    return _heal(*args, **kwargs)
