"""
Synapse Compiler Package.
Provides AST caching, compilation utilities, and bytecode generation.
"""

from synapse.compiler.cache import ASTCache, clean_cache, clean
from synapse.compiler.options import (
    CompilationProfile,
    CompilationMode,
    CompilerOptions,
    StandaloneViolationError,
    E_INTEROP_STANDALONE_VIOLATION,
    check_profile_compliance,
    heal_and_execute,
)

__all__ = [
    "ASTCache",
    "clean_cache",
    "clean",
    "CompilationProfile",
    "CompilationMode",
    "CompilerOptions",
    "StandaloneViolationError",
    "E_INTEROP_STANDALONE_VIOLATION",
    "check_profile_compliance",
    "heal_and_execute",
]
