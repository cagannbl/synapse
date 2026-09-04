"""
Unit and Integration Tests for SynapseStubGenerator (synapse-stubgen).
Validates Phase 1: Automated Type Stub & Interface Generator.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest

from synapse.tools import (
    SynapseStubGenerator,
    generate_stub_for_module,
    save_stub,
)
from synapse.tools.stubgen import SynapseTypeFormatter


# =============================================================================
# 1. Standard Library Introspection Tests
# =============================================================================

def test_stubgen_stdlib_math():
    """Validates introspection of C-extension math module."""
    stub = generate_stub_for_module("math")

    assert stub.startswith("// Synapse Interface Stub for: math")
    # Check constants
    assert "const pi = 3.14" in stub
    assert "const e = 2.71" in stub
    assert "const tau = 6.28" in stub
    # Check standard functions
    assert "fn sin(" in stub
    assert "fn cos(" in stub
    assert "fn sqrt(" in stub
    # Check built-in functions without inspect signature (e.g. hypot, log)
    assert "fn hypot(" in stub
    assert "fn log(" in stub


def test_stubgen_stdlib_math_from_module_object():
    """Validates introspection when passing the module object directly."""
    stub = generate_stub_for_module(math)
    assert "// Synapse Interface Stub for: math" in stub
    assert "fn tan(" in stub
    assert "const pi = " in stub


def test_stubgen_stdlib_json():
    """Validates introspection of standard library json module."""
    stub = generate_stub_for_module("json")

    assert stub.startswith("// Synapse Interface Stub for: json")
    # Functions
    assert "fn dumps(" in stub
    assert "fn loads(" in stub
    assert "fn dump(" in stub
    assert "fn load(" in stub
    # Classes
    assert "class JSONDecoder:" in stub
    assert "class JSONEncoder:" in stub
    assert "class JSONDecodeError(ValueError):" in stub


def test_stubgen_stdlib_os_path():
    """Validates introspection of standard library os.path module."""
    stub = generate_stub_for_module("os.path")

    assert stub.startswith("// Synapse Interface Stub for: os.path")
    # Path functions
    assert "fn join(" in stub
    assert "fn split(" in stub
    assert "fn abspath(" in stub
    assert "fn exists(" in stub
    # Path constants
    assert 'const curdir = "."' in stub
    assert "const sep = " in stub


# =============================================================================
# 2. Custom Class Introspection Tests
# =============================================================================

class SampleVector2D:
    """A 2D geometric vector representation."""
    x: float
    y: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        """Initialize coordinates."""
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        """Compute the Euclidean norm."""
        return (self.x**2 + self.y**2)**0.5

    def scale(self, factor: float) -> "SampleVector2D":
        return SampleVector2D(self.x * factor, self.y * factor)


def test_stubgen_custom_class():
    """Validates generation of stubs for custom classes with methods and docstrings."""
    gen = SynapseStubGenerator(include_docstrings=True)
    stub = gen.generate_stub_for_class(SampleVector2D)

    assert "class SampleVector2D:" in stub
    assert '"""A 2D geometric vector representation."""' in stub
    assert "x: float" in stub
    assert "y: float = 0.0" in stub

    # self parameter must NOT have type annotation
    assert "fn __init__(self, x: float = 0.0, y: float = 0.0) -> void:" in stub
    assert '"""Initialize coordinates."""' in stub

    assert "fn magnitude(self) -> float:" in stub
    assert '"""Compute the Euclidean norm."""' in stub

    assert "fn scale(self, factor: float) -> SampleVector2D: ..." in stub


def test_stubgen_class_docstring_toggle():
    """Validates that include_docstrings=False produces clean compact stubs."""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_for_class(SampleVector2D)

    assert '"""' not in stub
    assert "fn __init__(self, x: float = 0.0, y: float = 0.0) -> void: ..." in stub
    assert "fn magnitude(self) -> float: ..." in stub
    assert "fn scale(self, factor: float) -> SampleVector2D: ..." in stub


class BaseService:
    service_name: str = "base"


class AuthService(BaseService):
    """Authentication and identity service."""
    DEFAULT_TIMEOUT = 30

    @staticmethod
    def hash_token(raw: str) -> str:
        return f"hashed_{raw}"

    @classmethod
    def create_default(cls) -> "AuthService":
        return cls()


def test_stubgen_static_class_methods_and_inheritance():
    """Validates handling of base classes, @staticmethod, and @classmethod."""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_for_class(AuthService)

    assert "class AuthService(BaseService):" in stub
    assert "const DEFAULT_TIMEOUT = 30" in stub
    # staticmethod does not have self or cls
    assert "fn hash_token(raw: str) -> str: ..." in stub
    # classmethod has cls parameter
    assert "fn create_default(cls) -> AuthService: ..." in stub


# =============================================================================
# 3. Fallback for Untyped Functions
# =============================================================================

def untyped_calculator(a, b=10):
    """Calculates something without type annotations."""
    return a + b


def test_stubgen_untyped_fallback():
    """Validates that untyped parameters and return types fallback to Any."""
    gen = SynapseStubGenerator(fallback_type="Any", include_docstrings=False)
    stub = gen.generate_stub_for_function(untyped_calculator)

    assert stub == "fn untyped_calculator(a: Any, b: Any = 10) -> Any: ..."


def test_stubgen_untyped_fallback_disabled():
    """Validates that when fallback_type is empty, untyped parameters omit type annotations."""
    gen = SynapseStubGenerator(fallback_type="", include_docstrings=False)
    stub = gen.generate_stub_for_function(untyped_calculator)

    assert stub == "fn untyped_calculator(a, b = 10): ..."


# =============================================================================
# 4. Module Constants Extraction
# =============================================================================

def test_stubgen_module_constants_from_source():
    """Validates extraction of module-level constants of various types."""
    source_code = '''
"""Configuration module."""
PI = 3.14159265
MAX_RETRIES: int = 5
APP_NAME = "SynapseCore"
IS_DEBUG: bool = True
FEATURE_FLAGS = ["fast_math", "cuda"]
'''
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_from_source(source_code, module_name="config")

    assert stub.startswith("// Synapse Interface Stub for: config")
    assert "const PI = 3.14159265" in stub
    assert "const MAX_RETRIES: int = 5" in stub
    assert 'const APP_NAME = "SynapseCore"' in stub
    assert "const IS_DEBUG: bool = true" in stub
    assert 'const FEATURE_FLAGS = ["fast_math", "cuda"]' in stub


# =============================================================================
# 5. Disk Persistence (.syni) Tests
# =============================================================================

def test_stubgen_save_stub():
    """Validates saving .syni stub file to disk and reading back its contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "math.syni")
        save_stub("math", out_path, include_docstrings=False)

        assert os.path.isfile(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert content.startswith("// Synapse Interface Stub for: math")
        assert "fn sin(" in content
        assert "const pi = " in content


def test_stubgen_save_stub_nested_directory():
    """Validates that save_stub creates parent directories automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "stubs", "py", "json.syni")
        gen = SynapseStubGenerator(include_docstrings=False)
        gen.save_stub("json", out_path)

        assert os.path.isfile(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert content.startswith("// Synapse Interface Stub for: json")
        assert "class JSONDecoder:" in content


# =============================================================================
# 6. Type Mapping Engine Tests
# =============================================================================

def test_stubgen_type_mapping():
    """Validates SynapseTypeFormatter mapping of Python types to Synapse syntax."""
    # Primitive types
    assert SynapseTypeFormatter.format_runtime_type(int) == "int"
    assert SynapseTypeFormatter.format_runtime_type(float) == "float"
    assert SynapseTypeFormatter.format_runtime_type(str) == "str"
    assert SynapseTypeFormatter.format_runtime_type(bool) == "bool"
    assert SynapseTypeFormatter.format_runtime_type(None, is_return=True) == "void"
    assert SynapseTypeFormatter.format_runtime_type(None, is_return=False) == "None"

    # Generic collections
    assert SynapseTypeFormatter.format_runtime_type(List[int]) == "List[int]"
    assert SynapseTypeFormatter.format_runtime_type(list) == "List[Any]"
    assert SynapseTypeFormatter.format_runtime_type(Dict[str, float]) == "Dict[str, float]"
    assert SynapseTypeFormatter.format_runtime_type(dict) == "Dict[str, Any]"
    assert SynapseTypeFormatter.format_runtime_type(Tuple[int, str]) == "Tuple[int, str]"

    # Optional and Union
    assert SynapseTypeFormatter.format_runtime_type(Optional[int]) == "Optional[int]"
    assert SynapseTypeFormatter.format_runtime_type(Union[str, None]) == "Optional[str]"
    assert SynapseTypeFormatter.format_runtime_type(Union[int, str]) == "int | str"


def test_stubgen_tensor_type_mapping():
    """Validates mapping of Tensor types from string or AST."""
    import ast
    node1 = ast.parse("Tensor[32, 64]", mode="eval").body
    assert SynapseTypeFormatter.format_ast_type(node1) == "Tensor[32, 64]"

    node2 = ast.parse("torch.Tensor", mode="eval").body
    assert SynapseTypeFormatter.format_ast_type(node2) == "Tensor"


# =============================================================================
# 7. AST Source Generation & Error Handling
# =============================================================================

def test_stubgen_generate_from_source_string():
    """Validates generation directly from a multi-construct source string."""
    source = """
# Module: neural_layer
ALPHA: float = 0.01

def forward(inputs: List[float], bias: float = 0.0) -> List[float]:
    \"\"\"Perform forward pass.\"\"\"
    return [x + bias for x in inputs]

class DenseLayer:
    units: int
    def __init__(self, units: int = 64) -> None:
        self.units = units

    def get_units(self) -> int:
        return self.units
"""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_from_source(source, module_name="neural_layer")

    assert stub.startswith("// Synapse Interface Stub for: neural_layer")
    assert "const ALPHA: float = 0.01" in stub
    assert "fn forward(inputs: List[float], bias: float = 0.0) -> List[float]: ..." in stub
    assert "class DenseLayer:" in stub
    assert "units: int" in stub
    assert "fn __init__(self, units: int = 64) -> void: ..." in stub
    assert "fn get_units(self) -> int: ..." in stub


def test_stubgen_nonexistent_module_raises_error():
    """Validates that importing a nonexistent module raises ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        generate_stub_for_module("definitely_non_existent_synapse_mod_9999")


# =============================================================================
# 8. Regression and Deep Edge Case Tests
# =============================================================================

def test_stubgen_imported_symbols_not_leaked():
    """Validates that imported third-party/stdlib symbols do not leak into module stubs."""
    stub = generate_stub_for_module("synapse.parser.parser")

    # Local definitions MUST be present
    assert "class Parser:" in stub
    assert "class ParseError" in stub
    assert "fn parse_source(" in stub

    # External imported classes and typing constructs MUST NOT be emitted
    assert "class ASTCache:" not in stub
    assert "class Any:" not in stub
    assert "class Stmt:" not in stub
    assert "class Token:" not in stub
    assert "fn Optional(" not in stub


def test_stubgen_untyped_init_returns_void():
    """Validates that AST function __init__ without explicit return hint returns void."""
    source = """
class Model:
    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
"""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_from_source(source)

    assert "fn __init__(self, hidden_dim: int = 128) -> void: ..." in stub
    assert "-> Any" not in stub


def test_stubgen_empty_class_with_docstring():
    """Validates that empty classes with docstrings end with ellipsis body."""
    source = """
class EmptyConfig:
    \"\"\"An empty configuration placeholder.\"\"\"
    pass
"""
    gen = SynapseStubGenerator(include_docstrings=True)
    stub = gen.generate_stub_from_source(source)

    assert "class EmptyConfig:" in stub
    assert '"""An empty configuration placeholder."""' in stub
    assert stub.strip().endswith("...")


def test_stubgen_default_values_nested_and_pointer_redaction():
    """Validates recursive default value formatting without memory pointer leaks."""
    class CustomObject:
        pass

    obj = CustomObject()
    # Tuples with booleans and arbitrary objects
    tup_val = (True, False, obj)
    formatted_tup = SynapseTypeFormatter.format_default_value(tup_val)
    assert formatted_tup == "(true, false, ...)"
    assert "0x" not in formatted_tup

    # Sets
    set_val = {True, False}
    formatted_set = SynapseTypeFormatter.format_default_value(set_val)
    assert formatted_set == "{false, true}"


def test_stubgen_generic_runtime_types_enhanced():
    """Validates runtime formatting of Callable, Sequence, Mapping, Literal, and ForwardRef."""
    from typing import Callable, ForwardRef, Literal, Mapping, Sequence

    # Callable
    c1 = Callable[[int, float], str]
    assert SynapseTypeFormatter.format_runtime_type(c1) == "Callable[[int, float], str]"
    c2 = Callable[..., int]
    assert SynapseTypeFormatter.format_runtime_type(c2) == "Callable[..., int]"

    # Sequence -> List, Mapping -> Dict
    seq = Sequence[int]
    assert SynapseTypeFormatter.format_runtime_type(seq) == "List[int]"
    mapping = Mapping[str, float]
    assert SynapseTypeFormatter.format_runtime_type(mapping) == "Dict[str, float]"

    # Literal
    lit = Literal["fast", "accurate"]
    assert SynapseTypeFormatter.format_runtime_type(lit) == 'Literal["fast", "accurate"]'

    # ForwardRef
    fref = ForwardRef("DenseLayer")
    assert SynapseTypeFormatter.format_runtime_type(fref) == "DenseLayer"


def test_stubgen_keyword_only_args_separator():
    """Validates that keyword-only arguments without *args insert a bare * separator."""
    source = """
def query(endpoint: str, *, timeout: float = 30.0, retry: bool = True) -> str:
    return endpoint
"""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_from_source(source)

    assert "fn query(endpoint: str, *, timeout: float = 30.0, retry: bool = true) -> str: ..." in stub


def test_stubgen_union_multiple_types_with_none():
    """Validates that Union with multiple types and None formats to Optional[T1 | T2]."""
    from typing import Union
    t = Union[int, str, None]
    assert SynapseTypeFormatter.format_runtime_type(t) == "Optional[int | str]"

    # AST BinOp int | str | None
    source = """
def parse_id(val: int | str | None) -> void:
    pass
"""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_from_source(source)
    assert "fn parse_id(val: Optional[int | str]) -> void: ..." in stub


def test_stubgen_annotated_self_cls_stripped():
    """Validates that explicit type annotations on self or cls parameters are cleanly omitted."""
    source = """
class Layer:
    def forward(self: "Layer", x: float) -> float:
        return x

    @classmethod
    def create(cls: type) -> "Layer":
        return cls()
"""
    gen = SynapseStubGenerator(include_docstrings=False)
    stub = gen.generate_stub_from_source(source)

    assert "fn forward(self, x: float) -> float: ..." in stub
    assert "fn create(cls) -> Layer: ..." in stub


def test_stubgen_cli_entrypoint(monkeypatch, capsys):
    """Validates CLI main() invocation."""
    from synapse.tools.stubgen import main

    monkeypatch.setattr("sys.argv", ["synapse-stubgen", "math", "--no-docstrings"])
    exit_code = main()
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "// Synapse Interface Stub for: math" in captured.out
    assert '"""' not in captured.out
    assert "fn sin(" in captured.out

