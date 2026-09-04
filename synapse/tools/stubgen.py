"""
Synapse Stub Generator (synapse-stubgen)
Phase 1: Automated Type Stub & Interface Generator.

Introspects Python modules, classes, and callables using inspect and AST parsing
to generate clean Synapse interface stub files (.syni).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# Type Formatting Engine
# =============================================================================

class SynapseTypeFormatter:
    """
    Translates Python type annotations, AST nodes, and runtime type objects
    into clean Synapse interface stub types.
    """

    PRIMITIVE_TYPE_MAP: Dict[str, str] = {
        "int": "int",
        "float": "float",
        "str": "str",
        "bool": "bool",
        "bytes": "bytes",
        "None": "void",
        "NoneType": "void",
        "Any": "Any",
        "object": "Any",
    }

    GENERIC_TYPE_MAP: Dict[str, str] = {
        "list": "List",
        "List": "List",
        "Sequence": "List",
        "Iterable": "List",
        "dict": "Dict",
        "Dict": "Dict",
        "Mapping": "Dict",
        "tuple": "Tuple",
        "Tuple": "Tuple",
        "set": "Set",
        "Set": "Set",
        "Optional": "Optional",
        "Union": "Union",
        "Tensor": "Tensor",
        "Option": "Option",
        "Result": "Result",
    }

    @classmethod
    def format_ast_type(
        cls,
        node: Optional[ast.AST],
        is_return: bool = False,
        fallback: str = "Any",
    ) -> str:
        """
        Formats an AST annotation node into Synapse type syntax.
        """
        if node is None:
            return fallback if fallback else ""

        if isinstance(node, ast.Name):
            name = node.id
            if name in ("None", "NoneType"):
                return "void" if is_return else "None"
            if name == "object":
                return "Any"
            if name in ("list", "List"):
                return "List[Any]"
            if name in ("dict", "Dict"):
                return "Dict[str, Any]"
            if name in ("tuple", "Tuple"):
                return "Tuple[Any, ...]"
            if name in ("set", "Set"):
                return "Set[Any]"
            return cls.PRIMITIVE_TYPE_MAP.get(name, name)

        if isinstance(node, ast.Constant):
            if node.value is None:
                return "void" if is_return else "None"
            if isinstance(node.value, str):
                # Possible forward reference string: e.g. "Vector2D" or "List[int]"
                raw_str = node.value.strip()
                try:
                    parsed = ast.parse(raw_str, mode="eval")
                    return cls.format_ast_type(parsed.body, is_return=is_return, fallback=fallback)
                except Exception:
                    return raw_str.strip("'\"")
            return str(node.value)

        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                base_str = node.value.id
            elif isinstance(node.value, ast.Attribute) and cls.format_ast_type(node.value.value) in ("typing", "t"):
                base_str = node.value.attr
            else:
                base_str = cls.format_ast_type(node.value, is_return=False, fallback=fallback)
            mapped_base = cls.GENERIC_TYPE_MAP.get(base_str, base_str)

            # Handle slice arguments
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                args = [
                    cls.format_ast_type(elt, is_return=False, fallback=fallback)
                    for elt in slice_node.elts
                ]
            else:
                args = [cls.format_ast_type(slice_node, is_return=False, fallback=fallback)]

            # Clean Union / Optional mapping
            if mapped_base in ("Union", "Optional"):
                non_none = [a for a in args if a not in ("None", "void")]
                has_none = any(a in ("None", "void") for a in args)
                if has_none or mapped_base == "Optional":
                    inner = " | ".join(non_none) if non_none else "Any"
                    return f"Optional[{inner}]"
                return " | ".join(args)

            if mapped_base == "Tensor":
                return f"Tensor[{', '.join(args)}]"

            if mapped_base == "Callable":
                if len(args) == 2 and args[0] == "...":
                    return f"Callable[..., {args[1]}]"
                elif len(args) >= 2:
                    inners = ", ".join(args[:-1])
                    return f"Callable[[{inners}], {args[-1]}]"
                return f"Callable[{', '.join(args)}]"

            return f"{mapped_base}[{', '.join(args)}]"

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left_str = cls.format_ast_type(node.left, is_return=False, fallback=fallback)
            right_str = cls.format_ast_type(node.right, is_return=False, fallback=fallback)
            if right_str in ("None", "void") and left_str not in ("None", "void"):
                return f"Optional[{left_str}]"
            if left_str in ("None", "void") and right_str not in ("None", "void"):
                return f"Optional[{right_str}]"
            return f"{left_str} | {right_str}"

        if isinstance(node, ast.Attribute):
            val_str = cls.format_ast_type(node.value, is_return=False, fallback=fallback)
            attr = node.attr
            if val_str in ("typing", "t"):
                if attr in cls.GENERIC_TYPE_MAP:
                    return cls.GENERIC_TYPE_MAP[attr]
                if attr in cls.PRIMITIVE_TYPE_MAP:
                    return cls.PRIMITIVE_TYPE_MAP[attr]
                return attr
            if (val_str == "torch" and attr == "Tensor") or (val_str in ("np", "numpy") and attr == "ndarray"):
                return "Tensor"
            return attr

        try:
            return ast.unparse(node)
        except Exception:
            return fallback if fallback else ""

    @classmethod
    def format_runtime_type(
        cls,
        type_hint: Any,
        is_return: bool = False,
        fallback: str = "Any",
    ) -> str:
        """
        Formats a runtime type annotation object (e.g. from inspect.signature or typing)
        into Synapse type syntax.
        """
        if type_hint is inspect.Parameter.empty or type_hint is inspect.Signature.empty:
            return fallback if fallback else ""

        if type_hint is None or type_hint is type(None):
            return "void" if is_return else "None"

        if type_hint is int:
            return "int"
        if type_hint is float:
            return "float"
        if type_hint is str:
            return "str"
        if type_hint is bool:
            return "bool"
        if type_hint is bytes:
            return "bytes"
        if type_hint is Any or type_hint is object:
            return "Any"
        if type_hint is list or type_hint is List:
            return "List[Any]"
        if type_hint is dict or type_hint is Dict:
            return "Dict[str, Any]"
        if type_hint is tuple or type_hint is Tuple:
            return "Tuple[Any, ...]"
        if type_hint is set or type_hint is Set:
            return "Set[Any]"

        # Handle ForwardRef
        if hasattr(type_hint, "__forward_arg__"):
            clean = str(type_hint.__forward_arg__).strip()
            try:
                parsed = ast.parse(clean, mode="eval")
                return cls.format_ast_type(parsed.body, is_return=is_return, fallback=fallback)
            except Exception:
                return clean.strip("'\"")

        origin = getattr(type_hint, "__origin__", None)
        if origin is None and hasattr(types, "UnionType") and isinstance(type_hint, types.UnionType):
            origin = types.UnionType

        if origin is not None:
            args = getattr(type_hint, "__args__", ())
            origin_name = getattr(origin, "__name__", str(origin))

            if origin in (list, List) or origin_name == "list":
                inner = cls.format_runtime_type(args[0], fallback=fallback) if args else "Any"
                return f"List[{inner}]"

            if origin in (dict, Dict) or origin_name == "dict":
                k = cls.format_runtime_type(args[0], fallback=fallback) if len(args) > 0 else "str"
                v = cls.format_runtime_type(args[1], fallback=fallback) if len(args) > 1 else "Any"
                return f"Dict[{k}, {v}]"

            if origin in (tuple, Tuple) or origin_name == "tuple":
                if args:
                    if len(args) == 2 and args[1] is ...:
                        inner = cls.format_runtime_type(args[0], fallback=fallback)
                        return f"Tuple[{inner}, ...]"
                    inner = ", ".join(cls.format_runtime_type(a, fallback=fallback) for a in args)
                    return f"Tuple[{inner}]"
                return "Tuple[Any, ...]"

            if origin in (set, Set) or origin_name == "set":
                inner = cls.format_runtime_type(args[0], fallback=fallback) if args else "Any"
                return f"Set[{inner}]"

            if origin is Union or origin_name == "Union" or (hasattr(types, "UnionType") and origin is types.UnionType):
                non_none = [a for a in args if a is not type(None) and a is not None]
                has_none = len(non_none) < len(args)
                if has_none:
                    inner = " | ".join(cls.format_runtime_type(a, is_return=False, fallback=fallback) for a in non_none)
                    return f"Optional[{inner}]"
                return " | ".join(cls.format_runtime_type(a, is_return=False, fallback=fallback) for a in args)

            if origin_name == "Optional":
                inner = cls.format_runtime_type(args[0], fallback=fallback) if args else "Any"
                return f"Optional[{inner}]"

            # Check for Tensor
            if "Tensor" in origin_name:
                inner = ", ".join(cls.format_runtime_type(a, fallback=fallback) for a in args)
                return f"Tensor[{inner}]" if inner else "Tensor"

            # Check for Callable
            if origin_name == "Callable" or origin is Callable or (isinstance(origin, type) and issubclass(origin, Callable)):
                if not args:
                    return "Callable"
                if len(args) == 2 and args[0] is ...:
                    ret = cls.format_runtime_type(args[1], is_return=True, fallback=fallback)
                    return f"Callable[..., {ret}]"
                arg_types = args[:-1]
                ret_type = args[-1]
                inners = ", ".join(cls.format_runtime_type(a, is_return=False, fallback=fallback) for a in arg_types)
                ret = cls.format_runtime_type(ret_type, is_return=True, fallback=fallback)
                return f"Callable[[{inners}], {ret}]"

            # Check for Literal
            if origin_name == "Literal":
                literals = [cls.format_default_value(a) for a in args]
                return f"Literal[{', '.join(literals)}]"

            # Map generic collections e.g. Sequence, Mapping, Iterable
            mapped_origin = cls.GENERIC_TYPE_MAP.get(origin_name, origin_name)
            if args:
                inners = ", ".join(cls.format_runtime_type(a, is_return=False, fallback=fallback) for a in args)
                return f"{mapped_origin}[{inners}]"
            return mapped_origin

        if isinstance(type_hint, str):
            clean = type_hint.strip()
            try:
                parsed = ast.parse(clean, mode="eval")
                return cls.format_ast_type(parsed.body, is_return=is_return, fallback=fallback)
            except Exception:
                return clean.strip("'\"")

        if hasattr(type_hint, "__name__"):
            name = type_hint.__name__
            if name == "NoneType":
                return "void" if is_return else "None"
            if name in ("Tensor", "ndarray") or getattr(type_hint, "__module__", None) in ("torch", "numpy"):
                return "Tensor"
            return cls.PRIMITIVE_TYPE_MAP.get(name, name)

        str_rep = str(type_hint)
        if str_rep.startswith("typing."):
            str_rep = str_rep[7:]
        return str_rep

    @classmethod
    def format_default_value(cls, val: Any) -> str:
        """
        Formats a Python runtime default value into Synapse syntax literal.
        Ensures nested collections are formatted recursively and memory addresses are redacted.
        """
        if val is None:
            return "None"
        if val is True:
            return "true"
        if val is False:
            return "false"
        if isinstance(val, (int, float)):
            return repr(val)
        if isinstance(val, str):
            return json.dumps(val)
        if isinstance(val, list):
            if not val:
                return "[]"
            return "[" + ", ".join(cls.format_default_value(v) for v in val) + "]"
        if isinstance(val, tuple):
            if not val:
                return "()"
            items = ", ".join(cls.format_default_value(v) for v in val)
            return f"({items},)" if len(val) == 1 else f"({items})"
        if isinstance(val, set):
            if not val:
                return "set()"
            items = ", ".join(cls.format_default_value(v) for v in sorted(val, key=lambda x: str(x)))
            return "{" + items + "}"
        if isinstance(val, dict):
            if not val:
                return "{}"
            entries = [f"{json.dumps(str(k))}: {cls.format_default_value(v)}" for k, v in val.items()]
            return "{" + ", ".join(entries) + "}"
        if inspect.isfunction(val) or inspect.isclass(val):
            return getattr(val, "__name__", "...")
        r = repr(val)
        if "<" in r and ">" in r:
            return "..."
        return r

    @classmethod
    def format_ast_default(cls, node: Optional[ast.AST]) -> Optional[str]:
        """
        Formats an AST default expression into Synapse syntax.
        """
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return cls.format_default_value(node.value)
        if isinstance(node, ast.List):
            return "[" + ", ".join(cls.format_ast_default(e) or "..." for e in node.elts) + "]"
        if isinstance(node, ast.Tuple):
            items = ", ".join(cls.format_ast_default(e) or "..." for e in node.elts)
            return f"({items},)" if len(node.elts) == 1 else f"({items})"
        if isinstance(node, ast.Dict):
            entries = []
            for k, v in zip(node.keys, node.values):
                k_str = cls.format_ast_default(k) if k else "..."
                v_str = cls.format_ast_default(v) if v else "..."
                entries.append(f"{k_str}: {v_str}")
            return "{" + ", ".join(entries) + "}"
        try:
            expr_str = ast.unparse(node)
            if expr_str == "True":
                return "true"
            if expr_str == "False":
                return "false"
            if expr_str == "None":
                return "None"
            return expr_str
        except Exception:
            return "..."


# =============================================================================
# Stub Intermediate Representations
# =============================================================================

@dataclass
class StubParam:
    """Represents a parameter in a function or method signature."""
    name: str
    type_str: str = ""
    default_str: Optional[str] = None
    is_vararg: bool = False
    is_kwarg: bool = False

    def format(self) -> str:
        prefix = "**" if self.is_kwarg else ("*" if self.is_vararg else "")
        display_name = f"{prefix}{self.name}"

        # Bare '*' separator for keyword-only arguments
        if display_name == "*":
            return "*"

        if self.type_str and self.default_str is not None:
            return f"{display_name}: {self.type_str} = {self.default_str}"
        elif self.type_str:
            return f"{display_name}: {self.type_str}"
        elif self.default_str is not None:
            return f"{display_name} = {self.default_str}"
        else:
            return display_name


@dataclass
class StubFunction:
    """Represents a function or method stub."""
    name: str
    params: List[StubParam] = field(default_factory=list)
    return_type: str = ""
    docstring: Optional[str] = None
    is_method: bool = False
    is_static: bool = False
    is_class_method: bool = False

    def format(self, indent: str = "", include_docstrings: bool = True) -> str:
        params_str = ", ".join(p.format() for p in self.params)
        ret_part = f" -> {self.return_type}" if self.return_type else ""
        header = f"{indent}fn {self.name}({params_str}){ret_part}:"

        clean_doc = inspect.cleandoc(self.docstring) if (self.docstring and include_docstrings) else None

        if clean_doc:
            doc_indent = indent + "    "
            safe_doc = clean_doc.replace('"""', r'\"\"\"')
            lines = [header]
            if "\n" in safe_doc:
                lines.append(f'{doc_indent}"""')
                for d_line in safe_doc.splitlines():
                    lines.append(f"{doc_indent}{d_line}" if d_line else "")
                lines.append(f'{doc_indent}"""')
            else:
                lines.append(f'{doc_indent}"""{safe_doc}"""')
            lines.append(f"{doc_indent}...")
            return "\n".join(lines)
        else:
            return f"{header} ..."


@dataclass
class StubClass:
    """Represents a class interface stub."""
    name: str
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    fields: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)
    methods: List[StubFunction] = field(default_factory=list)

    def format(self, indent: str = "", include_docstrings: bool = True) -> str:
        bases_part = f"({', '.join(self.bases)})" if self.bases else ""
        header = f"{indent}class {self.name}{bases_part}:"

        clean_doc = inspect.cleandoc(self.docstring) if (self.docstring and include_docstrings) else None
        body_indent = indent + "    "
        lines = [header]

        if clean_doc:
            safe_doc = clean_doc.replace('"""', r'\"\"\"')
            if "\n" in safe_doc:
                lines.append(f'{body_indent}"""')
                for d_line in safe_doc.splitlines():
                    lines.append(f"{body_indent}{d_line}" if d_line else "")
                lines.append(f'{body_indent}"""')
            else:
                lines.append(f'{body_indent}"""{safe_doc}"""')

        # Fields / constants
        if clean_doc and self.fields:
            lines.append("")
        for f_name, f_type, f_default in self.fields:
            if f_type and f_default is not None:
                lines.append(f"{body_indent}{f_name}: {f_type} = {f_default}")
            elif f_type:
                lines.append(f"{body_indent}{f_name}: {f_type}")
            elif f_default is not None:
                lines.append(f"{body_indent}const {f_name} = {f_default}")

        # Methods
        for m in self.methods:
            if lines[-1] != header and not lines[-1].endswith('"""'):
                lines.append("")
            lines.append(m.format(indent=body_indent, include_docstrings=include_docstrings))

        if len(lines) == 1:
            return f"{header} ..."

        if clean_doc and not self.fields and not self.methods:
            lines.append(f"{body_indent}...")

        return "\n".join(lines)


@dataclass
class StubConstant:
    """Represents a module-level constant."""
    name: str
    value_str: str
    type_str: Optional[str] = None

    def format(self) -> str:
        if self.type_str:
            return f"const {self.name}: {self.type_str} = {self.value_str}"
        return f"const {self.name} = {self.value_str}"


# =============================================================================
# SynapseStubGenerator
# =============================================================================

class SynapseStubGenerator:
    """
    Automated Type Stub & Interface Generator for Synapse (.syni).
    Introspects Python modules, classes, and functions via AST parsing and runtime inspection.
    """

    def __init__(
        self,
        include_docstrings: bool = True,
        fallback_type: str = "Any",
        include_private: bool = False,
        indent: str = "    ",
    ) -> None:
        self.include_docstrings = include_docstrings
        self.fallback_type = fallback_type
        self.include_private = include_private
        self.indent = indent

    # -------------------------------------------------------------------------
    # Public Generation API
    # -------------------------------------------------------------------------

    def generate_stub_for_module(self, module_or_name: Union[str, Any]) -> str:
        """
        Generates a complete .syni interface stub string for a Python module or module name.
        """
        if isinstance(module_or_name, str):
            if module_or_name.endswith(".py") and os.path.isfile(module_or_name):
                # File path passed
                mod_name = os.path.splitext(os.path.basename(module_or_name))[0]
                with open(module_or_name, "r", encoding="utf-8") as f:
                    source = f.read()
                return self.generate_stub_from_source(source, module_name=mod_name)
            else:
                # Module import name e.g. "math", "json", "os.path"
                module = importlib.import_module(module_or_name)
                mod_name = module_or_name
        else:
            module = module_or_name
            mod_name = getattr(module, "__name__", "module")

        # Try AST source introspection if source is available
        source = None
        try:
            source = inspect.getsource(module)
        except (TypeError, OSError):
            source = None

        if source is not None:
            # We have source code! Combine AST precision with runtime members.
            return self._generate_from_module_ast_and_runtime(module, mod_name, source)
        else:
            # Pure C-extension or built-in module without Python source (e.g. math)
            return self._generate_from_runtime_module(module, mod_name)

    def generate_stub(self, module_or_name: Union[str, Any]) -> str:
        """Alias for generate_stub_for_module."""
        return self.generate_stub_for_module(module_or_name)

    def generate_stub_from_source(self, source: str, module_name: str = "module") -> str:
        """
        Generates a .syni interface stub string directly from Python source code.
        """
        tree = ast.parse(source)
        mod_doc = ast.get_docstring(tree) if self.include_docstrings else None

        constants: List[StubConstant] = []
        functions: List[StubFunction] = []
        classes: List[StubClass] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") or self.include_private:
                    functions.append(self._ast_function_to_stub(node))
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") or self.include_private:
                    classes.append(self._ast_class_to_stub(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name == "__all__":
                            continue
                        if not name.startswith("_") or self.include_private:
                            val_str = SynapseTypeFormatter.format_ast_default(node.value)
                            if val_str is not None:
                                constants.append(StubConstant(name=name, value_str=val_str))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if not name.startswith("_") or self.include_private:
                        val_str = SynapseTypeFormatter.format_ast_default(node.value) if node.value else "..."
                        type_str = SynapseTypeFormatter.format_ast_type(node.annotation, fallback=self.fallback_type)
                        constants.append(StubConstant(name=name, value_str=val_str or "...", type_str=type_str))

        return self._render_module_stub(module_name, mod_doc, constants, functions, classes)

    def generate_stub_for_class(self, cls: type) -> str:
        """
        Generates a stub for an individual Python class.
        """
        stub_cls = self._inspect_class(cls)
        return stub_cls.format(include_docstrings=self.include_docstrings)

    def generate_stub_for_function(self, func: Any) -> str:
        """
        Generates a stub for an individual Python function or callable.
        """
        stub_func = self._inspect_function(func)
        return stub_func.format(include_docstrings=self.include_docstrings)

    def save_stub(self, module_or_name: Union[str, Any], output_path: str) -> None:
        """
        Generates and saves a .syni stub file to disk.
        """
        content = self.generate_stub_for_module(module_or_name)
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    # -------------------------------------------------------------------------
    # Internal Module Introspection
    # -------------------------------------------------------------------------

    def _render_module_stub(
        self,
        module_name: str,
        docstring: Optional[str],
        constants: List[StubConstant],
        functions: List[StubFunction],
        classes: List[StubClass],
    ) -> str:
        """Assembles all stub components into a final .syni document."""
        lines: List[str] = [f"// Synapse Interface Stub for: {module_name}"]

        if docstring and self.include_docstrings:
            clean_doc = inspect.cleandoc(docstring).replace('"""', r'\"\"\"')
            lines.append('"""')
            lines.append(clean_doc)
            lines.append('"""')

        if constants:
            lines.append("")
            for c in constants:
                lines.append(c.format())

        if functions:
            lines.append("")
            for f in functions:
                lines.append(f.format(include_docstrings=self.include_docstrings))
                lines.append("")
            if lines[-1] == "":
                lines.pop()

        if classes:
            lines.append("")
            for c in classes:
                lines.append(c.format(include_docstrings=self.include_docstrings))
                lines.append("")
            if lines[-1] == "":
                lines.pop()

        return "\n".join(lines).rstrip() + "\n"

    def _generate_from_module_ast_and_runtime(
        self,
        module: types.ModuleType,
        module_name: str,
        source: str,
    ) -> str:
        """
        Combines AST analysis of source with runtime module inspection.
        Preserves definition order and prevents imported symbols from leaking into stubs.
        """
        has_all = hasattr(module, "__all__") and isinstance(module.__all__, (list, tuple))
        all_set = set(module.__all__) if has_all else None

        mod_doc = inspect.getdoc(module) if self.include_docstrings else None

        # Parse AST to gather source definitions
        try:
            tree = ast.parse(source)
        except Exception:
            # If AST parsing fails, fallback to runtime inspection
            return self._generate_from_runtime_module(module, module_name)

        constants: List[StubConstant] = []
        functions: List[StubFunction] = []
        classes: List[StubClass] = []
        processed_names: Set[str] = set()

        # 1. Walk AST body in natural order
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if (not name.startswith("_") or self.include_private) and (all_set is None or name in all_set):
                    functions.append(self._ast_function_to_stub(node))
                    processed_names.add(name)
            elif isinstance(node, ast.ClassDef):
                name = node.name
                if (not name.startswith("_") or self.include_private) and (all_set is None or name in all_set):
                    classes.append(self._ast_class_to_stub(node))
                    processed_names.add(name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name == "__all__":
                            continue
                        if (not name.startswith("_") or self.include_private) and (all_set is None or name in all_set):
                            val_str = SynapseTypeFormatter.format_ast_default(node.value)
                            if val_str is not None:
                                constants.append(StubConstant(name=name, value_str=val_str))
                                processed_names.add(name)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if (not name.startswith("_") or self.include_private) and (all_set is None or name in all_set):
                        val_str = SynapseTypeFormatter.format_ast_default(node.value) if node.value else "..."
                        type_str = SynapseTypeFormatter.format_ast_type(node.annotation, fallback=self.fallback_type)
                        constants.append(StubConstant(name=name, value_str=val_str or "...", type_str=type_str))
                        processed_names.add(name)

        # 2. Check for additional members (e.g. __all__ re-exports or dynamic attributes)
        if all_set is not None:
            remaining_names = [n for n in all_set if n not in processed_names]
        else:
            # Only local symbols defined in this module
            remaining_names = [
                n for n in dir(module)
                if n not in processed_names
                and (not n.startswith("_") or self.include_private)
                and getattr(getattr(module, n, None), "__module__", None) == module.__name__
            ]

        for name in remaining_names:
            if not hasattr(module, name):
                continue
            val = getattr(module, name)
            if isinstance(val, types.ModuleType):
                continue

            if inspect.isclass(val):
                classes.append(self._inspect_class(val))
            elif inspect.isroutine(val) or callable(val):
                functions.append(self._inspect_function(val, name=name))
            else:
                val_str = SynapseTypeFormatter.format_default_value(val)
                type_str = None
                annotations = getattr(module, "__annotations__", {})
                if name in annotations:
                    type_str = SynapseTypeFormatter.format_runtime_type(annotations[name], fallback=self.fallback_type)
                constants.append(StubConstant(name=name, value_str=val_str, type_str=type_str))

        return self._render_module_stub(module_name, mod_doc, constants, functions, classes)

    def _generate_from_runtime_module(self, module: types.ModuleType, module_name: str) -> str:
        """
        Introspects purely runtime objects (useful for C-extensions like math or dynamic modules).
        """
        has_all = hasattr(module, "__all__") and isinstance(module.__all__, (list, tuple))
        all_set = set(module.__all__) if has_all else None

        mod_doc = inspect.getdoc(module) if self.include_docstrings else None

        constants: List[StubConstant] = []
        functions: List[StubFunction] = []
        classes: List[StubClass] = []

        names = [n for n in dir(module) if not n.startswith("_") or self.include_private]
        if all_set is not None:
            names = [n for n in names if n in all_set]
        else:
            # Filter out imported symbols from other modules
            filtered_names = []
            for n in names:
                try:
                    v = getattr(module, n)
                except Exception:
                    continue
                v_mod = getattr(v, "__module__", None)
                # Keep if local to module or primitive value without module attribute (e.g. math.pi)
                if v_mod is None or v_mod == module.__name__:
                    filtered_names.append(n)
            names = filtered_names

        for name in names:
            try:
                val = getattr(module, name)
            except Exception:
                continue

            if isinstance(val, types.ModuleType):
                continue

            if inspect.isclass(val):
                classes.append(self._inspect_class(val))
            elif inspect.isroutine(val) or callable(val):
                functions.append(self._inspect_function(val, name=name))
            else:
                if isinstance(val, (int, float, str, bool, bytes, list, dict, tuple, set)) or val is None:
                    val_str = SynapseTypeFormatter.format_default_value(val)
                    type_str = None
                    annotations = getattr(module, "__annotations__", {})
                    if name in annotations:
                        type_str = SynapseTypeFormatter.format_runtime_type(annotations[name], fallback=self.fallback_type)
                    constants.append(StubConstant(name=name, value_str=val_str, type_str=type_str))

        return self._render_module_stub(module_name, mod_doc, constants, functions, classes)

    # -------------------------------------------------------------------------
    # AST Introspection Helpers
    # -------------------------------------------------------------------------

    def _ast_function_to_stub(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_method: bool = False,
    ) -> StubFunction:
        """Converts an AST function node to StubFunction."""
        doc = ast.get_docstring(node) if self.include_docstrings else None

        # Check decorators for staticmethod / classmethod
        is_static = False
        is_class_method = False
        for dec in node.decorator_list:
            dec_id = ""
            if isinstance(dec, ast.Name):
                dec_id = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_id = dec.attr
            if dec_id == "staticmethod":
                is_static = True
            elif dec_id == "classmethod":
                is_class_method = True

        return_type = SynapseTypeFormatter.format_ast_type(node.returns, is_return=True, fallback=self.fallback_type)
        if node.name == "__init__" and (node.returns is None or return_type in ("None", "Any", "")):
            return_type = "void"

        params: List[StubParam] = []

        all_pos = node.args.posonlyargs + node.args.args
        defaults_offset = len(all_pos) - len(node.args.defaults)

        for i, arg in enumerate(all_pos):
            p_name = arg.arg
            is_self = is_method and i == 0 and not is_static and p_name in ("self", "cls")

            p_type = ""
            if not is_self:
                if arg.annotation:
                    p_type = SynapseTypeFormatter.format_ast_type(arg.annotation, fallback=self.fallback_type)
                else:
                    p_type = self.fallback_type if self.fallback_type else ""

            default_val = None
            if i >= defaults_offset:
                def_node = node.args.defaults[i - defaults_offset]
                default_val = SynapseTypeFormatter.format_ast_default(def_node)

            params.append(StubParam(name=p_name, type_str=p_type, default_str=default_val))

        # Vararg *args
        if node.args.vararg:
            va = node.args.vararg
            va_type = SynapseTypeFormatter.format_ast_type(va.annotation, fallback=self.fallback_type) if va.annotation else self.fallback_type
            params.append(StubParam(name=va.arg, type_str=va_type, default_str=None, is_vararg=True))
        elif node.args.kwonlyargs:
            # Bare '*' separator before keyword-only arguments when no *args is present
            params.append(StubParam(name="", type_str="", default_str=None, is_vararg=True))

        # Keyword-only args
        for i, kwarg in enumerate(node.args.kwonlyargs):
            kw_name = kwarg.arg
            kw_type = SynapseTypeFormatter.format_ast_type(kwarg.annotation, fallback=self.fallback_type) if kwarg.annotation else self.fallback_type
            kw_def_node = node.args.kw_defaults[i] if i < len(node.args.kw_defaults) else None
            kw_default = SynapseTypeFormatter.format_ast_default(kw_def_node) if kw_def_node else None
            params.append(StubParam(name=kw_name, type_str=kw_type, default_str=kw_default))

        # Kwarg **kwargs
        if node.args.kwarg:
            ka = node.args.kwarg
            ka_type = SynapseTypeFormatter.format_ast_type(ka.annotation, fallback=self.fallback_type) if ka.annotation else self.fallback_type
            params.append(StubParam(name=ka.arg, type_str=ka_type, default_str=None, is_kwarg=True))

        return StubFunction(
            name=node.name,
            params=params,
            return_type=return_type,
            docstring=doc,
            is_method=is_method,
            is_static=is_static,
            is_class_method=is_class_method,
        )

    def _ast_class_to_stub(self, node: ast.ClassDef) -> StubClass:
        """Converts an AST class node to StubClass."""
        doc = ast.get_docstring(node) if self.include_docstrings else None
        bases: List[str] = []
        for b in node.bases:
            b_name = SynapseTypeFormatter.format_ast_type(b, fallback="")
            if b_name and b_name != "object":
                bases.append(b_name)

        fields: List[Tuple[str, str, Optional[str]]] = []
        methods: List[StubFunction] = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not item.name.startswith("_") or item.name == "__init__" or self.include_private:
                    methods.append(self._ast_function_to_stub(item, is_method=True))
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    name = item.target.id
                    if not name.startswith("_") or self.include_private:
                        val_str = SynapseTypeFormatter.format_ast_default(item.value) if item.value else None
                        t_str = SynapseTypeFormatter.format_ast_type(item.annotation, fallback=self.fallback_type)
                        fields.append((name, t_str, val_str))
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        name = t.id
                        if not name.startswith("_") or self.include_private:
                            val_str = SynapseTypeFormatter.format_ast_default(item.value)
                            fields.append((name, "", val_str))

        return StubClass(
            name=node.name,
            bases=bases,
            docstring=doc,
            fields=fields,
            methods=methods,
        )

    # -------------------------------------------------------------------------
    # Runtime Introspection Helpers
    # -------------------------------------------------------------------------

    def _inspect_function(
        self,
        func: Any,
        name: Optional[str] = None,
        is_method: bool = False,
        is_static: bool = False,
        is_class_method: bool = False,
    ) -> StubFunction:
        """Introspects a runtime function or callable object."""
        fn_name = name or getattr(func, "__name__", "callable")
        doc = inspect.getdoc(func) if self.include_docstrings else None

        params: List[StubParam] = []
        return_type = self.fallback_type if self.fallback_type else ""

        try:
            sig = inspect.signature(func)
            ret_ann = sig.return_annotation
            if ret_ann is not inspect.Signature.empty:
                return_type = SynapseTypeFormatter.format_runtime_type(ret_ann, is_return=True, fallback=self.fallback_type)
            elif fn_name == "__init__":
                return_type = "void"
            else:
                return_type = self.fallback_type if self.fallback_type else ""

            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            seen_first_kw_only = False

            for i, (p_name, param) in enumerate(sig.parameters.items()):
                is_self = is_method and i == 0 and not is_static and p_name in ("self", "cls")

                if param.kind == inspect.Parameter.KEYWORD_ONLY and not has_var_pos and not seen_first_kw_only:
                    params.append(StubParam(name="", type_str="", default_str=None, is_vararg=True))
                    seen_first_kw_only = True

                p_type = ""
                if not is_self:
                    if param.annotation is not inspect.Parameter.empty:
                        p_type = SynapseTypeFormatter.format_runtime_type(param.annotation, is_return=False, fallback=self.fallback_type)
                    else:
                        p_type = self.fallback_type if self.fallback_type else ""

                default_str = None
                if param.default is not inspect.Parameter.empty:
                    default_str = SynapseTypeFormatter.format_default_value(param.default)

                is_var = param.kind == inspect.Parameter.VAR_POSITIONAL
                is_kw = param.kind == inspect.Parameter.VAR_KEYWORD

                params.append(StubParam(
                    name=p_name,
                    type_str=p_type,
                    default_str=default_str,
                    is_vararg=is_var,
                    is_kwarg=is_kw,
                ))

        except (ValueError, TypeError):
            # Fallback for C builtins without text signatures (e.g. math.hypot, math.log)
            params, return_type = self._parse_builtin_docstring_signature(fn_name, doc)

        return StubFunction(
            name=fn_name,
            params=params,
            return_type=return_type,
            docstring=doc,
            is_method=is_method,
            is_static=is_static,
            is_class_method=is_class_method,
        )

    def _parse_builtin_docstring_signature(self, fn_name: str, doc: Optional[str]) -> Tuple[List[StubParam], str]:
        """
        Extracts signature information from a builtin function's docstring
        when inspect.signature fails.
        """
        fallback_ret = "void" if fn_name == "__init__" else (self.fallback_type if self.fallback_type else "")

        if not doc:
            return [
                StubParam("args", self.fallback_type, None, is_vararg=True),
                StubParam("kwargs", self.fallback_type, None, is_kwarg=True),
            ], fallback_ret

        first_line = doc.splitlines()[0].strip()
        match = re.match(rf"^(?:[\w\.]+\.)?{re.escape(fn_name)}\s*\((.*?)\)(?:\s*->\s*(.*))?", first_line)
        if not match:
            return [
                StubParam("args", self.fallback_type, None, is_vararg=True),
                StubParam("kwargs", self.fallback_type, None, is_kwarg=True),
            ], fallback_ret

        raw_params = match.group(1).strip()
        raw_ret = match.group(2).strip() if match.group(2) else ""

        ret_type = fallback_ret
        if raw_ret:
            ret_type = SynapseTypeFormatter.PRIMITIVE_TYPE_MAP.get(raw_ret, self.fallback_type or raw_ret)

        params: List[StubParam] = []
        if raw_params:
            # Clean up brackets from docstring signatures e.g. [base=math.e]
            cleaned_params = raw_params.replace("[", "").replace("]", "").replace("/", "").strip()
            for part in cleaned_params.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" in part:
                    p_name, p_def = part.split("=", 1)
                    p_name = p_name.strip()
                    p_def = p_def.strip()
                else:
                    p_name = part
                    p_def = None

                is_var = p_name.startswith("*") and not p_name.startswith("**")
                is_kw = p_name.startswith("**")
                clean_name = p_name.lstrip("*")

                if clean_name == "" and is_var:
                    # Bare '*' keyword separator
                    params.append(StubParam(name="", type_str="", default_str=None, is_vararg=True))
                else:
                    params.append(StubParam(
                        name=clean_name,
                        type_str=self.fallback_type,
                        default_str=p_def,
                        is_vararg=is_var,
                        is_kwarg=is_kw,
                    ))

        return params, ret_type

    def _inspect_class(self, cls: type) -> StubClass:
        """Introspects a runtime class object."""
        doc = inspect.getdoc(cls) if self.include_docstrings else None
        bases = [b.__name__ for b in cls.__bases__ if b is not object]

        fields: List[Tuple[str, str, Optional[str]]] = []
        methods: List[StubFunction] = []

        cls_dict = cls.__dict__
        annotations = getattr(cls, "__annotations__", {})

        # Extract annotated fields
        for f_name, f_type_hint in annotations.items():
            if not f_name.startswith("_") or self.include_private:
                type_str = SynapseTypeFormatter.format_runtime_type(f_type_hint, fallback=self.fallback_type)
                default_val = None
                if f_name in cls_dict:
                    val = cls_dict[f_name]
                    if not inspect.isroutine(val) and not isinstance(val, (staticmethod, classmethod, property)):
                        default_val = SynapseTypeFormatter.format_default_value(val)
                fields.append((f_name, type_str, default_val))

        # Extract class-level constants without annotations
        for name, val in cls_dict.items():
            if name in annotations or name.startswith("__"):
                continue
            if not name.startswith("_") or self.include_private:
                if isinstance(val, (int, float, str, bool)) or val is None:
                    val_str = SynapseTypeFormatter.format_default_value(val)
                    fields.append((name, "", val_str))

        # Extract methods
        for name, val in cls_dict.items():
            if name.startswith("_") and name != "__init__" and not self.include_private:
                continue

            underlying_func = None
            is_static = False
            is_cls_method = False

            if isinstance(val, staticmethod):
                underlying_func = val.__func__
                is_static = True
            elif isinstance(val, classmethod):
                underlying_func = val.__func__
                is_cls_method = True
            elif isinstance(val, property):
                underlying_func = val.fget
            elif inspect.isfunction(val) or inspect.isroutine(val) or callable(val):
                underlying_func = val

            if underlying_func is not None:
                m_stub = self._inspect_function(
                    underlying_func,
                    name=name,
                    is_method=True,
                    is_static=is_static,
                    is_class_method=is_cls_method,
                )
                methods.append(m_stub)

        return StubClass(
            name=cls.__name__,
            bases=bases,
            docstring=doc,
            fields=fields,
            methods=methods,
        )


# =============================================================================
# Functional Helper Interface
# =============================================================================

def generate_stub_for_module(
    module_or_name: Union[str, Any],
    include_docstrings: bool = True,
    fallback_type: str = "Any",
) -> str:
    """
    Generates a .syni interface stub string for a Python module.
    """
    generator = SynapseStubGenerator(
        include_docstrings=include_docstrings,
        fallback_type=fallback_type,
    )
    return generator.generate_stub_for_module(module_or_name)


def save_stub(
    module_or_name: Union[str, Any],
    output_path: str,
    include_docstrings: bool = True,
    fallback_type: str = "Any",
) -> None:
    """
    Generates and saves a .syni interface stub file to disk.
    """
    generator = SynapseStubGenerator(
        include_docstrings=include_docstrings,
        fallback_type=fallback_type,
    )
    generator.save_stub(module_or_name, output_path)


def main() -> int:
    """CLI entrypoint for synapse-stubgen."""
    import argparse
    parser = argparse.ArgumentParser(description="Synapse Type Stub & Interface Generator (.syni)")
    parser.add_argument("module", help="Python module name or file path to introspect")
    parser.add_argument("-o", "--output", help="Output .syni file path (default: stdout)", default=None)
    parser.add_argument("--no-docstrings", action="store_true", help="Omit docstrings from stub")
    parser.add_argument("--fallback", default="Any", help="Fallback type for untyped parameters (default: Any)")
    parser.add_argument("--include-private", action="store_true", help="Include private symbols starting with _")
    args = parser.parse_args()

    try:
        generator = SynapseStubGenerator(
            include_docstrings=not args.no_docstrings,
            fallback_type=args.fallback,
            include_private=args.include_private,
        )
        stub = generator.generate_stub_for_module(args.module)
        if args.output:
            dir_name = os.path.dirname(args.output)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(stub)
            print(f"Generated Synapse stub saved to: {args.output}")
        else:
            sys.stdout.write(stub)
        return 0
    except Exception as e:
        sys.stderr.write(f"Error generating stub: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
