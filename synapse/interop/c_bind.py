"""
Synapse C-Bind: Direct C/C++ Header Parser and Dynamic Library Binder.

Parses standard C99/C++ header (.h) declarations and automatically binds native
functions from dynamic libraries (.dll on Windows, .so on Linux, .dylib on macOS).
Supports pointer qualifiers, stdint types, Synapse FFI zero-copy tensor buffers,
Python buffer protocol (bytearray, memoryview), function pointer callbacks (CFUNCTYPE),
argument and return type conversions, typedef resolution, and result caching.
"""
from __future__ import annotations

import ctypes
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from synapse.interop.c_ffi import (
    CDynamicLibrary,
    CFunction,
    SynapseFFIError,
    load_library,
    resolve_type,
)

# =========================================================================
# Custom Exceptions
# =========================================================================

class CHeaderParseError(SynapseFFIError):
    """Raised when a C header prototype cannot be parsed or contains invalid syntax."""
    pass


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class CParameter:
    """Represents a single parameter in a C function prototype."""
    type_str: str
    name: str = ""
    ctype: Any = None
    ffi_type: Any = None
    is_function_pointer: bool = False

    def __post_init__(self) -> None:
        if self.ffi_type is None and self.ctype is not None:
            self.ffi_type = self.ctype

    @property
    def type_name(self) -> str:
        """Alias for type_str."""
        return self.type_str

    def __repr__(self) -> str:
        if self.name:
            return f"<CParameter {self.type_str} {self.name}>"
        return f"<CParameter {self.type_str}>"


@dataclass
class CFunctionPrototype:
    """Represents a parsed C function prototype declaration."""
    name: str
    return_type: str
    parameters: List[CParameter] = field(default_factory=list)
    restype: Any = None
    argtypes: List[Any] = field(default_factory=list)
    raw_signature: str = ""
    ffi_restype: Any = None
    ffi_argtypes: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.ffi_restype is None:
            self.ffi_restype = self.restype
        if not self.ffi_argtypes and self.argtypes:
            self.ffi_argtypes = list(self.argtypes)

    @property
    def param_types(self) -> List[str]:
        """Returns the list of parameter type strings."""
        return [p.type_str for p in self.parameters]

    @property
    def param_names(self) -> List[str]:
        """Returns the list of parameter names."""
        return [p.name for p in self.parameters]

    def __repr__(self) -> str:
        params_repr = ", ".join(
            f"{p.type_str} {p.name}".strip() for p in self.parameters
        )
        return f"<CFunctionPrototype {self.return_type} {self.name}({params_repr})>"


# =========================================================================
# Type Keywords and Mapping Tables
# =========================================================================

TYPE_KEYWORDS: set[str] = {
    "void",
    "bool",
    "_bool",
    "char",
    "wchar_t",
    "wchar_p",
    "short",
    "int",
    "long",
    "float",
    "double",
    "signed",
    "unsigned",
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "intptr_t",
    "uintptr_t",
    "intmax_t",
    "uintmax_t",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
    "char_p",
    "void_p",
    "byte",
    "word",
    "dword",
    "qword",
    "handle",
    "hwnd",
    "hmodule",
    "hinstance",
    "bool_t",
    "float_t",
    "int_t",
}

QUALIFIERS: set[str] = {
    "const",
    "volatile",
    "restrict",
    "__restrict",
    "__restrict__",
}

BASE_TYPE_MAP: dict[str, Any] = {
    "void": None,
    "bool": ctypes.c_bool,
    "_bool": ctypes.c_bool,
    "char": ctypes.c_char,
    "wchar_t": ctypes.c_wchar,
    "signed char": ctypes.c_int8,
    "unsigned char": ctypes.c_uint8,
    "short": ctypes.c_int16,
    "short int": ctypes.c_int16,
    "signed short": ctypes.c_int16,
    "signed short int": ctypes.c_int16,
    "unsigned short": ctypes.c_uint16,
    "unsigned short int": ctypes.c_uint16,
    "int": ctypes.c_int32,
    "signed int": ctypes.c_int32,
    "signed": ctypes.c_int32,
    "unsigned int": ctypes.c_uint32,
    "unsigned": ctypes.c_uint32,
    "long": ctypes.c_long,
    "signed long": ctypes.c_long,
    "long int": ctypes.c_long,
    "signed long int": ctypes.c_long,
    "unsigned long": ctypes.c_ulong,
    "unsigned long int": ctypes.c_ulong,
    "long long": ctypes.c_int64,
    "signed long long": ctypes.c_int64,
    "long long int": ctypes.c_int64,
    "signed long long int": ctypes.c_int64,
    "unsigned long long": ctypes.c_uint64,
    "unsigned long long int": ctypes.c_uint64,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "long double": ctypes.c_longdouble,
    "size_t": ctypes.c_size_t,
    "ssize_t": ctypes.c_ssize_t,
    "ptrdiff_t": ctypes.c_ssize_t,
    "intptr_t": ctypes.c_ssize_t,
    "uintptr_t": ctypes.c_size_t,
    "intmax_t": ctypes.c_int64,
    "uintmax_t": ctypes.c_uint64,
    # stdint.h exact width types
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    # Synapse / Windows aliases
    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "int16": ctypes.c_int16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
    "char_p": ctypes.c_char_p,
    "wchar_p": ctypes.c_wchar_p,
    "void_p": ctypes.c_void_p,
    "byte": ctypes.c_uint8,
    "word": ctypes.c_uint16,
    "dword": ctypes.c_uint32,
    "qword": ctypes.c_uint64,
    "handle": ctypes.c_void_p,
    "hwnd": ctypes.c_void_p,
    "hmodule": ctypes.c_void_p,
    "hinstance": ctypes.c_void_p,
    "bool_t": ctypes.c_bool,
    "float_t": ctypes.c_float,
    "int_t": ctypes.c_int,
}


# =========================================================================
# CHeaderParser: C99 / Windows x64 Header Declaration Parser
# =========================================================================

class CHeaderParser:
    """
    Parses standard C99 header declarations (.h string or file), extracts
    function prototypes, resolves parameter and return types to ctypes and Synapse
    FFI types, and strips C comments, preprocessor directives, and function bodies.
    """

    @staticmethod
    def strip_comments(code: str) -> str:
        """
        Strips C-style single-line (//) and multi-line (/* ... */) comments,
        preserving string literals.
        """
        pattern = r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(/\*[\s\S]*?\*/|//[^\r\n]*)'

        def replace(match: re.Match[str]) -> str:
            if match.group(1) is not None:
                return match.group(1)
            # Replace comment with space to prevent token merging
            return " "

        return re.sub(pattern, replace, code)

    @staticmethod
    def strip_preprocessor(code: str) -> str:
        """Strips preprocessor directives starting with #."""
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def canonicalize_type(type_str: str) -> str:
        """
        Normalizes a C type string: collapses whitespaces, attaches pointer asterisks
        consistently, e.g. 'const double *' -> 'const double*'.
        """
        s = type_str.strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*\*\s*", "*", s)
        # Ensure space between * and following word, e.g. '* const'
        s = re.sub(r"\*([a-zA-Z_])", r"* \1", s)
        return s.strip()

    @classmethod
    def map_type_to_ctypes(
        cls, c_type: str, custom_types: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Maps a C type string (e.g. 'int', 'const double*', 'char**', 'void*', 'size_t', 'wchar_t*')
        to its corresponding ctypes type representation.
        """
        s = c_type.strip()
        if not s:
            raise CHeaderParseError("Empty C type cannot be mapped to ctypes")

        # Strip qualifiers for ctypes representation
        clean = re.sub(
            r"\b(const|volatile|restrict|__restrict|__restrict__)\b", "", s
        ).strip()
        clean = re.sub(r"\s+", " ", clean)

        # Count pointer asterisks
        ptr_count = clean.count("*")
        base = clean.replace("*", "").strip()
        base_norm = re.sub(r"\s+", " ", base).lower()

        # Check custom/typedef types first
        if custom_types and base in custom_types:
            target = custom_types[base]
            if ptr_count == 0:
                return target
            for _ in range(ptr_count):
                target = ctypes.POINTER(target)
            return target
        if custom_types and base_norm in custom_types:
            target = custom_types[base_norm]
            if ptr_count == 0:
                return target
            for _ in range(ptr_count):
                target = ctypes.POINTER(target)
            return target

        # Handle 'struct Name' prefix
        if base_norm.startswith("struct "):
            base_norm = base_norm[7:].strip()

        # 1. Non-pointer types
        if ptr_count == 0:
            if base_norm in BASE_TYPE_MAP:
                return BASE_TYPE_MAP[base_norm]
            raise CHeaderParseError(f"Unknown C type: '{c_type}'")

        # 2. Pointer types
        # Special case: char* -> ctypes.c_char_p
        if base_norm == "char":
            if ptr_count == 1:
                return ctypes.c_char_p
            curr = ctypes.c_char_p
            for _ in range(ptr_count - 1):
                curr = ctypes.POINTER(curr)
            return curr

        # Special case: wchar_t* -> ctypes.c_wchar_p
        if base_norm == "wchar_t":
            if ptr_count == 1:
                return ctypes.c_wchar_p
            curr = ctypes.c_wchar_p
            for _ in range(ptr_count - 1):
                curr = ctypes.POINTER(curr)
            return curr

        # Special case: void* -> ctypes.c_void_p
        if base_norm == "void":
            if ptr_count == 1:
                return ctypes.c_void_p
            curr = ctypes.c_void_p
            for _ in range(ptr_count - 1):
                curr = ctypes.POINTER(curr)
            return curr

        # Known base type pointer
        if base_norm in BASE_TYPE_MAP:
            target = BASE_TYPE_MAP[base_norm]
            if target is None:
                # Pointer to void
                target = ctypes.c_void_p
                ptr_count -= 1
            for _ in range(ptr_count):
                target = ctypes.POINTER(target)
            return target

        # Opaque pointer to unrecognized struct or typedef (e.g. FILE*, Node*)
        target = ctypes.c_void_p
        for _ in range(ptr_count - 1):
            target = ctypes.POINTER(target)
        return target

    @classmethod
    def map_type_to_ffi(
        cls, c_type: str, custom_types: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Maps a C type string to its corresponding Synapse C-FFI type
        (synapse.interop.c_ffi).
        """
        try:
            resolved = resolve_type(c_type)
            if resolved is not None:
                return resolved
        except Exception:
            pass
        return cls.map_type_to_ctypes(c_type, custom_types=custom_types)

    @staticmethod
    def _split_param_list(params_str: str) -> List[str]:
        """Splits a C parameter list by comma, ignoring commas inside parentheses."""
        if not params_str:
            return []
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for ch in params_str:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        parts.append("".join(current).strip())
        return parts

    @classmethod
    def parse_function_pointer_param(
        cls, param_raw: str, custom_types: Optional[dict[str, Any]] = None
    ) -> Optional[CParameter]:
        """
        Parses a C function pointer parameter declaration, e.g.:
        'int (*compar)(const void*, const void*)' or 'void (*)(int)'.
        Returns a CParameter configured with ctypes.CFUNCTYPE.
        """
        p = param_raw.strip()
        m = re.match(r"^(.*?)\s*\(\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*\)\s*\((.*?)\)$", p)
        if not m:
            return None

        ret_type_raw = m.group(1).strip()
        fn_name = (m.group(2) or "").strip()
        param_list_raw = m.group(3).strip()

        ret_type_canon = cls.canonicalize_type(ret_type_raw)
        c_restype = cls.map_type_to_ctypes(ret_type_canon, custom_types=custom_types)

        c_argtypes: List[Any] = []
        if param_list_raw and param_list_raw != "void":
            for sub_p in cls._split_param_list(param_list_raw):
                sub_p = sub_p.strip()
                if not sub_p:
                    continue
                sub_param = cls.parse_parameter(sub_p, custom_types=custom_types)
                if sub_param is not None and sub_param.ctype is not None:
                    c_argtypes.append(sub_param.ctype)

        cfuntype = ctypes.CFUNCTYPE(c_restype, *c_argtypes)
        return CParameter(
            type_str=p,
            name=fn_name,
            ctype=cfuntype,
            ffi_type=cfuntype,
            is_function_pointer=True,
        )

    @classmethod
    def parse_parameter(
        cls, param_raw: str, custom_types: Optional[dict[str, Any]] = None
    ) -> Optional[CParameter]:
        """
        Parses a single parameter string from a prototype parameter list.
        Returns a CParameter instance or None if empty/void.
        """
        p = param_raw.strip()
        if not p or p == "void":
            return None

        # Varargs (...)
        if p == "...":
            return CParameter(type_str="...", name="...", ctype=None, ffi_type=None)

        # Function pointer parameter check: e.g. int (*cmp)(int, int)
        fn_ptr_param = cls.parse_function_pointer_param(p, custom_types=custom_types)
        if fn_ptr_param is not None:
            return fn_ptr_param

        # Array notation: e.g. 'double a[]' or 'const double arr[10]'
        m_arr = re.match(r"^(.*?)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*\d*\s*\]$", p)
        if m_arr:
            raw_type = cls.canonicalize_type(m_arr.group(1) + "*")
            param_name = m_arr.group(2)
            ctype = cls.map_type_to_ctypes(raw_type, custom_types=custom_types)
            ffi_type = cls.map_type_to_ffi(raw_type, custom_types=custom_types)
            return CParameter(type_str=raw_type, name=param_name, ctype=ctype, ffi_type=ffi_type)

        # Parameter ending with pointer asterisk has no name (unnamed parameter)
        if p.endswith("*"):
            raw_type = cls.canonicalize_type(p)
            ctype = cls.map_type_to_ctypes(raw_type, custom_types=custom_types)
            ffi_type = cls.map_type_to_ffi(raw_type, custom_types=custom_types)
            return CParameter(type_str=raw_type, name="", ctype=ctype, ffi_type=ffi_type)

        # Search for potential parameter name at the end
        m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*$", p)
        if not m:
            raw_type = cls.canonicalize_type(p)
            ctype = cls.map_type_to_ctypes(raw_type, custom_types=custom_types)
            ffi_type = cls.map_type_to_ffi(raw_type, custom_types=custom_types)
            return CParameter(type_str=raw_type, name="", ctype=ctype, ffi_type=ffi_type)

        cand_name = m.group(1)
        cand_type = p[: m.start()].strip()

        # Single word parameter: must be an unnamed type (e.g. 'int', 'size_t')
        if not cand_type:
            raw_type = cls.canonicalize_type(cand_name)
            ctype = cls.map_type_to_ctypes(raw_type, custom_types=custom_types)
            ffi_type = cls.map_type_to_ffi(raw_type, custom_types=custom_types)
            return CParameter(type_str=raw_type, name="", ctype=ctype, ffi_type=ffi_type)

        # In C, qualifiers or type keywords CANNOT be parameter identifiers!
        # E.g. 'int * const' or 'const char * const' or 'unsigned int' or 'long long'
        if cand_name.lower() in QUALIFIERS or cand_name.lower() in TYPE_KEYWORDS:
            raw_type = cls.canonicalize_type(p)
            ctype = cls.map_type_to_ctypes(raw_type, custom_types=custom_types)
            ffi_type = cls.map_type_to_ffi(raw_type, custom_types=custom_types)
            return CParameter(type_str=raw_type, name="", ctype=ctype, ffi_type=ffi_type)

        # cand_name is the parameter name, cand_type is the type
        raw_type = cls.canonicalize_type(cand_type)
        ctype = cls.map_type_to_ctypes(raw_type, custom_types=custom_types)
        ffi_type = cls.map_type_to_ffi(raw_type, custom_types=custom_types)
        return CParameter(type_str=raw_type, name=cand_name, ctype=ctype, ffi_type=ffi_type)

    @classmethod
    def parse_prototype(
        cls, raw: str, custom_types: Optional[dict[str, Any]] = None
    ) -> CFunctionPrototype:
        """
        Parses a single C function declaration statement into a CFunctionPrototype.
        Example: 'double* matmul(const double* a, const double* b, int n);'
        """
        s = raw.strip()
        if s.endswith(";"):
            s = s[:-1].strip()

        # Strip attributes with nested parentheses support
        s = re.sub(r"__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)", "", s)
        s = re.sub(r"__declspec\s*\((?:[^()]|\([^()]*\))*\)", "", s)

        # Strip modifiers, calling conventions, dll attributes
        s = re.sub(
            r"\b(__cdecl|_cdecl|__stdcall|_stdcall|__fastcall|_fastcall|__vectorcall|WINAPI|APIENTRY|PASCAL|CDECL|STDCALL|DLL_EXPORT|DLL_IMPORT|inline|__inline|__inline__|static|extern)\b",
            "",
            s,
        )
        s = s.strip()

        first_paren = s.find("(")
        last_paren = s.rfind(")")
        if first_paren == -1 or last_paren <= first_paren:
            raise CHeaderParseError(
                f"Invalid C function prototype (missing parameter parentheses): '{raw}'"
            )

        prefix = s[:first_paren].strip()
        params_str = s[first_paren + 1 : last_paren].strip()

        # Extract function name and return type from prefix
        m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*$", prefix)
        if not m:
            raise CHeaderParseError(
                f"Could not extract function name from prototype prefix: '{prefix}' (in '{raw}')"
            )

        func_name = m.group(1)
        ret_type_raw = prefix[: m.start()].strip()
        if not ret_type_raw:
            raise CHeaderParseError(
                f"Missing return type in function prototype: '{raw}'"
            )

        ret_type = cls.canonicalize_type(ret_type_raw)
        restype = cls.map_type_to_ctypes(ret_type, custom_types=custom_types)
        ffi_restype = cls.map_type_to_ffi(ret_type, custom_types=custom_types)

        # Parse parameter list with strict C99 validation
        parameters: List[CParameter] = []
        argtypes: List[Any] = []
        ffi_argtypes: List[Any] = []

        if not params_str or params_str == "void":
            pass
        else:
            raw_parts = cls._split_param_list(params_str)
            for idx, p_raw in enumerate(raw_parts):
                if not p_raw:
                    raise CHeaderParseError(
                        f"Empty parameter (stray comma) in function prototype: '{raw}'"
                    )
                if p_raw == "void":
                    raise CHeaderParseError(
                        f"Cannot use 'void' as parameter type when combined with other parameters in '{raw}'"
                    )
                if p_raw == "...":
                    if idx != len(raw_parts) - 1:
                        raise CHeaderParseError(
                            f"Variadic ellipsis '...' must be the last parameter in '{raw}'"
                        )
                    p = CParameter(type_str="...", name="...", ctype=None, ffi_type=None)
                    parameters.append(p)
                    continue

                p = cls.parse_parameter(p_raw, custom_types=custom_types)
                if p is None or p.ctype is None:
                    if p is not None and ("void" in p.type_str or "void" in p_raw):
                        raise CHeaderParseError(
                            f"Cannot use 'void' as parameter type in '{raw}'"
                        )
                    raise CHeaderParseError(
                        f"Cannot resolve parameter type for '{p_raw}' in '{raw}'"
                    )
                parameters.append(p)
                argtypes.append(p.ctype)
                ffi_argtypes.append(p.ffi_type or p.ctype)

        return CFunctionPrototype(
            name=func_name,
            return_type=ret_type,
            parameters=parameters,
            restype=restype,
            argtypes=argtypes,
            raw_signature=raw.strip(),
            ffi_restype=ffi_restype,
            ffi_argtypes=ffi_argtypes,
        )

    @staticmethod
    def strip_function_bodies_and_braces(code: str) -> str:
        """
        Removes function definitions with { ... } bodies, struct/enum bodies,
        and stray braces, preserving semicolons and function declarations.
        """
        result = []
        i = 0
        n = len(code)
        last_stmt_start = 0

        while i < n:
            ch = code[i]
            if ch == '{':
                prefix = code[last_stmt_start:i].strip()
                # Find matching '}'
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if code[j] == '{':
                        depth += 1
                    elif code[j] == '}':
                        depth -= 1
                    j += 1

                # If prefix contains ')' it is a function definition with a body
                if ')' in prefix:
                    result = result[:last_stmt_start]
                    i = j
                    last_stmt_start = len(result)
                    continue
                else:
                    # Struct, union, enum body or block
                    i = j
                    continue
            elif ch == ';':
                result.append(ch)
                i += 1
                last_stmt_start = len(result)
            elif ch == '}':
                # Stray closing brace (e.g. from extern "C" {)
                i += 1
            else:
                result.append(ch)
                i += 1

        return "".join(result)

    def parse(self, header_content_or_path: str) -> List[CFunctionPrototype]:
        """
        Parses a C header string or header file on disk and extracts all
        function prototypes.
        """
        # Check if argument is an existing file on disk
        if os.path.isfile(header_content_or_path):
            return self.parse_file(header_content_or_path)

        # If it looks like a file path (.h/.hpp) but doesn't exist, raise FileNotFoundError
        if not ("(" in header_content_or_path or ";" in header_content_or_path) and (
            header_content_or_path.endswith((".h", ".hpp", ".h++"))
            or "/" in header_content_or_path
            or "\\" in header_content_or_path
        ):
            raise FileNotFoundError(
                f"C header file not found: '{header_content_or_path}'"
            )

        content = header_content_or_path

        # Step 1: Strip comments
        cleaned = self.strip_comments(content)

        # Step 2: Strip preprocessor directives
        cleaned = self.strip_preprocessor(cleaned)

        # Step 3: Strip extern "C"
        cleaned = re.sub(r'extern\s+"C"\s*\{?', ' ', cleaned)

        # Step 4: Strip function bodies and braced constructs
        cleaned = self.strip_function_bodies_and_braces(cleaned)

        # Step 5: Split statements by semicolon
        prototypes: List[CFunctionPrototype] = []
        statements = cleaned.split(";")

        # Collect typedef aliases
        custom_types: dict[str, Any] = {}
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt.startswith("typedef"):
                continue
            rest = re.sub(r"^typedef\s+", "", stmt).strip()
            if "(" in rest or "{" in rest:
                continue
            m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*$", rest)
            if not m:
                continue
            alias = m.group(1)
            base_type = rest[: m.start()].strip()
            if not base_type:
                continue
            try:
                c_type_val = self.map_type_to_ctypes(base_type, custom_types=custom_types)
                custom_types[alias] = c_type_val
                custom_types[alias.lower()] = c_type_val
            except Exception:
                pass

        # Parse function prototypes
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt or "{" in stmt or "}" in stmt:
                continue
            if "(" not in stmt or ")" not in stmt:
                continue
            if stmt.startswith("typedef"):
                continue

            proto = self.parse_prototype(stmt + ";", custom_types=custom_types)
            prototypes.append(proto)

        return prototypes

    def parse_file(self, file_path: str) -> List[CFunctionPrototype]:
        """Reads a header file from disk and parses its function prototypes."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"C header file not found: '{file_path}'")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self.parse(content)


# =========================================================================
# Callable Bound C Function Wrapper
# =========================================================================

class CBoundFunction(CFunction):
    """
    Callable wrapper around a bound ctypes C function with automatic
    bidirectional argument and return value conversion, zero-copy Synapse
    Tensor / NumPy array buffer passing, Python buffer protocol (bytearray, memoryview),
    function pointer callbacks (CFUNCTYPE), Python list conversion, and string handling.
    """

    def __init__(
        self,
        name: str,
        func: Any,
        prototype: CFunctionPrototype,
    ):
        super().__init__(
            name=name,
            func=func,
            arg_types=prototype.argtypes,
            return_type=prototype.restype,
        )
        self.prototype = prototype

    def _convert_arg(self, arg: Any, expected_type: Any) -> Any:
        """Converts argument to the expected ctypes representation."""
        # 1. Synapse Tensor zero-copy pointer extraction
        if hasattr(arg, "data") and hasattr(arg.data, "ctypes"):
            arr = np.ascontiguousarray(arg.data)
            if expected_type is not None and hasattr(expected_type, "_type_"):
                return arr.ctypes.data_as(expected_type)
            return arr.ctypes.data_as(ctypes.c_void_p)

        # 2. NumPy ndarray zero-copy pointer extraction
        if isinstance(arg, np.ndarray):
            arr = np.ascontiguousarray(arg)
            if expected_type is not None and hasattr(expected_type, "_type_"):
                return arr.ctypes.data_as(expected_type)
            return arr.ctypes.data_as(ctypes.c_void_p)

        # 3. Python bytearray zero-copy buffer passing
        if isinstance(arg, bytearray):
            c_arr = (ctypes.c_char * len(arg)).from_buffer(arg)
            if expected_type is not None and expected_type != ctypes.c_void_p and hasattr(expected_type, "_type_"):
                try:
                    return ctypes.cast(c_arr, expected_type)
                except Exception:
                    pass
            return c_arr

        # 4. Python memoryview zero-copy buffer passing
        if isinstance(arg, memoryview):
            c_arr = (ctypes.c_char * arg.nbytes).from_buffer(arg)
            if expected_type is not None and expected_type != ctypes.c_void_p and hasattr(expected_type, "_type_"):
                try:
                    return ctypes.cast(c_arr, expected_type)
                except Exception:
                    pass
            return c_arr

        # 5. Python callable for CFUNCTYPE parameter
        if callable(arg) and expected_type is not None and hasattr(expected_type, "_argtypes_"):
            return expected_type(arg)

        # 6. Python str to C char* / wchar_t* / void*
        if isinstance(arg, str):
            if expected_type == ctypes.c_wchar_p:
                return arg
            if expected_type == ctypes.c_char_p or expected_type == ctypes.c_void_p:
                return arg.encode("utf-8")

        # 7. Python bytes to C void*
        if isinstance(arg, bytes) and expected_type == ctypes.c_void_p:
            return ctypes.cast(ctypes.c_char_p(arg), ctypes.c_void_p)

        # 8. Python list/tuple to ctypes array if a pointer is expected
        if isinstance(arg, (list, tuple)) and expected_type is not None:
            elem_type = getattr(expected_type, "_type_", None)
            if isinstance(elem_type, type):
                arr_type = elem_type * len(arg)
                return arr_type(*arg)
            if (expected_type == ctypes.c_void_p or expected_type is None) and len(arg) > 0:
                first = arg[0]
                if isinstance(first, float):
                    arr_type = ctypes.c_double * len(arg)
                    return arr_type(*arg)
                elif isinstance(first, int):
                    arr_type = ctypes.c_int64 * len(arg)
                    return arr_type(*arg)

        # 9. Standard argument passing
        return arg

    def __call__(self, *args: Any) -> Any:
        converted_args = []
        for i, arg in enumerate(args):
            expected = self.arg_types[i] if i < len(self.arg_types) else None
            converted_args.append(self._convert_arg(arg, expected))

        try:
            result = self.func(*converted_args)
        except Exception as e:
            raise SynapseFFIError(
                f"Error calling C function '{self.name}': {e}"
            ) from e

        # Automatic decoding for char_p return
        if self.return_type == ctypes.c_char_p and isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")

        return result

    def __repr__(self) -> str:
        sig = self.prototype.raw_signature or f"{self.prototype.return_type} {self.name}(...)"
        return f"<CBoundFunction: {sig}>"


# =========================================================================
# CBindResult: Dictionary with Attribute Access
# =========================================================================

class CBindResult(dict[str, Any]):
    """
    Dictionary mapping function names to callable CBoundFunction instances,
    supporting both dict-key access (bindings['add'](...)) and attribute access
    (bindings.add(...)).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No bound C function named '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __repr__(self) -> str:
        fn_keys = ", ".join(self.keys())
        return f"<CBindResult functions=[{fn_keys}]>"


# =========================================================================
# CBindManager: Header Binding & Cache Manager
# =========================================================================

class CBindManager:
    """
    Manages direct C header parsing and dynamic library bindings.
    Caches parsed prototypes, loaded dynamic libraries, and generated bindings.
    """

    def __init__(self, parser: Optional[CHeaderParser] = None):
        self.parser = parser or CHeaderParser()
        self._header_cache: dict[str, List[CFunctionPrototype]] = {}
        self._lib_cache: dict[str, CDynamicLibrary] = {}
        self._binding_cache: dict[tuple[str, str], CBindResult] = {}

    def _canonicalize_lib_key(self, lib_path_or_name: Any) -> str:
        """Determines canonical cache key for a library name, path, or instance."""
        if lib_path_or_name is None:
            return "msvcrt" if os.name == "nt" else "libc"
        if isinstance(lib_path_or_name, CDynamicLibrary):
            return f"CDynamicLibrary_{id(lib_path_or_name)}"
        if isinstance(lib_path_or_name, ctypes.CDLL):
            return f"CDLL_{id(lib_path_or_name)}"
        if not isinstance(lib_path_or_name, str):
            return str(lib_path_or_name)
        lowered = lib_path_or_name.lower()
        if os.name == "nt" and lowered in ("msvcrt", "msvcrt.dll", "libc", "libc.so", "libc.so.6", "c"):
            return "msvcrt"
        if os.name != "nt" and lowered in ("libc", "libc.so", "libc.so.6", "c"):
            return "libc"
        return os.path.abspath(lib_path_or_name) if os.path.isfile(lib_path_or_name) else lib_path_or_name

    def bind_header(
        self,
        header_content_or_path: str,
        library_path_or_name: Union[str, CDynamicLibrary, ctypes.CDLL, None] = None,
    ) -> CBindResult:
        """
        Parses a C header declaration string or file path, loads the target dynamic library,
        sets argtypes and restype on loaded C functions, and returns a dictionary of
        callable wrapper functions.

        Args:
            header_content_or_path: C header code string or absolute/relative path to .h file.
            library_path_or_name: Name or path to dynamic library (.dll/.so/.dylib),
                                  or standard C runtime ('msvcrt' on Windows, 'libc' on Unix),
                                  or an existing CDynamicLibrary / ctypes.CDLL instance.
                                  If None, defaults to platform C runtime.

        Returns:
            CBindResult containing callable CBoundFunction wrappers.
        """
        # Determine header cache key
        if os.path.isfile(header_content_or_path):
            header_key = os.path.abspath(header_content_or_path)
        else:
            header_key = header_content_or_path

        # Canonicalize library key
        lib_key = self._canonicalize_lib_key(library_path_or_name)
        binding_key = (header_key, lib_key)

        # Check binding cache
        if binding_key in self._binding_cache:
            return self._binding_cache[binding_key]

        # Check / populate header cache
        if header_key in self._header_cache:
            prototypes = self._header_cache[header_key]
        else:
            prototypes = self.parser.parse(header_content_or_path)
            self._header_cache[header_key] = prototypes

        # Check / populate library cache
        if lib_key in self._lib_cache:
            cdll = self._lib_cache[lib_key]
        elif isinstance(library_path_or_name, CDynamicLibrary):
            cdll = library_path_or_name
            self._lib_cache[lib_key] = cdll
        elif isinstance(library_path_or_name, ctypes.CDLL):
            cdll = CDynamicLibrary.__new__(CDynamicLibrary)
            cdll.lib_path = getattr(library_path_or_name, "_name", "cdll")
            cdll._lib = library_path_or_name
            cdll._bound_functions = {}
            self._lib_cache[lib_key] = cdll
        else:
            target_lib = library_path_or_name
            if os.name == "nt" and target_lib and isinstance(target_lib, str) and target_lib.lower() in ("libc", "libc.so", "libc.so.6"):
                target_lib = "msvcrt"
            cdll = load_library(target_lib)
            self._lib_cache[lib_key] = cdll

        # Bind each prototype
        result = CBindResult()
        for proto in prototypes:
            if not hasattr(cdll._lib, proto.name):
                raise SynapseFFIError(
                    f"Symbol '{proto.name}' declared in header was not found in library '{lib_key}'"
                )

            raw_func = getattr(cdll._lib, proto.name)
            raw_func.argtypes = proto.argtypes
            raw_func.restype = proto.restype

            bound_fn = CBoundFunction(
                name=proto.name,
                func=raw_func,
                prototype=proto,
            )
            result[proto.name] = bound_fn

        # Cache bindings
        self._binding_cache[binding_key] = result
        return result

    def clear_cache(self) -> None:
        """Clears all internal caches (headers, libraries, and bindings)."""
        self._header_cache.clear()
        self._lib_cache.clear()
        self._binding_cache.clear()

    def get_cached_prototypes(self, header_key: str) -> Optional[List[CFunctionPrototype]]:
        """Returns cached prototypes for a given header key if present."""
        return self._header_cache.get(header_key)

    def get_cached_bindings(
        self, header_key: str, library_key: str
    ) -> Optional[CBindResult]:
        """Returns cached bindings for a given header and library key if present."""
        return self._binding_cache.get((header_key, library_key))


# =========================================================================
# Global Singleton & Module Convenience Functions
# =========================================================================

_GLOBAL_BIND_MANAGER = CBindManager()


def bind_header(
    header_content_or_path: str,
    library_path_or_name: Union[str, CDynamicLibrary, ctypes.CDLL, None] = None,
) -> CBindResult:
    """
    Convenience function that parses a C header and binds functions using the
    global CBindManager instance.
    """
    return _GLOBAL_BIND_MANAGER.bind_header(header_content_or_path, library_path_or_name)


def parse_header(header_content_or_path: str) -> List[CFunctionPrototype]:
    """Convenience function to parse a C header string or file."""
    return _GLOBAL_BIND_MANAGER.parser.parse(header_content_or_path)


def parse_prototype(prototype_str: str) -> CFunctionPrototype:
    """Convenience function to parse a single C function prototype string."""
    return _GLOBAL_BIND_MANAGER.parser.parse_prototype(prototype_str)
