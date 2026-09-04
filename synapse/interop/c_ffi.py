"""
Synapse C-FFI Bridge (Native Dynamic Library Interoperability).

Provides seamless bindings to native C dynamic libraries (.dll on Windows,
.so on Linux, .dylib on macOS) and C standard runtimes (msvcrt on Windows,
libc on Unix). Enables zero-copy memory operations directly on Synapse Tensor
underlying data buffers.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Any, Callable, Optional, Sequence, Union
import numpy as np

# =========================================================================
# Custom Exceptions
# =========================================================================

class SynapseFFIError(Exception):
    """Raised when an error occurs in the Synapse C-FFI bridge."""
    pass


# =========================================================================
# C-FFI Type Definitions & Mappings
# =========================================================================

int8 = ctypes.c_int8
int16 = ctypes.c_int16
int32 = ctypes.c_int32
int64 = ctypes.c_int64
uint8 = ctypes.c_uint8
uint16 = ctypes.c_uint16
uint32 = ctypes.c_uint32
uint64 = ctypes.c_uint64
float32 = ctypes.c_float
float64 = ctypes.c_double
char_p = ctypes.c_char_p
void_p = ctypes.c_void_p
size_t = ctypes.c_size_t
bool_t = ctypes.c_bool
double = ctypes.c_double
float_t = ctypes.c_float
int_t = ctypes.c_int

# Pointer helpers
int32_p = ctypes.POINTER(ctypes.c_int32)
int64_p = ctypes.POINTER(ctypes.c_int64)
float32_p = ctypes.POINTER(ctypes.c_float)
float64_p = ctypes.POINTER(ctypes.c_double)

TYPE_MAP: dict[Union[str, None], Any] = {
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "int": ctypes.c_int,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "float": ctypes.c_float,
    "float32": ctypes.c_float,
    "double": ctypes.c_double,
    "float64": ctypes.c_double,
    "char_p": ctypes.c_char_p,
    "char*": ctypes.c_char_p,
    "str": ctypes.c_char_p,
    "string": ctypes.c_char_p,
    "void_p": ctypes.c_void_p,
    "void*": ctypes.c_void_p,
    "ptr": ctypes.c_void_p,
    "pointer": ctypes.c_void_p,
    "size_t": ctypes.c_size_t,
    "bool": ctypes.c_bool,
    "void": None,
    None: None,
}


def resolve_type(t: Any) -> Any:
    """
    Resolves a type specification (string name or ctypes type) to a valid ctypes type.
    Supports pointer syntax like 'float64*', 'int32*', 'double*'.
    """
    if t is None:
        return None
    if isinstance(t, str):
        cleaned = t.strip()
        if cleaned in TYPE_MAP:
            return TYPE_MAP[cleaned]
        if cleaned.endswith("*"):
            base = cleaned[:-1].strip()
            base_type = resolve_type(base)
            if base_type is None or base_type == ctypes.c_void_p:
                return ctypes.c_void_p
            return ctypes.POINTER(base_type)
        raise SynapseFFIError(f"Unknown FFI type name: '{t}'")
    # Assume it is already a ctypes type or pointer
    return t


def pointer_to(t: Any) -> Any:
    """Returns ctypes POINTER for the specified type or type name."""
    resolved = resolve_type(t)
    if resolved is None or resolved == ctypes.c_void_p:
        return ctypes.c_void_p
    return ctypes.POINTER(resolved)


# =========================================================================
# Tensor Pointer & Zero-Copy Helpers
# =========================================================================

def get_tensor_pointer(tensor_or_array: Any, ctype: Any = None) -> Any:
    """
    Extracts a raw ctypes pointer from a Synapse Tensor or NumPy array buffer.
    Enables zero-copy C interoperability without memory duplication.

    Args:
        tensor_or_array: A Synapse Tensor or numpy.ndarray.
        ctype: Target ctypes pointer type (e.g. void_p or ctypes.POINTER(ctypes.c_double)).
               Defaults to ctypes.c_void_p.

    Returns:
        A ctypes pointer pointing to the contiguous memory buffer.
    """
    target = resolve_type(ctype) if ctype is not None else ctypes.c_void_p

    # Check for Synapse Tensor
    if hasattr(tensor_or_array, "data") and hasattr(tensor_or_array.data, "ctypes"):
        arr = np.ascontiguousarray(tensor_or_array.data)
        return arr.ctypes.data_as(target)

    # Check for NumPy array
    if isinstance(tensor_or_array, np.ndarray):
        arr = np.ascontiguousarray(tensor_or_array)
        return arr.ctypes.data_as(target)

    # Already a ctypes pointer or address
    if isinstance(tensor_or_array, (ctypes.c_void_p, int)):
        if isinstance(tensor_or_array, int):
            return ctypes.cast(ctypes.c_void_p(tensor_or_array), target)
        return ctypes.cast(tensor_or_array, target)

    raise TypeError(
        f"Expected Synapse Tensor or NumPy ndarray, got {type(tensor_or_array).__name__}"
    )


# =========================================================================
# Callable C Function Wrapper
# =========================================================================

class CFunction:
    """
    Callable wrapper around a bound ctypes function with automatic argument conversion,
    including zero-copy conversion of Synapse Tensors.
    """

    def __init__(
        self,
        name: str,
        func: Any,
        arg_types: Sequence[Any],
        return_type: Any,
    ):
        self.name = name
        self.func = func
        self.arg_types = list(arg_types)
        self.return_type = return_type

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

        # 3. String to C char* (bytes)
        if isinstance(arg, str) and expected_type == ctypes.c_char_p:
            return arg.encode("utf-8")

        # 4. Standard argument passing (ctypes handles numbers and pointers)
        return arg

    def __call__(self, *args: Any) -> Any:
        converted_args = []
        for i, arg in enumerate(args):
            expected = self.arg_types[i] if i < len(self.arg_types) else None
            converted_args.append(self._convert_arg(arg, expected))

        try:
            result = self.func(*converted_args)
        except Exception as e:
            raise SynapseFFIError(f"Error calling C function '{self.name}': {e}") from e

        # Automatic decoding for char_p return
        if self.return_type == ctypes.c_char_p and isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")

        return result

    def __repr__(self) -> str:
        args_repr = ", ".join(getattr(t, "__name__", str(t)) for t in self.arg_types)
        ret_repr = getattr(self.return_type, "__name__", str(self.return_type))
        return f"<CFunction {self.name}({args_repr}) -> {ret_repr}>"


# =========================================================================
# CDynamicLibrary: Dynamic Library Loader
# =========================================================================

class CDynamicLibrary:
    """
    Manages loading of dynamic C libraries (.dll on Windows, .so on Linux, .dylib on macOS)
    or default system C runtime libraries, and binding C functions.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path
        self._lib = self._load_library(lib_path)
        self._bound_functions: dict[str, CFunction] = {}

    def _load_library(self, lib_path: Optional[str]) -> ctypes.CDLL:
        """Loads the requested library or the platform standard C runtime."""
        # Case 1: None -> Load system C runtime
        if lib_path is None:
            if os.name == "nt":
                # Windows target: libc.so.6 does NOT exist on Windows.
                # Must use ctypes.cdll.msvcrt
                try:
                    return ctypes.cdll.msvcrt
                except Exception as e:
                    raise SynapseFFIError(f"Failed to load Windows C runtime (msvcrt): {e}") from e
            else:
                # Linux / macOS
                libc_name = ctypes.util.find_library("c")
                if libc_name:
                    return ctypes.CDLL(libc_name)
                for fallback in ("libc.so.6", "libc.dylib", "libc.so"):
                    try:
                        return ctypes.CDLL(fallback)
                    except Exception:
                        pass
                try:
                    return ctypes.CDLL(None)  # Standard C symbols loaded in process
                except Exception as e:
                    raise SynapseFFIError(f"Failed to load system C runtime: {e}") from e

        # Case 2: Explicit 'msvcrt' or 'libc' on Windows
        if os.name == "nt" and lib_path.lower() in ("msvcrt", "msvcrt.dll", "libc", "libc.so", "libc.so.6", "c"):
            return ctypes.cdll.msvcrt

        # Case 3: Absolute or relative file path on disk
        if os.path.isfile(lib_path):
            try:
                return ctypes.CDLL(os.path.abspath(lib_path))
            except Exception as e:
                raise SynapseFFIError(f"Failed to load library file '{lib_path}': {e}") from e

        # Case 4: Load by name via standard search path
        try:
            return ctypes.CDLL(lib_path)
        except Exception:
            pass

        # Case 5: Locate via ctypes.util.find_library
        found = ctypes.util.find_library(lib_path)
        if found:
            try:
                return ctypes.CDLL(found)
            except Exception as e:
                raise SynapseFFIError(f"Failed to load library '{lib_path}' (found '{found}'): {e}") from e

        raise SynapseFFIError(f"Dynamic library not found or could not be loaded: '{lib_path}'")

    def bind(
        self,
        func_name: str,
        arg_types: Optional[Sequence[Any]] = None,
        return_type: Any = None,
    ) -> CFunction:
        """
        Binds a function from the dynamic library with typed signatures.

        Args:
            func_name: Name of the exported C function (e.g. 'abs', 'ceil', 'memcpy').
            arg_types: List or tuple of argument types (e.g. [float64] or ['float64']).
            return_type: Return type (e.g. float64, int32, void_p, or None).

        Returns:
            A callable CFunction instance.
        """
        try:
            raw_func = getattr(self._lib, func_name)
        except AttributeError:
            raise SynapseFFIError(
                f"Symbol '{func_name}' not found in library: {self.lib_path or 'C Runtime'}"
            )

        resolved_args = [resolve_type(t) for t in (arg_types or [])]
        resolved_ret = resolve_type(return_type)

        raw_func.argtypes = resolved_args
        raw_func.restype = resolved_ret

        c_func = CFunction(
            name=func_name,
            func=raw_func,
            arg_types=resolved_args,
            return_type=resolved_ret,
        )
        self._bound_functions[func_name] = c_func
        return c_func

    def has_symbol(self, func_name: str) -> bool:
        """Checks if a function symbol exists in the library."""
        return hasattr(self._lib, func_name)

    def get_symbol(self, name: str) -> Any:
        """Retrieves raw symbol attribute from the underlying CDLL."""
        if not hasattr(self._lib, name):
            raise SynapseFFIError(f"Symbol '{name}' not found in library")
        return getattr(self._lib, name)

    def raw_pointer(self, tensor_or_array: Any, ctype: Any = None) -> Any:
        """Convenience method for zero-copy pointer extraction from Tensor or array."""
        return get_tensor_pointer(tensor_or_array, ctype)

    def __getattr__(self, name: str) -> Any:
        """Provides direct access to library functions or bound functions."""
        if name in self._bound_functions:
            return self._bound_functions[name]
        try:
            return getattr(self._lib, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute or symbol '{name}'")

    def __repr__(self) -> str:
        name = self.lib_path if self.lib_path else ("msvcrt (Windows)" if os.name == "nt" else "libc (Unix)")
        return f"<CDynamicLibrary: {name}>"


# =========================================================================
# Library Loader Factory Function
# =========================================================================

def load_library(name_or_path: Optional[str] = None) -> CDynamicLibrary:
    """
    Loads a dynamic library or default system C runtime.

    Args:
        name_or_path: Optional library name (e.g. 'msvcrt', 'kernel32', 'm')
                      or file path (.dll, .so, .dylib). If None, automatically loads
                      the platform standard C runtime (msvcrt on Windows, libc on Unix).

    Returns:
        CDynamicLibrary instance.
    """
    return CDynamicLibrary(name_or_path)


# =========================================================================
# cuBLAS Acceleration Engine Singleton
# =========================================================================

_GLOBAL_CUBLAS_ENGINE: Optional[Any] = None


def get_cublas_engine() -> Any:
    """
    Returns the global singleton CuBlasEngine instance for hardware-accelerated
    cuBLAS matrix multiplication.

    If cuBLAS or CUDA hardware is unavailable, returns an engine where
    is_available() == False without raising any exceptions, allowing safe fallback.
    """
    global _GLOBAL_CUBLAS_ENGINE
    if _GLOBAL_CUBLAS_ENGINE is None:
        from synapse.interop.cuda_cublas import CuBlasEngine
        _GLOBAL_CUBLAS_ENGINE = CuBlasEngine()
    return _GLOBAL_CUBLAS_ENGINE

