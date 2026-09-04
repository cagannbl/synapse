"""
Unit and integration tests for Synapse C-Bind: Direct C/C++ Header Parser and Binder.

Covers:
- Parsing primitive C function prototypes
- Parsing pointer, double pointer, and qualifier prototypes
- Parsing stdint, size_t, and unsigned types
- Parsing array notation parameters
- Parsing unnamed parameters
- Stripping comments (single line, multi-line, inline) and whitespace variations
- Stripping preprocessor directives and extern "C" blocks
- Binding to standard C runtime functions on Windows (msvcrt.dll or libc: abs, sqrt, puts)
- Calling bound functions with valid arguments and verifying numerical return values
- String parameter and return type handling (puts, strlen, atoi, atof)
- Zero-copy Synapse Tensor and NumPy buffer passing with native memory operations (memcpy, memset)
- Binding from header files on disk
- Attribute-based and dictionary-based access to bound functions
- CBindManager header, library, and binding caching mechanics
- Error handling for invalid C signatures, unknown types, nonexistent libraries, and missing symbols
"""
from __future__ import annotations

import ctypes
import os
import sys
import numpy as np
import pytest

from synapse.core.tensor import Tensor
from synapse.interop.c_ffi import SynapseFFIError
from synapse.interop.c_bind import (
    CBindManager,
    CBoundFunction,
    CBindResult,
    CFunctionPrototype,
    CHeaderParser,
    CHeaderParseError,
    CParameter,
    bind_header,
    parse_header,
    parse_prototype,
)


# =========================================================================
# 1. Prototype Parsing Tests
# =========================================================================

def test_parse_primitive_prototypes():
    """Tests parsing basic C function prototypes with primitive types."""
    parser = CHeaderParser()

    proto1 = parser.parse_prototype("int add(int a, int b);")
    assert proto1.name == "add"
    assert proto1.return_type == "int"
    assert len(proto1.parameters) == 2
    assert proto1.parameters[0].name == "a"
    assert proto1.parameters[0].type_str == "int"
    assert proto1.parameters[1].name == "b"
    assert proto1.parameters[1].type_str == "int"
    assert proto1.restype == ctypes.c_int32
    assert proto1.argtypes == [ctypes.c_int32, ctypes.c_int32]
    assert proto1.param_names == ["a", "b"]
    assert proto1.param_types == ["int", "int"]

    proto2 = parser.parse_prototype("double divide(double x, double y);")
    assert proto2.name == "divide"
    assert proto2.return_type == "double"
    assert proto2.restype == ctypes.c_double
    assert proto2.argtypes == [ctypes.c_double, ctypes.c_double]

    proto3 = parser.parse_prototype("void reset(void);")
    assert proto3.name == "reset"
    assert proto3.return_type == "void"
    assert proto3.restype is None
    assert proto3.parameters == []
    assert proto3.argtypes == []


def test_parse_pointer_and_qualifier_prototypes():
    """Tests parsing prototypes containing single/double pointers and qualifiers (const, restrict)."""
    parser = CHeaderParser()

    # matmul prototype with const pointers
    proto = parser.parse_prototype(
        "double* matmul(const double* a, const double* b, int n);"
    )
    assert proto.name == "matmul"
    assert proto.return_type == "double*"
    assert proto.restype == ctypes.POINTER(ctypes.c_double)
    assert len(proto.parameters) == 3
    assert proto.parameters[0].name == "a"
    assert proto.parameters[0].type_str == "const double*"
    assert proto.parameters[0].ctype == ctypes.POINTER(ctypes.c_double)
    assert proto.parameters[1].name == "b"
    assert proto.parameters[1].type_str == "const double*"
    assert proto.parameters[2].name == "n"
    assert proto.parameters[2].type_str == "int"
    assert proto.argtypes == [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
    ]

    # void* and size_t
    proto_alloc = parser.parse_prototype("void* malloc(size_t size);")
    assert proto_alloc.name == "malloc"
    assert proto_alloc.return_type == "void*"
    assert proto_alloc.restype == ctypes.c_void_p
    assert proto_alloc.argtypes == [ctypes.c_size_t]

    # double pointer
    proto_split = parser.parse_prototype(
        "char** split(const char* text, char delim, int* count);"
    )
    assert proto_split.name == "split"
    assert proto_split.return_type == "char**"
    assert proto_split.restype == ctypes.POINTER(ctypes.c_char_p)
    assert proto_split.argtypes == [
        ctypes.c_char_p,
        ctypes.c_char,
        ctypes.POINTER(ctypes.c_int32),
    ]


def test_parse_stdint_and_unsigned_types():
    """Tests parsing exact-width stdint types, unsigned types, and int64."""
    header = """
    uint32_t hash_func(const uint8_t* data, size_t len);
    int64_t sum64(int64_t a, int64_t b);
    unsigned int get_flags(unsigned short mask, unsigned long timeout);
    """
    parser = CHeaderParser()
    prototypes = parser.parse(header)
    assert len(prototypes) == 3

    p_hash, p_sum, p_flags = prototypes
    assert p_hash.name == "hash_func"
    assert p_hash.restype == ctypes.c_uint32
    assert p_hash.argtypes == [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

    assert p_sum.name == "sum64"
    assert p_sum.restype == ctypes.c_int64
    assert p_sum.argtypes == [ctypes.c_int64, ctypes.c_int64]

    assert p_flags.name == "get_flags"
    assert p_flags.restype == ctypes.c_uint32
    assert p_flags.argtypes == [ctypes.c_uint16, ctypes.c_ulong]


def test_parse_array_notation_parameters():
    """Tests that C array notation (e.g. double a[], float arr[10]) is parsed as pointer types."""
    parser = CHeaderParser()

    proto = parser.parse_prototype(
        "double dot_product(const double a[], const double b[], int n);"
    )
    assert proto.name == "dot_product"
    assert proto.return_type == "double"
    assert proto.parameters[0].name == "a"
    assert proto.parameters[0].type_str == "const double*"
    assert proto.parameters[0].ctype == ctypes.POINTER(ctypes.c_double)
    assert proto.parameters[1].name == "b"
    assert proto.parameters[1].type_str == "const double*"

    proto2 = parser.parse_prototype("void normalize(float arr[3]);")
    assert proto2.parameters[0].name == "arr"
    assert proto2.parameters[0].type_str == "float*"
    assert proto2.parameters[0].ctype == ctypes.POINTER(ctypes.c_float)


def test_parse_unnamed_parameters():
    """Tests parsing prototypes where parameter names are omitted (type-only)."""
    parser = CHeaderParser()

    proto = parser.parse_prototype("int add(int, int);")
    assert proto.name == "add"
    assert len(proto.parameters) == 2
    assert proto.parameters[0].name == ""
    assert proto.parameters[0].type_str == "int"
    assert proto.parameters[1].name == ""
    assert proto.parameters[1].type_str == "int"
    assert proto.argtypes == [ctypes.c_int32, ctypes.c_int32]

    proto2 = parser.parse_prototype("double* process(const double*, int);")
    assert proto2.parameters[0].type_str == "const double*"
    assert proto2.parameters[0].name == ""
    assert proto2.parameters[1].type_str == "int"
    assert proto2.parameters[1].name == ""


# =========================================================================
# 2. Stripping Comments, Directives & Formatting Variations
# =========================================================================

def test_strip_comments_and_whitespace_variations():
    """Tests stripping single-line, multi-line, and inline comments, and handling multi-line formatting."""
    header = """
    // Single line comment
    /* Multi-line
       comment spanning lines */
    int add(
        int /* inline comment */ a, // trailing comment
        int b
    );

    /* Another comment */
    double sqrt(double x);
    """
    parser = CHeaderParser()
    prototypes = parser.parse(header)
    assert len(prototypes) == 2

    assert prototypes[0].name == "add"
    assert prototypes[0].return_type == "int"
    assert prototypes[0].param_names == ["a", "b"]

    assert prototypes[1].name == "sqrt"
    assert prototypes[1].return_type == "double"
    assert prototypes[1].param_names == ["x"]


def test_strip_preprocessor_and_extern_guards():
    """Tests that header guards (#ifndef/#define), #include, and extern blocks are stripped cleanly."""
    header = """
    #ifndef MATH_UTILS_H
    #define MATH_UTILS_H

    #include <stddef.h>
    #include <stdint.h>

    #pragma once

    extern "C" {
        __declspec(dllexport) int abs(int n);
        extern double sqrt(double x);
    }

    #endif // MATH_UTILS_H
    """
    parser = CHeaderParser()
    prototypes = parser.parse(header)
    assert len(prototypes) == 2

    p_abs, p_sqrt = prototypes
    assert p_abs.name == "abs"
    assert p_abs.return_type == "int"
    assert p_abs.restype == ctypes.c_int32

    assert p_sqrt.name == "sqrt"
    assert p_sqrt.return_type == "double"
    assert p_sqrt.restype == ctypes.c_double


# =========================================================================
# 3. Dynamic Library Binding & Standard C Runtime Invocation
# =========================================================================

def test_bind_standard_c_runtime_math_functions():
    """Binds to standard C runtime (msvcrt on Windows) and tests abs, sqrt, sin, cos numerical calls."""
    header = """
    int abs(int n);
    double sqrt(double x);
    double sin(double x);
    double cos(double x);
    double ceil(double x);
    double floor(double x);
    """
    bindings = bind_header(header, "msvcrt")
    assert isinstance(bindings, CBindResult)
    assert isinstance(bindings["abs"], CBoundFunction)

    # 1. abs: int -> int
    assert bindings.abs(-42) == 42
    assert bindings["abs"](-99) == 99
    assert bindings.abs(0) == 0

    # 2. sqrt: double -> double
    assert bindings.sqrt(0.0) == 0.0
    assert bindings.sqrt(4.0) == 2.0
    assert bindings.sqrt(16.0) == 4.0
    assert bindings.sqrt(144.0) == 12.0
    assert abs(bindings.sqrt(2.0) - 1.41421356) < 1e-6

    # 3. sin & cos
    assert abs(bindings.sin(0.0)) < 1e-9
    assert abs(bindings.cos(0.0) - 1.0) < 1e-9

    # 4. ceil & floor
    assert bindings.ceil(3.2) == 4.0
    assert bindings.floor(3.8) == 3.0


def test_bind_standard_c_runtime_string_functions():
    """Tests string parameter and return value handling with puts, strlen, atoi, and atof."""
    header = """
    int puts(const char* str);
    size_t strlen(const char* s);
    int atoi(const char* str);
    double atof(const char* str);
    """
    bindings = bind_header(header, "msvcrt")

    # puts returns int (0 on success) and writes to stdout
    ret = bindings.puts("Testing puts from Synapse C-Bind")
    assert ret == 0

    # strlen converts str -> char_p and returns size_t
    assert bindings.strlen("Synapse AI") == 10
    assert bindings.strlen("") == 0
    assert bindings.strlen("Hello World!") == 12

    # atoi: string to integer
    assert bindings.atoi("12345") == 12345
    assert bindings.atoi("-987") == -987

    # atof: string to float
    assert abs(bindings.atof("3.14159") - 3.14159) < 1e-5
    assert abs(bindings.atof("-2.718") - (-2.718)) < 1e-5


def test_bind_memory_operations_with_synapse_tensors():
    """Tests zero-copy pointer passing of Synapse Tensors and NumPy arrays using native memcpy and memset."""
    header = """
    void* memcpy(void* dest, const void* src, size_t n);
    void* memset(void* s, int c, size_t n);
    """
    bindings = bind_header(header, "msvcrt")

    # 1. NumPy array zero-copy memcpy
    src_np = np.array([10.5, 20.5, 30.5, 40.5], dtype=np.float64)
    dst_np = np.zeros(4, dtype=np.float64)
    bindings.memcpy(dst_np, src_np, src_np.nbytes)
    assert np.array_equal(dst_np, src_np)

    # 2. Synapse Tensor zero-copy memcpy
    src_tensor = Tensor([1.0, 2.0, 3.0, 4.0])
    dst_tensor = Tensor([0.0, 0.0, 0.0, 0.0])
    byte_count = 4 * 8  # 4 double elements (8 bytes each)
    bindings.memcpy(dst_tensor, src_tensor, byte_count)
    assert dst_tensor.data[0] == 1.0
    assert dst_tensor.data[1] == 2.0
    assert dst_tensor.data[2] == 3.0
    assert dst_tensor.data[3] == 4.0

    # 3. memset on byte buffer
    buf = np.zeros(8, dtype=np.uint8)
    bindings.memset(buf, 0x41, 4)  # set first 4 bytes to 'A' (ASCII 65)
    assert buf[0] == 65
    assert buf[1] == 65
    assert buf[2] == 65
    assert buf[3] == 65
    assert buf[4] == 0


def test_bind_with_python_list_conversion():
    """Tests that passing Python lists to pointer arguments automatically converts to ctypes arrays."""
    header = """
    void* memcpy(void* dest, const void* src, size_t n);
    """
    bindings = bind_header(header, "msvcrt")

    # Pass Python list to src pointer
    src_list = [100.0, 200.0, 300.0]
    dst_np = np.zeros(3, dtype=np.float64)
    bindings.memcpy(dst_np, src_list, 3 * 8)
    assert dst_np[0] == 100.0
    assert dst_np[1] == 200.0
    assert dst_np[2] == 300.0


def test_bind_header_from_file_on_disk(tmp_path):
    """Tests reading and binding from a physical .h file created on disk."""
    header_path = tmp_path / "math_test.h"
    header_path.write_text(
        """
        #ifndef MATH_TEST_H
        #define MATH_TEST_H

        int abs(int n);
        double sqrt(double x);

        #endif
        """,
        encoding="utf-8",
    )

    bindings = bind_header(str(header_path), "msvcrt")
    assert "abs" in bindings
    assert "sqrt" in bindings
    assert bindings.abs(-77) == 77
    assert bindings.sqrt(81.0) == 9.0


def test_bind_result_dict_and_attribute_access():
    """Tests that CBindResult supports both dict-style access and attribute access."""
    header = "int abs(int n); double sqrt(double x);"
    res = bind_header(header, "msvcrt")

    # Dict access
    assert res["abs"](-10) == 10
    assert res["sqrt"](100.0) == 10.0

    # Attribute access
    assert res.abs(-10) == 10
    assert res.sqrt(100.0) == 10.0

    # Dict methods
    assert set(res.keys()) == {"abs", "sqrt"}
    assert len(res) == 2
    assert "abs" in res
    assert "<CBindResult" in repr(res)

    # Missing attribute
    with pytest.raises(AttributeError):
        _ = res.non_existent_fn


def test_bind_header_libc_alias_on_windows():
    """Verifies that passing 'libc' or 'msvcrt.dll' on Windows correctly resolves to msvcrt."""
    header = "int abs(int n);"
    bindings_libc = bind_header(header, "libc")
    assert bindings_libc.abs(-15) == 15

    bindings_dll = bind_header(header, "msvcrt.dll")
    assert bindings_dll.abs(-25) == 25


# =========================================================================
# 4. Caching Mechanics
# =========================================================================

def test_cbind_manager_caching():
    """Tests that CBindManager caches parsed prototypes, loaded dynamic libraries, and bindings."""
    mgr = CBindManager()
    header = "int abs(int n); double sqrt(double x);"

    # First bind: populates cache
    bindings1 = mgr.bind_header(header, "msvcrt")
    assert len(mgr._header_cache) == 1
    assert len(mgr._lib_cache) == 1
    assert len(mgr._binding_cache) == 1

    # Second bind with identical header and lib: retrieved from cache
    bindings2 = mgr.bind_header(header, "msvcrt")
    assert bindings1 is bindings2

    # Verify cached prototypes accessor
    cached_protos = mgr.get_cached_prototypes(header)
    assert cached_protos is not None
    assert len(cached_protos) == 2

    # Clear cache
    mgr.clear_cache()
    assert len(mgr._header_cache) == 0
    assert len(mgr._lib_cache) == 0
    assert len(mgr._binding_cache) == 0


# =========================================================================
# 5. Error Handling & Edge Cases
# =========================================================================

def test_error_missing_parentheses():
    """Tests that invalid C prototypes missing parameter parentheses raise CHeaderParseError."""
    parser = CHeaderParser()
    with pytest.raises(CHeaderParseError, match="missing parameter parentheses"):
        parser.parse_prototype("int add int a, int b;")


def test_error_missing_function_name():
    """Tests that invalid prototypes missing a function name raise CHeaderParseError."""
    parser = CHeaderParser()
    with pytest.raises(CHeaderParseError):
        parser.parse_prototype("int (int a, int b);")


def test_error_missing_return_type():
    """Tests that prototypes missing a return type raise CHeaderParseError."""
    parser = CHeaderParser()
    with pytest.raises(CHeaderParseError):
        parser.parse_prototype("(int a, int b);")


def test_error_unknown_c_type():
    """Tests that unrecognized C types raise CHeaderParseError."""
    parser = CHeaderParser()
    with pytest.raises(CHeaderParseError, match="Unknown C type"):
        parser.parse_prototype("UnknownCustomType calc(int a);")


def test_error_void_as_parameter_type():
    """Tests that using 'void' as a typed parameter raises CHeaderParseError."""
    parser = CHeaderParser()
    with pytest.raises(CHeaderParseError, match="Cannot use 'void' as parameter type"):
        parser.parse_prototype("int bad_func(void a);")


def test_error_nonexistent_shared_library():
    """Tests that requesting a nonexistent dynamic library raises SynapseFFIError."""
    header = "int abs(int n);"
    with pytest.raises(SynapseFFIError):
        bind_header(header, "nonexistent_lib_xyz_12345.dll")


def test_error_missing_symbol_in_library():
    """Tests that declaring a function not exported by the target library raises SynapseFFIError."""
    header = "void definitely_non_existent_function_9999(void);"
    with pytest.raises(SynapseFFIError, match="not found in library"):
        bind_header(header, "msvcrt")


def test_error_nonexistent_header_file():
    """Tests that passing a path to a nonexistent .h file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="C header file not found"):
        parse_header("nonexistent_header_file_xyz_888.h")


# =========================================================================
# 6. Advanced Edge Cases, Qualifiers, Wide Chars & Callbacks
# =========================================================================

def test_parse_unnamed_pointer_qualifiers_not_treated_as_names():
    """Tests that unnamed parameters ending with qualifiers (const, restrict, volatile) or type keywords are not treated as parameter names."""
    parser = CHeaderParser()

    p_const = parser.parse_parameter("int * const")
    assert p_const is not None
    assert p_const.name == ""
    assert p_const.ctype == ctypes.POINTER(ctypes.c_int32)
    assert "const" in p_const.type_str

    p_restrict = parser.parse_parameter("const char * restrict")
    assert p_restrict is not None
    assert p_restrict.name == ""
    assert p_restrict.ctype == ctypes.c_char_p

    p_u64 = parser.parse_parameter("unsigned long long")
    assert p_u64 is not None
    assert p_u64.name == ""
    assert p_u64.ctype == ctypes.c_uint64

    # Combined in full prototype
    proto = parser.parse_prototype("int process(int * const, const char * restrict);")
    assert proto.parameters[0].name == ""
    assert proto.parameters[1].name == ""
    assert proto.argtypes == [ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p]


def test_parse_syntax_error_void_combinations():
    """Tests that invalid C99 parameter combinations like (int, void) or trailing commas raise CHeaderParseError."""
    parser = CHeaderParser()

    with pytest.raises(CHeaderParseError):
        parser.parse_prototype("int bad1(int a, void);")

    with pytest.raises(CHeaderParseError):
        parser.parse_prototype("int bad2(void, int b);")

    with pytest.raises(CHeaderParseError):
        parser.parse_prototype("int bad3(int a, );")

    with pytest.raises(CHeaderParseError):
        parser.parse_prototype("int bad4(..., int a);")


def test_map_c_types_to_ffi():
    """Tests mapping standard C types to Synapse C-FFI types and prototype FFI attributes."""
    parser = CHeaderParser()

    assert parser.map_type_to_ffi("int") == ctypes.c_int32
    assert parser.map_type_to_ffi("float64") == ctypes.c_double
    assert parser.map_type_to_ffi("char_p") == ctypes.c_char_p

    proto = parser.parse_prototype("double compute(int x, float y);")
    assert proto.ffi_restype == ctypes.c_double
    assert proto.ffi_argtypes == [ctypes.c_int32, ctypes.c_float]
    assert proto.parameters[0].ffi_type == ctypes.c_int32
    assert proto.parameters[1].ffi_type == ctypes.c_float


def test_windows_wchar_support():
    """Tests wide character (wchar_t and wchar_t*) binding and execution on Windows msvcrt."""
    header = """
    size_t wcslen(const wchar_t* str);
    wchar_t towlower(wchar_t c);
    wchar_t towupper(wchar_t c);
    """
    bindings = bind_header(header, "msvcrt")
    assert bindings.wcslen("Synapse") == 7
    assert bindings.wcslen("Merhaba Dünya") == 13
    assert bindings.wcslen("") == 0

    assert bindings.towlower("A") == "a"
    assert bindings.towupper("z") == "Z"


def test_header_with_inline_functions_and_structs():
    """Tests that inline functions with { ... } bodies and struct definitions do not corrupt subsequent prototypes."""
    header = """
    struct Point {
        double x;
        double y;
    };

    static inline double get_x(struct Point* p) {
        return p->x;
    }

    int abs(int n);
    double sqrt(double x);
    """
    parser = CHeaderParser()
    prototypes = parser.parse(header)
    names = [p.name for p in prototypes]
    assert "abs" in names
    assert "sqrt" in names
    assert "get_x" not in names  # inline definition discarded, declarations preserved

    bindings = bind_header(header, "msvcrt")
    assert bindings.abs(-50) == 50
    assert bindings.sqrt(25.0) == 5.0


def test_nested_compiler_attributes_parsing():
    """Tests that complex attributes with nested parentheses like __attribute__((visibility("default"))) are stripped cleanly."""
    header = """
    __attribute__((visibility("default"))) int abs(int n);
    __declspec(align(16)) double sqrt(double x);
    """
    parser = CHeaderParser()
    protos = parser.parse(header)
    assert len(protos) == 2
    assert protos[0].name == "abs"
    assert protos[0].return_type == "int"
    assert protos[1].name == "sqrt"
    assert protos[1].return_type == "double"


def test_buffer_protocol_bytearray_and_memoryview():
    """Tests zero-copy buffer operations using Python bytearray and memoryview directly with native memset/memcpy."""
    header = """
    void* memset(void* s, int c, size_t n);
    void* memcpy(void* dest, const void* src, size_t n);
    """
    bindings = bind_header(header, "msvcrt")

    # 1. bytearray with memset
    ba = bytearray(10)
    bindings.memset(ba, 65, 5)
    assert ba[:5] == b"AAAAA"
    assert ba[5:] == b"\x00\x00\x00\x00\x00"

    # 2. memoryview with memset
    mv = memoryview(ba)
    bindings.memset(mv, 66, 3)
    assert ba[:3] == b"BBB"
    assert ba[3:5] == b"AA"

    # 3. bytearray to bytearray memcpy
    src = bytearray(b"SYNAPSE")
    dst = bytearray(7)
    bindings.memcpy(dst, src, len(src))
    assert dst == b"SYNAPSE"


def test_typedef_resolution_in_headers():
    """Tests that typedef declarations inside headers resolve correctly for function parameters and return types."""
    header = """
    typedef double real_t;
    typedef unsigned long long uint64;

    real_t sqrt(real_t x);
    int abs(int n);
    """
    parser = CHeaderParser()
    protos = parser.parse(header)
    assert len(protos) == 2

    p_sqrt = protos[0]
    assert p_sqrt.name == "sqrt"
    assert p_sqrt.restype == ctypes.c_double
    assert p_sqrt.argtypes == [ctypes.c_double]

    bindings = bind_header(header, "msvcrt")
    assert bindings.sqrt(49.0) == 7.0


def test_function_pointer_callback_with_qsort():
    """Tests function pointer callback parameters by sorting an integer array using standard C runtime qsort."""
    header = """
    void qsort(void* base, size_t num, size_t size, int (*compar)(const void*, const void*));
    """
    parser = CHeaderParser()
    proto = parser.parse_prototype("void qsort(void* base, size_t num, size_t size, int (*compar)(const void*, const void*));")
    assert proto.name == "qsort"
    assert len(proto.parameters) == 4
    assert proto.parameters[3].is_function_pointer is True

    bindings = bind_header(header, "msvcrt")

    # Native qsort call with Python comparator function
    def py_compare(a_ptr, b_ptr):
        val_a = ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
        val_b = ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
        return val_a - val_b

    arr = (ctypes.c_int * 6)(40, 10, 50, 20, 60, 30)
    bindings.qsort(arr, len(arr), ctypes.sizeof(ctypes.c_int), py_compare)
    assert list(arr) == [10, 20, 30, 40, 50, 60]


def test_bind_manager_with_cdll_and_cdynamiclibrary_instances():
    """Tests passing existing CDynamicLibrary or ctypes.CDLL instances directly to CBindManager."""
    from synapse.interop.c_ffi import load_library
    mgr = CBindManager()

    header = "int abs(int n); double sqrt(double x);"

    # Pass ctypes.CDLL directly
    cdll = ctypes.cdll.msvcrt
    res1 = mgr.bind_header(header, cdll)
    assert res1.abs(-123) == 123

    # Pass Synapse CDynamicLibrary directly
    syn_lib = load_library("msvcrt")
    res2 = mgr.bind_header(header, syn_lib)
    assert res2.sqrt(64.0) == 8.0

