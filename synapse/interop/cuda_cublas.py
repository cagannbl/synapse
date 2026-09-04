"""
Synapse CUDA cuBLAS C-FFI & GPU Acceleration Bridge.

Provides high-performance matrix multiplication acceleration by dynamically
loading NVIDIA cuBLAS dynamic libraries (.dll on Windows, .so on Linux) via ctypes.

When cuBLAS or CUDA hardware is unavailable, all functions safely degrade
with zero exceptions, enabling the caller to gracefully fall back to CPU/OpenMP.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import logging
import os
import sys
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np

logger = logging.getLogger("synapse.cuda.cublas")

# =========================================================================
# cuBLAS Status & Operation Enumerations
# =========================================================================

CUBLAS_STATUS_SUCCESS = 0
CUBLAS_STATUS_NOT_INITIALIZED = 1
CUBLAS_STATUS_ALLOC_FAILED = 3
CUBLAS_STATUS_INVALID_VALUE = 7
CUBLAS_STATUS_ARCH_MISMATCH = 8
CUBLAS_STATUS_MAPPING_ERROR = 11
CUBLAS_STATUS_EXECUTION_FAILED = 13
CUBLAS_STATUS_INTERNAL_ERROR = 14
CUBLAS_STATUS_NOT_SUPPORTED = 15
CUBLAS_STATUS_LICENSE_ERROR = 16

CUBLAS_STATUS_NAMES: dict[int, str] = {
    0: "CUBLAS_STATUS_SUCCESS",
    1: "CUBLAS_STATUS_NOT_INITIALIZED",
    3: "CUBLAS_STATUS_ALLOC_FAILED",
    7: "CUBLAS_STATUS_INVALID_VALUE",
    8: "CUBLAS_STATUS_ARCH_MISMATCH",
    11: "CUBLAS_STATUS_MAPPING_ERROR",
    13: "CUBLAS_STATUS_EXECUTION_FAILED",
    14: "CUBLAS_STATUS_INTERNAL_ERROR",
    15: "CUBLAS_STATUS_NOT_SUPPORTED",
    16: "CUBLAS_STATUS_LICENSE_ERROR",
}

# cublasOperation_t
CUBLAS_OP_N = 0  # Non-transpose
CUBLAS_OP_T = 1  # Transpose
CUBLAS_OP_C = 2  # Conjugate transpose

# cudaMemcpyKind
CUDA_MEMCPY_HOST_TO_HOST = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


# =========================================================================
# Dynamic Library Search Helpers
# =========================================================================

def _get_search_paths() -> list[str]:
    """Gathers prospective search directories for CUDA and cuBLAS binaries."""
    paths: list[str] = []

    # Environment variables
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        val = os.environ.get(env_var)
        if val and os.path.isdir(val):
            paths.append(val)
            bin_dir = os.path.join(val, "bin")
            if os.path.isdir(bin_dir):
                paths.append(bin_dir)
            lib_dir = os.path.join(val, "lib64" if sys.platform != "win32" else "lib\\x64")
            if os.path.isdir(lib_dir):
                paths.append(lib_dir)

    # Specific version variables on Windows (e.g. CUDA_PATH_V12_0, CUDA_PATH_V11_8)
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH_V") and os.path.isdir(v):
            paths.append(v)
            bin_dir = os.path.join(v, "bin")
            if os.path.isdir(bin_dir):
                paths.append(bin_dir)

    if sys.platform.startswith("win"):
        # Common Windows installation directories
        toolkit_pattern = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin"
        for p in glob.glob(toolkit_pattern):
            if os.path.isdir(p) and p not in paths:
                paths.append(p)
    else:
        # Common Linux / Unix directories
        unix_dirs = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
        ]
        for p in unix_dirs:
            if os.path.isdir(p) and p not in paths:
                paths.append(p)

    return paths


def find_cublas_library() -> Optional[str]:
    """
    Searches for the NVIDIA cuBLAS dynamic shared library across standard locations.

    Windows: cublas64_12.dll, cublas64_11.dll, cublas64_10.dll, etc.
    Linux: libcublas.so, libcublas.so.12, libcublas.so.11, etc.
    macOS: libcublas.dylib
    """
    if sys.platform.startswith("win"):
        candidates = [
            "cublas64_12.dll",
            "cublas64_11.dll",
            "cublas64_10.dll",
            "cublas64_100.dll",
            "cublas64_92.dll",
            "cublas.dll",
        ]
    elif sys.platform == "darwin":
        candidates = ["libcublas.dylib"]
    else:
        candidates = [
            "libcublas.so",
            "libcublas.so.12",
            "libcublas.so.11",
            "libcublas.so.10",
        ]

    # 1. Search in custom CUDA paths
    search_dirs = _get_search_paths()
    for directory in search_dirs:
        for name in candidates:
            candidate_path = os.path.join(directory, name)
            if os.path.isfile(candidate_path):
                return candidate_path

    # 2. Search using ctypes.util.find_library
    for base in ("cublas", "cublas64_12", "cublas64_11", "cublas64_10"):
        found = ctypes.util.find_library(base)
        if found:
            return found

    # 3. Try plain names directly
    for name in candidates:
        try:
            # Check if OS loader can find it directly in system PATH / LD_LIBRARY_PATH
            if sys.platform.startswith("win"):
                lib = ctypes.windll.LoadLibrary(name)
            else:
                lib = ctypes.CDLL(name)
            if lib is not None:
                return name
        except Exception:
            continue

    return None


def find_cudart_library() -> Optional[str]:
    """
    Searches for the CUDA Runtime library (cudart) for GPU memory management
    (cudaMalloc, cudaFree, cudaMemcpy).
    """
    if sys.platform.startswith("win"):
        candidates = [
            "cudart64_12.dll",
            "cudart64_11.dll",
            "cudart64_10.dll",
            "cudart64_100.dll",
            "cudart.dll",
        ]
    elif sys.platform == "darwin":
        candidates = ["libcudart.dylib"]
    else:
        candidates = [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.so.10",
        ]

    search_dirs = _get_search_paths()
    for directory in search_dirs:
        for name in candidates:
            candidate_path = os.path.join(directory, name)
            if os.path.isfile(candidate_path):
                return candidate_path

    for base in ("cudart", "cudart64_12", "cudart64_11", "cudart64_10"):
        found = ctypes.util.find_library(base)
        if found:
            return found

    for name in candidates:
        try:
            if sys.platform.startswith("win"):
                lib = ctypes.windll.LoadLibrary(name)
            else:
                lib = ctypes.CDLL(name)
            if lib is not None:
                return name
        except Exception:
            continue

    return None


def _load_library_safe(lib_path_or_name: str) -> Optional[Any]:
    """Safely loads a dynamic library via ctypes.cdll or ctypes.windll."""
    # 1. Try ctypes.cdll
    try:
        return ctypes.cdll.LoadLibrary(lib_path_or_name)
    except Exception:
        pass

    # 2. Try ctypes.CDLL
    try:
        return ctypes.CDLL(lib_path_or_name)
    except Exception:
        pass

    # 3. Try ctypes.windll on Windows
    if sys.platform.startswith("win"):
        try:
            return ctypes.windll.LoadLibrary(lib_path_or_name)
        except Exception:
            pass

    return None


def _set_fn_signature(fn: Any, argtypes: Sequence[Any], restype: Any) -> None:
    """Safely assigns argtypes and restype to a C function pointer or mock callable."""
    try:
        fn.argtypes = list(argtypes)
    except (AttributeError, TypeError):
        pass
    try:
        fn.restype = restype
    except (AttributeError, TypeError):
        pass


# =========================================================================
# CuBlasEngine: High-Performance cuBLAS Interoperability
# =========================================================================

class CuBlasEngine:
    """
    Synapse cuBLAS C-FFI Hardware Acceleration Engine.

    Dynamically binds cublasCreate_v2, cublasDestroy_v2, cublasSgemm_v2,
    and cublasDgemm_v2. Manages device memory allocations and transfers
    for zero-overhead matrix multiplication on NVIDIA GPUs.

    If CUDA or cuBLAS is unavailable or encounters an error, the engine
    never raises exceptions: is_available() returns False and matmul()
    returns None, allowing callers to gracefully fall back to CPU/OpenMP.
    """

    def __init__(
        self,
        lib: Optional[Any] = None,
        cudart_lib: Optional[Any] = None,
    ):
        self._lib: Optional[Any] = None
        self._cudart: Optional[Any] = None
        self._handle: Optional[ctypes.c_void_p] = None
        self._available: bool = False
        self._lib_path: Optional[str] = None

        # Bound cuBLAS function pointers
        self._cublas_create: Optional[Any] = None
        self._cublas_destroy: Optional[Any] = None
        self._cublas_sgemm: Optional[Any] = None
        self._cublas_dgemm: Optional[Any] = None
        self._cublas_set_matrix: Optional[Any] = None
        self._cublas_get_matrix: Optional[Any] = None

        # Bound CUDA Runtime memory management function pointers
        self._cuda_malloc: Optional[Any] = None
        self._cuda_free: Optional[Any] = None
        self._cuda_memcpy: Optional[Any] = None

        self._initialize(lib, cudart_lib)

    def _initialize(self, lib: Optional[Any] = None, cudart_lib: Optional[Any] = None) -> None:
        """Attempts to load libraries and bind C-FFI function signatures."""
        try:
            # 1. Load cuBLAS library
            if lib is not None:
                if isinstance(lib, str):
                    self._lib = _load_library_safe(lib)
                    self._lib_path = lib
                else:
                    self._lib = lib
                    self._lib_path = getattr(lib, "_name", "custom_cublas_mock")
            else:
                cublas_path = find_cublas_library()
                if cublas_path:
                    self._lib = _load_library_safe(cublas_path)
                    self._lib_path = cublas_path

            if self._lib is None:
                self._available = False
                return

            # 2. Bind cuBLAS functions
            if not self._bind_cublas_signatures():
                self._available = False
                return

            # 3. Load CUDA Runtime (cudart) for memory management
            if cudart_lib is not None:
                if isinstance(cudart_lib, str):
                    self._cudart = _load_library_safe(cudart_lib)
                else:
                    self._cudart = cudart_lib
            else:
                cudart_path = find_cudart_library()
                if cudart_path:
                    self._cudart = _load_library_safe(cudart_path)
                elif self._lib is not None:
                    # In some builds, cudart functions are exported directly from cuBLAS
                    if hasattr(self._lib, "cudaMalloc"):
                        self._cudart = self._lib

            # Bind CUDA memory functions if available
            self._bind_cuda_signatures()

            self._available = True
            logger.debug(f"cuBLAS dynamic library loaded successfully from {self._lib_path}")

        except Exception as e:
            logger.debug(f"cuBLAS initialization suppressed error: {e}")
            self._available = False
            self._lib = None

    def _bind_cublas_signatures(self) -> bool:
        """Binds typed C signatures for cublasCreate, cublasDestroy, cublasSgemm, cublasDgemm."""
        try:
            # cublasCreate_v2 (fallback: cublasCreate)
            create_fn = getattr(self._lib, "cublasCreate_v2", None) or getattr(self._lib, "cublasCreate", None)
            if create_fn is None:
                return False
            _set_fn_signature(create_fn, [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
            self._cublas_create = create_fn

            # cublasDestroy_v2 (fallback: cublasDestroy)
            destroy_fn = getattr(self._lib, "cublasDestroy_v2", None) or getattr(self._lib, "cublasDestroy", None)
            if destroy_fn is None:
                return False
            _set_fn_signature(destroy_fn, [ctypes.c_void_p], ctypes.c_int)
            self._cublas_destroy = destroy_fn

            # cublasSgemm_v2 (fallback: cublasSgemm)
            sgemm_fn = getattr(self._lib, "cublasSgemm_v2", None) or getattr(self._lib, "cublasSgemm", None)
            if sgemm_fn is None:
                return False
            _set_fn_signature(
                sgemm_fn,
                [
                    ctypes.c_void_p,                     # handle
                    ctypes.c_int,                        # transa (cublasOperation_t)
                    ctypes.c_int,                        # transb (cublasOperation_t)
                    ctypes.c_int,                        # m
                    ctypes.c_int,                        # n
                    ctypes.c_int,                        # k
                    ctypes.POINTER(ctypes.c_float),      # alpha
                    ctypes.c_void_p,                     # A (device pointer)
                    ctypes.c_int,                        # lda
                    ctypes.c_void_p,                     # B (device pointer)
                    ctypes.c_int,                        # ldb
                    ctypes.POINTER(ctypes.c_float),      # beta
                    ctypes.c_void_p,                     # C (device pointer)
                    ctypes.c_int,                        # ldc
                ],
                ctypes.c_int,
            )
            self._cublas_sgemm = sgemm_fn

            # cublasDgemm_v2 (fallback: cublasDgemm)
            dgemm_fn = getattr(self._lib, "cublasDgemm_v2", None) or getattr(self._lib, "cublasDgemm", None)
            if dgemm_fn is None:
                return False
            _set_fn_signature(
                dgemm_fn,
                [
                    ctypes.c_void_p,                     # handle
                    ctypes.c_int,                        # transa (cublasOperation_t)
                    ctypes.c_int,                        # transb (cublasOperation_t)
                    ctypes.c_int,                        # m
                    ctypes.c_int,                        # n
                    ctypes.c_int,                        # k
                    ctypes.POINTER(ctypes.c_double),     # alpha
                    ctypes.c_void_p,                     # A (device pointer)
                    ctypes.c_int,                        # lda
                    ctypes.c_void_p,                     # B (device pointer)
                    ctypes.c_int,                        # ldb
                    ctypes.POINTER(ctypes.c_double),     # beta
                    ctypes.c_void_p,                     # C (device pointer)
                    ctypes.c_int,                        # ldc
                ],
                ctypes.c_int,
            )
            self._cublas_dgemm = dgemm_fn

            # Optional: cublasSetMatrix / cublasGetMatrix
            set_mat = getattr(self._lib, "cublasSetMatrix", None)
            if set_mat is not None:
                _set_fn_signature(
                    set_mat,
                    [
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int
                    ],
                    ctypes.c_int,
                )
                self._cublas_set_matrix = set_mat

            get_mat = getattr(self._lib, "cublasGetMatrix", None)
            if get_mat is not None:
                _set_fn_signature(
                    get_mat,
                    [
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int
                    ],
                    ctypes.c_int,
                )
                self._cublas_get_matrix = get_mat

            return True
        except Exception as e:
            logger.debug(f"Failed to bind cuBLAS signatures: {e}")
            return False

    def _bind_cuda_signatures(self) -> None:
        """Binds typed signatures for cudaMalloc, cudaFree, cudaMemcpy if cudart is present."""
        if self._cudart is None:
            return

        try:
            malloc_fn = getattr(self._cudart, "cudaMalloc", None)
            if malloc_fn is not None:
                _set_fn_signature(malloc_fn, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int)
                self._cuda_malloc = malloc_fn

            free_fn = getattr(self._cudart, "cudaFree", None)
            if free_fn is not None:
                _set_fn_signature(free_fn, [ctypes.c_void_p], ctypes.c_int)
                self._cuda_free = free_fn

            memcpy_fn = getattr(self._cudart, "cudaMemcpy", None)
            if memcpy_fn is not None:
                _set_fn_signature(memcpy_fn, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int)
                self._cuda_memcpy = memcpy_fn
        except Exception as e:
            logger.debug(f"CUDA memory function binding note: {e}")

            logger.debug(f"CUDA memory function binding note: {e}")

    def is_available(self) -> bool:
        """Returns True if the cuBLAS dynamic library is successfully loaded and bound."""
        return self._available

    def _get_or_create_handle(self) -> Optional[ctypes.c_void_p]:
        """Lazily creates and caches the cuBLAS context handle."""
        if self._handle is not None:
            return self._handle

        if not self._available or self._cublas_create is None:
            return None

        try:
            handle = ctypes.c_void_p()
            status = self._cublas_create(ctypes.byref(handle))
            if status == CUBLAS_STATUS_SUCCESS and handle.value:
                self._handle = handle
                return self._handle
            else:
                status_name = CUBLAS_STATUS_NAMES.get(status, f"CODE_{status}")
                logger.debug(f"cublasCreate failed with status: {status_name}")
                return None
        except Exception as e:
            logger.debug(f"cublasCreate exception: {e}")
            return None

    def matmul(self, A: Any, B: Any) -> Optional[Any]:
        """
        Performs matrix multiplication (A @ B) accelerated by cuBLAS on GPU.

        Args:
            A: Left-hand Synapse Tensor or NumPy 2D array.
            B: Right-hand Synapse Tensor or NumPy 2D array.

        Returns:
            A new Synapse Tensor containing the matrix product, or None if
            cuBLAS is unavailable or an error occurs (triggering graceful fallback).
        """
        if not self._available:
            return None

        # 1. Validate inputs and extract continuous numpy buffers
        try:
            a_data = A.data if hasattr(A, "data") else np.asarray(A)
            b_data = B.data if hasattr(B, "data") else np.asarray(B)
        except Exception:
            return None

        if not isinstance(a_data, np.ndarray) or not isinstance(b_data, np.ndarray):
            return None

        # Must be 2-dimensional matrices
        if a_data.ndim != 2 or b_data.ndim != 2:
            return None

        M, K_A = a_data.shape
        K_B, N = b_data.shape

        if K_A != K_B:
            # Dimension mismatch: caller can raise or fall back
            return None

        K = K_A

        # Ensure supported floating-point dtype (float32 or float64)
        if a_data.dtype == np.float32 and b_data.dtype == np.float32:
            target_dtype = np.float32
            use_double = False
        else:
            target_dtype = np.float64
            use_double = True

        a_contiguous = np.ascontiguousarray(a_data, dtype=target_dtype)
        b_contiguous = np.ascontiguousarray(b_data, dtype=target_dtype)

        # 2. Acquire cuBLAS handle
        handle = self._get_or_create_handle()
        if handle is None:
            return None

        # 3. Check memory management functions
        if self._cuda_malloc is None or self._cuda_free is None or self._cuda_memcpy is None:
            return None

        d_A = ctypes.c_void_p()
        d_B = ctypes.c_void_p()
        d_C = ctypes.c_void_p()

        elem_size = 8 if use_double else 4
        size_A = M * K * elem_size
        size_B = K * N * elem_size
        size_C = M * N * elem_size

        try:
            # Allocate device memory
            if self._cuda_malloc(ctypes.byref(d_A), size_A) != 0:
                return None
            if self._cuda_malloc(ctypes.byref(d_B), size_B) != 0:
                return None
            if self._cuda_malloc(ctypes.byref(d_C), size_C) != 0:
                return None

            # Copy data from host to device
            a_ptr = a_contiguous.ctypes.data_as(ctypes.c_void_p)
            b_ptr = b_contiguous.ctypes.data_as(ctypes.c_void_p)

            if self._cuda_memcpy(d_A, a_ptr, size_A, CUDA_MEMCPY_HOST_TO_DEVICE) != 0:
                return None
            if self._cuda_memcpy(d_B, b_ptr, size_B, CUDA_MEMCPY_HOST_TO_DEVICE) != 0:
                return None

            # 4. Invoke cuBLAS GEMM
            #
            # Matrix Layout Note:
            # Synapse and NumPy use row-major (C) ordering, whereas cuBLAS expects
            # column-major (Fortran) ordering.
            # Identity: C = A @ B in row-major is equivalent to C^T = B^T @ A^T in column-major.
            # Passing B as matrix 1 (dim N x K) and A as matrix 2 (dim K x M) computes
            # result matrix C^T (dim N x M) in column-major, which in row-major memory
            # is identical to C (dim M x N).
            #
            # Dimensions in cuBLAS call:
            # m = N, n = M, k = K
            # lda = N, ldb = K, ldc = N
            m_cublas = N
            n_cublas = M
            k_cublas = K
            lda = N
            ldb = K
            ldc = N

            if use_double:
                if self._cublas_dgemm is None:
                    return None
                alpha = ctypes.c_double(1.0)
                beta = ctypes.c_double(0.0)
                status = self._cublas_dgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m_cublas,
                    n_cublas,
                    k_cublas,
                    ctypes.byref(alpha),
                    d_B,
                    lda,
                    d_A,
                    ldb,
                    ctypes.byref(beta),
                    d_C,
                    ldc,
                )
            else:
                if self._cublas_sgemm is None:
                    return None
                alpha = ctypes.c_float(1.0)
                beta = ctypes.c_float(0.0)
                status = self._cublas_sgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m_cublas,
                    n_cublas,
                    k_cublas,
                    ctypes.byref(alpha),
                    d_B,
                    lda,
                    d_A,
                    ldb,
                    ctypes.byref(beta),
                    d_C,
                    ldc,
                )

            if status != CUBLAS_STATUS_SUCCESS:
                status_name = CUBLAS_STATUS_NAMES.get(status, f"CODE_{status}")
                logger.debug(f"cuBLAS GEMM failed with status: {status_name}")
                return None

            # 5. Copy result back from device to host
            c_host = np.empty((M, N), dtype=target_dtype)
            c_ptr = c_host.ctypes.data_as(ctypes.c_void_p)
            if self._cuda_memcpy(c_ptr, d_C, size_C, CUDA_MEMCPY_DEVICE_TO_HOST) != 0:
                return None

            # 6. Wrap into Synapse Tensor
            from synapse.core.tensor import Tensor
            from synapse.core.device import Device

            requires_grad = (
                getattr(A, "requires_grad", False) or getattr(B, "requires_grad", False)
            )
            target_device = (
                getattr(A, "device", Device("cuda"))
                if getattr(A, "device", None) and getattr(A.device, "device_type", None) == "cuda"
                else Device("cuda")
            )

            return Tensor(
                c_host,
                requires_grad=requires_grad,
                device=target_device,
                _children=(A, B) if isinstance(A, Tensor) and isinstance(B, Tensor) else (),
                _op="@cublas",
            )

        except Exception as e:
            logger.debug(f"cuBLAS matmul exception: {e}")
            return None
        finally:
            # Always ensure allocated device buffers are freed
            if d_A.value and self._cuda_free is not None:
                try:
                    self._cuda_free(d_A)
                except Exception:
                    pass
            if d_B.value and self._cuda_free is not None:
                try:
                    self._cuda_free(d_B)
                except Exception:
                    pass
            if d_C.value and self._cuda_free is not None:
                try:
                    self._cuda_free(d_C)
                except Exception:
                    pass

    def close(self) -> None:
        """Releases the cuBLAS context handle."""
        if self._handle is not None and self._cublas_destroy is not None:
            try:
                self._cublas_destroy(self._handle)
            except Exception:
                pass
            finally:
                self._handle = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "Available" if self._available else "Unavailable"
        return f"<CuBlasEngine [{status}] library={self._lib_path}>"
