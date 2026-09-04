"""
Unit and integration tests for Synapse CUDA cuBLAS C-FFI Bridge and GPU Acceleration.

Covers:
- Safe import and dynamic loading on systems without CUDA / GPU hardware
- CuBlasEngine.is_available() safe boolean query without exceptions
- CuBlasEngine.matmul() graceful fallback (returning None) when cuBLAS is unavailable
- Shape and dimension mismatch handling
- Global get_cublas_engine() singleton verification
- Mocked cuBLAS C-FFI execution verifying cublasCreate_v2, cublasDestroy_v2,
  cublasSgemm_v2, cublasDgemm_v2, memory transfers, and error resilience
"""
import ctypes
from typing import Any
import numpy as np
import pytest

from synapse.core.tensor import Tensor
from synapse.interop.c_ffi import get_cublas_engine
from synapse.interop.cuda_cublas import (
    CuBlasEngine,
    find_cublas_library,
    find_cudart_library,
    CUBLAS_STATUS_SUCCESS,
    CUBLAS_STATUS_EXECUTION_FAILED,
    CUBLAS_STATUS_ALLOC_FAILED,
    CUBLAS_OP_N,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CUDA_MEMCPY_DEVICE_TO_HOST,
)
from synapse.interop import CuBlasEngine as ExportedCuBlasEngine
from synapse.interop import get_cublas_engine as exported_get_cublas_engine


# =========================================================================
# 1. Non-GPU / Safe Degraded Environment Tests
# =========================================================================

def test_cublas_import_and_exports():
    """Verifies that cuBLAS bridge can be imported safely from all entrypoints."""
    assert CuBlasEngine is ExportedCuBlasEngine
    assert get_cublas_engine is exported_get_cublas_engine

    # Search functions return None or str without error
    cublas_path = find_cublas_library()
    assert cublas_path is None or isinstance(cublas_path, str)

    cudart_path = find_cudart_library()
    assert cudart_path is None or isinstance(cudart_path, str)


def test_cublas_engine_availability_safe():
    """Verifies that is_available() always returns a boolean and never raises."""
    engine = CuBlasEngine()
    available = engine.is_available()
    assert isinstance(available, bool)
    assert repr(engine).startswith("<CuBlasEngine")


def test_cublas_engine_singleton():
    """Verifies get_cublas_engine() singleton helper behavior."""
    e1 = get_cublas_engine()
    e2 = get_cublas_engine()
    assert e1 is e2
    assert isinstance(e1, CuBlasEngine)


def test_cublas_matmul_graceful_fallback_when_unavailable():
    """
    Verifies that matmul returns None when cuBLAS is unavailable,
    allowing the caller to gracefully fall back to CPU/OpenMP.
    """
    # Create an engine with no library
    engine = CuBlasEngine(lib=None)
    assert not engine.is_available()

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    result = engine.matmul(a, b)
    assert result is None


def test_cublas_matmul_invalid_inputs_handled_safely():
    """Verifies that invalid shapes and types do not cause crashes."""
    engine = CuBlasEngine()

    # 1D tensors
    a_1d = Tensor([1.0, 2.0])
    b_1d = Tensor([3.0, 4.0])
    assert engine.matmul(a_1d, b_1d) is None

    # Shape mismatch: (2, 3) @ (4, 2)
    a_mismatch = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b_mismatch = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    assert engine.matmul(a_mismatch, b_mismatch) is None

    # None inputs
    assert engine.matmul(None, None) is None


# =========================================================================
# 2. Mock cuBLAS C-FFI Functionality & Execution Tests
# =========================================================================

class MockCUDAContext:
    """Simulates device memory buffers and C-level memory copy operations."""

    def __init__(self):
        self._buffers: dict[int, bytearray] = {}
        self._next_addr = 0x10000

    def malloc(self, size: int) -> int:
        addr = self._next_addr
        self._next_addr += size + 64
        self._buffers[addr] = bytearray(size)
        return addr

    def free(self, addr: int) -> None:
        self._buffers.pop(addr, None)

    def get_buffer(self, addr: int) -> bytearray:
        return self._buffers[addr]


def _get_ctypes_target(ptr: Any) -> Any:
    target = getattr(ptr, "_obj", None)
    if target is None:
        target = getattr(ptr, "contents", None)
    return target


class MockCuBlasLib:
    """Mock dynamic library providing cuBLAS and cudart C functions."""

    def __init__(self, fail_create: bool = False, fail_gemm: bool = False, fail_malloc: bool = False):
        self._name = "mock_cublas.dll"
        self.mem = MockCUDAContext()
        self.fail_create = fail_create
        self.fail_gemm = fail_gemm
        self.fail_malloc = fail_malloc
        self.handle_created = False
        self.handle_destroyed = False
        self.sgemm_called = 0
        self.dgemm_called = 0

    def cublasCreate_v2(self, handle_ptr):
        if self.fail_create:
            return CUBLAS_STATUS_ALLOC_FAILED
        target = _get_ctypes_target(handle_ptr)
        if target is not None:
            target.value = 0xCAFE
        self.handle_created = True
        return CUBLAS_STATUS_SUCCESS

    def cublasDestroy_v2(self, handle):
        self.handle_destroyed = True
        return CUBLAS_STATUS_SUCCESS

    def cudaMalloc(self, dev_ptr, size):
        if self.fail_malloc:
            return 1  # cudaErrorMemoryAllocation
        addr = self.mem.malloc(size)
        target = _get_ctypes_target(dev_ptr)
        if target is not None:
            target.value = addr
        return 0

    def cudaFree(self, dev_ptr):
        addr = getattr(dev_ptr, "value", dev_ptr) if getattr(dev_ptr, "value", None) is not None else dev_ptr
        self.mem.free(addr)
        return 0

    def cudaMemcpy(self, dst, src, count, kind):
        dst_val = getattr(dst, "value", dst) if getattr(dst, "value", None) is not None else dst
        src_val = getattr(src, "value", src) if getattr(src, "value", None) is not None else src
        if kind == CUDA_MEMCPY_HOST_TO_DEVICE:
            # src is host ctypes pointer, dst is simulated device address integer
            src_bytes = ctypes.string_at(src_val, count)
            buf = self.mem.get_buffer(dst_val)
            buf[:count] = src_bytes
            return 0
        elif kind == CUDA_MEMCPY_DEVICE_TO_HOST:
            # src is simulated device address integer, dst is host pointer
            buf = self.mem.get_buffer(src_val)
            ctypes.memmove(dst_val, (ctypes.c_char * count).from_buffer_copy(buf[:count]), count)
            return 0
        return 0

    def cublasSgemm_v2(self, handle, transa, transb, m, n, k, alpha_p, d_B, lda, d_A, ldb, beta_p, d_C, ldc):
        self.sgemm_called += 1
        if self.fail_gemm:
            return CUBLAS_STATUS_EXECUTION_FAILED

        addr_B = getattr(d_B, "value", d_B) if getattr(d_B, "value", None) is not None else d_B
        addr_A = getattr(d_A, "value", d_A) if getattr(d_A, "value", None) is not None else d_A
        addr_C = getattr(d_C, "value", d_C) if getattr(d_C, "value", None) is not None else d_C

        # Reconstruct B (dim n x k in col-major = k x n in row-major)
        b_raw = self.mem.get_buffer(addr_B)
        b_arr = np.frombuffer(b_raw, dtype=np.float32).reshape(k, m)  # k x n

        # Reconstruct A (dim k x m in col-major = m x k in row-major)
        a_raw = self.mem.get_buffer(addr_A)
        a_arr = np.frombuffer(a_raw, dtype=np.float32).reshape(n, k)  # m x k

        # Matrix multiply: C = A @ B (shape: n x m -> M x N)
        c_res = (a_arr @ b_arr).astype(np.float32)

        c_buf = self.mem.get_buffer(addr_C)
        c_bytes = c_res.tobytes()
        c_buf[:len(c_bytes)] = c_bytes
        return CUBLAS_STATUS_SUCCESS

    def cublasDgemm_v2(self, handle, transa, transb, m, n, k, alpha_p, d_B, lda, d_A, ldb, beta_p, d_C, ldc):
        self.dgemm_called += 1
        if self.fail_gemm:
            return CUBLAS_STATUS_EXECUTION_FAILED

        addr_B = getattr(d_B, "value", d_B) if getattr(d_B, "value", None) is not None else d_B
        addr_A = getattr(d_A, "value", d_A) if getattr(d_A, "value", None) is not None else d_A
        addr_C = getattr(d_C, "value", d_C) if getattr(d_C, "value", None) is not None else d_C

        b_raw = self.mem.get_buffer(addr_B)
        b_arr = np.frombuffer(b_raw, dtype=np.float64).reshape(k, m)

        a_raw = self.mem.get_buffer(addr_A)
        a_arr = np.frombuffer(a_raw, dtype=np.float64).reshape(n, k)

        c_res = (a_arr @ b_arr).astype(np.float64)

        c_buf = self.mem.get_buffer(addr_C)
        c_bytes = c_res.tobytes()
        c_buf[:len(c_bytes)] = c_bytes
        return CUBLAS_STATUS_SUCCESS


def test_mock_cublas_sgemm_float32():
    """Tests successful single-precision cuBLAS GEMM via CuBlasEngine."""
    mock_lib = MockCuBlasLib()
    engine = CuBlasEngine(lib=mock_lib, cudart_lib=mock_lib)

    assert engine.is_available()

    a_mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_mat = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a = Tensor(a_mat, dtype="float32")
    b = Tensor(b_mat, dtype="float32")

    c = engine.matmul(a, b)
    assert c is not None
    assert isinstance(c, Tensor)
    assert mock_lib.sgemm_called == 1
    assert mock_lib.dgemm_called == 0

    expected = a_mat @ b_mat
    assert np.allclose(c.data, expected)


def test_mock_cublas_dgemm_float64():
    """Tests successful double-precision cuBLAS GEMM via CuBlasEngine."""
    mock_lib = MockCuBlasLib()
    engine = CuBlasEngine(lib=mock_lib, cudart_lib=mock_lib)

    assert engine.is_available()

    a_mat = np.array([[2.5, -1.0, 3.0], [0.5, 4.0, -2.0]], dtype=np.float64)
    b_mat = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]], dtype=np.float64)

    a = Tensor(a_mat, dtype="float64")
    b = Tensor(b_mat, dtype="float64")

    c = engine.matmul(a, b)
    assert c is not None
    assert isinstance(c, Tensor)
    assert mock_lib.dgemm_called == 1
    assert mock_lib.sgemm_called == 0

    expected = a_mat @ b_mat
    assert np.allclose(c.data, expected)


def test_mock_cublas_gemm_failure_returns_none_gracefully():
    """Verifies that if cuBLAS GEMM returns an error status, matmul returns None."""
    mock_lib = MockCuBlasLib(fail_gemm=True)
    engine = CuBlasEngine(lib=mock_lib, cudart_lib=mock_lib)

    assert engine.is_available()

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    result = engine.matmul(a, b)
    assert result is None  # Graceful fallback


def test_mock_cublas_malloc_failure_returns_none_gracefully():
    """Verifies that if cudaMalloc fails, matmul returns None without crashing."""
    mock_lib = MockCuBlasLib(fail_malloc=True)
    engine = CuBlasEngine(lib=mock_lib, cudart_lib=mock_lib)

    assert engine.is_available()

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    result = engine.matmul(a, b)
    assert result is None  # Graceful fallback


def test_mock_cublas_create_failure_returns_none_gracefully():
    """Verifies that if cublasCreate fails, matmul returns None without crashing."""
    mock_lib = MockCuBlasLib(fail_create=True)
    engine = CuBlasEngine(lib=mock_lib, cudart_lib=mock_lib)

    assert engine.is_available()

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    result = engine.matmul(a, b)
    assert result is None  # Graceful fallback
