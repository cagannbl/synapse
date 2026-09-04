import ctypes
import gc
import numpy as np
import pytest

from synapse.core.tensor import Tensor
from synapse.interop.dlpack import (
    DLDevice,
    DLDataType,
    DLTensor,
    DLManagedTensor,
    DLDeviceType,
    DLDataTypeCode,
    is_dlpack_available,
    from_dlpack,
    to_dlpack,
    _ACTIVE_MANAGED_TENSORS,
)
from synapse.interop.python_bridge import to_synapse_tensor, to_numpy


# =========================================================================
# 1. Availability and Structure Layout Tests
# =========================================================================
def test_dlpack_availability():
    """DLPack runtime should be available via ctypes.pythonapi."""
    assert is_dlpack_available() is True


def test_dlpack_structure_sizes_and_alignments():
    """Verify that DLPack C-ABI structures match standard 64-bit sizes."""
    assert ctypes.sizeof(DLDevice) == 8
    assert ctypes.sizeof(DLDataType) == 4
    assert ctypes.sizeof(DLTensor) == 48
    assert ctypes.sizeof(DLManagedTensor) == 64

    # Check field offsets
    assert DLDevice.device_type.offset == 0
    assert DLDevice.device_id.offset == 4

    assert DLDataType.code.offset == 0
    assert DLDataType.bits.offset == 1
    assert DLDataType.lanes.offset == 2

    assert DLTensor.data.offset == 0
    assert DLTensor.device.offset == 8
    assert DLTensor.ndim.offset == 16
    assert DLTensor.dtype.offset == 20
    assert DLTensor.shape.offset == 24
    assert DLTensor.strides.offset == 32
    assert DLTensor.byte_offset.offset == 40


# =========================================================================
# 2. to_dlpack Tests
# =========================================================================
def test_to_dlpack_returns_valid_capsule():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cap = to_dlpack(t)
    assert type(cap).__name__ == "PyCapsule"

    pythonapi = ctypes.pythonapi
    PyCapsule_GetName = pythonapi.PyCapsule_GetName
    PyCapsule_GetName.restype = ctypes.c_char_p
    PyCapsule_GetName.argtypes = [ctypes.py_object]

    assert PyCapsule_GetName(cap) == b"dltensor"


def test_to_dlpack_direct_tensor_method():
    t = Tensor([10, 20, 30], dtype=np.int32)
    assert hasattr(t, "to_dlpack")
    cap = t.to_dlpack()
    assert type(cap).__name__ == "PyCapsule"


def test_to_dlpack_roundtrip_zero_copy():
    """Test exporting to_dlpack and importing with from_dlpack achieves zero-copy."""
    t = Tensor([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    cap = to_dlpack(t)

    t2 = from_dlpack(cap)
    assert isinstance(t2, Tensor)
    assert t2.shape == (2, 2)
    assert t2.dtype == np.float64
    assert t2.data.ctypes.data == t.data.ctypes.data

    # Mutation test
    t2.data[0, 1] = 999.0
    assert t.data[0, 1] == 999.0


# =========================================================================
# 3. from_dlpack Zero-Copy & GC Lifecycle Tests
# =========================================================================
def test_from_dlpack_numpy_zero_copy():
    """Ensure from_dlpack with NumPy array shares memory pointer and binds owner."""
    orig = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = from_dlpack(orig)

    assert isinstance(t, Tensor)
    assert t.shape == (3, 4)
    assert t.dtype == np.float32
    assert t.data.ctypes.data == orig.ctypes.data

    # Owner attribute must be set
    assert hasattr(t, "_dlpack_owner")
    assert type(t._dlpack_owner).__name__ == "PyCapsule"

    # Bidirectional mutation
    orig[1, 2] = 100.0
    assert t.data[1, 2] == 100.0

    t.data[0, 0] = 55.0
    assert orig[0, 0] == 55.0


def test_from_dlpack_sliced_non_contiguous():
    """Ensure non-contiguous strided arrays are correctly imported zero-copy."""
    orig = np.arange(30, dtype=np.int64).reshape(5, 6)
    sliced = orig[::2, 1::2]  # shape (3, 3)

    t = from_dlpack(sliced)
    assert t.shape == sliced.shape
    assert t.data.ctypes.data == sliced.ctypes.data
    assert np.allclose(t.data, sliced)

    t.data[0, 0] = 777
    assert orig[0, 1] == 777


def test_from_dlpack_dtypes():
    """Ensure various dtypes are properly supported across DLPack C-ABI."""
    dtypes = [np.float32, np.float64, np.int32, np.int64, np.uint8, np.int8, bool]
    for dt in dtypes:
        arr = np.array([0, 1, 0, 1], dtype=dt)
        t = from_dlpack(arr)
        assert t.dtype == np.dtype(dt)
        assert np.array_equal(t.data, arr)
        assert t.data.ctypes.data == arr.ctypes.data


def test_dlpack_lifecycle_cleanup():
    """Verify that unused capsules are cleaned up from _ACTIVE_MANAGED_TENSORS."""
    initial_count = len(_ACTIVE_MANAGED_TENSORS)

    t = Tensor([1.0, 2.0, 3.0])
    cap = to_dlpack(t)
    assert len(_ACTIVE_MANAGED_TENSORS) == initial_count + 1

    del cap
    gc.collect()
    assert len(_ACTIVE_MANAGED_TENSORS) == initial_count


# =========================================================================
# 4. python_bridge to_synapse_tensor Integration Tests
# =========================================================================
def test_to_synapse_tensor_dlpack_integration():
    """to_synapse_tensor should use DLPack zero-copy on objects with __dlpack__."""
    arr = np.array([[10, 20], [30, 40]], dtype=np.float32)
    t = to_synapse_tensor(arr)

    assert isinstance(t, Tensor)
    assert hasattr(t, "_dlpack_owner")
    assert t.data.ctypes.data == arr.ctypes.data

    t.data[1, 1] = 999
    assert arr[1, 1] == 999


def test_to_synapse_tensor_fallback():
    """If __dlpack__ raises an exception, to_synapse_tensor falls back gracefully."""
    class BrokenDLPackArray:
        def __init__(self, arr):
            self._arr = arr

        def __dlpack__(self):
            raise RuntimeError("DLPack export failed!")

        def numpy(self):
            return self._arr

    mock = BrokenDLPackArray(np.array([100, 200, 300]))
    t = to_synapse_tensor(mock)

    assert isinstance(t, Tensor)
    assert np.allclose(t.data, [100, 200, 300])


def test_no_torch_import_error():
    """Verify that synapse.interop.dlpack works completely without PyTorch installed."""
    import sys
    assert "torch" not in sys.modules
    import synapse.interop.dlpack
    assert hasattr(synapse.interop.dlpack, "from_dlpack")
    assert hasattr(synapse.interop.dlpack, "to_dlpack")
