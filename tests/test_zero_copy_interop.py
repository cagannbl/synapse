"""Tests for Synapse Zero-Copy DLPack and Python Bridge Interoperability.

Verifies:
1. Synapse Tensor -> DLPack -> NumPy zero-copy memory sharing and bidirectional mutation.
2. NumPy ndarray -> DLPack -> Synapse Tensor zero-copy sharing and bidirectional mutation.
3. DLManagedTensor & DLManagedTensorVersioned lifecycle, deleter callbacks, and memory safety.
4. python_bridge 'import py.numpy as np' and 'from python import numpy as np' transparent import experience.
5. End-to-end Synapse VM execution with 'from python import numpy as np'.
6. Mock PyTorch Tensor zero-copy interop via __dlpack__.
"""

from __future__ import annotations

import ctypes
import gc
import weakref
import numpy as np
import pytest

from synapse.core.tensor import Tensor, tensor
from synapse.interop.dlpack import (
    DLDevice,
    DLDataType,
    DLTensor,
    DLPackVersion,
    DLManagedTensor,
    DLManagedTensorVersioned,
    DLManagedTensorDeleter,
    DLManagedTensorVersionedDeleter,
    DLDeviceType,
    DLDataTypeCode,
    DLPACK_FLAG_BITMASK_READ_ONLY,
    DLPACK_FLAG_BITMASK_IS_COPIED,
    is_dlpack_available,
    from_dlpack,
    to_dlpack,
    _ACTIVE_MANAGED_TENSORS,
    _PyCapsule_New,
    _PyCapsule_GetPointer,
    _PyCapsule_GetName,
    PyCapsule_Destructor,
)
from synapse.interop.python_bridge import (
    to_synapse_tensor,
    to_numpy,
    to_synapse,
    from_synapse,
    PythonModuleWrapper,
    install_import_hooks,
)
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_synapse_code(source: str) -> VirtualMachine:
    """Helper to compile and execute Synapse source code in the VM."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    bytecode = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(bytecode)
    return vm


# =========================================================================
# 1. Synapse Tensor -> DLPack -> NumPy Zero-Copy Tests
# =========================================================================
class TestSynapseToNumPyZeroCopy:
    def test_1d_tensor_to_numpy_zero_copy(self):
        """Synapse Tensor exported to NumPy shares exact memory and allows in-place mutation."""
        t = Tensor([10.0, 20.0, 30.0], dtype=np.float32)
        arr = np.from_dlpack(t)

        assert isinstance(arr, np.ndarray)
        assert arr.flags.writeable is True
        assert arr.ctypes.data == t.data.ctypes.data
        assert arr.shape == (3,)
        assert arr.dtype == np.float32

        # Mutation on NumPy reflects in Synapse Tensor
        arr[0] = 999.0
        assert t.data[0] == 999.0

        # Mutation on Synapse Tensor reflects in NumPy array
        t.data[1] = 777.0
        assert arr[1] == 777.0

    def test_2d_tensor_to_numpy_zero_copy(self):
        """2D Tensor exported to NumPy retains matrix layout and shares memory pointer."""
        raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        t = Tensor(raw)
        arr = np.from_dlpack(t)

        assert arr.shape == (2, 3)
        assert arr.ctypes.data == t.data.ctypes.data
        assert arr.flags.writeable is True

        arr[1, 2] = 42.0
        assert t.data[1, 2] == 42.0

        t.data[0, 0] = -99.0
        assert arr[0, 0] == -99.0

    def test_various_dtypes_to_numpy(self):
        """Ensures int32, int64, float32, float64, uint8, and bool export cleanly."""
        dtypes = [np.int32, np.int64, np.float32, np.float64, np.uint8, bool]
        for dt in dtypes:
            raw = np.array([0, 1, 1, 0], dtype=dt)
            t = Tensor(raw)
            arr = np.from_dlpack(t)

            assert arr.dtype == np.dtype(dt)
            assert arr.ctypes.data == t.data.ctypes.data
            assert np.array_equal(arr, t.data)


# =========================================================================
# 2. NumPy -> DLPack -> Synapse Tensor Zero-Copy Tests
# =========================================================================
class TestNumPyToSynapseZeroCopy:
    def test_1d_numpy_to_synapse_zero_copy(self):
        """NumPy array imported into Synapse shares exact memory pointer."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        t = from_dlpack(arr)

        assert isinstance(t, Tensor)
        assert t.data.ctypes.data == arr.ctypes.data
        assert t.shape == (3,)
        assert t.dtype == np.float64

        # Mutation in Synapse reflects in NumPy
        t.data[0] = 12345.0
        assert arr[0] == 12345.0

        # Mutation in NumPy reflects in Synapse
        arr[2] = 98765.0
        assert t.data[2] == 98765.0

    def test_non_contiguous_sliced_numpy_to_synapse(self):
        """Sliced strided NumPy array imported into Synapse shares memory correctly."""
        big = np.arange(30, dtype=np.int64).reshape(5, 6)
        sliced = big[::2, 1::2]  # shape (3, 3)

        t = from_dlpack(sliced)
        assert t.shape == (3, 3)
        assert t.data.ctypes.data == sliced.ctypes.data
        assert np.array_equal(t.data, sliced)

        # Mutate through Synapse
        t.data[0, 0] = 8888
        assert sliced[0, 0] == 8888
        assert big[0, 1] == 8888

    def test_to_synapse_tensor_dlpack_priority(self):
        """to_synapse_tensor uses DLPack zero-copy for numpy arrays."""
        arr = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        t = to_synapse_tensor(arr)

        assert isinstance(t, Tensor)
        assert t.data.ctypes.data == arr.ctypes.data

        t.data[0] = -1.0
        assert arr[0] == -1.0


# =========================================================================
# 3. DLManagedTensor Lifecycle, Memory Management & Deleter Tests
# =========================================================================
class TestDLPackLifecycleAndDeleter:
    def test_active_tensor_registered_and_freed_by_numpy_deleter(self):
        """Verify that exporting to NumPy registers the tensor, and NumPy frees it on deletion."""
        initial_count = len(_ACTIVE_MANAGED_TENSORS)

        t = Tensor([1.0, 2.0, 3.0], dtype=np.float32)
        arr = np.from_dlpack(t)
        assert len(_ACTIVE_MANAGED_TENSORS) == initial_count + 1

        # Delete consumer array and collect garbage
        del arr
        gc.collect()
        assert len(_ACTIVE_MANAGED_TENSORS) == initial_count

    def test_capsule_destructor_called_if_not_consumed(self):
        """Verify that an unconsumed capsule is cleaned up by its PyCapsule destructor."""
        initial_count = len(_ACTIVE_MANAGED_TENSORS)

        t = Tensor([4.0, 5.0, 6.0], dtype=np.float32)
        cap = to_dlpack(t, max_version=(1, 0))
        assert len(_ACTIVE_MANAGED_TENSORS) == initial_count + 1

        del cap
        gc.collect()
        assert len(_ACTIVE_MANAGED_TENSORS) == initial_count

    def test_double_consumption_prevented(self):
        """A capsule consumed once cannot be consumed again ('used_dltensor')."""
        t = Tensor([1.0, 2.0])
        cap = to_dlpack(t)

        t1 = from_dlpack(cap)
        assert isinstance(t1, Tensor)

        with pytest.raises(ValueError, match="already been consumed"):
            from_dlpack(cap)

    def test_producer_deleter_invoked_on_synapse_tensor_gc(self):
        """When Synapse imports a foreign DLManagedTensor, it invokes deleter upon GC."""
        deleter_called = []

        def custom_deleter(ptr):
            deleter_called.append(True)

        c_deleter = DLManagedTensorVersionedDeleter(custom_deleter)
        raw_arr = np.array([10.0, 20.0], dtype=np.float32)

        dlmv = DLManagedTensorVersioned()
        dlmv.version.major = 1
        dlmv.version.minor = 0
        dlmv.manager_ctx = None
        dlmv.deleter = c_deleter
        dlmv.flags = 0

        shape_type = ctypes.c_int64 * 1
        shape_arr = shape_type(2)
        strides_arr = shape_type(1)

        dlmv.dl_tensor.data = ctypes.c_void_p(raw_arr.ctypes.data)
        dlmv.dl_tensor.device = DLDevice(DLDeviceType.kDLCPU, 0)
        dlmv.dl_tensor.ndim = 1
        dlmv.dl_tensor.dtype = DLDataType(DLDataTypeCode.kDLFloat, 32, 1)
        dlmv.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
        dlmv.dl_tensor.strides = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int64))
        dlmv.dl_tensor.byte_offset = 0

        # Pin C objects to prevent GC
        pin = (dlmv, shape_arr, strides_arr, c_deleter, raw_arr)

        cap = _PyCapsule_New(ctypes.byref(dlmv), b"dltensor_versioned", PyCapsule_Destructor(0))

        syn_t = from_dlpack(cap)
        assert isinstance(syn_t, Tensor)
        assert len(deleter_called) == 0

        del syn_t
        gc.collect()
        assert len(deleter_called) == 1
        del pin


# =========================================================================
# 4. python_bridge Transparent Import Experience Tests
# =========================================================================
class TestPythonBridgeImportExperience:
    def test_import_py_numpy_in_python(self):
        """'import py.numpy as np' provides a wrapped numpy module returning Synapse Tensors."""
        install_import_hooks()
        import py.numpy as np

        assert isinstance(np, PythonModuleWrapper)

        # Calling np.zeros returns Synapse Tensor
        zeros = np.zeros([2, 3])
        assert isinstance(zeros, Tensor)
        assert zeros.shape == (2, 3)

        # Calling np.add with Synapse Tensors
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([10.0, 20.0, 30.0])
        c = np.add(a, b)
        assert isinstance(c, Tensor)
        assert np.allclose(c.data, [11.0, 22.0, 33.0])

    def test_from_python_import_numpy(self):
        """'from python import numpy as np_from_py' works seamlessly."""
        install_import_hooks()
        from python import numpy as np_from_py

        assert isinstance(np_from_py, PythonModuleWrapper)
        ones = np_from_py.ones([3, 2])
        assert isinstance(ones, Tensor)
        assert ones.shape == (3, 2)
        assert np.allclose(ones.data, 1.0)

    def test_submodule_import(self):
        """'import py.numpy.linalg as la' correctly resolves submodules."""
        install_import_hooks()
        import py.numpy as np
        import py.numpy.linalg as la

        assert isinstance(la, PythonModuleWrapper)
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        det = la.det(mat)
        assert abs(det - (-2.0)) < 1e-4


# =========================================================================
# 5. Mock PyTorch Tensor Zero-Copy Interop Tests
# =========================================================================
class MockPyTorchTensor:
    """Simulates a PyTorch Tensor implementing __dlpack__ and __dlpack_device__."""
    def __init__(self, arr: np.ndarray, is_cuda: bool = False):
        self._arr = arr
        self.is_cuda = is_cuda

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        from synapse.interop.dlpack import to_dlpack
        return to_dlpack(self._arr, stream=stream, max_version=max_version, dl_device=dl_device, copy=copy)

    def __dlpack_device__(self):
        return (DLDeviceType.kDLCPU, 0)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class TestMockPyTorchZeroCopy:
    def test_pytorch_mock_to_synapse_zero_copy(self):
        """PyTorch-like tensor transfers zero-copy to Synapse Tensor via __dlpack__."""
        raw = np.array([5.0, 15.0, 25.0], dtype=np.float32)
        mock = MockPyTorchTensor(raw)

        syn_t = to_synapse_tensor(mock)
        assert isinstance(syn_t, Tensor)
        assert syn_t.data.ctypes.data == raw.ctypes.data

        syn_t.data[0] = 777.0
        assert raw[0] == 777.0

        raw[1] = 888.0
        assert syn_t.data[1] == 888.0


# =========================================================================
# 6. End-to-End Synapse VM Import and Execution Tests
# =========================================================================
class TestSynapseVMInteropScripts:
    def test_vm_from_python_import_numpy(self):
        """Synapse script: 'from python import numpy as np'."""
        source = """
from python import numpy as np
let t = tensor([4.0, 9.0, 16.0])
let root_t = np.sqrt(t)
let sum_val = np.sum(root_t)
"""
        vm = run_synapse_code(source)
        root_t = vm.globals["root_t"]
        assert isinstance(root_t, Tensor)
        assert np.allclose(root_t.data, [2.0, 3.0, 4.0])
        assert vm.globals["sum_val"] == 9.0

    def test_vm_from_py_import_numpy(self):
        """Synapse script: 'from py import numpy as np'."""
        source = """
from py import numpy as np
let z = np.zeros([2, 3])
"""
        vm = run_synapse_code(source)
        z = vm.globals["z"]
        assert isinstance(z, Tensor)
        assert z.shape == (2, 3)

    def test_vm_import_py_numpy_zero_copy_in_place(self):
        """Synapse script: 'import py.numpy as np' zero-copy interaction."""
        source = """
import py.numpy as np
let a = tensor([1.0, 2.0, 3.0])
let b = tensor([10.0, 20.0, 30.0])
let c = np.add(a, b)
"""
        vm = run_synapse_code(source)
        c = vm.globals["c"]
        assert isinstance(c, Tensor)
        assert np.allclose(c.data, [11.0, 22.0, 33.0])
