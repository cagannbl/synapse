import ctypes
import numpy as np
import pytest

from synapse.core.tensor import Tensor, tensor
from synapse.interop.dlpack import to_dlpack, from_dlpack, is_dlpack_available
from synapse.interop.eco_bridge import TransparentPyResolver, PackageNotFoundError
from synapse.interop.python_bridge import PythonBridge, PythonModuleWrapper, SynapseInteropError, to_synapse, from_synapse
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError


def run_synapse_code(code: str) -> VirtualMachine:
    """Helper to compile and execute Synapse source code in the VM."""
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    compiled = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(compiled)
    return vm


# =========================================================================
# 1. Standard Python Modules Transparent Import Tests
# =========================================================================
def test_resolver_import_math_module():
    """Verify transparent import and invocation of standard math module."""
    math_mod = TransparentPyResolver.resolve_import("math")
    assert isinstance(math_mod, PythonModuleWrapper)

    # Function calls
    assert math_mod.sqrt(144.0) == 12.0
    assert math_mod.cos(0.0) == 1.0
    assert math_mod.sin(0.0) == 0.0
    assert math_mod.floor(3.99) == 3

    # Attributes / constants
    assert abs(math_mod.pi - 3.14159265) < 1e-5
    assert abs(math_mod.e - 2.71828182) < 1e-5


def test_resolver_import_json_module():
    """Verify transparent import and serialization with json module."""
    json_mod = TransparentPyResolver.resolve_import("json")
    assert isinstance(json_mod, PythonModuleWrapper)

    data = {"status": "ok", "count": 42, "items": ["a", "b"]}
    serialized = json_mod.dumps(data)
    assert isinstance(serialized, str)
    assert '"status": "ok"' in serialized

    deserialized = json_mod.loads(serialized)
    assert deserialized["count"] == 42
    assert deserialized["items"] == ["a", "b"]


def test_resolver_import_os_path_module():
    """Verify transparent import of dotted module (os.path)."""
    path_mod = TransparentPyResolver.resolve_import("os.path")
    assert isinstance(path_mod, PythonModuleWrapper)

    joined = path_mod.join("dir", "subdir", "file.txt")
    assert "dir" in joined and "file.txt" in joined

    basename = path_mod.basename(joined)
    assert basename == "file.txt"


def test_resolver_import_with_py_prefix():
    """Verify that 'py.' prefixed module names are automatically normalized."""
    m1 = TransparentPyResolver.resolve_import("py.math")
    assert m1.sqrt(25.0) == 5.0

    p1 = TransparentPyResolver.resolve_import("py.os.path")
    assert p1.basename("path/to/file.syn") == "file.syn"


def test_resolver_submodule_traversal():
    """Verify that accessing submodules from a root module wrapper works dynamically."""
    os_mod = TransparentPyResolver.resolve_import("os")
    assert hasattr(os_mod, "path")
    joined = os_mod.path.join("a", "b")
    assert "a" in joined and "b" in joined


def test_resolver_raw_import_without_wrap():
    """Verify wrap=False returns the underlying Python module directly."""
    import math
    raw_math = TransparentPyResolver.resolve_import("math", wrap=False)
    assert raw_math is math


def test_python_bridge_load_delegation():
    """Verify PythonBridge.load delegates directly to TransparentPyResolver."""
    m = PythonBridge.load("math")
    assert m.sqrt(49.0) == 7.0


# =========================================================================
# 2. Missing Module Guidance Error Message Tests
# =========================================================================
def test_missing_module_raises_actionable_error():
    """Verify missing package raises PackageNotFoundError with exact installation guide."""
    pkg_name = "non_existent_ml_package_xyz"
    with pytest.raises(PackageNotFoundError) as exc_info:
        TransparentPyResolver.resolve_import(pkg_name)

    err = exc_info.value
    expected_msg = (
        f"Package '{pkg_name}' is not installed in the environment. "
        f"Run 'pip install {pkg_name}' or 'synapse pkg install py:{pkg_name}' to use it in Synapse."
    )
    assert str(err) == expected_msg
    assert err.module_name == pkg_name

    # Inherits from both ModuleNotFoundError and SynapseInteropError
    assert isinstance(err, ModuleNotFoundError)
    assert isinstance(err, SynapseInteropError)
    assert isinstance(err, ImportError)


def test_missing_module_with_py_prefix_actionable_error():
    """Verify missing package with 'py.' prefix produces clean actionable error message."""
    with pytest.raises(PackageNotFoundError) as exc_info:
        TransparentPyResolver.resolve_import("py.deep_quantum_ai")

    err = exc_info.value
    expected_msg = (
        "Package 'deep_quantum_ai' is not installed in the environment. "
        "Run 'pip install deep_quantum_ai' or 'synapse pkg install py:deep_quantum_ai' to use it in Synapse."
    )
    assert str(err) == expected_msg
    assert err.module_name == "deep_quantum_ai"


def test_python_bridge_load_missing_module_error():
    """Verify PythonBridge.load also surfaces the rich PackageNotFoundError."""
    with pytest.raises(PackageNotFoundError) as exc_info:
        PythonBridge.load("super_custom_lib")

    assert "Run 'pip install super_custom_lib'" in str(exc_info.value)
    assert "synapse pkg install py:super_custom_lib" in str(exc_info.value)


def test_vm_runtime_error_enrichment_on_missing_import():
    """Verify that Synapse VM translates missing Python imports into actionable VMRuntimeError."""
    source = """
import py.totally_unknown_framework as tuf
let x = 10
"""
    with pytest.raises(VMRuntimeError) as exc_info:
        run_synapse_code(source)

    vm_err = str(exc_info.value)
    assert "Failed to import Python module 'totally_unknown_framework'" in vm_err
    assert "Run 'pip install totally_unknown_framework'" in vm_err
    assert "synapse pkg install py:totally_unknown_framework" in vm_err


# =========================================================================
# 3. DLPack Zero-Copy Tensor Interop Tests
# =========================================================================
def test_numpy_zero_copy_via_dlpack_resolver():
    """Verify NumPy array conversion achieves zero-copy memory sharing and capsule ownership."""
    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    t = TransparentPyResolver.from_dlpack(arr)

    assert isinstance(t, Tensor)
    assert t.shape == (2, 2)
    assert t.dtype == np.float32
    assert t.data.ctypes.data == arr.ctypes.data

    # DLPack owner capsule attached
    assert hasattr(t, "_dlpack_owner")
    assert type(t._dlpack_owner).__name__ == "PyCapsule"

    # Bidirectional zero-copy mutation test
    arr[0, 1] = 999.0
    assert t.data[0, 1] == 999.0

    t.data[1, 0] = 777.0
    assert arr[1, 0] == 777.0


def test_numpy_strided_zero_copy():
    """Verify non-contiguous (strided) NumPy arrays maintain zero-copy DLPack mapping."""
    base = np.arange(20, dtype=np.float64).reshape(4, 5)
    sliced = base[::2, ::2]  # shape (2, 3)

    t = TransparentPyResolver.from_dlpack(sliced)
    assert t.shape == sliced.shape
    assert t.data.ctypes.data == sliced.ctypes.data

    sliced[1, 2] = 4321.0
    assert t.data[1, 2] == 4321.0


class MockTorchTensor:
    """Mock representing a PyTorch torch.Tensor implementing standard DLPack protocol."""
    def __init__(self, arr: np.ndarray, is_cuda: bool = False):
        self._arr = arr
        self.is_cuda = is_cuda

    def __dlpack__(self, stream=None, max_version=None, dl_device=None):
        return self._arr.__dlpack__()

    def __dlpack_device__(self):
        return (1, 0)  # kDLCPU, id 0

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype


def test_mock_torch_tensor_zero_copy_dlpack():
    """Verify that PyTorch-like tensors with __dlpack__ convert zero-copy."""
    raw = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    mock_torch = MockTorchTensor(raw)

    t = TransparentPyResolver.from_dlpack(mock_torch)
    assert isinstance(t, Tensor)
    assert t.data.ctypes.data == raw.ctypes.data
    assert hasattr(t, "_dlpack_owner")

    # Bidirectional mutation test
    raw[0] = 100.5
    assert t.data[0] == 100.5

    t.data[3] = 400.5
    assert raw[3] == 400.5


def test_to_synapse_auto_detects_torch_and_numpy_tensors():
    """Verify to_synapse() transparently routes PyTorch/NumPy tensors through DLPack zero-copy."""
    raw = np.array([[1, 2], [3, 4]], dtype=np.int64)
    mock_torch = MockTorchTensor(raw)

    syn_t = to_synapse(mock_torch)
    assert isinstance(syn_t, Tensor)
    assert syn_t.data.ctypes.data == raw.ctypes.data

    raw[0, 0] = 9999
    assert syn_t.data[0, 0] == 9999


def test_module_wrapper_auto_converts_tensors():
    """Verify function calls through PythonModuleWrapper automatically convert return tensors."""
    np_mod = TransparentPyResolver.resolve_import("numpy")

    # Calling np.zeros via the wrapper returns a Synapse Tensor
    zeros_t = np_mod.zeros([3, 3])
    assert isinstance(zeros_t, Tensor)
    assert zeros_t.shape == (3, 3)

    # Calling np.add with Synapse Tensors
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([10.0, 20.0, 30.0])
    added = np_mod.add(a, b)
    assert isinstance(added, Tensor)
    assert np.allclose(added.data, [11.0, 22.0, 33.0])


def test_roundtrip_to_and_from_dlpack():
    """Verify roundtrip export and import: Tensor -> to_dlpack -> from_dlpack."""
    original = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    capsule = TransparentPyResolver.to_dlpack(original)
    assert type(capsule).__name__ == "PyCapsule"

    imported = TransparentPyResolver.from_dlpack(capsule)
    assert isinstance(imported, Tensor)
    assert imported.data.ctypes.data == original.data.ctypes.data

    # Zero-copy mutation verification
    imported.data[0, 1] = 12345.0
    assert original.data[0, 1] == 12345.0


def test_vm_end_to_end_python_tensor_script():
    """Verify end-to-end VM execution with transparent python numpy import and tensor operations."""
    source = """
import py.numpy as np
let t = tensor([4.0, 9.0, 16.0])
let root_t = np.sqrt(t)
let sum_val = np.sum(root_t)
"""
    vm = run_synapse_code(source)
    root_t = vm.globals["root_t"]
    assert isinstance(root_t, Tensor)
    assert np.allclose(root_t.data, [2.0, 3.0, 4.0])
    assert vm.globals["sum_val"] == 9.0
