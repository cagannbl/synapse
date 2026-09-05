"""Synapse DLPack & High-Performance Python Interop Bridge.

Provides standard C-ABI structures (DLDevice, DLDataType, DLTensor, DLManagedTensor)
and zero-copy tensor sharing between Synapse and Python array frameworks
(PyTorch, CuPy, NumPy, JAX, etc.).
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from synapse.core.tensor import Tensor

logger = logging.getLogger("synapse.interop.dlpack")

# =========================================================================
# DLPack C-ABI Enumerations & Constants (DLPack v0.2 - v0.8+)
# =========================================================================
class DLDeviceType:
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCm = 10
    kDLROCmHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    kDLMAIA = 17


class DLDataTypeCode:
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


# =========================================================================
# DLPack C-ABI Structures (ctypes.Structure)
# =========================================================================
class DLDevice(ctypes.Structure):
    """Standard DLPack device descriptor: {device_type, device_id}."""
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]

    def __repr__(self) -> str:
        return f"DLDevice(device_type={self.device_type}, device_id={self.device_id})"


class DLDataType(ctypes.Structure):
    """Standard DLPack data type descriptor: {code, bits, lanes}."""
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

    def __repr__(self) -> str:
        return f"DLDataType(code={self.code}, bits={self.bits}, lanes={self.lanes})"


class DLPackVersion(ctypes.Structure):
    """DLPack ABI version descriptor: {major, minor}."""
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]

    def __repr__(self) -> str:
        return f"DLPackVersion(major={self.major}, minor={self.minor})"


class DLTensor(ctypes.Structure):
    """Plain C DLTensor struct representing multidimensional array buffer."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    """C DLManagedTensor struct managing tensor lifecycle with custom deleter (DLPack <= 0.8)."""
    pass


DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorDeleter),
]


class DLManagedTensorVersioned(ctypes.Structure):
    """C DLManagedTensorVersioned struct managing tensor lifecycle with ABI versioning (DLPack 1.0+, PEP 652)."""
    pass


DLManagedTensorVersionedDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))

DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorVersionedDeleter),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", DLTensor),
]

# DLPack Bitmask Flags
DLPACK_FLAG_BITMASK_READ_ONLY = 1 << 0
DLPACK_FLAG_BITMASK_IS_COPIED = 1 << 1


# =========================================================================
# PyCapsule C API Bindings
# =========================================================================
_pythonapi = ctypes.pythonapi

PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

# C function pointer versions for use inside C callbacks
_PyCapsule_GetPointer_C = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", _pythonapi)
)
_PyCapsule_GetName_C = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)(
    ("PyCapsule_GetName", _pythonapi)
)

# Python object calling versions
_PyCapsule_New = _pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor]

_PyCapsule_GetPointer = _pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_PyCapsule_GetName = _pythonapi.PyCapsule_GetName
_PyCapsule_GetName.restype = ctypes.c_char_p
_PyCapsule_GetName.argtypes = [ctypes.py_object]

_PyCapsule_IsValid = _pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

_PyCapsule_SetName = _pythonapi.PyCapsule_SetName
_PyCapsule_SetName.restype = ctypes.c_int
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]


# =========================================================================
# DLManagedTensor Lifecycle & Memory Management
# =========================================================================
# Keeps memory for DLManagedTensor/DLManagedTensorVersioned, shape arrays, strides arrays,
# and owner alive until the consumer calls deleter(dlm) or the capsule is garbage collected.
import threading
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_MANAGED_TENSORS: Dict[int, tuple] = {}


def _global_dlpack_deleter(dlm_ptr):
    """C deleter callback invoked by consumer when legacy DLManagedTensor is freed."""
    if not dlm_ptr:
        return
    ptr_val = ctypes.cast(dlm_ptr, ctypes.c_void_p).value
    with _ACTIVE_LOCK:
        _ACTIVE_MANAGED_TENSORS.pop(ptr_val, None)


_C_GLOBAL_DELETER = DLManagedTensorDeleter(_global_dlpack_deleter)


def _global_dlpack_versioned_deleter(dlmv_ptr):
    """C deleter callback invoked by consumer when DLManagedTensorVersioned is freed."""
    if not dlmv_ptr:
        return
    ptr_val = ctypes.cast(dlmv_ptr, ctypes.c_void_p).value
    with _ACTIVE_LOCK:
        _ACTIVE_MANAGED_TENSORS.pop(ptr_val, None)


_C_GLOBAL_VERSIONED_DELETER = DLManagedTensorVersionedDeleter(_global_dlpack_versioned_deleter)


def _global_capsule_destructor(cap_ptr):
    """PyCapsule destructor invoked if legacy capsule is GC'd without being consumed."""
    if not cap_ptr:
        return
    name = _PyCapsule_GetName_C(cap_ptr)
    if name == b"dltensor":
        raw_ptr = _PyCapsule_GetPointer_C(cap_ptr, b"dltensor")
        if raw_ptr:
            dlm = ctypes.cast(raw_ptr, ctypes.POINTER(DLManagedTensor))
            if dlm and dlm.contents.deleter:
                dlm.contents.deleter(dlm)


_C_GLOBAL_CAPSULE_DESTRUCTOR = PyCapsule_Destructor(_global_capsule_destructor)


def _global_versioned_capsule_destructor(cap_ptr):
    """PyCapsule destructor invoked if versioned capsule is GC'd without being consumed."""
    if not cap_ptr:
        return
    name = _PyCapsule_GetName_C(cap_ptr)
    if name == b"dltensor_versioned":
        raw_ptr = _PyCapsule_GetPointer_C(cap_ptr, b"dltensor_versioned")
        if raw_ptr:
            dlmv = ctypes.cast(raw_ptr, ctypes.POINTER(DLManagedTensorVersioned))
            if dlmv and dlmv.contents.deleter:
                dlmv.contents.deleter(dlmv)


_C_GLOBAL_VERSIONED_CAPSULE_DESTRUCTOR = PyCapsule_Destructor(_global_versioned_capsule_destructor)


# =========================================================================
# DType Mappings
# =========================================================================
def _dlpack_to_numpy_dtype(dl_dtype: DLDataType) -> np.dtype:
    """Converts a DLDataType struct to a corresponding numpy.dtype."""
    code = dl_dtype.code
    bits = dl_dtype.bits
    lanes = dl_dtype.lanes

    if lanes != 1:
        raise ValueError(f"Unsupported DLPack lanes: {lanes} (only lanes=1 supported)")

    if code == DLDataTypeCode.kDLInt:
        if bits == 8:
            return np.dtype(np.int8)
        if bits == 16:
            return np.dtype(np.int16)
        if bits == 32:
            return np.dtype(np.int32)
        if bits == 64:
            return np.dtype(np.int64)
    elif code == DLDataTypeCode.kDLUInt:
        if bits == 8:
            return np.dtype(np.uint8)
        if bits == 16:
            return np.dtype(np.uint16)
        if bits == 32:
            return np.dtype(np.uint32)
        if bits == 64:
            return np.dtype(np.uint64)
    elif code == DLDataTypeCode.kDLFloat:
        if bits == 16:
            return np.dtype(np.float16)
        if bits == 32:
            return np.dtype(np.float32)
        if bits == 64:
            return np.dtype(np.float64)
    elif code == DLDataTypeCode.kDLBfloat:
        try:
            return np.dtype("bfloat16")
        except TypeError:
            return np.dtype(np.uint16)
    elif code == DLDataTypeCode.kDLComplex:
        if bits == 64:
            return np.dtype(np.complex64)
        if bits == 128:
            return np.dtype(np.complex128)
    elif code == DLDataTypeCode.kDLBool:
        if bits == 8:
            return np.dtype(bool)

    raise ValueError(f"Unsupported DLPack data type: code={code}, bits={bits}")


def _numpy_to_dlpack_dtype(dtype: Union[np.dtype, str, type]) -> DLDataType:
    """Converts a numpy.dtype to a DLDataType struct."""
    dt = np.dtype(dtype)
    kind = dt.kind
    itemsize = dt.itemsize
    bits = itemsize * 8

    if kind == "i":
        return DLDataType(DLDataTypeCode.kDLInt, bits, 1)
    if kind == "u":
        return DLDataType(DLDataTypeCode.kDLUInt, bits, 1)
    if kind == "f":
        return DLDataType(DLDataTypeCode.kDLFloat, bits, 1)
    if kind == "c":
        return DLDataType(DLDataTypeCode.kDLComplex, bits, 1)
    if kind == "b":
        return DLDataType(DLDataTypeCode.kDLBool, 8, 1)

    raise ValueError(f"NumPy dtype '{dtype}' is not supported by DLPack standard")


# =========================================================================
# Public Interop Functions
# =========================================================================
def is_dlpack_available() -> bool:
    """Checks whether the DLPack interop runtime and C-API bindings are available."""
    try:
        return (
            hasattr(ctypes, "pythonapi")
            and hasattr(ctypes.pythonapi, "PyCapsule_New")
            and hasattr(ctypes.pythonapi, "PyCapsule_GetPointer")
            and hasattr(ctypes.pythonapi, "PyCapsule_GetName")
        )
    except Exception:
        return False


def from_dlpack(obj: Any, requires_grad: bool = False) -> Tensor:
    """Constructs a Synapse Tensor from a PyTorch/CuPy/NumPy DLPack object or PyCapsule.

    Supports both modern versioned capsules ('dltensor_versioned' / DLManagedTensorVersioned, DLPack 1.0+)
    and legacy capsules ('dltensor' / DLManagedTensor).
    Achieves true zero-copy data sharing by mapping ctypes memory pointers to NumPy ndarray,
    renames consumed capsule to prevent premature GC destruction, and binds a safe finalizer callback
    to invoke the producer's deleter when the Synapse Tensor is destroyed.
    """
    if isinstance(obj, Tensor):
        return obj

    capsule = None
    source_obj = None

    if type(obj).__name__ == "PyCapsule":
        capsule = obj
    elif hasattr(obj, "__dlpack__"):
        source_obj = obj
        # If CUDA PyTorch tensor on host without CUDA interop, transfer to CPU
        if getattr(obj, "is_cuda", False) and hasattr(obj, "cpu"):
            source_obj = obj.cpu()
        try:
            capsule = source_obj.__dlpack__(max_version=(1, 0))
        except (TypeError, ValueError, AttributeError):
            capsule = source_obj.__dlpack__()
    elif hasattr(obj, "to_dlpack") and callable(obj.to_dlpack):
        source_obj = obj
        capsule = obj.to_dlpack()
    else:
        raise TypeError(
            f"Object of type {type(obj).__name__} does not implement DLPack protocol (__dlpack__) "
            "and is not a 'dltensor' or 'dltensor_versioned' PyCapsule."
        )

    # Validate capsule name
    cap_name = _PyCapsule_GetName(capsule)
    if cap_name in (b"used_dltensor", b"used_dltensor_versioned"):
        raise ValueError("PyCapsule has already been consumed ('used_dltensor').")

    deleter_fn = None
    deleter_arg = None
    is_read_only = False

    if cap_name == b"dltensor_versioned":
        raw_ptr = _PyCapsule_GetPointer(capsule, b"dltensor_versioned")
        if not raw_ptr:
            raise ValueError("PyCapsule does not contain a valid pointer or has been corrupted.")
        dlmv_ptr = ctypes.cast(raw_ptr, ctypes.POINTER(DLManagedTensorVersioned))
        dlmv = dlmv_ptr.contents
        dlt = dlmv.dl_tensor
        deleter_fn = dlmv.deleter
        deleter_arg = dlmv_ptr
        is_read_only = bool(dlmv.flags & DLPACK_FLAG_BITMASK_READ_ONLY)
        # Rename capsule to signify consumption per DLPack spec
        _PyCapsule_SetName(capsule, b"used_dltensor_versioned")
    elif cap_name == b"dltensor":
        raw_ptr = _PyCapsule_GetPointer(capsule, b"dltensor")
        if not raw_ptr:
            raise ValueError("PyCapsule does not contain a valid pointer or has been corrupted.")
        dlm_ptr = ctypes.cast(raw_ptr, ctypes.POINTER(DLManagedTensor))
        dlm = dlm_ptr.contents
        dlt = dlm.dl_tensor
        deleter_fn = dlm.deleter
        deleter_arg = dlm_ptr
        # Rename capsule to signify consumption per DLPack spec
        _PyCapsule_SetName(capsule, b"used_dltensor")
    else:
        raise ValueError(f"Expected PyCapsule named 'dltensor' or 'dltensor_versioned', got {cap_name!r}")

    # Device verification
    dev_type = dlt.device.device_type
    dev_id = dlt.device.device_id
    if dev_type == DLDeviceType.kDLCUDA:
        raise ValueError(
            f"DLTensor is on CUDA device ({dev_id}). Synapse zero-copy CPU interop requires host memory. "
            "Please call .cpu() on the source tensor before passing to from_dlpack."
        )
    elif dev_type not in (DLDeviceType.kDLCPU, DLDeviceType.kDLCUDAHost, DLDeviceType.kDLCUDAManaged):
        raise ValueError(f"Unsupported DLPack device type: {dev_type}")

    # DataType and element size
    np_dtype = _dlpack_to_numpy_dtype(dlt.dtype)
    itemsize = np_dtype.itemsize

    # Shape and strides
    ndim = int(dlt.ndim)
    if ndim < 0:
        raise ValueError(f"Invalid negative ndim: {ndim}")

    shape = tuple(int(dlt.shape[i]) for i in range(ndim))

    if bool(dlt.strides):
        byte_strides = tuple(int(dlt.strides[i]) * itemsize for i in range(ndim))
    else:
        byte_strides = None

    data_address = (dlt.data or 0) + (dlt.byte_offset or 0)

    # Compute array size
    size = 1
    for s in shape:
        size *= s

    if size == 0:
        arr = np.empty(shape, dtype=np_dtype)
    else:
        if byte_strides is not None:
            max_offset = 0
            for dim, s in zip(shape, byte_strides):
                if dim > 0:
                    max_offset += (dim - 1) * abs(s)
            total_bytes = max(max_offset + itemsize, itemsize)
            c_buf = (ctypes.c_char * total_bytes).from_address(data_address)
            arr = np.ndarray(shape=shape, dtype=np_dtype, buffer=c_buf, strides=byte_strides)
        else:
            total_bytes = max(size * itemsize, itemsize)
            c_buf = (ctypes.c_char * total_bytes).from_address(data_address)
            arr = np.ndarray(shape=shape, dtype=np_dtype, buffer=c_buf)

    if is_read_only:
        arr.flags.writeable = False

    # Instantiate Synapse Tensor
    synapse_tensor = Tensor(arr, requires_grad=requires_grad, device="cpu", dtype=np_dtype)
    # Guarantee zero-copy direct array reference
    synapse_tensor.data = arr

    # Store capsule reference on tensor to prevent GC-induced dangling pointer
    synapse_tensor._dlpack_owner = capsule
    if source_obj is not None and source_obj is not capsule:
        synapse_tensor._dlpack_source = source_obj

    # Register safe deleter callback when Synapse Tensor is garbage collected
    if deleter_fn:
        import weakref

        def _safe_invoke_deleter(fn, arg):
            try:
                fn(arg)
            except Exception:
                pass

        synapse_tensor._dlpack_finalizer = weakref.finalize(
            synapse_tensor, _safe_invoke_deleter, deleter_fn, deleter_arg
        )

    return synapse_tensor


def to_dlpack(
    tensor: Any,
    stream: Any = None,
    max_version: Optional[Tuple[int, int]] = None,
    dl_device: Optional[Tuple[int, int]] = None,
    copy: Optional[bool] = None,
) -> Any:
    """Exports a Synapse Tensor (or compatible array) as a DLPack PyCapsule.

    When max_version is >= (1, 0), wraps a DLManagedTensorVersioned struct in a 'dltensor_versioned'
    capsule, enabling true writeable zero-copy data exchange with NumPy 2.x+, PyTorch, and JAX.
    Otherwise, wraps a DLManagedTensor in a legacy 'dltensor' capsule for backward compatibility.
    """
    if isinstance(tensor, Tensor):
        arr = tensor.data
        dev_type_str = tensor.device.device_type
        dev_id = getattr(tensor.device, "index", 0) or 0
    elif isinstance(tensor, np.ndarray):
        arr = tensor
        dev_type_str = "cpu"
        dev_id = 0
    else:
        # Fallback convert to Tensor
        from synapse.interop.python_bridge import to_synapse_tensor
        tensor = to_synapse_tensor(tensor)
        arr = tensor.data
        dev_type_str = tensor.device.device_type
        dev_id = getattr(tensor.device, "index", 0) or 0

    if copy is True:
        arr = arr.copy()

    if dev_type_str == "cuda":
        device = DLDevice(DLDeviceType.kDLCUDA, dev_id)
    else:
        device = DLDevice(DLDeviceType.kDLCPU, 0)

    if dl_device is not None:
        expected_type, expected_id = dl_device
        if device.device_type != expected_type or device.device_id != expected_id:
            raise ValueError(
                f"Device mismatch: requested {dl_device}, but tensor is on {(device.device_type, device.device_id)}"
            )

    dtype = _numpy_to_dlpack_dtype(arr.dtype)
    ndim = arr.ndim

    if ndim > 0:
        shape_type = ctypes.c_int64 * ndim
        shape_arr = shape_type(*arr.shape)
        shape_ptr = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))

        strides_arr = shape_type(*(s // arr.itemsize for s in arr.strides))
        strides_ptr = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int64))
    else:
        shape_arr = None
        shape_ptr = None
        strides_arr = None
        strides_ptr = None

    data_ptr = ctypes.c_void_p(arr.ctypes.data)

    # Versioned DLPack 1.0+ (PEP 652) requested
    if max_version is not None and max_version >= (1, 0):
        dlmv = DLManagedTensorVersioned()
        dlmv.version.major = 1
        dlmv.version.minor = 0
        dlmv.manager_ctx = None
        dlmv.deleter = _C_GLOBAL_VERSIONED_DELETER
        dlmv.flags = 0 if arr.flags.writeable else DLPACK_FLAG_BITMASK_READ_ONLY
        dlmv.dl_tensor.data = data_ptr
        dlmv.dl_tensor.device = device
        dlmv.dl_tensor.ndim = ndim
        dlmv.dl_tensor.dtype = dtype
        dlmv.dl_tensor.shape = shape_ptr
        dlmv.dl_tensor.strides = strides_ptr
        dlmv.dl_tensor.byte_offset = 0

        ptr_val = ctypes.cast(ctypes.byref(dlmv), ctypes.c_void_p).value
        with _ACTIVE_LOCK:
            _ACTIVE_MANAGED_TENSORS[ptr_val] = (
                dlmv,
                shape_arr,
                strides_arr,
                tensor,
                arr,
                _C_GLOBAL_VERSIONED_DELETER,
            )

        capsule = _PyCapsule_New(ctypes.byref(dlmv), b"dltensor_versioned", _C_GLOBAL_VERSIONED_CAPSULE_DESTRUCTOR)
        return capsule

    # Legacy DLPack <= 0.8 unversioned capsule
    dlm = DLManagedTensor()
    dlm.dl_tensor.data = data_ptr
    dlm.dl_tensor.device = device
    dlm.dl_tensor.ndim = ndim
    dlm.dl_tensor.dtype = dtype
    dlm.dl_tensor.shape = shape_ptr
    dlm.dl_tensor.strides = strides_ptr
    dlm.dl_tensor.byte_offset = 0
    dlm.manager_ctx = None
    dlm.deleter = _C_GLOBAL_DELETER

    # Pin memory in active registry to prevent GC while held by consumer
    ptr_val = ctypes.cast(ctypes.byref(dlm), ctypes.c_void_p).value
    with _ACTIVE_LOCK:
        _ACTIVE_MANAGED_TENSORS[ptr_val] = (dlm, shape_arr, strides_arr, tensor, arr, _C_GLOBAL_DELETER)

    capsule = _PyCapsule_New(ctypes.byref(dlm), b"dltensor", _C_GLOBAL_CAPSULE_DESTRUCTOR)
    return capsule


# Bind standard Array API DLPack protocol to Synapse Tensor
def _tensor_dlpack(self, stream=None, max_version=None, dl_device=None, copy=None):
    return to_dlpack(self, stream=stream, max_version=max_version, dl_device=dl_device, copy=copy)


def _tensor_dlpack_device(self):
    dev_type = DLDeviceType.kDLCUDA if self.device.device_type == "cuda" else DLDeviceType.kDLCPU
    dev_id = getattr(self.device, "index", 0) or 0
    return (dev_type, dev_id)


if not hasattr(Tensor, "__dlpack__"):
    Tensor.__dlpack__ = _tensor_dlpack
if not hasattr(Tensor, "__dlpack_device__"):
    Tensor.__dlpack_device__ = _tensor_dlpack_device
if not hasattr(Tensor, "to_dlpack"):
    Tensor.to_dlpack = lambda self, *args, **kwargs: to_dlpack(self, *args, **kwargs)
if not hasattr(Tensor, "from_dlpack"):
    Tensor.from_dlpack = staticmethod(from_dlpack)


__all__ = [
    "DLDeviceType",
    "DLDataTypeCode",
    "DLDevice",
    "DLDataType",
    "DLTensor",
    "DLPackVersion",
    "DLManagedTensor",
    "DLManagedTensorVersioned",
    "DLManagedTensorDeleter",
    "DLManagedTensorVersionedDeleter",
    "DLPACK_FLAG_BITMASK_READ_ONLY",
    "DLPACK_FLAG_BITMASK_IS_COPIED",
    "is_dlpack_available",
    "from_dlpack",
    "to_dlpack",
    "_ACTIVE_MANAGED_TENSORS",
]
