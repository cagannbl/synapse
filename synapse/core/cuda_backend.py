from __future__ import annotations
import ctypes
import logging
import os
import sys
from typing import Any, Optional, Sequence, Union
import numpy as np

from synapse.core.device import Device

logger = logging.getLogger("synapse.cuda")

# =========================================================================
# Hardware & Driver Level Detection
# =========================================================================
_DRIVER_INITIALIZED = False
_CUDA_DRIVER_LIB = None
_DRIVER_DEVICE_COUNT = 0
_DRIVER_DEVICE_NAMES: dict[int, str] = {}


def _init_cuda_driver():
    global _DRIVER_INITIALIZED, _CUDA_DRIVER_LIB, _DRIVER_DEVICE_COUNT, _DRIVER_DEVICE_NAMES
    if _DRIVER_INITIALIZED:
        return

    _DRIVER_INITIALIZED = True

    # Attempt to load nvcuda (Windows) or libcuda (Linux)
    lib = None
    if sys.platform.startswith("win"):
        try:
            lib = ctypes.windll.LoadLibrary("nvcuda.dll")
        except Exception:
            try:
                lib = ctypes.windll.LoadLibrary("nvml.dll")
            except Exception:
                lib = None
    else:
        for libname in ("libcuda.so.1", "libcuda.so", "libnvidia-ml.so"):
            try:
                lib = ctypes.CDLL(libname)
                break
            except Exception:
                continue

    if lib is None:
        return

    _CUDA_DRIVER_LIB = lib

    # Initialize driver API: cuInit(0)
    try:
        if hasattr(lib, "cuInit"):
            res = lib.cuInit(0)
            if res == 0:  # CUDA_SUCCESS
                count = ctypes.c_int(0)
                if hasattr(lib, "cuDeviceGetCount") and lib.cuDeviceGetCount(ctypes.byref(count)) == 0:
                    _DRIVER_DEVICE_COUNT = count.value
                    for i in range(_DRIVER_DEVICE_COUNT):
                        dev = ctypes.c_int(0)
                        if lib.cuDeviceGet(ctypes.byref(dev), i) == 0:
                            name_buf = ctypes.create_string_buffer(256)
                            if lib.cuDeviceGetName(name_buf, 256, dev.value) == 0:
                                _DRIVER_DEVICE_NAMES[i] = name_buf.value.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        logger.debug(f"CUDA driver initialization note: {e}")


# =========================================================================
# Framework Detection (PyTorch / CuPy)
# =========================================================================
def _check_torch() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _check_cupy() -> bool:
    try:
        import cupy as cp
        return bool(cp.cuda.is_available())
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Returns True if CUDA hardware/runtime is available on the host machine."""
    # 1. Check PyTorch
    if _check_torch():
        return True
    # 2. Check CuPy
    if _check_cupy():
        return True
    # 3. Check direct CUDA driver API
    _init_cuda_driver()
    if _DRIVER_DEVICE_COUNT > 0:
        return True
    return False


def device_count() -> int:
    """Returns the total number of accessible CUDA devices."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass

    try:
        import cupy as cp
        if cp.cuda.is_available():
            return cp.cuda.runtime.getDeviceCount()
    except Exception:
        pass

    _init_cuda_driver()
    return _DRIVER_DEVICE_COUNT


def get_device_name(index: int = 0) -> str:
    """Returns the descriptive model name of the CUDA GPU at given index."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(index)
    except Exception:
        pass

    _init_cuda_driver()
    if index in _DRIVER_DEVICE_NAMES:
        return _DRIVER_DEVICE_NAMES[index]

    if is_cuda_available():
        return f"CUDA Device {index}"
    return "No CUDA Device"


# =========================================================================
# CUDA Backend Execution & Acceleration Bridge
# =========================================================================
class CUDABackend:
    """High-performance tensor acceleration backend.
    
    Routes tensor operations to CUDA tensor cores via PyTorch / CuPy when available,
    and provides seamless graceful fallback to CPU/NumPy with zero crash risk.
    """

    @staticmethod
    def is_available() -> bool:
        return is_cuda_available()

    @staticmethod
    def device_count() -> int:
        return device_count()

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        return get_device_name(index)

    @staticmethod
    def matmul(a: np.ndarray, b: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Matrix multiplication accelerated on CUDA cores with graceful CPU fallback."""
        if device is not None and device.device_type == "cuda":
            # 1. Try PyTorch CUDA acceleration
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    t_a = torch.from_numpy(a).to(dev_str)
                    t_b = torch.from_numpy(b).to(dev_str)
                    res = torch.matmul(t_a, t_b)
                    return res.cpu().numpy()
            except Exception as e:
                logger.debug(f"PyTorch CUDA matmul fallback: {e}")

            # 2. Try CuPy CUDA acceleration
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        c_a = cp.asarray(a)
                        c_b = cp.asarray(b)
                        res = cp.matmul(c_a, c_b)
                        return cp.asnumpy(res)
            except Exception as e:
                logger.debug(f"CuPy CUDA matmul fallback: {e}")

            # 3. Graceful Fallback to CPU OpenBLAS/NumPy
            logger.info(
                f"CUDA acceleration requested on {device}, executing via CPU tensor fallback."
            )

        return a @ b

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Elementwise addition accelerated on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return (torch.from_numpy(a).to(dev_str) + torch.from_numpy(b).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        return cp.asnumpy(cp.asarray(a) + cp.asarray(b))
            except Exception:
                pass
        return a + b

    @staticmethod
    def sub(a: np.ndarray, b: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Elementwise subtraction accelerated on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return (torch.from_numpy(a).to(dev_str) - torch.from_numpy(b).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        return cp.asnumpy(cp.asarray(a) - cp.asarray(b))
            except Exception:
                pass
        return a - b

    @staticmethod
    def mul(a: np.ndarray, b: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Elementwise multiplication accelerated on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return (torch.from_numpy(a).to(dev_str) * torch.from_numpy(b).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        return cp.asnumpy(cp.asarray(a) * cp.asarray(b))
            except Exception:
                pass
        return a * b

    @staticmethod
    def truediv(a: np.ndarray, b: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Elementwise division accelerated on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return (torch.from_numpy(a).to(dev_str) / torch.from_numpy(b).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        return cp.asnumpy(cp.asarray(a) / cp.asarray(b))
            except Exception:
                pass
        return a / b

    @staticmethod
    def sum(a: np.ndarray, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, device: Optional[Device] = None) -> np.ndarray:
        """Sum reduction accelerated on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    ta = torch.from_numpy(a).to(dev_str)
                    dim = tuple(axis) if isinstance(axis, (list, tuple)) else axis
                    res = torch.sum(ta, dim=dim, keepdim=keepdims) if dim is not None else torch.sum(ta)
                    return res.cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        ca = cp.asarray(a)
                        return cp.asnumpy(cp.sum(ca, axis=axis, keepdims=keepdims))
            except Exception:
                pass
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def relu(a: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """ReLU activation on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return torch.relu(torch.from_numpy(a).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        return cp.asnumpy(cp.maximum(0, cp.asarray(a)))
            except Exception:
                pass
        return np.maximum(0, a)

    @staticmethod
    def sigmoid(a: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Sigmoid activation on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return torch.sigmoid(torch.from_numpy(a).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        ca = cp.asarray(a)
                        s = 1.0 / (1.0 + cp.exp(-cp.clip(ca, -500, 500)))
                        return cp.asnumpy(s)
            except Exception:
                pass
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))
        return s

    @staticmethod
    def tanh(a: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """Tanh activation on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return torch.tanh(torch.from_numpy(a).to(dev_str)).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        return cp.asnumpy(cp.tanh(cp.asarray(a)))
            except Exception:
                pass
        return np.tanh(a)

    @staticmethod
    def gelu(a: np.ndarray, device: Optional[Device] = None) -> np.ndarray:
        """GELU (Gaussian Error Linear Unit) on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return torch.nn.functional.gelu(torch.from_numpy(a).to(dev_str)).cpu().numpy()
            except Exception:
                pass
        # CPU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (a + 0.044715 * (a ** 3))
        return 0.5 * a * (1.0 + np.tanh(inner))

    @staticmethod
    def softmax(a: np.ndarray, axis: int = -1, device: Optional[Device] = None) -> np.ndarray:
        """Softmax along axis on GPU with CPU fallback."""
        if device is not None and device.device_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device.index}"
                    return torch.softmax(torch.from_numpy(a).to(dev_str), dim=axis).cpu().numpy()
            except Exception:
                pass
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    with cp.cuda.Device(device.index):
                        ca = cp.asarray(a)
                        e = cp.exp(ca - cp.max(ca, axis=axis, keepdims=True))
                        return cp.asnumpy(e / cp.sum(e, axis=axis, keepdims=True))
            except Exception:
                pass
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
