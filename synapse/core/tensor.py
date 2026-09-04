from __future__ import annotations
import logging
from typing import Any, Callable, Optional, Sequence, Union
import numpy as np

from synapse.core.device import Device, cpu, cuda
from synapse.core.cuda_backend import CUDABackend, is_cuda_available

logger = logging.getLogger("synapse.tensor")


# =========================================================================
# FP8 Lookups & Quantization Tables (E4M3 and E5M2 Formats)
# =========================================================================
def _build_fp8_e4m3_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 3) & 0xF
        m = b & 0x7
        sign = -1.0 if s else 1.0
        if e == 0:
            val = sign * (2.0 ** -6) * (m / 8.0)
        elif e == 15 and m == 7:
            val = np.nan
        else:
            val = sign * (2.0 ** (e - 7)) * (1.0 + m / 8.0)
        table[b] = val
    return table


def _build_fp8_e5m2_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 2) & 0x1F
        m = b & 0x3
        sign = -1.0 if s else 1.0
        if e == 0:
            val = sign * (2.0 ** -14) * (m / 4.0)
        elif e == 31:
            val = np.nan
        else:
            val = sign * (2.0 ** (e - 15)) * (1.0 + m / 4.0)
        table[b] = val
    return table


_FP8_E4M3_TABLE = _build_fp8_e4m3_table()
_FP8_E5M2_TABLE = _build_fp8_e5m2_table()
_FP8_E4M3_POS_TABLE = _FP8_E4M3_TABLE[:127]  # 0 to 126 positive non-NaN
_FP8_E5M2_POS_TABLE = _FP8_E5M2_TABLE[:124]  # 0 to 123 positive non-NaN


def _quantize_fp8_e4m3(data: np.ndarray, scale: float) -> np.ndarray:
    scaled = (data / scale).astype(np.float32)
    signs = (scaled < 0).astype(np.uint8)
    abs_scaled = np.clip(np.abs(scaled), 0.0, 448.0)

    idx = np.searchsorted(_FP8_E4M3_POS_TABLE, abs_scaled)
    idx = np.clip(idx, 0, len(_FP8_E4M3_POS_TABLE) - 1)

    prev_idx = np.maximum(idx - 1, 0)
    diff_curr = np.abs(_FP8_E4M3_POS_TABLE[idx] - abs_scaled)
    diff_prev = np.abs(_FP8_E4M3_POS_TABLE[prev_idx] - abs_scaled)
    best_idx = np.where(diff_prev < diff_curr, prev_idx, idx).astype(np.uint8)

    code = best_idx | (signs << 7)
    return code.view(np.int8)


def _quantize_fp8_e5m2(data: np.ndarray, scale: float) -> np.ndarray:
    scaled = (data / scale).astype(np.float32)
    signs = (scaled < 0).astype(np.uint8)
    abs_scaled = np.clip(np.abs(scaled), 0.0, 57344.0)

    idx = np.searchsorted(_FP8_E5M2_POS_TABLE, abs_scaled)
    idx = np.clip(idx, 0, len(_FP8_E5M2_POS_TABLE) - 1)

    prev_idx = np.maximum(idx - 1, 0)
    diff_curr = np.abs(_FP8_E5M2_POS_TABLE[idx] - abs_scaled)
    diff_prev = np.abs(_FP8_E5M2_POS_TABLE[prev_idx] - abs_scaled)
    best_idx = np.where(diff_prev < diff_curr, prev_idx, idx).astype(np.uint8)

    code = best_idx | (signs << 7)
    return code.view(np.int8)


class TensorShapeMismatchError(Exception):
    """Raised when tensor shapes are incompatible for an operation, such as matrix multiplication."""
    pass


class Tensor:
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        device: Union[Device, str] = "cpu",
        dtype: Optional[Union[np.dtype, str]] = None,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ):
        if dtype is not None:
            target_dtype = np.dtype(dtype)
        elif isinstance(data, np.ndarray):
            target_dtype = data.dtype
        else:
            target_dtype = np.float64
        if isinstance(data, np.ndarray):
            self.data = data.astype(target_dtype)
        elif isinstance(data, (int, float)):
            self.data = np.array(data, dtype=target_dtype)
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=target_dtype)
        elif isinstance(data, Tensor):
            self.data = data.data.copy().astype(target_dtype) if dtype is not None else data.data.copy()
            if device == "cpu" and data.device != "cpu":
                device = data.device
        else:
            raise TypeError(f"Unsupported data type for Tensor: {type(data)}")

        self.device: Device = Device(device) if isinstance(device, str) else (device if isinstance(device, Device) else Device(device))
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __getitem__(self, idx: Any) -> Any:
        res = self.data[idx]
        if isinstance(res, np.ndarray):
            return Tensor(res, device=self.device)
        return float(res) if np.issubdtype(type(res), np.number) else res

    def __setitem__(self, idx: Any, value: Any):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    def data_ptr(self) -> int:
        """Returns the memory address of the tensor data buffer as an integer."""
        return int(self.data.ctypes.data)

    def c_pointer(self, ctype: Any = None) -> Any:
        """Returns a ctypes pointer to the tensor data buffer for zero-copy C interoperability."""
        import ctypes
        from synapse.interop.c_ffi import resolve_type
        target = resolve_type(ctype) if ctype is not None else ctypes.c_void_p
        return self.data.ctypes.data_as(target)


    # ==========================================
    # Device Management & Transfers
    # ==========================================
    def to(self, device: Union[Device, str]) -> Tensor:
        """Transfers tensor to the target device (CPU, CUDA, MPS).
        
        If the target device is CUDA and no CUDA hardware is available,
        gracefully falls back to CPU with a log message.
        """
        target_device = Device(device) if isinstance(device, str) else device
        if self.device == target_device:
            return self

        if target_device.device_type == "cuda" and not is_cuda_available():
            logger.warning(
                f"CUDA is not available on this machine. Falling back to CPU for tensor device '{target_device}'."
            )
            target_device = Device("cpu")

        out = Tensor(
            self.data.copy(),
            requires_grad=self.requires_grad,
            device=target_device,
            _children=(self,),
            _op=f"to({target_device})",
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_self = out.grad.to(self.device)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad.data += grad_self.data

        out._backward = _backward
        return out

    def cuda(self, index: int = 0) -> Tensor:
        """Shortcut to transfer tensor to CUDA GPU at index."""
        return self.to(Device("cuda", index))

    def cpu(self) -> Tensor:
        """Shortcut to transfer tensor to CPU."""
        return self.to(Device("cpu"))

    def _match_device(self, other: Tensor) -> tuple[Device, Tensor, Tensor]:
        """Reconciles device mismatches between operands, prioritizing CUDA."""
        if self.device == other.device:
            return self.device, self, other

        if self.device.device_type == "cuda":
            return self.device, self, other.to(self.device)
        elif other.device.device_type == "cuda":
            return other.device, self.to(other.device), other
        else:
            return self.device, self, other.to(self.device)

    # ==========================================
    # Properties & Views
    # ==========================================
    @property
    def T(self) -> Tensor:
        out = Tensor(self.data.T, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="T")

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_t = out.grad.data.T
                if self.grad is None:
                    self.grad = Tensor(grad_t, device=self.device)
                else:
                    self.grad.data += grad_t

        out._backward = _backward
        return out

    # ==========================================
    # Operators & Autograd Backwards
    # ==========================================
    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        target_device, s, o = self._match_device(other)
        out_data = CUDABackend.add(s.data, o.data, target_device)
        out = Tensor(out_data, requires_grad=s.requires_grad or o.requires_grad, device=target_device, _children=(s, o), _op="+")

        def _backward():
            if out.grad is None:
                return
            if s.requires_grad:
                grad_self = _unbroadcast(out.grad.data, s.shape)
                if s.grad is None:
                    s.grad = Tensor(grad_self, device=s.device)
                else:
                    s.grad.data += grad_self

            if o.requires_grad:
                grad_other = _unbroadcast(out.grad.data, o.shape)
                if o.grad is None:
                    o.grad = Tensor(grad_other, device=o.device)
                else:
                    o.grad.data += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        target_device, s, o = self._match_device(other)
        out_data = CUDABackend.sub(s.data, o.data, target_device)
        out = Tensor(out_data, requires_grad=s.requires_grad or o.requires_grad, device=target_device, _children=(s, o), _op="-")

        def _backward():
            if out.grad is None:
                return
            if s.requires_grad:
                grad_self = _unbroadcast(out.grad.data, s.shape)
                if s.grad is None:
                    s.grad = Tensor(grad_self, device=s.device)
                else:
                    s.grad.data += grad_self

            if o.requires_grad:
                grad_other = _unbroadcast(-out.grad.data, o.shape)
                if o.grad is None:
                    o.grad = Tensor(grad_other, device=o.device)
                else:
                    o.grad.data += grad_other

        out._backward = _backward
        return out

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other - self

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        target_device, s, o = self._match_device(other)
        out_data = CUDABackend.mul(s.data, o.data, target_device)
        out = Tensor(out_data, requires_grad=s.requires_grad or o.requires_grad, device=target_device, _children=(s, o), _op="*")

        def _backward():
            if out.grad is None:
                return
            if s.requires_grad:
                grad_self = _unbroadcast(CUDABackend.mul(out.grad.data, o.data, target_device), s.shape)
                if s.grad is None:
                    s.grad = Tensor(grad_self, device=s.device)
                else:
                    s.grad.data += grad_self

            if o.requires_grad:
                grad_other = _unbroadcast(CUDABackend.mul(out.grad.data, s.data, target_device), o.shape)
                if o.grad is None:
                    o.grad = Tensor(grad_other, device=o.device)
                else:
                    o.grad.data += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        target_device, s, o = self._match_device(other)
        out_data = CUDABackend.truediv(s.data, o.data, target_device)
        out = Tensor(out_data, requires_grad=s.requires_grad or o.requires_grad, device=target_device, _children=(s, o), _op="/")

        def _backward():
            if out.grad is None:
                return
            if s.requires_grad:
                grad_self = _unbroadcast(CUDABackend.truediv(out.grad.data, o.data, target_device), s.shape)
                if s.grad is None:
                    s.grad = Tensor(grad_self, device=s.device)
                else:
                    s.grad.data += grad_self

            if o.requires_grad:
                grad_other = _unbroadcast(-out.grad.data * s.data / (o.data ** 2), o.shape)
                if o.grad is None:
                    o.grad = Tensor(grad_other, device=o.device)
                else:
                    o.grad.data += grad_other

        out._backward = _backward
        return out

    def __matmul__(self, other: Union[Tensor, QuantizedTensor]) -> Tensor:
        if isinstance(other, QuantizedTensor):
            if len(self.shape) >= 2 and len(other.shape) >= 2:
                if self.shape[-1] != other.shape[-2]:
                    raise TensorShapeMismatchError(
                        f"Cannot multiply tensor of shape {self.shape} with quantized tensor of shape {other.shape}. Inner dimensions must match: {self.shape[-1]} != {other.shape[-2]}."
                    )
            return self @ other.dequantize()

        if not isinstance(other, Tensor):
            raise TypeError(f"Matrix multiplication requires another Tensor, got {type(other)}")

        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise TensorShapeMismatchError(
                    f"Cannot multiply tensor of shape {self.shape} with tensor of shape {other.shape}. Inner dimensions must match: {self.shape[-1]} != {other.shape[-2]}."
                )

        target_device, s, o = self._match_device(other)
        out_data = CUDABackend.matmul(s.data, o.data, target_device)
        out = Tensor(out_data, requires_grad=s.requires_grad or o.requires_grad, device=target_device, _children=(s, o), _op="@")

        def _backward():
            if out.grad is None:
                return
            if s.requires_grad:
                # d(A @ B)/dA = grad @ B.T
                grad_self = CUDABackend.matmul(out.grad.data, o.data.T, target_device)
                if s.grad is None:
                    s.grad = Tensor(grad_self, device=s.device)
                else:
                    s.grad.data += grad_self

            if o.requires_grad:
                # d(A @ B)/dB = A.T @ grad
                grad_other = CUDABackend.matmul(s.data.T, out.grad.data, target_device)
                if o.grad is None:
                    o.grad = Tensor(grad_other, device=o.device)
                else:
                    o.grad.data += grad_other

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __pow__(self, power: Union[int, float]) -> Tensor:
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op=f"**{power}")

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_val = out.grad.data * (power * (self.data ** (power - 1)))
                if self.grad is None:
                    self.grad = Tensor(grad_val, device=self.device)
                else:
                    self.grad.data += grad_val

        out._backward = _backward
        return out

    # ==========================================
    # Reductions & Activations
    # ==========================================
    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
        out_data = CUDABackend.sum(self.data, axis=axis, keepdims=keepdims, device=self.device)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_data = out.grad.data
                if not keepdims and axis is not None:
                    axes = [axis] if isinstance(axis, int) else list(axis)
                    shape = list(self.shape)
                    for ax in axes:
                        shape[ax] = 1
                    grad_data = grad_data.reshape(shape)

                # Broadcast to self.shape
                grad_self = np.broadcast_to(grad_data, self.shape)
                if self.grad is None:
                    self.grad = Tensor(grad_self, device=self.device)
                else:
                    self.grad.data += grad_self

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
        total = self.sum(axis=axis, keepdims=keepdims)
        denom = float(self.size if axis is None else (np.prod([self.shape[i] for i in ([axis] if isinstance(axis, int) else axis)])))
        return total * (1.0 / denom)

    def relu(self) -> Tensor:
        out_data = CUDABackend.relu(self.data, device=self.device)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_mask = (self.data > 0).astype(np.float64)
                grad_self = out.grad.data * grad_mask
                if self.grad is None:
                    self.grad = Tensor(grad_self, device=self.device)
                else:
                    self.grad.data += grad_self

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        s = CUDABackend.sigmoid(self.data, device=self.device)
        out = Tensor(s, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_self = out.grad.data * (s * (1.0 - s))
                if self.grad is None:
                    self.grad = Tensor(grad_self, device=self.device)
                else:
                    self.grad.data += grad_self

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t_val = CUDABackend.tanh(self.data, device=self.device)
        out = Tensor(t_val, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_self = out.grad.data * (1.0 - t_val * t_val)
                if self.grad is None:
                    self.grad = Tensor(grad_self, device=self.device)
                else:
                    self.grad.data += grad_self

        out._backward = _backward
        return out

    def gelu(self) -> Tensor:
        g_val = CUDABackend.gelu(self.data, device=self.device)
        out = Tensor(g_val, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="gelu")

        def _backward():
            if self.requires_grad and out.grad is not None:
                sqrt_2_over_pi = 0.7978845608028654
                x = self.data
                u = sqrt_2_over_pi * (x + 0.044715 * (x ** 3))
                tanh_u = np.tanh(u)
                du_dx = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * (x ** 2))
                grad_gelu = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u ** 2) * du_dx
                grad_self = out.grad.data * grad_gelu
                if self.grad is None:
                    self.grad = Tensor(grad_self, device=self.device)
                else:
                    self.grad.data += grad_self

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> Tensor:
        s_val = CUDABackend.softmax(self.data, axis=axis, device=self.device)
        out = Tensor(s_val, requires_grad=self.requires_grad, device=self.device, _children=(self,), _op="softmax")

        def _backward():
            if self.requires_grad and out.grad is not None:
                # dL/dx_i = s_i * (dL/ds_i - sum_k(dL/ds_k * s_k))
                sum_grad_s = np.sum(out.grad.data * s_val, axis=axis, keepdims=True)
                grad_self = s_val * (out.grad.data - sum_grad_s)
                if self.grad is None:
                    self.grad = Tensor(grad_self, device=self.device)
                else:
                    self.grad.data += grad_self

        out._backward = _backward
        return out

    # ==========================================
    # Autograd: Backward Engine
    # ==========================================
    def backward(self, gradient: Optional[Tensor] = None):
        if not self.requires_grad:
            self.requires_grad = True

        if gradient is None:
            if self.data.ndim == 0 or self.data.size == 1:
                self.grad = Tensor(np.ones_like(self.data, dtype=np.float64), device=self.device)
            else:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs. Pass gradient explicitly.")
        else:
            self.grad = gradient.to(self.device) if isinstance(gradient, Tensor) else Tensor(gradient, device=self.device)

        # Topolojik sıralama
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Geriye doğru gradyanları işlet
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = None

    def item(self) -> float:
        return float(self.data.item())

    def tolist(self) -> list:
        return self.data.tolist()

    def to_dataframe(self, columns: Optional[Sequence[str]] = None) -> Any:
        """2D tensörü doğrudan Synapse DataFrame nesnesine dönüştürür."""
        from synapse.core.dataframe import DataFrame
        arr = self.cpu().data
        if arr.ndim != 2:
            raise ValueError(f"Only 2D tensors can be converted to DataFrame, got shape {arr.shape}")
        return DataFrame(arr, columns=columns)

    def quantize(
        self,
        dtype: str = "int8",
        scale: Optional[float] = None,
        zero_point: Optional[int] = None,
        symmetric: bool = False,
    ) -> QuantizedTensor:
        """Quantizes tensor to INT8 or FP8 format.

        Calculates symmetric or asymmetric affine quantization parameters:
        - Asymmetric affine (INT8):
          scale = (max_val - min_val) / 255.0
          zero_point = round(-min_val / scale) - 128
        - Symmetric (INT8):
          scale = max_abs / 127.0
          zero_point = 0
        - FP8 (E4M3 / E5M2):
          scale = max_abs / max_fp8_value
          zero_point = 0

        Stores quantized data in np.int8 and returns a QuantizedTensor.
        """
        dtype_lower = dtype.lower()

        if dtype_lower == "int8":
            if scale is None:
                if symmetric:
                    max_abs = float(np.max(np.abs(self.data))) if self.size > 0 else 1.0
                    calc_scale = float(max_abs / 127.0) if max_abs > 1e-12 else 1.0
                    calc_zp = 0
                else:
                    min_val = float(np.min(self.data)) if self.size > 0 else 0.0
                    max_val = float(np.max(self.data)) if self.size > 0 else 0.0
                    diff = max_val - min_val
                    if diff > 1e-12:
                        calc_scale = float(diff / 255.0)
                        calc_zp = int(np.round(-min_val / calc_scale) - 128)
                    else:
                        calc_scale = 1.0
                        calc_zp = 0
            else:
                calc_scale = float(scale)
                calc_zp = int(zero_point) if zero_point is not None else 0

            if zero_point is not None:
                calc_zp = int(zero_point)

            q_float = np.round(self.data / calc_scale) + calc_zp
            qdata = np.clip(q_float, -128, 127).astype(np.int8)

            return QuantizedTensor(
                qdata=qdata,
                scale=calc_scale,
                zero_point=calc_zp,
                original_shape=self.shape,
                dtype="int8",
                device=self.device,
            )

        elif dtype_lower in ("fp8", "fp8_e4m3"):
            max_abs = float(np.max(np.abs(self.data))) if self.size > 0 else 1.0
            calc_scale = float(scale) if scale is not None else (float(max_abs / 448.0) if max_abs > 1e-12 else 1.0)
            qdata = _quantize_fp8_e4m3(self.data, calc_scale)
            return QuantizedTensor(
                qdata=qdata,
                scale=calc_scale,
                zero_point=0,
                original_shape=self.shape,
                dtype="fp8",
                device=self.device,
            )

        elif dtype_lower == "fp8_e5m2":
            max_abs = float(np.max(np.abs(self.data))) if self.size > 0 else 1.0
            calc_scale = float(scale) if scale is not None else (float(max_abs / 57344.0) if max_abs > 1e-12 else 1.0)
            qdata = _quantize_fp8_e5m2(self.data, calc_scale)
            return QuantizedTensor(
                qdata=qdata,
                scale=calc_scale,
                zero_point=0,
                original_shape=self.shape,
                dtype="fp8_e5m2",
                device=self.device,
            )

        else:
            raise ValueError(f"Unsupported quantization dtype: '{dtype}'. Supported: 'int8', 'fp8', 'fp8_e4m3', 'fp8_e5m2'")

    def dequantize(self) -> Tensor:
        """Dequantizes tensor. Returns self for unquantized tensors."""
        return self

    def __repr__(self) -> str:
        req = ", requires_grad=True" if self.requires_grad else ""
        dev = f", device='{self.device}'" if self.device.device_type != "cpu" else ""
        return f"tensor({self.data.tolist()}{req}{dev})"


class QuantizedTensor(Tensor):
    """Represents a low-precision quantized tensor (INT8 / FP8) with hardware compression."""

    def __init__(
        self,
        qdata: np.ndarray,
        scale: float,
        zero_point: int = 0,
        original_shape: Optional[tuple[int, ...]] = None,
        dtype: str = "int8",
        device: Union[Device, str] = "cpu",
    ):
        target_device = Device(device) if isinstance(device, str) else (device if isinstance(device, Device) else Device(device))
        self.device: Device = target_device
        self._dtype_str: str = dtype
        self.scale: float = float(scale)
        self.zero_point: int = int(zero_point)

        if dtype == "int8":
            self.qdata: np.ndarray = np.asarray(qdata, dtype=np.int8)
        else:
            self.qdata: np.ndarray = np.asarray(qdata, dtype=np.int8 if qdata.dtype == np.int8 else qdata.dtype)

        self.quantized_data: np.ndarray = self.qdata
        self.data: np.ndarray = self.qdata
        self.original_shape: tuple[int, ...] = tuple(original_shape) if original_shape is not None else tuple(self.qdata.shape)
        self.requires_grad: bool = False
        self.grad: Optional[Tensor] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set()
        self._op: str = f"quantize({dtype})"

    @property
    def dtype(self) -> str:
        return self._dtype_str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.original_shape

    @property
    def ndim(self) -> int:
        return len(self.original_shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.original_shape))

    def compression_ratio(self) -> float:
        """Memory compression metric vs standard float32 (4 bytes per element)."""
        elem_bytes = self.qdata.itemsize
        return float(4.0 / max(elem_bytes, 1))

    def dequantize(self) -> Tensor:
        """Reconstructs float32 tensor: data = (quantized_data - zero_point) * scale."""
        if self._dtype_str == "int8":
            float_data = (self.qdata.astype(np.float32) - self.zero_point) * self.scale
        elif self._dtype_str in ("fp8", "fp8_e4m3"):
            raw_codes = self.qdata.view(np.uint8)
            float_data = (_FP8_E4M3_TABLE[raw_codes] * self.scale).astype(np.float32)
        elif self._dtype_str == "fp8_e5m2":
            raw_codes = self.qdata.view(np.uint8)
            float_data = (_FP8_E5M2_TABLE[raw_codes] * self.scale).astype(np.float32)
        else:
            float_data = (self.qdata.astype(np.float32) - self.zero_point) * self.scale

        float_data = float_data.reshape(self.original_shape)
        return Tensor(float_data, dtype=np.float32, device=self.device)

    def __matmul__(self, other: Union[QuantizedTensor, Tensor]) -> Tensor:
        if isinstance(other, QuantizedTensor):
            if len(self.shape) >= 2 and len(other.shape) >= 2:
                if self.shape[-1] != other.shape[-2]:
                    raise TensorShapeMismatchError(
                        f"Cannot multiply quantized tensor of shape {self.shape} with {other.shape}. "
                        f"Inner dimensions must match: {self.shape[-1]} != {other.shape[-2]}."
                    )

            if self._dtype_str == "int8" and other._dtype_str == "int8":
                # Integer GEMM: int8 @ int8 with int32 accumulator
                int_a = self.qdata.astype(np.int32) - self.zero_point
                int_b = other.qdata.astype(np.int32) - other.zero_point
                int_gemm = np.matmul(int_a, int_b)
                rescaled = int_gemm.astype(np.float32) * (self.scale * other.scale)
                return Tensor(rescaled, dtype=np.float32, device=self.device)
            else:
                # FP8 or mixed GEMM
                a_float = self.dequantize().data.astype(np.float32)
                b_float = other.dequantize().data.astype(np.float32)
                out_data = np.matmul(a_float, b_float)
                return Tensor(out_data, dtype=np.float32, device=self.device)

        elif isinstance(other, Tensor):
            return self.dequantize() @ other
        else:
            raise TypeError(f"Matrix multiplication requires Tensor or QuantizedTensor, got {type(other)}")

    def __rmatmul__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return other @ self.dequantize()
        return NotImplemented

    def to(self, device: Union[Device, str]) -> QuantizedTensor:
        target_device = Device(device) if isinstance(device, str) else device
        return QuantizedTensor(
            self.qdata.copy(),
            scale=self.scale,
            zero_point=self.zero_point,
            original_shape=self.original_shape,
            dtype=self._dtype_str,
            device=target_device,
        )

    def cpu(self) -> QuantizedTensor:
        return self.to("cpu")

    def cuda(self, index: int = 0) -> QuantizedTensor:
        return self.to(Device("cuda", index))

    def __repr__(self) -> str:
        return (
            f"QuantizedTensor(shape={self.shape}, dtype='{self._dtype_str}', "
            f"scale={self.scale:.6g}, zero_point={self.zero_point}, "
            f"compression={self.compression_ratio():.1f}x)"
        )


# ==========================================
# Factory Functions & Helpers
# ==========================================
def tensor(data: Any, requires_grad: bool = False, device: Union[Device, str] = "cpu", dtype: Optional[Union[np.dtype, str]] = None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)


def quantize(
    tensor_or_data: Any,
    dtype: str = "int8",
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    symmetric: bool = False,
    device: Union[Device, str] = "cpu",
) -> QuantizedTensor:
    """Quantizes a Tensor or array-like data to low-precision QuantizedTensor."""
    if isinstance(tensor_or_data, QuantizedTensor):
        return tensor_or_data
    if isinstance(tensor_or_data, Tensor):
        return tensor_or_data.quantize(dtype=dtype, scale=scale, zero_point=zero_point, symmetric=symmetric)
    t = Tensor(tensor_or_data, device=device)
    return t.quantize(dtype=dtype, scale=scale, zero_point=zero_point, symmetric=symmetric)


def dequantize(quant_tensor: Union[QuantizedTensor, Tensor]) -> Tensor:
    """Reconstructs float32 Tensor from QuantizedTensor."""
    if isinstance(quant_tensor, QuantizedTensor):
        return quant_tensor.dequantize()
    elif isinstance(quant_tensor, Tensor):
        return quant_tensor
    raise TypeError(f"Expected QuantizedTensor or Tensor, got {type(quant_tensor)}")


def zeros(shape: Union[int, Sequence[int]], requires_grad: bool = False, device: Union[Device, str] = "cpu", dtype: Optional[Union[np.dtype, str]] = None) -> Tensor:
    s = (shape,) if isinstance(shape, int) else tuple(shape)
    target_dtype = np.dtype(dtype) if dtype is not None else np.float64
    return Tensor(np.zeros(s, dtype=target_dtype), requires_grad=requires_grad, device=device, dtype=dtype)


def ones(shape: Union[int, Sequence[int]], requires_grad: bool = False, device: Union[Device, str] = "cpu", dtype: Optional[Union[np.dtype, str]] = None) -> Tensor:
    s = (shape,) if isinstance(shape, int) else tuple(shape)
    target_dtype = np.dtype(dtype) if dtype is not None else np.float64
    return Tensor(np.ones(s, dtype=target_dtype), requires_grad=requires_grad, device=device, dtype=dtype)


def randn(shape: Union[int, Sequence[int]], requires_grad: bool = False, device: Union[Device, str] = "cpu", dtype: Optional[Union[np.dtype, str]] = None) -> Tensor:
    s = (shape,) if isinstance(shape, int) else tuple(shape)
    target_dtype = np.dtype(dtype) if dtype is not None else np.float64
    return Tensor(np.random.randn(*s).astype(target_dtype), requires_grad=requires_grad, device=device, dtype=dtype)


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcasting sonrası boyut uyumu için gradyanı toplayarak küçültür."""
    if grad.shape == target_shape:
        return grad

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)

    for _ in range(grad_ndim - target_ndim):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
