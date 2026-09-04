from __future__ import annotations
import importlib
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from synapse.core.tensor import Tensor, tensor


class SynapseInteropError(Exception):
    """Python-Synapse FFI Köprüsü sırasında oluşan hataları temsil eder."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


def to_synapse_tensor(obj: Any, requires_grad: bool = False) -> Tensor:
    """Python/NumPy/PyTorch tensörünü doğrudan Synapse Tensor nesnesine çevirir."""
    if isinstance(obj, Tensor):
        return obj

    # DLPack sıfır-kopyalı interop denemesi
    if hasattr(obj, "__dlpack__") or type(obj).__name__ == "PyCapsule":
        try:
            from synapse.interop.dlpack import from_dlpack, is_dlpack_available
            if is_dlpack_available():
                return from_dlpack(obj, requires_grad=requires_grad)
        except Exception:
            # Graceful fallback: DLPack başarısız olursa standart dönüştürme akışına düş
            pass

    # PyTorch/Array tensörü kontrolü (DLPack kullanılamazsa veya fallback durumunda)
    if hasattr(obj, "detach") and hasattr(obj, "numpy"):
        detached = obj.detach()
        if hasattr(detached, "cpu"):
            detached = detached.cpu()
        return Tensor(detached.numpy(), requires_grad=requires_grad)

    if hasattr(obj, "numpy") and callable(obj.numpy):
        return Tensor(obj.numpy(), requires_grad=requires_grad)

    if isinstance(obj, np.ndarray):
        return Tensor(obj, requires_grad=requires_grad)

    if isinstance(obj, (list, tuple, int, float)):
        return Tensor(obj, requires_grad=requires_grad)

    raise TypeError(f"Cannot convert object of type {type(obj)} to Synapse Tensor")


def to_numpy(t: Tensor) -> np.ndarray:
    """Synapse Tensor'ü NumPy ndarray formatına çevirir."""
    return t.data


def from_synapse(obj: Any) -> Any:
    """Synapse nesnesini Python uyumlu veri türüne dönüştürür."""
    if isinstance(obj, Tensor):
        return obj.data
    # DataFrame kontrolü
    if hasattr(obj, "to_dict") and hasattr(obj, "_columns"):
        return obj.to_dict()
    # SynapseFunction veya çağrılabilir fonksiyon sarmalayıcısı
    if callable(obj):
        def py_callback(*args, **kwargs):
            clean_args = [to_synapse(a) for a in args]
            clean_kwargs = {k: to_synapse(v) for k, v in kwargs.items()}
            res = obj(*clean_args, **clean_kwargs)
            return from_synapse(res)
        return py_callback
    return obj


def to_synapse(obj: Any) -> Any:
    """Python çıktısını Synapse uyumlu nesneye dönüştürür."""
    if isinstance(obj, Tensor):
        return obj
    # PyTorch, NumPy veya DLPack uyumlu tensörleri otomatik sıfır-kopyalı dönüştür
    if (
        isinstance(obj, np.ndarray)
        or hasattr(obj, "__dlpack__")
        or type(obj).__name__ == "PyCapsule"
        or (hasattr(obj, "detach") and hasattr(obj, "numpy"))
    ):
        return to_synapse_tensor(obj)
    # Pandas DataFrame kontrolü
    if hasattr(obj, "to_dict") and hasattr(obj, "columns") and not isinstance(obj, dict):
        try:
            from synapse.core.dataframe import DataFrame
            return DataFrame(obj.to_dict(orient="list"))
        except Exception:
            return obj
    return obj


class PythonModuleWrapper:
    """
    Python modülünü veya alt nesnesini Synapse uyumlu olarak sarmalayan proxy.
    Tüm argümanları ve dönüş tiplerini otomatik olarak iki yönlü dönüştürür.
    """
    def __init__(self, mod: Any, name: Optional[str] = None):
        self._mod = mod
        self._name = name or getattr(mod, "__name__", str(mod))

    def __getattr__(self, name: str) -> Any:
        try:
            attr = getattr(self._mod, name)
        except AttributeError:
            # Alt modül yükleme denemesi: örn. scipy.optimize
            try:
                sub_mod_name = f"{self._name}.{name}"
                sub_mod = importlib.import_module(sub_mod_name)
                return PythonModuleWrapper(sub_mod, name=sub_mod_name)
            except ImportError:
                raise AttributeError(f"Python module/object '{self._name}' has no attribute or submodule '{name}'")

        if callable(attr):
            def wrapped(*args, **kwargs):
                clean_args = [from_synapse(a) for a in args]
                clean_kwargs = {k: from_synapse(v) for k, v in kwargs.items()}
                try:
                    res = attr(*clean_args, **clean_kwargs)
                    return to_synapse(res)
                except Exception as e:
                    exc_msg = f"Python Interop Error in '{self._name}.{name}()': {e}"
                    raise SynapseInteropError(exc_msg, original_exception=e) from e
            return wrapped

        if hasattr(attr, "__dict__") and not isinstance(attr, (int, float, str, bool, list, tuple, dict)):
            return PythonModuleWrapper(attr, name=f"{self._name}.{name}")

        return to_synapse(attr)

    def __call__(self, *args, **kwargs) -> Any:
        clean_args = [from_synapse(a) for a in args]
        clean_kwargs = {k: from_synapse(v) for k, v in kwargs.items()}
        try:
            res = self._mod(*clean_args, **clean_kwargs)
            return to_synapse(res)
        except Exception as e:
            exc_msg = f"Python Interop Error calling '{self._name}': {e}"
            raise SynapseInteropError(exc_msg, original_exception=e) from e

    def __repr__(self) -> str:
        return f"<SynapsePythonBridge for {self._name}>"


class PythonBridge:
    @staticmethod
    def load(module_name: str) -> PythonModuleWrapper:
        """Belirtilen Python modülünü sıfır sürtünmeyle Synapse'e bağlar."""
        from synapse.interop.eco_bridge import TransparentPyResolver
        return TransparentPyResolver.resolve_import(module_name)
