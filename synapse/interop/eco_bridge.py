"""Synapse Transparent Ecosystem Bridge & PyPI Package Resolver.

Provides seamless dynamic import resolution, actionable installation guidance for
missing packages, and zero-copy DLPack tensor interop between Synapse and the
Python scientific ecosystem (NumPy, PyTorch, CuPy, JAX, etc.).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import numpy as np

from synapse.core.tensor import Tensor
from synapse.interop.auto_pip import AutoPipManager
from synapse.interop.dlpack import (
    from_dlpack,
    to_dlpack,
    is_dlpack_available,
)
from synapse.interop.python_bridge import (
    PythonModuleWrapper,
    SynapseInteropError,
    to_synapse,
    from_synapse,
    to_synapse_tensor,
    to_numpy,
)

logger = logging.getLogger("synapse.interop.eco_bridge")


class PackageNotFoundError(SynapseInteropError, ModuleNotFoundError):
    """Açıklayıcı ve eyleme geçirilebilir PyPI paket bulunamadı hatası."""

    def __init__(self, module_name: str, original_exception: Optional[Exception] = None):
        self.module_name = module_name
        self.original_exception = original_exception
        clean_name = module_name[3:] if module_name.startswith("py.") else module_name
        message = (
            f"Package '{clean_name}' is not installed in the environment. "
            f"Run 'pip install {clean_name}' or 'synapse pkg install py:{clean_name}' to use it in Synapse."
        )
        super().__init__(message, original_exception=original_exception)


class TransparentPyResolver:
    """
    Şeffaf Python Ekosistem Çözümleyicisi.
    Python modüllerini dinamik olarak çözümler, eksik paketlerde JIT otomatik pip kurulumu yapar
    ve PyTorch/NumPy tensörlerini DLPack C-ABI üzerinden sıfır kopyalı Synapse tensörlerine dönüştürür.
    """

    @staticmethod
    def resolve_import(module_name: str, wrap: bool = True, auto_install: bool = True) -> Any:
        """Dinamik olarak Python modülünü içeri aktarır.

        Args:
            module_name: Modül adı ('math', 'json', 'os.path', 'py.numpy', vb.)
            wrap: True ise Synapse çağrı/dönüşüm proxy'si (PythonModuleWrapper) ile sarmalar.
            auto_install: Modül bulunamadığında AutoPipManager ile JIT kurulum yapılıp yapılmayacağı.

        Returns:
            PythonModuleWrapper veya ham Python modülü.

        Raises:
            PackageNotFoundError: Paket kurulu değilse veya kurulum başarısızsa açıklayıcı rehber mesajla fırlatılır.
        """
        # 'py.' önekini kaldır (örn: py.numpy -> numpy, py.torch -> torch)
        clean_name = module_name[3:] if module_name.startswith("py.") else module_name

        try:
            mod = importlib.import_module(clean_name)
        except (ModuleNotFoundError, ImportError) as exc:
            # JIT otomatik kurulum dene
            if auto_install:
                try:
                    pip_mgr = AutoPipManager()
                    if pip_mgr.ensure_package(clean_name, auto_install=True):
                        try:
                            mod = importlib.import_module(clean_name)
                            if wrap:
                                return PythonModuleWrapper(mod, name=clean_name)
                            return mod
                        except (ModuleNotFoundError, ImportError):
                            pass
                except Exception as install_exc:
                    logger.debug("Auto-pip installation attempt failed: %s", install_exc)

            # Modül veya bağımlı paket bulunamadıysa açıklayıcı rehber hata fırlat
            raise PackageNotFoundError(clean_name, original_exception=exc) from exc

        if wrap:
            return PythonModuleWrapper(mod, name=clean_name)
        return mod

    @staticmethod
    def from_dlpack(obj: Any, requires_grad: bool = False) -> Tensor:
        """PyTorch, NumPy veya DLPack nesnesinden sıfır kopyalı Synapse Tensor üretir."""
        return from_dlpack(obj, requires_grad=requires_grad)

    @staticmethod
    def to_dlpack(tensor: Any) -> Any:
        """Synapse Tensor nesnesini DLPack PyCapsule formatına ihraç eder."""
        return to_dlpack(tensor)

    @staticmethod
    def to_synapse(obj: Any) -> Any:
        """Python çıktısını DLPack sıfır-kopyalı Synapse nesnesine dönüştürür."""
        return to_synapse(obj)

    @staticmethod
    def from_synapse(obj: Any) -> Any:
        """Synapse nesnesini Python uyumlu veri türüne dönüştürür."""
        return from_synapse(obj)

    @staticmethod
    def to_synapse_tensor(obj: Any, requires_grad: bool = False) -> Tensor:
        """NumPy veya PyTorch tensörünü sıfır kopyalı Synapse Tensor'e dönüştürür."""
        return to_synapse_tensor(obj, requires_grad=requires_grad)

    @staticmethod
    def to_numpy(t: Tensor) -> np.ndarray:
        """Synapse Tensor'ü NumPy ndarray formatına çevirir."""
        return to_numpy(t)

    @staticmethod
    def is_dlpack_supported() -> bool:
        """DLPack C-ABI desteğinin mevcut olup olmadığını kontrol eder."""
        return is_dlpack_available()


__all__ = [
    "AutoPipManager",
    "PackageNotFoundError",
    "TransparentPyResolver",
]
