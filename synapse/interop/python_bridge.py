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
    """Converts a Python/NumPy/PyTorch tensor to a Synapse Tensor via zero-copy DLPack.

    Prioritizes the PEP 652 DLPack C-ABI standard (__dlpack__) to achieve zero-copy memory sharing.
    Falls back gracefully to NumPy array buffer sharing if DLPack is unavailable.
    """
    if isinstance(obj, Tensor):
        return obj

    # 1. DLPack zero-copy interop (PEP 652 primary protocol)
    if hasattr(obj, "__dlpack__") or type(obj).__name__ == "PyCapsule":
        try:
            from synapse.interop.dlpack import from_dlpack, is_dlpack_available
            if is_dlpack_available():
                return from_dlpack(obj, requires_grad=requires_grad)
        except Exception:
            # Graceful fallback: fall back to secondary array extraction if DLPack raises
            pass

    # 2. PyTorch tensor extraction (if DLPack was bypassed or unavailable)
    if hasattr(obj, "detach") and hasattr(obj, "numpy"):
        detached = obj.detach()
        if hasattr(detached, "cpu"):
            detached = detached.cpu()
        np_arr = detached.numpy()
        syn_t = Tensor(np_arr, requires_grad=requires_grad)
        syn_t.data = np_arr
        return syn_t

    if hasattr(obj, "numpy") and callable(obj.numpy):
        np_arr = obj.numpy()
        syn_t = Tensor(np_arr, requires_grad=requires_grad)
        syn_t.data = np_arr
        return syn_t

    # 3. Direct NumPy ndarray buffer sharing (guaranteed zero-copy pointer)
    if isinstance(obj, np.ndarray):
        syn_t = Tensor(obj, requires_grad=requires_grad)
        syn_t.data = obj
        return syn_t

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


# =========================================================================
# Transparent Python Import Hooks ('import py.*' & 'from python import ...')
# =========================================================================
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from types import ModuleType


class SynapsePyLoader(Loader):
    """Dynamically loads and wraps Python modules for transparent interop."""
    def __init__(self, fullname: str, is_pkg: bool = False):
        self.fullname = fullname
        self.is_pkg = is_pkg

    def create_module(self, spec: ModuleSpec) -> Any:
        if self.is_pkg:
            mod = ModuleType(spec.name)
            mod.__path__ = []
            return mod
        clean_name = spec.name
        for prefix in ("py.", "python."):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        from synapse.interop.eco_bridge import TransparentPyResolver
        return TransparentPyResolver.resolve_import(clean_name)

    def exec_module(self, module: Any) -> None:
        pass


class SynapsePyFinder(MetaPathFinder):
    """MetaPathFinder intercepting 'py.*' and 'python.*' import expressions."""
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Optional[ModuleSpec]:
        if fullname in ("py", "python"):
            return ModuleSpec(fullname, SynapsePyLoader(fullname, is_pkg=True), is_package=True)
        if fullname.startswith("py.") or fullname.startswith("python."):
            clean = fullname
            for prefix in ("py.", "python."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    break
            import importlib.util
            try:
                real_spec = importlib.util.find_spec(clean)
                is_pkg = bool(real_spec and real_spec.submodule_search_locations is not None)
            except Exception:
                is_pkg = False
            return ModuleSpec(fullname, SynapsePyLoader(fullname, is_pkg=False), is_package=is_pkg)
        return None


class DynamicPyNamespace(ModuleType):
    """Dynamic namespace module for 'py' and 'python' allowing attribute-style access (e.g. py.numpy)."""
    def __init__(self, name: str, prefix: str = ""):
        super().__init__(name)
        self.__path__ = []
        self._prefix = prefix

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        target = f"{self._prefix}.{item}" if self._prefix else item
        from synapse.interop.eco_bridge import TransparentPyResolver
        wrapped = TransparentPyResolver.resolve_import(target)
        setattr(self, item, wrapped)
        sys.modules[f"{self.__name__}.{item}"] = wrapped
        return wrapped


_HOOK_INSTALLED = False


def install_import_hooks() -> None:
    """Installs Synapse Python interop import hooks into sys.meta_path and sys.modules."""
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        if not any(isinstance(f, SynapsePyFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, SynapsePyFinder())
        if "py" not in sys.modules or not isinstance(sys.modules["py"], (DynamicPyNamespace, PythonModuleWrapper)):
            sys.modules["py"] = DynamicPyNamespace("py", "")
        if "python" not in sys.modules or not isinstance(sys.modules["python"], (DynamicPyNamespace, PythonModuleWrapper)):
            sys.modules["python"] = DynamicPyNamespace("python", "")
        _HOOK_INSTALLED = True


def uninstall_import_hooks() -> None:
    """Uninstalls Synapse Python interop import hooks."""
    global _HOOK_INSTALLED
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, SynapsePyFinder)]
    _HOOK_INSTALLED = False


# Automatically install hooks on module load
install_import_hooks()


__all__ = [
    "SynapseInteropError",
    "to_synapse_tensor",
    "to_numpy",
    "from_synapse",
    "to_synapse",
    "PythonModuleWrapper",
    "PythonBridge",
    "SynapsePyLoader",
    "SynapsePyFinder",
    "DynamicPyNamespace",
    "install_import_hooks",
    "uninstall_import_hooks",
]
