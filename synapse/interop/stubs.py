"""
Synapse Dynamic Python Reflection & Stub Generator (Phase 1)
============================================================
Inspects Python modules dynamically using CPython reflection, inspect,
and typing modules to generate Synapse-compatible type signatures,
documentation tooltips, and LSP autocompletion stubs for PyPI packages.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("synapse.interop.stubs")


class DynamicStubGenerator:
    """
    Reflective Stub Generator for Python packages.
    Inspects functions, classes, methods, docstrings, and type annotations
    to generate Synapse type stubs and autocompletion dictionaries.
    """

    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_module_stubs(cls, module_name: str) -> Dict[str, Any]:
        """Returns or generates the reflective stub index for a given Python module."""
        clean_name = module_name[3:] if module_name.startswith("py.") else module_name

        if clean_name in cls._cache:
            return cls._cache[clean_name]

        try:
            mod = importlib.import_module(clean_name)
        except Exception as e:
            logger.debug("Could not import '%s' for stub generation: %s", clean_name, e)
            return {"module": clean_name, "error": str(e), "members": {}}

        stubs = cls._inspect_module(mod, clean_name)
        cls._cache[clean_name] = stubs
        return stubs

    @classmethod
    def _inspect_module(cls, mod: Any, mod_name: str) -> Dict[str, Any]:
        members: Dict[str, Dict[str, Any]] = {}

        for name in dir(mod):
            if name.startswith("__") and name.endswith("__"):
                continue

            try:
                obj = getattr(mod, name)
            except Exception:
                continue

            doc = inspect.getdoc(obj) or ""
            kind = "unknown"

            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                kind = "function"
                sig = cls._get_signature_str(obj)
                members[name] = {
                    "kind": kind,
                    "name": name,
                    "signature": sig,
                    "doc": doc,
                    "synapse_decl": f"fn {name}{sig}",
                }
            elif inspect.isclass(obj):
                kind = "class"
                class_methods = {}
                for m_name in dir(obj):
                    if m_name.startswith("_") and not m_name.startswith("__init__"):
                        continue
                    try:
                        m_obj = getattr(obj, m_name)
                        if inspect.isfunction(m_obj) or inspect.ismethod(m_obj) or inspect.isbuiltin(m_obj):
                            m_sig = cls._get_signature_str(m_obj)
                            class_methods[m_name] = {
                                "signature": m_sig,
                                "doc": inspect.getdoc(m_obj) or "",
                            }
                    except Exception:
                        continue

                members[name] = {
                    "kind": kind,
                    "name": name,
                    "doc": doc,
                    "methods": class_methods,
                    "synapse_decl": f"struct {name}",
                }
            elif isinstance(obj, (int, float, str, bool)):
                kind = "constant"
                members[name] = {
                    "kind": kind,
                    "name": name,
                    "type": type(obj).__name__,
                    "value": str(obj),
                    "doc": doc,
                    "synapse_decl": f"const {name}: {type(obj).__name__} = {repr(obj)}",
                }
            else:
                kind = "object"
                members[name] = {
                    "kind": kind,
                    "name": name,
                    "type": type(obj).__name__,
                    "doc": doc,
                }

        return {
            "module": mod_name,
            "doc": inspect.getdoc(mod) or "",
            "members": members,
        }

    @classmethod
    def _get_signature_str(cls, func: Any) -> str:
        try:
            sig = inspect.signature(func)
            return str(sig)
        except Exception:
            return "()"

    @classmethod
    def generate_synapse_stubs(cls, module_name: str) -> str:
        """Generates a human-readable and parser-compliant Synapse stub header (.syn)."""
        stubs = cls.get_module_stubs(module_name)
        lines = [
            f"// Synapse Type Stubs for Python module '{stubs.get('module', module_name)}'",
            f"// Generated dynamically via CPython Reflection",
            "",
        ]

        members = stubs.get("members", {})
        for name, item in sorted(members.items()):
            decl = item.get("synapse_decl")
            if decl:
                doc = item.get("doc")
                if doc:
                    first_line = doc.splitlines()[0]
                    lines.append(f"// {first_line}")
                lines.append(decl)
                lines.append("")

        return "\n".join(lines)

    @classmethod
    def get_completions_for_lsp(cls, module_name: str) -> List[Dict[str, Any]]:
        """Returns structured completion list ready for LSP servers."""
        stubs = cls.get_module_stubs(module_name)
        completions = []
        members = stubs.get("members", {})

        for name, item in members.items():
            kind_str = item.get("kind", "text")
            # LSP CompletionItemKind mapping
            # 3: Function, 7: Class, 21: Constant, 6: Variable
            kind_id = 3 if kind_str == "function" else (7 if kind_str == "class" else 21)

            detail = item.get("synapse_decl") or item.get("signature") or kind_str
            doc = item.get("doc", "")

            completions.append({
                "label": name,
                "kind": kind_id,
                "detail": detail,
                "documentation": doc,
                "insertText": name,
            })

        return completions
