"""Synapse Built-in Standard Library (std).

Provides native, zero-dependency modules for AI, math, file system,
cryptography, time, and system integration.
"""

from typing import Any, Dict, Iterator, List, Tuple
from synapse.stdlib import math, fs, crypto, time, sys, http, regex, db, compress, path


class StandardLibrary:
    """Namespace container and registry for the Synapse Standard Library."""

    def __init__(self) -> None:
        self.math = math
        self.fs = fs
        self.crypto = crypto
        self.time = time
        self.sys = sys
        self.http = http
        self.regex = regex
        self.db = db
        self.compress = compress
        self.path = path

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Standard library module '{key}' not found")

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in ("math", "fs", "crypto", "time", "sys", "http", "regex", "db", "compress", "path")

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return ["math", "fs", "crypto", "time", "sys", "http", "regex"]

    def values(self) -> List[Any]:
        return [self.math, self.fs, self.crypto, self.time, self.sys, self.http, self.regex]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}

    def __repr__(self) -> str:
        return "<StandardLibrary [math, fs, crypto, time, sys, http, regex]>"


std = StandardLibrary()

__all__ = [
    "std",
    "StandardLibrary",
    "math",
    "fs",
    "crypto",
    "time",
    "sys",
    "http",
    "regex",
]
