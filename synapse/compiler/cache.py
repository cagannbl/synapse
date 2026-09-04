"""
Synapse Compiler AST Cache Module.

Provides disk serialization and memory-mapped loading for parsed Synapse ASTs
indexed by SHA-256 source code hashes. Utilizes Pickle Protocol 5 and in-memory LRU caching.
"""

from __future__ import annotations
import os
import mmap
import pickle
import hashlib
import time
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Any

from synapse.parser.ast_nodes import Program


class ASTCache:
    """
    Two-tier AST Cache (Memory LRU + Disk with Pickle Protocol 5 and memory mapping).
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".syn_cache",
        max_memory_entries: int = 128,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max(1, max_memory_entries)
        self.enabled = enabled
        self._memory_cache: OrderedDict[str, Program] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def compute_hash(source: str) -> str:
        """Calculates SHA-256 hash hex string of the given source code."""
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _resolve_source(self, source: str, filename: str = "") -> str:
        if not source and filename and os.path.isfile(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError:
                pass
        return source

    def _evict_memory_if_needed(self) -> None:
        while len(self._memory_cache) > self.max_memory_entries:
            self._memory_cache.popitem(last=False)

    def get(self, source: str, filename: str = "") -> Optional[Program]:
        """
        Retrieves AST from in-memory LRU cache or disk cache.
        Returns None if not found or if cache is disabled.
        """
        if not self.enabled:
            self.misses += 1
            return None

        effective_source = self._resolve_source(source, filename)
        if not effective_source:
            self.misses += 1
            return None

        key = self.compute_hash(effective_source)

        # 1. Check in-memory LRU cache
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self.hits += 1
            return self._memory_cache[key]

        # 2. Check disk cache
        cache_file = self.cache_dir / f"{key}.ast"
        if cache_file.is_file():
            try:
                file_size = cache_file.stat().st_size
                if file_size > 0:
                    with open(cache_file, "rb") as f:
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            ast = pickle.loads(mm)
                    if isinstance(ast, Program):
                        self._memory_cache[key] = ast
                        self._memory_cache.move_to_end(key)
                        self._evict_memory_if_needed()
                        self.hits += 1
                        return ast
            except Exception:
                # Corrupt cache file or I/O failure, fall back to miss
                pass

        self.misses += 1
        return None

    def set(self, source: str, ast: Program, filename: str = "") -> None:
        """
        Serializes and stores AST in memory LRU cache and to disk (.syn_cache/<hash>.ast)
        using Pickle Protocol 5.
        """
        if not self.enabled or ast is None:
            return

        effective_source = self._resolve_source(source, filename)
        if not effective_source:
            return

        key = self.compute_hash(effective_source)

        # 1. Update memory LRU
        self._memory_cache[key] = ast
        self._memory_cache.move_to_end(key)
        self._evict_memory_if_needed()

        # 2. Atomic disk serialization
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{key}.ast"
            tmp_file = self.cache_dir / f"{key}.{os.getpid()}.{time.time_ns()}.tmp"
            with open(tmp_file, "wb") as f:
                pickle.dump(ast, f, protocol=5)
            os.replace(tmp_file, cache_file)
        except Exception:
            if "tmp_file" in locals() and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    @staticmethod
    def _clean_directory(target_dir: Path) -> int:
        if not target_dir.exists():
            return 0
        deleted_count = 0
        for item in list(target_dir.glob("*.ast")):
            try:
                item.unlink()
                deleted_count += 1
            except OSError:
                pass
        for item in list(target_dir.glob("*.tmp")):
            try:
                item.unlink()
            except OSError:
                pass
        return deleted_count

    def clean(self_or_dir: Any = ".syn_cache", cache_dir: Optional[Union[str, Path]] = None) -> int:
        """
        Cleans cached .ast and .tmp files from cache directory.
        Can be called on instance (`cache.clean()`) or directly on class (`ASTCache.clean(".syn_cache")`).
        Returns the number of deleted files.
        """
        if isinstance(self_or_dir, ASTCache):
            target = Path(cache_dir) if cache_dir is not None else self_or_dir.cache_dir
            self_or_dir._memory_cache.clear()
        else:
            target = Path(cache_dir) if cache_dir is not None else Path(self_or_dir)
        return ASTCache._clean_directory(target)


def clean_cache(cache_dir: Union[str, Path] = ".syn_cache") -> int:
    """Convenience function to clean the AST cache directory."""
    return ASTCache.clean(cache_dir)


clean = clean_cache
