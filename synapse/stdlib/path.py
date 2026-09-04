"""
Synapse Standard Library: std.path
==================================
Provides cross-platform path manipulation and filesystem inspection utilities
with standard POSIX / Windows normalization.
"""
from __future__ import annotations
import os
import posixpath
import ntpath
from typing import List, Optional, Tuple


def join(*parts: str) -> str:
    """Joins path components using the native platform separator."""
    return os.path.join(*parts)


def split(path: str) -> Tuple[str, str]:
    """Splits path into (dirname, basename)."""
    return os.path.split(path)


def dirname(path: str) -> str:
    """Returns directory path component."""
    return os.path.dirname(path)


def basename(path: str) -> str:
    """Returns filename component."""
    return os.path.basename(path)


def ext(path: str) -> str:
    """Returns file extension including the leading dot, e.g. '.syn'."""
    return os.path.splitext(path)[1]


def stem(path: str) -> str:
    """Returns filename without directory or extension."""
    return os.path.splitext(os.path.basename(path))[0]


def abspath(path: str) -> str:
    """Returns normalized absolute path."""
    return os.path.abspath(path)


def exists(path: str) -> bool:
    """Returns True if path points to an existing file or directory."""
    return os.path.exists(path)


def is_file(path: str) -> bool:
    """Returns True if path points to an existing regular file."""
    return os.path.isfile(path)


def is_dir(path: str) -> bool:
    """Returns True if path points to an existing directory."""
    return os.path.isdir(path)


def normalize(path: str) -> str:
    """Normalizes path separators and collapses redundant segments."""
    return os.path.normpath(path)
