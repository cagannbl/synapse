"""Synapse Built-in Standard Library: File System & Path Operations Module."""

import os
import shutil
from typing import Any, List, Optional, Tuple, Union

# =============================================================================
# File I/O Operations
# =============================================================================

def read_file(path: str, mode: str = "r") -> Union[str, bytes]:
    """Reads content from a file. Returns str for text modes and bytes for binary modes."""
    encoding = None if "b" in mode else "utf-8"
    with open(path, mode, encoding=encoding) as f:
        return f.read()


def write_file(path: str, content: Union[str, bytes, Any], mode: str = "w") -> int:
    """Writes content to a file. Returns number of characters or bytes written."""
    if "b" in mode:
        data = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8")
        with open(path, mode) as f:
            return f.write(data)
    else:
        text = content if isinstance(content, str) else str(content)
        with open(path, mode, encoding="utf-8") as f:
            return f.write(text)


def append_file(path: str, content: Union[str, Any]) -> int:
    """Appends text content to a file (UTF-8). Returns number of characters written."""
    text = content if isinstance(content, str) else str(content)
    with open(path, "a", encoding="utf-8") as f:
        return f.write(text)


def file_exists(path: str) -> bool:
    """Checks whether a regular file exists at the specified path."""
    return os.path.isfile(path)


def dir_exists(path: str) -> bool:
    """Checks whether a directory exists at the specified path."""
    return os.path.isdir(path)


# =============================================================================
# Directory & Path Operations
# =============================================================================

def mkdir(path: str, recursive: bool = True) -> bool:
    """Creates a new directory. If recursive is True, creates intermediate parent directories."""
    if recursive:
        os.makedirs(path, exist_ok=True)
    else:
        os.mkdir(path)
    return True


def rmdir(path: str, recursive: bool = False) -> bool:
    """Removes a directory. If recursive is True, removes directory and all its contents."""
    if recursive:
        shutil.rmtree(path)
    else:
        os.rmdir(path)
    return True


def remove_file(path: str) -> bool:
    """Deletes the specified file."""
    os.remove(path)
    return True


def list_dir(path: str = ".") -> List[str]:
    """Returns a list of entry names in the given directory."""
    return os.listdir(path)


def walk(path: str) -> List[Tuple[str, List[str], List[str]]]:
    """Walks the directory tree recursively, returning a list of (dirpath, dirnames, filenames)."""
    return list(os.walk(path))


def join_path(*parts: str) -> str:
    """Joins path components intelligently using the platform separator."""
    return os.path.join(*parts)


def base_name(path: str) -> str:
    """Returns the base filename or trailing component of a path."""
    return os.path.basename(path)


def dir_name(path: str) -> str:
    """Returns the directory component of a path."""
    return os.path.dirname(path)


def ext_name(path: str) -> str:
    """Returns the file extension (including the leading dot, e.g. '.txt')."""
    return os.path.splitext(path)[1]


def file_size(path: str) -> int:
    """Returns the size of a file in bytes."""
    return os.path.getsize(path)


__all__ = [
    "read_file",
    "write_file",
    "append_file",
    "file_exists",
    "dir_exists",
    "mkdir",
    "rmdir",
    "remove_file",
    "list_dir",
    "walk",
    "join_path",
    "base_name",
    "dir_name",
    "ext_name",
    "file_size",
]
