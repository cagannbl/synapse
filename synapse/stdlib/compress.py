"""
Synapse Standard Library: std.compress
======================================
Provides gzip, zlib, and base64 compression / decompression utilities
with zero external dependencies.
"""
from __future__ import annotations
import base64
import gzip as _gzip_module
import zlib
from typing import Union


def gzip_compress(data: Union[str, bytes], level: int = 9) -> bytes:
    """Compresses string or bytes using gzip format."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return _gzip_module.compress(raw, compresslevel=level)


def gzip_decompress(data: bytes, as_text: bool = True) -> Union[str, bytes]:
    """Decompresses gzip compressed bytes."""
    decomp = _gzip_module.decompress(data)
    if as_text:
        try:
            return decomp.decode("utf-8")
        except UnicodeDecodeError:
            return decomp
    return decomp


def zlib_compress(data: Union[str, bytes], level: int = 6) -> bytes:
    """Compresses string or bytes using zlib format."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return zlib.compress(raw, level=level)


def zlib_decompress(data: bytes, as_text: bool = True) -> Union[str, bytes]:
    """Decompresses zlib compressed bytes."""
    decomp = zlib.decompress(data)
    if as_text:
        try:
            return decomp.decode("utf-8")
        except UnicodeDecodeError:
            return decomp
    return decomp


def b64_encode(data: Union[str, bytes]) -> str:
    """Encodes bytes or string to Base64 string."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return base64.b64encode(raw).decode("ascii")


def b64_decode(data: str, as_text: bool = True) -> Union[str, bytes]:
    """Decodes Base64 string back to string or bytes."""
    decoded = base64.b64decode(data.encode("ascii"))
    if as_text:
        try:
            return decoded.decode("utf-8")
        except UnicodeDecodeError:
            return decoded
    return decoded
