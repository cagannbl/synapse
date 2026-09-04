"""Synapse Built-in Standard Library: Cryptography, Hashing, and Encodings Module."""

import base64 as _base64
import hashlib as _hashlib
import hmac as _hmac
import secrets as _secrets
from typing import Any, Optional, Union
import uuid as _uuid

# =============================================================================
# Cryptographic Hash Functions
# =============================================================================

def _to_bytes(data: Union[str, bytes, bytearray, Any]) -> bytes:
    """Helper to convert input strings or objects into raw bytes."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return str(data).encode("utf-8")


def sha256(data: Union[str, bytes, Any]) -> str:
    """Computes SHA-256 hash digest in hexadecimal representation."""
    return _hashlib.sha256(_to_bytes(data)).hexdigest()


def sha512(data: Union[str, bytes, Any]) -> str:
    """Computes SHA-512 hash digest in hexadecimal representation."""
    return _hashlib.sha512(_to_bytes(data)).hexdigest()


def md5(data: Union[str, bytes, Any]) -> str:
    """Computes MD5 hash digest in hexadecimal representation."""
    return _hashlib.md5(_to_bytes(data)).hexdigest()


def hmac(key: Union[str, bytes, Any], data: Union[str, bytes, Any], algo: str = "sha256") -> str:
    """Computes HMAC digest in hexadecimal representation using the specified algorithm."""
    k = _to_bytes(key)
    d = _to_bytes(data)
    algo_name = algo.lower().replace("-", "")
    digestmod = getattr(_hashlib, algo_name, None)
    if digestmod is None:
        raise ValueError(f"Unsupported hash algorithm '{algo}' for HMAC")
    return _hmac.new(k, d, digestmod).hexdigest()


# =============================================================================
# Encodings
# =============================================================================

def base64_encode(data: Union[str, bytes, Any]) -> str:
    """Encodes string or bytes into a Base64 string."""
    raw = _to_bytes(data)
    return _base64.b64encode(raw).decode("utf-8")


def base64_decode(data: Union[str, bytes], as_str: bool = True) -> Union[str, bytes]:
    """Decodes a Base64 string or bytes into decoded text or raw bytes."""
    raw = _to_bytes(data)
    decoded = _base64.b64decode(raw)
    if as_str:
        try:
            return decoded.decode("utf-8")
        except UnicodeDecodeError:
            return decoded
    return decoded


# =============================================================================
# Security & Identification
# =============================================================================

def random_token(nbytes: int = 16) -> str:
    """Generates a cryptographically secure random hexadecimal token."""
    return _secrets.token_hex(nbytes)


def uuid4() -> str:
    """Generates a random UUID (version 4) as a string."""
    return str(_uuid.uuid4())


__all__ = [
    "sha256",
    "sha512",
    "md5",
    "hmac",
    "base64_encode",
    "base64_decode",
    "random_token",
    "uuid4",
]
