"""
Synapse Enterprise Web Framework
================================
Modern, type-safe, dependency-injected full-stack web framework.
"""

from .enterprise import (
    SynapseApp,
    Request,
    Response,
    Depends,
    HTTPException,
    create_access_token,
    decode_access_token,
    jwt_required,
    JWTError,
    JWTExpiredError,
)

__all__ = [
    "SynapseApp",
    "Request",
    "Response",
    "Depends",
    "HTTPException",
    "create_access_token",
    "decode_access_token",
    "jwt_required",
    "JWTError",
    "JWTExpiredError",
]
