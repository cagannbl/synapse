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

PACKAGE_NAME = "synapse-web"
IS_UNBUNDLED = True

__all__ = [
    "PACKAGE_NAME",
    "IS_UNBUNDLED",
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

