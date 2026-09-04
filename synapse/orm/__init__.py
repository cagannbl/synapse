"""
Synapse Type-Safe Active-Record ORM & Multi-Dialect Engine
==========================================================
"""

from .dialects import (
    Dialect,
    SQLiteDialect,
    PostgresDialect,
    MySQLDialect,
    get_dialect,
    parse_connection_url,
)
from .models import Database, Field, Model, QuerySet

PACKAGE_NAME = "synapse-orm"
IS_UNBUNDLED = True

__all__ = [
    "PACKAGE_NAME",
    "IS_UNBUNDLED",
    "Database",
    "Field",
    "Model",
    "QuerySet",
    "Dialect",
    "SQLiteDialect",
    "PostgresDialect",
    "MySQLDialect",
    "get_dialect",
    "parse_connection_url",
]

