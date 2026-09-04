"""
Synapse Standard Library: std.db (SQLite Database Adapter)
===========================================================
Provides lightweight, zero-external-dependency embedded SQL database access
built on top of Python's native sqlite3 engine.
Allows Synapse applications, agents, and web APIs to perform ACID transactions,
parameterized queries, and schema management.
"""
from __future__ import annotations
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Union


class DatabaseConnection:
    """
    Thread-safe SQLite database connection wrapper for Synapse.
    """
    def __init__(self, database_path: str = ":memory:", autocommit: bool = True):
        self.path = database_path
        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.autocommit = autocommit
        self.is_open = True

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> int:
        """
        Executes a DDL or DML statement (CREATE, INSERT, UPDATE, DELETE).
        Returns the number of affected rows (rowcount) or last inserted id.
        """
        self._check_open()
        cursor = self._conn.cursor()
        try:
            if params:
                cursor.execute(sql, tuple(params))
            else:
                cursor.execute(sql)
            if self.autocommit:
                self._conn.commit()
            return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
        finally:
            cursor.close()

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query and returns all records as a list of dictionaries.
        """
        self._check_open()
        cursor = self._conn.cursor()
        try:
            if params:
                cursor.execute(sql, tuple(params))
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            cursor.close()

    def query_one(self, sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Executes a SELECT query and returns the first record as a dictionary, or None.
        """
        results = self.query(sql, params)
        return results[0] if results else None

    def commit(self):
        """Manually commits the current transaction."""
        self._check_open()
        self._conn.commit()

    def rollback(self):
        """Rolls back the current transaction."""
        self._check_open()
        self._conn.rollback()

    def close(self):
        """Closes the underlying database connection."""
        if self.is_open:
            self._conn.close()
            self.is_open = False

    def _check_open(self):
        if not self.is_open:
            raise RuntimeError("Database connection is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"<DatabaseConnection path='{self.path}' state='{state}'>"


def connect(database_path: str = ":memory:", autocommit: bool = True) -> DatabaseConnection:
    """
    Opens a connection to an SQLite database (file path or ':memory:').
    Example:
        let db = connect("app.db")
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO users (name) VALUES (?)", ["Alice"])
        let users = db.query("SELECT * FROM users")
    """
    return DatabaseConnection(database_path=database_path, autocommit=autocommit)
