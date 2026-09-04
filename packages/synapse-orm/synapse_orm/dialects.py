"""
Synapse Multi-Dialect SQL Generator & Connection Parser
======================================================
Provides database dialect abstractions, SQL query building, and URL parsing
for SQLite, PostgreSQL, and MySQL.
"""

from abc import ABC, abstractmethod
import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import urllib.parse


# =============================================================================
# 1. Base Dialect Abstract Base Class (ABC)
# =============================================================================

class Dialect(ABC):
    """Abstract base class defining SQL dialect behavior and query building."""

    name: str = "base"

    @abstractmethod
    def param_placeholder(self, idx: int = 1) -> str:
        """
        Returns the parameter placeholder string for this dialect.
        idx is 1-based (used by dialects like PostgreSQL: $1, $2).
        """
        pass

    @abstractmethod
    def quote_identifier(self, name: str) -> str:
        """
        Quotes a database identifier (table or column name).
        Handles compound identifiers like 'table.column' and preserves '*'.
        """
        pass

    @abstractmethod
    def type_to_sql(
        self,
        field_type: Any,
        primary_key: bool = False,
        auto_increment: Optional[bool] = None,
        nullable: bool = True,
        default: Any = None,
        unique: bool = False
    ) -> str:
        """
        Maps a Python type and constraints into a dialect-specific DDL column type.
        """
        pass

    def build_insert(
        self,
        table: str,
        columns: List[str],
        returning: Optional[str] = None
    ) -> str:
        """Builds a dialect-compliant INSERT statement."""
        quoted_table = self.quote_identifier(table)
        if not columns:
            sql = f"INSERT INTO {quoted_table} DEFAULT VALUES"
        else:
            quoted_cols = [self.quote_identifier(c) for c in columns]
            placeholders = [self.param_placeholder(i) for i in range(1, len(columns) + 1)]
            sql = f"INSERT INTO {quoted_table} ({', '.join(quoted_cols)}) VALUES ({', '.join(placeholders)})"

        if returning:
            sql += f" RETURNING {self.quote_identifier(returning)}"
        return sql

    def build_update(
        self,
        table: str,
        columns: List[str],
        where_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        start_idx: int = 1
    ) -> str:
        """
        Builds a dialect-compliant UPDATE statement.
        start_idx tracks parameter indexing (crucial for PostgreSQL $1, $2...).
        """
        if not columns:
            raise ValueError("Cannot build UPDATE statement without columns to set")

        quoted_table = self.quote_identifier(table)
        set_clauses: List[str] = []
        curr_idx = start_idx

        for col in columns:
            set_clauses.append(f"{self.quote_identifier(col)} = {self.param_placeholder(curr_idx)}")
            curr_idx += 1

        sql = f"UPDATE {quoted_table} SET {', '.join(set_clauses)}"

        where_parts: List[str] = []
        if where_columns:
            for col in where_columns:
                where_parts.append(f"{self.quote_identifier(col)} = {self.param_placeholder(curr_idx)}")
                curr_idx += 1
        if where_clause:
            where_parts.append(where_clause)

        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"

        return sql

    def _build_order_by(self, order_by: Optional[Union[str, List[str], Tuple[str, ...]]]) -> str:
        """Helper to build a dialect-quoted ORDER BY clause for single or multiple columns."""
        if not order_by:
            return ""
        if isinstance(order_by, str):
            items = [item.strip() for item in order_by.split(",") if item.strip()]
        elif isinstance(order_by, (list, tuple)):
            items = [str(item).strip() for item in order_by if str(item).strip()]
        else:
            return ""

        formatted_items: List[str] = []
        for item in items:
            if item.startswith("-"):
                formatted_items.append(f"{self.quote_identifier(item[1:].strip())} DESC")
            elif item.startswith("+"):
                formatted_items.append(f"{self.quote_identifier(item[1:].strip())} ASC")
            elif item.upper().endswith(" DESC"):
                col = item[:-5].strip()
                formatted_items.append(f"{self.quote_identifier(col)} DESC")
            elif item.upper().endswith(" ASC"):
                col = item[:-4].strip()
                formatted_items.append(f"{self.quote_identifier(col)} ASC")
            else:
                formatted_items.append(f"{self.quote_identifier(item)} ASC")

        if formatted_items:
            return f" ORDER BY {', '.join(formatted_items)}"
        return ""

    def _build_limit_offset(self, limit: Optional[int], offset: Optional[int]) -> str:
        """Helper to build LIMIT / OFFSET clauses according to dialect rules."""
        parts = ""
        if limit is not None:
            parts += f" LIMIT {int(limit)}"
        if offset is not None:
            parts += f" OFFSET {int(offset)}"
        return parts

    def build_select(
        self,
        table: str,
        columns: Optional[Union[List[str], str, Tuple[str, ...]]] = None,
        where_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        order_by: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        start_idx: int = 1
    ) -> str:
        """Builds a dialect-compliant SELECT statement."""
        quoted_table = self.quote_identifier(table)

        if columns is None or columns == "*":
            cols_str = "*"
        elif isinstance(columns, str):
            cols_str = columns
        else:
            cols_list = [self.quote_identifier(c) for c in columns]
            cols_str = ", ".join(cols_list) if cols_list else "*"

        sql = f"SELECT {cols_str} FROM {quoted_table}"

        curr_idx = start_idx
        where_parts: List[str] = []
        if where_columns:
            for col in where_columns:
                where_parts.append(f"{self.quote_identifier(col)} = {self.param_placeholder(curr_idx)}")
                curr_idx += 1
        if where_clause:
            where_parts.append(where_clause)

        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"

        sql += self._build_order_by(order_by)
        sql += self._build_limit_offset(limit, offset)

        return sql

    def build_delete(
        self,
        table: str,
        where_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        start_idx: int = 1
    ) -> str:
        """Builds a dialect-compliant DELETE statement."""
        quoted_table = self.quote_identifier(table)
        sql = f"DELETE FROM {quoted_table}"

        where_parts: List[str] = []
        curr_idx = start_idx
        if where_columns:
            for col in where_columns:
                where_parts.append(f"{self.quote_identifier(col)} = {self.param_placeholder(curr_idx)}")
                curr_idx += 1
        if where_clause:
            where_parts.append(where_clause)

        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"

        return sql

    def build_create_table(
        self,
        table_name: str,
        columns: Union[List[str], Dict[str, Any]],
        if_not_exists: bool = True
    ) -> str:
        """Builds a dialect-compliant CREATE TABLE statement."""
        quoted_table = self.quote_identifier(table_name)
        prefix = "CREATE TABLE IF NOT EXISTS" if if_not_exists else "CREATE TABLE"

        col_defs: List[str] = []
        if isinstance(columns, list):
            col_defs = list(columns)
        elif isinstance(columns, dict):
            for col_name, col_info in columns.items():
                quoted_col = self.quote_identifier(col_name)
                if isinstance(col_info, str):
                    col_defs.append(f"{quoted_col} {col_info}")
                elif hasattr(col_info, "type"):  # Field instance
                    auto_inc = getattr(col_info, "auto_increment", None)
                    if auto_inc is None:
                        auto_inc = getattr(col_info, "primary_key", False) and getattr(col_info, "type", None) is int
                    type_str = self.type_to_sql(
                        field_type=col_info.type,
                        primary_key=getattr(col_info, "primary_key", False),
                        auto_increment=auto_inc,
                        nullable=getattr(col_info, "nullable", True),
                        default=getattr(col_info, "default", None),
                        unique=getattr(col_info, "unique", False)
                    )
                    col_defs.append(f"{quoted_col} {type_str}")
                elif isinstance(col_info, dict):
                    auto_inc = col_info.get("auto_increment")
                    if auto_inc is None:
                        auto_inc = col_info.get("primary_key", False) and col_info.get("type", None) is int
                    type_str = self.type_to_sql(
                        field_type=col_info.get("type", str),
                        primary_key=col_info.get("primary_key", False),
                        auto_increment=auto_inc,
                        nullable=col_info.get("nullable", True),
                        default=col_info.get("default", None),
                        unique=col_info.get("unique", False)
                    )
                    col_defs.append(f"{quoted_col} {type_str}")
                else:
                    col_defs.append(f"{quoted_col} {col_info}")

        return f"{prefix} {quoted_table} ({', '.join(col_defs)})"

    def build_drop_table(self, table_name: str, if_exists: bool = True) -> str:
        """Builds a dialect-compliant DROP TABLE statement."""
        prefix = "DROP TABLE IF EXISTS" if if_exists else "DROP TABLE"
        return f"{prefix} {self.quote_identifier(table_name)}"


# =============================================================================
# 2. Concrete Dialect Implementations
# =============================================================================

class SQLiteDialect(Dialect):
    """SQLite dialect using ? placeholders, double quotes, and AUTOINCREMENT."""

    name = "sqlite"

    TYPE_MAP: Dict[Any, str] = {
        int: "INTEGER",
        str: "TEXT",
        float: "REAL",
        bool: "INTEGER",
        bytes: "BLOB",
        datetime.datetime: "TIMESTAMP",
        datetime.date: "DATE",
        dict: "TEXT",
        list: "TEXT",
    }

    def param_placeholder(self, idx: int = 1) -> str:
        return "?"

    def quote_identifier(self, name: str) -> str:
        if name == "*":
            return "*"
        if "." in name:
            return ".".join(self.quote_identifier(part) for part in name.split("."))
        clean = name.strip('"\'`')
        clean = clean.replace('"', '""')
        return f'"{clean}"'

    def _build_limit_offset(self, limit: Optional[int], offset: Optional[int]) -> str:
        """
        SQLite requires a LIMIT clause if OFFSET is specified.
        If OFFSET is specified without LIMIT, SQLite uses LIMIT -1.
        """
        if limit is not None and offset is not None:
            return f" LIMIT {int(limit)} OFFSET {int(offset)}"
        elif limit is not None:
            return f" LIMIT {int(limit)}"
        elif offset is not None:
            return f" LIMIT -1 OFFSET {int(offset)}"
        return ""

    def type_to_sql(
        self,
        field_type: Any,
        primary_key: bool = False,
        auto_increment: Optional[bool] = None,
        nullable: bool = True,
        default: Any = None,
        unique: bool = False
    ) -> str:
        sql_type = self.TYPE_MAP.get(
            field_type,
            str(field_type) if isinstance(field_type, str) else "TEXT"
        )

        if primary_key:
            is_int = (field_type is int or sql_type.upper() in ("INTEGER", "INT"))
            is_auto = is_int if auto_increment is None else auto_increment
            if is_auto:
                return "INTEGER PRIMARY KEY AUTOINCREMENT"
            return f"{sql_type} PRIMARY KEY"

        parts = [sql_type]
        if not nullable:
            parts.append("NOT NULL")
        if unique:
            parts.append("UNIQUE")
        if default is not None and not callable(default):
            if isinstance(default, bool):
                parts.append(f"DEFAULT {1 if default else 0}")
            elif isinstance(default, (int, float)):
                parts.append(f"DEFAULT {default}")
            else:
                escaped = str(default).replace("'", "''")
                parts.append(f"DEFAULT '{escaped}'")

        return " ".join(parts)


class PostgresDialect(Dialect):
    """PostgreSQL dialect using $1, $2 placeholders, double quotes, and SERIAL PRIMARY KEY."""

    name = "postgresql"

    TYPE_MAP: Dict[Any, str] = {
        int: "INTEGER",
        str: "TEXT",
        float: "DOUBLE PRECISION",
        bool: "BOOLEAN",
        bytes: "BYTEA",
        datetime.datetime: "TIMESTAMP",
        datetime.date: "DATE",
        dict: "JSONB",
        list: "JSONB",
    }

    def param_placeholder(self, idx: int = 1) -> str:
        return f"${max(1, idx)}"

    def quote_identifier(self, name: str) -> str:
        if name == "*":
            return "*"
        if "." in name:
            return ".".join(self.quote_identifier(part) for part in name.split("."))
        clean = name.strip('"\'`')
        clean = clean.replace('"', '""')
        return f'"{clean}"'

    def _build_limit_offset(self, limit: Optional[int], offset: Optional[int]) -> str:
        """PostgreSQL supports OFFSET with or without LIMIT."""
        parts = ""
        if limit is not None:
            parts += f" LIMIT {int(limit)}"
        if offset is not None:
            parts += f" OFFSET {int(offset)}"
        return parts

    def type_to_sql(
        self,
        field_type: Any,
        primary_key: bool = False,
        auto_increment: Optional[bool] = None,
        nullable: bool = True,
        default: Any = None,
        unique: bool = False
    ) -> str:
        sql_type = self.TYPE_MAP.get(
            field_type,
            str(field_type) if isinstance(field_type, str) else "TEXT"
        )

        if primary_key:
            is_int = (field_type is int or sql_type.upper() in ("INTEGER", "INT", "SERIAL"))
            is_auto = is_int if auto_increment is None else auto_increment
            if is_auto or sql_type.upper() == "SERIAL":
                return "SERIAL PRIMARY KEY"
            return f"{sql_type} PRIMARY KEY"

        parts = [sql_type]
        if not nullable:
            parts.append("NOT NULL")
        if unique:
            parts.append("UNIQUE")
        if default is not None and not callable(default):
            if isinstance(default, bool):
                parts.append(f"DEFAULT {'TRUE' if default else 'FALSE'}")
            elif isinstance(default, (int, float)):
                parts.append(f"DEFAULT {default}")
            else:
                escaped = str(default).replace("'", "''")
                parts.append(f"DEFAULT '{escaped}'")

        return " ".join(parts)


class MySQLDialect(Dialect):
    """MySQL dialect using %s placeholders, backtick quotes, and AUTO_INCREMENT."""

    name = "mysql"

    TYPE_MAP: Dict[Any, str] = {
        int: "INT",
        str: "VARCHAR(255)",
        float: "DOUBLE",
        bool: "TINYINT(1)",
        bytes: "BLOB",
        datetime.datetime: "DATETIME",
        datetime.date: "DATE",
        dict: "JSON",
        list: "JSON",
    }

    def param_placeholder(self, idx: int = 1) -> str:
        return "%s"

    def quote_identifier(self, name: str) -> str:
        if name == "*":
            return "*"
        if "." in name:
            return ".".join(self.quote_identifier(part) for part in name.split("."))
        clean = name.strip('"\'`')
        clean = clean.replace('`', '``')
        return f"`{clean}`"

    def _build_limit_offset(self, limit: Optional[int], offset: Optional[int]) -> str:
        """
        MySQL requires a LIMIT clause if OFFSET is specified.
        If OFFSET is specified without LIMIT, standard MySQL uses 18446744073709551615.
        """
        if limit is not None and offset is not None:
            return f" LIMIT {int(limit)} OFFSET {int(offset)}"
        elif limit is not None:
            return f" LIMIT {int(limit)}"
        elif offset is not None:
            return f" LIMIT 18446744073709551615 OFFSET {int(offset)}"
        return ""

    def type_to_sql(
        self,
        field_type: Any,
        primary_key: bool = False,
        auto_increment: Optional[bool] = None,
        nullable: bool = True,
        default: Any = None,
        unique: bool = False
    ) -> str:
        sql_type = self.TYPE_MAP.get(
            field_type,
            str(field_type) if isinstance(field_type, str) else "VARCHAR(255)"
        )

        if primary_key:
            is_int = (field_type is int or sql_type.upper() in ("INT", "INTEGER"))
            is_auto = is_int if auto_increment is None else auto_increment
            if is_auto:
                return "INT AUTO_INCREMENT PRIMARY KEY"
            return f"{sql_type} PRIMARY KEY"

        parts = [sql_type]
        if not nullable:
            parts.append("NOT NULL")
        if unique:
            parts.append("UNIQUE")
        if default is not None and not callable(default):
            if isinstance(default, bool):
                parts.append(f"DEFAULT {1 if default else 0}")
            elif isinstance(default, (int, float)):
                parts.append(f"DEFAULT {default}")
            else:
                escaped = str(default).replace("'", "''")
                parts.append(f"DEFAULT '{escaped}'")

        return " ".join(parts)

    def build_insert(
        self,
        table: str,
        columns: List[str],
        returning: Optional[str] = None
    ) -> str:
        """MySQL does not support standard RETURNING or DEFAULT VALUES clause."""
        quoted_table = self.quote_identifier(table)
        if not columns:
            sql = f"INSERT INTO {quoted_table} () VALUES ()"
        else:
            quoted_cols = [self.quote_identifier(c) for c in columns]
            placeholders = [self.param_placeholder(i) for i in range(1, len(columns) + 1)]
            sql = f"INSERT INTO {quoted_table} ({', '.join(quoted_cols)}) VALUES ({', '.join(placeholders)})"
        return sql


# =============================================================================
# 3. Dialect Registry & Factory
# =============================================================================

_DIALECT_REGISTRY: Dict[str, Type[Dialect]] = {
    "sqlite": SQLiteDialect,
    "sqlite3": SQLiteDialect,
    "postgres": PostgresDialect,
    "postgresql": PostgresDialect,
    "mysql": MySQLDialect,
    "mariadb": MySQLDialect,
}


def get_dialect(dialect_or_name: Union[str, Dialect]) -> Dialect:
    """Returns a Dialect instance given a dialect name string or Dialect instance."""
    if isinstance(dialect_or_name, Dialect):
        return dialect_or_name
    if isinstance(dialect_or_name, str):
        key = dialect_or_name.lower().strip()
        if key in _DIALECT_REGISTRY:
            return _DIALECT_REGISTRY[key]()
        raise ValueError(
            f"Unsupported SQL dialect: '{dialect_or_name}'. "
            f"Supported dialects: {list(_DIALECT_REGISTRY.keys())}"
        )
    raise TypeError(f"Expected str or Dialect, got {type(dialect_or_name).__name__}")


# =============================================================================
# 4. Connection URL Parser
# =============================================================================

def parse_connection_url(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parses a database connection URL or path into a tuple of:
    (dialect_name: str, params: Dict[str, Any]).

    Supported formats:
      - sqlite:///:memory:
      - sqlite:///path/to/db.sqlite
      - :memory:
      - path/to/db.sqlite (.db, .sqlite, .sqlite3)
      - postgresql://user:pass@host:port/dbname
      - postgres://user:pass@host/dbname
      - mysql://user:pass@host:port/dbname
      - mysql+pymysql://user:pass@host:port/dbname
      - mariadb://user:pass@host:port/dbname
    """
    if not url:
        return "sqlite", {"database": ":memory:", "path": ":memory:"}

    # Handle bare SQLite paths and :memory:
    if "://" not in url:
        return "sqlite", {"database": url, "path": url}

    # Handle sqlite:/// URLs directly
    if url.startswith("sqlite:///"):
        sub = url[10:]
        path, _, query_str = sub.partition("?")
        if not path or path == ":memory:":
            path = ":memory:"
        elif path.startswith("/") and len(path) > 2 and path[2] == ":":
            # Windows drive path with leading slash, e.g. /C:/path
            path = path[1:]
        params: Dict[str, Any] = {"database": path, "path": path}
        if query_str:
            q_dict = urllib.parse.parse_qs(query_str)
            flat_q = {k: v[0] if len(v) == 1 else v for k, v in q_dict.items()}
            params["options"] = flat_q
            for k, v in flat_q.items():
                if k not in params:
                    params[k] = v
        return "sqlite", params

    if url.startswith("sqlite://"):
        sub = url[9:]
        path, _, query_str = sub.partition("?")
        if not path or path == ":memory:":
            path = ":memory:"
        params = {"database": path, "path": path}
        if query_str:
            q_dict = urllib.parse.parse_qs(query_str)
            flat_q = {k: v[0] if len(v) == 1 else v for k, v in q_dict.items()}
            params["options"] = flat_q
            for k, v in flat_q.items():
                if k not in params:
                    params[k] = v
        return "sqlite", params

    # Use urllib.parse for network-based URLs (postgresql, mysql, mariadb)
    parsed = urllib.parse.urlparse(url)
    raw_scheme = parsed.scheme.lower()

    if (
        raw_scheme in ("postgres", "postgresql")
        or raw_scheme.startswith("postgresql+")
        or raw_scheme.startswith("postgres+")
    ):
        dialect = "postgresql"
        default_port = 5432
    elif (
        raw_scheme in ("mysql", "mariadb")
        or raw_scheme.startswith("mysql+")
        or raw_scheme.startswith("mariadb+")
    ):
        dialect = "mysql"
        default_port = 3306
    elif raw_scheme in ("sqlite",) or raw_scheme.startswith("sqlite+"):
        dialect = "sqlite"
        default_port = None
    else:
        dialect = raw_scheme
        default_port = None

    if dialect == "sqlite":
        path = parsed.path.lstrip("/")
        if not path or path == ":memory:":
            path = ":memory:"
        params = {"database": path, "path": path}
        if parsed.query:
            query_dict = urllib.parse.parse_qs(parsed.query)
            flat_query = {k: v[0] if len(v) == 1 else v for k, v in query_dict.items()}
            params["options"] = flat_query
            for k, v in flat_query.items():
                if k not in params:
                    params[k] = v
        return "sqlite", params

    params = {
        "user": parsed.username or "",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port if parsed.port is not None else default_port,
        "database": parsed.path.strip("/"),
    }

    # Extract query parameters (e.g., sslmode=require, charset=utf8mb4)
    if parsed.query:
        query_dict = urllib.parse.parse_qs(parsed.query)
        flat_query = {k: v[0] if len(v) == 1 else v for k, v in query_dict.items()}
        params["options"] = flat_query
        for k, v in flat_query.items():
            if k not in params:
                params[k] = v

    return dialect, params
