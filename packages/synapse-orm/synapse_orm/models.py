"""
Synapse Type-Safe Active-Record ORM & Multi-Dialect Engine
==========================================================
Zero-dependency Active-Record ORM featuring:
- Multi-dialect support: SQLite, PostgreSQL, MySQL
- Database connection manager with thread safety and transaction handling
- Field definitions with automatic multi-dialect type mapping
- Active-Record Model base: create(), filter().all(), filter().first(), get(id), save(), delete()
- Parameterized queries (? for SQLite, $1 for Postgres, %s for MySQL) ensuring 100% SQL injection resistance
- Safe schema generation with Database.create_all()
"""

import inspect
import re
import sqlite3
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

from .dialects import (
    Dialect,
    SQLiteDialect,
    PostgresDialect,
    MySQLDialect,
    get_dialect,
    parse_connection_url,
)


# =============================================================================
# 1. Database Connection & Schema Manager
# =============================================================================

class Database:
    """
    Thread-safe database connection manager, dialect controller, and transaction manager.
    Supports SQLite directly and manages query dialect compilation for Postgres & MySQL.
    """
    _default_instance: Optional["Database"] = None
    _registered_models: List[Type["Model"]] = []

    def __init__(
        self,
        connection_string_or_path: str = ":memory:",
        dialect: Optional[Union[str, Dialect]] = None,
        **kwargs
    ):
        # Support connection_string as keyword argument for full backward compatibility
        conn_str = kwargs.get("connection_string", connection_string_or_path)
        self.connection_string = conn_str

        # Auto-detect dialect and parse connection parameters
        detected_dialect_name, conn_params = parse_connection_url(conn_str)
        self.connection_params = conn_params

        # Explicit dialect overrides auto-detected dialect
        if dialect is not None:
            self.dialect: Dialect = get_dialect(dialect)
        else:
            self.dialect: Dialect = get_dialect(detected_dialect_name)

        self.db_path = conn_params.get("path", conn_str)
        self._lock = threading.RLock()
        self._in_transaction = False

        # SQLite connection initialization
        if self.dialect.name == "sqlite":
            if self.db_path != ":memory:":
                import os
                parent = os.path.dirname(self.db_path)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
            self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # autocommit mode unless explicit transaction
            )
            self._conn.row_factory = sqlite3.Row
        else:
            self._conn = None

        # Set as default database if none registered yet
        if Database._default_instance is None:
            Database._default_instance = self

    @classmethod
    def get_default(cls) -> "Database":
        """Returns the default Database instance, creating an in-memory one if needed."""
        if cls._default_instance is None:
            cls._default_instance = Database("sqlite:///:memory:")
        return cls._default_instance

    @classmethod
    def set_default(cls, db: "Database"):
        """Sets the active default database."""
        cls._default_instance = db

    @classmethod
    def register_model(cls, model_cls: Type["Model"]):
        """Registers a Model class for automatic schema migration."""
        if model_cls not in cls._registered_models:
            cls._registered_models.append(model_cls)

    def _parse_connection_string(self, conn_str: str) -> str:
        """Extracts database path from connection string for backward compatibility."""
        _, params = parse_connection_url(conn_str)
        return params.get("path", conn_str)

    def execute(self, sql: str, params: Union[Tuple, List] = ()) -> Any:
        """Executes a parameterized SQL query under thread-safe lock."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError(
                    f"Live database connection is not active for dialect '{self.dialect.name}'. "
                    "Database execution requires a running database driver and server."
                )
            cursor = self._conn.cursor()
            cursor.execute(sql, tuple(params))
            return cursor

    def executemany(self, sql: str, seq_of_params: List[Tuple]) -> Any:
        """Executes multiple parameterized SQL queries under thread-safe lock."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError(
                    f"Live database connection is not active for dialect '{self.dialect.name}'. "
                    "Database execution requires a running database driver and server."
                )
            cursor = self._conn.cursor()
            cursor.executemany(sql, seq_of_params)
            return cursor

    def query(self, sql: str, params: Union[Tuple, List] = ()) -> List[Dict[str, Any]]:
        """Executes a SELECT query and returns all records as dictionaries."""
        cursor = self.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: Union[Tuple, List] = ()) -> Optional[Dict[str, Any]]:
        """Executes a SELECT query and returns the first record as a dictionary or None."""
        results = self.query(sql, params)
        return results[0] if results else None

    def transaction(self):
        """Context manager for atomic database transactions."""
        class TransactionContext:
            def __init__(self, db: Database):
                self.db = db

            def __enter__(self):
                self.db._lock.acquire()
                if self.db._conn:
                    self.db._conn.execute("BEGIN")
                self.db._in_transaction = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    if self.db._conn:
                        if exc_type is None:
                            self.db._conn.execute("COMMIT")
                        else:
                            self.db._conn.execute("ROLLBACK")
                finally:
                    self.db._in_transaction = False
                    self.db._lock.release()

        return TransactionContext(self)

    def create_all(self):
        """Creates tables for all registered models if they don't already exist."""
        with self._lock:
            for model_cls in self._registered_models:
                table_name = model_cls._table_name
                column_defs: List[str] = []

                for col_name, field in model_cls._fields.items():
                    column_defs.append(field.get_sql_definition(col_name, dialect=self.dialect))

                sql = self.dialect.build_create_table(table_name, column_defs)
                self.execute(sql)

    def drop_all(self):
        """Drops all registered tables (useful for test tear-downs)."""
        with self._lock:
            for model_cls in reversed(self._registered_models):
                sql = self.dialect.build_drop_table(model_cls._table_name)
                self.execute(sql)

    def close(self):
        """Closes the underlying database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# =============================================================================
# 2. Field Definition & Multi-Dialect Type Mapping
# =============================================================================

class Field:
    """Field definition for Model attributes with multi-dialect type mapping."""

    TYPE_MAP = {
        int: "INTEGER",
        str: "TEXT",
        float: "REAL",
        bool: "INTEGER",
        bytes: "BLOB",
    }

    def __init__(
        self,
        type: type = str,
        primary_key: bool = False,
        nullable: bool = True,
        default: Any = None,
        unique: bool = False,
        auto_increment: Optional[bool] = None,
    ):
        self.type = type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.unique = unique
        self.auto_increment = auto_increment
        self.name: Optional[str] = None

    def get_sql_definition(self, col_name: str, dialect: Optional[Dialect] = None) -> str:
        """Generates the column specification string according to the active dialect."""
        if dialect is None:
            dialect = SQLiteDialect()
        quoted_col = dialect.quote_identifier(col_name)
        auto_inc = self.auto_increment
        if auto_inc is None:
            auto_inc = (self.primary_key and self.type is int)
        type_sql = dialect.type_to_sql(
            field_type=self.type,
            primary_key=self.primary_key,
            auto_increment=auto_inc,
            nullable=self.nullable,
            default=self.default,
            unique=self.unique
        )
        return f"{quoted_col} {type_sql}"


# =============================================================================
# 3. Chainable QuerySet
# =============================================================================

class QuerySet:
    """Chainable, lazy query builder for Model queries supporting multi-dialect compilation."""

    def __init__(
        self,
        model_cls: Type["Model"],
        conditions: Optional[Dict[str, Any]] = None,
        order_by_col: Optional[str] = None,
        limit_val: Optional[int] = None,
        offset_val: Optional[int] = None
    ):
        self.model_cls = model_cls
        self.conditions = dict(conditions or {})
        self._order_by = order_by_col
        self._limit = limit_val
        self._offset = offset_val

        # Validate column names strictly to prevent SQL injection in keys
        for key in self.conditions:
            if key not in self.model_cls._fields:
                raise ValueError(
                    f"Invalid query field: '{key}' on model '{self.model_cls.__name__}'"
                )

    def filter(self, **kwargs) -> "QuerySet":
        """Returns a new QuerySet with additional AND filter conditions."""
        new_conditions = dict(self.conditions)
        new_conditions.update(kwargs)
        return QuerySet(
            self.model_cls,
            conditions=new_conditions,
            order_by_col=self._order_by,
            limit_val=self._limit,
            offset_val=self._offset
        )

    def order_by(self, *columns: str) -> "QuerySet":
        """Orders the query result by the specified column name(s)."""
        if not columns:
            return self
        col_list: List[str] = []
        for c in columns:
            if "," in c:
                col_list.extend(part.strip() for part in c.split(",") if part.strip())
            else:
                col_list.append(c.strip())

        for c in col_list:
            clean = c.lstrip("-+").strip()
            if clean.upper().endswith(" DESC"):
                clean = clean[:-5].strip()
            elif clean.upper().endswith(" ASC"):
                clean = clean[:-4].strip()
            if clean not in self.model_cls._fields:
                raise ValueError(f"Unknown ordering column: '{clean}'")

        combined = ", ".join(col_list)
        return QuerySet(
            self.model_cls,
            conditions=self.conditions,
            order_by_col=combined,
            limit_val=self._limit,
            offset_val=self._offset
        )

    def limit(self, count: int) -> "QuerySet":
        """Limits the maximum number of rows returned."""
        return QuerySet(
            self.model_cls,
            conditions=self.conditions,
            order_by_col=self._order_by,
            limit_val=count,
            offset_val=self._offset
        )

    def offset(self, count: int) -> "QuerySet":
        """Skips the specified number of rows."""
        return QuerySet(
            self.model_cls,
            conditions=self.conditions,
            order_by_col=self._order_by,
            limit_val=self._limit,
            offset_val=count
        )

    def _build_select_sql(self, select_clause: str = "*") -> Tuple[str, Tuple]:
        db = self.model_cls.get_db()
        dialect = db.dialect
        table = self.model_cls._table_name
        where_clauses: List[str] = []
        params: List[Any] = []

        param_idx = 1
        for col, val in self.conditions.items():
            quoted_col = dialect.quote_identifier(col)
            if val is None:
                where_clauses.append(f"{quoted_col} IS NULL")
            else:
                placeholder = dialect.param_placeholder(param_idx)
                where_clauses.append(f"{quoted_col} = {placeholder}")
                params.append(val)
                param_idx += 1

        where_clause_str = " AND ".join(where_clauses) if where_clauses else None
        sql = dialect.build_select(
            table=table,
            columns=select_clause,
            where_clause=where_clause_str,
            order_by=self._order_by,
            limit=self._limit,
            offset=self._offset
        )
        return sql, tuple(params)

    def to_sql(self, select_clause: str = "*") -> Tuple[str, Tuple]:
        """Returns the compiled SELECT SQL query and parameter tuple for inspection."""
        return self._build_select_sql(select_clause)

    def to_delete_sql(self) -> Tuple[str, Tuple]:
        """Returns the compiled DELETE SQL query and parameter tuple for inspection."""
        db = self.model_cls.get_db()
        dialect = db.dialect
        table = self.model_cls._table_name
        where_clauses: List[str] = []
        params: List[Any] = []

        param_idx = 1
        for col, val in self.conditions.items():
            quoted_col = dialect.quote_identifier(col)
            if val is None:
                where_clauses.append(f"{quoted_col} IS NULL")
            else:
                placeholder = dialect.param_placeholder(param_idx)
                where_clauses.append(f"{quoted_col} = {placeholder}")
                params.append(val)
                param_idx += 1

        where_clause_str = " AND ".join(where_clauses) if where_clauses else None
        sql = dialect.build_delete(table=table, where_clause=where_clause_str)
        return sql, tuple(params)

    def all(self) -> List["Model"]:
        """Executes the query and returns a list of Model instances."""
        sql, params = self._build_select_sql()
        cursor = self.model_cls.get_db().execute(sql, params)
        rows = cursor.fetchall()

        results: List["Model"] = []
        for row in rows:
            data = dict(row)
            instance = self.model_cls(**data)
            instance._is_persisted = True
            results.append(instance)
        return results

    def first(self) -> Optional["Model"]:
        """Executes the query and returns the first matching Model instance or None."""
        limited = self.limit(1)
        sql, params = limited._build_select_sql()
        cursor = self.model_cls.get_db().execute(sql, params)
        row = cursor.fetchone()
        if not row:
            return None
        instance = self.model_cls(**dict(row))
        instance._is_persisted = True
        return instance

    def count(self) -> int:
        """Returns the count of matching records."""
        sql, params = self._build_select_sql("COUNT(*)")
        cursor = self.model_cls.get_db().execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else 0

    def delete(self) -> int:
        """Deletes all matching records from the database. Returns affected rows count."""
        db = self.model_cls.get_db()
        sql, params = self.to_delete_sql()
        cursor = db.execute(sql, params)
        return cursor.rowcount

    def __iter__(self) -> Iterator["Model"]:
        return iter(self.all())

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"<QuerySet {self.model_cls.__name__} conditions={self.conditions}>"


# =============================================================================
# 4. Model Base Class & Metaclass
# =============================================================================

class ModelMeta(type):
    """Metaclass that collects Fields, registers tables and injects primary key."""

    def __new__(mcs, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        if name == "Model":
            return super().__new__(mcs, name, bases, attrs)

        fields: Dict[str, Field] = {}

        # Inherit fields from parent models
        for base in bases:
            if hasattr(base, "_fields"):
                fields.update(base._fields)

        has_pk = False
        for k, v in list(attrs.items()):
            if isinstance(v, Field):
                v.name = k
                fields[k] = v
                if v.primary_key:
                    has_pk = True

        # Default table name: lowercased class name + 's'
        if "_table_name" not in attrs:
            attrs["_table_name"] = f"{name.lower()}s"

        # Inject default integer auto-increment primary key 'id' if not provided
        if not has_pk:
            if "id" in fields:
                fields["id"].primary_key = True
            else:
                id_field = Field(type=int, primary_key=True)
                id_field.name = "id"
                fields["id"] = id_field
                attrs["id"] = None

        # Determine primary key column name
        pk_col = "id"
        for col_name, fld in fields.items():
            if fld.primary_key:
                pk_col = col_name
                break
        attrs["_pk_name"] = pk_col
        attrs["_fields"] = fields

        cls = super().__new__(mcs, name, bases, attrs)
        Database.register_model(cls)
        return cls


class Model(metaclass=ModelMeta):
    """
    Active-Record ORM Model base class.
    Provides create(), filter().all(), filter().first(), get(), save(), delete().
    """
    _table_name: str
    _fields: Dict[str, Field]
    _pk_name: str
    _db: Optional[Database] = None

    def __init__(self, **kwargs):
        self._is_persisted = False

        for col_name, field in self._fields.items():
            if col_name in kwargs:
                setattr(self, col_name, kwargs[col_name])
            elif field.default is not None:
                default_val = field.default() if callable(field.default) else field.default
                setattr(self, col_name, default_val)
            else:
                setattr(self, col_name, None)

        if self.pk is not None:
            self._is_persisted = True

    @property
    def pk(self) -> Any:
        """Returns the value of the primary key field."""
        return getattr(self, self._pk_name, None)

    @pk.setter
    def pk(self, value: Any):
        """Sets the value of the primary key field."""
        setattr(self, self._pk_name, value)

    @classmethod
    def get_db(cls) -> Database:
        """Returns the bound Database or active default Database."""
        return cls._db or Database.get_default()

    @classmethod
    def bind(cls, db: Database):
        """Binds this Model class to a specific Database instance."""
        cls._db = db
        Database.register_model(cls)

    @classmethod
    def create_table_sql(cls, dialect: Optional[Dialect] = None) -> str:
        """Returns the DDL statement to create the table for this model."""
        d = dialect or cls.get_db().dialect
        col_defs = [f.get_sql_definition(name, dialect=d) for name, f in cls._fields.items()]
        return d.build_create_table(cls._table_name, col_defs)

    # -------------------------------------------------------------------------
    # Active-Record CRUD Operations & Inspection
    # -------------------------------------------------------------------------

    @classmethod
    def create(cls, **kwargs) -> "Model":
        """Creates, saves and returns a new Model instance in a single step."""
        instance = cls(**kwargs)
        instance.save()
        return instance

    @classmethod
    def filter(cls, **kwargs) -> QuerySet:
        """Begins a filtered query on this Model."""
        return QuerySet(cls, conditions=kwargs)

    @classmethod
    def all(cls) -> List["Model"]:
        """Returns all records in the table."""
        return cls.filter().all()

    @classmethod
    def get(cls, id: Any = None, **kwargs) -> Optional["Model"]:
        """Retrieves a single record by primary key or filter kwargs."""
        if id is not None:
            kwargs[cls._pk_name] = id
        return cls.filter(**kwargs).first()

    def to_insert_sql(self, dialect: Optional[Dialect] = None) -> Tuple[str, Tuple]:
        """Compiles the INSERT SQL and values tuple for this instance without executing."""
        d = dialect or self.get_db().dialect
        columns: List[str] = []
        values: List[Any] = []

        pk_field = self._fields[self._pk_name]
        auto_inc = getattr(pk_field, "auto_increment", None)
        if auto_inc is None:
            auto_inc = (pk_field.primary_key and pk_field.type is int)

        for col_name, field in self._fields.items():
            val = getattr(self, col_name, None)
            if field.primary_key and auto_inc and val is None:
                continue
            columns.append(col_name)
            values.append(val)

        returning = self._pk_name if d.name == "postgresql" else None
        sql = d.build_insert(self._table_name, columns, returning=returning)
        return sql, tuple(values)

    def to_update_sql(self, dialect: Optional[Dialect] = None) -> Tuple[str, Tuple]:
        """Compiles the UPDATE SQL and values tuple for this instance without executing."""
        d = dialect or self.get_db().dialect
        columns: List[str] = []
        values: List[Any] = []

        for col_name, field in self._fields.items():
            if col_name == self._pk_name:
                continue
            columns.append(col_name)
            values.append(getattr(self, col_name, None))

        if not columns:
            return "", ()

        values.append(self.pk)
        sql = d.build_update(
            self._table_name,
            columns=columns,
            where_columns=[self._pk_name]
        )
        return sql, tuple(values)

    def save(self) -> "Model":
        """
        Persists the current model instance to the database.
        Performs an INSERT if new, or an UPDATE if already persisted.
        """
        db = self.get_db()
        dialect = db.dialect

        if not self._is_persisted or self.pk is None:
            # INSERT operation
            columns: List[str] = []
            values: List[Any] = []

            pk_field = self._fields[self._pk_name]
            auto_inc = getattr(pk_field, "auto_increment", None)
            if auto_inc is None:
                auto_inc = (pk_field.primary_key and pk_field.type is int)

            for col_name, field in self._fields.items():
                val = getattr(self, col_name, None)
                if field.primary_key and auto_inc and val is None:
                    continue
                columns.append(col_name)
                values.append(val)

            returning = self._pk_name if dialect.name == "postgresql" else None
            sql = dialect.build_insert(self._table_name, columns, returning=returning)
            cursor = db.execute(sql, tuple(values))

            # Retrieve auto-increment id if applicable
            if pk_field.primary_key and auto_inc and self.pk is None:
                if hasattr(cursor, "lastrowid") and cursor.lastrowid:
                    self.pk = cursor.lastrowid
                elif hasattr(cursor, "fetchone"):
                    row = cursor.fetchone()
                    if row:
                        self.pk = row[0]

            self._is_persisted = True
        else:
            # UPDATE operation
            columns: List[str] = []
            values: List[Any] = []

            for col_name, field in self._fields.items():
                if col_name == self._pk_name:
                    continue
                columns.append(col_name)
                values.append(getattr(self, col_name, None))

            if not columns:
                return self

            values.append(self.pk)
            sql = dialect.build_update(
                self._table_name,
                columns=columns,
                where_columns=[self._pk_name]
            )
            db.execute(sql, tuple(values))

        return self

    def delete(self) -> bool:
        """Deletes the current record from the database."""
        if self.pk is None:
            return False

        db = self.get_db()
        dialect = db.dialect
        sql = dialect.build_delete(self._table_name, where_columns=[self._pk_name])
        cursor = db.execute(sql, (self.pk,))
        self._is_persisted = False
        return cursor.rowcount > 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts Model attributes to a standard dictionary."""
        return {col: getattr(self, col) for col in self._fields}

    def __repr__(self) -> str:
        attrs = [f"{col}={getattr(self, col)!r}" for col in self._fields]
        return f"<{self.__class__.__name__} {', '.join(attrs)}>"
