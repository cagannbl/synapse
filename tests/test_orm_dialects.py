"""
Synapse Multi-Dialect ORM Test Suite
====================================
Tests:
1. Placeholder formatting across SQLite (?), Postgres ($1, $2...), MySQL (%s)
2. Identifier quotation across SQLite ("..."), Postgres ("..."), MySQL (`...`)
3. DDL column type generation across dialects (Postgres SERIAL PRIMARY KEY, MySQL AUTO_INCREMENT, SQLite AUTOINCREMENT)
4. Column type mapping, nullability, unique constraints, and dialect default values
5. CREATE TABLE query building across dialects
6. INSERT query building across dialects (including Postgres RETURNING and MySQL empty insert)
7. UPDATE query building with sequential parameter index tracking
8. SELECT query building with WHERE, ORDER BY, LIMIT, and OFFSET
9. DELETE query building across dialects
10. Connection URL parsing for SQLite (:memory:, file paths, sqlite:///)
11. Connection URL parsing for PostgreSQL (standard URLs, ports, query options)
12. Connection URL parsing for MySQL (standard URLs, drivers, ports, query options)
13. Database dialect auto-detection from connection strings and paths
14. Database explicit dialect override and get_dialect factory validation
15. Live SQLite Active-Record operations with auto-detected dialect
16. Live SQLite Active-Record operations with explicitly set dialect
17. Model & QuerySet multi-dialect SQL compilation without live remote connections
18. Package exports validation from synapse.orm
"""

import pathlib
import pytest
from synapse.orm import (
    Database,
    Field,
    Model,
    QuerySet,
    Dialect,
    SQLiteDialect,
    PostgresDialect,
    MySQLDialect,
    get_dialect,
    parse_connection_url,
)


# =============================================================================
# 1. Dialect Placeholders & Identifiers Tests
# =============================================================================

def test_dialect_param_placeholders():
    """Verify parameter placeholder format for SQLite (?), Postgres ($1, $2), and MySQL (%s)."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # SQLite always uses '?'
    assert sqlite.param_placeholder(1) == "?"
    assert sqlite.param_placeholder(2) == "?"
    assert sqlite.param_placeholder(10) == "?"

    # Postgres uses 1-based sequential indexing: $1, $2, $3...
    assert postgres.param_placeholder(1) == "$1"
    assert postgres.param_placeholder(2) == "$2"
    assert postgres.param_placeholder(15) == "$15"
    # Defensive handling for 0 or negative index
    assert postgres.param_placeholder(0) == "$1"

    # MySQL uses '%s'
    assert mysql.param_placeholder(1) == "%s"
    assert mysql.param_placeholder(2) == "%s"
    assert mysql.param_placeholder(99) == "%s"


def test_dialect_quote_identifiers():
    """Verify identifier quotation across dialects (double quotes vs backticks)."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # Simple identifier
    assert sqlite.quote_identifier("users") == '"users"'
    assert postgres.quote_identifier("users") == '"users"'
    assert mysql.quote_identifier("users") == "`users`"

    # Compound table.column identifier
    assert sqlite.quote_identifier("users.email") == '"users"."email"'
    assert postgres.quote_identifier("users.email") == '"users"."email"'
    assert mysql.quote_identifier("users.email") == "`users`.`email`"

    # Asterisk should remain unquoted
    assert sqlite.quote_identifier("*") == "*"
    assert postgres.quote_identifier("*") == "*"
    assert mysql.quote_identifier("*") == "*"

    # Already quoted identifier should not be double quoted
    assert sqlite.quote_identifier('"users"') == '"users"'
    assert mysql.quote_identifier("`users`") == "`users`"


# =============================================================================
# 2. DDL Generation Tests (SERIAL, AUTO_INCREMENT, AUTOINCREMENT)
# =============================================================================

def test_dialect_ddl_generation_primary_keys():
    """
    Verify primary key DDL generation across dialects:
    - Postgres: SERIAL PRIMARY KEY
    - MySQL: AUTO_INCREMENT (INT AUTO_INCREMENT PRIMARY KEY)
    - SQLite: AUTOINCREMENT (INTEGER PRIMARY KEY AUTOINCREMENT)
    """
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # Integer Primary Key (Auto-Increment)
    sqlite_pk = sqlite.type_to_sql(int, primary_key=True, auto_increment=True)
    assert "AUTOINCREMENT" in sqlite_pk
    assert sqlite_pk == "INTEGER PRIMARY KEY AUTOINCREMENT"

    pg_pk = postgres.type_to_sql(int, primary_key=True, auto_increment=True)
    assert "SERIAL PRIMARY KEY" in pg_pk

    mysql_pk = mysql.type_to_sql(int, primary_key=True, auto_increment=True)
    assert "AUTO_INCREMENT" in mysql_pk
    assert "PRIMARY KEY" in mysql_pk

    # Non-integer primary key
    sqlite_str_pk = sqlite.type_to_sql(str, primary_key=True)
    assert sqlite_str_pk == "TEXT PRIMARY KEY"

    pg_str_pk = postgres.type_to_sql(str, primary_key=True)
    assert pg_str_pk == "TEXT PRIMARY KEY"

    mysql_str_pk = mysql.type_to_sql(str, primary_key=True)
    assert mysql_str_pk == "VARCHAR(255) PRIMARY KEY"


def test_dialect_ddl_column_types_and_defaults():
    """Verify column types, nullability, unique constraints, and default value formatting."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # NOT NULL and UNIQUE string
    sqlite_str = sqlite.type_to_sql(str, nullable=False, unique=True)
    assert sqlite_str == "TEXT NOT NULL UNIQUE"

    pg_str = postgres.type_to_sql(str, nullable=False, unique=True)
    assert pg_str == "TEXT NOT NULL UNIQUE"

    mysql_str = mysql.type_to_sql(str, nullable=False, unique=True)
    assert mysql_str == "VARCHAR(255) NOT NULL UNIQUE"

    # Boolean with default True
    sqlite_bool = sqlite.type_to_sql(bool, default=True)
    assert "DEFAULT 1" in sqlite_bool

    pg_bool = postgres.type_to_sql(bool, default=True)
    assert "DEFAULT TRUE" in pg_bool
    assert "BOOLEAN" in pg_bool

    pg_bool_false = postgres.type_to_sql(bool, default=False)
    assert "DEFAULT FALSE" in pg_bool_false

    mysql_bool = mysql.type_to_sql(bool, default=True)
    assert "DEFAULT 1" in mysql_bool

    # Float with default
    sqlite_float = sqlite.type_to_sql(float, default=99.9)
    assert "REAL DEFAULT 99.9" in sqlite_float

    pg_float = postgres.type_to_sql(float, default=99.9)
    assert "DOUBLE PRECISION DEFAULT 99.9" in pg_float

    mysql_float = mysql.type_to_sql(float, default=99.9)
    assert "DOUBLE DEFAULT 99.9" in mysql_float

    # Bytes / BLOB
    assert sqlite.type_to_sql(bytes) == "BLOB"
    assert postgres.type_to_sql(bytes) == "BYTEA"
    assert mysql.type_to_sql(bytes) == "BLOB"


def test_dialect_build_create_table():
    """Verify CREATE TABLE DDL construction across all dialects."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # List of column definition strings
    col_defs = ['"id" SERIAL PRIMARY KEY', '"name" TEXT NOT NULL']
    pg_sql = postgres.build_create_table("tenants", col_defs, if_not_exists=True)
    assert pg_sql == 'CREATE TABLE IF NOT EXISTS "tenants" ("id" SERIAL PRIMARY KEY, "name" TEXT NOT NULL)'

    # Dict of string definitions
    cols_dict = {"id": "INT AUTO_INCREMENT PRIMARY KEY", "name": "VARCHAR(255) NOT NULL"}
    mysql_sql = mysql.build_create_table("tenants", cols_dict, if_not_exists=False)
    assert mysql_sql == "CREATE TABLE `tenants` (`id` INT AUTO_INCREMENT PRIMARY KEY, `name` VARCHAR(255) NOT NULL)"

    # Dict with Field instances
    fields_dict = {
        "id": Field(type=int, primary_key=True),
        "title": Field(type=str, nullable=False),
        "active": Field(type=bool, default=True),
    }
    sqlite_sql = sqlite.build_create_table("posts", fields_dict)
    assert 'CREATE TABLE IF NOT EXISTS "posts"' in sqlite_sql
    assert '"id" INTEGER PRIMARY KEY AUTOINCREMENT' in sqlite_sql
    assert '"title" TEXT NOT NULL' in sqlite_sql
    assert '"active" INTEGER DEFAULT 1' in sqlite_sql


# =============================================================================
# 3. Query Generation Tests (INSERT, SELECT, UPDATE, DELETE)
# =============================================================================

def test_dialect_build_insert():
    """Verify INSERT query generation with appropriate placeholders and RETURNING clauses."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    columns = ["username", "email", "age"]

    # SQLite
    sql_sqlite = sqlite.build_insert("users", columns)
    assert sql_sqlite == 'INSERT INTO "users" ("username", "email", "age") VALUES (?, ?, ?)'

    # Postgres
    sql_pg = postgres.build_insert("users", columns)
    assert sql_pg == 'INSERT INTO "users" ("username", "email", "age") VALUES ($1, $2, $3)'

    # Postgres with RETURNING
    sql_pg_ret = postgres.build_insert("users", columns, returning="id")
    assert sql_pg_ret == 'INSERT INTO "users" ("username", "email", "age") VALUES ($1, $2, $3) RETURNING "id"'

    # MySQL
    sql_mysql = mysql.build_insert("users", columns)
    assert sql_mysql == "INSERT INTO `users` (`username`, `email`, `age`) VALUES (%s, %s, %s)"

    # Empty columns insert
    assert sqlite.build_insert("audit_logs", []) == 'INSERT INTO "audit_logs" DEFAULT VALUES'
    assert mysql.build_insert("audit_logs", []) == "INSERT INTO `audit_logs` () VALUES ()"


def test_dialect_build_update():
    """Verify UPDATE query generation with sequential parameter indexing."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    columns = ["name", "status"]
    where_columns = ["id", "tenant_id"]

    # SQLite (? placeholders)
    sql_sqlite = sqlite.build_update("organizations", columns, where_columns=where_columns)
    assert sql_sqlite == 'UPDATE "organizations" SET "name" = ?, "status" = ? WHERE "id" = ? AND "tenant_id" = ?'

    # Postgres (sequential $1, $2, $3, $4 placeholders)
    sql_pg = postgres.build_update("organizations", columns, where_columns=where_columns)
    assert sql_pg == 'UPDATE "organizations" SET "name" = $1, "status" = $2 WHERE "id" = $3 AND "tenant_id" = $4'

    # MySQL (%s placeholders)
    sql_mysql = mysql.build_update("organizations", columns, where_columns=where_columns)
    assert sql_mysql == "UPDATE `organizations` SET `name` = %s, `status` = %s WHERE `id` = %s AND `tenant_id` = %s"

    # Custom where clause
    sql_custom = sqlite.build_update("users", ["active"], where_clause='"id" > 10')
    assert sql_custom == 'UPDATE "users" SET "active" = ? WHERE "id" > 10'


def test_dialect_build_select_queries():
    """Verify SELECT query generation with WHERE, ORDER BY, LIMIT, and OFFSET."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # Basic SELECT *
    assert sqlite.build_select("items") == 'SELECT * FROM "items"'
    assert mysql.build_select("items") == "SELECT * FROM `items`"

    # Specific columns, WHERE, ORDER BY descending, LIMIT, OFFSET
    # Postgres
    pg_sql = postgres.build_select(
        table="orders",
        columns=["id", "total", "customer_id"],
        where_columns=["status", "customer_id"],
        order_by="-created_at",
        limit=20,
        offset=40
    )
    assert pg_sql == (
        'SELECT "id", "total", "customer_id" FROM "orders" '
        'WHERE "status" = $1 AND "customer_id" = $2 '
        'ORDER BY "created_at" DESC LIMIT 20 OFFSET 40'
    )

    # MySQL
    mysql_sql = mysql.build_select(
        table="orders",
        columns=["id", "total"],
        where_columns=["status"],
        order_by="+id",
        limit=10
    )
    assert mysql_sql == 'SELECT `id`, `total` FROM `orders` WHERE `status` = %s ORDER BY `id` ASC LIMIT 10'

    # SQLite with order_by string ending in DESC
    sqlite_sql = sqlite.build_select(
        table="orders",
        columns="COUNT(*)",
        order_by="score DESC"
    )
    assert sqlite_sql == 'SELECT COUNT(*) FROM "orders" ORDER BY "score" DESC'


def test_dialect_build_delete_queries():
    """Verify DELETE query generation across dialects."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    # Postgres with where_columns
    pg_del = postgres.build_delete("sessions", where_columns=["token", "user_id"])
    assert pg_del == 'DELETE FROM "sessions" WHERE "token" = $1 AND "user_id" = $2'

    # MySQL with where_columns
    mysql_del = mysql.build_delete("sessions", where_columns=["token"])
    assert mysql_del == "DELETE FROM `sessions` WHERE `token` = %s"

    # SQLite with custom where clause
    sqlite_del = sqlite.build_delete("sessions", where_clause='"expires_at" < 1000')
    assert sqlite_del == 'DELETE FROM "sessions" WHERE "expires_at" < 1000'

    # Total table delete (no WHERE)
    assert sqlite.build_delete("temp_data") == 'DELETE FROM "temp_data"'


# =============================================================================
# 4. Connection URL Parsing Tests
# =============================================================================

def test_parse_connection_url_sqlite():
    """Verify parsing of SQLite URLs and bare paths."""
    # In-memory URI
    d1, p1 = parse_connection_url("sqlite:///:memory:")
    assert d1 == "sqlite"
    assert p1["database"] == ":memory:"

    # Relative file URI
    d2, p2 = parse_connection_url("sqlite:///app.db")
    assert d2 == "sqlite"
    assert p2["database"] == "app.db"

    # Absolute file URI
    d3, p3 = parse_connection_url("sqlite:////var/data/synapse.db")
    assert d3 == "sqlite"
    assert p3["database"] == "/var/data/synapse.db"

    # Bare path :memory:
    d4, p4 = parse_connection_url(":memory:")
    assert d4 == "sqlite"
    assert p4["database"] == ":memory:"

    # Bare file path
    d5, p5 = parse_connection_url("my_custom_database.db")
    assert d5 == "sqlite"
    assert p5["database"] == "my_custom_database.db"

    # Empty string defaults to :memory:
    d6, p6 = parse_connection_url("")
    assert d6 == "sqlite"
    assert p6["database"] == ":memory:"


def test_parse_connection_url_postgresql():
    """Verify parsing of PostgreSQL connection URLs."""
    # Standard postgresql:// URL with port and credentials
    url1 = "postgresql://postgres:secretpassword@localhost:5432/enterprise_db"
    d1, p1 = parse_connection_url(url1)
    assert d1 == "postgresql"
    assert p1["user"] == "postgres"
    assert p1["password"] == "secretpassword"
    assert p1["host"] == "localhost"
    assert p1["port"] == 5432
    assert p1["database"] == "enterprise_db"

    # Short scheme postgres:// with query parameters and custom port
    url2 = "postgres://alice:token123@db.internal:5439/analytics?sslmode=require&application_name=synapse"
    d2, p2 = parse_connection_url(url2)
    assert d2 == "postgresql"
    assert p2["user"] == "alice"
    assert p2["password"] == "token123"
    assert p2["host"] == "db.internal"
    assert p2["port"] == 5439
    assert p2["database"] == "analytics"
    assert p2["sslmode"] == "require"
    assert p2["application_name"] == "synapse"

    # URL with driver prefix
    url3 = "postgresql+psycopg2://app:pass@10.0.0.1/prod"
    d3, p3 = parse_connection_url(url3)
    assert d3 == "postgresql"
    assert p3["port"] == 5432  # Default PostgreSQL port applied


def test_parse_connection_url_mysql():
    """Verify parsing of MySQL connection URLs."""
    # Standard mysql:// URL
    url1 = "mysql://root:admin_pass@127.0.0.1:3306/production_db"
    d1, p1 = parse_connection_url(url1)
    assert d1 == "mysql"
    assert p1["user"] == "root"
    assert p1["password"] == "admin_pass"
    assert p1["host"] == "127.0.0.1"
    assert p1["port"] == 3306
    assert p1["database"] == "production_db"

    # Driver prefix with query parameter
    url2 = "mysql+pymysql://synapse_user:secret@mysql.cluster:3307/ecommerce?charset=utf8mb4"
    d2, p2 = parse_connection_url(url2)
    assert d2 == "mysql"
    assert p2["user"] == "synapse_user"
    assert p2["host"] == "mysql.cluster"
    assert p2["port"] == 3307
    assert p2["database"] == "ecommerce"
    assert p2["charset"] == "utf8mb4"


# =============================================================================
# 5. Database Dialect Auto-detection & Factory Tests
# =============================================================================

def test_database_dialect_autodetection(tmp_path):
    """Verify Database automatically detects the appropriate dialect from URL or path."""
    db_mem = Database(":memory:")
    assert isinstance(db_mem.dialect, SQLiteDialect)
    assert db_mem.dialect.name == "sqlite"

    db_sqlite = Database("sqlite:///:memory:")
    assert isinstance(db_sqlite.dialect, SQLiteDialect)

    db_file = tmp_path / "synapse_test.db"
    db_path = Database(str(db_file))
    assert isinstance(db_path.dialect, SQLiteDialect)
    db_path.close()

    db_pg = Database("postgresql://user:pass@localhost:5432/mydb")
    assert isinstance(db_pg.dialect, PostgresDialect)
    assert db_pg.dialect.name == "postgresql"

    db_mysql = Database("mysql://root:pass@localhost:3306/mydb")
    assert isinstance(db_mysql.dialect, MySQLDialect)
    assert db_mysql.dialect.name == "mysql"


def test_database_explicit_dialect_override_and_factory():
    """Verify explicit dialect override via string name or Dialect instance."""
    # String override
    db_pg_str = Database(":memory:", dialect="postgresql")
    assert isinstance(db_pg_str.dialect, PostgresDialect)

    db_mysql_str = Database(":memory:", dialect="mysql")
    assert isinstance(db_mysql_str.dialect, MySQLDialect)

    # Instance override
    my_dialect = MySQLDialect()
    db_instance = Database(":memory:", dialect=my_dialect)
    assert db_instance.dialect is my_dialect

    # Factory get_dialect tests
    assert isinstance(get_dialect("sqlite"), SQLiteDialect)
    assert isinstance(get_dialect("postgres"), PostgresDialect)
    assert isinstance(get_dialect("postgresql"), PostgresDialect)
    assert isinstance(get_dialect("mysql"), MySQLDialect)

    with pytest.raises(ValueError, match="Unsupported SQL dialect"):
        get_dialect("oracle_unsupported")

    with pytest.raises(TypeError):
        get_dialect(12345)


# =============================================================================
# 6. Live SQLite Database Operations Tests
# =============================================================================

def test_live_sqlite_operations_auto_detected_dialect():
    """Verify complete Active-Record CRUD workflow on live SQLite with auto-detected dialect."""
    db = Database("sqlite:///:memory:")

    class InventoryItem(Model):
        _table_name = "inventory_items"
        _db = db
        sku = Field(type=str, unique=True, nullable=False)
        quantity = Field(type=int, default=0)
        unit_price = Field(type=float, default=1.0)

    db.create_all()

    # Create records
    item1 = InventoryItem.create(sku="SKU-100", quantity=50, unit_price=9.99)
    item2 = InventoryItem.create(sku="SKU-200", quantity=15, unit_price=24.50)

    assert item1.id == 1
    assert item2.id == 2

    # Query with .filter(), .all(), .first(), .count()
    all_items = InventoryItem.filter().all()
    assert len(all_items) == 2

    low_stock = InventoryItem.filter(quantity=15).first()
    assert low_stock is not None
    assert low_stock.sku == "SKU-200"

    assert InventoryItem.filter(quantity=50).count() == 1

    # Update record
    low_stock.quantity = 30
    low_stock.save()

    refetched = InventoryItem.get(2)
    assert refetched.quantity == 30

    # Delete record
    deleted = item1.delete()
    assert deleted is True
    assert InventoryItem.filter().count() == 1
    assert InventoryItem.get(1) is None


def test_live_sqlite_operations_explicit_dialect():
    """Verify live SQLite operations with explicitly injected SQLiteDialect instance."""
    explicit_dialect = SQLiteDialect()
    db = Database(":memory:", dialect=explicit_dialect)
    assert db.dialect is explicit_dialect

    class Metric(Model):
        _table_name = "metrics"
        _db = db
        name = Field(type=str, nullable=False)
        value = Field(type=float, default=0.0)

    db.create_all()

    # Active-Record operations
    m1 = Metric.create(name="cpu_load", value=42.5)
    m2 = Metric.create(name="mem_load", value=88.1)

    assert Metric.filter().count() == 2

    # Order by value descending
    highest = Metric.filter().order_by("-value").first()
    assert highest is not None
    assert highest.name == "mem_load"
    assert highest.value == 88.1

    # Bulk delete via QuerySet
    deleted_count = Metric.filter(name="cpu_load").delete()
    assert deleted_count == 1
    assert Metric.filter().count() == 1


# =============================================================================
# 7. Model Multi-Dialect Query Compilation Tests
# =============================================================================

def test_model_and_queryset_multi_dialect_sql_generation():
    """
    Verify Model.create_table_sql() and QuerySet.to_sql() compile appropriate
    dialect SQL (Postgres $1 vs MySQL %s vs SQLite ?) without connecting to remote DB.
    """
    class Invoice(Model):
        _table_name = "invoices"
        number = Field(type=str, nullable=False, unique=True)
        amount = Field(type=float, default=0.0)
        is_paid = Field(type=bool, default=False)

    # 1. DDL generation via Model.create_table_sql()
    sqlite_ddl = Invoice.create_table_sql(SQLiteDialect())
    assert 'CREATE TABLE IF NOT EXISTS "invoices"' in sqlite_ddl
    assert '"id" INTEGER PRIMARY KEY AUTOINCREMENT' in sqlite_ddl

    pg_ddl = Invoice.create_table_sql(PostgresDialect())
    assert 'CREATE TABLE IF NOT EXISTS "invoices"' in pg_ddl
    assert '"id" SERIAL PRIMARY KEY' in pg_ddl
    assert 'DEFAULT FALSE' in pg_ddl

    mysql_ddl = Invoice.create_table_sql(MySQLDialect())
    assert "CREATE TABLE IF NOT EXISTS `invoices`" in mysql_ddl
    assert "`id` INT AUTO_INCREMENT PRIMARY KEY" in mysql_ddl
    assert "DEFAULT 0" in mysql_ddl

    # 2. QuerySet query compilation bound to Postgres DB
    pg_db = Database("postgresql://user:pass@localhost:5432/testdb")
    Invoice.bind(pg_db)

    qs_pg = Invoice.filter(is_paid=True).order_by("-amount").limit(5).offset(10)
    sql_pg, params_pg = qs_pg.to_sql()
    assert sql_pg == 'SELECT * FROM "invoices" WHERE "is_paid" = $1 ORDER BY "amount" DESC LIMIT 5 OFFSET 10'
    assert params_pg == (True,)

    # 3. QuerySet query compilation bound to MySQL DB
    mysql_db = Database("mysql://root:pass@localhost:3306/testdb")
    Invoice.bind(mysql_db)

    qs_mysql = Invoice.filter(number="INV-2026-001")
    sql_mysql, params_mysql = qs_mysql.to_sql()
    assert sql_mysql == "SELECT * FROM `invoices` WHERE `number` = %s"
    assert params_mysql == ("INV-2026-001",)

    # 4. QuerySet query compilation bound to SQLite DB
    sqlite_db = Database(":memory:")
    Invoice.bind(sqlite_db)

    qs_sqlite = Invoice.filter(number="INV-2026-001")
    sql_sqlite, params_sqlite = qs_sqlite.to_sql()
    assert sql_sqlite == 'SELECT * FROM "invoices" WHERE "number" = ?'
    assert params_sqlite == ("INV-2026-001",)


def test_orm_exports_in_init():
    """Verify that all dialect classes and functions are cleanly exported from synapse.orm."""
    import synapse.orm as orm

    assert hasattr(orm, "Dialect")
    assert hasattr(orm, "SQLiteDialect")
    assert hasattr(orm, "PostgresDialect")
    assert hasattr(orm, "MySQLDialect")
    assert hasattr(orm, "get_dialect")
    assert hasattr(orm, "parse_connection_url")
    assert hasattr(orm, "Database")
    assert hasattr(orm, "Model")
    assert hasattr(orm, "Field")
    assert hasattr(orm, "QuerySet")


# =============================================================================
# 8. Advanced Edge Cases & Dialect Syntax Specifics
# =============================================================================

def test_dialect_offset_without_limit_handling():
    """
    Verify that offset without limit is correctly formatted according to each dialect:
    - SQLite: LIMIT -1 OFFSET {offset} (prevents syntax error: near '{offset}')
    - MySQL: LIMIT 18446744073709551615 OFFSET {offset}
    - Postgres: OFFSET {offset}
    And verify live execution on SQLite works without crashing.
    """
    sqlite = SQLiteDialect()
    mysql = MySQLDialect()
    postgres = PostgresDialect()

    # SQLite LIMIT -1 OFFSET
    sql_sqlite = sqlite.build_select("users", offset=5)
    assert sql_sqlite == 'SELECT * FROM "users" LIMIT -1 OFFSET 5'

    # MySQL LIMIT 18446744073709551615 OFFSET
    sql_mysql = mysql.build_select("users", offset=10)
    assert sql_mysql == "SELECT * FROM `users` LIMIT 18446744073709551615 OFFSET 10"

    # Postgres OFFSET
    sql_pg = postgres.build_select("users", offset=15)
    assert sql_pg == 'SELECT * FROM "users" OFFSET 15'

    # Live SQLite execution of offset without limit
    db = Database("sqlite:///:memory:")

    class Tag(Model):
        _table_name = "tags"
        _db = db
        label = Field(type=str)

    db.create_all()
    Tag.create(label="alpha")
    Tag.create(label="beta")
    Tag.create(label="gamma")

    # Offset only - must succeed and return the sliced records
    offset_results = Tag.filter().offset(1).all()
    assert len(offset_results) == 2
    assert [t.label for t in offset_results] == ["beta", "gamma"]


def test_dialect_multi_column_order_by_formatting():
    """
    Verify multi-column order_by properly quotes each column identifier individually
    instead of treating the entire comma-separated string as a single column name.
    """
    sqlite = SQLiteDialect()
    mysql = MySQLDialect()
    postgres = PostgresDialect()

    # String with multiple columns and mixed directions
    sql_sqlite = sqlite.build_select("users", order_by="-created_at, +id")
    assert sql_sqlite == 'SELECT * FROM "users" ORDER BY "created_at" DESC, "id" ASC'

    # List of column ordering specifications
    sql_mysql = mysql.build_select("users", order_by=["created_at DESC", "id ASC"])
    assert sql_mysql == "SELECT * FROM `users` ORDER BY `created_at` DESC, `id` ASC"

    # Postgres multi-column order_by
    sql_pg = postgres.build_select("users", order_by=["priority DESC", "created_at ASC"])
    assert sql_pg == 'SELECT * FROM "users" ORDER BY "priority" DESC, "created_at" ASC'

    # QuerySet order_by multi-column validation
    class TaskItem(Model):
        _table_name = "task_items"
        title = Field(type=str)
        priority = Field(type=int)
        created_at = Field(type=str)

    qs = TaskItem.filter().order_by("-priority", "title")
    sql, _ = qs.to_sql()
    assert 'ORDER BY "priority" DESC, "title" ASC' in sql


def test_dialect_explicit_auto_increment_false():
    """
    Verify that explicit auto_increment=False produces standard PRIMARY KEY
    without AUTOINCREMENT / SERIAL across all dialects.
    """
    sqlite = SQLiteDialect()
    mysql = MySQLDialect()
    postgres = PostgresDialect()

    # SQLite
    sqlite_pk = sqlite.type_to_sql(int, primary_key=True, auto_increment=False)
    assert sqlite_pk == "INTEGER PRIMARY KEY"
    assert "AUTOINCREMENT" not in sqlite_pk

    # Postgres
    pg_pk = postgres.type_to_sql(int, primary_key=True, auto_increment=False)
    assert pg_pk == "INTEGER PRIMARY KEY"
    assert "SERIAL" not in pg_pk

    # MySQL
    mysql_pk = mysql.type_to_sql(int, primary_key=True, auto_increment=False)
    assert mysql_pk == "INT PRIMARY KEY"
    assert "AUTO_INCREMENT" not in mysql_pk

    # Field instance with auto_increment=False
    f = Field(type=int, primary_key=True, auto_increment=False)
    assert f.get_sql_definition("external_id", dialect=postgres) == '"external_id" INTEGER PRIMARY KEY'
    assert f.get_sql_definition("external_id", dialect=mysql) == "`external_id` INT PRIMARY KEY"


def test_dialect_default_value_single_quote_escaping():
    """Verify that single quotes in default values are escaped properly in SQL DDL."""
    sqlite = SQLiteDialect()
    postgres = PostgresDialect()
    mysql = MySQLDialect()

    val = "O'Reilly Book"
    sqlite_ddl = sqlite.type_to_sql(str, default=val)
    assert "DEFAULT 'O''Reilly Book'" in sqlite_ddl

    pg_ddl = postgres.type_to_sql(str, default=val)
    assert "DEFAULT 'O''Reilly Book'" in pg_ddl

    mysql_ddl = mysql.type_to_sql(str, default=val)
    assert "DEFAULT 'O''Reilly Book'" in mysql_ddl


def test_dialect_quote_identifier_escaping():
    """Verify identifier quote escaping for special characters inside names."""
    sqlite = SQLiteDialect()
    mysql = MySQLDialect()

    assert sqlite.quote_identifier('user"name') == '"user""name"'
    assert mysql.quote_identifier("user`name") == "`user``name`"


def test_parse_connection_url_sqlite_query_params_and_windows_paths():
    """Verify parsing of SQLite URLs with query strings and Windows drive paths."""
    # SQLite with cache=shared query parameter
    d1, p1 = parse_connection_url("sqlite:///:memory:?cache=shared&mode=memory")
    assert d1 == "sqlite"
    assert p1["database"] == ":memory:"
    assert p1["options"]["cache"] == "shared"
    assert p1["options"]["mode"] == "memory"

    # SQLite relative path with query parameters
    d2, p2 = parse_connection_url("sqlite:///app.db?mode=ro")
    assert d2 == "sqlite"
    assert p2["database"] == "app.db"
    assert p2["options"]["mode"] == "ro"

    # Windows drive / absolute path dynamic test
    dynamic_db_path = str(pathlib.Path(__file__).resolve().parent / "test_data.db").replace("\\", "/")
    d3, p3 = parse_connection_url(f"sqlite:///{dynamic_db_path}")
    assert d3 == "sqlite"
    assert p3["database"] == dynamic_db_path

    # MariaDB scheme mapping to MySQL dialect
    d4, p4 = parse_connection_url("mariadb://admin:pass@db.local:3306/production")
    assert d4 == "mysql"
    assert p4["user"] == "admin"
    assert p4["port"] == 3306
    assert p4["database"] == "production"


def test_model_and_queryset_to_insert_update_delete_sql():
    """
    Verify Model.to_insert_sql(), Model.to_update_sql(), and QuerySet.to_delete_sql()
    produce accurate multi-dialect SQL statements without live database execution.
    """
    class CustomerOrder(Model):
        _table_name = "customer_orders"
        order_code = Field(type=str, nullable=False)
        total_amount = Field(type=float, default=0.0)

    pg_dialect = PostgresDialect()
    mysql_dialect = MySQLDialect()

    # 1. to_insert_sql for Postgres (includes RETURNING id)
    order = CustomerOrder(order_code="ORD-99", total_amount=129.50)
    sql_ins_pg, params_ins_pg = order.to_insert_sql(pg_dialect)
    assert sql_ins_pg == 'INSERT INTO "customer_orders" ("order_code", "total_amount") VALUES ($1, $2) RETURNING "id"'
    assert params_ins_pg == ("ORD-99", 129.50)

    # 2. to_insert_sql for MySQL
    sql_ins_my, params_ins_my = order.to_insert_sql(mysql_dialect)
    assert sql_ins_my == "INSERT INTO `customer_orders` (`order_code`, `total_amount`) VALUES (%s, %s)"
    assert params_ins_my == ("ORD-99", 129.50)

    # 3. to_update_sql for Postgres
    order.pk = 42
    sql_upd_pg, params_upd_pg = order.to_update_sql(pg_dialect)
    assert sql_upd_pg == 'UPDATE "customer_orders" SET "order_code" = $1, "total_amount" = $2 WHERE "id" = $3'
    assert params_upd_pg == ("ORD-99", 129.50, 42)

    # 4. to_delete_sql for MySQL QuerySet
    mysql_db = Database("mysql://root:pass@localhost:3306/testdb")
    CustomerOrder.bind(mysql_db)
    del_qs = CustomerOrder.filter(order_code="ORD-99")
    sql_del_my, params_del_my = del_qs.to_delete_sql()
    assert sql_del_my == "DELETE FROM `customer_orders` WHERE `order_code` = %s"
    assert params_del_my == ("ORD-99",)


def test_empty_update_columns_safely_handled():
    """Verify that updating a model with no non-PK fields returns safely without error."""
    db = Database("sqlite:///:memory:")

    class EmptyRecord(Model):
        _table_name = "empty_records"
        _db = db

    db.create_all()
    rec = EmptyRecord.create()
    assert rec.id == 1

    # Saving persisted record with no non-PK fields should not throw
    saved_again = rec.save()
    assert saved_again.id == 1

