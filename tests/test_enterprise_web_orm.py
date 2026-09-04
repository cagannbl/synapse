"""
Comprehensive Enterprise Web, Type-Safe ORM, and Task Queue Test Suite
======================================================================
Tests:
1. Web: Route methods (@app.get, @app.post, @app.put, @app.delete, @app.patch)
2. Web: URL path parameter extraction & auto type casting (/users/{id:int}, /items/{slug:str}, /prices/{val:float})
3. Web: JSON request body parsing & serialization
4. Web: Depends(...) dependency injection (single & nested)
5. Web: HTTP status codes (201, 404, 405, 500, HTTPException)
6. Web: CORS preflight (OPTIONS) and response headers
7. Web: JWT token creation, signature verification, expiration, and jwt_required dependency
8. Web: Programmatic handle_request() query strings and custom headers
9. Web: Enterprise middleware pipeline execution
10. Web: Live HTTP server serve() and graceful shutdown
11. ORM: Database initialization, Field types & create_all()
12. ORM: Active-Record create() and get(id)
13. ORM: Active-Record filter().all(), filter().first(), count(), order_by()
14. ORM: Active-Record save() (insert & update) and delete()
15. ORM: SQL injection resistance with parameterized ? bindings & column validation
16. ORM: Transaction commit and rollback
17. Tasks: TaskQueue registration with @queue.task and .delay(*args, **kwargs)
18. Tasks: Multi-worker pool parallel concurrency
19. Tasks: TimeoutError on wait(timeout) & TaskStatus.FAILED error capture
20. Full-Stack: End-to-end Web + ORM + Background Task Queue integration
"""

import json
import time
import urllib.request
import urllib.error
import pytest

from synapse.web import (
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
from synapse.orm import Database, Field, Model, QuerySet
from synapse.tasks import TaskQueue, TaskResult, TaskStatus


# =============================================================================
# 1. Web Framework Tests
# =============================================================================

def test_web_route_methods():
    """Verify @app.get, @app.post, @app.put, @app.delete, @app.patch handlers."""
    app = SynapseApp()

    @app.get("/items")
    def list_items():
        return {"action": "list"}

    @app.post("/items")
    def create_item(body: dict):
        return {"action": "create", "data": body}, 201

    @app.put("/items")
    def update_item(body: dict):
        return {"action": "update", "data": body}

    @app.delete("/items")
    def delete_item():
        return {"action": "delete"}

    @app.patch("/items")
    def patch_item():
        return {"action": "patch"}

    res_get = app.handle_request("GET", "/items")
    assert res_get.status_code == 200
    assert res_get.json()["action"] == "list"

    res_post = app.handle_request("POST", "/items", body={"name": "widget"})
    assert res_post.status_code == 201
    assert res_post.json()["data"]["name"] == "widget"

    res_put = app.handle_request("PUT", "/items", body={"id": 1, "name": "updated"})
    assert res_put.status_code == 200
    assert res_put.json()["action"] == "update"

    res_del = app.handle_request("DELETE", "/items")
    assert res_del.status_code == 200
    assert res_del.json()["action"] == "delete"

    res_patch = app.handle_request("PATCH", "/items")
    assert res_patch.status_code == 200
    assert res_patch.json()["action"] == "patch"


def test_web_url_path_parameter_type_casting():
    """Verify /users/{id:int}, /items/{slug:str}, /prices/{val:float} automatic casting."""
    app = SynapseApp()

    @app.get("/users/{id:int}")
    def get_user(id: int):
        assert isinstance(id, int)
        return {"user_id": id, "type": type(id).__name__}

    @app.get("/items/{slug:str}")
    def get_item(slug: str):
        assert isinstance(slug, str)
        return {"slug": slug}

    @app.get("/prices/{val:float}")
    def get_price(val: float):
        assert isinstance(val, float)
        return {"price": val}

    # Valid int
    res = app.handle_request("GET", "/users/1042")
    assert res.status_code == 200
    assert res.json()["user_id"] == 1042
    assert res.json()["type"] == "int"

    # Non-integer passed to int route should not match int regex (404)
    res_invalid = app.handle_request("GET", "/users/not-a-number")
    assert res_invalid.status_code == 404

    # String slug
    res_slug = app.handle_request("GET", "/items/fast-compiler")
    assert res_slug.status_code == 200
    assert res_slug.json()["slug"] == "fast-compiler"

    # Float price
    res_float = app.handle_request("GET", "/prices/49.99")
    assert res_float.status_code == 200
    assert res_float.json()["price"] == 49.99


def test_web_json_body_parsing():
    """Verify request body JSON parsing and automatic serialization."""
    app = SynapseApp()

    @app.post("/submit")
    def submit(body: dict):
        return {"received": body, "count": len(body)}

    # String JSON
    payload = json.dumps({"title": "Synapse V3", "stars": 500})
    res1 = app.handle_request("POST", "/submit", body=payload)
    assert res1.status_code == 200
    assert res1.json()["received"]["stars"] == 500

    # Bytes JSON
    res2 = app.handle_request("POST", "/submit", body=payload.encode("utf-8"))
    assert res2.status_code == 200
    assert res2.json()["received"]["title"] == "Synapse V3"

    # Direct Dict
    res3 = app.handle_request("POST", "/submit", body={"direct": True})
    assert res3.status_code == 200
    assert res3.json()["received"]["direct"] is True


def test_web_dependency_injection_depends():
    """Verify Depends(...) dependency injection, including nested dependencies."""
    app = SynapseApp()

    def get_api_key(req: Request) -> str:
        key = req.headers.get("x-api-key")
        if not key:
            raise HTTPException(status_code=401, detail="API key required")
        return key

    def get_client_session(api_key: str = Depends(get_api_key)) -> dict:
        return {"session_id": f"sess_{api_key}", "authenticated": True}

    @app.get("/secure/data")
    def secure_endpoint(session: dict = Depends(get_client_session)):
        return {"data": "classified", "session": session}

    # Missing header -> 401
    res_unauth = app.handle_request("GET", "/secure/data")
    assert res_unauth.status_code == 401
    assert "API key required" in res_unauth.json()["error"]

    # Valid header -> 200 with injected nested dependency
    res_auth = app.handle_request(
        "GET",
        "/secure/data",
        headers={"X-Api-Key": "secret123"}
    )
    assert res_auth.status_code == 200
    assert res_auth.json()["session"]["session_id"] == "sess_secret123"
    assert res_auth.json()["session"]["authenticated"] is True


def test_web_status_codes_and_error_handling():
    """Verify HTTP status codes: 201, 404, 405, and 500 error handling."""
    app = SynapseApp()

    @app.get("/exists")
    def exists():
        return {"ok": True}

    @app.post("/custom-created")
    def custom():
        return Response({"id": 99}, status_code=201)

    @app.get("/crash")
    def crash():
        raise RuntimeError("Unexpected internal crash")

    # 404 Not Found
    res404 = app.handle_request("GET", "/does-not-exist")
    assert res404.status_code == 404
    assert res404.json()["error"] == "Route not found"

    # 405 Method Not Allowed
    res405 = app.handle_request("POST", "/exists")
    assert res405.status_code == 405
    assert "GET" in res405.headers.get("Allow", "")

    # 201 Created
    res201 = app.handle_request("POST", "/custom-created")
    assert res201.status_code == 201
    assert res201.json()["id"] == 99

    # 500 Internal Error
    res500 = app.handle_request("GET", "/crash")
    assert res500.status_code == 500
    assert "Unexpected internal crash" in res500.json()["error"]


def test_web_cors_preflight_and_headers():
    """Verify CORS preflight OPTIONS request and CORS headers injection."""
    app = SynapseApp(
        cors_enabled=True,
        cors_origins=["https://synapse.dev"],
        cors_methods=["GET", "POST"]
    )

    @app.get("/api/test")
    def api_test():
        return {"status": "ok"}

    # Preflight OPTIONS
    options_res = app.handle_request(
        "OPTIONS",
        "/api/test",
        headers={"Origin": "https://synapse.dev"}
    )
    assert options_res.status_code == 204
    assert options_res.headers["Access-Control-Allow-Origin"] == "https://synapse.dev"
    assert "GET" in options_res.headers["Access-Control-Allow-Methods"]

    # Regular GET with CORS headers
    get_res = app.handle_request(
        "GET",
        "/api/test",
        headers={"Origin": "https://synapse.dev"}
    )
    assert get_res.status_code == 200
    assert get_res.headers["Access-Control-Allow-Origin"] == "https://synapse.dev"


def test_web_jwt_authentication():
    """Verify zero-dependency JWT token creation, expiration, and jwt_required dependency."""
    secret = "super-secret-synapse-key"

    # 1. Successful token generation and decoding
    payload = {"sub": "user_42", "role": "admin"}
    token = create_access_token(payload, secret_key=secret, expires_in=300)
    decoded = decode_access_token(token, secret_key=secret)
    assert decoded["sub"] == "user_42"
    assert decoded["role"] == "admin"

    # 2. Tampered token rejection
    tampered = token[:-4] + "abcd"
    with pytest.raises(JWTError):
        decode_access_token(tampered, secret_key=secret)

    # 3. Expired token handling
    expired_token = create_access_token(payload, secret_key=secret, expires_in=-10)
    with pytest.raises(JWTExpiredError):
        decode_access_token(expired_token, secret_key=secret)

    # 4. Route integration with jwt_required dependency
    app = SynapseApp()

    @app.get("/profile")
    def profile(user: dict = Depends(jwt_required(secret))):
        return {"welcome": user["sub"], "role": user["role"]}

    # Without token -> 401
    unauth = app.handle_request("GET", "/profile")
    assert unauth.status_code == 401

    # With valid Bearer token -> 200
    auth = app.handle_request(
        "GET",
        "/profile",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert auth.status_code == 200
    assert auth.json()["welcome"] == "user_42"
    assert auth.json()["role"] == "admin"


def test_web_programmatic_handle_request():
    """Verify handle_request() parses query strings and returns structured Response."""
    app = SynapseApp()

    @app.get("/search")
    def search(q: str = "", limit: int = 10):
        return {"query": q, "limit": limit}

    res = app.handle_request("GET", "/search?q=machine-learning&limit=25")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "machine-learning"
    assert data["limit"] == 25
    assert "Content-Type" in res.headers
    assert "application/json" in res.headers["Content-Type"]


def test_web_middleware_chain():
    """Verify enterprise custom middleware pipeline execution order."""
    app = SynapseApp()
    execution_order = []

    def middleware_one(request: Request, call_next):
        execution_order.append("mw1_start")
        request.state["mw1"] = True
        response = call_next(request)
        execution_order.append("mw1_end")
        response.headers["X-Custom-Header"] = "Processed"
        return response

    def middleware_two(request: Request, call_next):
        execution_order.append("mw2_start")
        response = call_next(request)
        execution_order.append("mw2_end")
        return response

    app.add_middleware(middleware_one)
    app.add_middleware(middleware_two)

    @app.get("/pipeline")
    def endpoint(req: Request):
        execution_order.append("endpoint")
        assert req.state.get("mw1") is True
        return {"ok": True}

    res = app.handle_request("GET", "/pipeline")
    assert res.status_code == 200
    assert res.headers["X-Custom-Header"] == "Processed"
    assert execution_order == ["mw1_start", "mw2_start", "endpoint", "mw2_end", "mw1_end"]


def test_web_live_server_serve():
    """Verify app.serve() runs a live HTTP server and responds over real sockets."""
    app = SynapseApp()

    @app.get("/live-ping")
    def ping():
        return {"status": "pong", "time": time.time()}

    # Non-blocking server on port 8192
    server = app.serve(host="127.0.0.1", port=8192, blocking=False)
    try:
        time.sleep(0.3)
        req = urllib.request.Request("http://127.0.0.1:8192/live-ping")
        with urllib.request.urlopen(req, timeout=2.0) as res:
            assert res.status == 200
            body = json.loads(res.read().decode("utf-8"))
            assert body["status"] == "pong"
    finally:
        server.stop()


# =============================================================================
# 2. Active-Record ORM Tests
# =============================================================================

def test_orm_database_create_all_and_schema():
    """Verify Database, Field definitions, and create_all() table creation."""
    db = Database("sqlite:///:memory:")

    class Product(Model):
        _table_name = "products"
        _db = db
        title = Field(type=str, nullable=False)
        price = Field(type=float, default=0.0)
        is_active = Field(type=bool, default=True)

    db.create_all()

    # Verify table schema exists in SQLite master
    cursor = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='products'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "products"


def test_orm_active_record_create_and_get():
    """Verify Model.create() auto-increment primary key and Model.get(id)."""
    db = Database("sqlite:///:memory:")

    class Account(Model):
        _table_name = "accounts"
        _db = db
        email = Field(type=str, unique=True, nullable=False)
        balance = Field(type=float, default=100.0)

    db.create_all()

    acc1 = Account.create(email="alice@synapse.org", balance=250.0)
    assert acc1.id == 1
    assert acc1.email == "alice@synapse.org"
    assert acc1.balance == 250.0

    acc2 = Account.create(email="bob@synapse.org")
    assert acc2.id == 2
    assert acc2.balance == 100.0  # default applied

    # Fetch by primary key
    fetched = Account.get(1)
    assert fetched is not None
    assert fetched.id == 1
    assert fetched.email == "alice@synapse.org"

    # Non-existent ID returns None
    assert Account.get(999) is None


def test_orm_active_record_filter_all_first_count():
    """Verify Model.filter().all(), .first(), .count(), .order_by(), .limit()."""
    db = Database("sqlite:///:memory:")

    class Article(Model):
        _table_name = "articles"
        _db = db
        title = Field(type=str)
        category = Field(type=str)
        views = Field(type=int, default=0)

    db.create_all()

    Article.create(title="AI Runtime", category="tech", views=500)
    Article.create(title="Web Assembly", category="tech", views=1200)
    Article.create(title="Coffee Brewing", category="lifestyle", views=150)

    # Filter with .all()
    tech_articles = Article.filter(category="tech").all()
    assert len(tech_articles) == 2

    # Filter with .count()
    assert Article.filter(category="tech").count() == 2
    assert Article.filter(category="lifestyle").count() == 1

    # Filter with .first()
    first_tech = Article.filter(category="tech").first()
    assert first_tech is not None
    assert first_tech.title == "AI Runtime"

    # Order by views descending
    top_viewed = Article.filter().order_by("-views").first()
    assert top_viewed is not None
    assert top_viewed.title == "Web Assembly"
    assert top_viewed.views == 1200

    # Limit
    limited = Article.filter().limit(1).all()
    assert len(limited) == 1


def test_orm_active_record_save_update_and_delete():
    """Verify Model.save() updating rows and Model.delete() deleting rows."""
    db = Database("sqlite:///:memory:")

    class Customer(Model):
        _table_name = "customers"
        _db = db
        name = Field(type=str)
        tier = Field(type=str, default="bronze")

    db.create_all()

    cust = Customer.create(name="Charlie", tier="bronze")
    assert cust.id == 1

    # Update attributes and save()
    cust.tier = "gold"
    cust.save()

    # Re-fetch and assert updated
    updated_cust = Customer.get(1)
    assert updated_cust.tier == "gold"

    # Delete instance
    deleted = cust.delete()
    assert deleted is True
    assert Customer.get(1) is None
    assert Customer.filter().count() == 0


def test_orm_sql_injection_safety():
    """Verify parameterized ? bindings and column validation prevent SQL injection."""
    db = Database("sqlite:///:memory:")

    class SafeRecord(Model):
        _table_name = "safe_records"
        _db = db
        username = Field(type=str)

    db.create_all()
    SafeRecord.create(username="legit_user")

    # 1. Malicious string payload inside filter value: should NOT execute injection
    malicious_val = "legit_user' OR '1'='1"
    match = SafeRecord.filter(username=malicious_val).first()
    assert match is None  # Matches literal string, doesn't match legit_user

    # 2. Malicious column name injection: should raise ValueError before executing
    with pytest.raises(ValueError, match="Invalid query field"):
        SafeRecord.filter(**{"username = 1 OR 1=1; --": "test"}).all()


def test_orm_transactions():
    """Verify Database transaction commit and rollback."""
    db = Database("sqlite:///:memory:")

    class LogEntry(Model):
        _table_name = "log_entries"
        _db = db
        msg = Field(type=str)

    db.create_all()

    # 1. Successful transaction
    with db.transaction():
        LogEntry.create(msg="message 1")
        LogEntry.create(msg="message 2")

    assert LogEntry.filter().count() == 2

    # 2. Rolled back transaction on exception
    try:
        with db.transaction():
            LogEntry.create(msg="message 3")
            raise RuntimeError("Transaction aborted")
    except RuntimeError:
        pass

    assert LogEntry.filter().count() == 2
    assert LogEntry.filter(msg="message 3").first() is None


# =============================================================================
# 3. Asynchronous Task Queue Tests
# =============================================================================

def test_task_queue_delay_and_result():
    """Verify @queue.task decorator, .delay() call, and TaskResult.wait()."""
    queue = TaskQueue(num_workers=2, name="test-single")

    @queue.task
    def compute_square(n: int) -> int:
        return n * n

    try:
        task = compute_square.delay(7)
        assert task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

        res = task.wait(timeout=3.0)
        assert res == 49
        assert task.status == TaskStatus.SUCCESS
        assert task.is_successful is True
        assert task.is_ready is True
    finally:
        queue.shutdown(wait=True)


def test_task_queue_worker_pool_concurrency():
    """Verify TaskQueue(num_workers=4) executes multiple jobs in parallel."""
    num_workers = 4
    queue = TaskQueue(num_workers=num_workers, name="test-pool")

    @queue.task
    def concurrent_worker(job_id: int) -> int:
        time.sleep(0.1)
        return job_id * 10

    try:
        start_time = time.time()
        tasks = [concurrent_worker.delay(i) for i in range(8)]

        results = [t.wait(timeout=5.0) for t in tasks]
        elapsed = time.time() - start_time

        assert results == [i * 10 for i in range(8)]
        # 8 tasks of 0.1s on 4 workers should finish in ~0.2s-0.5s, well under 1.5s
        assert elapsed < 1.5
    finally:
        queue.shutdown(wait=True)


def test_task_queue_timeout_and_error_handling():
    """Verify wait(timeout) raises TimeoutError and task errors are captured with FAILED status."""
    queue = TaskQueue(num_workers=2, name="test-errors")

    @queue.task
    def slow_task():
        time.sleep(1.0)
        return "done"

    @queue.task
    def failing_task():
        raise ValueError("Intentional task failure")

    try:
        # Timeout test
        slow_res = slow_task.delay()
        with pytest.raises(TimeoutError):
            slow_res.wait(timeout=0.05)

        # Failure test
        fail_res = failing_task.delay()
        fail_res.wait(timeout=2.0)

        assert fail_res.status == TaskStatus.FAILED
        assert fail_res.is_failed is True
        assert isinstance(fail_res.error, ValueError)
        assert "Intentional task failure" in str(fail_res.error)

        # wait(raise_on_error=True) should re-raise
        with pytest.raises(ValueError, match="Intentional task failure"):
            fail_res.wait(timeout=0.5, raise_on_error=True)
    finally:
        queue.shutdown(wait=True)


# =============================================================================
# 4. Full-Stack End-to-End Integration Test
# =============================================================================

def test_fullstack_web_orm_task_integration():
    """
    End-to-end full-stack integration:
    - SynapseApp REST endpoint receives user registration payload
    - Active-Record ORM models persist user into SQLite
    - TaskQueue asynchronously dispatches background welcome email job
    - Verifies database persistence and background task completion
    """
    db = Database("sqlite:///:memory:")
    queue = TaskQueue(num_workers=2, name="fullstack-tasks")

    class Member(Model):
        _table_name = "members"
        _db = db
        username = Field(type=str, unique=True, nullable=False)
        email = Field(type=str, nullable=False)
        notifications_sent = Field(type=int, default=0)

    db.create_all()

    @queue.task
    def send_welcome_notification(member_id: int):
        time.sleep(0.05)
        member = Member.get(member_id)
        if member:
            member.notifications_sent += 1
            member.save()
            return f"Welcome sent to {member.email}"
        return "Member not found"

    app = SynapseApp()

    @app.post("/api/members")
    def register_member(body: dict):
        username = body.get("username")
        email = body.get("email")
        if not username or not email:
            raise HTTPException(status_code=400, detail="username and email required")

        new_member = Member.create(username=username, email=email)
        # Dispatch background notification task
        task_res = send_welcome_notification.delay(new_member.id)

        return {
            "member_id": new_member.id,
            "username": new_member.username,
            "task_id": task_res.id
        }, 201

    try:
        # Submit registration request
        res = app.handle_request(
            "POST",
            "/api/members",
            body={"username": "dev_hero", "email": "hero@synapse.tech"}
        )
        assert res.status_code == 201
        data = res.json()
        member_id = data["member_id"]
        task_id = data["task_id"]

        # Verify member in database
        saved_member = Member.get(member_id)
        assert saved_member is not None
        assert saved_member.username == "dev_hero"

        # Wait for background notification task to complete
        bg_task = queue.get_task(task_id)
        assert bg_task is not None
        task_output = bg_task.wait(timeout=3.0)
        assert "Welcome sent to hero@synapse.tech" in task_output

        # Verify database record updated by background worker
        reloaded_member = Member.get(member_id)
        assert reloaded_member.notifications_sent == 1

    finally:
        queue.shutdown(wait=True)
