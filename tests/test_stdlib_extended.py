"""Comprehensive Test Suite for Synapse Extended Standard Library (HTTP and Regex)."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
from typing import Generator
import urllib.parse
import pytest

from synapse.stdlib import std, StandardLibrary
import synapse.stdlib.http as s_http
import synapse.stdlib.regex as s_regex
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_source(source: str, vm: VirtualMachine = None) -> VirtualMachine:
    """Helper to compile and execute Synapse source code in VM."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    if vm is None:
        vm = VirtualMachine()
    vm.execute(code)
    return vm


# =============================================================================
# Mock HTTP Server Fixture
# =============================================================================

class MockHTTPHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress stdout logging during tests
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/json":
            payload = json.dumps({"status": "success", "message": "hello synapse"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Custom-Echo", self.headers.get("X-Custom-Header", "none"))
            self.end_headers()
            self.wfile.write(payload)

        elif path == "/text":
            payload = b"Synapse Native HTTP Client"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif path == "/utf8":
            text = "Merhaba Dünya! 🚀 Türkçe karakterler: ç, ğ, ı, ö, ş, ü"
            payload = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif path == "/query":
            flat_query = {k: v[0] if len(v) == 1 else v for k, v in query.items()}
            payload = json.dumps({"params": flat_query}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/json")
            self.end_headers()

        elif path == "/slow":
            # Sleep 0.4s to trigger timeout tests
            time.sleep(0.4)
            payload = b"slow response"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif path == "/error500":
            payload = json.dumps({"error": "Internal Server Error"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            payload = b"Not Found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b""
        content_type = self.headers.get("Content-Type", "")

        if self.path == "/echo_json":
            data = json.loads(post_body.decode("utf-8")) if post_body else {}
            response_data = {"received": data, "type": "json"}
            payload = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/echo_form":
            parsed_form = urllib.parse.parse_qs(post_body.decode("utf-8"))
            flat = {k: v[0] if len(v) == 1 else v for k, v in parsed_form.items()}
            payload = json.dumps({"form": flat}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            payload = b"Created"
            self.send_response(201)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_PUT(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""
        data = json.loads(body.decode("utf-8")) if body else {}
        payload = json.dumps({"updated": True, "data": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_DELETE(self):
        payload = json.dumps({"deleted": True, "path": self.path}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Head-Check", "true")
        self.end_headers()


@pytest.fixture(scope="module")
def mock_server() -> Generator[str, None, None]:
    """Starts a local mock HTTP server on an ephemeral free port."""
    server = http.server.HTTPServer(("127.0.0.1", 0), MockHTTPHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


# =============================================================================
# 1. Native HTTP Client Module Tests
# =============================================================================

def test_http_get_json(mock_server: str):
    url = f"{mock_server}/json"
    resp = s_http.get(url)

    assert isinstance(resp, s_http.HttpResponse)
    assert resp.status_code == 200
    assert resp.ok is True
    assert bool(resp) is True
    assert resp.error is None
    assert "application/json" in resp.headers["content-type"]
    assert "application/json" in resp.headers["Content-Type"]

    data = resp.json()
    assert isinstance(data, dict)
    assert data["status"] == "success"
    assert data["message"] == "hello synapse"
    assert "200 OK" in repr(resp)


def test_http_get_text(mock_server: str):
    url = f"{mock_server}/text"
    resp = s_http.get(url)

    assert resp.status_code == 200
    assert resp.ok is True
    assert resp.text == "Synapse Native HTTP Client"
    assert resp.body == b"Synapse Native HTTP Client"


def test_http_get_utf8(mock_server: str):
    url = f"{mock_server}/utf8"
    resp = s_http.get(url)

    assert resp.status_code == 200
    assert resp.ok is True
    assert "Merhaba Dünya! 🚀" in resp.text
    assert "Türkçe karakterler" in resp.text


def test_http_get_query_params(mock_server: str):
    url = f"{mock_server}/query"
    # Query passed as dictionary
    resp = s_http.get(url, query={"q": "synapse", "page": "1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["params"]["q"] == "synapse"
    assert data["params"]["page"] == "1"

    # Query passed as string
    resp2 = s_http.get(url, query="filter=active&sort=desc")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["params"]["filter"] == "active"
    assert data2["params"]["sort"] == "desc"


def test_http_get_custom_headers(mock_server: str):
    url = f"{mock_server}/json"
    resp = s_http.get(url, headers={"X-Custom-Header": "Synapse-Test-Header"})
    assert resp.status_code == 200
    assert resp.headers["X-Custom-Echo"] == "Synapse-Test-Header"
    assert resp.headers.get("x-custom-echo") == "Synapse-Test-Header"


def test_http_get_redirect(mock_server: str):
    url = f"{mock_server}/redirect"
    resp = s_http.get(url)
    assert resp.status_code == 200
    assert resp.ok is True
    # Final URL should point to /json after following redirect
    assert resp.url.endswith("/json")
    data = resp.json()
    assert data["status"] == "success"


def test_http_post_json(mock_server: str):
    url = f"{mock_server}/echo_json"
    payload = {"model": "synapse-ai", "temperature": 0.7, "stream": False}
    resp = s_http.post(url, json=payload)

    assert resp.status_code == 200
    assert resp.ok is True
    data = resp.json()
    assert data["received"] == payload
    assert data["type"] == "json"


def test_http_post_form_data(mock_server: str):
    url = f"{mock_server}/echo_form"
    form = {"username": "agent007", "action": "login"}
    resp = s_http.post(url, data=form)

    assert resp.status_code == 200
    assert resp.ok is True
    data = resp.json()
    assert data["form"]["username"] == "agent007"
    assert data["form"]["action"] == "login"


def test_http_put(mock_server: str):
    url = f"{mock_server}/update"
    resp = s_http.put(url, json={"id": 42, "status": "updated"})

    assert resp.status_code == 200
    assert resp.ok is True
    data = resp.json()
    assert data["updated"] is True
    assert data["data"]["id"] == 42


def test_http_delete(mock_server: str):
    url = f"{mock_server}/remove/item_99"
    resp = s_http.delete(url)

    assert resp.status_code == 200
    assert resp.ok is True
    data = resp.json()
    assert data["deleted"] is True
    assert "/remove/item_99" in data["path"]


def test_http_head(mock_server: str):
    url = f"{mock_server}/json"
    resp = s_http.head(url)

    assert resp.status_code == 200
    assert resp.ok is True
    assert resp.headers["X-Head-Check"] == "true"
    assert resp.body == b""


def test_http_error_responses(mock_server: str):
    # 404 Not Found
    resp_404 = s_http.get(f"{mock_server}/non_existent_route")
    assert resp_404.status_code == 404
    assert resp_404.ok is False
    assert bool(resp_404) is False
    assert resp_404.text == "Not Found"

    # 500 Server Error
    resp_500 = s_http.get(f"{mock_server}/error500")
    assert resp_500.status_code == 500
    assert resp_500.ok is False
    data_500 = resp_500.json()
    assert data_500["error"] == "Internal Server Error"


def test_http_timeout_graceful(mock_server: str):
    # Route /slow delays for 0.4s; setting timeout=0.05 will trigger a timeout
    resp = s_http.get(f"{mock_server}/slow", timeout=0.05)
    assert resp.status_code == 408
    assert resp.ok is False
    assert resp.error is not None
    assert "timed out" in resp.error.lower() or "timeout" in resp.error.lower()


def test_http_connection_refused_graceful():
    # Connecting to an unassigned port on localhost (may return 408 on Windows or 0)
    resp = s_http.get("http://127.0.0.1:59999", timeout=0.1)
    assert resp.status_code in (0, 408)
    assert resp.ok is False
    assert resp.error is not None

    # Host resolution failure (DNS error returns 0)
    resp2 = s_http.get("http://non-existent-domain-123456789.xyz", timeout=0.5)
    assert resp2.status_code == 0
    assert resp2.ok is False
    assert resp2.error is not None


def test_http_response_empty_json():
    r = s_http.HttpResponse(status_code=200, body=b"")
    assert r.json() is None

    r_invalid = s_http.HttpResponse(status_code=200, body=b"not-json")
    with pytest.raises(ValueError):
        r_invalid.json()


# =============================================================================
# 2. Native Regular Expression Module Tests
# =============================================================================

def test_regex_match():
    # Full match at beginning
    m = s_regex.match(r"(\w+)\s+(\d+)", "synapse 2026")
    assert m is not None
    assert m["match"] == "synapse 2026"
    assert m["groups"] == ["synapse", "2026"]
    assert m["start"] == 0
    assert m["end"] == 12

    # Match fails when pattern is not at the start
    m2 = s_regex.match(r"\d+", "abc 123")
    assert m2 is None

    # Flags support
    m_case = s_regex.match(r"synapse", "SYNAPSE", flags=s_regex.IGNORECASE)
    assert m_case is not None
    assert m_case["match"] == "SYNAPSE"

    # String shorthand flag support
    m_case_str = s_regex.match(r"synapse", "SYNAPSE", flags="i")
    assert m_case_str is not None
    assert m_case_str["match"] == "SYNAPSE"


def test_regex_search():
    # Finds match anywhere in the string
    m = s_regex.search(r"\d+", "item 42 is available")
    assert m is not None
    assert m["match"] == "42"
    assert m["start"] == 5
    assert m["end"] == 7
    assert m["groups"] == []

    # Named groups
    m_named = s_regex.search(r"(?P<key>\w+)=(?P<val>\d+)", "config: port=8080 active")
    assert m_named is not None
    assert m_named["match"] == "port=8080"
    assert m_named["named_groups"] == {"key": "port", "val": "8080"}

    # Non match
    assert s_regex.search(r"xyz", "abc def") is None


def test_regex_find_all():
    text = "Extract 100, then 200, and finally 3500 items."
    matches = s_regex.find_all(r"\d+", text)
    assert matches == ["100", "200", "3500"]

    # No match returns empty list
    assert s_regex.find_all(r"[A-Z]{5}", text) == []

    # Patterns with groups still return list of matched strings
    email_text = "Contact us at info@synapse.ai or dev@google.com"
    emails = s_regex.find_all(r"(\w+)@([\w\.]+)", email_text)
    assert emails == ["info@synapse.ai", "dev@google.com"]


def test_regex_replace():
    text = "The price is $10 and tax is $2."
    result = s_regex.replace(r"\$\d+", "[REDACTED]", text)
    assert result == "The price is [REDACTED] and tax is [REDACTED]."

    # Replace with count limit
    result_once = s_regex.replace(r"\$\d+", "[REDACTED]", text, count=1)
    assert result_once == "The price is [REDACTED] and tax is $2."

    # Replace with function / lambda or replacement pattern
    swapped = s_regex.replace(r"(\w+)\s+(\w+)", r"\2 \1", "hello world")
    assert swapped == "world hello"


def test_regex_split():
    text = "apple, banana; orange,   grape"
    parts = s_regex.split(r"[,;]\s*", text)
    assert parts == ["apple", "banana", "orange", "grape"]

    # Split with maxsplit
    limited = s_regex.split(r"\s+", "one two three four", maxsplit=2)
    assert limited == ["one", "two", "three four"]


def test_regex_helpers():
    assert s_regex.is_match(r"\d{3}", "abc123def") is True
    assert s_regex.is_match(r"^\d+$", "abc123def") is False

    escaped = s_regex.escape("price is $100 (usd)")
    assert "\\" in escaped
    assert s_regex.is_match(escaped, "price is $100 (usd)") is True


# =============================================================================
# 3. Synapse VM Script Execution with std.http & std.regex
# =============================================================================

def test_vm_std_http_execution(mock_server: str):
    source = f"""
let target = "{mock_server}/json"
let resp = std.http.get(target)
let status = resp.status_code
let is_ok = resp.ok
let data = resp.json()
let status_msg = data.status
"""
    vm = run_source(source)
    assert vm.globals["status"] == 200
    assert vm.globals["is_ok"] is True
    assert vm.globals["status_msg"] == "success"


def test_vm_std_http_post_execution(mock_server: str):
    source = f"""
let target = "{mock_server}/echo_json"
let payload = {{"name": "SynapseVM", "version": 2}}
let resp = std.http.post(target, json=payload)
let status = resp.status_code
let is_ok = resp.ok
let body_data = resp.json()
let received_name = body_data.received.name
"""
    vm = run_source(source)
    assert vm.globals["status"] == 200
    assert vm.globals["is_ok"] is True
    assert vm.globals["received_name"] == "SynapseVM"


def test_vm_std_regex_execution():
    source = """
let m = std.regex.match("([a-z]+)-([0-9]+)", "synapse-42")
let full_match = m.match
let groups = m.groups
let prefix = groups[0]
let num = groups[1]

let s = std.regex.search("[0-9]+", "order id: 98765 completed")
let search_match = s.match
let search_start = s.start

let numbers = std.regex.find_all("[0-9]+", "a10b20c30")
let replaced = std.regex.replace("[0-9]", "*", "code123")
let parts = std.regex.split("-", "alpha-beta-gamma")
"""
    vm = run_source(source)
    assert vm.globals["full_match"] == "synapse-42"
    assert vm.globals["prefix"] == "synapse"
    assert vm.globals["num"] == "42"
    assert vm.globals["search_match"] == "98765"
    assert vm.globals["search_start"] == 10
    assert vm.globals["numbers"] == ["10", "20", "30"]
    assert vm.globals["replaced"] == "code***"
    assert vm.globals["parts"] == ["alpha", "beta", "gamma"]


def test_vm_global_shortcuts_execution(mock_server: str):
    source = f"""
let get_target = "{mock_server}/json"
let g_resp = http_get(get_target)
let g_status = g_resp.status_code

let post_target = "{mock_server}/echo_json"
let p_resp = http_post(post_target, json={{"ping": "pong"}})
let p_ok = p_resp.ok

let reg_m = regex_match("[A-Z]+", "SYNAPSE")
let reg_m_val = reg_m.match

let reg_s = regex_search("version-[0-9]+", "system version-3 ready")
let reg_s_val = reg_s.match

let reg_rep = regex_replace("bad", "good", "bad vibes only")
"""
    vm = run_source(source)
    assert vm.globals["g_status"] == 200
    assert vm.globals["p_ok"] is True
    assert vm.globals["reg_m_val"] == "SYNAPSE"
    assert vm.globals["reg_s_val"] == "version-3"
    assert vm.globals["reg_rep"] == "good vibes only"
