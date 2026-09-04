import os
import re
import json
import time
import struct
import base64
import hashlib
import inspect
import threading
from typing import Any, Callable, Optional, Sequence, Union, Iterable, Iterator
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Threading server model for concurrent request handling
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    import socketserver

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# RFC 6455 WebSocket Opcodes
OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA


class SynapseRequest:
    """Represents an incoming HTTP request in Synapse."""
    def __init__(self, path: str, method: str, query: dict[str, Any], headers: dict[str, str], body: Any):
        self.path = path
        self.method = method
        self.query = query
        self.headers = headers
        self.body = body

    def __repr__(self) -> str:
        return f"<SynapseRequest {self.method} {self.path}>"


class SynapseResponse:
    """Represents a standard HTTP response in Synapse."""
    def __init__(
        self,
        body: Any,
        status_code: int = 200,
        content_type: str = "application/json",
        headers: Optional[dict[str, str]] = None
    ):
        self.body = body
        self.status_code = status_code
        self.content_type = content_type
        self.headers = headers or {}

    def __repr__(self) -> str:
        return f"<SynapseResponse {self.status_code} {self.content_type}>"


class SynapseSSEResponse:
    """
    Server-Sent Events (SSE) streaming response.
    Streams events in standard 'text/event-stream' format without buffering.
    """
    def __init__(
        self,
        generator_or_iterable: Any,
        status_code: int = 200,
        headers: Optional[dict[str, str]] = None
    ):
        self.generator = generator_or_iterable
        self.status_code = status_code
        self.headers = headers or {}

    def __iter__(self) -> Iterator[Any]:
        if hasattr(self.generator, "__iter__"):
            return iter(self.generator)
        return iter([self.generator])

    def __repr__(self) -> str:
        return f"<SynapseSSEResponse {self.status_code}>"


class SynapseWebSocketRoute:
    """Wrapper to mark a route handler explicitly as a WebSocket endpoint."""
    def __init__(self, handler: Callable[..., Any]):
        self.handler = handler
        self.is_websocket = True

    def __call__(self, *args, **kwargs) -> Any:
        return self.handler(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<SynapseWebSocketRoute handler={getattr(self.handler, '__name__', str(self.handler))}>"


class SynapseWebSocket:
    """
    RFC 6455 compliant lightweight WebSocket connection handler.
    Provides frame encoding/decoding for text frames (opcode 0x1),
    masked client frames, unmasked server frames, ping/pong and close handshakes.
    """
    def __init__(
        self,
        sock: Any,
        rfile: Optional[Any] = None,
        path: str = "",
        headers: Optional[dict[str, str]] = None,
        query: Optional[dict[str, Any]] = None
    ):
        self.sock = sock
        self.rfile = rfile
        self.path = path
        self.headers = headers or {}
        self.query = query or {}
        self.closed = False
        self._lock = threading.Lock()

    def _read_exact(self, num_bytes: int) -> bytes:
        """Reads exactly num_bytes from buffer or socket."""
        buf = bytearray()
        while len(buf) < num_bytes:
            needed = num_bytes - len(buf)
            if self.rfile is not None:
                chunk = self.rfile.read(needed)
            else:
                chunk = self.sock.recv(needed)
            if not chunk:
                raise ConnectionResetError("WebSocket connection closed prematurely")
            buf.extend(chunk)
        return bytes(buf)

    def recv_text(self) -> Optional[str]:
        """
        Receives the next text message from the WebSocket.
        Handles masked client frames, fragments, ping/pong and close frames.
        Returns None when the socket is closed or disconnected.
        """
        fragments = bytearray()
        target_opcode = None

        while not self.closed:
            try:
                head = self._read_exact(2)
            except (ConnectionResetError, BrokenPipeError, TimeoutError, OSError):
                self.closed = True
                return None

            b0, b1 = head[0], head[1]
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            is_masked = bool(b1 & 0x80)
            payload_len = b1 & 0x7F

            if payload_len == 126:
                ext = self._read_exact(2)
                payload_len = struct.unpack("!H", ext)[0]
            elif payload_len == 127:
                ext = self._read_exact(8)
                payload_len = struct.unpack("!Q", ext)[0]

            mask_key = None
            if is_masked:
                mask_key = self._read_exact(4)

            raw_payload = self._read_exact(payload_len) if payload_len > 0 else b""

            if is_masked and mask_key:
                payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(raw_payload))
            else:
                payload = raw_payload

            # Handle control frames
            if opcode == OPCODE_CLOSE:
                self.closed = True
                try:
                    # Echo close frame if not already closed
                    self._send_frame(OPCODE_CLOSE, payload)
                except Exception:
                    pass
                return None

            elif opcode == OPCODE_PING:
                # Reply with pong
                try:
                    self._send_frame(OPCODE_PONG, payload)
                except Exception:
                    pass
                continue

            elif opcode == OPCODE_PONG:
                continue

            # Handle data frames
            if opcode in (OPCODE_TEXT, OPCODE_BINARY):
                target_opcode = opcode
                fragments.extend(payload)
                if fin:
                    if target_opcode == OPCODE_TEXT:
                        return fragments.decode("utf-8", errors="replace")
                    else:
                        return fragments.decode("utf-8", errors="replace")

            elif opcode == OPCODE_CONTINUATION:
                fragments.extend(payload)
                if fin:
                    if target_opcode == OPCODE_TEXT:
                        return fragments.decode("utf-8", errors="replace")
                    else:
                        return fragments.decode("utf-8", errors="replace")

        return None

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        """Encodes and sends an unmasked RFC 6455 frame to the client."""
        if self.closed and opcode != OPCODE_CLOSE:
            raise ConnectionResetError("WebSocket is closed")

        payload_len = len(payload)
        b0 = 0x80 | (opcode & 0x0F)  # FIN=1

        # Server to client frames MUST be unmasked (mask bit = 0)
        if payload_len <= 125:
            header = bytes([b0, payload_len])
        elif payload_len <= 65535:
            header = struct.pack("!BBH", b0, 126, payload_len)
        else:
            header = struct.pack("!BBQ", b0, 127, payload_len)

        with self._lock:
            self.sock.sendall(header + payload)

    def send_text(self, msg: str) -> None:
        """Sends a text message to the WebSocket client."""
        data = msg.encode("utf-8")
        self._send_frame(OPCODE_TEXT, data)

    def send_json(self, data: Any) -> None:
        """Serializes and sends a JSON text frame."""
        self.send_text(json.dumps(data, ensure_ascii=False))

    def recv_json(self) -> Optional[Any]:
        """Receives a text message and parses it as JSON."""
        txt = self.recv_text()
        if txt is None:
            return None
        return json.loads(txt)

    def close(self, code: int = 1000, reason: str = "") -> None:
        """Sends a WebSocket close frame and closes the underlying socket."""
        if self.closed:
            return
        self.closed = True
        try:
            payload = struct.pack("!H", code) + reason.encode("utf-8")
            self._send_frame(OPCODE_CLOSE, payload)
        except Exception:
            pass
        finally:
            try:
                self.sock.close()
            except Exception:
                pass

    def __iter__(self) -> Iterator[str]:
        """Allows iterating over incoming text messages: for msg in ws: ..."""
        while not self.closed:
            msg = self.recv_text()
            if msg is None:
                break
            yield msg

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"<SynapseWebSocket path='{self.path}' state={state}>"


def _compute_ws_accept(key: str) -> str:
    """Computes Sec-WebSocket-Accept token according to RFC 6455."""
    combined = (key.strip() + WS_GUID).encode("utf-8")
    sha1_digest = hashlib.sha1(combined).digest()
    return base64.b64encode(sha1_digest).decode("ascii")


def _py_type_to_schema(t: Any) -> dict[str, Any]:
    """Maps Python types, typing constructs, and model classes to OpenAPI 3.1 JSON schemas."""
    if t is None or t is inspect.Parameter.empty or t is inspect.Signature.empty:
        return {"type": "object"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is str:
        return {"type": "string"}
    if t is bool:
        return {"type": "boolean"}
    if t is bytes:
        return {"type": "string", "contentMediaType": "application/octet-stream"}
    if t is list:
        return {"type": "array"}
    if t is dict:
        return {"type": "object"}

    # Handle typing primitives (list[T], dict[K, V], Optional[T], Union[...])
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", None)

    if origin in (list, Sequence, set, tuple, Iterable):
        if args and len(args) > 0:
            return {"type": "array", "items": _py_type_to_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if origin is Union:
        # Check for Optional[T] which is Union[T, NoneType]
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1 and len(args) == 2:
            base_schema = _py_type_to_schema(non_none_args[0])
            base_type = base_schema.get("type")
            if isinstance(base_type, str):
                return {**base_schema, "type": [base_type, "null"]}
            return base_schema
        return {"anyOf": [_py_type_to_schema(a) for a in args]}

    # Pydantic v2 / v1 model compatibility
    if hasattr(t, "model_json_schema") and callable(t.model_json_schema):
        return t.model_json_schema()
    if hasattr(t, "schema") and callable(t.schema):
        return t.schema()

    # Generic dataclass or class with __annotations__
    if hasattr(t, "__annotations__") and isinstance(t.__annotations__, dict):
        props = {}
        for prop_name, prop_type in t.__annotations__.items():
            props[prop_name] = _py_type_to_schema(prop_type)
        return {
            "type": "object",
            "properties": props
        }

    return {"type": "object"}


def generate_openapi_spec(
    server: Any,
    title: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None
) -> dict[str, Any]:
    """
    Inspects registered routes (`routes` and `ws_routes`) on a SynapseServer or dict,
    extracts HTTP method, clean path, docstrings, parameters, request body, and
    return types to generate a valid OpenAPI 3.1.0 specification dictionary.
    """
    if isinstance(server, dict):
        routes = server
        ws_routes = {}
        server_title = title or "Synapse API"
        server_version = version or "1.0.0"
        server_desc = description or "Synapse High-Performance AI & Web Runtime API"
    else:
        routes = getattr(server, "routes", {})
        ws_routes = getattr(server, "ws_routes", {})
        server_title = title or getattr(server, "title", "Synapse API")
        server_version = version or getattr(server, "version", "1.0.0")
        server_desc = description or getattr(server, "description", "Synapse High-Performance AI & Web Runtime API")

    paths: dict[str, dict[str, Any]] = {}
    schemas: dict[str, Any] = {}

    # 1. Process HTTP Routes
    for route_key, handler in routes.items():
        # Exclude built-in docs endpoints from spec
        if route_key in ("/docs", "/openapi.json", "GET /docs", "GET /openapi.json"):
            continue

        # Extract HTTP method and route path
        route_key_clean = route_key.strip()
        parts = route_key_clean.split(" ", 1)
        if len(parts) == 2 and parts[0].upper() in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD", "TRACE"):
            method = parts[0].lower()
            raw_path = parts[1].strip()
        else:
            raw_path = route_key_clean
            if hasattr(handler, "method") and isinstance(handler.method, str):
                method = handler.method.lower()
            elif hasattr(handler, "methods") and isinstance(handler.methods, (list, tuple)) and handler.methods:
                method = handler.methods[0].lower()
            else:
                fn_name = getattr(handler, "__name__", "").lower()
                if any(fn_name.startswith(p) for p in ("get_", "list_", "read_", "fetch_", "search_", "find_")):
                    method = "get"
                elif any(fn_name.startswith(p) for p in ("post_", "create_", "add_", "insert_", "chat_", "send_")):
                    method = "post"
                elif any(fn_name.startswith(p) for p in ("put_", "update_", "set_")):
                    method = "put"
                elif any(fn_name.startswith(p) for p in ("delete_", "remove_")):
                    method = "delete"
                elif fn_name.startswith("patch_"):
                    method = "patch"
                else:
                    doc_check = inspect.getdoc(handler) or getattr(handler, "__doc__", "") or ""
                    doc_check_lower = doc_check.lower()
                    if "@method post" in doc_check_lower or "method: post" in doc_check_lower:
                        method = "post"
                    elif "@method put" in doc_check_lower or "method: put" in doc_check_lower:
                        method = "put"
                    elif "@method delete" in doc_check_lower or "method: delete" in doc_check_lower:
                        method = "delete"
                    elif "@method patch" in doc_check_lower or "method: patch" in doc_check_lower:
                        method = "patch"
                    else:
                        method = "get"

        # Normalize path formatting: :param and <param> -> {param}
        clean_path = raw_path
        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path
        clean_path = re.sub(r':([a-zA-Z_0-9]+)', r'{\1}', clean_path)
        clean_path = re.sub(r'<([a-zA-Z_0-9]+)>', r'{\1}', clean_path)

        # Detect path variables
        path_vars = re.findall(r'\{([a-zA-Z_0-9]+)\}', clean_path)

        # Extract docstring: first line = summary, subsequent = description
        doc = inspect.getdoc(handler) or getattr(handler, "__doc__", "") or ""
        doc_lines = [l.strip() for l in doc.splitlines() if l.strip()]
        summary = doc_lines[0] if doc_lines else f"{method.upper()} {clean_path}"
        description_text = "\n".join(doc_lines[1:]) if len(doc_lines) > 1 else summary

        # Inspect handler signature
        actual_fn = handler.handler if isinstance(handler, SynapseWebSocketRoute) else handler
        sig = None
        if callable(actual_fn):
            try:
                sig = inspect.signature(actual_fn)
            except Exception:
                sig = None

        parameters = []
        # Path parameters
        for pv in path_vars:
            pv_schema = {"type": "string"}
            if sig and pv in sig.parameters:
                pv_annot = sig.parameters[pv].annotation
                if pv_annot is not inspect.Parameter.empty:
                    pv_schema = _py_type_to_schema(pv_annot)
            parameters.append({
                "name": pv,
                "in": "path",
                "required": True,
                "schema": pv_schema
            })

        # Query parameters and request body
        body_schema = None
        if sig:
            for pname, param in sig.parameters.items():
                if pname in ("req", "request", "ws", "websocket", "self", "cls"):
                    if param.annotation not in (inspect.Parameter.empty, Any, SynapseRequest, SynapseWebSocket):
                        body_schema = _py_type_to_schema(param.annotation)
                    continue
                if pname in path_vars:
                    continue
                if pname in ("body", "payload", "data"):
                    if param.annotation is not inspect.Parameter.empty:
                        body_schema = _py_type_to_schema(param.annotation)
                    continue

                p_required = (param.default is inspect.Parameter.empty)
                p_schema = _py_type_to_schema(param.annotation)
                parameters.append({
                    "name": pname,
                    "in": "query",
                    "required": p_required,
                    "schema": p_schema
                })

        # Request body for mutations
        request_body = None
        if method in ("post", "put", "patch"):
            request_body = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": body_schema or {"type": "object"}
                    }
                }
            }

        # Response schema
        ret_annot = sig.return_annotation if (sig and sig.return_annotation is not inspect.Signature.empty) else None
        if ret_annot is SynapseSSEResponse or inspect.isgeneratorfunction(actual_fn):
            resp_content = {
                "text/event-stream": {
                    "schema": {"type": "string"}
                }
            }
        elif ret_annot is str:
            resp_content = {
                "text/plain": {
                    "schema": {"type": "string"}
                }
            }
        else:
            resp_schema = _py_type_to_schema(ret_annot) if ret_annot else {"type": "object"}
            resp_content = {
                "application/json": {
                    "schema": resp_schema
                }
            }

        op_id = f"{method}_{clean_path.strip('/').replace('/', '_').replace('{', '').replace('}', '') or 'root'}"
        op_obj: dict[str, Any] = {
            "summary": summary,
            "description": description_text,
            "operationId": op_id,
            "responses": {
                "200": {
                    "description": "Successful Response",
                    "content": resp_content
                }
            }
        }
        if parameters:
            op_obj["parameters"] = parameters
        if request_body:
            op_obj["requestBody"] = request_body

        if clean_path not in paths:
            paths[clean_path] = {}
        paths[clean_path][method] = op_obj

    # 2. Process WebSocket Routes
    for ws_path, ws_handler in ws_routes.items():
        clean_ws = ws_path.strip()
        if not clean_ws.startswith("/"):
            clean_ws = "/" + clean_ws
        clean_ws = re.sub(r':([a-zA-Z_0-9]+)', r'{\1}', clean_ws)
        clean_ws = re.sub(r'<([a-zA-Z_0-9]+)>', r'{\1}', clean_ws)

        ws_doc = inspect.getdoc(ws_handler) or getattr(ws_handler, "__doc__", "") or ""
        ws_lines = [l.strip() for l in ws_doc.splitlines() if l.strip()]
        summary = ws_lines[0] if ws_lines else f"WebSocket Endpoint for {clean_ws}"
        description_text = "\n".join(ws_lines[1:]) if len(ws_lines) > 1 else "RFC 6455 compliant WebSocket endpoint."

        ws_op = {
            "summary": summary,
            "description": description_text,
            "tags": ["WebSockets"],
            "operationId": f"ws_{clean_ws.strip('/').replace('/', '_').replace('{', '').replace('}', '') or 'root'}",
            "responses": {
                "101": {
                    "description": "Switching Protocols to WebSocket (RFC 6455)"
                }
            }
        }
        if clean_ws not in paths:
            paths[clean_ws] = {}
        paths[clean_ws]["get"] = ws_op

    return {
        "openapi": "3.1.0",
        "info": {
            "title": server_title,
            "version": server_version,
            "description": server_desc,
        },
        "paths": paths,
        "components": {
            "schemas": schemas
        }
    }


def render_swagger_ui_html(
    openapi_url: str = "/openapi.json",
    title: str = "Synapse API Docs"
) -> str:
    """
    Renders an interactive Swagger UI HTML page styled with a modern minimalist dark theme.
    Loads Swagger UI from CDN with a built-in offline fallback that fetches openapi.json.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #09090b;
      color: #fafafa;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    .swagger-ui .topbar {{ display: none !important; }}
    .synapse-nav {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 32px;
      background: #18181b;
      border-bottom: 1px solid #27272a;
    }}
    .synapse-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 18px;
      font-weight: 600;
      color: #fafafa;
    }}
    .synapse-badge {{
      padding: 2px 8px;
      background: #27272a;
      border: 1px solid #3f3f46;
      border-radius: 6px;
      font-size: 12px;
      color: #a1a1aa;
    }}
    .synapse-link {{
      color: #38bdf8;
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
      padding: 6px 12px;
      border: 1px solid #27272a;
      border-radius: 6px;
      background: #09090b;
    }}
    .synapse-link:hover {{
      background: #27272a;
      text-decoration: none;
    }}
    #swagger-ui {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    /* Offline fallback container */
    #offline-fallback {{
      display: none;
      max-width: 1000px;
      margin: 40px auto;
      padding: 32px;
      background: #18181b;
      border: 1px solid #27272a;
      border-radius: 12px;
    }}
    .route-item {{
      margin-top: 14px;
      padding: 16px;
      background: #09090b;
      border: 1px solid #27272a;
      border-radius: 8px;
    }}
    .method-badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .badge-get {{ background: #0284c7; color: white; }}
    .badge-post {{ background: #16a34a; color: white; }}
    .badge-put {{ background: #d97706; color: white; }}
    .badge-delete {{ background: #dc2626; color: white; }}
    .badge-patch {{ background: #9333ea; color: white; }}
    .badge-ws {{ background: #4f46e5; color: white; }}
  </style>
</head>
<body>
  <nav class="synapse-nav">
    <div class="synapse-brand">
      <span>⚡ {title}</span>
      <span class="synapse-badge">OpenAPI 3.1.0</span>
    </div>
    <div>
      <a class="synapse-link" href="{openapi_url}" target="_blank">View OpenAPI JSON</a>
    </div>
  </nav>

  <div id="swagger-ui"></div>

  <div id="offline-fallback">
    <h2 style="margin-top: 0; color: #fafafa;">Synapse Native API Catalog (Offline Mode)</h2>
    <p style="color: #a1a1aa; font-size: 14px;">Swagger UI CDN is unreachable or offline. Loaded local endpoints dynamically from <code>{openapi_url}</code>.</p>
    <div id="fallback-list">Loading endpoints...</div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js" onerror="renderFallback()"></script>
  <script>
    function renderFallback() {{
      var uiDiv = document.getElementById("swagger-ui");
      if (uiDiv) uiDiv.style.display = "none";
      var fb = document.getElementById("offline-fallback");
      if (fb) fb.style.display = "block";
      fetch("{openapi_url}")
        .then(function(res) {{ return res.json(); }})
        .then(function(spec) {{
          var listDiv = document.getElementById("fallback-list");
          listDiv.innerHTML = "";
          var paths = spec.paths || {{}};
          for (var p in paths) {{
            for (var m in paths[p]) {{
              var op = paths[p][m];
              var card = document.createElement("div");
              card.className = "route-item";
              var badgeClass = "badge-" + m.toLowerCase();
              card.innerHTML = "<span class='method-badge " + badgeClass + "'>" + m.toUpperCase() + "</span> " +
                               "<strong style='margin-left: 10px; font-family: monospace; font-size: 14px;'>" + p + "</strong>" +
                               "<div style='color: #a1a1aa; font-size: 13px; margin-top: 8px;'>" + (op.summary || op.description || "No description provided.") + "</div>";
              listDiv.appendChild(card);
            }}
          }}
        }})
        .catch(function(err) {{
          var listDiv = document.getElementById("fallback-list");
          if (listDiv) listDiv.innerHTML = "<p style='color: #ef4444;'>Failed to load OpenAPI spec: " + err + "</p>";
        }});
    }}

    window.addEventListener("DOMContentLoaded", function() {{
      setTimeout(function() {{
        if (typeof SwaggerUIBundle !== "undefined") {{
          window.ui = SwaggerUIBundle({{
            url: "{openapi_url}",
            dom_id: "#swagger-ui",
            deepLinking: true,
            presets: [
              SwaggerUIBundle.presets.apis
            ],
            layout: "BaseLayout"
          }});
        }} else {{
          renderFallback();
        }}
      }}, 250);
    }});
  </script>
</body>
</html>
"""


def _create_handler_class(
    routes: dict[str, Any],
    ws_routes: dict[str, Any],
    static_dir: Optional[str],
    vm_ref: Any,
    server_ref: Any = None
):
    class SynapseHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE, PATCH")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def do_GET(self):
            # Check for RFC 6455 WebSocket Upgrade request
            upgrade = self.headers.get("Upgrade", "")
            connection = self.headers.get("Connection", "")
            if upgrade.lower() == "websocket" and "upgrade" in connection.lower():
                self._handle_websocket_upgrade()
                return

            self._handle_request("GET")

        def do_POST(self):
            self._handle_request("POST")

        def do_PUT(self):
            self._handle_request("PUT")

        def do_DELETE(self):
            self._handle_request("DELETE")

        def do_PATCH(self):
            self._handle_request("PATCH")

        def _handle_websocket_upgrade(self):
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_raw = parse_qs(parsed_url.query)
            query = {k: v[0] if len(v) == 1 else v for k, v in query_raw.items()}

            # Match WebSocket route
            ws_handler = ws_routes.get(path)
            if ws_handler is None:
                # Also check standard routes for WebSocketRoute marker
                standard_handler = routes.get(path)
                if isinstance(standard_handler, SynapseWebSocketRoute) or getattr(standard_handler, "is_websocket", False):
                    ws_handler = standard_handler

            if ws_handler is None:
                self.send_response(404)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "WebSocket endpoint not found", "path": path}).encode("utf-8"))
                return

            ws_key = self.headers.get("Sec-WebSocket-Key", "")
            if not ws_key:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Missing Sec-WebSocket-Key header")
                return

            accept_token = _compute_ws_accept(ws_key)

            # Send 101 Switching Protocols response
            handshake_resp = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_token}\r\n"
                "\r\n"
            )
            self.wfile.write(handshake_resp.encode("ascii"))
            self.wfile.flush()

            # Prevent BaseHTTPRequestHandler from handling further HTTP requests on this connection
            self.close_connection = True

            ws = SynapseWebSocket(
                sock=self.connection,
                rfile=self.rfile,
                path=path,
                headers=dict(self.headers),
                query=query
            )

            try:
                # Execute WebSocket handler
                actual_fn = ws_handler.handler if isinstance(ws_handler, SynapseWebSocketRoute) else ws_handler
                sig = inspect.signature(actual_fn)
                if len(sig.parameters) >= 2:
                    req = SynapseRequest(path=path, method="GET", query=query, headers=dict(self.headers), body=None)
                    actual_fn(ws, req)
                else:
                    actual_fn(ws)
            except Exception:
                pass
            finally:
                ws.close()

        def _handle_request(self, method: str):
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_raw = parse_qs(parsed_url.query)
            query = {k: v[0] if len(v) == 1 else v for k, v in query_raw.items()}

            # Check built-in OpenAPI 3.1 & Swagger UI docs endpoints
            if server_ref and getattr(server_ref, "enable_docs", True) and method == "GET":
                if path == "/openapi.json" and path not in routes and f"GET {path}" not in routes:
                    spec = generate_openapi_spec(server_ref)
                    self._send_result(SynapseResponse(spec, status_code=200, content_type="application/json"))
                    return
                elif path == "/docs" and path not in routes and f"GET {path}" not in routes:
                    html = render_swagger_ui_html(
                        openapi_url="/openapi.json",
                        title=getattr(server_ref, "title", "Synapse API Docs")
                    )
                    self._send_result(SynapseResponse(html, status_code=200, content_type="text/html; charset=utf-8"))
                    return

            # Body oku
            body = None
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
                try:
                    body = json.loads(raw_body)
                except Exception:
                    body = raw_body

            # 1. Tanımlı Route var mı? (Method-specific e.g. 'GET /path' or generic '/path')
            route_handler = routes.get(f"{method.upper()} {path}") or routes.get(path)

            # Check parameterized routes if no direct match
            if route_handler is None:
                for r_pattern, r_h in routes.items():
                    r_method = None
                    p_parts = r_pattern.strip().split(" ", 1)
                    if len(p_parts) == 2 and p_parts[0].upper() in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
                        r_method = p_parts[0].upper()
                        pat = p_parts[1].strip()
                    else:
                        pat = r_pattern.strip()

                    if r_method and r_method != method.upper():
                        continue

                    # If route pattern contains {param} or :param
                    if "{" in pat or ":" in pat:
                        regex_pat = "^" + re.sub(r'\{[a-zA-Z_0-9]+\}', r'([^/]+)', re.sub(r':[a-zA-Z_0-9]+', r'([^/]+)', pat)) + "$"
                        if re.match(regex_pat, path):
                            route_handler = r_h
                            break

            if route_handler is not None:
                self._dispatch_route(route_handler, path, method, query, body)
                return

            # 2. Statik dosya sunumu (Örn. public/index.html)
            if static_dir and os.path.isdir(static_dir):
                clean_rel = path.lstrip("/") or "index.html"
                full_path = os.path.abspath(os.path.join(static_dir, clean_rel))

                # Güvenli dizin kontrolü
                if full_path.startswith(os.path.abspath(static_dir)) and os.path.isfile(full_path):
                    self._serve_static_file(full_path)
                    return

            # 3. Bulunamadı (404)
            self.send_response(404)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not Found", "path": path}).encode("utf-8"))

        def _dispatch_route(self, handler: Any, path: str, method: str, query: dict, body: Any):
            req = SynapseRequest(path=path, method=method, query=query, headers=dict(self.headers), body=body)

            try:
                # Handler callable mı? (SynapseFunction veya Python callable)
                if callable(handler):
                    if hasattr(handler, "code") and hasattr(handler, "env"):
                        result = handler(vm_ref, req)
                    else:
                        try:
                            result = handler(req)
                        except TypeError:
                            result = handler()
                else:
                    result = handler

                self._send_result(result)

            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal Server Error", "message": str(e)}).encode("utf-8"))

        def _send_result(self, result: Any):
            # SSE streaming check
            is_sse = (
                isinstance(result, SynapseSSEResponse)
                or inspect.isgenerator(result)
                or (hasattr(result, "__next__") and not isinstance(result, (dict, list, tuple, set, str, bytes)))
            )
            if is_sse:
                self._send_sse_stream(result)
                return

            if isinstance(result, SynapseResponse):
                status = result.status_code
                content_type = result.content_type
                body_data = result.body
                extra_headers = result.headers
            else:
                status = 200
                extra_headers = {}
                if isinstance(result, (dict, list)):
                    content_type = "application/json"
                    body_data = json.dumps(result, ensure_ascii=False)
                elif isinstance(result, str):
                    if result.strip().startswith("<") and result.strip().endswith(">"):
                        content_type = "text/html; charset=utf-8"
                    else:
                        content_type = "text/plain; charset=utf-8"
                    body_data = result
                else:
                    content_type = "text/plain; charset=utf-8"
                    body_data = str(result)

            if isinstance(body_data, (dict, list)):
                encoded = json.dumps(body_data, ensure_ascii=False).encode("utf-8")
            elif isinstance(body_data, str):
                encoded = body_data.encode("utf-8")
            elif isinstance(body_data, bytes):
                encoded = body_data
            else:
                encoded = str(body_data).encode("utf-8")

            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            for k, v in extra_headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(encoded)

        def _send_sse_stream(self, sse_obj: Any):
            """Streams SSE chunks immediately without buffering."""
            status = getattr(sse_obj, "status_code", 200)
            extra_headers = getattr(sse_obj, "headers", {})

            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            for k, v in extra_headers.items():
                self.send_header(k, v)
            self.end_headers()

            gen = sse_obj.generator if hasattr(sse_obj, "generator") else sse_obj
            try:
                for item in gen:
                    chunk = self._format_sse_item(item)
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                pass
            finally:
                self.close_connection = True

        def _format_sse_item(self, item: Any) -> str:
            """Formats a yielded item into standard SSE 'data: <content>\n\n' syntax."""
            if isinstance(item, dict):
                lines = []
                if "event" in item:
                    lines.append(f"event: {item['event']}")
                if "id" in item:
                    lines.append(f"id: {item['id']}")
                if "retry" in item:
                    lines.append(f"retry: {item['retry']}")

                if "data" in item:
                    d = item["data"]
                    if isinstance(d, (dict, list)):
                        lines.append(f"data: {json.dumps(d, ensure_ascii=False)}")
                    else:
                        lines.append(f"data: {d}")
                else:
                    lines.append(f"data: {json.dumps(item, ensure_ascii=False)}")
                return "\n".join(lines) + "\n\n"

            elif isinstance(item, str):
                if item.startswith(("data:", "event:", ":")):
                    return item if item.endswith("\n\n") else (item.rstrip("\n") + "\n\n")
                return f"data: {item}\n\n"
            else:
                return f"data: {item}\n\n"

        def _serve_static_file(self, full_path: str):
            ext = os.path.splitext(full_path)[1].lower()
            mime_types = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".ico": "image/x-icon",
            }
            content_type = mime_types.get(ext, "application/octet-stream")

            try:
                with open(full_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))

    return SynapseHTTPHandler


class SynapseServer:
    """
    High-performance multi-threaded web server for Synapse.
    Supports concurrent requests, SSE streaming, native RFC 6455 WebSockets,
    and automatic OpenAPI 3.1 schema generation with native Swagger UI.
    """
    def __init__(
        self,
        port: int = 8080,
        routes: Optional[dict[str, Any]] = None,
        ws_routes: Optional[dict[str, Any]] = None,
        host: str = "127.0.0.1",
        static_dir: Optional[str] = "public",
        vm_ref: Any = None,
        enable_docs: bool = True,
        title: str = "Synapse API",
        version: str = "1.0.0",
        description: str = "Synapse High-Performance AI & Web Runtime API",
    ):
        self.port = port
        self.host = host
        self.routes: dict[str, Any] = routes.copy() if routes else {}
        self.ws_routes: dict[str, Any] = ws_routes.copy() if ws_routes else {}
        self.static_dir = static_dir
        self.vm_ref = vm_ref
        self.enable_docs = enable_docs
        self.title = title
        self.version = version
        self.description = description
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

        # Detect any WebSocket routes configured in routes dict
        for p, h in list(self.routes.items()):
            if isinstance(h, SynapseWebSocketRoute) or getattr(h, "is_websocket", False):
                self.ws_routes[p] = h.handler if isinstance(h, SynapseWebSocketRoute) else h
            elif p.startswith("WS "):
                clean_p = p[3:].strip()
                self.ws_routes[clean_p] = h

    def add_route(self, path: str, handler: Any, is_ws: bool = False, method: Optional[str] = None) -> None:
        """Adds an HTTP or WebSocket route to the server."""
        if is_ws or isinstance(handler, SynapseWebSocketRoute) or getattr(handler, "is_websocket", False):
            self.ws_routes[path] = handler.handler if isinstance(handler, SynapseWebSocketRoute) else handler
        else:
            if method:
                self.routes[f"{method.upper()} {path}"] = handler
            else:
                self.routes[path] = handler

    def add_ws_route(self, path: str, ws_handler_fn: Callable[[SynapseWebSocket], Any]) -> None:
        """Registers a native RFC 6455 WebSocket endpoint."""
        self.ws_routes[path] = ws_handler_fn

    def start(self, blocking: bool = True) -> None:
        """Starts the multi-threaded HTTP/WebSocket server."""
        handler_cls = _create_handler_class(
            self.routes,
            self.ws_routes,
            self.static_dir,
            self.vm_ref,
            server_ref=self
        )
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self.httpd.daemon_threads = True
        docs_str = f" | Swagger UI: http://{self.host}:{self.port}/docs" if self.enable_docs else ""
        print(f"[Synapse Web Server] Running at http://{self.host}:{self.port}/ (Static: '{self.static_dir}'{docs_str})")

        if blocking:
            try:
                self.httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[Synapse Web Server] Stopped by user.")
                self.httpd.server_close()
        else:
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Shuts down the server and closes all sockets."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def serve(
    port: int = 8080,
    routes: Optional[dict[str, Any]] = None,
    ws_routes: Optional[dict[str, Any]] = None,
    host: str = "127.0.0.1",
    static_dir: str = "public",
    blocking: bool = True,
    vm_ref: Any = None,
    enable_docs: bool = True,
    title: str = "Synapse API",
    version: str = "1.0.0",
    description: str = "Synapse High-Performance AI & Web Runtime API",
) -> SynapseServer:
    """Starts the built-in Synapse multi-threaded HTTP/SSE/WebSocket server."""
    server = SynapseServer(
        port=port,
        routes=routes,
        ws_routes=ws_routes,
        host=host,
        static_dir=static_dir,
        vm_ref=vm_ref,
        enable_docs=enable_docs,
        title=title,
        version=version,
        description=description,
    )
    server.start(blocking=blocking)
    return server

