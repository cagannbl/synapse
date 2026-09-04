"""
Synapse Enterprise Web Framework
================================
Type-safe, modern web framework featuring:
- Expressive decorators (@app.get, @app.post, @app.put, @app.delete)
- Dynamic URL path parameter extraction with automatic type casting (/users/{id:int}, /items/{slug:str})
- Automatic JSON request parsing and structured Response objects
- FastAPI-style Depends(...) dependency injection
- Zero-dependency built-in CORS & JWT token authentication
- Programmatic handle_request() for lightning-fast testing
- Embedded ThreadingHTTPServer for live production serving
"""

import base64
import functools
import hashlib
import hmac
import http.server
import inspect
import json
import re
import socketserver
import threading
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


# =============================================================================
# 1. Exceptions
# =============================================================================

class HTTPException(Exception):
    """HTTP Exception with customizable status code and detail message."""
    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail or http.server.HTTPStatus(status_code).phrase

    def __repr__(self) -> str:
        return f"<HTTPException status_code={self.status_code} detail={self.detail!r}>"


class JWTError(Exception):
    """Base exception for JWT-related failures."""
    pass


class JWTExpiredError(JWTError):
    """Raised when a JWT token has expired."""
    pass


# =============================================================================
# 2. Zero-Dependency JWT Implementation (HS256)
# =============================================================================

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    pad_len = (-len(data)) % 4
    return base64.urlsafe_b64decode(data + ("=" * pad_len))


def create_access_token(
    payload: Dict[str, Any],
    secret_key: str,
    expires_in: int = 3600,
    algorithm: str = "HS256"
) -> str:
    """Creates a signed HMAC-SHA256 JWT token with expiration."""
    if algorithm != "HS256":
        raise ValueError("Currently only HS256 algorithm is supported.")

    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    token_payload = dict(payload)
    if "iat" not in token_payload:
        token_payload["iat"] = now
    if "exp" not in token_payload and expires_in is not None:
        token_payload["exp"] = now + expires_in

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_json = json.dumps(token_payload, separators=(",", ":")).encode("utf-8")

    header_b64 = _b64url_encode(header_json)
    payload_b64 = _b64url_encode(payload_json)
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
    verify_exp: bool = True
) -> Dict[str, Any]:
    """Decodes and validates a signed JWT token."""
    if algorithm != "HS256":
        raise ValueError("Currently only HS256 algorithm is supported.")

    parts = token.split(".")
    if len(parts) != 3:
        raise JWTError("Invalid token format: expected 3 dot-separated segments.")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    expected_sig_b64 = _b64url_encode(expected_sig)

    if not hmac.compare_digest(sig_b64, expected_sig_b64):
        raise JWTError("Invalid token signature.")

    try:
        payload_data = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as e:
        raise JWTError(f"Failed to decode token payload: {e}") from e

    if verify_exp and "exp" in payload_data:
        if int(time.time()) > payload_data["exp"]:
            raise JWTExpiredError("Token has expired.")

    return payload_data


def jwt_required(secret_key: str, header_name: str = "Authorization") -> Callable:
    """Dependency provider that extracts and verifies Bearer JWT token from request headers."""
    def dependency(request: "Request") -> Dict[str, Any]:
        auth_header = request.headers.get(header_name.lower(), "")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing authorization header")

        parts = auth_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        else:
            token = auth_header.strip()

        try:
            payload = decode_access_token(token, secret_key)
            request.state["user"] = payload
            return payload
        except JWTExpiredError as e:
            raise HTTPException(status_code=401, detail=f"Token expired: {e}") from e
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e

    return dependency


# =============================================================================
# 3. Dependency Injection Primitive: Depends
# =============================================================================

class Depends:
    """Dependency Injection marker for route handlers."""
    def __init__(self, dependency: Callable):
        self.dependency = dependency

    def __repr__(self) -> str:
        name = getattr(self.dependency, "__name__", str(self.dependency))
        return f"<Depends({name})>"


# =============================================================================
# 4. Request & Response
# =============================================================================

class Request:
    """Represents an incoming HTTP request."""
    def __init__(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Any = None,
        query: Optional[Dict[str, Any]] = None,
        path_params: Optional[Dict[str, Any]] = None
    ):
        self.method = method.upper()
        self.path = path
        # Normalize headers to lowercase keys for case-insensitive lookup
        self.headers: Dict[str, str] = {k.lower(): str(v) for k, v in (headers or {}).items()}
        self.raw_headers: Dict[str, str] = headers or {}
        self.raw_body = body
        self.query: Dict[str, Any] = query or {}
        self.path_params: Dict[str, Any] = path_params or {}
        self.state: Dict[str, Any] = {}
        self._parsed_json = None
        self._json_parsed = False

    def json(self) -> Any:
        """Parses and caches the request body as JSON."""
        if not self._json_parsed:
            if isinstance(self.raw_body, (dict, list)):
                self._parsed_json = self.raw_body
            elif isinstance(self.raw_body, (str, bytes)):
                try:
                    self._parsed_json = json.loads(self.raw_body)
                except Exception:
                    self._parsed_json = None
            else:
                self._parsed_json = None
            self._json_parsed = True
        return self._parsed_json

    @property
    def body(self) -> Any:
        """Returns JSON parsed body if available, otherwise raw_body."""
        parsed = self.json()
        return parsed if parsed is not None else self.raw_body

    def __repr__(self) -> str:
        return f"<Request {self.method} {self.path}>"


class Response:
    """Represents an HTTP response with automatic serialization."""
    def __init__(
        self,
        content: Any = None,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ):
        self.status_code = status_code
        self.headers: Dict[str, str] = dict(headers) if headers else {}
        self.content = content

        # Check existing content-type
        ct_key = None
        for k in self.headers:
            if k.lower() == "content-type":
                ct_key = k
                break

        if isinstance(content, (dict, list)):
            if not ct_key:
                self.headers["Content-Type"] = "application/json"
            self.body_text = json.dumps(content)
        elif isinstance(content, str):
            if not ct_key:
                self.headers["Content-Type"] = content_type or "text/plain; charset=utf-8"
            self.body_text = content
        elif isinstance(content, bytes):
            if not ct_key:
                self.headers["Content-Type"] = content_type or "application/octet-stream"
            self.body_text = content.decode("utf-8", errors="replace")
        elif content is None:
            self.body_text = ""
        else:
            # Primitive scalar or other serializable
            if not ct_key:
                self.headers["Content-Type"] = "application/json"
            self.body_text = json.dumps(content)

    def json(self) -> Any:
        """Returns the deserialized JSON content."""
        if isinstance(self.content, (dict, list)):
            return self.content
        try:
            return json.loads(self.body_text)
        except Exception:
            return None

    @property
    def text(self) -> str:
        """Returns the text body."""
        return self.body_text

    def __repr__(self) -> str:
        return f"<Response status_code={self.status_code}>"


# =============================================================================
# 5. Route Definition & URL Pattern Matching
# =============================================================================

class Route:
    """Represents an individual route with parameterized URL parsing."""
    PARAM_REGEX = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([a-zA-Z_][a-zA-Z0-9_]*))?\}")

    TYPE_CONVERTERS: Dict[str, Tuple[str, Callable[[str], Any]]] = {
        "int": (r"-?\d+", int),
        "float": (r"-?\d+(?:\.\d+)?", float),
        "str": (r"[^/]+", str),
        "uuid": (r"[0-9a-fA-F-]{36}", str),
        "path": (r".+", str),
        "slug": (r"[a-zA-Z0-9_-]+", str),
    }

    def __init__(self, path_template: str, method: str, handler: Callable):
        self.path_template = path_template
        self.method = method.upper()
        self.handler = handler
        self.param_names: List[str] = []
        self.converters: Dict[str, Callable[[str], Any]] = {}
        self.regex = self._compile_path(path_template)

    def _compile_path(self, template: str) -> re.Pattern:
        parts: List[str] = []
        last_end = 0

        for match in self.PARAM_REGEX.finditer(template):
            parts.append(re.escape(template[last_end:match.start()]))
            param_name = match.group(1)
            param_type = match.group(2) or "str"

            self.param_names.append(param_name)
            pattern_str, converter = self.TYPE_CONVERTERS.get(param_type, (r"[^/]+", str))
            self.converters[param_name] = converter

            parts.append(f"(?P<{param_name}>{pattern_str})")
            last_end = match.end()

        parts.append(re.escape(template[last_end:]))
        regex_str = f"^{''.join(parts)}/?$"
        return re.compile(regex_str)

    def match(self, path: str) -> Optional[Dict[str, Any]]:
        """Matches path and converts extracted parameter values."""
        clean_path = path if path.startswith("/") else f"/{path}"
        m = self.regex.match(clean_path)
        if not m:
            return None

        extracted = m.groupdict()
        params: Dict[str, Any] = {}
        for name, raw_val in extracted.items():
            converter = self.converters.get(name, str)
            try:
                params[name] = converter(raw_val)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid parameter format for '{name}': {e}"
                ) from e
        return params


# =============================================================================
# 6. SynapseApp Class
# =============================================================================

class SynapseApp:
    """Enterprise full-stack web application engine for Synapse."""

    def __init__(
        self,
        title: str = "Synapse Enterprise API",
        version: str = "1.0.0",
        cors_enabled: bool = True,
        cors_origins: Optional[List[str]] = None,
        cors_methods: Optional[List[str]] = None,
        cors_headers: Optional[List[str]] = None,
        cors_allow_credentials: bool = True
    ):
        self.title = title
        self.version = version
        self.routes: List[Route] = []
        self.middlewares: List[Callable] = []

        # CORS Configuration
        self.cors_enabled = cors_enabled
        self.cors_origins = cors_origins or ["*"]
        self.cors_methods = cors_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.cors_headers = cors_headers or ["*"]
        self.cors_allow_credentials = cors_allow_credentials

    # -------------------------------------------------------------------------
    # Route Registration Decorators
    # -------------------------------------------------------------------------

    def route(self, path: str, methods: Optional[List[str]] = None):
        """Generic route decorator supporting multiple HTTP methods."""
        def decorator(fn: Callable) -> Callable:
            target_methods = methods or ["GET"]
            for m in target_methods:
                self.routes.append(Route(path, m, fn))
            return fn
        return decorator

    def get(self, path: str):
        """Register a GET route."""
        return self.route(path, methods=["GET"])

    def post(self, path: str):
        """Register a POST route."""
        return self.route(path, methods=["POST"])

    def put(self, path: str):
        """Register a PUT route."""
        return self.route(path, methods=["PUT"])

    def delete(self, path: str):
        """Register a DELETE route."""
        return self.route(path, methods=["DELETE"])

    def patch(self, path: str):
        """Register a PATCH route."""
        return self.route(path, methods=["PATCH"])

    def add_middleware(self, middleware: Callable):
        """Registers a custom middleware (request, call_next) -> Response."""
        self.middlewares.append(middleware)

    # -------------------------------------------------------------------------
    # Dependency Injection Resolver
    # -------------------------------------------------------------------------

    def _resolve_dependency(self, dep_fn: Callable, request: Request) -> Any:
        """Recursively resolves dependencies and passes parameters to dependency function."""
        sig = inspect.signature(dep_fn)
        dep_kwargs: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if isinstance(param.default, Depends):
                dep_kwargs[param_name] = self._resolve_dependency(param.default.dependency, request)
            elif param_name in ("request", "req") or (param.annotation and param.annotation in (Request, "Request")):
                dep_kwargs[param_name] = request
            elif param_name in request.path_params:
                dep_kwargs[param_name] = request.path_params[param_name]
            elif param_name in request.query:
                dep_kwargs[param_name] = request.query[param_name]
            elif param_name in ("body", "data", "payload"):
                dep_kwargs[param_name] = request.json()
            elif param.default is not inspect.Parameter.empty:
                dep_kwargs[param_name] = param.default
            else:
                dep_kwargs[param_name] = None

        return dep_fn(**dep_kwargs)

    def _resolve_handler_args(
        self,
        handler: Callable,
        request: Request,
        path_params: Dict[str, Any]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Resolves parameters for the target route handler."""
        sig = inspect.signature(handler)
        kwargs: Dict[str, Any] = {}

        for name, param in sig.parameters.items():
            # 1. Check for Depends(...)
            if isinstance(param.default, Depends):
                kwargs[name] = self._resolve_dependency(param.default.dependency, request)
            # 2. Path parameter match
            elif name in path_params:
                kwargs[name] = path_params[name]
            # 3. Request object injection
            elif name in ("request", "req") or (param.annotation and param.annotation in (Request, "Request")):
                kwargs[name] = request
            # 4. Body / JSON payload injection
            elif name in ("body", "data", "payload") or (param.annotation and param.annotation in (dict, list)):
                kwargs[name] = request.json()
            # 5. Query parameters
            elif name in request.query:
                raw_query_val = request.query[name]
                if param.annotation and param.annotation is int:
                    try:
                        kwargs[name] = int(raw_query_val)
                    except Exception:
                        kwargs[name] = raw_query_val
                elif param.annotation and param.annotation is float:
                    try:
                        kwargs[name] = float(raw_query_val)
                    except Exception:
                        kwargs[name] = raw_query_val
                elif param.annotation and param.annotation is bool:
                    kwargs[name] = str(raw_query_val).lower() in ("true", "1", "yes")
                else:
                    kwargs[name] = raw_query_val
            # 6. Default parameter value
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
            else:
                kwargs[name] = None

        return [], kwargs

    # -------------------------------------------------------------------------
    # CORS Headers Applier
    # -------------------------------------------------------------------------

    def _apply_cors_headers(self, response: Response, origin: Optional[str] = None) -> Response:
        if not self.cors_enabled:
            return response

        matched_origin = "*"
        if origin and "*" not in self.cors_origins:
            if origin in self.cors_origins:
                matched_origin = origin
            else:
                matched_origin = self.cors_origins[0]

        if "Access-Control-Allow-Origin" not in response.headers:
            response.headers["Access-Control-Allow-Origin"] = matched_origin
        if "Access-Control-Allow-Methods" not in response.headers:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.cors_methods)
        if "Access-Control-Allow-Headers" not in response.headers:
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.cors_headers)
        if self.cors_allow_credentials and "Access-Control-Allow-Credentials" not in response.headers:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response

    # -------------------------------------------------------------------------
    # Programmatic Request Handler (handle_request)
    # -------------------------------------------------------------------------

    def handle_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Any = None
    ) -> Response:
        """
        Processes an incoming request programmatically and returns a Response object.
        Perfect for lightning-fast unit tests and zero-socket integration testing.
        """
        # Parse path & query string
        parsed_url = urllib.parse.urlparse(path)
        clean_path = parsed_url.path or "/"
        query_dict: Dict[str, Any] = {}
        if parsed_url.query:
            raw_qs = urllib.parse.parse_qs(parsed_url.query)
            query_dict = {k: v[0] if len(v) == 1 else v for k, v in raw_qs.items()}

        request = Request(
            method=method,
            path=clean_path,
            headers=headers,
            body=body,
            query=query_dict
        )

        origin = request.headers.get("origin")

        # CORS Preflight OPTIONS Check
        if method.upper() == "OPTIONS" and self.cors_enabled:
            preflight_resp = Response(content="", status_code=204)
            return self._apply_cors_headers(preflight_resp, origin)

        # Core Route Matching Logic
        def core_handler(req: Request) -> Response:
            matched_routes: List[Route] = []
            path_params: Dict[str, Any] = {}
            target_route: Optional[Route] = None

            for route in self.routes:
                try:
                    params = route.match(req.path)
                except HTTPException as e:
                    return Response({"error": e.detail, "status_code": e.status_code}, status_code=e.status_code)

                if params is not None:
                    matched_routes.append(route)
                    if route.method == req.method:
                        target_route = route
                        path_params = params
                        break

            if not matched_routes:
                return Response(
                    {"error": "Route not found", "path": req.path},
                    status_code=404
                )

            if target_route is None:
                allowed = [r.method for r in matched_routes]
                return Response(
                    {"error": "Method Not Allowed", "allowed": allowed},
                    status_code=405,
                    headers={"Allow": ", ".join(allowed)}
                )

            req.path_params = path_params

            try:
                args, kwargs = self._resolve_handler_args(target_route.handler, req, path_params)
                result = target_route.handler(*args, **kwargs)

                # Format return value to Response
                if isinstance(result, Response):
                    return result
                elif isinstance(result, tuple):
                    if len(result) == 2:
                        content, code = result
                        return Response(content, status_code=code)
                    elif len(result) == 3:
                        content, code, resp_headers = result
                        return Response(content, status_code=code, headers=resp_headers)
                return Response(result, status_code=200)

            except HTTPException as e:
                return Response(
                    {"error": e.detail, "detail": e.detail, "status_code": e.status_code},
                    status_code=e.status_code
                )
            except Exception as e:
                return Response(
                    {"error": str(e), "type": type(e).__name__, "status_code": 500},
                    status_code=500
                )

        # Middleware Chain Execution
        def build_middleware_chain(index: int) -> Callable[[Request], Response]:
            if index >= len(self.middlewares):
                return core_handler

            current_mw = self.middlewares[index]
            next_handler = build_middleware_chain(index + 1)

            def chained(req: Request) -> Response:
                return current_mw(req, next_handler)

            return chained

        pipeline = build_middleware_chain(0)
        try:
            final_response = pipeline(request)
        except HTTPException as e:
            final_response = Response(
                {"error": e.detail, "detail": e.detail, "status_code": e.status_code},
                status_code=e.status_code
            )
        except Exception as e:
            final_response = Response(
                {"error": str(e), "type": type(e).__name__, "status_code": 500},
                status_code=500
            )

        return self._apply_cors_headers(final_response, origin)

    # -------------------------------------------------------------------------
    # Live HTTP Server (serve)
    # -------------------------------------------------------------------------

    def serve(self, host: str = "127.0.0.1", port: int = 8000, blocking: bool = False):
        """Starts a live ThreadingHTTPServer serving this SynapseApp."""
        app = self

        class SynapseHTTPHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any):
                pass  # Suppress default noisy console logs

            def do_GET(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

            def do_PUT(self):
                self._dispatch("PUT")

            def do_DELETE(self):
                self._dispatch("DELETE")

            def do_PATCH(self):
                self._dispatch("PATCH")

            def do_OPTIONS(self):
                self._dispatch("OPTIONS")

            def _dispatch(self, method: str):
                content_len = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_len) if content_len > 0 else None

                headers_dict = {k: v for k, v in self.headers.items()}
                response = app.handle_request(method, self.path, headers=headers_dict, body=body_bytes)

                self.send_response(response.status_code)
                for header_k, header_v in response.headers.items():
                    self.send_header(header_k, header_v)
                self.end_headers()

                if response.body_text:
                    self.wfile.write(response.body_text.encode("utf-8"))

        class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        server = ThreadedServer((host, port), SynapseHTTPHandler)

        if blocking:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.shutdown()
        else:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def stop():
                server.shutdown()
                server.server_close()

            server.stop = stop
            return server
