#!/usr/bin/env python3
"""
Synapse Playground local server.

Serves the `playground/` directory and exposes `POST /api/run`, which executes
the submitted Synapse program with the real interpreter (`synapse run --vm`) in
a separate process with a time limit, and returns its stdout/stderr.

The server only binds to localhost, and /api/run only accepts requests whose
Host and Origin are this server, so other websites open in the same browser
cannot use it to run code on your machine.
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import os
import socketserver
import subprocess
import sys
import tempfile
import time
from typing import Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_TIMEOUT_SECONDS = 10
MAX_SOURCE_BYTES = 100_000


# Ensure MIME types are registered at the Python system level
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/plain", ".syn")


def run_synapse_source(source: str, timeout: float = RUN_TIMEOUT_SECONDS) -> dict:
    """Runs a Synapse program with the bytecode VM in a subprocess."""
    with tempfile.TemporaryDirectory(prefix="synapse-playground-") as tmp:
        path = os.path.join(tmp, "main.syn")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        env["NO_COLOR"] = "1"
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "synapse.cli", "run", "--vm", path],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = f"Stopped after {timeout:g}s time limit."
            exit_code = -1
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
        duration_ms = (time.perf_counter() - start) * 1000
    return {"stdout": stdout, "stderr": stderr.strip(), "exit_code": exit_code, "duration_ms": round(duration_ms, 2)}


class PlaygroundRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Static file handler for the playground, plus POST /api/run.
    Sends COOP/COEP isolation headers and disables caching.
    """

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".html": "text/html; charset=utf-8",
        ".htm": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".syn": "text/plain; charset=utf-8",
    }

    def end_headers(self) -> None:
        """Injects cross-origin isolation and no-cache headers. No CORS: same-origin only."""
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _is_same_origin_request(self) -> bool:
        """Rejects cross-site requests and DNS-rebinding attempts."""
        port = self.server.server_address[1]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        # Browsers always send Origin on cross-origin POSTs; same-origin fetches may omit it.
        return origin is None or origin in {f"http://{h}" for h in allowed_hosts}

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json(404, {"error": "not found"})
            return
        if not self._is_same_origin_request():
            self._send_json(403, {"error": "cross-origin requests are not allowed"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._send_json(415, {"error": "expected application/json"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_SOURCE_BYTES:
            self._send_json(413, {"error": f"program larger than {MAX_SOURCE_BYTES} bytes"})
            return
        try:
            source = json.loads(self.rfile.read(length) or b"{}").get("code", "")
        except (ValueError, AttributeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return
        if not isinstance(source, str):
            self._send_json(400, {"error": "'code' must be a string"})
            return
        self._send_json(200, run_synapse_source(source))

    def guess_type(self, path: str) -> str:
        """Explicitly guarantees MIME type resolution for WebAssembly files."""
        base, ext = os.path.splitext(path)
        ext_lower = ext.lower()
        if ext_lower == ".wasm":
            return "application/wasm"
        if ext_lower in (".js", ".mjs"):
            return "text/javascript"
        if ext_lower in (".html", ".htm"):
            return "text/html; charset=utf-8"
        if ext_lower == ".css":
            return "text/css; charset=utf-8"
        if ext_lower == ".json":
            return "application/json"
        if ext_lower == ".syn":
            return "text/plain; charset=utf-8"
        return super().guess_type(path)


class ThreadingPlaygroundServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP Server for non-blocking browser asset requests."""
    daemon_threads = True
    allow_reuse_address = True


def get_default_directory() -> str:
    """Returns the absolute path to the playground assets directory."""
    return os.path.abspath(os.path.dirname(__file__))


def create_server(
    host: str = "127.0.0.1",
    port: int = 3000,
    directory: Optional[str] = None
) -> Tuple[ThreadingPlaygroundServer, str]:
    """
    Creates and binds a ThreadingPlaygroundServer instance.
    Returns (server_instance, serving_directory).
    """
    # Security: Strictly enforce localhost binding to prevent remote exposure
    if host not in ("127.0.0.1", "localhost"):
        host = "127.0.0.1"

    serve_dir = os.path.abspath(directory or get_default_directory())

    handler = lambda *args, **kwargs: PlaygroundRequestHandler(
        *args, directory=serve_dir, **kwargs
    )

    server = ThreadingPlaygroundServer((host, port), handler)
    return server, serve_dir


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command line options."""
    parser = argparse.ArgumentParser(
        description="Synapse WebAssembly Playground Local Server"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3000,
        help="Port to serve playground on (default: 3000)"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="Directory to serve (default: playground/ directory)"
    )
    return parser.parse_args(args)


def run_server(
    host: str = "127.0.0.1",
    port: int = 3000,
    directory: Optional[str] = None
) -> None:
    """Starts the development server and serves until interrupted."""
    server, serve_dir = create_server(host=host, port=port, directory=directory)
    url = f"http://{host}:{port}"
    print(f"============================================================")
    print(f" Synapse WebAssembly Interactive Playground")
    print(f" Serving Directory : {serve_dir}")
    print(f" Local URL         : {url}")
    print(f" COOP / COEP       : Enabled (same-origin / require-corp)")
    print(f" Press Ctrl+C to stop.")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Synapse] Server stopped by user.")
    finally:
        server.server_close()


def main() -> None:
    parsed = parse_args()
    run_server(host=parsed.host, port=parsed.port, directory=parsed.dir)


if __name__ == "__main__":
    main()
