#!/usr/bin/env python3
"""
Synapse WebAssembly Interactive Playground Local Development Server.

Zero-dependency HTTP server based strictly on Python's built-in `http.server`.
Serves the `playground/` directory with proper WebAssembly MIME types and
Cross-Origin Isolation headers (COOP/COEP) required for high-performance
SharedArrayBuffer and WASM multi-threading / SIMD in modern browsers.
"""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import socketserver
import sys
from typing import Optional, Tuple


# Ensure MIME types are registered at the Python system level
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/plain", ".syn")


class PlaygroundRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP Request handler configured with:
    - Correct WebAssembly (application/wasm) MIME types
    - Cross-Origin-Opener-Policy: same-origin
    - Cross-Origin-Embedder-Policy: require-corp
    - Access-Control-Allow-Origin: *
    - Cache-Control: no-cache
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
        """Injects cross-origin isolation and developer-friendly headers."""
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handles CORS preflight checks gracefully."""
        self.send_response(200)
        self.end_headers()

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
