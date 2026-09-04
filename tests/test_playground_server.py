"""
Unit and integration tests for the Synapse WebAssembly Interactive Playground
and development server (`playground/server.py`).
"""

import io
import os
import tempfile
import threading
import urllib.error
import urllib.request
import pytest

from playground.server import (
    PlaygroundRequestHandler,
    ThreadingPlaygroundServer,
    create_server,
    get_default_directory,
    parse_args,
)


@pytest.fixture(scope="module")
def live_server():
    """Starts a live ThreadingPlaygroundServer on an ephemeral port for testing."""
    # Use port 0 for OS-assigned ephemeral port to avoid port conflicts
    server, serve_dir = create_server(host="127.0.0.1", port=0)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url, serve_dir

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_server_initialization_and_port_binding():
    """1. Test that server initializes and binds properly on ephemeral port."""
    server, serve_dir = create_server(host="127.0.0.1", port=0)
    try:
        assert isinstance(server, ThreadingPlaygroundServer)
        assert os.path.isdir(serve_dir)
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        server.server_close()


def test_server_serves_index_html(live_server):
    """2. Test serving index.html with status 200 and text/html MIME type."""
    base_url, _ = live_server

    for path in ["/", "/index.html"]:
        req = urllib.request.Request(f"{base_url}{path}")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert "text/html" in content_type
            body = resp.read().decode("utf-8")
            assert "<!DOCTYPE html>" in body or "<html" in body
            assert "Synapse" in body


def test_server_serves_worker_js(live_server):
    """3. Test serving worker.js with status 200 and text/javascript MIME type."""
    base_url, _ = live_server
    req = urllib.request.Request(f"{base_url}/worker.js")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "javascript" in content_type
        body = resp.read().decode("utf-8")
        assert "WasmCooperativeQueue" in body
        assert "onmessage" in body


def test_server_coop_coep_headers(live_server):
    """4. Test that Cross-Origin Isolation headers (COOP/COEP) are present."""
    base_url, _ = live_server
    req = urllib.request.Request(f"{base_url}/index.html")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
        assert resp.headers.get("Cross-Origin-Embedder-Policy") == "require-corp"


def test_server_cors_headers_and_options_preflight(live_server):
    """5. Test CORS allow-origin headers and OPTIONS preflight handler."""
    base_url, _ = live_server

    # GET request check
    req = urllib.request.Request(f"{base_url}/index.html")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "Cache-Control" in resp.headers

    # OPTIONS preflight check
    req_options = urllib.request.Request(f"{base_url}/index.html", method="OPTIONS")
    with urllib.request.urlopen(req_options, timeout=5.0) as resp:
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")


def test_server_wasm_mime_type_serving():
    """6. Test that .wasm files are served with application/wasm MIME type."""
    # Create a temporary directory with a dummy .wasm file
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_wasm = os.path.join(temp_dir, "test_module.wasm")
        # WASM binary magic header \0asm
        with open(dummy_wasm, "wb") as f:
            f.write(b"\x00asm\x01\x00\x00\x00")

        server, _ = create_server(host="127.0.0.1", port=0, directory=temp_dir)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            url = f"http://127.0.0.1:{port}/test_module.wasm"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                assert resp.status == 200
                content_type = resp.headers.get("Content-Type", "")
                assert content_type == "application/wasm"
                data = resp.read()
                assert data.startswith(b"\x00asm")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


def test_guess_type_extensions_map():
    """7. Test PlaygroundRequestHandler.guess_type for all key file extensions."""
    handler = PlaygroundRequestHandler

    # Test via handler class method
    dummy_handler = handler.__new__(handler)
    assert dummy_handler.guess_type("kernel.wasm") == "application/wasm"
    assert dummy_handler.guess_type("module.js") == "text/javascript"
    assert dummy_handler.guess_type("index.html") == "text/html; charset=utf-8"
    assert dummy_handler.guess_type("styles.css") == "text/css; charset=utf-8"
    assert dummy_handler.guess_type("manifest.json") == "application/json"
    assert dummy_handler.guess_type("script.syn") == "text/plain; charset=utf-8"


def test_server_404_not_found(live_server):
    """8. Test that requesting a non-existent file returns 404 Not Found."""
    base_url, _ = live_server
    req = urllib.request.Request(f"{base_url}/nonexistent_file_xyz_123.wasm")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5.0)
    assert exc_info.value.code == 404


def test_index_html_ui_components():
    """9. Test that index.html contains all required UI components and examples."""
    assets_dir = get_default_directory()
    html_path = os.path.join(assets_dir, "index.html")
    assert os.path.isfile(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Verify linear / vercel minimalist styling elements
    assert "zinc-950" in html or "#09090b" in html
    assert "border" in html
    assert "Inter" in html or "JetBrains Mono" in html

    # Verify editor and console
    assert "codeEditor" in html
    assert "lineNumbers" in html
    assert "terminal" in html
    assert "STDOUT" in html
    assert "STDERR" in html

    # Verify action buttons
    assert "runBtn" in html or "runCode" in html
    assert "resetBtn" in html or "resetEditor" in html
    assert "clearBtn" in html or "clearTerminal" in html

    # Verify all 4 required examples
    assert "Tensor Matris" in html
    assert "AI Prompt" in html
    assert "LazyFrame" in html
    assert "CSP" in html or "Kanalları" in html


def test_worker_js_protocol_and_cooperative_queue():
    """10. Test that worker.js contains the Web Worker protocol and cooperative queue."""
    assets_dir = get_default_directory()
    worker_path = os.path.join(assets_dir, "worker.js")
    assert os.path.isfile(worker_path)

    with open(worker_path, "r", encoding="utf-8") as f:
        worker_code = f.read()

    # Verify Web Worker messaging API
    assert "onmessage" in worker_code
    assert "postMessage" in worker_code

    # Verify cooperative queue & single-threaded WASM CSP channel
    assert "WasmCooperativeQueue" in worker_code
    assert "WasmChannel" in worker_code
    assert "enqueue" in worker_code
    assert "runNext" in worker_code
    assert "runAll" in worker_code

    # Verify example handlers
    assert "executeExample1_Tensor" in worker_code
    assert "executeExample2_AIPrompt" in worker_code
    assert "executeExample3_LazyFrame" in worker_code
    assert "executeExample4_CSP" in worker_code

    # Verify stdout / stderr streaming
    assert "stdout" in worker_code
    assert "stderr" in worker_code
    assert "done" in worker_code


def test_cli_argument_parsing():
    """11. Test server CLI argument parser."""
    # Default args
    args_default = parse_args([])
    assert args_default.port == 3000
    assert args_default.host == "127.0.0.1"
    assert args_default.dir is None

    # Custom args
    args_custom = parse_args(["--port", "8080", "--host", "0.0.0.0", "--dir", "/tmp/playground"])
    assert args_custom.port == 8080
    assert args_custom.host == "0.0.0.0"
    assert args_custom.dir == "/tmp/playground"


def test_c_runtime_wasm_cooperative_queue_declarations():
    """12. Test that synapse_runtime.h and .c include Emscripten cooperative queue."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runtime_h = os.path.join(root_dir, "synapse", "runtime", "synapse_runtime.h")
    runtime_c = os.path.join(root_dir, "synapse", "runtime", "synapse_runtime.c")

    with open(runtime_h, "r", encoding="utf-8") as f:
        h_code = f.read()

    assert "syn_channel_t" in h_code
    assert "syn_channel_create" in h_code
    assert "syn_channel_send" in h_code
    assert "syn_channel_recv" in h_code
    assert "syn_spawn" in h_code
    assert "#if defined(__EMSCRIPTEN__)" in h_code
    assert "syn_wasm_enqueue_task" in h_code
    assert "syn_wasm_run_microtasks" in h_code

    with open(runtime_c, "r", encoding="utf-8") as f:
        c_code = f.read()

    assert "#if defined(__EMSCRIPTEN__)" in c_code
    assert "g_wasm_task_head" in c_code
    assert "syn_wasm_enqueue_task" in c_code
    assert "syn_wasm_run_microtasks" in c_code
    assert "syn_channel_send" in c_code
    assert "syn_channel_recv" in c_code
