import io
import json
import pytest
from synapse.lsp.server import SynapseLanguageServer, LSPServer
from synapse.lsp.features import get_hover, get_completions


def test_lsp_server_alias():
    assert LSPServer is SynapseLanguageServer
    server = LSPServer()
    assert isinstance(server, SynapseLanguageServer)


def test_lsp_initialize_handshake():
    reader = io.BytesIO()
    writer = io.BytesIO()
    server = SynapseLanguageServer(reader=reader, writer=writer)

    # Dispatch initialize
    server.handle_initialize(msg_id=1, params={"capabilities": {}})
    assert server.is_initialized is True

    # Read output response
    writer.seek(0)
    raw = writer.read().decode("utf-8")
    assert "Content-Length:" in raw

    body_json = raw.split("\r\n\r\n", 1)[1]
    resp = json.loads(body_json)
    assert resp["id"] == 1
    assert "capabilities" in resp["result"]
    assert resp["result"]["capabilities"]["hoverProvider"] is True
    assert "completionProvider" in resp["result"]["capabilities"]


def test_lsp_hover_request():
    reader = io.BytesIO()
    writer = io.BytesIO()
    server = SynapseLanguageServer(reader=reader, writer=writer)

    uri = "file:///test.syn"
    code = "let t = tensor([[1, 2], [3, 4]])\n"
    server.documents[uri] = code

    # Hover over 'tensor' at line 0, column 10
    server.handle_hover(msg_id=2, params={"textDocument": {"uri": uri}, "position": {"line": 0, "character": 10}})

    writer.seek(0)
    raw = writer.read().decode("utf-8")
    body_json = raw.split("\r\n\r\n", 1)[1]
    resp = json.loads(body_json)

    assert resp["id"] == 2
    assert resp["result"] is not None
    assert "contents" in resp["result"]
    assert "tensor" in resp["result"]["contents"]["value"]


def test_lsp_completion_request():
    reader = io.BytesIO()
    writer = io.BytesIO()
    server = SynapseLanguageServer(reader=reader, writer=writer)

    uri = "file:///test.syn"
    server.documents[uri] = "let x = 1\n"

    # Completion with trigger character '.'
    server.handle_completion(
        msg_id=3,
        params={
            "textDocument": {"uri": uri},
            "position": {"line": 0, "character": 9},
            "context": {"triggerCharacter": "."}
        }
    )

    writer.seek(0)
    raw = writer.read().decode("utf-8")
    body_json = raw.split("\r\n\r\n", 1)[1]
    resp = json.loads(body_json)

    assert resp["id"] == 3
    items = resp["result"]["items"]
    assert len(items) > 0
    labels = [item["label"] for item in items]
    assert "matmul" in labels or "relu" in labels or "grad" in labels


def test_lsp_diagnostics_and_didchange():
    reader = io.BytesIO()
    writer = io.BytesIO()
    server = SynapseLanguageServer(reader=reader, writer=writer)

    uri = "file:///err.syn"
    bad_code = "fn bad_func(\n"  # Unterminated parenthesis / syntax error

    # Trigger didOpen
    server.handle_did_open({"textDocument": {"uri": uri, "text": bad_code}})
    assert uri in server.documents

    writer.seek(0)
    raw = writer.read().decode("utf-8")
    body_json = raw.split("\r\n\r\n", 1)[1]
    notification = json.loads(body_json)

    assert notification["method"] == "textDocument/publishDiagnostics"
    assert notification["params"]["uri"] == uri
    diags = notification["params"]["diagnostics"]
    assert len(diags) > 0
    assert diags[0]["severity"] == 1  # Error
    assert "💡" in diags[0]["message"] or "error" in diags[0]["message"].lower()

    # Now fix it with didChange
    fixed_code = "fn good_func():\n    return 42\n"
    writer.seek(0)
    writer.truncate(0)

    server.handle_did_change({"textDocument": {"uri": uri}, "contentChanges": [{"text": fixed_code}]})

    writer.seek(0)
    raw_fix = writer.read().decode("utf-8")
    body_fix = raw_fix.split("\r\n\r\n", 1)[1]
    fix_notif = json.loads(body_fix)

    assert fix_notif["method"] == "textDocument/publishDiagnostics"
    # No error diagnostics on valid code
    assert len(fix_notif["params"]["diagnostics"]) == 0
