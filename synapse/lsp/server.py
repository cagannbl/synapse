"""
Synapse Language Server Protocol (LSP) Server.
Standards-compliant JSON-RPC 2.0 stdio language server implementation for Synapse.
Supports:
- Capabilities negotiation (hover, completion)
- Real-time syntax & semantic diagnostics with self-healing AI hints
- Rich Markdown hover documentation
- Context-aware code completions (keywords, builtins, tensor methods)
"""
from __future__ import annotations
import io
import json
import re
import sys
from typing import Any, BinaryIO, Optional

from synapse.core.diagnostics import diagnose_code
from synapse.lsp.features import (
    get_hover, get_completions, find_definition,
    get_signature_help, get_document_symbols, format_document
)


class SynapseLanguageServer:
    """
    Language Server Protocol 3.x implementation for the Synapse programming language.
    """

    def __init__(self, reader: Optional[BinaryIO] = None, writer: Optional[BinaryIO] = None):
        self.reader = reader or sys.stdin.buffer
        self.writer = writer or sys.stdout.buffer
        self.documents: dict[str, str] = {}
        self.is_running = True
        self.is_initialized = False

    # =========================================================================
    # JSON-RPC Wire Protocol
    # =========================================================================

    def read_message(self) -> Optional[dict[str, Any]]:
        """Reads a single JSON-RPC message from reader stream."""
        content_length: Optional[int] = None

        while True:
            line = self.reader.readline()
            if not line:
                return None
            line_str = line.decode("latin1").strip()
            if not line_str:
                # Empty line signifies end of HTTP headers
                break
            if line_str.lower().startswith("content-length:"):
                try:
                    content_length = int(line_str.split(":", 1)[1].strip())
                except ValueError:
                    content_length = None

        if content_length is None:
            return None

        body = self.reader.read(content_length)
        if not body:
            return None

        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None

    def send_message(self, payload: dict[str, Any]) -> None:
        """Sends a JSON-RPC message to the writer stream."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("latin1")
        self.writer.write(header + body)
        self.writer.flush()

    def send_response(self, msg_id: Any, result: Any = None, error: Optional[dict[str, Any]] = None) -> None:
        """Sends JSON-RPC 2.0 response."""
        resp: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": msg_id,
        }
        if error is not None:
            resp["error"] = error
        else:
            resp["result"] = result
        self.send_message(resp)

    def send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Sends JSON-RPC 2.0 notification (no id)."""
        self.send_message({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        })

    # =========================================================================
    # Diagnostics Publishing
    # =========================================================================

    def publish_diagnostics(self, uri: str, code: str) -> dict[str, Any]:
        """
        Runs synapse.core.diagnostics.diagnose_code and sends textDocument/publishDiagnostics.
        Returns the notification payload for inspection/testing.
        """
        report = diagnose_code(code, filepath=uri)
        diagnostics: list[dict[str, Any]] = []

        if report.status == "error":
            # LSP line & column are 0-indexed; report is 1-indexed
            lsp_line = max(0, report.line - 1)
            lsp_col = max(0, report.column - 1)

            # Build enriched message with AI self-healing hint
            message_parts = [report.message]
            if report.suggested_fix:
                message_parts.append(f"💡 Suggested Fix: {report.suggested_fix}")
            if report.ai_prompt_hint:
                message_parts.append(f"🤖 AI Hint: {report.ai_prompt_hint}")
            full_message = "\n\n".join(message_parts)

            end_col = lsp_col + 1
            if report.source_line and lsp_col < len(report.source_line):
                # Highlight the offending identifier/character
                sub = report.source_line[lsp_col:]
                m = re.match(r"^[A-Za-z0-9_]+", sub)
                if m:
                    end_col = lsp_col + len(m.group(0))

            diag_item = {
                "range": {
                    "start": {"line": lsp_line, "character": lsp_col},
                    "end": {"line": lsp_line, "character": end_col}
                },
                "severity": 1,  # 1 = Error
                "source": "synapse",
                "message": full_message,
                "data": {
                    "error_type": report.error_type,
                    "suggested_fix": report.suggested_fix,
                    "ai_prompt_hint": report.ai_prompt_hint
                }
            }
            diagnostics.append(diag_item)

        notification = {
            "uri": uri,
            "diagnostics": diagnostics
        }
        self.send_notification("textDocument/publishDiagnostics", notification)
        return notification

    # =========================================================================
    # LSP Handlers
    # =========================================================================

    def handle_initialize(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Responds with server capabilities."""
        capabilities = {
            "textDocumentSync": 1,  # 1 = Full sync
            "hoverProvider": True,
            "completionProvider": {
                "resolveProvider": False,
                "triggerCharacters": [".", "@", ":"]
            },
            "definitionProvider": True,
            "documentFormattingProvider": True,
            "signatureHelpProvider": {
                "triggerCharacters": ["(", ","]
            },
            "documentSymbolProvider": True,
        }
        server_info = {
            "name": "synapse-lsp",
            "version": "0.2.0"
        }
        self.is_initialized = True
        self.send_response(msg_id, result={"capabilities": capabilities, "serverInfo": server_info})

    def handle_hover(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Extracts identifier at position and returns Markdown hover tooltip."""
        uri = params.get("textDocument", {}).get("uri", "")
        pos = params.get("position", {})
        line_no = pos.get("line", 0)
        col_no = pos.get("character", 0)

        text = self.documents.get(uri, "")
        word = self._extract_word_at_pos(text, line_no, col_no)

        hover_data = get_hover(word) if word else None
        self.send_response(msg_id, result=hover_data)

    def handle_completion(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Returns auto-completion items."""
        uri = params.get("textDocument", {}).get("uri", "")
        pos = params.get("position", {})
        line_no = pos.get("line", 0)
        col_no = pos.get("character", 0)
        context = params.get("context", {})
        trigger_char = context.get("triggerCharacter")

        text = self.documents.get(uri, "")
        line_prefix = ""
        lines = text.splitlines()
        if 0 <= line_no < len(lines):
            line_prefix = lines[line_no][:col_no]

        items = get_completions(trigger_char=trigger_char, line_prefix=line_prefix)
        self.send_response(msg_id, result={"isIncomplete": False, "items": items})

    def handle_definition(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Handles textDocument/definition request."""
        uri = params.get("textDocument", {}).get("uri", "")
        pos = params.get("position", {})
        line_no = pos.get("line", 0)
        col_no = pos.get("character", 0)
        text = self.documents.get(uri, "")
        word = self._extract_word_at_pos(text, line_no, col_no)
        loc = find_definition(text, word) if word else None
        if loc:
            result = {
                "uri": uri,
                "range": {
                    "start": {"line": loc["line"], "character": loc["character"]},
                    "end": {"line": loc["line"], "character": loc["character"] + len(word)}
                }
            }
        else:
            result = None
        self.send_response(msg_id, result=result)

    def handle_formatting(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Handles textDocument/formatting request."""
        uri = params.get("textDocument", {}).get("uri", "")
        text = self.documents.get(uri, "")
        formatted = format_document(text)
        lines = text.splitlines()
        last_line = max(0, len(lines) - 1)
        last_col = len(lines[last_line]) if lines else 0
        edits = [{
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": last_line + 1, "character": last_col}
            },
            "newText": formatted
        }]
        self.send_response(msg_id, result=edits)

    def handle_signature_help(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Handles textDocument/signatureHelp request."""
        uri = params.get("textDocument", {}).get("uri", "")
        pos = params.get("position", {})
        line_no = pos.get("line", 0)
        col_no = pos.get("character", 0)
        text = self.documents.get(uri, "")
        sig_data = get_signature_help(text, line_no, col_no)
        self.send_response(msg_id, result=sig_data)

    def handle_document_symbol(self, msg_id: Any, params: dict[str, Any]) -> None:
        """Handles textDocument/documentSymbol request."""
        uri = params.get("textDocument", {}).get("uri", "")
        text = self.documents.get(uri, "")
        symbols = get_document_symbols(text)
        self.send_response(msg_id, result=symbols)

    def handle_did_open(self, params: dict[str, Any]) -> None:
        """Handles textDocument/didOpen and publishes initial diagnostics."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri", "")
        text = doc.get("text", "")
        self.documents[uri] = text
        self.publish_diagnostics(uri, text)

    def handle_did_change(self, params: dict[str, Any]) -> None:
        """Handles textDocument/didChange (Full sync) and updates diagnostics."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri", "")
        changes = params.get("contentChanges", [])
        if changes:
            # Under Full sync (1), the last change contains the full source text
            text = changes[-1].get("text", "")
            self.documents[uri] = text
            self.publish_diagnostics(uri, text)

    def handle_did_close(self, params: dict[str, Any]) -> None:
        """Handles textDocument/didClose and clears diagnostics."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri", "")
        self.documents.pop(uri, None)
        self.send_notification("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": []})

    def handle_shutdown(self, msg_id: Any) -> None:
        """Handles shutdown request."""
        self.send_response(msg_id, result=None)

    def handle_exit(self) -> None:
        """Terminates server process."""
        self.is_running = False

    # =========================================================================
    # Request Dispatcher
    # =========================================================================

    def handle_message(self, msg: dict[str, Any]) -> None:
        """Dispatches an incoming JSON-RPC message."""
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            self.handle_initialize(msg_id, params)
        elif method == "initialized":
            pass  # Client confirmation notification
        elif method == "shutdown":
            self.handle_shutdown(msg_id)
        elif method == "exit":
            self.handle_exit()
        elif method == "textDocument/didOpen":
            self.handle_did_open(params)
        elif method == "textDocument/didChange":
            self.handle_did_change(params)
        elif method == "textDocument/didClose":
            self.handle_did_close(params)
        elif method == "textDocument/hover":
            self.handle_hover(msg_id, params)
        elif method == "textDocument/completion":
            self.handle_completion(msg_id, params)
        elif method == "textDocument/definition":
            self.handle_definition(msg_id, params)
        elif method == "textDocument/formatting":
            self.handle_formatting(msg_id, params)
        elif method == "textDocument/signatureHelp":
            self.handle_signature_help(msg_id, params)
        elif method == "textDocument/documentSymbol":
            self.handle_document_symbol(msg_id, params)
        else:
            if msg_id is not None:
                # Method not found
                self.send_response(
                    msg_id,
                    error={"code": -32601, "message": f"Method not found: {method}"}
                )

    def start(self) -> None:
        """Starts the main stdio loop."""
        while self.is_running:
            try:
                msg = self.read_message()
                if msg is None:
                    break
                self.handle_message(msg)
            except Exception as e:
                sys.stderr.write(f"[Synapse LSP Error] {e}\n")
                sys.stderr.flush()

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _extract_word_at_pos(self, text: str, line_no: int, col_no: int) -> str:
        """Extracts the identifier/word at the specified 0-indexed line and character."""
        lines = text.splitlines()
        if not (0 <= line_no < len(lines)):
            return ""
        line = lines[line_no]
        if not line or col_no < 0:
            return ""

        # Clamp col_no to line length
        idx = min(col_no, len(line) - 1) if len(line) > 0 else 0

        # Scan backwards to word boundary
        start = idx
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            start -= 1

        # Scan forwards to word boundary
        end = idx
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1

        return line[start:end]


# Alias for standard naming
LSPServer = SynapseLanguageServer

__all__ = ["SynapseLanguageServer", "LSPServer", "run_server"]


def run_server() -> None:
    """Entry point for running Synapse LSP server via stdio."""
    server = SynapseLanguageServer()
    server.start()


if __name__ == "__main__":
    run_server()
