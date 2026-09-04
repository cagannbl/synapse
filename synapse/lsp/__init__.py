"""
Synapse Language Server Protocol (LSP) Package.
"""
from synapse.lsp.server import SynapseLanguageServer, LSPServer, run_server
from synapse.lsp.features import get_hover, get_completions, HOVER_DOCS

__all__ = [
    "SynapseLanguageServer",
    "LSPServer",
    "run_server",
    "get_hover",
    "get_completions",
    "HOVER_DOCS",
]
