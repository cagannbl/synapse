"""
Tests for Synapse v3+ Engineering Enhancements:
- std.db (SQLite embedded database)
- std.compress (gzip, zlib, b64)
- std.path (cross-platform path operations)
- VM CSP Channels & try_catch exception handling
- LSP Advanced Features (Go-to-Definition, Formatting, Signature Help, Document Symbols)
- Package Manager SemVer & Outdated checks
- CLI Documentation Generator (`synapse doc`)
"""
import io
import os
import tempfile
import pytest

from synapse.stdlib import db, compress, path
from synapse.core.task_pool import channel, spawn, ChannelClosed
from synapse.vm.virtual_machine import VirtualMachine
from synapse.lsp.features import (
    find_definition, get_document_symbols, get_signature_help, format_document
)
from synapse.lsp.server import SynapseLanguageServer
from synapse.pkg.manager import SemverHelper, PackageManager
from synapse.cli import generate_docs


# =========================================================================
# 1. std.db (SQLite) Tests
# =========================================================================

def test_stdlib_db_in_memory():
    conn = db.connect(":memory:")
    assert conn.is_open

    # DDL
    conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT, params INT)")

    # DML Insert
    rowid = conn.execute("INSERT INTO models (name, params) VALUES (?, ?)", ["GPT-4o", 1800])
    assert rowid == 1

    # Query
    rows = conn.query("SELECT * FROM models WHERE name = ?", ["GPT-4o"])
    assert len(rows) == 1
    assert rows[0]["name"] == "GPT-4o"
    assert rows[0]["params"] == 1800

    # Query One
    one = conn.query_one("SELECT * FROM models WHERE id = 1")
    assert one is not None
    assert one["name"] == "GPT-4o"

    conn.close()
    assert not conn.is_open


def test_stdlib_db_context_manager(tmp_path):
    db_file = str(tmp_path / "test.db")
    with db.connect(db_file) as conn:
        conn.execute("CREATE TABLE items (val TEXT)")
        conn.execute("INSERT INTO items VALUES ('synapse')")

    with db.connect(db_file) as conn:
        res = conn.query("SELECT * FROM items")
        assert len(res) == 1
        assert res[0]["val"] == "synapse"


# =========================================================================
# 2. std.compress (gzip, zlib, base64) Tests
# =========================================================================

def test_stdlib_compress():
    text = "Synapse AI Programming Language 2026! " * 20

    # gzip
    gz = compress.gzip_compress(text)
    assert len(gz) < len(text)
    decomp_gz = compress.gzip_decompress(gz)
    assert decomp_gz == text

    # zlib
    zl = compress.zlib_compress(text)
    assert len(zl) < len(text)
    decomp_zl = compress.zlib_decompress(zl)
    assert decomp_zl == text

    # Base64
    b64 = compress.b64_encode(text)
    assert isinstance(b64, str)
    unb64 = compress.b64_decode(b64)
    assert unb64 == text


# =========================================================================
# 3. std.path Tests
# =========================================================================

def test_stdlib_path():
    p = path.join("home", "user", "project", "main.syn")
    assert path.basename(p) == "main.syn"
    assert path.dirname(p) == path.join("home", "user", "project")
    assert path.ext(p) == ".syn"
    assert path.stem(p) == "main"
    assert path.normalize(p) == p


# =========================================================================
# 4. VM Task Pool Channels & try_catch Tests
# =========================================================================

def test_vm_channels_and_try_catch():
    vm = VirtualMachine()

    # Test Channel
    ch = vm.globals["channel"](capacity=2)
    ch.send("ping")
    assert ch.recv() == "ping"

    # Test try_catch in VM
    safe_div = vm.globals["try_catch"](
        lambda: 10 / 0,
        lambda err: "error_caught"
    )
    assert safe_div == "error_caught"

    # Test try_catch success
    success = vm.globals["try_catch"](
        lambda: 42 * 2,
        lambda err: "error"
    )
    assert success == 84


# =========================================================================
# 5. LSP Advanced Features Tests
# =========================================================================

def test_lsp_features():
    code = """
fn compute_loss(pred, target):
    return (pred - target) * 2

agent Evaluator:
    model: "gpt-4o"

let alpha = 0.05
"""
    # Go-to-Definition
    def_pos = find_definition(code, "compute_loss")
    assert def_pos is not None
    assert def_pos["line"] == 1

    def_agent = find_definition(code, "Evaluator")
    assert def_agent is not None
    assert def_agent["line"] == 4

    # Document Symbols
    symbols = get_document_symbols(code)
    names = [s["name"] for s in symbols]
    assert "compute_loss" in names
    assert "Evaluator" in names
    assert "alpha" in names

    # Signature Help
    call_line = "let l = compute_loss(p, "
    doc_with_call = code.strip() + "\n" + call_line
    doc_lines = doc_with_call.splitlines()
    sig = get_signature_help(doc_with_call, line_no=len(doc_lines) - 1, col_no=len(call_line))
    assert sig is not None
    assert sig["activeParameter"] == 1
    assert "compute_loss" in sig["signatures"][0]["label"]

    # Formatting
    formatted = format_document(code)
    assert "fn compute_loss" in formatted


def test_lsp_server_handlers():
    in_buf = io.BytesIO()
    out_buf = io.BytesIO()
    server = SynapseLanguageServer(reader=in_buf, writer=out_buf)

    code = "fn greet(name):\n    return 'Hello ' + name\n"
    server.documents["file:///test.syn"] = code

    # Initialize
    server.handle_initialize(1, {})
    assert server.is_initialized

    # Definition
    server.handle_definition(2, {
        "textDocument": {"uri": "file:///test.syn"},
        "position": {"line": 0, "character": 4}
    })

    # Document Symbol
    server.handle_document_symbol(3, {
        "textDocument": {"uri": "file:///test.syn"}
    })

    # Formatting
    server.handle_formatting(4, {
        "textDocument": {"uri": "file:///test.syn"}
    })

    output = out_buf.getvalue().decode("utf-8")
    assert '"result"' in output


# =========================================================================
# 6. SemVer & Package Manager Tests
# =========================================================================

def test_pkg_semver_constraints():
    # Caret
    assert SemverHelper.satisfies("1.2.3", "^1.0.0")
    assert not SemverHelper.satisfies("2.0.0", "^1.0.0")
    assert SemverHelper.satisfies("0.2.5", "^0.2.0")
    assert not SemverHelper.satisfies("0.3.0", "^0.2.0")

    # Tilde
    assert SemverHelper.satisfies("1.2.9", "~1.2.0")
    assert not SemverHelper.satisfies("1.3.0", "~1.2.0")

    # Comparisons
    assert SemverHelper.satisfies("2.5.0", ">=2.0.0")
    assert not SemverHelper.satisfies("1.9.0", ">=2.0.0")
    assert SemverHelper.satisfies("1.0.0", "==1.0.0")


def test_pkg_outdated(tmp_path):
    pm = PackageManager()
    manifest_path = pm.init_project(str(tmp_path), name="demo")
    
    # Simulate old package installed
    syn_modules = tmp_path / "syn_modules" / "synapse-nn"
    syn_modules.mkdir(parents=True)
    with open(syn_modules / "synapse.toml", "w") as f:
        f.write('[package]\nname = "synapse-nn"\nversion = "0.1.0"\n')

    outdated_items = pm.outdated(str(tmp_path))
    assert len(outdated_items) == 1
    assert outdated_items[0]["name"] == "synapse-nn"
    assert outdated_items[0]["current"] == "0.1.0"
    assert outdated_items[0]["latest"] == "0.2.0"


# =========================================================================
# 7. CLI Documentation Generator Tests
# =========================================================================

def test_cli_doc_generation(tmp_path):
    source_file = tmp_path / "module.syn"
    out_md = tmp_path / "module.md"

    source_code = (
        "fn calculate_metric(a, b):\n"
        "    return a * b + 1\n\n"
        "agent MetricsBot:\n"
        "    model: \"claude-3-5-sonnet\"\n"
        "    instructions: \"Analyze metrics.\"\n"
    )
    with open(source_file, "w", encoding="utf-8") as f:
        f.write(source_code)

    generate_docs(str(source_file), output=str(out_md))

    assert os.path.isfile(out_md)
    with open(out_md, "r", encoding="utf-8") as f:
        md = f.read()

    assert "calculate_metric" in md
    assert "MetricsBot" in md
    assert "claude-3-5-sonnet" in md
