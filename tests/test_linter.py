import json
import os
import subprocess
import sys
import pytest

from synapse.tools.linter import SynapseLinter, LintDiagnostic


def run_synapse_cli(*args):
    cmd = [sys.executable, "-m", "synapse.cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


# =============================================================================
# SYN001: Tanımlanıp kullanılmayan değişkenler (Unused variable)
# =============================================================================

def test_syn001_unused_variable():
    linter = SynapseLinter()
    source = """let unused_var = 100
let used_var = 200
print(used_var)
"""
    diags = linter.lint_source(source)
    syn001_diags = [d for d in diags if d.code == "SYN001"]

    assert len(syn001_diags) == 1
    assert "unused_var" in syn001_diags[0].message
    assert syn001_diags[0].line == 1
    assert syn001_diags[0].severity == "warning"


def test_syn001_ignored_underscore():
    linter = SynapseLinter()
    source = """let _unused = 100
let _ = 200
"""
    diags = linter.lint_source(source)
    syn001_diags = [d for d in diags if d.code == "SYN001"]
    assert len(syn001_diags) == 0


def test_syn001_unused_in_function_scope():
    linter = SynapseLinter()
    source = """fn compute():
    let temp = 42
    let result = 10
    return result
"""
    diags = linter.lint_source(source)
    syn001_diags = [d for d in diags if d.code == "SYN001"]

    assert len(syn001_diags) == 1
    assert "temp" in syn001_diags[0].message
    assert syn001_diags[0].line == 2


# =============================================================================
# SYN002: Ulaşılamaz kod (Unreachable code after return/break/continue)
# =============================================================================

def test_syn002_unreachable_after_return():
    linter = SynapseLinter()
    source = """fn test_func():
    let a = 1
    return a
    let dead = 999
    print(dead)
"""
    diags = linter.lint_source(source)
    syn002_diags = [d for d in diags if d.code == "SYN002"]

    assert len(syn002_diags) >= 1
    assert syn002_diags[0].line == 4
    assert "return" in syn002_diags[0].message


def test_syn002_unreachable_after_break_and_continue():
    linter = SynapseLinter()
    source = """fn loop_test():
    while true:
        break
        let dead1 = 1
    for i in [1, 2]:
        continue
        let dead2 = 2
"""
    diags = linter.lint_source(source)
    syn002_diags = [d for d in diags if d.code == "SYN002"]

    assert len(syn002_diags) == 2
    assert any("break" in d.message for d in syn002_diags)
    assert any("continue" in d.message for d in syn002_diags)


# =============================================================================
# SYN003: Python sözdizimi sapması (def yerine fn, eksik let)
# =============================================================================

def test_syn003_python_def_divergence():
    linter = SynapseLinter()
    source = """def calculate(x, y):
    return x + y
"""
    diags = linter.lint_source(source)
    syn003_diags = [d for d in diags if d.code == "SYN003"]

    assert len(syn003_diags) >= 1
    assert any("use 'fn' instead of 'def'" in d.message for d in syn003_diags)
    assert any(d.severity == "error" for d in syn003_diags)


def test_syn003_missing_let_divergence():
    linter = SynapseLinter()
    source = """total = 100
print(total)
"""
    diags = linter.lint_source(source)
    syn003_diags = [d for d in diags if d.code == "SYN003"]

    assert len(syn003_diags) >= 1
    assert any("missing 'let' or 'const'" in d.message for d in syn003_diags)
    assert any(d.severity == "error" for d in syn003_diags)


def test_syn003_python_dl_import():
    linter = SynapseLinter()
    source = """import numpy as np
let x = 10
print(x)
"""
    diags = linter.lint_source(source)
    syn003_diags = [d for d in diags if d.code == "SYN003"]

    assert len(syn003_diags) >= 1
    assert any("external deep learning" in d.message for d in syn003_diags)


# =============================================================================
# SYN004: Eksik tensör şekil bildirimi uyarısı
# =============================================================================

def test_syn004_missing_tensor_shape_warning():
    linter = SynapseLinter()
    source = """let t = tensor([1, 2, 3])
let z = zeros(4, 4)
print(t)
print(z)
"""
    diags = linter.lint_source(source)
    syn004_diags = [d for d in diags if d.code == "SYN004"]

    assert len(syn004_diags) == 2
    assert any("t" in d.message for d in syn004_diags)
    assert any("z" in d.message for d in syn004_diags)


def test_syn004_valid_tensor_shape_no_warning():
    linter = SynapseLinter()
    source = """let t: Tensor[3] = tensor([1, 2, 3])
let z: Tensor[4, 4] = zeros(4, 4)
print(t)
print(z)
"""
    diags = linter.lint_source(source)
    syn004_diags = [d for d in diags if d.code == "SYN004"]
    assert len(syn004_diags) == 0


def test_syn004_plain_tensor_type_warning():
    linter = SynapseLinter()
    source = """let t: Tensor = tensor([1, 2])
print(t)
"""
    diags = linter.lint_source(source)
    syn004_diags = [d for d in diags if d.code == "SYN004"]
    assert len(syn004_diags) == 1
    assert "Missing tensor shape dimensions" in syn004_diags[0].message


# =============================================================================
# Clean Code & Format Report Tests
# =============================================================================

def test_linter_clean_code():
    linter = SynapseLinter()
    source = """fn add(a: int, b: int) -> int:
    return a + b

let x: int = 10
let y: int = 20
let result: int = add(x, y)
let data: Tensor[2] = tensor([1, 2])
print(result)
print(data)
"""
    diags = linter.lint_source(source)
    assert len(diags) == 0

    report = linter.format_report(diags)
    assert report == "No lint issues found."


def test_linter_format_report_json():
    linter = SynapseLinter()
    source = """let unused = 42
"""
    diags = linter.lint_source(source, filename="test.syn")
    json_str = linter.format_report(diags, as_json=True)

    data = json.loads(json_str)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["code"] == "SYN001"
    assert data[0]["filename"] == "test.syn"
    assert data[0]["severity"] == "warning"
    assert data[0]["line"] == 1


def test_linter_format_report_text():
    linter = SynapseLinter()
    source = """let unused = 42
"""
    diags = linter.lint_source(source, filename="example.syn")
    text = linter.format_report(diags, as_json=False)

    assert "example.syn:1:1: [SYN001] WARNING:" in text
    assert "Found 1 issue (0 errors, 1 warning)." in text


# =============================================================================
# CLI Subcommand Tests: synapse lint [files/dirs] [--json]
# =============================================================================

def test_cli_lint_clean_file(tmp_path):
    clean_file = str(tmp_path / "clean.syn")
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write("let x = 10\nprint(x)\n")

    res = run_synapse_cli("lint", clean_file)
    assert res.returncode == 0
    assert "No lint issues found." in res.stdout


def test_cli_lint_issues_with_json(tmp_path):
    issue_file = str(tmp_path / "issue.syn")
    with open(issue_file, "w", encoding="utf-8") as f:
        f.write("let dead = 100\n")

    res = run_synapse_cli("lint", issue_file, "--json")
    assert res.returncode == 0  # Only warning -> exits 0 by default
    data = json.loads(res.stdout.strip())
    assert isinstance(data, list)
    assert any(item["code"] == "SYN001" for item in data)


def test_cli_lint_directory(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "mod1.syn").write_text("let a = 1\nprint(a)\n", encoding="utf-8")
    (sub / "mod2.syn").write_text("let b = 2\nprint(b)\n", encoding="utf-8")

    res = run_synapse_cli("lint", str(sub))
    assert res.returncode == 0
    assert "No lint issues found." in res.stdout


def test_cli_lint_strict_mode(tmp_path):
    warn_file = str(tmp_path / "warn.syn")
    with open(warn_file, "w", encoding="utf-8") as f:
        f.write("let unused = 5\n")

    res = run_synapse_cli("lint", warn_file, "--strict")
    assert res.returncode == 1
    assert "SYN001" in res.stdout


# =============================================================================
# VS Code Extension Validation Tests
# =============================================================================

def test_vscode_package_json_valid():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "editors", "vscode")
    pkg_path = os.path.join(base_dir, "package.json")

    assert os.path.isfile(pkg_path), "editors/vscode/package.json missing"
    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)

    assert pkg["name"] == "synapse-lang"
    assert pkg["publisher"] == "synapse"
    lang_contrib = pkg["contributes"]["languages"][0]
    assert lang_contrib["id"] == "synapse"
    assert ".syn" in lang_contrib["extensions"]
    assert lang_contrib["configuration"] == "./language-configuration.json"

    grammar_contrib = pkg["contributes"]["grammars"][0]
    assert grammar_contrib["language"] == "synapse"
    assert grammar_contrib["scopeName"] == "source.synapse"
    assert grammar_contrib["path"] == "./syntaxes/synapse.tmLanguage.json"


def test_vscode_tm_language_json_valid():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "editors", "vscode")
    syntax_path = os.path.join(base_dir, "syntaxes", "synapse.tmLanguage.json")

    assert os.path.isfile(syntax_path), "synapse.tmLanguage.json missing"
    with open(syntax_path, "r", encoding="utf-8") as f:
        syntax = json.load(f)

    assert syntax["name"] == "Synapse"
    assert syntax["scopeName"] == "source.synapse"
    assert "patterns" in syntax
    assert "repository" in syntax

    repo = syntax["repository"]
    assert "keywords" in repo
    assert "operators" in repo
    assert "comments" in repo
    assert "strings" in repo

    # Verify key tokens are captured in grammar regexes
    keywords_json = json.dumps(repo)
    required_keywords = [
        "fn", "let", "const", "return", "if", "elif", "else", "while",
        "for", "in", "break", "continue", "prompt", "agent", "tool",
        "import", "enum", "tensor"
    ]
    for kw in required_keywords:
        assert kw in keywords_json, f"Keyword '{kw}' missing from TextMate grammar"

    # Operators |> and @
    assert "\\|>" in keywords_json or "|>" in keywords_json
    assert "@" in keywords_json


def test_vscode_language_configuration_json_valid():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "editors", "vscode")
    cfg_path = os.path.join(base_dir, "language-configuration.json")

    assert os.path.isfile(cfg_path), "language-configuration.json missing"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["comments"]["lineComment"] == "#"
    assert ["{", "}"] in cfg["brackets"]
    assert ["[", "]"] in cfg["brackets"]
    assert ["(", ")"] in cfg["brackets"]
    assert "indentationRules" in cfg
    assert "increaseIndentPattern" in cfg["indentationRules"]
    assert "decreaseIndentPattern" in cfg["indentationRules"]
