import json
import os
import pytest


def test_vscode_extension_files():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "vscode-synapse")

    pkg_path = os.path.join(base_dir, "package.json")
    lang_cfg_path = os.path.join(base_dir, "language-configuration.json")
    syntax_path = os.path.join(base_dir, "syntaxes", "synapse.tmLanguage.json")

    assert os.path.exists(pkg_path), "package.json missing"
    assert os.path.exists(lang_cfg_path), "language-configuration.json missing"
    assert os.path.exists(syntax_path), "synapse.tmLanguage.json missing"

    # JSON geçerlilik testi
    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)
        assert pkg["name"] == "synapse-language"
        exts = pkg["contributes"]["languages"][0]["extensions"]
        assert ".syn" in exts
        assert ".ai" in exts

    with open(lang_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg["comments"]["lineComment"] == "#"

    with open(syntax_path, "r", encoding="utf-8") as f:
        syntax = json.load(f)
        assert syntax["name"] == "Synapse"
        assert "patterns" in syntax
        assert "repository" in syntax
