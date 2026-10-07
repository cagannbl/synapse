"""
Executes the Synapse code shown in the READMEs, launch drafts and examples, so
documentation can't silently drift away from what the language actually does.
"""

import io
import json
import os
import re
from contextlib import redirect_stdout

import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
READMES = ["README.md", "README.tr.md"]
LAUNCH_DRAFTS = ["docs/launch/SHOW_HN.md", "docs/launch/REDDIT_POSTS.md", "docs/launch/TWITTER_THREAD.md"]
# Snippets/examples that start a blocking web server are parsed but not executed.
BLOCKING_MARKER = "serve("


def _readme_snippets():
    for readme in READMES + LAUNCH_DRAFTS:
        with open(os.path.join(BASE_DIR, readme), encoding="utf-8") as f:
            text = f.read()
        for i, block in enumerate(re.findall(r"```python\n(.*?)```", text, re.S)):
            yield pytest.param(block, id=f"{readme}#{i}")


def _example_files():
    examples_dir = os.path.join(BASE_DIR, "examples")
    for root, _, files in os.walk(examples_dir):
        for name in sorted(files):
            if name.endswith(".syn"):
                path = os.path.join(root, name)
                yield pytest.param(path, id=os.path.relpath(path, examples_dir))


def _playground_programs():
    with open(os.path.join(BASE_DIR, "playground", "index.html"), encoding="utf-8") as f:
        html = f.read()
    block = html[html.index("const EXAMPLES = {"):html.index("// DOM References")]
    for key, code in re.findall(r'"(\d)": `(.*?)`', block, re.S):
        yield pytest.param(code.replace("\\`", "`").replace("\\\\", "\\"), id=f"playground-example-{key}")
    with open(os.path.join(BASE_DIR, "playground", "tour.json"), encoding="utf-8") as f:
        for i, step in enumerate(json.load(f)["steps"]):
            yield pytest.param(step["code"], id=f"playground-tour-{i + 1}")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No real LLM calls, and any files a program writes land in a temp dir."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def _run(source: str) -> str:
    ast = Parser(Lexer(source).tokenize()).parse()
    code = Compiler().compile(ast)
    if BLOCKING_MARKER in source:
        return ""
    out = io.StringIO()
    with redirect_stdout(out):
        VirtualMachine().execute(code)
    return out.getvalue()


@pytest.mark.parametrize("snippet", list(_readme_snippets()))
def test_readme_snippet_runs(snippet, offline):
    _run(snippet)


@pytest.mark.parametrize("path", list(_example_files()))
def test_example_runs(path, offline, monkeypatch):
    # Examples may reference sibling files relative to their own directory.
    monkeypatch.syspath_prepend(os.path.dirname(path))
    with open(path, encoding="utf-8") as f:
        _run(f.read())


@pytest.mark.parametrize("readme", READMES)
def test_quickstart_output_matches_readme(readme, offline):
    with open(os.path.join(BASE_DIR, readme), encoding="utf-8") as f:
        text = f.read()
    program = re.search(r"```python\n(# pipeline\.syn\n.*?)```", text, re.S).group(1)
    shown = re.search(r"\$ synapse run pipeline\.syn\n(.*?)\n```", text, re.S).group(1)
    assert _run(program).strip() == shown.strip()


@pytest.mark.parametrize("program", list(_playground_programs()))
def test_playground_program_runs(program, offline):
    _run(program)
