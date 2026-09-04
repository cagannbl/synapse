import json
import os
import subprocess
import sys
import pytest


def run_synapse_cli(*args):
    cmd = [sys.executable, "-m", "synapse.cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def test_cli_build_c_only(tmp_path):
    syn_file = str(tmp_path / "hello.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("let x = 42\nprint(x)\n")

    res = run_synapse_cli("build", syn_file, "--c-only")
    assert res.returncode == 0
    assert "C source emitted:" in res.stdout

    expected_c = str(tmp_path / "hello.c")
    assert os.path.isfile(expected_c)
    with open(expected_c, "r", encoding="utf-8") as f:
        c_code = f.read()
    assert "synapse_runtime.h" in c_code


def test_cli_pkg_lifecycle(tmp_path):
    proj_dir = str(tmp_path / "cli_proj")

    # 1. pkg init
    res_init = run_synapse_cli("pkg", "init", proj_dir, "--name", "cli_test_pkg")
    assert res_init.returncode == 0
    assert "Initialized Synapse project:" in res_init.stdout
    assert os.path.isfile(os.path.join(proj_dir, "synapse.toml"))

    # 2. pkg add
    res_add = run_synapse_cli("pkg", "add", "demo-module", "--dir", proj_dir)
    assert res_add.returncode == 0
    assert "Added package 'demo-module'" in res_add.stdout

    # 3. pkg list
    res_list = run_synapse_cli("pkg", "list", "--dir", proj_dir)
    assert res_list.returncode == 0
    assert "demo-module" in res_list.stdout

    # 4. pkg remove
    res_rm = run_synapse_cli("pkg", "remove", "demo-module", "--dir", proj_dir)
    assert res_rm.returncode == 0
    assert "Removed package 'demo-module'" in res_rm.stdout


def test_cli_device_and_info():
    res_dev = run_synapse_cli("device")
    assert res_dev.returncode == 0
    assert "Synapse Hardware & Acceleration Info" in res_dev.stdout
    assert "CUDA Available:" in res_dev.stdout

    res_info = run_synapse_cli("info")
    assert res_info.returncode == 0
    assert "Platform:" in res_info.stdout


def test_cli_check_command(tmp_path):
    good_syn = str(tmp_path / "good.syn")
    with open(good_syn, "w", encoding="utf-8") as f:
        f.write("fn calc():\n    return 10 + 20\ncalc()\n")

    res_good = run_synapse_cli("check", good_syn)
    assert res_good.returncode == 0
    assert "Check PASSED:" in res_good.stdout

    # JSON check output
    res_json = run_synapse_cli("check", good_syn, "--json")
    assert res_json.returncode == 0
    data = json.loads(res_json.stdout.strip())
    assert data["status"] == "ok"

    # Bad syntax file
    bad_syn = str(tmp_path / "bad.syn")
    with open(bad_syn, "w", encoding="utf-8") as f:
        f.write("fn invalid(\n")

    res_bad = run_synapse_cli("check", bad_syn)
    assert res_bad.returncode == 1
    assert "Check FAILED" in res_bad.stdout

    # Type contract violation
    type_err_syn = str(tmp_path / "type_err.syn")
    with open(type_err_syn, "w", encoding="utf-8") as f:
        f.write('let x: int = "string_value"\n')

    res_type_err = run_synapse_cli("check", type_err_syn)
    assert res_type_err.returncode == 1
    assert "Check FAILED" in res_type_err.stdout


def test_cli_pkg_search():
    res = run_synapse_cli("pkg", "search", "nn")
    assert res.returncode == 0
    assert "synapse-nn" in res.stdout


def test_cli_test_help():
    res = run_synapse_cli("test", "--help")
    assert res.returncode == 0
    assert "Run Synapse test suite" in res.stdout
    assert "--verbose" in res.stdout


def test_cli_test_execution(tmp_path):
    # Test passing .syn file
    test_syn = str(tmp_path / "test_sample.syn")
    with open(test_syn, "w", encoding="utf-8") as f:
        f.write("fn test_ok():\n    assert(10 + 20 == 30)\n")

    res = run_synapse_cli("test", str(tmp_path))
    assert res.returncode == 0
    assert "PASSED" in res.stdout
    assert "test_ok" in res.stdout

    # Test failing .syn file
    fail_syn = str(tmp_path / "test_fail.syn")
    with open(fail_syn, "w", encoding="utf-8") as f:
        f.write("fn test_bad():\n    assert(1 == 2)\n")

    res_fail = run_synapse_cli("test", str(tmp_path))
    assert res_fail.returncode == 1
    assert "FAILED" in res_fail.stdout
    assert "test_bad" in res_fail.stdout


def test_cli_repl_sanitization():
    from synapse.cli import _sanitize_repl_input
    assert _sanitize_repl_input("synapse> let a = tensor([1.0, 2.0, 3.0])") == "let a = tensor([1.0, 2.0, 3.0])"
    assert _sanitize_repl_input("synapse> synapse> print(a * 2)") == "print(a * 2)"
    assert _sanitize_repl_input(">>> let b = 42") == "let b = 42"
    assert _sanitize_repl_input("$ exit") == "exit"
    assert _sanitize_repl_input("let x = 5 > 3") == "let x = 5 > 3"


def test_cli_repl_interactive_execution():
    from synapse.lexer.lexer import Lexer
    from synapse.parser.parser import Parser
    from synapse.vm.compiler import Compiler
    from synapse.vm.virtual_machine import VirtualMachine
    import io
    from contextlib import redirect_stdout

    vm = VirtualMachine()
    # 1. State preservation across statements
    ast1 = Parser(Lexer("let a = tensor([1.0, 2.0, 3.0])").tokenize()).parse()
    vm.execute(Compiler(name="<repl>", repl_mode=True).compile(ast1))
    assert "a" in vm.globals

    # 2. Interactive auto-display for expressions
    buf = io.StringIO()
    with redirect_stdout(buf):
        ast2 = Parser(Lexer("a * 2").tokenize()).parse()
        vm.execute(Compiler(name="<repl>", repl_mode=True).compile(ast2))
    assert "tensor([2.0, 4.0, 6.0])" in buf.getvalue()


def test_cli_repl_piped_self_healing():
    input_text = "synapse> let a = tensor([1.0, 2.0])\nsynapse> print(a * 2)\nsynapse> exit()\n"
    cmd = [sys.executable, "-m", "synapse.cli", "repl"]
    res = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    assert res.returncode == 0
    assert "tensor([2.0, 4.0])" in res.stdout
    assert "Exiting Synapse REPL." in res.stdout

