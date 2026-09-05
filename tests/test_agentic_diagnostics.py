import json
import os
import subprocess
import sys
import pytest

from synapse.core.diagnostics import Diagnostic, DiagnosticReport, diagnose_code


def run_synapse_cli(*args):
    cmd = [sys.executable, "-m", "synapse.cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return result


def test_diagnostic_dataclass_fields_and_methods():
    diag = Diagnostic(
        file="neural_net.syn",
        line=15,
        column=10,
        severity="error",
        code="SYN-E202",
        message="Cannot multiply tensor of shape (32, 64) with (128, 64)",
        expected="64",
        actual="128",
        suggested_fix="Transpose tensor with .T",
        diff="--- a/neural_net.syn\n+++ b/neural_net.syn\n@@ -15 +15 @@\n-let y = x @ w\n+let y = x @ w.T\n",
    )

    d_dict = diag.to_dict()
    assert d_dict["file"] == "neural_net.syn"
    assert d_dict["line"] == 15
    assert d_dict["column"] == 10
    assert d_dict["severity"] == "error"
    assert d_dict["code"] == "SYN-E202"
    assert d_dict["message"] == "Cannot multiply tensor of shape (32, 64) with (128, 64)"
    assert d_dict["expected"] == "64"
    assert d_dict["actual"] == "128"
    assert d_dict["suggested_fix"] == "Transpose tensor with .T"
    assert "w.T" in d_dict["diff"]

    json_str = diag.to_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed == d_dict


def test_diagnostic_utf8_determinism():
    diag = Diagnostic(
        file="türkçe_şeması.syn",
        line=1,
        column=1,
        severity="warning",
        code="SYN-W101",
        message="Öğrenme oranı ve tensör boyutu uyuşmazlığı: ğüşiöç",
        expected="Tensör[32, 64]",
        actual="Tensör[128, 64]",
        suggested_fix="Tensörü .T ile devrik yapın",
    )
    json_str = diag.to_json(indent=2)
    assert "türkçe_şeması.syn" in json_str
    assert "öç" in json_str
    parsed = json.loads(json_str)
    assert parsed["message"] == "Öğrenme oranı ve tensör boyutu uyuşmazlığı: ğüşiöç"


def test_diagnostic_report_methods_and_backward_compatibility():
    report_ok = DiagnosticReport(status="ok", diagnostics=[])
    dict_ok = report_ok.to_dict()
    assert dict_ok["status"] == "ok"
    assert dict_ok["diagnostics"] == []

    parsed_ok = json.loads(report_ok.to_json(indent=2))
    assert parsed_ok["status"] == "ok"
    assert parsed_ok["diagnostics"] == []

    # Error report
    diag = Diagnostic(
        file="test.syn",
        line=5,
        column=3,
        severity="error",
        code="SYN-E101",
        message="Unexpected token",
    )
    report_err = DiagnosticReport(status="error", diagnostics=[diag])
    dict_err = report_err.to_dict()
    assert dict_err["status"] == "error"
    assert len(dict_err["diagnostics"]) == 1
    assert dict_err["diagnostics"][0]["code"] == "SYN-E101"

    # Backward compatibility properties and indexing
    assert report_err.file == "test.syn"
    assert report_err.line == 5
    assert report_err.message == "Unexpected token"
    assert report_err["status"] == "error"
    assert report_err.get("status") == "error"


def test_tensor_shape_mismatch_static_diagnostics():
    source = """let a = zeros([32, 64])
let b = zeros([128, 64])
let c = a @ b
"""
    report = diagnose_code(source, filepath="matmul_err.syn")
    assert report.status == "error"
    assert len(report.diagnostics) >= 1

    diag = report.diagnostics[0]
    assert diag.code == "SYN-E202"
    assert diag.expected == "64"
    assert diag.actual == "128"
    assert "b.T" in diag.suggested_fix or "Transpose" in diag.suggested_fix
    assert diag.diff is not None
    assert "b.T" in diag.diff
    assert "--- a/matmul_err.syn" in diag.diff

    # Ensure JSON conversion matches agentic requirements
    parsed = json.loads(report.to_json(indent=2))
    assert parsed["status"] == "error"
    assert len(parsed["diagnostics"]) == 1
    assert parsed["diagnostics"][0]["code"] == "SYN-E202"
    assert parsed["diagnostics"][0]["expected"] == "64"
    assert parsed["diagnostics"][0]["actual"] == "128"
    assert "b.T" in parsed["diagnostics"][0]["suggested_fix"] or "Transpose" in parsed["diagnostics"][0]["suggested_fix"]


def test_cli_check_format_json_and_agent_success(tmp_path):
    good_syn = str(tmp_path / "valid.syn")
    with open(good_syn, "w", encoding="utf-8") as f:
        f.write("fn add(a: int, b: int) -> int:\n    return a + b\nlet res = add(10, 20)\n")

    # 1. Test check --format=json
    res_fmt = run_synapse_cli("check", good_syn, "--format=json")
    assert res_fmt.returncode == 0
    data_fmt = json.loads(res_fmt.stdout.strip())
    assert data_fmt["status"] == "ok"
    assert data_fmt["diagnostics"] == []

    # 2. Test check --agent
    res_agent = run_synapse_cli("check", good_syn, "--agent")
    assert res_agent.returncode == 0
    data_agent = json.loads(res_agent.stdout.strip())
    assert data_agent["status"] == "ok"
    assert data_agent["diagnostics"] == []

    # Ensure completely silent stdout (pure valid JSON without ANSI or banners)
    assert "\033[" not in res_agent.stdout
    assert "Synapse" not in res_agent.stdout.replace('"diagnostics"', "")


def test_cli_check_format_json_and_agent_syntax_error(tmp_path):
    bad_syn = str(tmp_path / "syntax_error.syn")
    with open(bad_syn, "w", encoding="utf-8") as f:
        f.write("fn broken(\n")

    res_fmt = run_synapse_cli("check", bad_syn, "--format=json")
    assert res_fmt.returncode == 1
    data_fmt = json.loads(res_fmt.stdout.strip())
    assert data_fmt["status"] == "error"
    assert len(data_fmt["diagnostics"]) >= 1
    assert data_fmt["diagnostics"][0]["severity"] == "error"
    assert data_fmt["diagnostics"][0]["code"].startswith("SYN-")

    res_agent = run_synapse_cli("check", bad_syn, "--agent")
    assert res_agent.returncode == 1
    data_agent = json.loads(res_agent.stdout.strip())
    assert data_agent["status"] == "error"
    assert len(data_agent["diagnostics"]) >= 1


def test_cli_check_tensor_shape_mismatch_agent(tmp_path):
    mismatch_syn = str(tmp_path / "tensor_err.syn")
    with open(mismatch_syn, "w", encoding="utf-8") as f:
        f.write("let a = zeros([32, 64])\nlet b = zeros([128, 64])\nlet c = a @ b\n")

    res_agent = run_synapse_cli("check", mismatch_syn, "--agent")
    assert res_agent.returncode == 1

    data = json.loads(res_agent.stdout.strip())
    assert data["status"] == "error"
    assert len(data["diagnostics"]) >= 1

    diag = data["diagnostics"][0]
    assert diag["code"] == "SYN-E202"
    assert diag["expected"] == "64"
    assert diag["actual"] == "128"
    assert "b.T" in diag["suggested_fix"] or "Transpose" in diag["suggested_fix"]
    assert diag["diff"] is not None
    assert "b.T" in diag["diff"]


def test_cli_run_agent_mode(tmp_path):
    good_syn = str(tmp_path / "run_ok.syn")
    with open(good_syn, "w", encoding="utf-8") as f:
        f.write("let x = 100\nprint(x)\n")

    res_ok = run_synapse_cli("run", good_syn, "--agent", "--vm")
    assert res_ok.returncode == 0
    data_ok = json.loads(res_ok.stdout.strip())
    assert data_ok["status"] == "ok"
    assert data_ok["diagnostics"] == []

    bad_syn = str(tmp_path / "run_bad.syn")
    with open(bad_syn, "w", encoding="utf-8") as f:
        f.write("fn broken(\n")

    res_err = run_synapse_cli("run", bad_syn, "--agent", "--vm")
    assert res_err.returncode == 1
    data_err = json.loads(res_err.stdout.strip())
    assert data_err["status"] == "error"
    assert len(data_err["diagnostics"]) >= 1


def test_cli_lint_agent_mode(tmp_path):
    good_syn = str(tmp_path / "lint_clean.syn")
    with open(good_syn, "w", encoding="utf-8") as f:
        f.write("fn calculate(x: int) -> int:\n    return x * 2\n")

    res = run_synapse_cli("lint", good_syn, "--agent")
    assert res.returncode == 0
    data = json.loads(res.stdout.strip())
    assert data["status"] == "ok"
    assert data["diagnostics"] == []

    bad_syn = str(tmp_path / "lint_err.syn")
    with open(bad_syn, "w", encoding="utf-8") as f:
        f.write("fn broken(\n")

    res_err = run_synapse_cli("lint", bad_syn, "--agent")
    assert res_err.returncode == 1
    data_err = json.loads(res_err.stdout.strip())
    assert data_err["status"] == "error"
    assert len(data_err["diagnostics"]) >= 1


def test_cli_build_agent_mode(tmp_path):
    good_syn = str(tmp_path / "build_ok.syn")
    with open(good_syn, "w", encoding="utf-8") as f:
        f.write("let x = 42\nprint(x)\n")

    res_ok = run_synapse_cli("build", good_syn, "--c-only", "--agent")
    assert res_ok.returncode == 0
    data_ok = json.loads(res_ok.stdout.strip())
    assert data_ok["status"] == "ok"
    assert data_ok["diagnostics"] == []

    bad_syn = str(tmp_path / "build_bad.syn")
    with open(bad_syn, "w", encoding="utf-8") as f:
        f.write("fn broken(\n")

    res_err = run_synapse_cli("build", bad_syn, "--c-only", "--agent")
    assert res_err.returncode == 1
    data_err = json.loads(res_err.stdout.strip())
    assert data_err["status"] == "error"
    assert len(data_err["diagnostics"]) >= 1
