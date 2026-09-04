import os
import shutil
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from synapse.codegen.wasm_compiler import WasmCompiler, WasmBuildResult


def test_is_emcc_available(monkeypatch):
    """Test detection of emcc in PATH."""
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/emcc" if "emcc" in cmd else None)
    assert WasmCompiler.is_emcc_available() is True

    monkeypatch.setattr("shutil.which", lambda cmd: None)
    assert WasmCompiler.is_emcc_available() is False


def test_command_construction_default_flags():
    """Test emcc command generation with production flags."""
    wc = WasmCompiler()
    c_source = "source.c"
    js_output = "dist/app.js"

    cmd = wc.build_emcc_command(c_source, js_output)
    cmd_str = " ".join(cmd)

    # 1. Compiler binary
    assert cmd[0] == "emcc"

    # 2. Optimization flag
    assert "-O3" in cmd

    # 3. WebAssembly production flags
    assert "-s WASM=1" in cmd_str
    assert "-s ALLOW_MEMORY_GROWTH=1" in cmd_str
    assert "-s EXPORTED_FUNCTIONS=['_main']" in cmd_str
    assert "-s EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap', 'getValue', 'setValue']" in cmd_str
    assert "-s MODULARIZE=1" in cmd_str
    assert '-s EXPORT_NAME="SynapseModule"' in cmd_str

    # 4. Source & runtime files
    assert c_source in cmd
    assert wc.runtime_c in cmd
    assert f"-I{wc.runtime_dir}" in cmd
    assert "-lm" in cmd

    # 5. Output file
    assert "-o" in cmd
    out_idx = cmd.index("-o") + 1
    assert cmd[out_idx] == js_output


def test_command_construction_custom_exports():
    """Test custom exported functions formatting, trimming, and deduplication."""
    wc = WasmCompiler()
    exports = ["add", "_multiply", "relu_forward", "main", "   ", ""]

    cmd = wc.build_emcc_command("source.c", "app.js", exported_functions=exports)
    cmd_str = " ".join(cmd)

    # Every function should have leading underscore, _main present once, empty trimmed
    assert "EXPORTED_FUNCTIONS=['_main', '_add', '_multiply', '_relu_forward']" in cmd_str
    assert "'_'" not in cmd_str


def test_command_construction_optimization_levels():
    """Test different optimization flags (-O2, -Os, -Oz, -O0)."""
    wc = WasmCompiler()
    for opt in ["-O0", "-O1", "-O2", "-Os", "-Oz", "O3"]:
        cmd = wc.build_emcc_command("source.c", "app.js", optimize_level=opt)
        expected_opt = opt if opt.startswith("-") else f"-{opt}"
        assert expected_opt in cmd


def test_command_construction_extra_flags_and_export_name():
    """Test custom export name and extra compiler flags."""
    wc = WasmCompiler()
    cmd = wc.build_emcc_command(
        "source.c",
        "app.js",
        export_name="MyNeuralEngine",
        extra_flags=["-s", "ASSERTIONS=1", "--profiling"],
    )
    cmd_str = " ".join(cmd)

    assert '-s EXPORT_NAME="MyNeuralEngine"' in cmd_str
    assert "-s ASSERTIONS=1" in cmd_str
    assert "--profiling" in cmd


def test_transpile_synapse_code():
    """Test transpilation of Synapse code to C code via CEmitter."""
    wc = WasmCompiler()
    syn_code = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
print(C.item())
"""
    c_code = wc.transpile(syn_code)
    assert "syn_tensor_create" in c_code
    assert "syn_matmul" in c_code
    assert "int main(" in c_code


def test_transpile_file(tmp_path):
    """Test transpiling a .syn file to a .c file on disk."""
    wc = WasmCompiler()
    syn_file = tmp_path / "model.syn"
    c_file = tmp_path / "model.c"

    syn_file.write_text("let x = 42\nprint(x)\n", encoding="utf-8")
    wc.transpile_file(str(syn_file), str(c_file))

    assert c_file.exists()
    content = c_file.read_text(encoding="utf-8")
    assert "synapse_runtime.h" in content
    assert "int main(" in content


def test_extract_functions_from_synapse():
    """Test automatic extraction of top-level functions and signatures from Synapse source."""
    syn_code = """
fn predict(x: f64, threshold: f64) -> f64:
    if x > threshold:
        return 1.0
    return 0.0

fn reset():
    return 0

let a = 1
"""
    fns, sigs = WasmCompiler.extract_functions(syn_code)
    assert fns == ["predict", "reset"]
    assert "predict" in sigs
    assert sigs["predict"]["return_type"] == "f64"
    assert sigs["predict"]["arg_types"] == ["f64", "f64"]
    assert "reset" in sigs


def test_generate_html_harness_basic():
    """Test generation of companion HTML test runner without custom signatures."""
    wc = WasmCompiler()
    html = wc.generate_html_harness("app.js")

    assert "<!DOCTYPE html>" in html
    assert '<script src="app.js"></script>' in html
    assert "SynapseModule" in html
    assert "Execute _main()" in html
    assert "Execution Logs &amp; Terminal Output" in html
    assert "statusBadge" in html
    # Linear / Vercel style dark theme
    assert "#09090b" in html


def test_generate_html_harness_with_signatures():
    """Test generation of HTML runner with interactive function cards and ccall types."""
    wc = WasmCompiler()
    signatures = {
        "infer_step": {"return_type": "number", "arg_types": ["number", "number"]},
        "reset_state": {"return_type": "void", "arg_types": []},
        "get_loss": ("number", ["number"]),
    }
    html = wc.generate_html_harness("neural.js", function_signatures=signatures, module_name="NeuralNet")

    assert '<script src="neural.js"></script>' in html
    assert "NeuralNet" in html
    assert "card_infer_step" in html
    assert "card_reset_state" in html
    assert "card_get_loss" in html
    assert "infer_step_arg_0" in html
    assert "infer_step_arg_1" in html
    assert "invokeFunction('infer_step'" in html


def test_generate_html_harness_void_return():
    """Test that void return type is correctly mapped to null for ccall and displays success."""
    wc = WasmCompiler()
    signatures = {
        "clear_buffers": {"return_type": "void", "arg_types": []}
    }
    html = wc.generate_html_harness("test.js", function_signatures=signatures)
    assert "invokeFunction('clear_buffers', 'void'" in html
    assert "void (success)" in html


def test_write_html_harness_relative_paths(tmp_path):
    """Test writing HTML harness calculates correct relative path to JS file."""
    dist = tmp_path / "dist"
    dist.mkdir()
    out_html = dist / "runner.html"
    js_file = dist / "module.js"

    # Passing relative path from workspace root
    content = WasmCompiler.write_html_harness(str(out_html), str(js_file))

    assert out_html.exists()
    # In runner.html, the script tag should refer to module.js directly (not dist/module.js)
    assert '<script src="module.js"></script>' in content


def test_generate_node_runner():
    """Test Node.js runner script generation."""
    js_runner = WasmCompiler.generate_node_runner("app.js", module_name="CustomModule")

    assert "const path = require('path');" in js_runner
    assert "require(path.resolve(__dirname, 'app.js'))" in js_runner
    assert "CustomModule" in js_runner
    assert "async function run" in js_runner
    assert "moduleInstance._main()" in js_runner
    assert "module.exports = { run, createModule };" in js_runner


def test_write_node_runner_relative_paths(tmp_path):
    """Test writing Node.js runner calculates correct relative path without double dir."""
    out_dir = tmp_path / "build" / "out"
    out_dir.mkdir(parents=True)
    runner_path = out_dir / "run.js"
    js_path = out_dir / "app.js"

    content = WasmCompiler.write_node_runner(str(runner_path), str(js_path))
    assert runner_path.exists()
    # Should resolve to app.js relative to run.js
    assert "path.resolve(__dirname, 'app.js')" in content


def test_real_node_runner_execution(tmp_path):
    """Test actual execution of generated Node.js runner using the local Node runtime."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js not installed on host")

    # Create a mock Emscripten module in tmp_path
    mock_mod_js = tmp_path / "mock_module.js"
    mock_mod_js.write_text(
        """
module.exports = function createMockModule(opts = {}) {
    return Promise.resolve({
        _main: () => 0,
        ccall: (name) => 42
    });
};
""",
        encoding="utf-8",
    )

    runner_js = tmp_path / "test_runner.js"
    WasmCompiler.write_node_runner(str(runner_js), str(mock_mod_js), module_name="MockModule")

    res = subprocess.run([node_bin, str(runner_js)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "[Synapse WASM] Initializing MockModule in Node.js..." in res.stdout
    assert "[Synapse WASM] _main() returned: 0" in res.stdout


def test_compile_c_to_wasm_missing_emcc(tmp_path, monkeypatch):
    """Test that compile_c_to_wasm handles missing emcc gracefully."""
    wc = WasmCompiler()
    monkeypatch.setattr(wc, "is_emcc_available", lambda: False)

    c_file = tmp_path / "code.c"
    c_file.write_text("int main() { return 0; }", encoding="utf-8")
    wasm_out = tmp_path / "code.wasm"

    result = wc.compile_c_to_wasm(str(c_file), str(wasm_out))

    assert isinstance(result, WasmBuildResult)
    assert result.success is False
    assert result.wasm_path == str(wasm_out)
    assert result.js_path == str(tmp_path / "code.js")
    assert "Emscripten compiler 'emcc' not found" in result.error_message
    assert result.command is not None
    assert "emcc" in result.command[0]


def test_compile_c_to_wasm_non_existent_file_with_main(tmp_path):
    """Test that a non-existent file path containing 'main' is not misclassified as inline C code."""
    wc = WasmCompiler()
    missing_file = str(tmp_path / "my_main_program.c")

    result = wc.compile_c_to_wasm(missing_file, str(tmp_path / "out.wasm"))
    assert result.success is False
    assert f"C source file does not exist: {missing_file}" in result.error_message


def test_compile_c_to_wasm_from_code_string(tmp_path, monkeypatch):
    """Test compiling raw C code string without an existing file."""
    wc = WasmCompiler()
    monkeypatch.setattr(wc, "is_emcc_available", lambda: True)

    raw_c = "int main() { return 42; }"
    wasm_out = tmp_path / "direct.wasm"
    js_out = tmp_path / "direct.js"

    def mock_run(cmd, capture_output=True, text=True):
        c_src = [arg for arg in cmd if arg.endswith(".c") and "synapse_runtime" not in arg][0]
        assert os.path.isfile(c_src)
        with open(c_src, "r", encoding="utf-8") as f:
            assert "return 42" in f.read()

        with open(js_out, "w", encoding="utf-8") as f:
            f.write("// js")
        with open(wasm_out, "wb") as f:
            f.write(b"\x00asm")
        res = MagicMock()
        res.returncode = 0
        res.stdout = ""
        res.stderr = ""
        return res

    with patch("subprocess.run", side_effect=mock_run):
        result = wc.compile_c_to_wasm(raw_c, str(wasm_out))
        assert result.success is True


def test_compile_c_to_wasm_mock_success(tmp_path, monkeypatch):
    """Test mock successful execution of emcc with both HTML and Node runner emission."""
    wc = WasmCompiler()
    monkeypatch.setattr(wc, "is_emcc_available", lambda: True)

    c_file = tmp_path / "test.c"
    c_file.write_text("int main() { return 0; }", encoding="utf-8")
    wasm_out = tmp_path / "test.wasm"
    js_out = tmp_path / "test.js"

    def mock_run(cmd, capture_output=True, text=True):
        with open(js_out, "w", encoding="utf-8") as f:
            f.write("// emcc generated js")
        with open(wasm_out, "wb") as f:
            f.write(b"\x00asm\x01\x00\x00\x00")
        res = MagicMock()
        res.returncode = 0
        res.stdout = "Compiled successfully"
        res.stderr = ""
        return res

    with patch("subprocess.run", side_effect=mock_run):
        result = wc.compile_c_to_wasm(str(c_file), str(wasm_out), emit_html=True, emit_node=True)

        assert result.success is True
        assert result.wasm_path == str(wasm_out)
        assert result.js_path == str(js_out)
        assert result.html_path == str(tmp_path / "test.html")
        assert os.path.isfile(result.html_path)
        assert result.node_runner_path == str(tmp_path / "test_runner.js")
        assert os.path.isfile(result.node_runner_path)
        assert result.error_message is None


def test_compile_c_to_wasm_mock_failure(tmp_path, monkeypatch):
    """Test handling of compilation failure from emcc."""
    wc = WasmCompiler()
    monkeypatch.setattr(wc, "is_emcc_available", lambda: True)

    c_file = tmp_path / "bad.c"
    c_file.write_text("int main() { syntax_error; }", encoding="utf-8")
    wasm_out = tmp_path / "bad.wasm"

    def mock_run(cmd, capture_output=True, text=True):
        res = MagicMock()
        res.returncode = 1
        res.stdout = ""
        res.stderr = "error: use of undeclared identifier 'syntax_error'"
        return res

    with patch("subprocess.run", side_effect=mock_run):
        result = wc.compile_c_to_wasm(str(c_file), str(wasm_out))
        assert result.success is False
        assert "use of undeclared identifier 'syntax_error'" in result.error_message


def test_compile_synapse_to_wasm_convenience(tmp_path, monkeypatch):
    """Test compile_synapse_to_wasm end-to-end wrapper with auto function discovery."""
    wc = WasmCompiler()
    monkeypatch.setattr(wc, "is_emcc_available", lambda: True)

    syn_file = tmp_path / "app.syn"
    syn_file.write_text("fn add(a: f64, b: f64) -> f64:\n    return a + b\nlet x = add(1.0, 2.0)\n", encoding="utf-8")
    wasm_out = tmp_path / "app.wasm"
    js_out = tmp_path / "app.js"

    def mock_run(cmd, capture_output=True, text=True):
        cmd_str = " ".join(cmd)
        # Ensure auto-discovered 'add' function was exported
        assert "'_add'" in cmd_str
        with open(js_out, "w", encoding="utf-8") as f:
            f.write("// js")
        with open(wasm_out, "wb") as f:
            f.write(b"\x00asm")
        res = MagicMock()
        res.returncode = 0
        res.stdout = "OK"
        res.stderr = ""
        return res

    with patch("subprocess.run", side_effect=mock_run):
        res = wc.compile_synapse_to_wasm(str(syn_file), str(wasm_out), emit_html=True)
        assert res.success is True
        assert os.path.isfile(str(tmp_path / "app.c"))
        assert res.html_path == str(tmp_path / "app.html")
        html_text = open(res.html_path, encoding="utf-8").read()
        assert "card_add" in html_text


def test_cli_build_help_includes_wasm_target():
    """Test that synapse build --help includes --target with native and wasm and --node."""
    res = subprocess.run(
        [sys.executable, "-m", "synapse.cli", "build", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--target" in res.stdout
    assert "native" in res.stdout
    assert "wasm" in res.stdout
    assert "--html" in res.stdout
    assert "--node" in res.stdout


def test_cli_build_wasm_c_only(tmp_path):
    """Test CLI build with --target wasm and --c-only flag."""
    syn_file = tmp_path / "wasm_demo.syn"
    syn_file.write_text("let a = 123\nprint(a)\n", encoding="utf-8")

    res = subprocess.run(
        [sys.executable, "-m", "synapse.cli", "build", str(syn_file), "--target", "wasm", "--c-only"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "C source emitted:" in res.stdout
    c_out = tmp_path / "wasm_demo.c"
    assert c_out.exists()


def test_cli_build_wasm_notice_when_emcc_missing(tmp_path):
    """Test CLI build with --target wasm outputs notice if emcc is not installed."""
    syn_file = tmp_path / "wasm_notice.syn"
    syn_file.write_text("let a = 123\nprint(a)\n", encoding="utf-8")

    res = subprocess.run(
        [sys.executable, "-m", "synapse.cli", "build", str(syn_file), "--target", "wasm"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Build notice:" in res.stdout
    assert "emcc" in res.stdout


def test_cli_build_missing_file_error(tmp_path):
    """Test CLI build reports clean error when input file does not exist."""
    res = subprocess.run(
        [sys.executable, "-m", "synapse.cli", "build", str(tmp_path / "missing.syn"), "--target", "wasm"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "Error: File not found" in res.stderr
