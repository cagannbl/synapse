import os
import tempfile
import subprocess
import pytest
from synapse.codegen.c_emitter import CEmitter
from synapse.codegen.native_compiler import NativeCompiler
from synapse.parser.parser import Parser
from synapse.lexer.lexer import Lexer


def compile_and_run_synapse(source: str, expected_output_substr: str = ""):
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    emitter = CEmitter()
    c_code = emitter.emit(ast, filename="escape_test.syn")

    nc = NativeCompiler()
    comp_info = nc.find_c_compiler()
    if not comp_info:
        pytest.skip("No C compiler available")
    compiler_path, comp_type = comp_info

    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "test.c")
        exe_file = os.path.join(tmpdir, "test.exe")

        with open(c_file, "w", encoding="utf-8") as f:
            f.write(c_code)

        if comp_type == "zig":
            cmd = [compiler_path, "cc", c_file, nc.runtime_c, f"-I{nc.runtime_dir}", "-O2", "-o", exe_file]
        elif comp_type in ("gcc", "clang"):
            cmd = [compiler_path, c_file, nc.runtime_c, f"-I{nc.runtime_dir}", "-O2", "-o", exe_file]
        elif comp_type == "cl":
            cmd = [compiler_path, c_file, nc.runtime_c, f"/I{nc.runtime_dir}", "/O2", f"/Fe:{exe_file}"]
        else:
            pytest.skip(f"Unsupported compiler {comp_type}")

        comp_res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert comp_res.returncode == 0, f"Compilation failed:\n{comp_res.stderr}\n{comp_res.stdout}\nCode:\n{c_code}"

        run_res = subprocess.run([exe_file], capture_output=True, text=True, timeout=10)
        assert run_res.returncode == 0, f"Execution failed:\n{run_res.stderr}\n{run_res.stdout}"
        if expected_output_substr:
            assert expected_output_substr in run_res.stdout, f"Expected '{expected_output_substr}' in output:\n{run_res.stdout}"
        return c_code, run_res.stdout


def test_loop_outer_tensor_accumulation_escape_safety():
    """
    Verifies that accumulating tensors from within a loop into an outer variable
    does not cause Use-After-Free when arena scopes reset each iteration.
    """
    source = """
fn compute_sum() -> Tensor:
    let total = tensor([0.0])
    for i in range(100):
        let step = tensor([1.0])
        total = total + step
    return total

let result = compute_sum()
print("TOTAL:", result.item())
"""
    c_code, stdout = compile_and_run_synapse(source, expected_output_substr="TOTAL: 100.000000")
    assert "syn_tensor_escape" in c_code


def test_function_return_tensor_from_loop_escape_safety():
    """
    Verifies that returning a tensor computed inside a loop escapes safely to the caller.
    """
    source = """
fn find_first_positive() -> Tensor:
    for i in range(10):
        if i == 5:
            let found = tensor([42.0])
            return found
    return tensor([0.0])

let res = find_first_positive()
print("FOUND:", res.item())
"""
    c_code, stdout = compile_and_run_synapse(source, expected_output_substr="FOUND: 42.000000")
    assert "syn_tensor_escape" in c_code
