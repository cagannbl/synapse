import os
import pytest
from synapse.codegen.native_compiler import NativeCompiler


def test_native_compiler_transpile():
    nc = NativeCompiler()
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
"""
    c_code = nc.transpile(source)
    assert "syn_tensor_create" in c_code
    assert "syn_matmul" in c_code
    assert "int main(" in c_code


def test_native_compiler_file_output(tmp_path):
    nc = NativeCompiler()
    syn_file = tmp_path / "test.syn"
    c_file = tmp_path / "test.c"

    syn_file.write_text("let x = tensor(42.0)\nprint(x.item())\n", encoding="utf-8")
    nc.transpile_file(str(syn_file), str(c_file))

    assert c_file.exists()
    content = c_file.read_text(encoding="utf-8")
    assert "syn_tensor_scalar(42.0" in content


def test_build_executable_compiler_flags_gcc(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    nc = NativeCompiler()
    c_file = tmp_path / "test.c"
    c_file.write_text("int main() { return 0; }", encoding="utf-8")
    exe_file = tmp_path / "output_prog"

    monkeypatch.setattr(nc, "find_c_compiler", lambda: ("gcc", "gcc"))

    called_cmds = []

    def mock_run(cmd, capture_output=True, text=True):
        called_cmds.append(cmd)
        # Create output file to simulate successful compilation
        out_idx = cmd.index("-o") + 1
        with open(cmd[out_idx], "wb") as f:
            f.write(b"fake_binary")
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    with patch("subprocess.run", side_effect=mock_run):
        ok, res = nc.build_executable(str(c_file), str(exe_file))
        assert ok is True
        assert len(called_cmds) > 0
        # -O3 ve -mavx2 bayraklarının ilk denemede bulunduğunu doğrula
        first_cmd = called_cmds[0]
        assert "-fopenmp" in first_cmd
        assert "-O3" in first_cmd
        assert "-mavx2" in first_cmd
        assert "-mfma" in first_cmd
        assert first_cmd[-1].endswith(".exe") if os.name == "nt" else True


def test_build_executable_compiler_flags_msvc(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    nc = NativeCompiler()
    c_file = tmp_path / "test.c"
    c_file.write_text("int main() { return 0; }", encoding="utf-8")
    exe_file = tmp_path / "output_prog"

    monkeypatch.setattr(nc, "find_c_compiler", lambda: ("cl.exe", "cl"))

    called_cmds = []

    def mock_run(cmd, capture_output=True, text=True):
        called_cmds.append(cmd)
        # Extract output exe path from /Fe:
        for arg in cmd:
            if arg.startswith("/Fe:"):
                out_path = arg[4:]
                with open(out_path, "wb") as f:
                    f.write(b"fake_binary")
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    with patch("subprocess.run", side_effect=mock_run):
        ok, res = nc.build_executable(str(c_file), str(exe_file))
        assert ok is True
        assert len(called_cmds) > 0
        first_cmd = called_cmds[0]
        assert "/openmp" in first_cmd
        assert "/O2" in first_cmd
        assert "/arch:AVX2" in first_cmd

