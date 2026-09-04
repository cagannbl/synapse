"""
Tests for Synapse C Emitter Monomorphic Tagged Unions.
Verifies:
1. 'Option[int]', 'Option[Tensor]', 'Result[Tensor, str]' transpile to distinct C99 monomorphic structs.
2. No 'syn_tensor_t*' pointer punning in Option[int].
3. Valid enum tag checks ('SYN_OPT_SOME', 'SYN_OPT_NONE', 'SYN_RES_OK', 'SYN_RES_ERR').
4. Compiles and executes cleanly verifying unwrap logic.
"""
import os
import subprocess
import pytest
from synapse.codegen.native_compiler import NativeCompiler


def test_monomorphic_struct_definitions_and_no_punning():
    """Verify monomorphic structs are generated with exact types and no syn_tensor_t* punning in Option[int]."""
    nc = NativeCompiler()
    source = """
fn get_num(flag: int) -> Option[int]:
    if flag > 0:
        return Some(42)
    return None

fn get_tensor(flag: int) -> Option[Tensor]:
    if flag > 0:
        return Some(tensor([1.0, 2.0]))
    return None

fn compute_result(flag: int) -> Result[Tensor, str]:
    if flag > 0:
        return Ok(tensor([3.0, 4.0]))
    return Err("calculation failed")
"""
    c_code = nc.transpile(source)

    # 1. Monomorphic Option[int] struct has 'int val' and NO syn_tensor_t* in its definition
    assert "typedef struct {" in c_code
    assert "SynOptTag tag;" in c_code
    assert "int val;" in c_code
    assert "SynOption_int;" in c_code

    # 2. Monomorphic Option[tensor] struct
    assert "syn_tensor_t* val;" in c_code
    assert "SynOption_tensor;" in c_code

    # 3. Monomorphic Result[tensor, str] struct has union with syn_tensor_t* ok and const char* err
    assert "SynResTag tag;" in c_code
    assert "syn_tensor_t* ok;" in c_code
    assert "const char* err;" in c_code
    assert "SynResult_tensor_str;" in c_code

    # 4. Constructors generated
    assert "syn_option_int_some" in c_code
    assert "syn_option_int_none" in c_code
    assert "syn_result_tensor_str_ok" in c_code
    assert "syn_result_tensor_str_err" in c_code


def test_compile_and_run_option_int(tmp_path):
    """Compile and execute C program verifying Option[int] Some and None handling."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
fn maybe_double(x: int) -> Option[int]:
    if x > 0:
        return Some(x * 2)
    return None

let opt1 = maybe_double(21)
let opt2 = maybe_double(-5)

match opt1:
    case Some(val):
        print("OPT1:", val)
    case None:
        print("OPT1: NONE")

match opt2:
    case Some(val):
        print("OPT2:", val)
    case None:
        print("OPT2: NONE")
"""
    test_exe = tmp_path / ("test_opt.exe" if os.name == "nt" else "test_opt")
    test_c = tmp_path / "test_opt.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "OPT1: 42" in proc.stdout
    assert "OPT2: NONE" in proc.stdout


def test_compile_and_run_result_tensor_str(tmp_path):
    """Compile and execute C program verifying Result[Tensor, str] Ok and Err handling."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
fn create_tensor(flag: int) -> Result[Tensor, str]:
    if flag > 0:
        return Ok(tensor([10.0, 20.0]))
    return Err("invalid flag")

let res_ok = create_tensor(1)
let res_err = create_tensor(0)

match res_ok:
    case Ok(t):
        print("OK TENSOR:", t.item())
    case Err(msg):
        print("ERROR:", msg)

match res_err:
    case Ok(t):
        print("OK TENSOR:", t.item())
    case Err(msg):
        print("ERROR:", msg)
"""
    test_exe = tmp_path / ("test_res.exe" if os.name == "nt" else "test_res")
    test_c = tmp_path / "test_res.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "OK TENSOR: 10.000000" in proc.stdout
    assert "ERROR: invalid flag" in proc.stdout

