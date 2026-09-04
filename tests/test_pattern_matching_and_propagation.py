"""
Tests for Synapse C Emitter Pattern Matching and '?' Error Propagation.
Verifies:
1. 'match' statements correctly unpack Option.Some/None and Result.Ok/Err in compiled C code.
2. The '?' (TryExpr) postfix operator properly short-circuits and propagates errors across function chains.
"""
import os
import subprocess
import pytest
from synapse.codegen.native_compiler import NativeCompiler


def test_pattern_matching_option_some_and_none(tmp_path):
    """Compile and execute C program testing pattern matching on Option."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
fn check_option(v: int) -> Option[int]:
    if v == 0:
        return None
    return Some(v * 10)

let a = check_option(5)
let b = check_option(0)

match a:
    case Some(val):
        print("A_SOME:", val)
    case None:
        print("A_NONE")

match b:
    case Some(val):
        print("B_SOME:", val)
    case None:
        print("B_NONE")
"""
    test_exe = tmp_path / ("match_opt.exe" if os.name == "nt" else "match_opt")
    test_c = tmp_path / "match_opt.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "A_SOME: 50" in proc.stdout
    assert "B_NONE" in proc.stdout


def test_pattern_matching_result_ok_and_err(tmp_path):
    """Compile and execute C program testing pattern matching on Result."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
fn divide(a: int, b: int) -> Result[int, str]:
    if b == 0:
        return Err("division by zero")
    return Ok(a / b)

let res1 = divide(100, 4)
let res2 = divide(10, 0)

match res1:
    case Ok(val):
        print("DIV1_OK:", val)
    case Err(err):
        print("DIV1_ERR:", err)

match res2:
    case Ok(val):
        print("DIV2_OK:", val)
    case Err(err):
        print("DIV2_ERR:", err)
"""
    test_exe = tmp_path / ("match_res.exe" if os.name == "nt" else "match_res")
    test_c = tmp_path / "match_res.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "DIV1_OK: 25" in proc.stdout
    assert "DIV2_ERR: division by zero" in proc.stdout


def test_try_operator_error_propagation_chain(tmp_path):
    """Compile and execute C program testing '?' operator propagating errors through function chain."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
fn parse_number(x: int) -> Result[int, str]:
    if x < 0:
        return Err("negative input")
    return Ok(x + 10)

fn validate_range(x: int) -> Result[int, str]:
    let num = parse_number(x)?
    if num > 50:
        return Err("number exceeds limit")
    return Ok(num * 2)

fn execute_pipeline(x: int) -> Result[int, str]:
    let val = validate_range(x)?
    return Ok(val + 1)

let ok_run = execute_pipeline(10)
let err_neg = execute_pipeline(-5)
let err_lim = execute_pipeline(45)

match ok_run:
    case Ok(v):
        print("RUN1:", v)
    case Err(e):
        print("ERR1:", e)

match err_neg:
    case Ok(v):
        print("RUN2:", v)
    case Err(e):
        print("ERR2:", e)

match err_lim:
    case Ok(v):
        print("RUN3:", v)
    case Err(e):
        print("ERR3:", e)
"""
    test_exe = tmp_path / ("prop_test.exe" if os.name == "nt" else "prop_test")
    test_c = tmp_path / "prop_test.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    # 10 + 10 = 20 -> 20 * 2 = 40 -> 40 + 1 = 41
    assert "RUN1: 41" in proc.stdout
    assert "ERR2: negative input" in proc.stdout
    # 45 + 10 = 55 > 50 -> "number exceeds limit"
    assert "ERR3: number exceeds limit" in proc.stdout


def test_try_operator_option_propagation(tmp_path):
    """Compile and execute C program testing '?' operator propagating None on Option."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
fn lookup(id: int) -> Option[int]:
    if id == 1:
        return Some(100)
    return None

fn compute(id: int) -> Option[int]:
    let val = lookup(id)?
    return Some(val + 50)

let found = compute(1)
let not_found = compute(2)

match found:
    case Some(v):
        print("FOUND:", v)
    case None:
        print("FOUND: NONE")

match not_found:
    case Some(v):
        print("NOT_FOUND:", v)
    case None:
        print("NOT_FOUND: NONE")
"""
    test_exe = tmp_path / ("prop_opt.exe" if os.name == "nt" else "prop_opt")
    test_c = tmp_path / "prop_opt.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "FOUND: 150" in proc.stdout
    assert "NOT_FOUND: NONE" in proc.stdout

