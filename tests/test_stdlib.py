"""Comprehensive Test Suite for Synapse Built-in Standard Library (std)."""

import math
import os
import sys
import time
import pytest

from synapse.stdlib import std, StandardLibrary
import synapse.stdlib.math as s_math
import synapse.stdlib.fs as s_fs
import synapse.stdlib.crypto as s_crypto
import synapse.stdlib.time as s_time
import synapse.stdlib.sys as s_sys
import synapse.stdlib.http as s_http
import synapse.stdlib.regex as s_regex
from synapse.core.tensor import Tensor, tensor
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_source(source: str, vm: VirtualMachine = None) -> VirtualMachine:
    """Helper to compile and execute Synapse source code in VM."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    if vm is None:
        vm = VirtualMachine()
    vm.execute(code)
    return vm


# =============================================================================
# 1. StandardLibrary Container & Namespace Tests
# =============================================================================

def test_stdlib_container_structure():
    assert isinstance(std, StandardLibrary)
    assert std.math is s_math
    assert std.fs is s_fs
    assert std.crypto is s_crypto
    assert std.time is s_time
    assert std.sys is s_sys
    assert std.http is s_http
    assert std.regex is s_regex

    # Dict-like access
    assert std["math"] is s_math
    assert std["fs"] is s_fs
    assert std["crypto"] is s_crypto
    assert std["time"] is s_time
    assert std["sys"] is s_sys
    assert std["http"] is s_http
    assert std["regex"] is s_regex

    with pytest.raises(KeyError):
        _ = std["unknown_module"]

    assert "math" in std
    assert "crypto" in std
    assert "http" in std
    assert "regex" in std
    assert "unknown" not in std

    keys = std.keys()
    assert set(keys) == {"math", "fs", "crypto", "time", "sys", "http", "regex"}
    assert len(std) == 7
    assert list(iter(std)) == keys
    assert std.get("math") is s_math
    assert std.get("unknown", "default") == "default"

    d = std.to_dict()
    assert d["math"] is s_math
    assert "<StandardLibrary" in repr(std)


# =============================================================================
# 2. Math & Statistics Module Tests
# =============================================================================

def test_math_constants():
    assert s_math.PI == math.pi
    assert s_math.E == math.e
    assert s_math.INF == math.inf
    assert math.isnan(s_math.NAN)


def test_math_trigonometry():
    assert s_math.sin(0.0) == pytest.approx(0.0)
    assert s_math.sin(s_math.PI / 2) == pytest.approx(1.0)
    assert s_math.cos(0.0) == pytest.approx(1.0)
    assert s_math.cos(s_math.PI) == pytest.approx(-1.0)
    assert s_math.tan(0.0) == pytest.approx(0.0)
    assert s_math.asin(1.0) == pytest.approx(math.pi / 2)
    assert s_math.acos(1.0) == pytest.approx(0.0)
    assert s_math.atan(0.0) == pytest.approx(0.0)
    assert s_math.atan2(1.0, 1.0) == pytest.approx(math.pi / 4)


def test_math_hyperbolic_and_exponential():
    assert s_math.sinh(0.0) == pytest.approx(0.0)
    assert s_math.cosh(0.0) == pytest.approx(1.0)
    assert s_math.tanh(0.0) == pytest.approx(0.0)

    assert s_math.exp(0.0) == pytest.approx(1.0)
    assert s_math.exp(1.0) == pytest.approx(math.e)

    assert s_math.log(math.e) == pytest.approx(1.0)
    assert s_math.log(100.0, 10) == pytest.approx(2.0)
    assert s_math.log2(8.0) == pytest.approx(3.0)
    assert s_math.log10(1000.0) == pytest.approx(3.0)

    assert s_math.sqrt(25.0) == pytest.approx(5.0)
    assert s_math.pow(2.0, 3.0) == pytest.approx(8.0)


def test_math_statistics():
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # Mean: (2+4+4+4+5+5+7+9)/8 = 40/8 = 5.0
    assert s_math.mean(data) == pytest.approx(5.0)
    # Median: 4.5
    assert s_math.median(data) == pytest.approx(4.5)
    assert s_math.median([1, 2, 3]) == pytest.approx(2.0)

    # Variance: sample variance with n=8, sum of squared diffs:
    # (2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2 = 9 + 3 + 0 + 4 + 16 = 32
    # Sample variance: 32 / 7 = 4.57142857...
    # Population variance: 32 / 8 = 4.0
    assert s_math.variance(data, sample=True) == pytest.approx(32 / 7)
    assert s_math.variance(data, sample=False) == pytest.approx(4.0)

    # Std
    assert s_math.std(data, sample=True) == pytest.approx(math.sqrt(32 / 7))
    assert s_math.std(data, sample=False) == pytest.approx(2.0)

    # Quantile
    q_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert s_math.quantile(q_data, 0.0) == pytest.approx(1.0)
    assert s_math.quantile(q_data, 0.5) == pytest.approx(3.0)
    assert s_math.quantile(q_data, 1.0) == pytest.approx(5.0)
    assert s_math.quantile(q_data, 0.25) == pytest.approx(2.0)
    assert s_math.quantile(q_data, 0.75) == pytest.approx(4.0)

    # Min / Max
    assert s_math.min([3, 1, 4]) == 1
    assert s_math.min(3, 1, 4) == 1
    assert s_math.max([3, 1, 4]) == 4
    assert s_math.max(3, 1, 4) == 4

    # Clamp
    assert s_math.clamp(5, 0, 10) == 5
    assert s_math.clamp(-5, 0, 10) == 0
    assert s_math.clamp(15, 0, 10) == 10
    assert s_math.clamp(5, 10, 0) == 5  # inverted low/high

    # Lerp
    assert s_math.lerp(0.0, 100.0, 0.5) == pytest.approx(50.0)
    assert s_math.lerp(10.0, 20.0, 0.0) == pytest.approx(10.0)
    assert s_math.lerp(10.0, 20.0, 1.0) == pytest.approx(20.0)


def test_math_statistics_edge_cases():
    # Single element
    assert s_math.mean([42]) == 42.0
    assert s_math.median([42]) == 42.0
    assert s_math.variance([42]) == 0.0
    assert s_math.std([42]) == 0.0
    assert s_math.quantile([42], 0.5) == 42.0

    # Empty sequences raise ValueError
    with pytest.raises(ValueError):
        s_math.mean([])
    with pytest.raises(ValueError):
        s_math.median([])
    with pytest.raises(ValueError):
        s_math.variance([])
    with pytest.raises(ValueError):
        s_math.quantile([], 0.5)

    # Invalid quantile q
    with pytest.raises(ValueError):
        s_math.quantile([1, 2, 3], -0.1)
    with pytest.raises(ValueError):
        s_math.quantile([1, 2, 3], 1.1)


def test_math_with_tensor_input():
    t = tensor([10.0, 20.0, 30.0, 40.0])
    assert s_math.mean(t) == pytest.approx(25.0)
    assert s_math.median(t) == pytest.approx(25.0)
    assert s_math.min(t) == 10.0
    assert s_math.max(t) == 40.0


# =============================================================================
# 3. File System (fs) Module Tests
# =============================================================================

def test_fs_file_io(tmp_path):
    file_path = str(tmp_path / "sample.txt")

    # Initial state
    assert not s_fs.file_exists(file_path)

    # Write file
    chars = s_fs.write_file(file_path, "Hello, Synapse!\n")
    assert chars > 0
    assert s_fs.file_exists(file_path)

    # Read file
    content = s_fs.read_file(file_path)
    assert content == "Hello, Synapse!\n"

    # Append file
    s_fs.append_file(file_path, "Second line.")
    content2 = s_fs.read_file(file_path)
    assert content2 == "Hello, Synapse!\nSecond line."

    # Binary mode write and read
    bin_path = str(tmp_path / "binary.dat")
    bin_data = b"\x00\x01\x02\x03\xFF"
    s_fs.write_file(bin_path, bin_data, mode="wb")
    assert s_fs.file_exists(bin_path)
    read_bin = s_fs.read_file(bin_path, mode="rb")
    assert read_bin == bin_data


def test_fs_directory_and_path_operations(tmp_path):
    sub_dir = str(tmp_path / "nested" / "sub")
    assert not s_fs.dir_exists(sub_dir)

    # mkdir recursive
    s_fs.mkdir(sub_dir, recursive=True)
    assert s_fs.dir_exists(sub_dir)

    # create files inside
    f1 = s_fs.join_path(sub_dir, "file1.txt")
    f2 = s_fs.join_path(sub_dir, "file2.json")
    s_fs.write_file(f1, "file1 content")
    s_fs.write_file(f2, '{"key": "value"}')

    # list_dir
    entries = s_fs.list_dir(sub_dir)
    assert "file1.txt" in entries
    assert "file2.json" in entries

    # file_size
    assert s_fs.file_size(f1) == len("file1 content")

    # base_name, dir_name, ext_name
    assert s_fs.base_name(f1) == "file1.txt"
    assert s_fs.ext_name(f1) == ".txt"
    assert s_fs.ext_name(f2) == ".json"
    assert s_fs.dir_name(f1) == sub_dir

    # walk
    walk_res = s_fs.walk(str(tmp_path))
    assert len(walk_res) >= 2

    # remove_file
    s_fs.remove_file(f1)
    assert not s_fs.file_exists(f1)

    # rmdir recursive
    nested_root = str(tmp_path / "nested")
    s_fs.rmdir(nested_root, recursive=True)
    assert not s_fs.dir_exists(nested_root)


# =============================================================================
# 4. Cryptography (crypto) Module Tests
# =============================================================================

def test_crypto_hashes():
    data = "synapse-ai-native"
    h256 = s_crypto.sha256(data)
    assert len(h256) == 64
    assert h256 == s_crypto.sha256(data.encode("utf-8"))

    h512 = s_crypto.sha512(data)
    assert len(h512) == 128

    h_md5 = s_crypto.md5(data)
    assert len(h_md5) == 32


def test_crypto_hmac():
    key = "secret-key"
    msg = "authenticated-message"

    mac256 = s_crypto.hmac(key, msg, algo="sha256")
    assert len(mac256) == 64

    mac512 = s_crypto.hmac(key, msg, algo="sha512")
    assert len(mac512) == 128

    mac_md5 = s_crypto.hmac(key, msg, algo="md5")
    assert len(mac_md5) == 32

    with pytest.raises(ValueError):
        s_crypto.hmac(key, msg, algo="non_existent_algo")


def test_crypto_base64():
    raw_text = "Standard Library for Synapse"
    encoded = s_crypto.base64_encode(raw_text)
    assert isinstance(encoded, str)
    decoded = s_crypto.base64_decode(encoded)
    assert decoded == raw_text

    # Binary mode
    binary_data = b"\xde\xad\xbe\xef"
    b_encoded = s_crypto.base64_encode(binary_data)
    b_decoded = s_crypto.base64_decode(b_encoded, as_str=False)
    assert b_decoded == binary_data


def test_crypto_security():
    token1 = s_crypto.random_token(16)
    token2 = s_crypto.random_token(16)
    assert len(token1) == 32  # 16 bytes = 32 hex chars
    assert len(token2) == 32
    assert token1 != token2

    token_32 = s_crypto.random_token(32)
    assert len(token_32) == 64

    uid1 = s_crypto.uuid4()
    uid2 = s_crypto.uuid4()
    assert len(uid1) == 36
    assert uid1 != uid2
    assert uid1.count("-") == 4


# =============================================================================
# 5. Time & Sys Module Tests
# =============================================================================

def test_time_module():
    t_start = s_time.now()
    assert isinstance(t_start, float)
    assert t_start > 0

    iso_str = s_time.now_iso()
    assert isinstance(iso_str, str)
    assert "T" in iso_str

    parsed_ts = s_time.parse_iso(iso_str)
    assert abs(parsed_ts - t_start) < 2.0

    p1 = s_time.perf_counter()
    s_time.sleep(0.01)
    p2 = s_time.perf_counter()
    assert p2 > p1

    formatted = s_time.format_time(t_start, "%Y")
    assert len(formatted) == 4
    default_formatted = s_time.format_time(t_start)
    assert len(default_formatted) == 19  # YYYY-MM-DD HH:MM:SS


def test_sys_module():
    # Environment variables
    test_key = "SYNAPSE_TEST_ENV_VAR"
    s_sys.env_set(test_key, "synapse_val_123")
    assert s_sys.env_get(test_key) == "synapse_val_123"
    assert s_sys.env_get("NON_EXISTENT_VAR", "fallback") == "fallback"

    # Platform & CPU count
    plat = s_sys.platform()
    assert plat == sys.platform
    assert s_sys.cpu_count() >= 1

    # Exit
    with pytest.raises(SystemExit) as exc_info:
        s_sys.exit(0)
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info2:
        s_sys.exit(42)
    assert exc_info2.value.code == 42


# =============================================================================
# 6. Synapse VM Script Execution with std & shortcuts Tests
# =============================================================================

def test_vm_std_math_execution():
    source = """
let s = std.math.sin(0.0)
let sq = std.math.sqrt(16.0)
let cl = std.math.clamp(25.0, 0.0, 10.0)
let lp = std.math.lerp(10.0, 30.0, 0.5)
let pi_val = std.math.PI
"""
    vm = run_source(source)
    assert vm.globals["s"] == pytest.approx(0.0)
    assert vm.globals["sq"] == pytest.approx(4.0)
    assert vm.globals["cl"] == pytest.approx(10.0)
    assert vm.globals["lp"] == pytest.approx(20.0)
    assert vm.globals["pi_val"] == pytest.approx(math.pi)


def test_vm_std_fs_execution(tmp_path):
    test_file = str(tmp_path / "vm_test.txt").replace("\\", "/")
    source = f"""
let path = "{test_file}"
let written = std.fs.write_file(path, "Synapse VM FS Integration")
let exists_before = std.fs.file_exists(path)
let content = std.fs.read_file(path)
"""
    vm = run_source(source)
    assert vm.globals["exists_before"] is True
    assert vm.globals["content"] == "Synapse VM FS Integration"


def test_vm_std_crypto_execution():
    source = """
let hash = std.crypto.sha256("synapse-stdlib")
let token = std.crypto.random_token(16)
let uid = std.crypto.uuid4()
let enc = std.crypto.base64_encode("hello world")
let dec = std.crypto.base64_decode(enc)
"""
    vm = run_source(source)
    assert len(vm.globals["hash"]) == 64
    assert len(vm.globals["token"]) == 32
    assert len(vm.globals["uid"]) == 36
    assert vm.globals["dec"] == "hello world"


def test_vm_std_time_and_sys_execution():
    source = """
let t = std.time.now()
let plat = std.sys.platform()
let cpus = std.sys.cpu_count()
"""
    vm = run_source(source)
    assert vm.globals["t"] > 0
    assert vm.globals["plat"] == sys.platform
    assert vm.globals["cpus"] >= 1


def test_vm_global_shortcuts_execution(tmp_path):
    test_file = str(tmp_path / "shortcut_test.txt").replace("\\", "/")
    source = f"""
let p = "{test_file}"
write_file(p, "shortcut content")
let ex = file_exists(p)
let text = read_file(p)
let current_time = now()
let h = sha256("shortcut")
let cl = clamp(100, 0, 50)
let lp = lerp(0.0, 50.0, 0.2)
"""
    vm = run_source(source)
    assert vm.globals["ex"] is True
    assert vm.globals["text"] == "shortcut content"
    assert vm.globals["current_time"] > 0
    assert len(vm.globals["h"]) == 64
    assert vm.globals["cl"] == 50
    assert vm.globals["lp"] == pytest.approx(10.0)
