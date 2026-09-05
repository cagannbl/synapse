"""
Synapse Core Scope Discipline, Package Decoupling, and Regression Verification Test
===================================================================================
Verifies:
1. Core compiler (Lexer, Parser, TypeChecker, ShapeGuard, CEmitter, VM) runs
   flawlessly with ZERO dependency on external packages ('synapse-orm', 'synapse-web').
2. C runtime arena memory lifecycle declarations ('syn_arena_scope_enter', 'syn_arena_scope_leave').
3. CUDA detection engine and safe CPU fallback execution under CUDABackend.
4. Absence of hardcoded external package imports across the core compiler codebase.
"""

import os
import sys
import numpy as np
import pytest

from synapse.core.device import Device
from synapse.core.cuda_backend import is_cuda_available, device_count, get_device_name, CUDABackend
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.core.type_checker import check_source
from synapse.analyzer.shape_guard import TensorShapeGuard
from synapse.codegen.c_emitter import CEmitter
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def test_core_compiler_runs_without_external_packages(monkeypatch):
    """
    Test that when 'synapse-orm' and 'synapse-web' are strictly absent from sys.path
    and sys.modules, the core compiler (Lexer, Parser, TypeChecker, ShapeGuard,
    CEmitter, VM) executes smoothly with zero missing dependency errors.
    """
    # 1. Cleanse sys.path from any references to packages/synapse-orm or packages/synapse-web
    cleaned_path = [
        p for p in sys.path
        if "synapse-orm" not in p and "synapse-web" not in p and not p.endswith(os.path.join("packages", "synapse_orm"))
    ]
    monkeypatch.setattr(sys, "path", cleaned_path)

    # 2. Ensure neither external package is imported
    assert "synapse_orm" not in sys.modules
    assert "synapse_web" not in sys.modules

    # 3. Full Compiler Pipeline Test on an AI-Native Synapse Program
    syn_code = """let a = tensor([[1.0, 2.0], [3.0, 4.0]])
let b = tensor([[2.0, 0.0], [1.0, 3.0]])
let c = a @ b

fn compute(x: int) -> int:
    let sum = 0
    for i in range(x):
        sum = sum + i
    return sum

let result = compute(5)
"""

    # Lexer
    tokens = Lexer(syn_code).tokenize()
    assert len(tokens) > 0

    # Parser
    ast = Parser(tokens).parse()
    assert ast is not None
    assert len(ast.statements) > 0

    # TypeChecker
    tc_result = check_source(syn_code)
    assert tc_result.is_valid
    assert len(tc_result.errors) == 0

    # ShapeGuard
    guard = TensorShapeGuard(source_text=syn_code, filename="core_test.syn")
    guard_errors = guard.verify_source()
    assert len(guard_errors) == 0

    # CEmitter
    emitter = CEmitter(include_line_directives=False, current_filename="core_test.syn")
    c_code = emitter.emit(ast)
    assert "syn_tensor_t*" in c_code or "syn_matmul" in c_code or "double" in c_code

    # Virtual Machine execution
    bytecode = Compiler(name="core_test").compile(ast)
    vm = VirtualMachine()
    vm.execute(bytecode)

    # Verify runtime state
    assert vm.globals["result"] == 10
    c_val = vm.globals["c"]
    assert c_val is not None
    expected_c = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[2.0, 0.0], [1.0, 3.0]])
    np.testing.assert_allclose(c_val.data, expected_c)


def test_core_codebase_has_no_external_package_imports():
    """
    Static audit: verify that no files inside 'synapse/' import or depend on
    'synapse_orm', 'synapse_web', or 'packages'.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    core_dir = os.path.join(repo_root, "synapse")

    forbidden_patterns = [
        "import synapse_orm",
        "from synapse_orm",
        "import synapse_web",
        "from synapse_web",
        "from packages",
        "import packages",
    ]

    violations = []
    for root, _, files in os.walk(core_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_idx, line in enumerate(f, start=1):
                        stripped = line.strip()
                        for pat in forbidden_patterns:
                            if pat in stripped and not stripped.startswith("#"):
                                violations.append(f"{os.path.relpath(filepath, repo_root)}:{line_idx} - {stripped}")

    assert violations == [], f"Found forbidden coupling to external packages in core:\n" + "\n".join(violations)


def test_arena_memory_lifecycle_c_declarations():
    """
    Audit C runtime source and header to verify that arena memory lifecycle
    primitives ('syn_arena_scope_enter', 'syn_arena_scope_leave') and arena
    allocator definitions exist and are fully declared.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    header_path = os.path.join(repo_root, "synapse", "runtime", "synapse_runtime.h")
    c_source_path = os.path.join(repo_root, "synapse", "runtime", "synapse_runtime.c")

    assert os.path.isfile(header_path), f"synapse_runtime.h not found at {header_path}"
    assert os.path.isfile(c_source_path), f"synapse_runtime.c not found at {c_source_path}"

    with open(header_path, "r", encoding="utf-8") as f:
        header_content = f.read()

    with open(c_source_path, "r", encoding="utf-8") as f:
        c_content = f.read()

    # Verify declaration in header
    assert "syn_arena_scope_enter" in header_content, "syn_arena_scope_enter missing from synapse_runtime.h"
    assert "syn_arena_scope_leave" in header_content, "syn_arena_scope_leave missing from synapse_runtime.h"
    assert "syn_arena_t" in header_content, "syn_arena_t missing from synapse_runtime.h"
    assert "syn_arena_scope_t" in header_content, "syn_arena_scope_t missing from synapse_runtime.h"
    assert "syn_arena_create" in header_content, "syn_arena_create missing from synapse_runtime.h"
    assert "syn_arena_alloc" in header_content, "syn_arena_alloc missing from synapse_runtime.h"
    assert "syn_arena_reset" in header_content, "syn_arena_reset missing from synapse_runtime.h"
    assert "syn_arena_free" in header_content, "syn_arena_free missing from synapse_runtime.h"

    # Verify implementation in C runtime source
    assert "syn_arena_scope_t syn_arena_scope_enter" in c_content, "syn_arena_scope_enter implementation missing from synapse_runtime.c"
    assert "void syn_arena_scope_leave" in c_content, "syn_arena_scope_leave implementation missing from synapse_runtime.c"
    assert "syn_arena_t* syn_arena_create" in c_content, "syn_arena_create implementation missing from synapse_runtime.c"
    assert "void* syn_arena_alloc" in c_content, "syn_arena_alloc implementation missing from synapse_runtime.c"
    assert "void syn_arena_reset" in c_content, "syn_arena_reset implementation missing from synapse_runtime.c"
    assert "void syn_arena_free" in c_content, "syn_arena_free implementation missing from synapse_runtime.c"


def test_cuda_detection_and_safe_cpu_fallback():
    """
    Test that CUDA device detection functions return valid types and values,
    and CUDABackend gracefully falls back to CPU tensor operations without crashing
    when a CUDA device target is requested.
    """
    # 1. Detection Engine Invariants
    avail = is_cuda_available()
    assert isinstance(avail, bool)

    count = device_count()
    assert isinstance(count, int)
    assert count >= 0

    name = get_device_name(0)
    assert isinstance(name, str)
    assert len(name) > 0

    # 2. CUDABackend API Invariants
    assert CUDABackend.is_available() == avail
    assert CUDABackend.device_count() == count
    assert CUDABackend.get_device_name(0) == name

    # 3. Graceful Fallback Execution with CUDA Device Target
    cuda_dev = Device("cuda:0")
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

    # Matmul fallback
    res_matmul = CUDABackend.matmul(a, b, device=cuda_dev)
    expected_matmul = a @ b
    np.testing.assert_allclose(res_matmul, expected_matmul)

    # Addition fallback
    res_add = CUDABackend.add(a, b, device=cuda_dev)
    expected_add = a + b
    np.testing.assert_allclose(res_add, expected_add)

    # Subtraction fallback
    res_sub = CUDABackend.sub(a, b, device=cuda_dev)
    expected_sub = a - b
    np.testing.assert_allclose(res_sub, expected_sub)


def test_arena_scope_c_emitter_loop_generation():
    """
    Verify that CEmitter generates arena scopes in high-iteration loops
    for bounded memory overhead.
    """
    syn_code = """fn loop_test():
    for i in range(100):
        let t = zeros([10, 10])
"""
    tokens = Lexer(syn_code).tokenize()
    ast = Parser(tokens).parse()
    emitter = CEmitter(include_line_directives=False)
    c_code = emitter.emit(ast)

    assert "syn_arena_scope_enter" in c_code or "syn_arena" in c_code
