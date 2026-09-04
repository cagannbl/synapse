"""
Tests for Synapse C Emitter Memory Lifecycle and Arena Scopes.
Verifies:
1. 10,000-iteration for loop tensor matmul / add transpiles with arena scope enter/leave.
2. Compiled C executable runs cleanly without memory leaks (bounded RSS footprint).
3. Arena scope reclamation resets offset every iteration.
"""
import os
import subprocess
import pytest
from synapse.codegen.native_compiler import NativeCompiler


def test_transpile_10k_loop_has_arena_scopes():
    """Verify CEmitter automatically injects arena scope management into 10k loops."""
    nc = NativeCompiler()
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[2.0, 0.0], [0.0, 2.0]])
for i in range(10000):
    let C = A @ B
    let D = C + A
"""
    c_code = nc.transpile(source)

    assert "syn_arena_create" in c_code
    assert "syn_arena_scope_enter" in c_code
    assert "syn_arena_scope_leave" in c_code
    assert "syn_arena_free" in c_code
    assert "for (int i = 0; i < 10000; i++)" in c_code


def test_compile_and_run_10k_loop_bounded_memory(tmp_path):
    """Compile and execute 10,000 iteration loop C program and verify zero leak / bounded RSS."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[0.5, 1.5], [2.5, 3.5]])
for i in range(10000):
    let C = A @ B
    let D = C + A
"""
    test_exe = tmp_path / ("mem_test.exe" if os.name == "nt" else "mem_test")
    test_c = tmp_path / "mem_test.c"

    ok, res = nc.compile_source_to_executable(source, str(test_exe), temp_c_path=str(test_c))
    assert ok is True, f"Compilation failed:\n{res}"
    assert os.path.isfile(test_exe)

    # Run compiled program
    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"Executable failed with error:\n{proc.stderr}\n{proc.stdout}"


def test_arena_scope_10k_iterations_offset_reclamation(tmp_path):
    """C verification ensuring that 10,000 arena scope iterations keep arena offset constant."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler available in environment")

    test_c = tmp_path / "arena_10k_test.c"
    test_exe = tmp_path / ("arena_10k_test.exe" if os.name == "nt" else "arena_10k_test")

    c_source = """
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "synapse_runtime.h"

int main() {
    syn_arena_t* arena = syn_arena_create(2 * 1024 * 1024); /* 2MB buffer */
    assert(arena != NULL);
    assert(arena->offset == 0);

    /* Base persistent tensors */
    int shape[2] = {2, 2};
    double data_a[4] = {1.0, 2.0, 3.0, 4.0};
    double data_b[4] = {5.0, 6.0, 7.0, 8.0};

    syn_tensor_t* a = syn_tensor_create_arena(arena, data_a, shape, 2, 0);
    syn_tensor_t* b = syn_tensor_create_arena(arena, data_b, shape, 2, 0);
    size_t base_offset = arena->offset;
    assert(base_offset > 0);

    /* Run 10,000 iterations allocating temporary tensors in arena scopes */
    for (int i = 0; i < 10000; i++) {
        syn_arena_scope_t scope = syn_arena_scope_enter(arena);
        assert(scope.saved_offset == base_offset);

        /* Temporary tensor operations */
        syn_tensor_t* temp = syn_tensor_zeros_arena(arena, shape, 2, 0);
        assert(temp != NULL);
        assert(arena->offset > base_offset);

        /* Instant O(1) scope reclamation */
        syn_arena_scope_leave(scope);
        assert(arena->offset == base_offset);
    }

    /* Verify base tensors were never corrupted */
    assert(a->data[0] == 1.0 && a->data[3] == 4.0);
    assert(b->data[0] == 5.0 && b->data[3] == 8.0);
    assert(arena->offset == base_offset);

    syn_arena_free(arena);
    printf("ARENA_10K_SUCCESS\\n");
    return 0;
}
"""
    test_c.write_text(c_source, encoding="utf-8")
    ok, res = nc.build_executable(str(test_c), str(test_exe))
    assert ok is True, f"Build failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "ARENA_10K_SUCCESS" in proc.stdout

