"""
Unit tests for Synapse C Runtime Arena Scope and Memory Management.
Verifies syn_arena_scope_t, syn_arena_scope_enter, syn_arena_scope_leave,
and idempotent / double-free safe syn_tensor_free.
"""
import os
import subprocess
import pytest
from synapse.codegen.native_compiler import NativeCompiler


def test_runtime_header_has_arena_scope_api():
    """Verify synapse_runtime.h declares syn_arena_scope_t and scope functions."""
    runtime_h = os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime", "synapse_runtime.h")
    with open(runtime_h, "r", encoding="utf-8") as f:
        content = f.read()

    assert "typedef struct syn_arena_scope {" in content
    assert "syn_arena_t* arena;" in content
    assert "size_t saved_offset;" in content
    assert "} syn_arena_scope_t;" in content
    assert "syn_arena_scope_t syn_arena_scope_enter(syn_arena_t* arena);" in content
    assert "void syn_arena_scope_leave(syn_arena_scope_t scope);" in content


def test_runtime_c_has_arena_scope_implementation():
    """Verify synapse_runtime.c implements syn_arena_scope_enter, syn_arena_scope_leave, and safe syn_tensor_free."""
    runtime_c = os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime", "synapse_runtime.c")
    with open(runtime_c, "r", encoding="utf-8") as f:
        content = f.read()

    assert "syn_arena_scope_t syn_arena_scope_enter(syn_arena_t* arena)" in content
    assert "void syn_arena_scope_leave(syn_arena_scope_t scope)" in content
    assert "scope.saved_offset" in content
    # Verify idempotent / safe syn_tensor_free
    assert "if (!t || !t->data) return;" in content


def test_arena_scope_execution(tmp_path):
    """Compile and execute C program verifying O(1) scope reclamation and nested arena scopes."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler found in environment")

    test_c = tmp_path / "test_arena_scope.c"
    test_exe = tmp_path / ("test_arena_scope.exe" if os.name == "nt" else "test_arena_scope")

    c_source = """
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "synapse_runtime.h"

int main() {
    /* 1. NULL arena safety */
    syn_arena_scope_t null_scope = syn_arena_scope_enter(NULL);
    assert(null_scope.arena == NULL);
    assert(null_scope.saved_offset == 0);
    syn_arena_scope_leave(null_scope);

    /* 2. Create Arena */
    syn_arena_t* arena = syn_arena_create(1024 * 1024); /* 1MB */
    assert(arena != NULL);
    assert(arena->offset == 0);

    /* 3. Outer Scope (scope1) */
    syn_arena_scope_t scope1 = syn_arena_scope_enter(arena);
    assert(scope1.arena == arena);
    assert(scope1.saved_offset == 0);

    int shape1[2] = {2, 3};
    double data1[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    syn_tensor_t* t1 = syn_tensor_create_arena(arena, data1, shape1, 2, 0);
    assert(t1 != NULL);
    assert(t1->size == 6);
    assert(t1->data[0] == 1.0 && t1->data[5] == 6.0);
    size_t offset_after_t1 = arena->offset;
    assert(offset_after_t1 > 0);

    /* 4. Nested Scope (scope2) - Temporary loop tensor simulation */
    {
        syn_arena_scope_t scope2 = syn_arena_scope_enter(arena);
        assert(scope2.arena == arena);
        assert(scope2.saved_offset == offset_after_t1);

        int shape2[1] = {256};
        syn_tensor_t* t_temp = syn_tensor_zeros_arena(arena, shape2, 1, 0);
        assert(t_temp != NULL);
        assert(t_temp->size == 256);
        assert(arena->offset > offset_after_t1);

        /* Deeply Nested Scope (scope3) */
        {
            syn_arena_scope_t scope3 = syn_arena_scope_enter(arena);
            size_t offset_before_s3 = arena->offset;
            int shape3[1] = {64};
            syn_tensor_t* t_inner = syn_tensor_zeros_arena(arena, shape3, 1, 0);
            assert(t_inner != NULL);
            assert(arena->offset > offset_before_s3);
            syn_arena_scope_leave(scope3);
            assert(arena->offset == offset_before_s3);
        }

        /* Leave nested scope2: O(1) instant memory reclamation */
        syn_arena_scope_leave(scope2);
        assert(arena->offset == offset_after_t1);
    }

    /* 5. Memory Reuse in outer scope after nested leave */
    int shape_reuse[2] = {2, 3};
    double data_reuse[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    syn_tensor_t* t_reuse = syn_tensor_create_arena(arena, data_reuse, shape_reuse, 2, 0);
    assert(t_reuse != NULL);
    /* t1 data must remain unmodified */
    assert(t1->data[0] == 1.0 && t1->data[5] == 6.0);
    assert(t_reuse->data[0] == 10.0 && t_reuse->data[5] == 60.0);

    /* Leave outer scope1: completely resets offset to 0 */
    syn_arena_scope_leave(scope1);
    assert(arena->offset == 0);

    syn_arena_free(arena);

    /* 6. Safe Tensor Free verification */
    syn_tensor_free(NULL);

    syn_tensor_t dummy;
    dummy.data = NULL;
    dummy.shape = NULL;
    dummy.grad = NULL;
    syn_tensor_free(&dummy); /* Must not crash when t->data is NULL */

    /* Standard heap tensor free */
    int shape_heap[1] = {4};
    double data_heap[4] = {7.0, 8.0, 9.0, 10.0};
    syn_tensor_t* heap_t = syn_tensor_create(data_heap, shape_heap, 1, 0);
    assert(heap_t != NULL);
    assert(heap_t->data != NULL);
    syn_tensor_free(heap_t);

    return 0;
}
"""
    test_c.write_text(c_source, encoding="utf-8")

    ok, res = nc.build_executable(str(test_c), str(test_exe))
    assert ok is True, f"Compilation failed: {res}"
    assert os.path.isfile(test_exe)

    # Run compiled test
    run_res = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=10)
    assert run_res.returncode == 0, f"Runtime failure: {run_res.stderr}\n{run_res.stdout}"
