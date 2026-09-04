import pytest
import os
import subprocess
import tempfile
from synapse.codegen.native_compiler import NativeCompiler


def test_thread_local_storage_definition_in_runtime_c():
    runtime_c = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime", "synapse_runtime.c"))
    with open(runtime_c, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SYN_THREAD_LOCAL" in content
    assert "static SYN_THREAD_LOCAL syn_arena_t* g_syn_active_arena = NULL;" in content
    assert "_Thread_local" in content or "__declspec(thread)" in content or "__thread" in content


def test_concurrent_threads_isolated_arenas_execution():
    """
    Spawns multiple threads allocating tensors in their own thread-local active arenas
    and verifies there is no data race or memory corruption across threads.
    """
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler found on system")

    compiler_path, compiler_type = compiler_info
    runtime_dir = nc.runtime_dir
    runtime_c = nc.runtime_c

    test_c_source = f"""
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "synapse_runtime.h"

#define NUM_THREADS 4
#define ITERS 500

void thread_worker(void* arg) {{
    int thread_id = *(int*)arg;
    syn_arena_t* arena = syn_arena_create(2 * 1024 * 1024);
    assert(arena != NULL);
    syn_arena_set_active(arena);

    /* Thread-local active arena must match this thread's arena */
    assert(syn_arena_get_active() == arena);

    for (int i = 0; i < ITERS; i++) {{
        syn_arena_scope_t scope = syn_arena_scope_enter(arena);
        int shape[2] = {{8, 8}};
        syn_tensor_t* a = syn_tensor_zeros(shape, 2, 0);
        syn_tensor_t* b = syn_tensor_ones(shape, 2, 0);
        syn_tensor_t* c = syn_add(a, b);
        assert(c != NULL);
        assert(syn_tensor_item(c) == 1.0);
        syn_arena_scope_leave(scope);
    }}

    syn_arena_set_active(NULL);
    syn_arena_free(arena);
}}

int main(void) {{
    syn_task_handle_t* handles[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {{
        thread_ids[i] = i;
        handles[i] = syn_spawn(thread_worker, &thread_ids[i]);
        assert(handles[i] != NULL);
    }}

    for (int i = 0; i < NUM_THREADS; i++) {{
        syn_task_wait(handles[i]);
        syn_task_handle_free(handles[i]);
    }}

    printf("THREAD_SAFE_ARENA_SUCCESS\\n");
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "test_threads.c")
        exe_file = os.path.join(tmpdir, "test_threads.exe")

        with open(src_file, "w", encoding="utf-8") as f:
            f.write(test_c_source)

        if compiler_type == "zig":
            cmd = [compiler_path, "cc", src_file, runtime_c, f"-I{runtime_dir}", "-O2", "-o", exe_file]
        elif compiler_type in ("gcc", "clang"):
            cmd = [compiler_path, src_file, runtime_c, f"-I{runtime_dir}", "-O2", "-pthread", "-o", exe_file]
        elif compiler_type == "cl":
            cmd = [compiler_path, src_file, runtime_c, f"/I{runtime_dir}", "/O2", f"/Fe:{exe_file}"]
        else:
            pytest.skip(f"Unsupported compiler {compiler_type}")

        comp_res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}\n{comp_res.stdout}"

        run_res = subprocess.run([exe_file], capture_output=True, text=True, timeout=10)
        assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}\n{run_res.stdout}"
        assert "THREAD_SAFE_ARENA_SUCCESS" in run_res.stdout
