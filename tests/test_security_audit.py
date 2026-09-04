"""
Synapse AI Security Audit & Hardening Test Suite (OWASP / CWE / AppSec Standards).
Validates:
1. C99 Runtime Arithmetic Overflow Protection (syn_tensor_create, syn_arena_alloc).
2. Auto-Pip Command Injection Rejection (strict regex package name validation).
3. Playground Development Server Localhost Interface Enforcement (Anti-RCE).
4. Git Tracked Tree Sterility (Zero binaries, zero PDAs, zero secrets, zero personal paths).
5. Web Route Parameter Validation & Anti-Traversal Safety.
"""

import os
import re
import sys
import subprocess
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# 1. C99 Runtime Arithmetic Overflow & Boundary Safety Tests
# ==========================================================================

def test_c99_tensor_arithmetic_overflow_protection(tmp_path):
    """Verify that C runtime and core refuse to allocate tensors with overflowing or negative dimensions."""
    from synapse.core.tensor import zeros

    # Negative shape dimensions must be rejected
    with pytest.raises((ValueError, RuntimeError)):
        zeros([-4, 10])

    # Astronomical dimension multiplication causing size_t overflow
    with pytest.raises((ValueError, RuntimeError, OverflowError, MemoryError)):
        zeros([2**31 - 1, 2**31 - 1])

    # Direct C runtime validation
    from synapse.codegen.native_compiler import NativeCompiler
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        return

    test_c = tmp_path / "overflow_test.c"
    test_exe = tmp_path / ("overflow_test.exe" if os.name == "nt" else "overflow_test")
    c_source = """
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "synapse_runtime.h"

int main() {
    syn_arena_t* arena = syn_arena_create(1024 * 1024);
    assert(arena != NULL);

    int neg_shape[2] = {-4, 10};
    syn_tensor_t* t_neg = syn_tensor_create_arena(arena, NULL, neg_shape, 2, 0);
    assert(t_neg == NULL); /* Must reject negative dimension */

    int big_shape[2] = {1000000000, 1000000000};
    syn_tensor_t* t_overflow = syn_tensor_create_arena(arena, NULL, big_shape, 2, 0);
    assert(t_overflow == NULL); /* Must reject overflow */

    syn_arena_free(arena);
    printf("OVERFLOW_GUARDS_ACTIVE\\n");
    return 0;
}
"""
    test_c.write_text(c_source, encoding="utf-8")
    ok, res = nc.build_executable(str(test_c), str(test_exe))
    assert ok is True, f"Build failed:\n{res}"

    proc = subprocess.run([str(test_exe)], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0
    assert "OVERFLOW_GUARDS_ACTIVE" in proc.stdout


def test_c99_source_code_overflow_guards_presence():
    """Direct static inspection of synapse_runtime.c to guarantee overflow guards exist."""
    runtime_c = REPO_ROOT / "synapse" / "runtime" / "synapse_runtime.c"
    assert runtime_c.is_file(), "synapse_runtime.c must exist"
    content = runtime_c.read_text(encoding="utf-8")

    # Verify SIZE_MAX overflow check in syn_arena_alloc
    assert "bytes > SIZE_MAX - 7" in content
    assert "arena->offset > arena->capacity - aligned" in content

    # Verify arithmetic overflow checks in syn_tensor_create_arena and _alloc_tensor
    assert "total_elements > SIZE_MAX / (size_t)shape[i]" in content
    assert "total_elements > SIZE_MAX / sizeof(double)" in content


# ===========================================================================
# 2. Auto-Pip Command Injection & Input Hardening Tests
# ==========================================================================
def test_auto_pip_command_injection_rejection():
    """Verify that auto_pip rejects shell metacharacters and injected command flags."""
    from synapse.interop.auto_pip import AutoPipManager

    manager = AutoPipManager()
    malicious_inputs = [
        "numpy; rm -rf /",
        "package & echo evil",
        "pkg | whoami",
        "pkg`id`",
        "pkg$(calc.exe)",
        "pkg --extra-index-url https://malicious.org",
        "pkg-name && touch /tmp/pwned",
        "name with spaces",
        "",
        "   ",
    ]

    for evil in malicious_inputs:
        with pytest.raises(ValueError, match="Invalid or unsafe package name"):
            manager._normalize_name(evil)


def test_auto_pip_valid_package_names_accepted():
    """Verify standard PyPI package names pass validation cleanly."""
    from synapse.interop.auto_pip import AutoPipManager

    manager = AutoPipManager()
    valid_names = [
        "numpy",
        "scipy",
        "scikit-learn",
        "torch_vision",
        "py.pillow",
        "urllib3",
    ]
    for valid in valid_names:
        norm = manager._normalize_name(valid)
        assert norm
        assert ";" not in norm


# ==========================================================================
# 3. Playground Server Localhost Interface Enforcement (Anti-RCE)
# ==========================================================================

def test_playground_server_restricts_to_localhost():
    """Verify that playground server refuses 0.0.0.0 or unsafe external interface binding."""
    from playground.server import create_server

    # Explicit 0.0.0.0 must be safely overridden to 127.0.0.1
    server, serve_dir = create_server(host="0.0.0.0", port=3099)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()

    # Default host must be strictly 127.0.0.1
    server_default, _ = create_server(port=3098)
    try:
        assert server_default.server_address[0] == "127.0.0.1"
    finally:
        server_default.server_close()


# ==========================================================================
# 4. Git Tracked Tree Sterility & Secret Leak Prevention
# ==========================================================================

def test_git_tracked_files_zero_binaries_and_caches():
    """Verify that git index tracks 100% clean source files without binaries or caches."""
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0
    files = res.stdout.splitlines()
    assert len(files) > 100

    disallowed_exts = (
        ".exe", ".pdb", ".dll", ".lib", ".obj", ".o", ".ilk", ".exp",
        ".so", ".dylib", ".vsix", ".pyc", ".pyo", ".pyd", ".arrow", ".safetensors"
    )
    disallowed_dirs = ("__pycache__", ".pytest_cache", ".syn_cache", ".synapse_env")

    for f in files:
        f_lower = f.lower()
        assert not f_lower.endswith(disallowed_exts), f"Binary or debug file tracked in git: {f}"
        assert not any(d in f for d in disallowed_dirs), f"Cache directory tracked in git: {f}"


def test_git_tracked_files_zero_hardcoded_paths_and_secrets():
    """Verify that no personal usernames or real API secrets exist in tracked files."""
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0
    files = res.stdout.splitlines()

    secret_pattern = re.compile(r"(?i)ghp_[a-zA-Z0-9]{36}")
    path_pattern = re.compile(r"(?i)[a-z]:[/\\]users[/\\]")

    for rel_path in files:
        file_path = REPO_ROOT / rel_path
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        assert not secret_pattern.search(content), f"Potential GitHub token leak in {rel_path}"
        assert not path_pattern.search(content), f"Hardcoded Windows user path found in {rel_path}"


# ===========================================================================
# 5. Web Route Parameter Anti-Traversal & Validation
# ==========================================================================
def test_web_route_parameter_type_safety():
    """Verify that parameterized routes safely reject malformed types and path traversal."""
    from synapse.web.enterprise import SynapseApp

    app = SynapseApp()

    @app.get("/users/{id:int}")
    def get_user(req, id: int):
        return {"id": id}

    # Valid integer request
    res_ok = app.handle_request("GET", "/users/42")
    assert res_ok.status_code == 200

    # Path traversal / non-integer attack attempts must be rejected with 400 Bad Request
    res_traversal = app.handle_request("GET", "/users/../etc/passwd")
    assert res_traversal.status_code in (400, 404)

    res_invalid_int = app.handle_request("GET", "/users/abc")
    assert res_invalid_int.status_code in (400, 404)
