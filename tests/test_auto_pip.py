"""Tests for Synapse JIT Auto-Pip Package Resolver & Isolated Environment Manager."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synapse.interop.auto_pip import AutoPipManager, FileLock
from synapse.interop.eco_bridge import PackageNotFoundError, TransparentPyResolver
from synapse.interop.python_bridge import PythonModuleWrapper
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_synapse_code(code: str) -> VirtualMachine:
    """Helper to compile and execute Synapse source code in the VM."""
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    compiled = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(compiled)
    return vm


# =========================================================================
# 1. Package Installation Detection Tests
# =========================================================================
def test_is_package_installed_builtin_modules(tmp_path):
    """Verify built-in standard library packages are detected without pip."""
    mgr = AutoPipManager(env_dir=tmp_path / "env")
    assert mgr.is_package_installed("math") is True
    assert mgr.is_package_installed("json") is True
    assert mgr.is_package_installed("sys") is True
    assert mgr.is_package_installed("os.path") is True
    assert mgr.is_package_installed("py.math") is True


def test_is_package_installed_non_existent(tmp_path):
    """Verify non-existent packages return False."""
    mgr = AutoPipManager(env_dir=tmp_path / "env")
    assert mgr.is_package_installed("non_existent_fake_package_xyz123") is False


# =========================================================================
# 2. ensure_package Tests (No Pip When Already Installed)
# =========================================================================
def test_ensure_package_already_installed_no_pip(tmp_path):
    """Verify that ensure_package does not invoke pip if module is already installed."""
    mgr = AutoPipManager(env_dir=tmp_path / "env")
    with patch("subprocess.run") as mock_run:
        result = mgr.ensure_package("math")
        assert result is True
        mock_run.assert_not_called()


def test_ensure_package_already_installed_with_py_prefix(tmp_path):
    """Verify that 'py.' prefixed existing modules bypass pip execution."""
    mgr = AutoPipManager(env_dir=tmp_path / "env")
    with patch("subprocess.run") as mock_run:
        result = mgr.ensure_package("py.json")
        assert result is True
        mock_run.assert_not_called()


# =========================================================================
# 3. ensure_package with Mock Subprocess (Pip Invocation & sys.path)
# =========================================================================
def test_ensure_package_success_with_mock_pip(tmp_path):
    """Verify pip install arguments, sys.path injection, and cache invalidation on missing package."""
    env_dir = tmp_path / "env"
    mgr = AutoPipManager(env_dir=env_dir)

    pkg_name = "custom_demo_scientific_lib"
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "Successfully installed custom_demo_scientific_lib"
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc) as mock_run, \
         patch("importlib.invalidate_caches") as mock_invalidate:
        result = mgr.ensure_package(pkg_name, auto_install=True)

        assert result is True
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args

        # Expected command: [sys.executable, "-m", "pip", "install", "--target", str(env_dir), "--quiet", pkg_name]
        cmd = args[0]
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["-m", "pip", "install"]
        assert "--target" in cmd
        assert str(env_dir) in cmd
        assert "--quiet" in cmd
        assert pkg_name in cmd

        # Verify sys.path injection
        assert str(env_dir) in sys.path
        assert sys.path[0] == str(env_dir)

        # Verify cache invalidation
        mock_invalidate.assert_called_once()


def test_ensure_package_auto_install_false(tmp_path):
    """Verify ensure_package returns False and skips pip when auto_install=False."""
    mgr = AutoPipManager(env_dir=tmp_path / "env")
    with patch("subprocess.run") as mock_run:
        result = mgr.ensure_package("uninstalled_lib_abc", auto_install=False)
        assert result is False
        mock_run.assert_not_called()


def test_ensure_package_pip_failure_returns_false(tmp_path):
    """Verify ensure_package returns False gracefully when pip install fails."""
    mgr = AutoPipManager(env_dir=tmp_path / "env")
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "ERROR: No matching distribution found for uninstalled_lib_abc"

    with patch("subprocess.run", return_value=mock_proc):
        result = mgr.ensure_package("uninstalled_lib_abc", auto_install=True)
        assert result is False


# =========================================================================
# 4. FileLock Concurrency and Mutual Exclusion Tests
# =========================================================================
def test_file_lock_mutual_exclusion_and_release(tmp_path):
    """Verify atomic file locking, timeout prevention, and clean release."""
    lock_file = tmp_path / ".lock"

    # 1. Acquire first lock
    with FileLock(lock_file, timeout=1.0) as lock1:
        assert lock_file.exists()

        # 2. Competing lock attempt should raise TimeoutError
        competing_lock = FileLock(lock_file, timeout=0.15, poll_interval=0.03)
        with pytest.raises(TimeoutError):
            competing_lock.acquire()

    # 3. Once released, lock file is removed and new lock can be acquired
    assert not lock_file.exists()
    with FileLock(lock_file, timeout=0.5):
        assert lock_file.exists()

    assert not lock_file.exists()


def test_clean_env_removes_files_and_preserves_dir(tmp_path):
    """Verify clean_env empties the environment directory and returns removed count."""
    env_dir = tmp_path / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    mgr = AutoPipManager(env_dir=env_dir)

    # Create dummy files and directories
    (env_dir / "module_a.py").write_text("# dummy a")
    (env_dir / "module_b.py").write_text("# dummy b")
    sub_pkg = env_dir / "pkg_c"
    sub_pkg.mkdir()
    (sub_pkg / "__init__.py").write_text("# dummy c")

    assert (env_dir / "module_a.py").exists()
    assert (env_dir / "pkg_c").exists()

    removed = mgr.clean_env()
    assert removed == 3
    assert env_dir.exists()
    assert list(env_dir.iterdir()) == []


# =========================================================================
# 5. TransparentPyResolver JIT Auto-Pip Integration Tests
# =========================================================================
def test_transparent_py_resolver_jit_auto_pip_success():
    """Verify TransparentPyResolver automatically triggers JIT install and returns wrapper."""
    fake_mod_name = "test_jit_mock_package_xyz"
    fake_mod = types.ModuleType(fake_mod_name)
    fake_mod.greet = lambda name: f"Hello, {name} from JIT!"

    def mock_ensure(pkg_name, auto_install=True):
        sys.modules[fake_mod_name] = fake_mod
        return True

    try:
        with patch.object(AutoPipManager, "ensure_package", side_effect=mock_ensure) as mock_ensure_pkg:
            wrapper = TransparentPyResolver.resolve_import(f"py.{fake_mod_name}")
            assert isinstance(wrapper, PythonModuleWrapper)
            assert wrapper.greet("Synapse") == "Hello, Synapse from JIT!"
            mock_ensure_pkg.assert_called_once_with(fake_mod_name, auto_install=True)
    finally:
        sys.modules.pop(fake_mod_name, None)


def test_transparent_py_resolver_jit_failure_raises_package_not_found():
    """Verify TransparentPyResolver raises PackageNotFoundError if auto-pip fails."""
    with patch.object(AutoPipManager, "ensure_package", return_value=False):
        with pytest.raises(PackageNotFoundError) as exc_info:
            TransparentPyResolver.resolve_import("totally_broken_pkg_12345")

        err = exc_info.value
        assert "totally_broken_pkg_12345" in str(err)
        assert "Run 'pip install totally_broken_pkg_12345'" in str(err)


def test_transparent_py_resolver_auto_install_disabled():
    """Verify auto_install=False prevents pip install attempt and raises immediately."""
    with patch.object(AutoPipManager, "ensure_package") as mock_ensure:
        with pytest.raises(PackageNotFoundError):
            TransparentPyResolver.resolve_import("uninstalled_test_package_999", auto_install=False)

        mock_ensure.assert_not_called()


# =========================================================================
# 6. End-to-End VM Execution with JIT Auto-Pip
# =========================================================================
def test_vm_execution_with_jit_auto_pip():
    """Verify Synapse VM imports dynamically resolved JIT package and executes methods."""
    fake_pkg = "jit_vm_compute_lib"
    fake_mod = types.ModuleType(fake_pkg)
    fake_mod.calculate = lambda a, b: a * 10 + b

    def mock_ensure(pkg_name, auto_install=True):
        sys.modules[fake_pkg] = fake_mod
        return True

    source = f"""
import py.{fake_pkg} as comp
let res = comp.calculate(5, 7)
"""
    try:
        with patch.object(AutoPipManager, "ensure_package", side_effect=mock_ensure):
            vm = run_synapse_code(source)
            assert vm.globals["res"] == 57
    finally:
        sys.modules.pop(fake_pkg, None)
