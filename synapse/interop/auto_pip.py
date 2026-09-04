"""Synapse JIT Auto-Pip Package Resolver & Isolated Environment Manager.

Provides automatic on-demand package resolution, isolated site-packages management (.synapse_env),
cross-process file locking for concurrency safety, and seamless sys.path injection.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("synapse.interop.auto_pip")


class FileLock:
    """Platform-independent atomic file lock using OS-level file creation flags.
    
    Protects .synapse_env from concurrent pip install race conditions across multiple
    threads, subagents, or processes.
    """

    def __init__(
        self,
        lock_path: Union[str, Path],
        timeout: float = 60.0,
        poll_interval: float = 0.05,
        stale_timeout: float = 180.0,
    ) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.stale_timeout = stale_timeout
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Acquires the lock atomically or waits until timeout."""
        start_time = time.time()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # O_CREAT | O_EXCL is atomic across Windows and POSIX
                self._fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                # Write current PID into lockfile for diagnostic / stale cleanup
                try:
                    os.write(self._fd, f"{os.getpid()}\n".encode("utf-8"))
                except OSError:
                    pass
                return True
            except FileExistsError:
                # Check stale lock condition
                try:
                    mtime = self.lock_path.stat().st_mtime
                    if time.time() - mtime > self.stale_timeout:
                        logger.warning(
                            "Detected stale lock file (%s, age > %s s). Breaking lock.",
                            self.lock_path,
                            self.stale_timeout,
                        )
                        try:
                            self.lock_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                except OSError:
                    pass

                if time.time() - start_time >= self.timeout:
                    raise TimeoutError(
                        f"Timed out after {self.timeout:.1f}s waiting for lock: {self.lock_path}"
                    )
                time.sleep(self.poll_interval)

    def release(self) -> None:
        """Releases the lock and cleans up the lock file."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        try:
            if self.lock_path.exists():
                self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class AutoPipManager:
    """Manages on-demand automatic package installation in an isolated Synapse environment.
    
    Default directory is `.synapse_env` in the project root, automatically added to `sys.path`.
    """

    def __init__(
        self,
        env_dir: Optional[Union[str, Path]] = None,
        timeout: float = 120.0,
    ) -> None:
        if env_dir is not None:
            self.env_dir = Path(env_dir).resolve()
        else:
            custom_env = os.environ.get("SYNAPSE_ENV_DIR")
            if custom_env:
                self.env_dir = Path(custom_env).resolve()
            else:
                # Default to project root .synapse_env
                # File location: synapse/interop/auto_pip.py -> parents[2] is project root
                project_root = Path(__file__).resolve().parents[2]
                self.env_dir = project_root / ".synapse_env"

        self.lock_file = self.env_dir / ".lock"
        self.timeout = timeout
        self._ensure_env_initialized()

    def _ensure_env_initialized(self) -> None:
        """Ensures that the target environment directory exists and is in sys.path."""
        self.env_dir.mkdir(parents=True, exist_ok=True)
        str_env = str(self.env_dir)
        if str_env not in sys.path:
            sys.path.insert(0, str_env)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strips 'py.' prefix, validates against command injection, and normalizes package name."""
        import re
        clean = name.strip()
        if clean.startswith("py."):
            clean = clean[3:]
        clean = clean.strip()
        # Security validation: strict regex to prevent command/argument injection
        if not clean or not re.match(r"^[a-zA-Z0-9_.-]+$", clean):
            raise ValueError(f"Invalid or unsafe package name: '{name}'")
        return clean

    def is_package_installed(self, package_name: str) -> bool:
        """Checks if a module or package can already be imported or exists in environment.
        
        Args:
            package_name: Package or module name (e.g. 'numpy', 'py.scipy', 'os.path')

        Returns:
            True if package is installed in the current Python environment or .synapse_env.
        """
        clean_name = self._normalize_name(package_name)
        root_name = clean_name.split(".")[0]

        # Ensure env_dir is on sys.path
        str_env = str(self.env_dir)
        if str_env not in sys.path and self.env_dir.exists():
            sys.path.insert(0, str_env)

        # 1. Already loaded in sys.modules
        if root_name in sys.modules:
            return True

        # 2. Can importlib find its specification?
        try:
            spec = importlib.util.find_spec(root_name)
            if spec is not None:
                return True
        except (ModuleNotFoundError, ValueError, AttributeError, ImportError):
            pass

        # 3. Direct inspection of .synapse_env directory
        if self.env_dir.exists():
            # Package directory with __init__.py or namespace package
            if (self.env_dir / root_name).is_dir():
                return True
            # Single-file module (e.g. root_name.py)
            if (self.env_dir / f"{root_name}.py").is_file():
                return True
            # Compiled extensions (.pyd on Windows, .so on POSIX)
            for ext in (".pyd", ".so"):
                if list(self.env_dir.glob(f"{root_name}*{ext}")):
                    return True
            # Egg-info or dist-info directory matching package
            for info_dir in self.env_dir.glob(f"{root_name.replace('-', '_')}*.dist-info"):
                if info_dir.is_dir():
                    return True

        return False

    def ensure_package(self, package_name: str, auto_install: bool = True) -> bool:
        """Ensures that the requested package is installed and importable.
        
        If already installed, immediately returns True without running pip.
        If missing and auto_install is True, installs via `pip install --target <env_dir> --quiet`
        under a concurrency file lock.

        Args:
            package_name: PyPI package or module name.
            auto_install: Whether to trigger automatic installation if missing.

        Returns:
            True if package is installed / successfully installed, False otherwise.
        """
        clean_name = self._normalize_name(package_name)

        # 1. Immediate check if already installed
        if self.is_package_installed(clean_name):
            return True

        # Check global environment toggle (e.g. SYNAPSE_AUTO_PIP=0)
        env_toggle = os.environ.get("SYNAPSE_AUTO_PIP", "1").strip().lower()
        if env_toggle in ("0", "false", "no", "off"):
            auto_install = False

        if not auto_install:
            return False

        # 2. Acquire lock for synchronized installation
        try:
            with FileLock(self.lock_file, timeout=self.timeout):
                # Double-checked locking pattern: check again if another process installed it
                if self.is_package_installed(clean_name):
                    return True

                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(self.env_dir),
                    "--quiet",
                    clean_name,
                ]
                logger.info(
                    "Auto-pip installing '%s' into %s: %s",
                    clean_name,
                    self.env_dir,
                    " ".join(cmd),
                )

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    shell=False,
                )

                if proc.returncode != 0:
                    logger.warning(
                        "Auto-pip install failed for '%s' (exit code %d): %s",
                        clean_name,
                        proc.returncode,
                        proc.stderr.strip() if proc.stderr else proc.stdout.strip(),
                    )
                    return False

                # Ensure env_dir is on sys.path
                str_env = str(self.env_dir)
                if str_env not in sys.path:
                    sys.path.insert(0, str_env)

                importlib.invalidate_caches()
                return True

        except Exception as exc:
            logger.error("Auto-pip error while installing '%s': %s", clean_name, exc)
            return False

    def clean_env(self) -> int:
        """Cleans all installed packages from .synapse_env.
        
        Returns:
            int: Number of items (files/directories) removed.
        """
        if not self.env_dir.exists():
            return 0

        removed_count = 0
        with FileLock(self.lock_file, timeout=self.timeout):
            for item in list(self.env_dir.iterdir()):
                if item.name == ".lock":
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    removed_count += 1
                except OSError as exc:
                    logger.warning("Failed to remove item %s during clean_env: %s", item, exc)

            importlib.invalidate_caches()

        return removed_count


__all__ = ["AutoPipManager", "FileLock"]
