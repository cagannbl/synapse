"""Synapse Built-in Standard Library: System & Environment Module."""

import os as _os
import sys as _sys
from typing import Any, Optional

# =============================================================================
# Environment Variables
# =============================================================================

def env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieves an environment variable value, or returns default if not set."""
    return _os.environ.get(str(key), default)


def env_set(key: str, val: Any) -> None:
    """Sets an environment variable to the specified string value."""
    _os.environ[str(key)] = str(val)


# =============================================================================
# System Information & Process Control
# =============================================================================

def platform() -> str:
    """Returns the platform identifier string (e.g. 'win32', 'linux', 'darwin')."""
    return _sys.platform


def cpu_count() -> int:
    """Returns the number of logical CPUs available on the system."""
    return _os.cpu_count() or 1


def exit(code: int = 0) -> None:
    """Exits the interpreter with the specified status code."""
    _sys.exit(code)


__all__ = [
    "env_get",
    "env_set",
    "platform",
    "cpu_count",
    "exit",
]
