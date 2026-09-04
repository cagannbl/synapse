"""Synapse Built-in Standard Library: Time & Date Module."""

import datetime as _datetime
import time as _time
from typing import Optional, Union

# =============================================================================
# Time & Clock Functions
# =============================================================================

def now() -> float:
    """Returns the current Unix timestamp in seconds as a floating-point number."""
    return _time.time()


def now_iso() -> str:
    """Returns the current local date and time formatted according to ISO 8601."""
    return _datetime.datetime.now().isoformat()


def sleep(seconds: float) -> None:
    """Suspends execution of the current thread for the given number of seconds."""
    _time.sleep(seconds)


def perf_counter() -> float:
    """Returns the current value of a fractional second performance counter for benchmarking."""
    return _time.perf_counter()


def format_time(timestamp: Optional[Union[float, int]] = None, fmt: Optional[str] = None) -> str:
    """Formats a Unix timestamp into a human-readable string. Defaults to '%Y-%m-%d %H:%M:%S'."""
    ts = _time.time() if timestamp is None else float(timestamp)
    pattern = fmt or "%Y-%m-%d %H:%M:%S"
    return _time.strftime(pattern, _time.localtime(ts))


def parse_iso(iso_str: str) -> float:
    """Parses an ISO 8601 formatted datetime string and returns its Unix timestamp (seconds)."""
    dt = _datetime.datetime.fromisoformat(iso_str)
    return dt.timestamp()


__all__ = [
    "now",
    "now_iso",
    "sleep",
    "perf_counter",
    "format_time",
    "parse_iso",
]
