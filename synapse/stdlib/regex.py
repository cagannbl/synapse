"""Synapse Built-in Standard Library: Native Regular Expressions Module.

Provides pure Python standard-library based regex matching, searching,
replacing, splitting, and pattern compilation helpers.
"""

from __future__ import annotations

import functools as _functools
import re as _re
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

# Flag Constants
IGNORECASE = I = _re.IGNORECASE
MULTILINE = M = _re.MULTILINE
DOTALL = S = _re.DOTALL
VERBOSE = X = _re.VERBOSE
ASCII = A = _re.ASCII


def _parse_flags(flags: Union[int, str]) -> int:
    """Normalizes flags given either as an integer bitmask or a shorthand string (e.g. 'i', 'ms')."""
    if isinstance(flags, int):
        return flags
    if isinstance(flags, str):
        flag_val = 0
        for char in flags.lower():
            if char == "i":
                flag_val |= _re.IGNORECASE
            elif char == "m":
                flag_val |= _re.MULTILINE
            elif char == "s":
                flag_val |= _re.DOTALL
            elif char == "x":
                flag_val |= _re.VERBOSE
            elif char == "a":
                flag_val |= _re.ASCII
        return flag_val
    return 0


@_functools.lru_cache(maxsize=512)
def _compile_cached(pattern: str, flags: int) -> Pattern[str]:
    """Compiles a regex pattern string into a Pattern object with caching."""
    return _re.compile(pattern, flags)


def _get_pattern(pattern: Union[str, Pattern[str]], flags: Union[int, str] = 0) -> Pattern[str]:
    """Resolves pattern argument into a compiled Pattern object."""
    flag_val = _parse_flags(flags)
    if isinstance(pattern, _re.Pattern):
        return pattern
    return _compile_cached(str(pattern), flag_val)


def _format_match(m: Optional[_re.Match[str]]) -> Optional[Dict[str, Any]]:
    """Formats a regex Match object into a dictionary containing match, groups, start, and end."""
    if m is None:
        return None
    return {
        "match": m.group(0),
        "groups": list(m.groups()),
        "start": m.start(),
        "end": m.end(),
        "named_groups": m.groupdict(),
    }


def match(
    pattern: Union[str, Pattern[str]],
    string: str,
    flags: Union[int, str] = 0,
) -> Optional[Dict[str, Any]]:
    """Matches pattern at the start of string.

    Returns a dict with 'match', 'groups', 'start', 'end' or None if no match.
    """
    s = str(string)
    compiled = _get_pattern(pattern, flags)
    m = compiled.match(s)
    return _format_match(m)


def search(
    pattern: Union[str, Pattern[str]],
    string: str,
    flags: Union[int, str] = 0,
) -> Optional[Dict[str, Any]]:
    """Scans through string looking for the first location where pattern produces a match.

    Returns a dict with 'match', 'groups', 'start', 'end' or None if no match.
    """
    s = str(string)
    compiled = _get_pattern(pattern, flags)
    m = compiled.search(s)
    return _format_match(m)


def find_all(
    pattern: Union[str, Pattern[str]],
    string: str,
    flags: Union[int, str] = 0,
) -> List[str]:
    """Finds all non-overlapping matches of pattern in string.

    Returns a list of matched strings.
    """
    s = str(string)
    compiled = _get_pattern(pattern, flags)
    return [m.group(0) for m in compiled.finditer(s)]


def replace(
    pattern: Union[str, Pattern[str]],
    repl: Union[str, Any],
    string: str,
    count: int = 0,
    flags: Union[int, str] = 0,
) -> str:
    """Replaces occurrences of pattern in string with repl."""
    s = str(string)
    compiled = _get_pattern(pattern, flags)
    return compiled.sub(repl, s, count=count)


def split(
    pattern: Union[str, Pattern[str]],
    string: str,
    maxsplit: int = 0,
    flags: Union[int, str] = 0,
) -> List[str]:
    """Splits string by the occurrences of pattern."""
    s = str(string)
    compiled = _get_pattern(pattern, flags)
    return compiled.split(s, maxsplit=maxsplit)


def is_match(
    pattern: Union[str, Pattern[str]],
    string: str,
    flags: Union[int, str] = 0,
) -> bool:
    """Returns True if pattern produces a match anywhere in string, False otherwise."""
    return search(pattern, string, flags=flags) is not None


def escape(pattern: str) -> str:
    """Escapes special characters in pattern string."""
    return _re.escape(str(pattern))


def compile(pattern: str, flags: Union[int, str] = 0) -> Pattern[str]:
    """Compiles pattern string into a regular expression object."""
    return _get_pattern(pattern, flags)


__all__ = [
    "match",
    "search",
    "find_all",
    "replace",
    "split",
    "is_match",
    "escape",
    "compile",
    "IGNORECASE",
    "I",
    "MULTILINE",
    "M",
    "DOTALL",
    "S",
    "VERBOSE",
    "X",
    "ASCII",
    "A",
]
