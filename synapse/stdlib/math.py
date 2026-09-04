"""Synapse Built-in Standard Library: Math Module."""

import math as _math
from typing import Any, Sequence, Union

# =============================================================================
# Mathematical Constants
# =============================================================================

PI: float = _math.pi
E: float = _math.e
INF: float = _math.inf
NAN: float = _math.nan

# =============================================================================
# Trigonometric Functions
# =============================================================================

def sin(x: float) -> float:
    """Calculates sine of x in radians."""
    return _math.sin(x)


def cos(x: float) -> float:
    """Calculates cosine of x in radians."""
    return _math.cos(x)


def tan(x: float) -> float:
    """Calculates tangent of x in radians."""
    return _math.tan(x)


def asin(x: float) -> float:
    """Calculates arc sine of x in radians."""
    return _math.asin(x)


def acos(x: float) -> float:
    """Calculates arc cosine of x in radians."""
    return _math.acos(x)


def atan(x: float) -> float:
    """Calculates arc tangent of x in radians."""
    return _math.atan(x)


def atan2(y: float, x: float) -> float:
    """Calculates arc tangent of y / x in radians taking into account quadrant."""
    return _math.atan2(y, x)


# =============================================================================
# Hyperbolic & Exponential Functions
# =============================================================================

def sinh(x: float) -> float:
    """Calculates hyperbolic sine of x."""
    return _math.sinh(x)


def cosh(x: float) -> float:
    """Calculates hyperbolic cosine of x."""
    return _math.cosh(x)


def tanh(x: float) -> float:
    """Calculates hyperbolic tangent of x."""
    return _math.tanh(x)


def exp(x: float) -> float:
    """Calculates e raised to power x."""
    return _math.exp(x)


def log(x: float, base: Union[float, int, None] = None) -> float:
    """Calculates logarithm of x. Defaults to natural logarithm (base e)."""
    if base is None:
        return _math.log(x)
    return _math.log(x, base)


def log2(x: float) -> float:
    """Calculates base-2 logarithm of x."""
    return _math.log2(x)


def log10(x: float) -> float:
    """Calculates base-10 logarithm of x."""
    return _math.log10(x)


def sqrt(x: float) -> float:
    """Calculates square root of x."""
    return _math.sqrt(x)


def pow(x: float, y: float) -> float:
    """Calculates x raised to power y."""
    return _math.pow(x, y)


# =============================================================================
# Analytics & Statistics Functions
# =============================================================================

def _to_sequence(seq: Any) -> list[float]:
    """Helper to convert tensors, lists, tuples, or iterables into a flat float list."""
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    if isinstance(seq, (int, float)):
        return [float(seq)]
    flat: list[float] = []

    def _flatten(item: Any):
        if isinstance(item, (list, tuple)):
            for sub in item:
                _flatten(sub)
        else:
            flat.append(float(item))

    try:
        _flatten(seq)
        return flat
    except Exception:
        return [float(x) for x in seq]


def mean(seq: Any) -> float:
    """Calculates the arithmetic mean of a sequence."""
    data = _to_sequence(seq)
    if not data:
        raise ValueError("mean requires at least one data point")
    return sum(data) / len(data)


def median(seq: Any) -> float:
    """Calculates the median value of a sequence."""
    data = sorted(_to_sequence(seq))
    n = len(data)
    if n == 0:
        raise ValueError("median requires at least one data point")
    mid = n // 2
    if n % 2 == 1:
        return float(data[mid])
    return float((data[mid - 1] + data[mid]) / 2.0)


def variance(seq: Any, sample: bool = True) -> float:
    """Calculates variance of a sequence. Uses sample variance (n-1) if sample is True."""
    data = _to_sequence(seq)
    n = len(data)
    if n == 0:
        raise ValueError("variance requires at least one data point")
    if n == 1:
        return 0.0
    m = sum(data) / n
    ss = sum((x - m) ** 2 for x in data)
    divisor = (n - 1) if sample else n
    return float(ss / divisor)


def std(seq: Any, sample: bool = True) -> float:
    """Calculates standard deviation of a sequence."""
    return float(_math.sqrt(variance(seq, sample=sample)))


def quantile(seq: Any, q: float) -> float:
    """Calculates the q-th quantile (0.0 <= q <= 1.0) with linear interpolation."""
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be between 0.0 and 1.0")
    data = sorted(_to_sequence(seq))
    n = len(data)
    if n == 0:
        raise ValueError("quantile requires at least one data point")
    if n == 1:
        return float(data[0])
    pos = q * (n - 1)
    base = int(pos)
    rest = pos - base
    if base + 1 < n:
        return float(data[base] + rest * (data[base + 1] - data[base]))
    return float(data[base])


def min(seq: Any, *args: Any) -> Any:
    """Returns minimum element from a sequence or arguments."""
    import builtins
    if args:
        return builtins.min(seq, *args)
    if hasattr(seq, "tolist"):
        seq = _to_sequence(seq)
    return builtins.min(seq)


def max(seq: Any, *args: Any) -> Any:
    """Returns maximum element from a sequence or arguments."""
    import builtins
    if args:
        return builtins.max(seq, *args)
    if hasattr(seq, "tolist"):
        seq = _to_sequence(seq)
    return builtins.max(seq)


def clamp(val: Any, low: Any, high: Any) -> Any:
    """Clamps a value between low and high limits."""
    if low > high:
        low, high = high, low
    if val < low:
        return low
    if val > high:
        return high
    return val


def lerp(a: float, b: float, t: float) -> float:
    """Computes linear interpolation between a and b at ratio t."""
    return float(a + (b - a) * t)


__all__ = [
    "PI", "E", "INF", "NAN",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "log2", "log10", "sqrt", "pow",
    "mean", "median", "std", "variance", "quantile", "min", "max",
    "clamp", "lerp",
]
