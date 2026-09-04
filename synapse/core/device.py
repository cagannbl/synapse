from __future__ import annotations
from typing import Union, Optional


class Device:
    """Represents a computation device (CPU, CUDA, MPS).
    
    Supports string parsing such as 'cpu', 'cuda', 'cuda:0', 'mps', etc.
    """
    def __init__(self, device: Union[Device, str] = "cpu", index: Optional[int] = None):
        if isinstance(device, Device):
            self.device_type: str = device.device_type
            self.index: int = index if index is not None else device.index
        elif isinstance(device, str):
            dev_str = device.strip().lower()
            if ":" in dev_str:
                parts = dev_str.split(":", 1)
                self.device_type = parts[0]
                try:
                    self.index = int(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid device index in '{device}'")
            else:
                self.device_type = dev_str
                self.index = index if index is not None else 0
        else:
            raise TypeError(f"Expected Device or str, got {type(device).__name__}")

        if self.device_type not in ("cpu", "cuda", "mps"):
            # Allow custom but normalize
            pass

    @property
    def type(self) -> str:
        """Alias for device_type for PyTorch-like API compatibility."""
        return self.device_type

    def __str__(self) -> str:
        if self.device_type == "cuda":
            return f"cuda:{self.index}"
        return self.device_type

    def __repr__(self) -> str:
        return f"device(type='{self.device_type}', index={self.index})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Device):
            return self.device_type == other.device_type and self.index == other.index
        if isinstance(other, str):
            try:
                other_dev = Device(other)
                if ":" not in other and other == "cuda" and self.device_type == "cuda":
                    return True
                return self == other_dev
            except Exception:
                return False
        return False

    def __hash__(self) -> int:
        return hash((self.device_type, self.index))


def cpu() -> Device:
    """Helper function returning CPU device."""
    return Device("cpu", 0)


def cuda(index: int = 0) -> Device:
    """Helper function returning CUDA device with given index."""
    return Device("cuda", index)
