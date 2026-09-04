"""Synapse SafeTensors Implementation (Hugging Face SafeTensors Specification).

Pure Python implementation with zero external dependencies (standard library + NumPy/Synapse).
Supports zero-copy memory mapping, strict schema validation, 8-byte buffer alignment,
metadata preservation, and full Hugging Face API compatibility (load_file, save_file, safe_open).

Specification:
1. First 8 bytes: header_size (64-bit unsigned integer, little-endian, struct.unpack('<Q', ...))
2. Next header_size bytes: UTF-8 encoded JSON string containing tensor metadata and optional __metadata__.
3. Remainder: Raw binary tensor buffers contiguous or aligned.
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from synapse.core.device import Device
from synapse.core.tensor import Tensor

logger = logging.getLogger("synapse.nn.safetensors")

# Security threshold: prevent memory exhaustion from maliciously malformed header sizes
MAX_HEADER_SIZE = 100 * 1024 * 1024  # 100 MB

# =============================================================================
# SafeTensors Standard DType Mappings
# =============================================================================
DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "F64": np.dtype("float64"),
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "I64": np.dtype("int64"),
    "I32": np.dtype("int32"),
    "I16": np.dtype("int16"),
    "I8": np.dtype("int8"),
    "U64": np.dtype("uint64"),
    "U32": np.dtype("uint32"),
    "U16": np.dtype("uint16"),
    "U8": np.dtype("uint8"),
    "BOOL": np.dtype("bool"),
}

# Optional bfloat16 support
try:
    import ml_dtypes  # type: ignore
    DTYPE_TO_NUMPY["BF16"] = np.dtype(ml_dtypes.bfloat16)
except ImportError:
    DTYPE_TO_NUMPY["BF16"] = np.dtype("uint16")


NUMPY_NAME_TO_DTYPE: dict[str, str] = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint64": "U64",
    "uint32": "U32",
    "uint16": "U16",
    "uint8": "U8",
    "bool": "BOOL",
    "bool_": "BOOL",
}


def numpy_dtype_to_safetensors(dtype: Union[np.dtype, str]) -> str:
    """Converts a NumPy dtype or type name to SafeTensors dtype identifier."""
    dt = np.dtype(dtype)
    if dt.name in NUMPY_NAME_TO_DTYPE:
        return NUMPY_NAME_TO_DTYPE[dt.name]

    # Kind and itemsize fallback for cross-platform integer/float variants
    kind_map: dict[tuple[str, int], str] = {
        ("f", 8): "F64",
        ("f", 4): "F32",
        ("f", 2): "F16",
        ("i", 8): "I64",
        ("i", 4): "I32",
        ("i", 2): "I16",
        ("i", 1): "I8",
        ("u", 8): "U64",
        ("u", 4): "U32",
        ("u", 2): "U16",
        ("u", 1): "U8",
        ("b", 1): "BOOL",
    }
    key = (dt.kind, dt.itemsize)
    if key in kind_map:
        return kind_map[key]

    raise ValueError(f"Unsupported NumPy dtype for SafeTensors: {dtype} (name={dt.name}, kind={dt.kind}, size={dt.itemsize})")


# =============================================================================
# SafeTensors Container Dict
# =============================================================================
class SafeTensorsDict(dict):
    """Dictionary holding loaded Synapse Tensors with SafeTensors metadata
    and lifetime management for memory-mapped buffers.
    """

    def __init__(self, *args: Any, metadata: Optional[dict[str, str]] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.metadata: dict[str, str] = dict(metadata) if metadata else {}
        self._mmap_ref: Optional[mmap.mmap] = None
        self._file_ref: Optional[Any] = None

    def get_metadata(self) -> dict[str, str]:
        """Returns the SafeTensors metadata dictionary."""
        return self.metadata

    def __getitem__(self, key: str) -> Any:
        if key == "__metadata__":
            return self.metadata
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "__metadata__":
            return self.metadata
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key == "__metadata__":
            return True
        return super().__contains__(key)

    def close(self) -> None:
        """Closes memory map and file handles if open."""
        if self._mmap_ref is not None:
            try:
                self._mmap_ref.close()
            except Exception:
                pass
            self._mmap_ref = None
        if self._file_ref is not None:
            try:
                self._file_ref.close()
            except Exception:
                pass
            self._file_ref = None

    def __enter__(self) -> "SafeTensorsDict":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# =============================================================================
# Serialization
# =============================================================================
def serialize(
    tensors: dict[str, Union[Tensor, np.ndarray, Any]],
    metadata: Optional[dict[str, str]] = None,
) -> bytes:
    """Serializes a dictionary of tensors to in-memory SafeTensors binary bytes.

    Args:
        tensors: Dictionary mapping tensor names to Synapse Tensor or NumPy arrays.
        metadata: Optional dictionary mapping string keys to string values.

    Returns:
        SafeTensors formatted bytes.
    """
    header: dict[str, Any] = {}
    combined_metadata: dict[str, str] = {}

    if metadata:
        combined_metadata.update({str(k): str(v) for k, v in metadata.items()})

    # Prepare raw tensor buffers and offsets
    tensor_buffers: list[bytes] = []
    current_offset = 0

    for name, item in tensors.items():
        if name == "__metadata__":
            if isinstance(item, dict):
                combined_metadata.update({str(k): str(v) for k, v in item.items()})
            continue

        if isinstance(item, Tensor):
            arr = item.data
        elif isinstance(item, np.ndarray):
            arr = item
        elif isinstance(item, (int, float, list, tuple)):
            arr = np.array(item)
        else:
            raise TypeError(f"Unsupported value type for tensor '{name}': {type(item)}")

        # Ensure contiguous array while preserving 0-D scalar shape
        if arr.ndim > 0:
            arr = np.ascontiguousarray(arr)
        dtype_str = numpy_dtype_to_safetensors(arr.dtype)
        raw_bytes = arr.tobytes()
        nbytes = len(raw_bytes)

        begin = current_offset
        end = current_offset + nbytes
        current_offset = end

        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [begin, end],
        }
        tensor_buffers.append(raw_bytes)

    if combined_metadata:
        header["__metadata__"] = combined_metadata

    # Convert header to JSON bytes
    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    header_len = len(header_bytes)

    # Align data start to 8-byte boundary:
    # Buffer begins at offset (8 + header_len).
    # We pad the header with spaces so that (8 + header_len) % 8 == 0.
    pad_size = (8 - ((8 + header_len) % 8)) % 8
    if pad_size > 0:
        header_bytes += b" " * pad_size
        header_len = len(header_bytes)

    # 8-byte header_size uint64 little-endian
    header_prefix = struct.pack("<Q", header_len)

    # Concatenate all parts
    result = bytearray()
    result.extend(header_prefix)
    result.extend(header_bytes)
    for buf in tensor_buffers:
        result.extend(buf)

    return bytes(result)


def save_safetensors(
    tensors: dict[str, Union[Tensor, np.ndarray, Any]],
    filepath: Optional[Union[str, Path]] = None,
    metadata: Optional[dict[str, str]] = None,
    *,
    filename: Optional[Union[str, Path]] = None,
) -> None:
    """Saves a dictionary of tensors to a SafeTensors binary file.

    Format:
    - Bytes 0-7: 64-bit unsigned integer (header_size in bytes, little-endian).
    - Bytes 8 to (8 + header_size): UTF-8 JSON header padded for 8-byte alignment.
    - Remaining bytes: Raw binary tensor data buffer.

    Args:
        tensors: Dictionary mapping tensor names to Synapse Tensor or NumPy arrays.
        filepath: Destination file path.
        metadata: Optional string-to-string dictionary stored in JSON '__metadata__'.
        filename: Alias for filepath (Hugging Face API compatibility).
    """
    target_path = filepath if filepath is not None else filename
    if target_path is None:
        raise ValueError("Either 'filepath' or 'filename' must be provided to save_safetensors.")

    path_obj = Path(target_path)
    if path_obj.parent and not path_obj.parent.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    data = serialize(tensors, metadata=metadata)
    with open(path_obj, "wb") as f:
        f.write(data)


# =============================================================================
# Deserialization
# =============================================================================
def deserialize(
    data: Union[bytes, bytearray, memoryview],
    device: Union[Device, str] = "cpu",
    copy: bool = False,
) -> SafeTensorsDict:
    """Deserializes tensors from an in-memory SafeTensors byte buffer.

    Args:
        data: SafeTensors binary byte buffer.
        device: Target device for loaded Tensors (e.g., 'cpu', 'cuda').
        copy: If True, makes private copies of tensor buffers instead of zero-copy views.

    Returns:
        SafeTensorsDict mapping tensor names to Synapse Tensor objects.
    """
    total_size = len(data)
    if total_size < 8:
        raise ValueError(f"Invalid SafeTensors data: buffer size ({total_size} bytes) is smaller than 8 bytes.")

    header_size = struct.unpack("<Q", data[:8])[0]
    if header_size <= 0:
        raise ValueError(f"Corrupted SafeTensors header: header size is {header_size}")
    if header_size > MAX_HEADER_SIZE:
        raise ValueError(f"SafeTensors header size {header_size} exceeds maximum allowable size of {MAX_HEADER_SIZE} bytes.")
    if 8 + header_size > total_size:
        raise ValueError(f"Corrupted SafeTensors data: header size ({header_size}) exceeds buffer length ({total_size}).")

    raw_header = data[8 : 8 + header_size]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse SafeTensors JSON header: {e}") from e

    metadata: dict[str, str] = {}
    if "__metadata__" in header:
        meta_val = header.pop("__metadata__")
        if isinstance(meta_val, dict):
            metadata = {str(k): str(v) for k, v in meta_val.items()}

    buffer_start = 8 + header_size
    data_buffer_len = total_size - buffer_start
    result = SafeTensorsDict(metadata=metadata)

    for name, tensor_info in header.items():
        if not isinstance(tensor_info, dict):
            raise ValueError(f"Invalid tensor metadata entry for '{name}': {tensor_info}")

        if "dtype" not in tensor_info or "shape" not in tensor_info or "data_offsets" not in tensor_info:
            raise ValueError(f"Missing required SafeTensors keys in entry '{name}'. Required: dtype, shape, data_offsets.")

        dtype_str = tensor_info["dtype"]
        if dtype_str not in DTYPE_TO_NUMPY:
            raise ValueError(f"Unsupported SafeTensors dtype '{dtype_str}' for tensor '{name}'.")
        np_dtype = DTYPE_TO_NUMPY[dtype_str]

        shape = tuple(tensor_info["shape"])
        offsets = tensor_info["data_offsets"]
        if not (isinstance(offsets, (list, tuple)) and len(offsets) == 2):
            raise ValueError(f"Invalid data_offsets for '{name}': {offsets}")

        begin, end = offsets
        if not (0 <= begin <= end <= data_buffer_len):
            raise ValueError(f"Invalid tensor offsets for '{name}': [{begin}, {end}] in buffer of size {data_buffer_len}.")

        itemsize = np_dtype.itemsize
        expected_elements = math.prod(shape) if shape else 1
        expected_bytes = expected_elements * itemsize
        actual_bytes = end - begin
        if expected_bytes != actual_bytes:
            raise ValueError(
                f"Tensor '{name}' byte size mismatch: shape {shape} with dtype {dtype_str} "
                f"requires {expected_bytes} bytes, but data_offsets specify {actual_bytes} bytes."
            )

        # Zero-copy buffer slice
        tensor_arr = np.frombuffer(
            data,
            dtype=np_dtype,
            count=expected_elements,
            offset=buffer_start + begin,
        ).reshape(shape)

        if copy:
            tensor_arr = tensor_arr.copy()

        t = Tensor(tensor_arr, dtype=np_dtype, device=device)
        t.data = tensor_arr  # Guarantee zero-copy view alignment
        result[name] = t

    return result


def load_safetensors(
    filepath: Optional[Union[str, Path]] = None,
    device: Union[Device, str] = "cpu",
    use_mmap: bool = False,
    copy: bool = False,
    *,
    filename: Optional[Union[str, Path]] = None,
) -> SafeTensorsDict:
    """Loads a SafeTensors binary file into Synapse Tensors with zero-copy memory mapping.

    Args:
        filepath: Path to the .safetensors file.
        device: Target device for loaded Tensors ('cpu', 'cuda', etc.).
        use_mmap: If True, uses mmap.mmap for true zero-copy OS memory mapping.
                  Falls back gracefully to binary reading if mmap is unavailable.
        copy: If True, copies tensor arrays into heap memory (making them writeable).
        filename: Alias for filepath (Hugging Face API compatibility).

    Returns:
        SafeTensorsDict containing tensor names mapped to Synapse Tensor objects,
        with metadata accessible via `.metadata` or `["__metadata__"]`.
    """
    target_path = filepath if filepath is not None else filename
    if target_path is None:
        raise ValueError("Either 'filepath' or 'filename' must be provided to load_safetensors.")

    path_obj = Path(target_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SafeTensors file not found: {target_path}")

    file_size = path_obj.stat().st_size
    if file_size < 8:
        raise ValueError(f"Invalid SafeTensors file: size ({file_size} bytes) is smaller than 8 bytes.")

    f = open(path_obj, "rb")
    mm: Optional[mmap.mmap] = None
    buffer: Union[mmap.mmap, bytes]

    if use_mmap:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            buffer = mm
        except Exception as e:
            logger.debug(f"mmap failed on '{target_path}', falling back to binary read: {e}")
            buffer = f.read()
    else:
        buffer = f.read()

    try:
        result = deserialize(buffer, device=device, copy=copy)
    except Exception:
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
        f.close()
        raise

    if mm is not None:
        result._mmap_ref = mm
        result._file_ref = f
    else:
        f.close()

    return result


def load_metadata(filepath: Union[str, Path]) -> dict[str, str]:
    """Reads only the metadata header from a SafeTensors file without reading tensor data."""
    path_obj = Path(filepath)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path_obj, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"Invalid SafeTensors file: size is less than 8 bytes.")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > MAX_HEADER_SIZE:
            raise ValueError(f"Invalid header size: {header_size}")
        raw_header = f.read(header_size)
        if len(raw_header) < header_size:
            raise ValueError("Corrupted SafeTensors file: unexpected EOF while reading header.")
        header = json.loads(raw_header.decode("utf-8"))
        return header.get("__metadata__", {})


# =============================================================================
# Hugging Face SafeTensors API Compatibility Aliases
# =============================================================================
save_file = save_safetensors
load_file = load_safetensors


# =============================================================================
# Safe Open Context Manager (HF Compatible)
# =============================================================================
class safe_open:
    """Context manager for streaming, inspecting, or slicing SafeTensors on-demand.
    Compatible with Hugging Face's `safetensors.safe_open`.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        framework: str = "synapse",
        device: Union[Device, str] = "cpu",
    ):
        self.filename = str(filename)
        self.framework = framework
        self.device = device
        self._file: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._buffer: Optional[Union[bytes, mmap.mmap]] = None
        self._header: dict[str, Any] = {}
        self._metadata: dict[str, str] = {}
        self._buffer_start: int = 0

    def __enter__(self) -> "safe_open":
        with open(self.filename, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            if file_size < 8:
                raise ValueError(f"Invalid SafeTensors file: size {file_size} is less than 8 bytes.")
            self._buffer = f.read()

        header_size = struct.unpack("<Q", self._buffer[:8])[0]
        if header_size <= 0 or 8 + header_size > file_size:
            raise ValueError(f"Corrupted header size {header_size} in SafeTensors file.")
        if header_size > MAX_HEADER_SIZE:
            raise ValueError(f"Header size {header_size} exceeds maximum allowable {MAX_HEADER_SIZE}.")

        raw_header = self._buffer[8 : 8 + header_size]
        self._header = json.loads(raw_header.decode("utf-8"))
        if "__metadata__" in self._header:
            meta = self._header.pop("__metadata__")
            if isinstance(meta, dict):
                self._metadata = {str(k): str(v) for k, v in meta.items()}

        self._buffer_start = 8 + header_size
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._buffer = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def keys(self) -> list[str]:
        """Returns list of tensor names in the file."""
        return list(self._header.keys())

    def metadata(self) -> dict[str, str]:
        """Returns the SafeTensors metadata dictionary."""
        return self._metadata

    def get_tensor(self, name: str) -> Tensor:
        """Loads and returns a specific tensor by name."""
        if name not in self._header:
            raise KeyError(f"Tensor '{name}' not found in SafeTensors file '{self.filename}'.")

        info = self._header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        begin, end = info["data_offsets"]
        np_dtype = DTYPE_TO_NUMPY[dtype_str]

        expected_elements = math.prod(shape) if shape else 1
        tensor_arr = np.frombuffer(
            self._buffer,
            dtype=np_dtype,
            count=expected_elements,
            offset=self._buffer_start + begin,
        ).reshape(shape)

        # Make a copy so tensor remains valid after safe_open closes
        arr_copy = tensor_arr.copy()
        t = Tensor(arr_copy, dtype=np_dtype, device=self.device)
        return t

    def get_slice(self, name: str) -> np.ndarray:
        """Returns the raw numpy slice for the given tensor name."""
        if name not in self._header:
            raise KeyError(f"Tensor '{name}' not found in SafeTensors file '{self.filename}'.")

        info = self._header[name]
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        begin, end = info["data_offsets"]
        np_dtype = DTYPE_TO_NUMPY[dtype_str]

        expected_elements = math.prod(shape) if shape else 1
        return np.frombuffer(
            self._buffer,
            dtype=np_dtype,
            count=expected_elements,
            offset=self._buffer_start + begin,
        ).reshape(shape)
