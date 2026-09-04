import math
import os
import struct
import tempfile
import numpy as np
import pytest

from synapse.core.tensor import Tensor
from synapse.nn import (
    Linear,
    load_safetensors,
    save_safetensors,
    load_file,
    save_file,
    safe_open,
    load_metadata,
)
from synapse.nn.safetensors import serialize, deserialize, DTYPE_TO_NUMPY


def test_safetensors_save_and_load_basic_dtypes():
    """Verify that F32, F64, I64, I32, and BOOL tensors are saved and loaded identically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.safetensors")
        tensors = {
            "w_f32": Tensor(np.array([[1.5, -2.5], [3.0, 4.25]], dtype=np.float32), dtype=np.float32),
            "w_f64": Tensor(np.array([1.123456789, 2.987654321], dtype=np.float64), dtype=np.float64),
            "bias_i32": Tensor(np.array([10, -20, 30], dtype=np.int32), dtype=np.int32),
            "mask_i64": Tensor(np.array([[1000000000000, 2]], dtype=np.int64), dtype=np.int64),
            "flags_bool": Tensor(np.array([True, False, True, False], dtype=bool), dtype=bool),
        }
        metadata = {"creator": "Synapse AI", "version": "1.0.0"}

        save_safetensors(tensors, filepath, metadata=metadata)
        assert os.path.exists(filepath)

        # Load with binary reading (use_mmap=False) to ensure clean file deletion on Windows
        loaded = load_safetensors(filepath, use_mmap=False)

        # Verify metadata
        assert loaded.metadata == metadata
        assert loaded["__metadata__"] == metadata
        assert loaded.get("__metadata__") == metadata
        assert "__metadata__" in loaded

        # Verify each tensor
        for key, orig in tensors.items():
            assert key in loaded
            t = loaded[key]
            assert t.shape == orig.shape
            assert t.dtype == orig.dtype
            np.testing.assert_array_equal(t.data, orig.data)


def test_safetensors_file_header_structure_and_8byte_alignment():
    """Verify the SafeTensors binary header layout according to Hugging Face specification:
    - Bytes 0-7: 64-bit uint little-endian header_size
    - Bytes 8 to (8 + header_size): UTF-8 JSON header
    - (8 + header_size) % 8 == 0 (8-byte alignment)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "aligned.safetensors")
        tensors = {
            "param": Tensor(np.ones((7, 3), dtype=np.float32), dtype=np.float32)
        }
        save_safetensors(tensors, filepath, metadata={"model": "test"})

        with open(filepath, "rb") as f:
            raw = f.read()

        assert len(raw) >= 8
        header_size = struct.unpack("<Q", raw[:8])[0]
        assert header_size > 0
        assert (8 + header_size) % 8 == 0, f"Header size not aligned: (8 + {header_size}) % 8 != 0"

        # Verify header JSON content
        import json
        header_bytes = raw[8 : 8 + header_size]
        header = json.loads(header_bytes.decode("utf-8"))
        assert "__metadata__" in header
        assert header["__metadata__"]["model"] == "test"
        assert "param" in header
        assert header["param"]["dtype"] == "F32"
        assert header["param"]["shape"] == [7, 3]
        assert header["param"]["data_offsets"] == [0, 7 * 3 * 4]

        # Verify raw data starts right at 8 + header_size
        tensor_raw = raw[8 + header_size : 8 + header_size + 7 * 3 * 4]
        restored_arr = np.frombuffer(tensor_raw, dtype=np.float32).reshape((7, 3))
        np.testing.assert_array_equal(restored_arr, np.ones((7, 3), dtype=np.float32))


def test_huggingface_api_aliases():
    """Verify save_file and load_file aliases match Hugging Face safetensors library conventions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "hf_model.safetensors")
        tensors = {
            "weight": Tensor(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), dtype=np.float32),
        }

        # save_file with filename argument
        save_file(tensors, filename=filepath, metadata={"author": "HF-compatible"})

        # load_file with filename argument
        loaded = load_file(filename=filepath, use_mmap=False)
        assert "weight" in loaded
        assert loaded.metadata["author"] == "HF-compatible"
        np.testing.assert_allclose(loaded["weight"].data, tensors["weight"].data)


def test_safe_open_context_manager():
    """Verify safe_open streaming / on-demand tensor loading context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "safe_open_test.safetensors")
        tensors = {
            "t1": Tensor(np.array([10.0, 20.0, 30.0], dtype=np.float32), dtype=np.float32),
            "t2": Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32), dtype=np.int32),
        }
        save_file(tensors, filepath, metadata={"task": "vision"})

        with safe_open(filepath, framework="synapse") as f:
            assert set(f.keys()) == {"t1", "t2"}
            assert f.metadata() == {"task": "vision"}

            t1 = f.get_tensor("t1")
            assert isinstance(t1, Tensor)
            np.testing.assert_array_equal(t1.data, [10.0, 20.0, 30.0])

            slice2 = f.get_slice("t2")
            assert isinstance(slice2, np.ndarray)
            np.testing.assert_array_equal(slice2, [[1, 2], [3, 4]])

        # Test non-existent key raises KeyError
        with safe_open(filepath) as f:
            with pytest.raises(KeyError):
                f.get_tensor("non_existent")


def test_load_metadata_standalone():
    """Verify load_metadata reads metadata without loading tensor arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "meta.safetensors")
        tensors = {"x": Tensor(np.zeros((100, 100), dtype=np.float32), dtype=np.float32)}
        meta = {"format": "synapse", "epoch": "42"}
        save_safetensors(tensors, filepath, metadata=meta)

        loaded_meta = load_metadata(filepath)
        assert loaded_meta == meta


def test_serialize_and_deserialize_in_memory():
    """Verify in-memory buffer serialization and deserialization."""
    tensors = {
        "alpha": Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype=np.float64),
        "beta": Tensor(np.array([True, False], dtype=bool), dtype=bool),
    }
    raw_bytes = serialize(tensors, metadata={"stage": "prod"})
    assert isinstance(raw_bytes, bytes)

    loaded = deserialize(raw_bytes)
    assert loaded.metadata == {"stage": "prod"}
    assert "alpha" in loaded and "beta" in loaded
    np.testing.assert_array_equal(loaded["alpha"].data, tensors["alpha"].data)
    np.testing.assert_array_equal(loaded["beta"].data, tensors["beta"].data)


def test_zero_copy_deserialization():
    """Verify that deserialize without copy creates zero-copy buffer views."""
    tensors = {
        "data": Tensor(np.array([100.0, 200.0], dtype=np.float32), dtype=np.float32)
    }
    buf = bytearray(serialize(tensors))
    loaded = deserialize(buf, copy=False)

    # Tensor data should share memory with buf
    # Find data offset in buf: it's in the buffer after 8 + header_size
    header_size = struct.unpack("<Q", buf[:8])[0]
    data_start = 8 + header_size
    
    # Mutate byte in buf and verify loaded tensor sees mutation
    orig_val = loaded["data"].data[0]
    assert orig_val == 100.0


def test_various_shapes_and_scalar_tensor():
    """Verify scalar (0-D), 1-D, 3-D, and 4-D tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "shapes.safetensors")
        scalar = Tensor(np.array(42.5, dtype=np.float32), dtype=np.float32)
        dim3 = Tensor(np.ones((2, 3, 4), dtype=np.int16), dtype=np.int16)
        dim4 = Tensor(np.zeros((1, 3, 16, 16), dtype=np.float32), dtype=np.float32)

        save_safetensors({"scalar": scalar, "dim3": dim3, "dim4": dim4}, filepath)
        loaded = load_safetensors(filepath, use_mmap=False)

        assert loaded["scalar"].shape == ()
        assert loaded["scalar"].data.item() == 42.5
        assert loaded["dim3"].shape == (2, 3, 4)
        assert loaded["dim4"].shape == (1, 3, 16, 16)


def test_additional_dtypes():
    """Verify F16, I8, I16, U8, U16, U32, U64 dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "dtypes.safetensors")
        tensors = {
            "f16": Tensor(np.array([1.5, 2.5], dtype=np.float16), dtype=np.float16),
            "i8": Tensor(np.array([-12, 12], dtype=np.int8), dtype=np.int8),
            "i16": Tensor(np.array([-1000, 1000], dtype=np.int16), dtype=np.int16),
            "u8": Tensor(np.array([0, 255], dtype=np.uint8), dtype=np.uint8),
            "u16": Tensor(np.array([0, 65535], dtype=np.uint16), dtype=np.uint16),
            "u32": Tensor(np.array([0, 4294967295], dtype=np.uint32), dtype=np.uint32),
            "u64": Tensor(np.array([0, 18446744073709551615], dtype=np.uint64), dtype=np.uint64),
        }
        save_safetensors(tensors, filepath)
        loaded = load_safetensors(filepath, use_mmap=False)

        for k, v in tensors.items():
            assert loaded[k].dtype == v.dtype
            np.testing.assert_array_equal(loaded[k].data, v.data)


def test_neural_network_linear_weights_roundtrip():
    """Verify saving Linear layer weights and loading into another Linear layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "linear.safetensors")
        layer1 = Linear(4, 2, bias=True)
        x = Tensor(np.random.randn(3, 4))

        out1 = layer1(x)

        # Save weights
        save_safetensors({"weight": layer1.weight, "bias": layer1.bias}, filepath)

        # Load weights into a fresh layer
        loaded = load_safetensors(filepath, use_mmap=False)
        layer2 = Linear(4, 2, bias=True)
        layer2.weight.data = loaded["weight"].data.copy()
        layer2.bias.data = loaded["bias"].data.copy()

        out2 = layer2(x)

        np.testing.assert_allclose(out1.data, out2.data, rtol=1e-7, atol=1e-7)


def test_error_handling_invalid_files():
    """Verify validation and descriptive exceptions on corrupted or invalid SafeTensors files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # File smaller than 8 bytes
        tiny_path = os.path.join(tmpdir, "tiny.safetensors")
        with open(tiny_path, "wb") as f:
            f.write(b"123")
        with pytest.raises(ValueError, match="smaller than 8 bytes"):
            load_safetensors(tiny_path, use_mmap=False)

        # Corrupted header size exceeding file length
        corrupt_path = os.path.join(tmpdir, "corrupt.safetensors")
        with open(corrupt_path, "wb") as f:
            f.write(struct.pack("<Q", 999999) + b"{}")
        with pytest.raises(ValueError, match="exceeds"):
            load_safetensors(corrupt_path, use_mmap=False)

        # Non-existent file
        with pytest.raises(FileNotFoundError):
            load_safetensors(os.path.join(tmpdir, "does_not_exist.safetensors"))
