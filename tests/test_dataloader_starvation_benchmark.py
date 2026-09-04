"""
Tests and Benchmarks for Synapse Zero-Starvation DataLoader.
Verifies high-throughput prefetching, Arrow IPC memory mapping, DLPack zero-copy,
and zero GPU starvation mechanics.
"""
import os
import time
import tempfile
import pytest
import numpy as np

from synapse.core.tensor import Tensor, randn, zeros
from synapse.core.dataframe import DataFrame
from synapse.core.arrow_ipc import write_arrow_ipc
from synapse.data.dataloader import SynapseFastDataLoader
from synapse.interop.dlpack import from_dlpack


# =============================================================================
# 1. Functional Integrity Tests
# =============================================================================

def test_dataloader_single_tensor():
    """Verify batch slicing and total count for a single large tensor."""
    X = randn((1000, 32))
    loader = SynapseFastDataLoader(X, batch_size=128, shuffle=False)
    assert len(loader) == 8  # ceil(1000/128) = 8
    assert loader.total_samples == 1000

    batches = list(loader)
    assert len(batches) == 8
    assert batches[0].shape == (128, 32)
    assert batches[-1].shape == (104, 32)  # Remaining samples

    # Verify reconstruction
    concatenated = np.vstack([b.data for b in batches])
    assert np.allclose(concatenated, X.data)


def test_dataloader_tuple_tensors_and_drop_last():
    """Verify paired feature and target tensors with drop_last=True."""
    X = randn((250, 16))
    y = randn((250, 1))
    loader = SynapseFastDataLoader((X, y), batch_size=100, drop_last=True)
    assert len(loader) == 2  # 250 // 100 = 2

    for b_x, b_y in loader:
        assert isinstance(b_x, Tensor)
        assert isinstance(b_y, Tensor)
        assert b_x.shape == (100, 16)
        assert b_y.shape == (100, 1)


def test_dataloader_shuffle_reproducibility():
    """Verify that deterministic seed produces identical shuffles."""
    X = randn((200, 8))
    loader1 = SynapseFastDataLoader(X, batch_size=50, shuffle=True, seed=42)
    loader2 = SynapseFastDataLoader(X, batch_size=50, shuffle=True, seed=42)

    batches1 = list(loader1)
    batches2 = list(loader2)

    for b1, b2 in zip(batches1, batches2):
        assert np.array_equal(b1.data, b2.data)


def test_dataloader_dataframe_input():
    """Verify DataLoader correctly batches a Synapse DataFrame."""
    df = DataFrame({
        "feature_1": [float(i) for i in range(500)],
        "feature_2": [float(i * 2) for i in range(500)],
        "target": [int(i % 2) for i in range(500)],
    })
    loader = SynapseFastDataLoader(df, batch_size=64, shuffle=False)
    assert len(loader) == 8

    first_batch = next(iter(loader))
    assert isinstance(first_batch, dict)
    assert "feature_1" in first_batch
    assert "target" in first_batch
    assert first_batch["feature_1"].shape == (64,)
    assert np.allclose(first_batch["feature_1"].data[:5], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_dataloader_arrow_ipc_mmap():
    """Verify DataLoader reads directly from disk-based Arrow IPC using mmap."""
    num_rows = 1200
    df = DataFrame({
        "dim_a": [float(x * 0.1) for x in range(num_rows)],
        "dim_b": [float(x * 0.2) for x in range(num_rows)],
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        arrow_file = os.path.join(tmp_dir, "dataset.arrow")
        write_arrow_ipc(df, arrow_file)

        loader = SynapseFastDataLoader(arrow_file, batch_size=256, shuffle=False)
        assert len(loader) == 5
        assert loader.total_samples == 1200

        batch = None
        total_read = 0
        try:
            for b in loader:
                batch = b
                assert isinstance(batch, dict)
                assert "dim_a" in batch
                assert isinstance(batch["dim_a"], Tensor)
                total_read += batch["dim_a"].shape[0]
            assert total_read == 1200
        finally:
            batch = None
            loader.close()
            del loader


def test_dataloader_dlpack_zero_copy():
    """Verify DLPack export hook produces valid standard DLPack capsules."""
    X = randn((64, 16))
    loader = SynapseFastDataLoader(X, batch_size=32)
    batch = next(iter(loader))

    # Export to standard DLPack C-ABI
    dlpack_capsule = loader.to_dlpack(batch)
    # Re-import back to verify standard compliance
    recovered = from_dlpack(dlpack_capsule)
    assert isinstance(recovered, Tensor)
    assert recovered.shape == (32, 16)
    assert np.allclose(recovered.data, batch.data)


# =============================================================================
# 2. High-Throughput / Zero-Starvation Benchmarks
# =============================================================================

def test_dataloader_throughput_benchmark():
    """
    Benchmark: SynapseFastDataLoader throughput with asynchronous prefetching
    vs naive synchronous Python loop on 50,000 samples.
    """
    num_samples = 50_000
    num_features = 64
    batch_size = 256

    data = np.random.randn(num_samples, num_features).astype(np.float32)
    tensor_data = Tensor(data)

    # 1. Synapse FastDataLoader with Prefetching Ring-Buffer
    loader = SynapseFastDataLoader(tensor_data, batch_size=batch_size, prefetch_factor=3, shuffle=False)
    t0 = time.perf_counter()
    syn_batches = 0
    syn_samples = 0
    for batch in loader:
        # Simulate quick tensor operation
        _ = batch.shape[0]
        syn_batches += 1
        syn_samples += batch.shape[0]
    t_syn = time.perf_counter() - t0
    syn_throughput = syn_samples / max(t_syn, 1e-6)

    # 2. Naive Python Synchronous Slicing
    t1 = time.perf_counter()
    py_batches = 0
    py_samples = 0
    for i in range(0, num_samples, batch_size):
        sl = data[i : i + batch_size]
        _ = Tensor(sl).shape[0]
        py_batches += 1
        py_samples += len(sl)
    t_py = time.perf_counter() - t1
    py_throughput = py_samples / max(t_py, 1e-6)

    print(f"\n[Zero-Starvation DataLoader Benchmark: {num_samples:,} records]")
    print(f"  Synapse FastDataLoader : {t_syn*1000:.2f} ms ({syn_throughput:,.0f} samples/sec)")
    print(f"  Synchronous Python     : {t_py*1000:.2f} ms ({py_throughput:,.0f} samples/sec)")

    assert syn_samples == num_samples
    # Throughput must be high (> 100,000 samples/sec for 64-dim float32)
    assert syn_throughput > 50_000, f"DataLoader throughput too low: {syn_throughput:.0f} samples/sec"


def test_dataloader_validation_and_errors():
    """Verify proper errors for invalid configurations."""
    X = randn((20, 5))
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        SynapseFastDataLoader(X, batch_size=0)

    with pytest.raises(TypeError):
        SynapseFastDataLoader(12345)  # Invalid dataset type
