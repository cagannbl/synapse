"""
Tests for Synapse Native Jupyter Kernel & Interactive Notebook Engine (Phase 2)
================================================================================
Verifies:
1. Stateful cell execution (variable state persistence across cells).
2. Function definitions and reuse across cells.
3. Rich MIME display formatting for DataFrames (Linear/Vercel dark theme HTML).
4. Rich MIME display formatting for Tensors (shape badges, stats, dtype).
5. Autocompletion engine (keywords and dynamic VM globals).
6. Kernel spec installer and JSON manifest generation.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from synapse.tools.kernel import (
    SynapseKernelEngine,
    install_kernel_spec,
    get_default_kernelspec_dir,
)
from synapse.tools.display import (
    format_html_dataframe,
    format_html_tensor,
    render_mime_bundle,
)
from synapse.core.tensor import Tensor, tensor, zeros
from synapse.core.dataframe import DataFrame


def test_kernel_stateful_cell_execution():
    """Verifies variable state is retained across separate cell executions."""
    engine = SynapseKernelEngine()

    # Cell 1: Define variables
    cell1 = """
let x: int = 42
let y: int = 58
"""
    res1 = engine.execute_cell(cell1)
    assert res1["status"] == "ok"
    assert engine.vm.globals.get("x") == 42
    assert engine.vm.globals.get("y") == 58

    # Cell 2: Use variables from Cell 1
    cell2 = """
let z = x + y
print(z)
"""
    res2 = engine.execute_cell(cell2)
    assert res2["status"] == "ok"
    assert "100" in res2["stdout"]
    assert engine.vm.globals.get("z") == 100


def test_kernel_function_definition_across_cells():
    """Verifies functions defined in one cell can be called in subsequent cells."""
    engine = SynapseKernelEngine()

    cell1 = """
fn compute_loss(pred, target):
    let diff = pred - target
    return diff * diff
"""
    res1 = engine.execute_cell(cell1)
    assert res1["status"] == "ok"
    assert "compute_loss" in engine.vm.globals

    cell2 = """
let l = compute_loss(10, 7)
print(l)
"""
    res2 = engine.execute_cell(cell2)
    assert res2["status"] == "ok"
    assert "9" in res2["stdout"]


def test_kernel_rich_display_dataframe():
    """Verifies that DataFrames produce sleek HTML tables with badges."""
    df = DataFrame({
        "model": ["Transformer", "CNN", "RNN"],
        "params_m": [125, 45, 12],
        "latency_ms": [1.42, 0.85, 2.10],
    })

    bundle = render_mime_bundle(df)
    assert "text/plain" in bundle
    assert "text/html" in bundle

    html_content = bundle["text/html"]
    assert "Synapse DataFrame" in html_content
    assert "Transformer" in html_content
    assert "params_m" in html_content
    assert "Memory-Mapped / Arrow Ready" in html_content


def test_kernel_rich_display_tensor():
    """Verifies that Tensors produce sleek badges with shape, dtype, and device."""
    t = zeros([16, 8, 128, 64])

    bundle = render_mime_bundle(t)
    assert "text/plain" in bundle
    assert "text/html" in bundle

    html_content = bundle["text/html"]
    assert "Synapse Tensor" in html_content
    assert "[16, 8, 128, 64]" in html_content
    assert "float" in html_content
    assert "cpu" in html_content
    assert "DLPack Zero-Copy Ready" in html_content


def test_kernel_autocompletion():
    """Verifies autocompletion matches Synapse keywords and active VM globals."""
    engine = SynapseKernelEngine()

    # Before defining variables
    comp = engine.complete("da", 2)
    assert "dataframe" in comp["matches"]

    # Define user variable in VM
    engine.execute_cell("let dataset_loader = 100")

    comp2 = engine.complete("data", 4)
    assert "dataset_loader" in comp2["matches"]
    assert "dataframe" in comp2["matches"]


def test_kernel_spec_installation(tmp_path):
    """Verifies kernel.json is generated correctly in destination directory."""
    target_dir = tmp_path / "synapse_kernel_test"
    manifest_path = install_kernel_spec(dest_dir=str(target_dir))

    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["display_name"] == "Synapse AI"
    assert data["language"] == "synapse"
    assert "synapse.tools.kernel" in " ".join(data["argv"])
