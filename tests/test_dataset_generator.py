import os
import json
import pytest
from tools.generate_dataset import (
    verify_synapse_code,
    DatasetGenerator,
    SEED_TEMPLATES
)


def test_verify_synapse_code_valid():
    valid_code = """
let x = tensor([[1.0, 2.0], [3.0, 4.0]])
let y = x @ x.T
let res = y.sum().item()
"""
    success, msg, vm = verify_synapse_code(valid_code)
    assert success is True
    assert vm is not None
    assert vm.globals["res"] == 52.0


def test_verify_synapse_code_invalid_syntax():
    invalid_code = "def bad_syntax():\n    return 42"
    success, msg, vm = verify_synapse_code(invalid_code)
    # Synapse requires 'fn', not 'def' (wait, def is supported as alias for fn in KEYWORDS, but let's test a real syntax error)
    bad_syntax = "let = 42"
    success, msg, vm = verify_synapse_code(bad_syntax)
    assert success is False
    assert "Syntax error" in msg


def test_verify_synapse_code_runtime_error():
    runtime_err_code = "let a = tensor([1.0, 2.0])\nlet b = tensor([[1.0], [2.0], [3.0]])\nlet c = a @ b"
    success, msg, vm = verify_synapse_code(runtime_err_code)
    assert success is False


def test_dataset_generator_procedural_output(tmp_path):
    out_file = tmp_path / "dataset.jsonl"
    gen = DatasetGenerator()
    dataset = gen.build_dataset(count=5, mode="procedural", format_type="chatml")
    assert len(dataset) == 5

    # Check each sample format
    for item in dataset:
        assert "messages" in item
        assert len(item["messages"]) == 3
        assert item["messages"][0]["role"] == "system"
        assert item["messages"][1]["role"] == "user"
        assert item["messages"][2]["role"] == "assistant"
        assert "```synapse" in item["messages"][2]["content"]


def test_all_seed_templates_are_valid():
    # Ensure every single seed template compiles and runs cleanly
    for i, t in enumerate(SEED_TEMPLATES):
        code = t["code_fn"]()
        ok, msg, _ = verify_synapse_code(code)
        assert ok is True, f"Seed template {i} ({t['category']}) failed: {msg}\nCode:\n{code}"
