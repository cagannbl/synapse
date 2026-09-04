"""
Tests for Edge NanoGPT Reference Architecture, Tokenizer, SafeTensors Weights & C99 Inference
"""

import os
import sys
import subprocess
import pytest
from synapse.parser import parse_source


def test_edge_nanogpt_source_parses_cleanly():
    """model.syn must parse cleanly without syntax errors."""
    syn_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "edge_nanogpt", "model.syn")
    )
    assert os.path.isfile(syn_path)
    with open(syn_path, "r", encoding="utf-8") as f:
        source = f.read()

    ast = parse_source(source, filename="model.syn", use_cache=False)
    assert ast is not None
    assert len(ast.statements) > 10


def test_edge_nanogpt_tokenizer_roundtrip():
    """EdgeNanoTokenizer should encode and decode UTF-8 text perfectly."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "edge_nanogpt")))
    from tokenizer import EdgeNanoTokenizer

    tokenizer = EdgeNanoTokenizer()
    text = "Synapse AI Edge Transformer"
    tokens = tokenizer.encode(text)
    assert len(tokens) == len(text.encode("utf-8"))
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_edge_nanogpt_safetensors_weights_roundtrip(tmp_path):
    """weights.py should generate valid SafeTensors weights and load with correct shapes."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "edge_nanogpt")))
    from weights import generate_nanogpt_weights, load_nanogpt_weights

    weights_file = str(tmp_path / "nanogpt.safetensors")
    generate_nanogpt_weights(weights_file)
    assert os.path.isfile(weights_file)
    assert os.path.getsize(weights_file) > 0

    loaded = load_nanogpt_weights(weights_file)
    assert "transformer.h.0.attn.w_q.weight" in loaded
    assert "lm_head.weight" in loaded
    assert list(loaded["lm_head.weight"].shape) == [2, 3]


def test_edge_nanogpt_c99_binary_execution():
    """Built nanogpt binary should execute and print valid forward logits."""
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "edge_nanogpt"))
    is_win = sys.platform.startswith("win")
    binary_name = "nanogpt.exe" if is_win else "nanogpt"
    binary_path = os.path.join(current_dir, binary_name)

    if not os.path.isfile(binary_path):
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from build import build_edge_nanogpt
        binary_path = build_edge_nanogpt(binary_path)

    assert os.path.isfile(binary_path)
    res = subprocess.run([binary_path], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Next-Token Prediction Logits" in res.stdout
    assert "0 runtime allocations" in res.stdout
