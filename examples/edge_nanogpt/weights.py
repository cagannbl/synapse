"""
Synapse Edge NanoGPT SafeTensors Binary Weight Manager
======================================================
Serializes and maps Transformer weights using Hugging Face SafeTensors binary format.
"""

import os
from typing import Dict
from synapse.core.tensor import Tensor
from synapse.nn.safetensors import save_file, load_file


def generate_nanogpt_weights(weights_path: str) -> str:
    """Generates and saves a synthetic SafeTensors checkpoint for Edge NanoGPT."""
    weights: Dict[str, Tensor] = {
        "transformer.wte.weight": Tensor([[0.05, -0.02], [0.08, 0.12]], dtype="float32"),
        "transformer.h.0.attn.w_q.weight": Tensor([[0.1, 0.0], [0.0, 0.1]], dtype="float32"),
        "transformer.h.0.attn.w_k.weight": Tensor([[0.1, 0.0], [0.0, 0.1]], dtype="float32"),
        "transformer.h.0.attn.w_v.weight": Tensor([[0.1, 0.0], [0.0, 0.1]], dtype="float32"),
        "transformer.h.0.attn.w_o.weight": Tensor([[0.1, 0.0], [0.0, 0.1]], dtype="float32"),
        "transformer.h.0.mlp.w_fc1.weight": Tensor([[0.15, -0.05], [0.05, 0.15]], dtype="float32"),
        "transformer.h.0.mlp.w_fc2.weight": Tensor([[0.10, 0.08], [-0.02, 0.12]], dtype="float32"),
        "transformer.ln_f.gamma": Tensor([[1.0, 1.0]], dtype="float32"),
        "transformer.ln_f.beta": Tensor([[0.0, 0.0]], dtype="float32"),
        "lm_head.weight": Tensor([[0.5, -0.3, 0.8], [0.2, 0.9, -0.1]], dtype="float32"),
    }

    metadata = {
        "format": "synapse-safetensors",
        "model": "EdgeNanoGPT-15M",
        "d_model": "32",
        "layers": "2",
        "heads": "2",
    }

    save_file(weights, weights_path, metadata=metadata)
    return weights_path


def load_nanogpt_weights(weights_path: str) -> Dict[str, Tensor]:
    """Loads SafeTensors weights from disk with zero-copy mmap."""
    return load_file(weights_path)
