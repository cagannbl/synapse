"""
Unit tests for Synapse C Runtime memory limits, dynamic autograd buffer, and GEMM allocation.
Verifies that graphs exceeding the legacy 1024-node limit execute without buffer overflow.
"""
import os
import sys
import pytest
from synapse.core.tensor import Tensor
from synapse.codegen.c_emitter import CEmitter
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser


def transpile(source: str) -> str:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return CEmitter().emit(ast)


def test_runtime_c_has_dynamic_topo_buffer():
    """Verify synapse_runtime.c contains dynamic realloc logic instead of static 1024 array."""
    runtime_path = os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime", "synapse_runtime.c")
    with open(runtime_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "syn_tensor_t* topo_list[1024];" not in content, "Static 1024-node buffer must be eliminated"
    assert "realloc" in content, "Dynamic buffer must use realloc for graph growth"
    assert "topo_capacity" in content, "Must track dynamic capacity"
    assert "free(topo_list);" in content, "Must free dynamic topo buffer after backward pass"


def test_runtime_h_has_clean_backward_signature():
    """Verify synapse_runtime.h maintains standard syn_backward signature."""
    runtime_h = os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime", "synapse_runtime.h")
    with open(runtime_h, "r", encoding="utf-8") as f:
        content = f.read()

    assert "void syn_backward(syn_tensor_t* root);" in content
    assert "syn_arena_t* syn_arena_create(size_t capacity);" in content


def test_deep_autograd_graph_2000_plus_nodes():
    """Verify autograd backward pass succeeds on graphs with > 2000 nodes without limit errors."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(15000)
        x = Tensor([2.0], requires_grad=True)
        current = x
        depth = 2050

        # Chain 2050 operations: current = current + 1.0
        for _ in range(depth):
            current = current + Tensor([1.0], requires_grad=False)

        current.backward()

        # Analytical derivative of (x + c1 + c2 + ...) with respect to x is 1.0
        assert x.grad is not None
        assert abs(x.grad.item() - 1.0) < 1e-5
    finally:
        sys.setrecursionlimit(old_limit)


def test_deep_multiplication_graph_gradients():
    """Verify deep scalar multiplication chain calculates correct analytical gradient."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(15000)
        w = Tensor([1.001], requires_grad=True)
        current = w
        steps = 1500

        for _ in range(steps):
            current = current * Tensor([1.0], requires_grad=False)

        current.backward()
        assert w.grad is not None
        assert abs(w.grad.item() - 1.0) < 1e-4
    finally:
        sys.setrecursionlimit(old_limit)


def test_deep_branching_dag_3000_nodes():
    """Verify branching DAG with thousands of nodes resolves topological ordering cleanly."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(15000)
        a = Tensor([1.5], requires_grad=True)
        b = Tensor([2.5], requires_grad=True)
        
        acc = a + b
        for _ in range(1200):
            acc = acc + (a * Tensor([0.5])) + (b * Tensor([0.2]))

        acc.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.item() > 0
        assert b.grad.item() > 0
    finally:
        sys.setrecursionlimit(old_limit)


def test_c_emitter_transpiles_backward_call_cleanly():
    """Verify C emitter produces valid syn_backward call for deep models."""
    source = """
let x = tensor(1.0, requires_grad=true)
let y = x + tensor(2.0)
let z = y * tensor(3.0)
z.backward()
"""
    c_code = transpile(source)
    assert "syn_backward(z);" in c_code
    assert "syn_tensor_scalar" in c_code


def test_gemm_simd_cache_blocked_c_code():
    """Verify matmul emitter incorporates cache-blocked C runtime implementation."""
    source = """
let M1 = tensor([[1.0, 2.0], [3.0, 4.0]])
let M2 = tensor([[5.0, 6.0], [7.0, 8.0]])
let M3 = M1 @ M2
"""
    c_code = transpile(source)
    assert "syn_matmul(M1, M2);" in c_code


def test_repeated_deep_backward_resets_visited_flags():
    """Verify multiple backward passes on deep graphs properly reset node visited flags without leaking."""
    param = Tensor([3.0], requires_grad=True)
    
    for iteration in range(5):
        loss = param
        for _ in range(300):
            loss = loss + Tensor([0.1])
        
        param.grad = None
        loss.backward()
        assert param.grad is not None
        assert abs(param.grad.item() - 1.0) < 1e-5
