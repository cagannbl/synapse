import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine
from synapse.core.tensor import Tensor


def run_source(source: str) -> VirtualMachine:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(code)
    return vm


def test_python_math_module_interop():
    source = """
import py.math as m
let pi_val = m.pi
let cos_val = m.cos(0.0)
let sqrt_val = m.sqrt(16.0)
"""
    vm = run_source(source)
    assert abs(vm.globals["pi_val"] - 3.14159265) < 1e-4
    assert vm.globals["cos_val"] == 1.0
    assert vm.globals["sqrt_val"] == 4.0


def test_python_numpy_interop_with_synapse_tensor():
    source = """
import py.numpy as np
let t = tensor([1.0, 4.0, 9.0])
let root_t = np.sqrt(t)
"""
    vm = run_source(source)
    root_t = vm.globals["root_t"]
    assert isinstance(root_t, Tensor)
    assert root_t.shape == (3,)
    assert root_t.data[0] == 1.0
    assert root_t.data[1] == 2.0
    assert root_t.data[2] == 3.0
