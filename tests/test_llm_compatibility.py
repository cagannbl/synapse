import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine
from synapse.codegen.c_emitter import CEmitter


def run_code(source: str) -> VirtualMachine:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(code)
    return vm


def test_llm_template_matrix_operations():
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
let At = A.T
"""
    vm = run_code(source)
    assert vm.globals["C"].shape == (2, 2)
    assert vm.globals["At"].shape == (2, 2)

    # C transpilation testi
    ast = Parser(Lexer(source).tokenize()).parse()
    c_out = CEmitter().emit(ast)
    assert "syn_matmul(A, B)" in c_out


def test_llm_template_autograd_training():
    source = """
let w = tensor(0.5, requires_grad=true)
let x = tensor(2.0)
let y_true = tensor(6.0)
let lr = 0.05

let epoch = 1
while epoch <= 5:
    let y_pred = w * x
    let loss = (y_pred - y_true) ** 2
    loss.backward()
    let new_w = w.item() - lr * w.grad.item()
    w = tensor(new_w, requires_grad=true)
    epoch += 1
"""
    vm = run_code(source)
    assert vm.globals["w"].item() > 0.5  # Ağırlık 3.0 hedefine doğru öğreniyor

    # C transpilation testi
    ast = Parser(Lexer(source).tokenize()).parse()
    c_out = CEmitter().emit(ast)
    assert "syn_backward(loss)" in c_out


def test_llm_template_pipeline_relu():
    source = """
fn double(x):
    return x * 2.0

let veri = tensor([-2.0, 0.0, 3.0])
let cikti = veri |> relu |> double
"""
    vm = run_code(source)
    cikti = vm.globals["cikti"]
    assert cikti.data[0] == 0.0
    assert cikti.data[1] == 0.0
    assert cikti.data[2] == 6.0
