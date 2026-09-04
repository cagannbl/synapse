import pytest
import numpy as np
from synapse.core.tensor import Tensor, tensor
from synapse.nn import Linear, Sequential, ReLU, Sigmoid, MSELoss
from synapse.optim import SGD, Adam
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def test_linear_layer_forward_and_backward():
    layer = Linear(2, 3)
    x = tensor([[1.0, 2.0]])
    out = layer(x)

    assert out.shape == (1, 3)
    assert len(layer.parameters()) == 2  # weight ve bias

    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_sequential_and_activations():
    model = Sequential([
        Linear(3, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    ])
    x = tensor([[1.0, 2.0, 3.0]])
    out = model(x)

    assert out.shape == (1, 1)
    assert 0.0 <= out.item() <= 1.0


def test_adam_optimizer_step():
    w = tensor([[2.0]], requires_grad=True)
    optimizer = Adam([w], lr=0.1)

    loss = (w * 3.0).sum()
    loss.backward()

    initial_val = w.item()
    optimizer.step()

    # Gradyan pozitif olduğu için değer azalmalı
    assert w.item() < initial_val


def test_synapse_vm_neural_network_training():
    source = """
let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])
let optimizer = Adam(model.parameters(), lr=0.1)
let criterion = MSELoss()

let X = tensor([[1.0, 2.0], [2.0, 3.0]])
let Y = tensor([[5.0], [8.0]])

let initial_loss = criterion(model(X), Y).item()

let epoch = 1
while epoch <= 10:
    let y_pred = model(X)
    let loss = criterion(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1

let final_loss = criterion(model(X), Y).item()
"""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(code)

    initial_loss = vm.globals["initial_loss"]
    final_loss = vm.globals["final_loss"]

    # Model öğrenmeli ve kayıp azalmalı
    assert final_loss < initial_loss
