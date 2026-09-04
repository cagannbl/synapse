import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine
from synapse.core.tensor import Tensor


def run_source(source: str, vm: VirtualMachine = None) -> VirtualMachine:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    if vm is None:
        vm = VirtualMachine()
    vm.execute(code)
    return vm


def test_vm_basic_arithmetic_and_variables():
    source = """
let a = 10
let b = 25
let c = a + b * 2
"""
    vm = run_source(source)
    assert vm.globals["a"] == 10
    assert vm.globals["b"] == 25
    assert vm.globals["c"] == 60


def test_vm_function_call():
    source = """
fn multiply(x, y):
    return x * y

let res = multiply(6, 7)
"""
    vm = run_source(source)
    assert vm.globals["res"] == 42


def test_vm_loops_and_conditionals():
    source = """
let total = 0
let i = 1
while i <= 5:
    total += i
    i += 1

let status = "none"
if total > 10:
    status = "high"
else:
    status = "low"
"""
    vm = run_source(source)
    assert vm.globals["total"] == 15
    assert vm.globals["status"] == "high"


def test_vm_tensor_matrix_multiplication():
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[2.0, 0.0], [1.0, 2.0]])
let C = A @ B
"""
    vm = run_source(source)
    C = vm.globals["C"]
    assert isinstance(C, Tensor)
    assert C.shape == (2, 2)
    assert C.data[0, 0] == 4.0
    assert C.data[1, 0] == 10.0


def test_vm_pipeline_operator():
    source = """
fn double(x):
    return x * 2.0

let val = 5.0 |> double
"""
    vm = run_source(source)
    assert vm.globals["val"] == 10.0


def test_vm_autograd_execution():
    source = """
let x = tensor(3.0, requires_grad=true)
let y = x * x + x * 2.0
y.backward()
let slope = x.grad.item()
"""
    vm = run_source(source)
    assert abs(vm.globals["slope"] - 8.0) < 1e-5


def test_vm_prompt_definition_and_call():
    source = """
prompt greet(user_name):
    system: "You are a friendly greeter."
    user: user_name

let greeting = greet("Alice")
"""
    vm = run_source(source)
    greeting = vm.globals["greeting"]
    assert "Alice" in greeting
    assert "friendly greeter" in greeting
