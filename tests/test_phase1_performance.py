import time
import numpy as np
import pytest
from synapse.core.tensor import Tensor, tensor
from synapse.core.device import Device
from synapse.core.cuda_backend import is_cuda_available
from synapse.core.task_pool import spawn, channel, parallel_map, wait_all, ChannelClosed
from synapse.vm.virtual_machine import VirtualMachine
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.codegen.c_emitter import CEmitter


def run_synapse(code: str) -> VirtualMachine:
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    compiled = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(compiled)
    return vm


def transpile(source: str) -> str:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return CEmitter().emit(ast)


# =========================================================================
# 1. C Runtime Arena & Modern Activations Codegen
# =========================================================================
def test_c_emitter_phase1_activations_and_pipeline():
    source = """
let x = tensor([0.5, -1.2, 2.0])
let a = x |> sigmoid
let b = x |> tanh
let c = x |> gelu
let d = x |> softmax
"""
    c_code = transpile(source)
    assert "syn_sigmoid(x)" in c_code
    assert "syn_tanh(x)" in c_code
    assert "syn_gelu(x)" in c_code
    assert "syn_softmax(x)" in c_code


def test_c_emitter_phase1_member_methods():
    source = """
let x = tensor([0.1, 0.2])
let s = x.sigmoid()
let t = x.tanh()
let g = x.gelu()
let sm = x.softmax()
"""
    c_code = transpile(source)
    assert "syn_sigmoid(x)" in c_code
    assert "syn_tanh(x)" in c_code
    assert "syn_gelu(x)" in c_code
    assert "syn_softmax(x)" in c_code


# =========================================================================
# 2. Tensor Activations & Autograd (Tanh, GELU, Softmax)
# =========================================================================
def test_tensor_tanh_and_autograd():
    x = tensor([0.5, -0.5], requires_grad=True)
    y = x.tanh()
    assert np.allclose(y.data, np.tanh([0.5, -0.5]))

    loss = y.sum()
    loss.backward()
    expected_grad = 1.0 - np.tanh([0.5, -0.5]) ** 2
    assert x.grad is not None
    assert np.allclose(x.grad.data, expected_grad)


def test_tensor_gelu_and_autograd():
    x = tensor([0.0, 1.0], requires_grad=True)
    y = x.gelu()
    # At x=0, gelu(0) = 0
    assert np.isclose(y.data[0], 0.0)
    assert y.data[1] > 0.8  # gelu(1.0) approx 0.8413

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.data[0] == pytest.approx(0.5, abs=1e-3)


def test_tensor_softmax_and_autograd():
    x = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    sm = x.softmax(axis=-1)
    # Probabilities sum to 1
    assert np.isclose(np.sum(sm.data), 1.0)
    assert sm.data[0, 2] > sm.data[0, 1] > sm.data[0, 0]

    loss = sm.sum()
    loss.backward()
    assert x.grad is not None
    # Derivative of sum of softmax with respect to logits is 0
    assert np.allclose(x.grad.data, 0.0, atol=1e-5)


# =========================================================================
# 3. No-GIL CSP Channels & Multi-Threading Task Pool
# =========================================================================
def test_channel_send_recv_and_close():
    ch = channel(capacity=2)
    assert not ch.is_closed()

    ch.send(42)
    ch.send(100)
    assert ch.recv() == 42
    assert ch.recv() == 100

    ch.close()
    assert ch.is_closed()
    with pytest.raises(ChannelClosed):
        ch.recv(timeout=0.05)


def test_spawn_worker_and_future_result():
    def compute(a, b):
        return a * b + 10

    fut = spawn(compute, 6, 7)
    assert fut.result(timeout=2.0) == 52
    assert fut.done()


def test_channel_inter_thread_communication():
    ch = channel()

    def producer():
        for i in range(5):
            ch.send(i * 10)
            time.sleep(0.01)
        ch.close()

    spawn(producer)

    received = []
    for val in ch:
        received.append(val)

    assert received == [0, 10, 20, 30, 40]


def test_parallel_map_multi_core():
    inputs = list(range(1, 10))
    squares = parallel_map(lambda x: x * x, inputs, max_workers=4)
    assert squares == [1, 4, 9, 16, 25, 36, 49, 64, 81]


def test_wait_all_futures():
    futs = [spawn(lambda x: x + 100, i) for i in range(4)]
    results = wait_all(futs, timeout=2.0)
    assert results == [100, 101, 102, 103]


# =========================================================================
# 4. Virtual Machine Integration (Phase 1 Language Globals)
# =========================================================================
def test_synapse_vm_task_pool_and_channels():
    code = """
fn double_it(v):
    return v * 2

let ch = channel(5)
ch.send(999)
let item = ch.recv()

let fut = spawn(double_it, 21)
let task_res = fut.result()
"""
    vm = run_synapse(code)
    assert vm.globals["item"] == 999
    assert vm.globals["task_res"] == 42


def test_synapse_vm_modern_activations():
    code = """
let x = tensor([0.0, 1.0])
let t_val = tanh(x)
let g_val = gelu(x)
"""
    vm = run_synapse(code)
    assert isinstance(vm.globals["t_val"], Tensor)
    assert isinstance(vm.globals["g_val"], Tensor)
    assert np.isclose(vm.globals["t_val"].data[0], 0.0)


# =========================================================================
# 5. Hardware Acceleration (CUDA Validation)
# =========================================================================
def test_cuda_hardware_acceleration_if_available():
    if not is_cuda_available():
        pytest.skip("CUDA not present on this test environment.")

    # Matrix multiplication on GPU
    A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).to("cuda")
    B = tensor([[5.0, 6.0], [7.0, 8.0]]).to("cuda")

    C = A @ B
    assert C.device.device_type == "cuda"
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(C.data, expected)

    loss = C.sum()
    loss.backward()
    assert A.grad is not None
    assert A.grad.device.device_type == "cuda"
