import pytest
import numpy as np
from synapse.core.device import Device, cpu, cuda
from synapse.core.cuda_backend import is_cuda_available, get_device_name, device_count, CUDABackend
from synapse.core.tensor import Tensor, tensor
from synapse.vm.virtual_machine import VirtualMachine


def test_device_parsing_and_equality():
    d_cpu = Device("cpu")
    assert d_cpu.device_type == "cpu"
    assert d_cpu.index == 0
    assert str(d_cpu) == "cpu"

    d_cuda0 = Device("cuda:0")
    assert d_cuda0.device_type == "cuda"
    assert d_cuda0.index == 0
    assert str(d_cuda0) == "cuda:0"

    d_cuda1 = Device("cuda:1")
    assert d_cuda1.device_type == "cuda"
    assert d_cuda1.index == 1
    assert str(d_cuda1) == "cuda:1"

    # Equalities
    assert Device("cpu") == Device("cpu")
    assert Device("cuda:0") == Device("cuda:0")
    assert Device("cuda:0") != Device("cuda:1")
    assert Device("cuda:0") != Device("cpu")

    # String comparison
    assert Device("cpu") == "cpu"
    assert Device("cuda:0") == "cuda:0"
    assert Device("cuda:0") == "cuda"  # default 0 matches 'cuda'

    # Repr and hash
    assert "cpu" in repr(d_cpu)
    assert hash(d_cpu) == hash(Device("cpu"))


def test_device_helpers():
    c = cpu()
    assert isinstance(c, Device)
    assert c.device_type == "cpu"

    g0 = cuda(0)
    assert isinstance(g0, Device)
    assert g0.device_type == "cuda"
    assert g0.index == 0

    g1 = cuda(1)
    assert g1.index == 1


def test_tensor_device_transfer():
    t = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.device.device_type == "cpu"

    # Transfer to cuda (with fallback if no hardware, or native if present)
    t_cuda = t.to("cuda")
    assert isinstance(t_cuda, Tensor)
    assert np.allclose(t_cuda.data, t.data)

    # Shortcuts
    t_gpu = t.cuda()
    assert isinstance(t_gpu, Tensor)

    t_back_cpu = t_gpu.cpu()
    assert t_back_cpu.device.device_type == "cpu"
    assert np.allclose(t_back_cpu.data, t.data)


def test_cuda_autograd_consistency():
    w = Tensor([[2.0, 3.0]], requires_grad=True).to("cuda")
    x = Tensor([[4.0], [5.0]]).to("cuda")

    # Forward
    y = w @ x  # 2*4 + 3*5 = 23
    assert np.isclose(y.data.item(), 23.0)

    # Backward
    y.backward()
    assert w.grad is not None
    assert np.allclose(w.grad.data, [[4.0, 5.0]])


def test_cuda_backend_helpers():
    available = is_cuda_available()
    assert isinstance(available, bool)

    count = device_count()
    assert isinstance(count, int)
    assert count >= 0

    name = get_device_name(0)
    assert isinstance(name, str)
    assert len(name) > 0


def test_vm_device_globals():
    vm = VirtualMachine()
    assert "Device" in vm.globals
    assert "cuda_is_available" in vm.globals
    assert "device_count" in vm.globals

    assert vm.globals["Device"] is Device
    assert callable(vm.globals["cuda_is_available"])
    assert callable(vm.globals["device_count"])
