import math
import pytest
import numpy as np
from synapse.core.tensor import Tensor, randn, zeros
from synapse.nn.dataloader import DataLoader
from synapse.optim import SGD, Adam
from synapse.optim.lr_scheduler import StepLR, CosineAnnealingLR
from synapse.vm.virtual_machine import VirtualMachine


def test_dataloader_single_tensor():
    X = randn((100, 4))
    loader = DataLoader(X, batch_size=32, shuffle=False)

    assert len(loader) == 4  # 32 + 32 + 32 + 4

    batches = list(loader)
    assert len(batches) == 4
    assert batches[0].shape == (32, 4)
    assert batches[1].shape == (32, 4)
    assert batches[2].shape == (32, 4)
    assert batches[3].shape == (4, 4)

    # Reconstruct whole data
    reconstructed = np.vstack([b.data for b in batches])
    assert np.allclose(reconstructed, X.data)


def test_dataloader_tuple_tensors_and_shuffle():
    X = randn((100, 4))
    y = randn((100, 1))

    loader = DataLoader((X, y), batch_size=25, shuffle=True)
    assert len(loader) == 4

    total_samples = 0
    for b_x, b_y in loader:
        assert isinstance(b_x, Tensor)
        assert isinstance(b_y, Tensor)
        assert b_x.shape[0] == b_y.shape[0]
        total_samples += b_x.shape[0]

    assert total_samples == 100


def test_dataloader_drop_last_and_validation():
    X = randn((50, 2))
    loader = DataLoader(X, batch_size=20, drop_last=True)
    assert len(loader) == 2  # 50 // 20 = 2

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0].shape == (20, 2)
    assert batches[1].shape == (20, 2)

    with pytest.raises(ValueError):
        DataLoader(X, batch_size=0)


def test_steplr_schedule():
    w = randn((2, 2), requires_grad=True)
    opt = SGD([w], lr=0.1)
    scheduler = StepLR(opt, step_size=2, gamma=0.5)

    assert math.isclose(opt.lr, 0.1)

    # Step 1 (epoch 1) -> still 0.1
    scheduler.step()
    assert math.isclose(opt.lr, 0.1)

    # Step 2 (epoch 2) -> decays to 0.05
    scheduler.step()
    assert math.isclose(opt.lr, 0.05)

    # Step 3 (epoch 3) -> still 0.05
    scheduler.step()
    assert math.isclose(opt.lr, 0.05)

    # Step 4 (epoch 4) -> decays to 0.025
    scheduler.step()
    assert math.isclose(opt.lr, 0.025)


def test_cosine_annealing_lr():
    w = randn((2, 2), requires_grad=True)
    opt = Adam([w], lr=0.1)
    scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)

    assert math.isclose(opt.lr, 0.1)

    # Middle of cosine curve (epoch = 5)
    scheduler.step(5)
    assert math.isclose(opt.lr, 0.05, abs_tol=1e-5)

    # End of cosine curve (epoch = 10)
    scheduler.step(10)
    assert math.isclose(opt.lr, 0.0, abs_tol=1e-5)


def test_vm_globals_dataloader_scheduler():
    vm = VirtualMachine()
    assert "DataLoader" in vm.globals
    assert "StepLR" in vm.globals
    assert "CosineAnnealingLR" in vm.globals

    assert vm.globals["DataLoader"] is DataLoader
    assert vm.globals["StepLR"] is StepLR
    assert vm.globals["CosineAnnealingLR"] is CosineAnnealingLR
