import pytest
import numpy as np
from synapse.core.tensor import Tensor, tensor, zeros, ones, randn


def test_tensor_creation_and_properties():
    t = tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.ndim == 2
    assert t.size == 4
    assert t.requires_grad is False


def test_tensor_matrix_multiplication():
    A = tensor([[1.0, 2.0], [3.0, 4.0]])
    B = tensor([[2.0, 0.0], [1.0, 2.0]])
    C = A @ B

    expected = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[2.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(C.data, expected)


def test_tensor_broadcasting_addition():
    A = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tensor([10.0, 20.0, 30.0])
    C = A + b

    expected = np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
    np.testing.assert_allclose(C.data, expected)


def test_tensor_transpose():
    A = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    At = A.T
    assert At.shape == (3, 2)
    np.testing.assert_allclose(At.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_activations():
    x = tensor([-2.0, 0.0, 3.0])
    r = x.relu()
    np.testing.assert_allclose(r.data, [0.0, 0.0, 3.0])

    s = tensor([0.0]).sigmoid()
    assert abs(s.item() - 0.5) < 1e-6
