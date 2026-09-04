import pytest
import numpy as np
from synapse.core.tensor import Tensor, tensor
from synapse.core.autograd import grad


def test_scalar_autograd():
    # y = x^2, dy/dx = 2x, x=3 -> 6
    x = tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()

    assert x.grad is not None
    assert abs(x.grad.item() - 6.0) < 1e-6


def test_composite_expression_autograd():
    # z = (2*x + 3*y)^2
    # dz/dx = 2 * (2*x + 3*y) * 2
    x = tensor(2.0, requires_grad=True)
    y = tensor(1.0, requires_grad=True)
    z = (x * 2.0 + y * 3.0) ** 2
    z.backward()

    # 2*2 + 3*1 = 7. dz/dx = 2 * 7 * 2 = 28. dz/dy = 2 * 7 * 3 = 42.
    assert abs(x.grad.item() - 28.0) < 1e-5
    assert abs(y.grad.item() - 42.0) < 1e-5


def test_matrix_matmul_autograd():
    # Y = X @ W
    # L = sum(Y)
    # dL/dW = X.T @ ones
    X = tensor([[1.0, 2.0], [3.0, 4.0]])
    W = tensor([[0.5], [1.5]], requires_grad=True)

    Y = X @ W
    loss = Y.sum()
    loss.backward()

    # X.T @ [[1], [1]] = [[1, 3], [2, 4]] @ [[1], [1]] = [[4], [6]]
    expected_grad = np.array([[4.0], [6.0]])
    np.testing.assert_allclose(W.grad.data, expected_grad)


def test_functional_grad():
    def poly(x):
        return x * x + x * 3.0 + 5.0

    d_poly = grad(poly)
    # d/dx (x^2 + 3x + 5) = 2x + 3
    # x = 4 -> 2*4 + 3 = 11
    slope = d_poly(4.0)
    assert abs(slope.item() - 11.0) < 1e-5


def test_relu_autograd():
    x = tensor([-3.0, 2.0, 5.0], requires_grad=True)
    y = x.relu().sum()
    y.backward()

    # d(relu)/dx: x < 0 ise 0, x > 0 ise 1
    expected_grad = [0.0, 1.0, 1.0]
    np.testing.assert_allclose(x.grad.data, expected_grad)
