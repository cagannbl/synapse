from typing import Sequence
import numpy as np
from synapse.core.tensor import Tensor


class Optimizer:
    def __init__(self, params: Sequence[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stokastik Gradyan İnişi (SGD)."""
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) Optimizasyon Algoritması."""
    def __init__(
        self,
        params: Sequence[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.v: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.data

            # 1. ve 2. moment güncellemeleri
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            # Bias düzeltmesi
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Ağırlık güncellemesi
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


from synapse.optim.lr_scheduler import LRScheduler, StepLR, CosineAnnealingLR

__all__ = ["Optimizer", "SGD", "Adam", "LRScheduler", "StepLR", "CosineAnnealingLR"]
