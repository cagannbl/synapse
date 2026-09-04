from __future__ import annotations
import math
from typing import Optional
from synapse.optim import Optimizer


class LRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = last_epoch
        # Initial step to align with last_epoch=0
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> float:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        new_lr = self.get_lr()
        self.optimizer.lr = new_lr
        return new_lr


class StepLR(LRScheduler):
    """Decays the learning rate of the optimizer by gamma every step_size epochs.
    
    Formula:
        lr = base_lr * (gamma ** (epoch // step_size))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 10,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch <= 0:
            return self.base_lr
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return self.base_lr * factor


class CosineAnnealingLR(LRScheduler):
    """Sets the learning rate using a cosine annealing schedule.
    
    Formula:
        lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int = 100,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch <= 0:
            return self.base_lr
        if self.last_epoch >= self.T_max:
            return self.eta_min
        return (
            self.eta_min
            + 0.5
            * (self.base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.last_epoch / self.T_max))
        )


__all__ = ["LRScheduler", "StepLR", "CosineAnnealingLR"]
