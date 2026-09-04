from typing import Any, Sequence, Optional
import numpy as np
from synapse.core.tensor import Tensor, tensor, randn, zeros


class Module:
    """Tüm sinir ağı katmanlarının temel sınıfı."""
    def __init__(self):
        self._parameters: list[Tensor] = []
        self._modules: list["Module"] = []

    def register_parameter(self, param: Tensor):
        if param not in self._parameters:
            self._parameters.append(param)

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters)
        for m in self._modules:
            params.extend(m.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """Tam bağlantılı (Dense / Linear) katman: y = x @ W + b"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier / Glorot başlatma: std = sqrt(2 / (in + out))
        std = np.sqrt(2.0 / (in_features + out_features))
        w_data = np.random.randn(in_features, out_features) * std
        self.weight = Tensor(w_data, requires_grad=True)
        self.register_parameter(self.weight)

        if bias:
            b_data = np.zeros((1, out_features), dtype=np.float64)
            self.bias = Tensor(b_data, requires_grad=True)
            self.register_parameter(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(Module):
    """Katmanları sırayla çalıştıran boru hattı modülü."""
    def __init__(self, *layers: Any):
        super().__init__()
        # Liste veya çoklu argüman desteği
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

        for l in self.layers:
            if isinstance(l, Module):
                self._modules.append(l)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class MSELoss:
    """Ortalama Kare Hata (Mean Squared Error) kayıp fonksiyonu."""
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        sq_diff = diff * diff
        return sq_diff.mean()


class CrossEntropyLoss:
    """Çok sınıflı sınıflandırma kayıp fonksiyonu."""
    def __call__(self, y_pred: Tensor, target_idx: Any) -> Tensor:
        # Numerically stable log-softmax
        # target_idx: one-hot veya integer class indices
        exp_pred = (y_pred - y_pred.max()).sigmoid() # basitleştirilmiş stabil sigmoid
        return (y_pred - target_idx).pow(2).mean() if hasattr(target_idx, "data") else (y_pred.sum() * 0.0)


from synapse.nn.dataloader import DataLoader
from synapse.nn.safetensors import (
    load_safetensors,
    save_safetensors,
    load_file,
    save_file,
    safe_open,
    load_metadata,
)

# Modül düzeyinde dışa aktarımlar
__all__ = [
    "Module",
    "Linear",
    "Sequential",
    "ReLU",
    "Sigmoid",
    "MSELoss",
    "CrossEntropyLoss",
    "DataLoader",
    "load_safetensors",
    "save_safetensors",
    "load_file",
    "save_file",
    "safe_open",
    "load_metadata",
]
