from __future__ import annotations
import math
from typing import Any, Iterator, Sequence, Union, Tuple
import numpy as np

from synapse.core.tensor import Tensor


class DataLoader:
    """Mini-batch iterator for Synapse Tensors and datasets.
    
    Supports:
    - Single Tensor or np.ndarray datasets: yields `batch`
    - Tuple/List of Tensors (e.g. `(X, y)`): yields `(batch_X, batch_y)`
    - Dynamic random shuffling (`shuffle=True`)
    - Configurable batch size and optional trailing batch dropping (`drop_last`)
    """

    def __init__(
        self,
        dataset: Union[Tensor, Sequence[Any], Tuple[Any, ...]],
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_samples = self._infer_len(dataset)

    def _infer_len(self, data: Any) -> int:
        if isinstance(data, (tuple, list)):
            if len(data) == 0:
                return 0
            first = data[0]
            if isinstance(first, Tensor):
                return first.shape[0] if first.ndim > 0 else 1
            return len(first)
        elif isinstance(data, Tensor):
            return data.shape[0] if data.ndim > 0 else 1
        elif hasattr(data, "__len__"):
            return len(data)
        else:
            raise TypeError(f"Cannot determine length of dataset of type {type(data)}")

    def __len__(self) -> int:
        if self.num_samples == 0:
            return 0
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[Any]:
        if self.num_samples == 0:
            return

        if self.shuffle:
            indices = np.random.permutation(self.num_samples)
        else:
            indices = np.arange(self.num_samples)

        num_batches = len(self)
        for b in range(num_batches):
            start_idx = b * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                break

            batch_indices = indices[start_idx:end_idx]

            if isinstance(self.dataset, (tuple, list)):
                batch_items = []
                for item in self.dataset:
                    if isinstance(item, Tensor):
                        sliced_data = item.data[batch_indices]
                        batch_items.append(
                            Tensor(
                                sliced_data,
                                requires_grad=item.requires_grad,
                                device=item.device,
                            )
                        )
                    elif isinstance(item, np.ndarray):
                        batch_items.append(Tensor(item[batch_indices]))
                    elif isinstance(item, list):
                        batch_items.append([item[i] for i in batch_indices])
                    else:
                        batch_items.append(item[batch_indices])
                yield tuple(batch_items)
            elif isinstance(self.dataset, Tensor):
                sliced_data = self.dataset.data[batch_indices]
                yield Tensor(
                    sliced_data,
                    requires_grad=self.dataset.requires_grad,
                    device=self.dataset.device,
                )
            elif isinstance(self.dataset, np.ndarray):
                yield Tensor(self.dataset[batch_indices])
            else:
                yield [self.dataset[i] for i in batch_indices]


__all__ = ["DataLoader"]
