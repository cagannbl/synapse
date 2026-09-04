from typing import Callable, Any
from synapse.core.tensor import Tensor


class no_grad:
    """Gradyan hesaplamasını geçici olarak devre dışı bırakan context manager."""
    def __enter__(self):
        self._prev_state = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def grad(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Fonksiyonel otomatik türev alıcı.
    Verilen fonksiyona göre girdilerin gradyanlarını döndüren yeni bir fonksiyon üretir.
    """
    def grad_fn(*args, **kwargs):
        # Argümanları Tensor ve requires_grad=True olarak hazırla
        tensor_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                t = Tensor(arg.data, requires_grad=True)
            elif isinstance(arg, (int, float, list)):
                t = Tensor(arg, requires_grad=True)
            else:
                t = arg
            tensor_args.append(t)

        # Fonksiyonu çalıştır
        output = fn(*tensor_args, **kwargs)

        if not isinstance(output, Tensor):
            raise TypeError(f"Function output must be a Tensor for autograd, got {type(output)}")

        # Geriye doğru türet
        output.backward()

        grads = [t.grad for t in tensor_args if isinstance(t, Tensor)]
        if len(grads) == 1:
            return grads[0]
        return tuple(grads)

    return grad_fn
