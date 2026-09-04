"""Synapse Core: Tensor Engine, Autograd System, and Generators."""

from synapse.core.tensor import Tensor, tensor, QuantizedTensor, quantize, dequantize
from synapse.core.autograd import grad
from synapse.core.generators import (
    SynapseGenerator,
    StreamPipeline,
    TokenStream,
    GeneratorState,
    from_generator_func,
    synapse_generator,
)

__all__ = [
    "Tensor",
    "tensor",
    "QuantizedTensor",
    "quantize",
    "dequantize",
    "grad",
    "SynapseGenerator",
    "StreamPipeline",
    "TokenStream",
    "GeneratorState",
    "from_generator_func",
    "synapse_generator",
]
