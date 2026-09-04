import pytest
import numpy as np

from synapse.core.tensor import (
    Tensor,
    QuantizedTensor,
    tensor,
    randn,
    quantize,
    dequantize,
    TensorShapeMismatchError,
)
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.codegen.c_emitter import CEmitter


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(dot / (norm_a * norm_b))


# =========================================================================
# 1. Quantization & Dequantization Tests
# =========================================================================

def test_affine_quantization_dequantization_cosine_similarity():
    """Tests affine INT8 quantization & dequantization achieves cosine similarity > 0.99."""
    np.random.seed(42)
    # Random normal tensor with varied dynamic range
    t = randn((32, 32))

    q = t.quantize(dtype="int8")
    assert isinstance(q, QuantizedTensor)
    assert q.dtype == "int8"
    assert q.qdata.dtype == np.int8
    assert q.shape == (32, 32)
    assert q.size == 1024
    assert q.ndim == 2

    # Verify scale and zero_point are computed
    assert q.scale > 0.0
    assert isinstance(q.zero_point, int)

    # Dequantize back to float32
    rec = q.dequantize()
    assert isinstance(rec, Tensor)
    assert rec.shape == (32, 32)
    assert rec.dtype == np.float32

    # Calculate cosine similarity with original
    sim = _cosine_similarity(t.data, rec.data)
    assert sim > 0.99, f"Cosine similarity {sim} is not > 0.99"


def test_symmetric_quantization_cosine_similarity():
    """Tests symmetric INT8 quantization achieves zero_point == 0 and cosine similarity > 0.99."""
    np.random.seed(123)
    t = randn((64, 64))

    q_sym = t.quantize(dtype="int8", symmetric=True)
    assert q_sym.zero_point == 0
    assert q_sym.scale > 0.0

    rec = q_sym.dequantize()
    sim = _cosine_similarity(t.data, rec.data)
    assert sim > 0.99, f"Symmetric cosine similarity {sim} is not > 0.99"


def test_fp8_quantization_dequantization():
    """Tests FP8 (E4M3 and E5M2) quantization & dequantization accuracy."""
    np.random.seed(99)
    t = randn((16, 16))

    # FP8 E4M3
    q_fp8 = t.quantize(dtype="fp8")
    assert isinstance(q_fp8, QuantizedTensor)
    assert q_fp8.dtype == "fp8"
    rec_fp8 = q_fp8.dequantize()
    sim_fp8 = _cosine_similarity(t.data, rec_fp8.data)
    assert sim_fp8 > 0.99, f"FP8 E4M3 similarity {sim_fp8} is not > 0.99"

    # FP8 E5M2
    q_fp8_e5 = t.quantize(dtype="fp8_e5m2")
    assert q_fp8_e5.dtype == "fp8_e5m2"
    rec_fp8_e5 = q_fp8_e5.dequantize()
    sim_fp8_e5 = _cosine_similarity(t.data, rec_fp8_e5.data)
    assert sim_fp8_e5 > 0.99, f"FP8 E5M2 similarity {sim_fp8_e5} is not > 0.99"


# =========================================================================
# 2. Quantized Matrix Multiplication Tests
# =========================================================================

def test_quantized_matrix_multiplication_against_float32():
    """Tests QuantizedTensor @ QuantizedTensor integer GEMM against standard float32 matmul."""
    np.random.seed(7)
    A_data = np.random.uniform(-3.0, 3.0, size=(16, 32)).astype(np.float32)
    B_data = np.random.uniform(-3.0, 3.0, size=(32, 16)).astype(np.float32)

    A = tensor(A_data)
    B = tensor(B_data)

    # Standard float32 ground truth
    C_float = A @ B

    # Quantized matmul (int8 @ int8 rescaled by scale_a * scale_b)
    A_quant = A.quantize(dtype="int8")
    B_quant = B.quantize(dtype="int8")

    C_quant = A_quant @ B_quant

    assert isinstance(C_quant, Tensor)
    assert C_quant.shape == (16, 16)
    assert C_quant.dtype == np.float32

    # Verify cosine similarity with standard float32 matmul > 0.99
    sim = _cosine_similarity(C_float.data, C_quant.data)
    assert sim > 0.99, f"Matmul cosine similarity {sim} is not > 0.99"

    # Verify relative error is small
    mae = np.mean(np.abs(C_float.data - C_quant.data))
    max_val = np.max(np.abs(C_float.data))
    assert (mae / max_val) < 0.05, "Relative MAE is too high"


def test_symmetric_quantized_matmul():
    """Tests symmetric quantized matmul (strictly zero-point=0 GEMM)."""
    np.random.seed(42)
    A = randn((20, 20))
    B = randn((20, 20))

    C_true = A @ B

    A_q = A.quantize(dtype="int8", symmetric=True)
    B_q = B.quantize(dtype="int8", symmetric=True)

    assert A_q.zero_point == 0
    assert B_q.zero_point == 0

    C_q = A_q @ B_q
    sim = _cosine_similarity(C_true.data, C_q.data)
    assert sim > 0.99


def test_mixed_precision_matmul():
    """Tests QuantizedTensor @ Tensor and Tensor @ QuantizedTensor interop."""
    A = tensor([[1.0, 2.0], [3.0, 4.0]])
    B = tensor([[2.0, 0.0], [1.0, 2.0]])
    C_expected = A @ B

    A_q = A.quantize()
    B_q = B.quantize()

    # Quantized @ Float
    C1 = A_q @ B
    np.testing.assert_allclose(C1.data, C_expected.data, atol=0.1)

    # Float @ Quantized
    C2 = A @ B_q
    np.testing.assert_allclose(C2.data, C_expected.data, atol=0.1)


def test_quantized_matmul_shape_mismatch():
    """Verifies TensorShapeMismatchError is raised for incompatible quantized shapes."""
    A_q = tensor(np.ones((4, 8))).quantize()
    B_q = tensor(np.ones((6, 4))).quantize()

    with pytest.raises(TensorShapeMismatchError):
        _ = A_q @ B_q


# =========================================================================
# 3. Compression Metric Tests
# =========================================================================

def test_compression_ratio():
    """Tests compression_ratio() is exactly 4.0x vs float32."""
    t = randn((64, 64))
    q_int8 = t.quantize(dtype="int8")
    assert q_int8.compression_ratio() == 4.0

    q_fp8 = t.quantize(dtype="fp8")
    assert q_fp8.compression_ratio() == 4.0

    # Memory buffer verification
    assert q_int8.qdata.nbytes == t.size * 1  # 1 byte per element
    float32_bytes = t.size * 4                # 4 bytes per element
    assert float32_bytes / q_int8.qdata.nbytes == 4.0


# =========================================================================
# 4. Standalone Functions & Edge Cases
# =========================================================================

def test_standalone_quantize_and_dequantize_functions():
    """Tests top-level quantize() and dequantize() helper functions."""
    data = [1.5, 2.5, -3.5, 4.0]
    q = quantize(data, dtype="int8")
    assert isinstance(q, QuantizedTensor)
    assert q.shape == (4,)

    rec = dequantize(q)
    assert isinstance(rec, Tensor)
    np.testing.assert_allclose(rec.data, data, atol=0.05)


def test_quantized_tensor_repr_and_device():
    """Tests QuantizedTensor string representation and device methods."""
    t = tensor([10.0, 20.0, 30.0])
    q = t.quantize()

    rep = repr(q)
    assert "QuantizedTensor" in rep
    assert "dtype='int8'" in rep
    assert "compression=4.0x" in rep

    q_cpu = q.cpu()
    assert q_cpu.device.device_type == "cpu"
    assert q_cpu.shape == (3,)


# =========================================================================
# 5. C Emitter SIMD & Compiler Flag Tests
# =========================================================================

def transpile_source(source: str) -> str:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return CEmitter().emit(ast)


def test_c_emitter_contains_simd_directives_and_flags():
    """Tests c_emitter.py produces C code with #pragma omp simd and compiler recommendations."""
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
"""
    c_code = transpile_source(source)

    # Verify recommended compiler flags comment
    assert "-O3" in c_code
    assert "-mavx2" in c_code
    assert "-mfma" in c_code
    assert "-fopenmp" in c_code
    assert "-O3 -mavx2 -mfma -fopenmp" in c_code

    # Verify #pragma omp simd directives in emitted C code
    assert "#pragma omp simd" in c_code
    assert "syn_simd_tensor_add" in c_code
    assert "syn_simd_tensor_mul" in c_code
    assert "syn_simd_tensor_matmul" in c_code


def test_c_emitter_explicit_simd_loop_generators():
    """Tests explicit SIMD tensor loop generator methods in CEmitter."""
    emitter = CEmitter()

    add_loop = emitter.emit_simd_tensor_add_loop("vec_a", "vec_b", "vec_out", "N")
    assert "#pragma omp simd" in add_loop
    assert "vec_out[i] = vec_a[i] + vec_b[i];" in add_loop

    mul_loop = emitter.emit_simd_tensor_mul_loop("x", "y", "z", "count")
    assert "#pragma omp simd" in mul_loop
    assert "z[i] = x[i] * y[i];" in mul_loop

    gemm_loop = emitter.emit_simd_tensor_matmul_loop("mat_a", "mat_b", "mat_out", "M", "K", "N")
    assert "#pragma omp simd" in gemm_loop
    assert "#pragma omp simd reduction(+:sum)" in gemm_loop
    assert "mat_out[i * N + j] = sum;" in gemm_loop
