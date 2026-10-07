"""
Shape contracts on function signatures are rigid: dimensions named in a
signature must line up inside the body, and bare `Tensor` means "shape unknown",
not "scalar". These go through verify_shapes_in_file, which backs both
`synapse verify-shapes` and the pre-run check in `synapse run`.
"""

import pytest

from synapse.analyzer.shape_guard import verify_shapes_in_file


def verify(tmp_path, source: str):
    path = tmp_path / "prog.syn"
    path.write_text(source, encoding="utf-8")
    return verify_shapes_in_file(str(path))


def test_symbolic_inner_dimension_mismatch_is_rejected(tmp_path):
    ok, errors = verify(
        tmp_path,
        "fn f(a: Tensor[N, D], b: Tensor[K, M]) -> Tensor[N, M]:\n"
        "    return a @ b\n",
    )
    assert not ok
    assert any("D != K" in e.message for e in errors)


def test_declared_return_shape_is_enforced(tmp_path):
    ok, errors = verify(
        tmp_path,
        "fn f(a: Tensor[N, D], b: Tensor[D, M]) -> Tensor[M, N]:\n"
        "    return a @ b\n",
    )
    assert not ok
    assert any("Return shape mismatch" in e.message for e in errors)


def test_consistent_symbolic_signature_passes(tmp_path):
    ok, errors = verify(
        tmp_path,
        "fn f(a: Tensor[N, D], b: Tensor[D, M]) -> Tensor[N, M]:\n"
        "    return a @ b\n",
    )
    assert ok, [e.message for e in errors]


def test_bare_tensor_annotation_is_unknown_not_scalar(tmp_path):
    ok, errors = verify(
        tmp_path,
        "fn scale(x: Tensor, w: Tensor) -> Tensor:\n"
        "    return x @ w\n"
        "let y = scale(tensor([[1.0, 2.0]]), tensor([[3.0], [4.0]]))\n",
    )
    assert ok, [e.message for e in errors]


def test_concrete_mismatch_is_reported_once(tmp_path):
    ok, errors = verify(
        tmp_path,
        "fn f(a: Tensor[32, 64], b: Tensor[32, 128]) -> Tensor[32, 128]:\n"
        "    return a @ b\n",
    )
    assert not ok
    assert len([e for e in errors if e.line == 2]) == 1
