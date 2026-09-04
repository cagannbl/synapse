import os
import tempfile
import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import (
    Program, VarDeclStmt, EnumDeclStmt, GenericType, UnionType,
    TypeAnnotation, TensorType, Some, NoneOption, Option, Ok, Err, Result
)
from synapse.core.type_checker import (
    StaticTypeChecker, check_source, handle_check_cli, TypeCheckResult
)


def parse_code(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# =============================================================================
# 1. Enum Definition and Parsing Tests
# =============================================================================

def test_parse_enum_single_line():
    source = "enum Color: Red, Green, Blue"
    ast = parse_code(source)
    assert len(ast.statements) == 1
    enum_stmt = ast.statements[0]
    assert isinstance(enum_stmt, EnumDeclStmt)
    assert enum_stmt.name == "Color"
    assert enum_stmt.variants == ["Red", "Green", "Blue"]


def test_parse_enum_multiline():
    source = """
enum Status:
    Active
    Inactive
    Pending
"""
    ast = parse_code(source)
    assert len(ast.statements) == 1
    enum_stmt = ast.statements[0]
    assert isinstance(enum_stmt, EnumDeclStmt)
    assert enum_stmt.name == "Status"
    assert enum_stmt.variants == ["Active", "Inactive", "Pending"]


def test_parse_enum_multiline_with_commas_and_values():
    source = """
enum Priority:
    High = 1,
    Medium = 2,
    Low = 3
"""
    ast = parse_code(source)
    assert len(ast.statements) == 1
    enum_stmt = ast.statements[0]
    assert isinstance(enum_stmt, EnumDeclStmt)
    assert enum_stmt.name == "Priority"
    assert enum_stmt.variants == ["High", "Medium", "Low"]


def test_enum_variant_type_checking():
    source = """
enum Direction: North, South, East, West
let d1: Direction = Direction.North
let d2: Direction = Direction.East
"""
    res = check_source(source)
    assert res.is_valid is True
    assert len(res.errors) == 0


def test_enum_invalid_variant_rejected():
    source = """
enum Direction: North, South, East, West
let d: Direction = Direction.Upward
"""
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    err = res.errors[0]
    assert "Invalid variant 'Upward' for enum 'Direction'" in err.message
    assert "North" in err.message


def test_enum_mismatched_assignment_rejected():
    source = """
enum Direction: North, South
let d: Direction = "North"
"""
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert "Type contract violation" in res.errors[0].message


# =============================================================================
# 2. Generic Type Annotations Tests
# =============================================================================

def test_parse_generic_type_annotations():
    source = """
let l: List[int] = [1, 2, 3]
let opt: Option[str] = Some("hello")
let res: Result[int, str] = Ok(42)
"""
    ast = parse_code(source)
    assert len(ast.statements) == 3

    l_stmt = ast.statements[0]
    assert isinstance(l_stmt.type_annot, GenericType)
    assert l_stmt.type_annot.base == "List"
    assert len(l_stmt.type_annot.type_args) == 1
    assert l_stmt.type_annot.type_args[0].raw == "int"

    opt_stmt = ast.statements[1]
    assert isinstance(opt_stmt.type_annot, GenericType)
    assert opt_stmt.type_annot.base == "Option"
    assert opt_stmt.type_annot.type_args[0].raw == "str"

    res_stmt = ast.statements[2]
    assert isinstance(res_stmt.type_annot, GenericType)
    assert res_stmt.type_annot.base == "Result"
    assert len(res_stmt.type_annot.type_args) == 2
    assert res_stmt.type_annot.type_args[0].raw == "int"
    assert res_stmt.type_annot.type_args[1].raw == "str"


def test_generic_list_type_check():
    valid_source = "let numbers: List[int] = [1, 2, 3]"
    res_valid = check_source(valid_source)
    assert res_valid.is_valid is True

    invalid_source = "let numbers: List[int] = ['a', 'b', 'c']"
    res_invalid = check_source(invalid_source)
    assert res_invalid.is_valid is False
    assert "Type contract violation" in res_invalid.errors[0].message


def test_generic_option_type_check():
    valid_source = """
let o1: Option[int] = Some(42)
let o2: Option[int] = None
"""
    res_valid = check_source(valid_source)
    assert res_valid.is_valid is True

    invalid_source = "let o3: Option[int] = Some('not_an_int')"
    res_invalid = check_source(invalid_source)
    assert res_invalid.is_valid is False
    assert "Type contract violation" in res_invalid.errors[0].message


def test_generic_result_type_check():
    valid_source = """
let r1: Result[int, str] = Ok(200)
let r2: Result[int, str] = Err("Not Found")
"""
    res_valid = check_source(valid_source)
    assert res_valid.is_valid is True

    invalid_source1 = "let r3: Result[int, str] = Ok('not_int')"
    res_invalid1 = check_source(invalid_source1)
    assert res_invalid1.is_valid is False

    invalid_source2 = "let r4: Result[int, str] = Err(500)"
    res_invalid2 = check_source(invalid_source2)
    assert res_invalid2.is_valid is False


def test_generic_dict_type_check():
    valid_source = "let scores: Dict[str, int] = {'Alice': 100, 'Bob': 90}"
    assert check_source(valid_source).is_valid is True

    invalid_source = "let scores: Dict[str, int] = {'Alice': 'high'}"
    assert check_source(invalid_source).is_valid is False


# =============================================================================
# 3. Union Type Annotations Tests
# =============================================================================

def test_parse_union_type_annotation():
    source = "let val: int | str = 42"
    ast = parse_code(source)
    assert len(ast.statements) == 1
    stmt = ast.statements[0]
    assert isinstance(stmt.type_annot, UnionType)
    assert len(stmt.type_annot.types) == 2
    assert stmt.type_annot.types[0].raw == "int"
    assert stmt.type_annot.types[1].raw == "str"


def test_union_type_checking():
    source = """
let u1: int | str = 42
let u2: int | str = "hello"
"""
    res = check_source(source)
    assert res.is_valid is True
    assert len(res.errors) == 0


def test_union_type_violation():
    source = "let u: int | str = 3.14"
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert "Type contract violation" in res.errors[0].message


def test_union_with_option_and_none():
    source = """
let val1: Option[int] | None = Some(10)
let val2: Option[int] | None = None
let val3: Option[int] | None = "error"
"""
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert res.errors[0].line == 4


# =============================================================================
# 4. Option & Result Helper Constructs Tests
# =============================================================================

def test_option_helper_constructs():
    some_val = Some(42)
    assert some_val.unwrap() == 42
    assert some_val.unwrap_or(0) == 42
    assert some_val.is_some() is True
    assert some_val.is_none() is False

    none_val = NoneOption
    assert none_val.is_some() is False
    assert none_val.is_none() is True
    assert none_val.unwrap_or(99) == 99
    with pytest.raises(ValueError, match="unwrap"):
        none_val.unwrap()

    opt1 = Option(100)
    assert opt1 == Some(100)
    opt2 = Option(None)
    assert opt2 == NoneOption
    assert Option.some("test") == Some("test")
    assert Option.none() == NoneOption


def test_result_helper_constructs():
    ok_val = Ok(10)
    assert ok_val.unwrap() == 10
    assert ok_val.unwrap_or(0) == 10
    assert ok_val.is_ok() is True
    assert ok_val.is_err() is False

    err_val = Err("network failure")
    assert err_val.is_ok() is False
    assert err_val.is_err() is True
    assert err_val.unwrap_or(0) == 0
    with pytest.raises(ValueError, match="network failure"):
        err_val.unwrap()

    assert Result.ok(50) == Ok(50)
    assert Result.err("err") == Err("err")


# =============================================================================
# 5. Option & Result Safe Unwrapping Tests
# =============================================================================

def test_safe_unwrap_on_known_none():
    source = """
let x: Option[int] = None
let y = x.unwrap()
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Unsafe unwrap: Variable 'x' is known to be None" in e.message for e in res.errors)


def test_safe_unwrap_on_known_err():
    source = """
let r: Result[int, str] = Err("fail")
let y = r.unwrap()
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Unsafe unwrap: Variable 'r' is known to be Err" in e.message for e in res.errors)


def test_safe_unwrap_on_non_option():
    source = """
let n: int = 42
let y = n.unwrap()
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Cannot call 'unwrap()' on type 'int'" in e.message for e in res.errors)


def test_reject_direct_binary_operation_on_option():
    source = """
let opt: Option[int] = Some(10)
let res = opt + 5
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Cannot perform binary operation '+' directly on 'Option[int]'" in e.message for e in res.errors)


def test_unwrap_or_type_mismatch():
    source = """
let opt: Option[int] = Some(10)
let val = opt.unwrap_or("wrong_default")
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Type contract violation in 'unwrap_or'" in e.message for e in res.errors)


# =============================================================================
# 6. Static Type Checking with Line Pointers and Diagnostics
# =============================================================================

def test_type_contract_violation_exact_line_and_pointer():
    source = """let a: int = 10
let b: int = "string_literal"
let c: int = 30"""
    res = check_source(source, filepath="test_prog.syn")
    assert res.is_valid is False
    assert len(res.errors) == 1

    err = res.errors[0]
    assert err.line == 2
    assert err.source_line == 'let b: int = "string_literal"'
    assert "^" in err.pointer
    assert "Variable 'b'" in err.message
    assert err.error_type == "TypeContractViolationError"


def test_const_reassignment_violation():
    source = """
const MAX_LIMIT: int = 100
MAX_LIMIT = 200
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Cannot reassign to constant variable 'MAX_LIMIT'" in e.message for e in res.errors)


def test_variable_reassignment_type_mismatch():
    source = """
let x: int = 10
x = "changed_to_str"
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Type contract violation: Reassignment to variable 'x'" in e.message for e in res.errors)


# =============================================================================
# 7. Function Parameter and Return Type Tests
# =============================================================================

def test_function_parameter_type_checking():
    source = """
fn multiply(x: int, y: int) -> int:
    return x * y

let valid_call = multiply(4, 5)
let invalid_call = multiply("hello", 5)
"""
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert "Argument type mismatch for parameter 'x' of function 'multiply'" in res.errors[0].message


def test_function_return_type_checking():
    source = """
fn get_number() -> int:
    return "this is not an int"
"""
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert "Return type mismatch in function 'get_number'" in res.errors[0].message


# =============================================================================
# 8. TensorType Contracts and Operations Tests
# =============================================================================

def test_tensor_contract_shape_verification():
    source = """
let A: Tensor[2, 3] = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
let B: Tensor[2, 2] = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
"""
    res = check_source(source)
    assert res.is_valid is False
    assert len(res.errors) == 1
    assert "Variable 'B' declared with contract Tensor[2, 2]" in res.errors[0].message
    assert "assigned expression of shape (2, 3)" in res.errors[0].message


def test_tensor_matmul_inner_dimension_mismatch():
    source = """
let A: Tensor[2, 3] = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
let C = A @ A
"""
    res = check_source(source)
    assert res.is_valid is False
    assert any("Matrix multiplication shape mismatch" in e.message for e in res.errors)


# =============================================================================
# 9. Valid Typed Programs Passing Without Errors
# =============================================================================

def test_valid_comprehensive_program():
    source = """
enum State: Idle, Running, Finished

fn process_item(item: int) -> Option[int]:
    if item > 0:
        return Some(item * 2)
    return None

let current_state: State = State.Running
let input_val: int = 21
let maybe_result: Option[int] = process_item(input_val)
let unwrapped_result: int = maybe_result.unwrap_or(0)

let weight: Tensor[2, 3] = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
let x_input: Tensor[3, 1] = tensor([[1.0], [2.0], [3.0]])
let y_output = weight @ x_input
"""
    res = check_source(source)
    assert res.is_valid is True
    assert len(res.errors) == 0


# =============================================================================
# 10. handle_check_cli Tests
# =============================================================================

def test_handle_check_cli_valid_file():
    code = """
enum Mode: Fast, Slow
let current: Mode = Mode.Fast
let value: int = 100
let opt: Option[int] = Some(value)
"""
    with tempfile.NamedTemporaryFile(suffix=".syn", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        valid_path = f.name

    try:
        exit_code = handle_check_cli(valid_path, verbose=True)
        assert exit_code == 0
    finally:
        os.remove(valid_path)


def test_handle_check_cli_invalid_file():
    code = """
let x: int = "should be int"
"""
    with tempfile.NamedTemporaryFile(suffix=".syn", delete=False, mode="w", encoding="utf-8") as f:
        f.write(code)
        invalid_path = f.name

    try:
        exit_code = handle_check_cli(invalid_path, verbose=False)
        assert exit_code == 1
    finally:
        os.remove(invalid_path)


def test_handle_check_cli_nonexistent_file():
    exit_code = handle_check_cli("this_file_does_not_exist_at_all.syn")
    assert exit_code == 1
