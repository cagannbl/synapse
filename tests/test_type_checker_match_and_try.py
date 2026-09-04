import pytest

from synapse.analyzer.type_checker import (
    TypeChecker,
    NonExhaustiveMatchError,
    check_source,
)
from synapse.parser.ast_nodes import (
    Program,
    FunctionDef,
    Param,
    MatchStmt,
    MatchCase,
    TryExpr,
    CallExpr,
    IdentifierExpr,
    LiteralExpr,
    ReturnStmt,
    GenericType,
    TypeAnnotation,
    EnumDeclStmt,
    VarDeclStmt,
    BinaryExpr,
    Some,
    Ok,
    Err,
    NoneOption,
)


# =============================================================================
# 1. Pattern Matching Exhaustiveness Checker (NonExhaustiveMatchError)
# =============================================================================

def test_option_match_missing_none_raises_non_exhaustive():
    """Option with only Some branch must raise NonExhaustiveMatchError."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("int")]))

    match_stmt = MatchStmt(
        subject=IdentifierExpr("opt"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Some"), [IdentifierExpr("x")]),
                body=[ReturnStmt(IdentifierExpr("x"))],
            )
        ],
    )

    with pytest.raises(NonExhaustiveMatchError) as exc_info:
        checker.check_match(match_stmt)
    assert "None" in str(exc_info.value)
    assert "Option" in str(exc_info.value)


def test_option_match_missing_some_raises_non_exhaustive():
    """Option with only None branch must raise NonExhaustiveMatchError."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("str")]))

    match_stmt = MatchStmt(
        subject=IdentifierExpr("opt"),
        cases=[
            MatchCase(
                pattern=IdentifierExpr("None"),
                body=[ReturnStmt(LiteralExpr("default"))],
            )
        ],
    )

    with pytest.raises(NonExhaustiveMatchError) as exc_info:
        checker.check_match(match_stmt)
    assert "Some" in str(exc_info.value)


def test_result_match_missing_err_raises_non_exhaustive():
    """Result with only Ok branch must raise NonExhaustiveMatchError."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
    )

    match_stmt = MatchStmt(
        subject=IdentifierExpr("res"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Ok"), [IdentifierExpr("val")]),
                body=[ReturnStmt(IdentifierExpr("val"))],
            )
        ],
    )

    with pytest.raises(NonExhaustiveMatchError) as exc_info:
        checker.check_match(match_stmt)
    assert "Err" in str(exc_info.value)
    assert "Result" in str(exc_info.value)


def test_result_match_missing_ok_raises_non_exhaustive():
    """Result with only Err branch must raise NonExhaustiveMatchError."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
    )

    match_stmt = MatchStmt(
        subject=IdentifierExpr("res"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Err"), [IdentifierExpr("err")]),
                body=[ReturnStmt(LiteralExpr(-1))],
            )
        ],
    )

    with pytest.raises(NonExhaustiveMatchError) as exc_info:
        checker.check_match(match_stmt)
    assert "Ok" in str(exc_info.value)


def test_enum_match_missing_variant_raises_non_exhaustive():
    """Enum with missing variants must raise NonExhaustiveMatchError."""
    checker = TypeChecker()
    checker.enums["Color"] = {"Red", "Green", "Blue"}
    checker.set_var("c", "Color")

    match_stmt = MatchStmt(
        subject=IdentifierExpr("c"),
        cases=[
            MatchCase(pattern=IdentifierExpr("Red"), body=[ReturnStmt(LiteralExpr(1))]),
            MatchCase(pattern=IdentifierExpr("Green"), body=[ReturnStmt(LiteralExpr(2))]),
        ],
    )

    with pytest.raises(NonExhaustiveMatchError) as exc_info:
        checker.check_match(match_stmt)
    assert "Blue" in str(exc_info.value)
    assert "Color" in str(exc_info.value)


def test_enum_decl_stmt_as_subject_missing_variant_raises():
    """When subject is directly an EnumDeclStmt, missing variants must raise."""
    checker = TypeChecker()
    enum_decl = EnumDeclStmt(name="Priority", variants=["Low", "Medium", "High"])

    match_stmt = MatchStmt(
        subject=enum_decl,
        cases=[
            MatchCase(pattern=IdentifierExpr("Low"), body=[ReturnStmt(LiteralExpr(1))]),
            MatchCase(pattern=IdentifierExpr("Medium"), body=[ReturnStmt(LiteralExpr(2))]),
        ],
    )

    with pytest.raises(NonExhaustiveMatchError) as exc_info:
        checker.check_match(match_stmt)
    assert "High" in str(exc_info.value)


# =============================================================================
# 2. Exhaustive Pattern Matching & Type Inference Verification
# =============================================================================

def test_option_exhaustive_match_type_inference_and_scope():
    """Option with Some and None infers correct inner variable and return type."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("int")]))

    # case Some(x): return x + 10; case None: return 0
    match_stmt = MatchStmt(
        subject=IdentifierExpr("opt"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Some"), [IdentifierExpr("x")]),
                body=[ReturnStmt(BinaryExpr(IdentifierExpr("x"), "+", LiteralExpr(10)))],
            ),
            MatchCase(
                pattern=IdentifierExpr("None"),
                body=[ReturnStmt(LiteralExpr(0))],
            ),
        ],
    )

    res_type = checker.check_match(match_stmt)
    assert res_type == "int"
    # Verify inner variable 'x' is scoped locally and not leaked
    assert checker.get_var_type("x") is None


def test_option_match_with_wildcard_exhaustive():
    """Option with Some and wildcard '_' is exhaustive."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("str")]))

    match_stmt = MatchStmt(
        subject=IdentifierExpr("opt"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Some"), [IdentifierExpr("s")]),
                body=[ReturnStmt(IdentifierExpr("s"))],
            ),
            MatchCase(
                pattern=IdentifierExpr("_"),
                body=[ReturnStmt(LiteralExpr("fallback"))],
            ),
        ],
    )

    res_type = checker.check_match(match_stmt)
    assert res_type == "str"


def test_result_exhaustive_match_type_inference():
    """Result with Ok(val) and Err(err) binds inner types T and E properly."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
    )

    # Both branches reconcile on 'int' return type
    match_stmt = MatchStmt(
        subject=IdentifierExpr("res"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Ok"), [IdentifierExpr("val")]),
                body=[ReturnStmt(BinaryExpr(IdentifierExpr("val"), "*", LiteralExpr(2)))],
            ),
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Err"), [IdentifierExpr("err")]),
                body=[ReturnStmt(LiteralExpr(-1))],
            ),
        ],
    )

    res_type = checker.check_match(match_stmt)
    assert res_type == "int"
    assert checker.get_var_type("val") is None
    assert checker.get_var_type("err") is None


def test_result_match_with_variable_catch_all():
    """Result with Ok and a general variable pattern is exhaustive."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
    )

    match_stmt = MatchStmt(
        subject=IdentifierExpr("res"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Ok"), [IdentifierExpr("val")]),
                body=[ReturnStmt(IdentifierExpr("val"))],
            ),
            MatchCase(
                pattern=IdentifierExpr("other"),
                body=[ReturnStmt(LiteralExpr(0))],
            ),
        ],
    )

    res_type = checker.check_match(match_stmt)
    assert res_type == "int"


def test_enum_exhaustive_match():
    """Enum matching all variants is exhaustive and reconciles return types."""
    checker = TypeChecker()
    checker.enums["State"] = {"Running", "Paused", "Stopped"}
    checker.set_var("st", "State")

    match_stmt = MatchStmt(
        subject=IdentifierExpr("st"),
        cases=[
            MatchCase(pattern=IdentifierExpr("Running"), body=[ReturnStmt(LiteralExpr(1))]),
            MatchCase(pattern=IdentifierExpr("Paused"), body=[ReturnStmt(LiteralExpr(2))]),
            MatchCase(pattern=IdentifierExpr("Stopped"), body=[ReturnStmt(LiteralExpr(3))]),
        ],
    )

    res_type = checker.check_match(match_stmt)
    assert res_type == "int"


def test_match_case_return_types_incompatible_raises_type_error():
    """Match branches with incompatible return types (e.g. int and str) must raise TypeError."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("int")]))

    match_stmt = MatchStmt(
        subject=IdentifierExpr("opt"),
        cases=[
            MatchCase(
                pattern=CallExpr(IdentifierExpr("Some"), [IdentifierExpr("x")]),
                body=[ReturnStmt(LiteralExpr("string_result"))],
            ),
            MatchCase(
                pattern=IdentifierExpr("None"),
                body=[ReturnStmt(LiteralExpr(42))],
            ),
        ],
    )

    with pytest.raises(TypeError) as exc_info:
        checker.check_match(match_stmt)
    assert "reconcile" in str(exc_info.value).lower() or "incompatible" in str(exc_info.value).lower()


# =============================================================================
# 3. '?' TryExpr Operator Invalid Usage (TypeError)
# =============================================================================

def test_try_expr_on_invalid_primitive_types_raises_type_error():
    """? operator applied to non-Result/Option types (int, str, bool) must raise TypeError."""
    checker = TypeChecker()

    # On int
    with pytest.raises(TypeError) as exc1:
        checker.infer_try(TryExpr(LiteralExpr(42)))
    assert "? operator can only be applied to Result or Option types" in str(exc1.value)

    # On str
    with pytest.raises(TypeError) as exc2:
        checker.infer_try(TryExpr(LiteralExpr("hello")))
    assert "? operator can only be applied to Result or Option types" in str(exc2.value)

    # On bool
    with pytest.raises(TypeError) as exc3:
        checker.infer_try(TryExpr(LiteralExpr(True)))
    assert "? operator can only be applied to Result or Option types" in str(exc3.value)


def test_try_expr_inside_incompatible_function_return_type_raises():
    """? used on Result inside a function returning int must raise TypeError."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("str"), TypeAnnotation("int")],
        ),
    )

    fn = FunctionDef(
        name="compute",
        params=[],
        return_type="int",
        body=[],
    )
    checker._register_function_signature(fn)
    checker.current_fn = checker.functions["compute"]

    with pytest.raises(TypeError) as exc_info:
        checker.infer_try(TryExpr(IdentifierExpr("res")))
    assert "return" in str(exc_info.value).lower() or "result" in str(exc_info.value).lower()

    checker.current_fn = None


def test_try_expr_option_inside_result_function_raises():
    """? used on Option inside a function returning Result must raise TypeError."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("int")]))

    fn = FunctionDef(
        name="fetch",
        params=[],
        return_type=GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
        body=[],
    )
    checker._register_function_signature(fn)
    checker.current_fn = checker.functions["fetch"]

    with pytest.raises(TypeError) as exc_info:
        checker.infer_try(TryExpr(IdentifierExpr("opt")))
    assert "Option" in str(exc_info.value)

    checker.current_fn = None


# =============================================================================
# 4. '?' TryExpr Operator Valid Usage & Type Inference
# =============================================================================

def test_try_expr_resolves_inner_type():
    """TryExpr on Result[int, str] resolves to inner type 'int'."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
    )

    resolved = checker.infer_try(TryExpr(IdentifierExpr("res")))
    assert resolved == "int"


def test_try_expr_on_option_resolves_inner_type():
    """TryExpr on Option[float] resolves to inner type 'float'."""
    checker = TypeChecker()
    checker.set_var("opt", GenericType(base="Option", type_args=[TypeAnnotation("float")]))

    resolved = checker.infer_try(TryExpr(IdentifierExpr("opt")))
    assert resolved == "float"


def test_try_expr_in_compatible_function():
    """TryExpr in a function whose return type matches the Result/Option contract passes."""
    checker = TypeChecker()
    checker.set_var(
        "res",
        GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
    )

    fn = FunctionDef(
        name="workflow",
        params=[],
        return_type=GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
        body=[],
    )
    checker._register_function_signature(fn)
    checker.current_fn = checker.functions["workflow"]

    resolved = checker.infer_try(TryExpr(IdentifierExpr("res")))
    assert resolved == "int"
    checker.current_fn = None


# =============================================================================
# 5. Program and End-to-End Source Code Verification
# =============================================================================

def test_check_program_with_match_and_try():
    """End-to-end Program AST with function using TryExpr and MatchStmt."""
    fn_def = FunctionDef(
        name="calc",
        params=[
            Param(
                name="data",
                type_annot=GenericType(
                    base="Result",
                    type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
                ),
            )
        ],
        return_type=GenericType(
            base="Result",
            type_args=[TypeAnnotation("int"), TypeAnnotation("str")],
        ),
        body=[
            VarDeclStmt(
                name="val",
                type_annot="int",
                value=TryExpr(IdentifierExpr("data")),
            ),
            MatchStmt(
                subject=IdentifierExpr("data"),
                cases=[
                    MatchCase(
                        pattern=CallExpr(IdentifierExpr("Ok"), [IdentifierExpr("v")]),
                        body=[ReturnStmt(CallExpr(IdentifierExpr("Ok"), [IdentifierExpr("v")]))],
                    ),
                    MatchCase(
                        pattern=CallExpr(IdentifierExpr("Err"), [IdentifierExpr("e")]),
                        body=[ReturnStmt(CallExpr(IdentifierExpr("Err"), [IdentifierExpr("e")]))],
                    ),
                ],
            ),
        ],
    )

    program = Program(statements=[fn_def])
    checker = TypeChecker()
    result = checker.check_program(program)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_check_source_valid_match_and_try():
    """Synapse source code containing TryExpr ('?') and MatchStmt."""
    code = """
fn fetch_and_process(res: Result[int, str]) -> Result[int, str]:
    let num = res?
    match res:
        case Ok(val):
            return Ok(val + num)
        case Err(err):
            return Err(err)
"""
    result = check_source(code)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_check_source_non_exhaustive_match_raises():
    """Synapse source code with non-exhaustive match raises NonExhaustiveMatchError."""
    code = """
fn test_match(opt: Option[int]) -> int:
    match opt:
        case Some(x):
            return x
"""
    with pytest.raises(NonExhaustiveMatchError):
        check_source(code)


def test_check_source_try_expr_on_int_raises():
    """Synapse source code with '?' on int raises TypeError."""
    code = """
fn bad_try() -> int:
    let x = 42?
    return x
"""
    with pytest.raises(TypeError) as exc_info:
        check_source(code)
    assert "? operator can only be applied to Result or Option types" in str(exc_info.value)
