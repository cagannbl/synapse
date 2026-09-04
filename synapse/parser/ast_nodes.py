from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class ASTNode:
    line: int = 0
    column: int = 0


# ==========================================
# Type Annotations & Static Tensor Types
# ==========================================
@dataclass(init=False)
class TensorType(ASTNode):
    dims: tuple[Union[int, str], ...]
    dtype: str

    def __init__(
        self,
        dims: Union[tuple[Union[int, str], ...], list[Union[int, str]], int, str] = (),
        dtype: str = "float32",
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        new_dims = []
        if isinstance(dims, (list, tuple)):
            for d in dims:
                if isinstance(d, int):
                    new_dims.append(d)
                elif isinstance(d, str) and d.isdigit():
                    new_dims.append(int(d))
                else:
                    new_dims.append(d)
        elif isinstance(dims, int):
            new_dims.append(dims)
        elif isinstance(dims, str) and dims.isdigit():
            new_dims.append(int(dims))
        elif dims:
            new_dims.append(dims)
        self.dims = tuple(new_dims)
        self.dtype = dtype

    def to_shape_tuple(self) -> tuple[Union[int, str], ...]:
        return self.dims

    def is_static(self) -> bool:
        """Returns True if all dimensions are static integers."""
        return all(isinstance(d, int) for d in self.dims)

    def is_symbolic(self) -> bool:
        """Returns True if any dimension is a symbolic identifier or string."""
        return any(isinstance(d, str) for d in self.dims)

    def matches_shape(self, shape: tuple[int, ...]) -> bool:
        """
        Checks if an inferred or runtime shape matches this declared TensorType contract.
        Handles both static dimensions (exact equality) and symbolic/dynamic dimensions.
        """
        if len(self.dims) != len(shape):
            return False
        for expected, actual in zip(self.dims, shape):
            if isinstance(expected, int):
                if expected != actual:
                    return False
        return True

    def __str__(self) -> str:
        dims_str = ", ".join(str(d) for d in self.dims)
        if self.dtype and self.dtype != "float32":
            return f"Tensor[{dims_str}, dtype='{self.dtype}']"
        return f"Tensor[{dims_str}]"

    def __repr__(self) -> str:
        return f"TensorType(dims={self.dims!r}, dtype={self.dtype!r})"

    @classmethod
    def from_string(cls, annot_str: str) -> Optional["TensorType"]:
        if not annot_str or not isinstance(annot_str, str):
            return None
        s = annot_str.strip()
        is_upper = s.startswith("Tensor[") and s.endswith("]")
        is_lower = s.startswith("tensor[") and s.endswith("]")
        if not (is_upper or is_lower) and s not in ("Tensor", "tensor"):
            return None
        if s in ("Tensor", "tensor"):
            return cls(dims=(), dtype="float32")
        inner = s[7:-1].strip()
        if not inner:
            return cls(dims=(), dtype="float32")
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        dims: list[Union[int, str]] = []
        dtype = "float32"
        for part in parts:
            if part.startswith("dtype="):
                dtype = part.split("=", 1)[1].strip().strip("'\"")
            elif part in ("float32", "float64", "float16", "int32", "int64", "bool"):
                dtype = part
            else:
                try:
                    dims.append(int(part))
                except ValueError:
                    clean_part = part.strip("'\"")
                    dims.append(clean_part)
        return cls(dims=tuple(dims), dtype=dtype)


ShapeAnnotation = TensorType


@dataclass(init=False)
class TypeAnnotation(ASTNode):
    raw: str
    tensor_type: Optional[TensorType]

    def __init__(
        self,
        raw: str = "",
        tensor_type: Optional[TensorType] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.raw = raw
        self.tensor_type = tensor_type
        if self.tensor_type is None and self.raw:
            self.tensor_type = TensorType.from_string(self.raw)

    def to_shape_tuple(self) -> Optional[tuple[Union[int, str], ...]]:
        return self.tensor_type.to_shape_tuple() if self.tensor_type else None

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return f"TypeAnnotation({self.raw!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.raw == other
        if isinstance(other, TypeAnnotation):
            return self.raw == other.raw
        return False


@dataclass(init=False)
class GenericType(TypeAnnotation):
    base: str
    type_args: list[TypeAnnotation]

    def __init__(
        self,
        base: str = "",
        type_args: Optional[list[Union[str, TypeAnnotation]]] = None,
        line: int = 0,
        column: int = 0,
        raw: str = "",
    ):
        self.base = base
        converted_args: list[TypeAnnotation] = []
        if type_args:
            for arg in type_args:
                if isinstance(arg, TypeAnnotation):
                    converted_args.append(arg)
                elif isinstance(arg, str):
                    converted_args.append(TypeAnnotation(raw=arg, line=line, column=column))
                else:
                    converted_args.append(TypeAnnotation(raw=str(arg), line=line, column=column))
        self.type_args = converted_args

        computed_raw = raw or (
            f"{base}[{', '.join(str(a) for a in self.type_args)}]"
            if self.type_args
            else base
        )
        super().__init__(raw=computed_raw, line=line, column=column)
        if self.base.lower() == "tensor" and self.tensor_type is None:
            self.tensor_type = TensorType.from_string(computed_raw)

    def __repr__(self) -> str:
        return f"GenericType(base={self.base!r}, type_args={self.type_args!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.raw == other
        if isinstance(other, GenericType):
            return self.base == other.base and self.type_args == other.type_args
        if isinstance(other, TypeAnnotation):
            return self.raw == other.raw
        return False


@dataclass(init=False)
class UnionType(TypeAnnotation):
    types: list[TypeAnnotation]

    def __init__(
        self,
        types: Optional[list[Union[str, TypeAnnotation]]] = None,
        line: int = 0,
        column: int = 0,
        raw: str = "",
    ):
        converted_types: list[TypeAnnotation] = []
        if types:
            for t in types:
                if isinstance(t, TypeAnnotation):
                    converted_types.append(t)
                elif isinstance(t, str):
                    converted_types.append(TypeAnnotation(raw=t, line=line, column=column))
                else:
                    converted_types.append(TypeAnnotation(raw=str(t), line=line, column=column))
        self.types = converted_types

        computed_raw = raw or (" | ".join(str(t) for t in self.types))
        super().__init__(raw=computed_raw, line=line, column=column)

    def __repr__(self) -> str:
        return f"UnionType(types={self.types!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.raw == other
        if isinstance(other, UnionType):
            return self.types == other.types
        if isinstance(other, TypeAnnotation):
            return self.raw == other.raw
        return False


# ==========================================
# Option & Result ADT Helper Constructs
# ==========================================
@dataclass
class Some:
    value: Any

    def unwrap(self) -> Any:
        return self.value

    def unwrap_or(self, default: Any) -> Any:
        return self.value

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Some({self.value!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Some):
            return self.value == other.value
        return False


class _NoneOption:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def unwrap(self) -> Any:
        raise ValueError("Called unwrap() on None Option")

    def unwrap_or(self, default: Any) -> Any:
        return default

    def is_some(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "None"

    def __eq__(self, other: Any) -> bool:
        return other is None or isinstance(other, _NoneOption)


NoneOption = _NoneOption()


class Option:
    """Option ADT helper construct with Some(val) and None."""
    Some = Some
    None_ = NoneOption

    @classmethod
    def some(cls, val: Any) -> Some:
        return Some(val)

    @classmethod
    def none(cls) -> _NoneOption:
        return NoneOption

    def __new__(cls, val: Any = None) -> Union[Some, _NoneOption]:
        if val is None:
            return NoneOption
        return Some(val)


@dataclass
class Ok:
    value: Any

    def unwrap(self) -> Any:
        return self.value

    def unwrap_or(self, default: Any) -> Any:
        return self.value

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Ok):
            return self.value == other.value
        return False


@dataclass
class Err:
    error: Any

    def unwrap(self) -> Any:
        raise ValueError(f"Called unwrap() on Err: {self.error}")

    def unwrap_or(self, default: Any) -> Any:
        return default

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Err({self.error!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Err):
            return self.error == other.error
        return False


class Result:
    """Result ADT helper construct with Ok(val) and Err(err)."""
    Ok = Ok
    Err = Err

    @classmethod
    def ok(cls, val: Any) -> Ok:
        return Ok(val)

    @classmethod
    def err(cls, err: Any) -> Err:
        return Err(err)


# ==========================================
# Parameters & Helper Types
# ==========================================
@dataclass
class Param(ASTNode):
    name: str = ""
    type_annot: Optional[Union[str, TypeAnnotation]] = None
    default_value: Optional["Expr"] = None
    tensor_type: Optional[TensorType] = None

    def __post_init__(self):
        if self.tensor_type is None and self.type_annot:
            if isinstance(self.type_annot, TypeAnnotation):
                self.tensor_type = self.type_annot.tensor_type
            elif isinstance(self.type_annot, str):
                self.tensor_type = TensorType.from_string(self.type_annot)
            elif isinstance(self.type_annot, TensorType):
                self.tensor_type = self.type_annot
                self.type_annot = str(self.type_annot)

    def to_shape_tuple(self) -> Optional[tuple[Union[int, str], ...]]:
        return self.tensor_type.to_shape_tuple() if self.tensor_type else None


# ==========================================
# Expressions
# ==========================================
@dataclass
class Expr(ASTNode):
    pass


@dataclass(init=False)
class LiteralExpr(Expr):
    value: Any

    def __init__(self, value: Any = None, line: int = 0, column: int = 0):
        super().__init__(line=line, column=column)
        self.value = value

    def __repr__(self) -> str:
        return f"LiteralExpr({self.value!r})"


@dataclass(init=False)
class IdentifierExpr(Expr):
    name: str

    def __init__(self, name: str = "", line: int = 0, column: int = 0):
        super().__init__(line=line, column=column)
        self.name = name

    def __repr__(self) -> str:
        return f"IdentifierExpr({self.name!r})"


@dataclass(init=False)
class BinaryExpr(Expr):
    left: Expr
    op: str
    right: Expr

    def __init__(
        self,
        left: Optional[Expr] = None,
        op: str = "",
        right: Optional[Expr] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.left = left if left is not None else Expr()
        self.op = op
        self.right = right if right is not None else Expr()

    def __repr__(self) -> str:
        return f"BinaryExpr({self.left!r} {self.op} {self.right!r})"


@dataclass(init=False)
class UnaryExpr(Expr):
    op: str
    operand: Expr

    def __init__(
        self,
        op: str = "",
        operand: Optional[Expr] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.op = op
        self.operand = operand if operand is not None else Expr()

    def __repr__(self) -> str:
        return f"UnaryExpr({self.op} {self.operand!r})"



@dataclass
class PipeExpr(Expr):
    left: Expr = field(default_factory=Expr)
    right: Expr = field(default_factory=Expr)


@dataclass(init=False)
class CallExpr(Expr):
    callee: Expr
    args: list[Expr]
    kwargs: dict[str, Expr]

    def __init__(
        self,
        callee: Optional[Expr] = None,
        args: Optional[list[Expr]] = None,
        kwargs: Optional[dict[str, Expr]] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.callee = callee if callee is not None else Expr()
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def __repr__(self) -> str:
        return f"CallExpr(callee={self.callee!r}, args={self.args!r})"


@dataclass(init=False)
class MemberExpr(Expr):
    target: Expr
    member: str

    def __init__(
        self,
        target: Optional[Expr] = None,
        member: str = "",
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.target = target if target is not None else Expr()
        self.member = member

    def __repr__(self) -> str:
        return f"MemberExpr(target={self.target!r}, member={self.member!r})"



@dataclass
class IndexExpr(Expr):
    target: Expr = field(default_factory=Expr)
    index: Expr = field(default_factory=Expr)


@dataclass
class ListLiteralExpr(Expr):
    elements: list[Expr] = field(default_factory=list)


@dataclass
class DictLiteralExpr(Expr):
    entries: list[tuple[Expr, Expr]] = field(default_factory=list)


@dataclass
class TensorLiteralExpr(Expr):
    data: Expr = field(default_factory=Expr)
    kwargs: dict[str, Expr] = field(default_factory=dict)


@dataclass(init=False)
class TryExpr(Expr):
    expr: Expr

    def __init__(self, expr: Optional[Expr] = None, line: int = 0, column: int = 0):
        super().__init__(line=line, column=column)
        self.expr = expr if expr is not None else Expr()

    def __repr__(self) -> str:
        return f"TryExpr(expr={self.expr!r})"



# ==========================================
# Statements
# ==========================================
@dataclass
class Stmt(ASTNode):
    pass


@dataclass
class EnumDeclStmt(Stmt):
    name: str = ""
    variants: list[str] = field(default_factory=list)


@dataclass
class StructField(ASTNode):
    name: str = ""
    type_annot: Optional[Union[str, TypeAnnotation]] = None
    default_value: Optional["Expr"] = None
    docstring: Optional[str] = None
    tensor_type: Optional[TensorType] = None

    def __post_init__(self):
        if self.tensor_type is None and self.type_annot:
            if isinstance(self.type_annot, TypeAnnotation):
                self.tensor_type = self.type_annot.tensor_type
            elif isinstance(self.type_annot, str):
                self.tensor_type = TensorType.from_string(self.type_annot)
            elif isinstance(self.type_annot, TensorType):
                self.tensor_type = self.type_annot
                self.type_annot = str(self.type_annot)

    def to_shape_tuple(self) -> Optional[tuple[Union[int, str], ...]]:
        return self.tensor_type.to_shape_tuple() if self.tensor_type else None


@dataclass
class StructDef(Stmt):
    name: str = ""
    fields: list[StructField] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class Program(ASTNode):
    statements: list[Stmt] = field(default_factory=list)


@dataclass
class VarDeclStmt(Stmt):
    name: str = ""
    type_annot: Optional[Union[str, TypeAnnotation]] = None
    value: Expr = field(default_factory=Expr)
    is_const: bool = False
    tensor_type: Optional[TensorType] = None

    def __post_init__(self):
        if self.tensor_type is None and self.type_annot:
            if isinstance(self.type_annot, TypeAnnotation):
                self.tensor_type = self.type_annot.tensor_type
            elif isinstance(self.type_annot, str):
                self.tensor_type = TensorType.from_string(self.type_annot)
            elif isinstance(self.type_annot, TensorType):
                self.tensor_type = self.type_annot
                self.type_annot = str(self.type_annot)

    def to_shape_tuple(self) -> Optional[tuple[Union[int, str], ...]]:
        return self.tensor_type.to_shape_tuple() if self.tensor_type else None


@dataclass
class AssignStmt(Stmt):
    target: Expr = field(default_factory=Expr)
    op: str = "="
    value: Expr = field(default_factory=Expr)


@dataclass
class ExprStmt(Stmt):
    expr: Expr = field(default_factory=Expr)


@dataclass(init=False)
class ReturnStmt(Stmt):
    value: Optional[Expr]

    def __init__(
        self,
        value: Optional[Expr] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.value = value

    def __repr__(self) -> str:
        return f"ReturnStmt(value={self.value!r})"



@dataclass
class PassStmt(Stmt):
    pass


@dataclass
class BreakStmt(Stmt):
    pass


@dataclass
class ContinueStmt(Stmt):
    pass


@dataclass
class IfStmt(Stmt):
    condition: Expr = field(default_factory=Expr)
    then_branch: list[Stmt] = field(default_factory=list)
    elif_branches: list[tuple[Expr, list[Stmt]]] = field(default_factory=list)
    else_branch: Optional[list[Stmt]] = None


@dataclass
class WhileStmt(Stmt):
    condition: Expr = field(default_factory=Expr)
    body: list[Stmt] = field(default_factory=list)


@dataclass
class ForStmt(Stmt):
    target: str = ""
    iterable: Expr = field(default_factory=Expr)
    body: list[Stmt] = field(default_factory=list)


@dataclass(init=False)
class MatchCase(ASTNode):
    pattern: Any
    body: list[Stmt]
    guard: Optional[Expr]

    def __init__(
        self,
        pattern: Any = None,
        body: Optional[list[Stmt]] = None,
        guard: Optional[Expr] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.pattern = pattern
        self.body = body if body is not None else []
        self.guard = guard

    def __repr__(self) -> str:
        if self.guard is not None:
            return f"MatchCase(pattern={self.pattern!r}, body={self.body!r}, guard={self.guard!r})"
        return f"MatchCase(pattern={self.pattern!r}, body={self.body!r})"


@dataclass(init=False)
class MatchStmt(Stmt):
    subject: Expr
    cases: list[MatchCase]

    def __init__(
        self,
        subject: Optional[Expr] = None,
        cases: Optional[list[MatchCase]] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line=line, column=column)
        self.subject = subject if subject is not None else Expr()
        self.cases = cases if cases is not None else []

    def __repr__(self) -> str:
        return f"MatchStmt(subject={self.subject!r}, cases={self.cases!r})"



@dataclass
class FunctionDef(Stmt):
    name: str = ""
    params: list[Param] = field(default_factory=list)
    return_type: Optional[Union[str, TypeAnnotation]] = None
    body: list[Stmt] = field(default_factory=list)
    return_tensor_type: Optional[TensorType] = None

    def __post_init__(self):
        if self.return_tensor_type is None and self.return_type:
            if isinstance(self.return_type, TypeAnnotation):
                self.return_tensor_type = self.return_type.tensor_type
            elif isinstance(self.return_type, str):
                self.return_tensor_type = TensorType.from_string(self.return_type)
            elif isinstance(self.return_type, TensorType):
                self.return_tensor_type = self.return_type
                self.return_type = str(self.return_type)

    def to_shape_tuple(self) -> Optional[tuple[Union[int, str], ...]]:
        return self.return_tensor_type.to_shape_tuple() if self.return_tensor_type else None


@dataclass
class ImportStmt(Stmt):
    module_path: list[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_python: bool = False


# ==========================================
# AI-Native Declarations
# ==========================================
@dataclass
class PromptDef(Stmt):
    name: str = ""
    params: list[Param] = field(default_factory=list)
    return_type: Optional[str] = None
    fields: dict[str, Expr] = field(default_factory=dict)


@dataclass
class AgentDef(Stmt):
    name: str = ""
    fields: dict[str, Expr] = field(default_factory=dict)


@dataclass
class ToolDef(Stmt):
    name: str = ""
    params: list[Param] = field(default_factory=list)
    return_type: Optional[str] = None
    body: list[Stmt] = field(default_factory=list)
