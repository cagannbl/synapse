from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    # End of file / Whitespace tokens
    EOF = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()

    # Identifiers & Literals
    IDENTIFIER = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    NONE = auto()

    # Keywords
    LET = auto()
    CONST = auto()
    FN = auto()
    RETURN = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    PASS = auto()
    BREAK = auto()
    CONTINUE = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    ENUM = auto()
    STRUCT = auto()

    MATCH = auto()
    CASE = auto()

    # AI-Native Keywords (Retained for AST/parser compatibility)
    TENSOR = auto()
    PROMPT = auto()
    AGENT = auto()
    TOOL = auto()
    GRAD = auto()

    # Logical Operators
    AND = auto()
    OR = auto()
    NOT = auto()

    # Arithmetic & Tensor Operators
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    DOUBLE_SLASH = auto()   # //
    PERCENT = auto()        # %
    DOUBLE_STAR = auto()    # **
    AT = auto()             # @ (Matrix Multiplication)
    PIPE = auto()           # |> (Pipeline Operator)
    BAR = auto()            # | (Union / Bitwise OR Operator)
    ARROW = auto()          # -> (Return type arrow)
    FAT_ARROW = auto()      # =>

    # Assignment Operators
    EQUAL = auto()          # =
    PLUS_EQUAL = auto()     # +=
    MINUS_EQUAL = auto()    # -=
    STAR_EQUAL = auto()     # *=
    SLASH_EQUAL = auto()    # /=
    AT_EQUAL = auto()       # @=

    # Comparison Operators
    EQUAL_EQUAL = auto()    # ==
    BANG_EQUAL = auto()     # !=
    LESS = auto()           # <
    LESS_EQUAL = auto()     # <=
    GREATER = auto()        # >
    GREATER_EQUAL = auto()  # >=

    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    COLON = auto()          # :
    COMMA = auto()          # ,
    DOT = auto()            # .
    SEMICOLON = auto()      # ;
    QUESTION = auto()       # ? (Try / Error propagation operator)


KEYWORDS: dict[str, TokenType] = {
    "let": TokenType.LET,
    "const": TokenType.CONST,
    "fn": TokenType.FN,
    "def": TokenType.FN,  # Python uyumluluğu için def de desteklenir
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "elif": TokenType.ELIF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "pass": TokenType.PASS,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "as": TokenType.AS,
    "enum": TokenType.ENUM,
    "struct": TokenType.STRUCT,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "true": TokenType.BOOL,
    "false": TokenType.BOOL,
    "True": TokenType.BOOL,
    "False": TokenType.BOOL,
    "none": TokenType.NONE,
    "None": TokenType.NONE,
    "match": TokenType.MATCH,
    "case": TokenType.CASE,
}


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
