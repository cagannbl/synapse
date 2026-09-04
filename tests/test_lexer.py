import pytest
from synapse.lexer.lexer import Lexer, LexerError
from synapse.lexer.token import TokenType


def test_basic_tokens():
    source = "let x = 42\nconst PI = 3.14"
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    expected_types = [
        TokenType.LET, TokenType.IDENTIFIER, TokenType.EQUAL, TokenType.INT, TokenType.NEWLINE,
        TokenType.CONST, TokenType.IDENTIFIER, TokenType.EQUAL, TokenType.FLOAT, TokenType.NEWLINE,
        TokenType.EOF
    ]
    assert [t.type for t in tokens] == expected_types
    assert tokens[3].value == 42
    assert tokens[8].value == 3.14


def test_ai_and_matrix_operators():
    source = "let C = A @ B |> relu"
    tokens = Lexer(source).tokenize()

    assert tokens[0].type == TokenType.LET
    assert tokens[1].value == "C"
    assert tokens[3].value == "A"
    assert tokens[4].type == TokenType.AT
    assert tokens[5].value == "B"
    assert tokens[6].type == TokenType.PIPE
    assert tokens[7].value == "relu"


def test_indentation_block():
    source = """
fn add(a, b):
    let result = a + b
    return result

let x = 10
"""
    tokens = Lexer(source).tokenize()
    types = [t.type for t in tokens]

    assert TokenType.FN in types
    assert TokenType.INDENT in types
    assert TokenType.DEDENT in types
    assert TokenType.RETURN in types
    assert types[-1] == TokenType.EOF


def test_bracket_nesting_ignores_newlines():
    source = """
let matrix = [
    [1.0, 2.0],
    [3.0, 4.0]
]
let y = 5
"""
    tokens = Lexer(source).tokenize()
    types = [t.type for t in tokens]

    # Köşeli parantez içinde INDENT veya DEDENT üretilmemeli
    assert TokenType.INDENT not in types
    assert TokenType.DEDENT not in types
    assert TokenType.LBRACKET in types
    assert TokenType.RBRACKET in types


def test_prompt_and_tool_keywords():
    source = """
prompt classify(text):
    system: "Classify this"
    user: text
"""
    tokens = Lexer(source).tokenize()
    types = [t.type for t in tokens]

    assert types[0] == TokenType.IDENTIFIER
    assert tokens[0].value == "prompt"
    assert TokenType.INDENT in types
    assert TokenType.DEDENT in types


def test_invalid_indentation_raises_error():
    source = """
let a = 1
  let b = 2
    let c = 3
 let d = 4
"""
    with pytest.raises(LexerError):
        Lexer(source).tokenize()
