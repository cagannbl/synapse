import re
from typing import Iterator, Optional
from synapse.lexer.token import Token, TokenType, KEYWORDS


# Pre-compiled fast-path regex patterns for whitespace, numbers, and identifiers
_WS_RE = re.compile(r"[ \t\r]+")
_FLOAT_RE = re.compile(r"(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)")
_INT_RE = re.compile(r"\d+")
_IDENT_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Combined scanner for fast token matching
_COMBINED_SCANNER = re.compile(
    r"(?P<WS>[ \t\r]+)"
    r"|(?P<FLOAT>\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)"
    r"|(?P<INT>\d+)"
    r"|(?P<IDENT>[a-zA-Z_][a-zA-Z0-9_]*)"
)


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Lexer error at {line}:{column} - {message}")
        self.line = line
        self.column = column


class Lexer:
    has_fast_path: bool = True

    def __init__(self, source: str, use_fast_path: bool = True):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1
        self.use_fast_path = use_fast_path

        self.indent_stack: list[int] = [0]
        self.bracket_level = 0  # ( [ { içindeyken girintiler yoksayılır
        self.at_line_start = True

    def peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < self.length:
            return self.source[idx]
        return ""

    def advance(self) -> str:
        if self.pos >= self.length:
            return ""
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
            self.at_line_start = True
        else:
            self.column += 1
        return ch

    def match(self, expected: str) -> bool:
        if self.pos >= self.length or self.source[self.pos] != expected:
            return False
        self.advance()
        return True

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []

        while self.pos < self.length:
            # Satır başındaysak ve parantez içinde değilsek girintiyi hesapla
            if self.at_line_start and self.bracket_level == 0:
                indent_tokens = self._handle_indentation()
                tokens.extend(indent_tokens)
                if self.pos >= self.length:
                    break

            ch = self.peek()

            # Boşluklar (satır içi) - Fast-path scanner
            if ch in " \t\r":
                if self.use_fast_path:
                    m = _WS_RE.match(self.source, self.pos)
                    if m:
                        count = len(m.group(0))
                        self.pos += count
                        self.column += count
                        continue
                self.advance()
                continue

            # Yorum satırı (#)
            if ch == "#":
                while self.peek() and self.peek() != "\n":
                    self.advance()
                continue

            # Satır sonu
            if ch == "\n":
                line_no, col_no = self.line, self.column
                self.advance()
                if self.bracket_level == 0:
                    # Arka arkaya gelen NEWLINE'ları teke indir
                    if tokens and tokens[-1].type != TokenType.NEWLINE:
                        tokens.append(Token(TokenType.NEWLINE, "\n", line_no, col_no))
                continue

            # İki karakterli ve tek karakterli semboller
            start_line, start_col = self.line, self.column

            # Pipeline |> or Union / Bitwise |
            if ch == "|":
                if self.peek(1) == ">":
                    self.advance()
                    self.advance()
                    tokens.append(Token(TokenType.PIPE, "|>", start_line, start_col))
                    continue
                else:
                    self.advance()
                    tokens.append(Token(TokenType.BAR, "|", start_line, start_col))
                    continue

            # Arrow ->
            if ch == "-" and self.peek(1) == ">":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.ARROW, "->", start_line, start_col))
                continue

            # Fat Arrow =>
            if ch == "=" and self.peek(1) == ">":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.FAT_ARROW, "=>", start_line, start_col))
                continue

            # Matrix Multiply Equal @= veya @
            if ch == "@":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.AT_EQUAL, "@=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.AT, "@", start_line, start_col))
                continue

            # Matched compound operators
            if ch == "=":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.EQUAL_EQUAL, "==", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.EQUAL, "=", start_line, start_col))
                continue

            if ch == "!":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.BANG_EQUAL, "!=", start_line, start_col))
                else:
                    raise LexerError(f"Unexpected character '!' (did you mean '!=')?", start_line, start_col)
                continue

            if ch == "<":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.LESS_EQUAL, "<=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.LESS, "<", start_line, start_col))
                continue

            if ch == ">":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.GREATER_EQUAL, ">=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.GREATER, ">", start_line, start_col))
                continue

            if ch == "+":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.PLUS_EQUAL, "+=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.PLUS, "+", start_line, start_col))
                continue

            if ch == "-":
                self.advance()
                if self.match("="):
                    tokens.append(Token(TokenType.MINUS_EQUAL, "-=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.MINUS, "-", start_line, start_col))
                continue

            if ch == "*":
                self.advance()
                if self.match("*"):
                    tokens.append(Token(TokenType.DOUBLE_STAR, "**", start_line, start_col))
                elif self.match("="):
                    tokens.append(Token(TokenType.STAR_EQUAL, "*=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.STAR, "*", start_line, start_col))
                continue

            if ch == "/":
                self.advance()
                if self.match("/"):
                    tokens.append(Token(TokenType.DOUBLE_SLASH, "//", start_line, start_col))
                elif self.match("="):
                    tokens.append(Token(TokenType.SLASH_EQUAL, "/=", start_line, start_col))
                else:
                    tokens.append(Token(TokenType.SLASH, "/", start_line, start_col))
                continue

            if ch == "%":
                self.advance()
                tokens.append(Token(TokenType.PERCENT, "%", start_line, start_col))
                continue

            # Parantezler & Ayırıcılar
            if ch == "(":
                self.advance()
                self.bracket_level += 1
                tokens.append(Token(TokenType.LPAREN, "(", start_line, start_col))
                continue
            if ch == ")":
                self.advance()
                self.bracket_level = max(0, self.bracket_level - 1)
                tokens.append(Token(TokenType.RPAREN, ")", start_line, start_col))
                continue
            if ch == "[":
                self.advance()
                self.bracket_level += 1
                tokens.append(Token(TokenType.LBRACKET, "[", start_line, start_col))
                continue
            if ch == "]":
                self.advance()
                self.bracket_level = max(0, self.bracket_level - 1)
                tokens.append(Token(TokenType.RBRACKET, "]", start_line, start_col))
                continue
            if ch == "{":
                self.advance()
                self.bracket_level += 1
                tokens.append(Token(TokenType.LBRACE, "{", start_line, start_col))
                continue
            if ch == "}":
                self.advance()
                self.bracket_level = max(0, self.bracket_level - 1)
                tokens.append(Token(TokenType.RBRACE, "}", start_line, start_col))
                continue
            if ch == ":":
                self.advance()
                tokens.append(Token(TokenType.COLON, ":", start_line, start_col))
                continue
            if ch == ",":
                self.advance()
                tokens.append(Token(TokenType.COMMA, ",", start_line, start_col))
                continue
            if ch == ".":
                self.advance()
                tokens.append(Token(TokenType.DOT, ".", start_line, start_col))
                continue
            if ch == ";":
                self.advance()
                tokens.append(Token(TokenType.SEMICOLON, ";", start_line, start_col))
                continue
            if ch == "?":
                self.advance()
                tokens.append(Token(TokenType.QUESTION, "?", start_line, start_col))
                continue

            # String literals ("..." veya '...') ya da f-strings (f"...")
            if ch in ('"', "'") or (ch == 'f' and self.peek(1) in ('"', "'")):
                tokens.append(self._read_string())
                continue

            # Sayılar (Int ve Float)
            if ch.isdigit():
                tokens.append(self._read_number())
                continue

            # Tanımlayıcılar (Identifiers) ve Anahtar Kelimeler
            if ch.isalpha() or ch == "_":
                tokens.append(self._read_identifier())
                continue

            # Bilinmeyen karakter
            unknown = self.advance()
            raise LexerError(f"Unexpected character: {unknown!r}", start_line, start_col)

        # Kalan girintileri temizle (DEDENT)
        if tokens and tokens[-1].type != TokenType.NEWLINE:
            tokens.append(Token(TokenType.NEWLINE, "\n", self.line, self.column))

        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            tokens.append(Token(TokenType.DEDENT, "", self.line, self.column))

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens

    def _handle_indentation(self) -> list[Token]:
        tokens: list[Token] = []
        indent_spaces = 0
        start_pos = self.pos
        start_line = self.line

        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch == " ":
                indent_spaces += 1
                self.advance()
            elif ch == "\t":
                indent_spaces += 4  # tab = 4 boşluk
                self.advance()
            elif ch == "#":
                # Yorum satırı, sonuna kadar atla
                while self.pos < self.length and self.source[self.pos] != "\n":
                    self.advance()
                if self.pos < self.length and self.source[self.pos] == "\n":
                    self.advance()
                indent_spaces = 0
            elif ch == "\n":
                # Boş satır, atla
                self.advance()
                indent_spaces = 0
            else:
                break

        # Dosya sonuna geldiysek girinti üretme
        if self.pos >= self.length:
            self.at_line_start = False
            return tokens

        self.at_line_start = False
        current_indent = self.indent_stack[-1]

        if indent_spaces > current_indent:
            self.indent_stack.append(indent_spaces)
            tokens.append(Token(TokenType.INDENT, indent_spaces, start_line, 1))
        elif indent_spaces < current_indent:
            while self.indent_stack and indent_spaces < self.indent_stack[-1]:
                self.indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, indent_spaces, start_line, 1))

            if not self.indent_stack or self.indent_stack[-1] != indent_spaces:
                raise LexerError(
                    f"Unindent does not match any outer indentation level (expected one of {self.indent_stack}, got {indent_spaces})",
                    start_line,
                    1
                )

        return tokens

    def _read_string(self) -> Token:
        start_line, start_col = self.line, self.column
        is_fstring = False

        if self.peek() == "f":
            is_fstring = True
            self.advance()

        quote = self.advance()  # ' veya "
        is_triple = False
        if self.peek() == quote and self.peek(1) == quote:
            is_triple = True
            self.advance()
            self.advance()

        chars: list[str] = []

        while self.pos < self.length:
            ch = self.advance()
            if is_triple and ch == quote and self.peek() == quote and self.peek(1) == quote:
                self.advance()
                self.advance()
                self.at_line_start = False
                val = "".join(chars)
                return Token(TokenType.STRING, val, start_line, start_col)
            elif not is_triple and ch == quote:
                self.at_line_start = False
                val = "".join(chars)
                return Token(TokenType.STRING, val, start_line, start_col)
            elif ch == "\\":
                if self.pos >= self.length:
                    break
                next_ch = self.advance()
                if next_ch == "n":
                    chars.append("\n")
                elif next_ch == "t":
                    chars.append("\t")
                elif next_ch == "r":
                    chars.append("\r")
                elif next_ch == "\\":
                    chars.append("\\")
                elif next_ch == quote:
                    chars.append(quote)
                else:
                    chars.append(next_ch)
            elif ch == "\r":
                if self.peek() == "\n":
                    self.advance()
                if not is_triple:
                    raise LexerError("Unterminated string literal before newline", start_line, start_col)
                chars.append("\n")
            elif ch == "\n":
                if not is_triple:
                    raise LexerError("Unterminated string literal before newline", start_line, start_col)
                chars.append("\n")
            else:
                chars.append(ch)

        raise LexerError("Unterminated string literal at end of file", start_line, start_col)

    def scan_fast(self) -> Optional[Token]:
        """
        Fast-path scanner for whitespace, numbers, and identifiers.
        Advances lexer position and returns Token or None if skipped (e.g. whitespace).
        """
        if not self.use_fast_path or self.pos >= self.length:
            return None
        m = _COMBINED_SCANNER.match(self.source, self.pos)
        if not m:
            return None
        kind = m.lastgroup
        val = m.group(0)
        start_line, start_col = self.line, self.column
        length = len(val)

        if kind == "WS":
            self.pos += length
            self.column += length
            return None
        elif kind == "FLOAT":
            self.pos += length
            self.column += length
            return Token(TokenType.FLOAT, float(val), start_line, start_col)
        elif kind == "INT":
            self.pos += length
            self.column += length
            return Token(TokenType.INT, int(val), start_line, start_col)
        elif kind == "IDENT":
            if val == "f" and self.peek(1) in ('"', "'"):
                return None
            self.pos += length
            self.column += length
            if val in KEYWORDS:
                kw_type = KEYWORDS[val]
                if kw_type == TokenType.BOOL:
                    return Token(TokenType.BOOL, val.lower() == "true", start_line, start_col)
                if kw_type == TokenType.NONE:
                    return Token(TokenType.NONE, None, start_line, start_col)
                return Token(kw_type, val, start_line, start_col)
            return Token(TokenType.IDENTIFIER, val, start_line, start_col)
        return None

    def _read_number(self) -> Token:
        start_line, start_col = self.line, self.column
        if self.use_fast_path:
            m_float = _FLOAT_RE.match(self.source, self.pos)
            if m_float:
                num_str = m_float.group(0)
                length = len(num_str)
                self.pos += length
                self.column += length
                return Token(TokenType.FLOAT, float(num_str), start_line, start_col)
            m_int = _INT_RE.match(self.source, self.pos)
            if m_int:
                num_str = m_int.group(0)
                length = len(num_str)
                self.pos += length
                self.column += length
                return Token(TokenType.INT, int(num_str), start_line, start_col)

        chars: list[str] = []
        is_float = False

        while self.pos < self.length and (self.peek().isdigit() or self.peek() == "."):
            if self.peek() == ".":
                if is_float:
                    break  # İkinci bir nokta
                # Bir sonraki karakter de rakam mı veya metod çağrısı mı?
                if not self.peek(1).isdigit():
                    break
                is_float = True
            chars.append(self.advance())

        # Scientific notation (1e-4, 2E+3)
        if self.pos < self.length and self.peek() in ("e", "E"):
            is_float = True
            chars.append(self.advance())
            if self.pos < self.length and self.peek() in ("+", "-"):
                chars.append(self.advance())
            while self.pos < self.length and self.peek().isdigit():
                chars.append(self.advance())

        num_str = "".join(chars)
        if is_float:
            return Token(TokenType.FLOAT, float(num_str), start_line, start_col)
        return Token(TokenType.INT, int(num_str), start_line, start_col)

    def _read_identifier(self) -> Token:
        start_line, start_col = self.line, self.column
        if self.use_fast_path:
            m = _IDENT_RE.match(self.source, self.pos)
            if m:
                name = m.group(0)
                length = len(name)
                self.pos += length
                self.column += length
            else:
                chars: list[str] = []
                while self.pos < self.length and (self.peek().isalnum() or self.peek() == "_"):
                    chars.append(self.advance())
                name = "".join(chars)
        else:
            chars: list[str] = []
            while self.pos < self.length and (self.peek().isalnum() or self.peek() == "_"):
                chars.append(self.advance())
            name = "".join(chars)

        # Keyword kontrolü
        if name in KEYWORDS:
            token_type = KEYWORDS[name]
            if token_type == TokenType.BOOL:
                val = (name.lower() == "true")
                return Token(TokenType.BOOL, val, start_line, start_col)
            if token_type == TokenType.NONE:
                return Token(TokenType.NONE, None, start_line, start_col)
            return Token(token_type, name, start_line, start_col)

        return Token(TokenType.IDENTIFIER, name, start_line, start_col)
