from __future__ import annotations

import os
from typing import Optional, Any, TYPE_CHECKING
from synapse.lexer.token import Token, TokenType
from synapse.lexer.lexer import Lexer
if TYPE_CHECKING:
    from synapse.compiler.cache import ASTCache
from synapse.parser.ast_nodes import (
    Program, Stmt, Expr, LiteralExpr, IdentifierExpr, BinaryExpr, UnaryExpr,
    PipeExpr, CallExpr, MemberExpr, IndexExpr, ListLiteralExpr, DictLiteralExpr,
    TensorLiteralExpr, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, PassStmt,
    BreakStmt, ContinueStmt, IfStmt, WhileStmt, ForStmt, FunctionDef, ImportStmt,
    PromptDef, AgentDef, ToolDef, Param, EnumDeclStmt, GenericType, UnionType,
    TypeAnnotation, StructDef, StructField, MatchCase, MatchStmt, TryExpr
)


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(f"Parse error at line {token.line}, col {token.column}: {message} (got {token.type.name} {token.value!r})")
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]

    def peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def advance(self) -> Token:
        tok = self.current()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def check(self, expected_type: TokenType) -> bool:
        return self.current().type == expected_type

    def match(self, *types: TokenType) -> bool:
        if self.current().type in types:
            self.advance()
            return True
        return False

    def expect(self, expected_type: TokenType, message: str = "") -> Token:
        if self.check(expected_type):
            return self.advance()
        msg = message or f"Expected token {expected_type.name}"
        raise ParseError(msg, self.current())

    def skip_newlines(self):
        while self.match(TokenType.NEWLINE, TokenType.SEMICOLON):
            pass

    def _is_soft_keyword(self, name: str, offset: int = 0) -> bool:
        tok = self.peek(offset) if offset > 0 else self.current()
        return tok.type == TokenType.IDENTIFIER and tok.value == name

    def _match_soft_keyword(self, name: str) -> bool:
        if self._is_soft_keyword(name):
            self.advance()
            return True
        return False

    def _is_match_stmt(self) -> bool:
        tok = self.current()
        if tok.type != TokenType.MATCH and not (tok.type == TokenType.IDENTIFIER and tok.value == "match"):
            return False
        # Atama operatörleri geliyorsa 'match' bir değişkendir
        assign_ops = (
            TokenType.EQUAL, TokenType.PLUS_EQUAL, TokenType.MINUS_EQUAL,
            TokenType.STAR_EQUAL, TokenType.SLASH_EQUAL, TokenType.AT_EQUAL,
        )
        if self.peek(1).type in assign_ops:
            return False
        # İleri bak: satırda/ifadede ':' veya '{' ve ardından 'case' var mı?
        i = self.pos + 1
        depth = 0
        while i < len(self.tokens):
            t = self.tokens[i]
            if t.type in (TokenType.LPAREN, TokenType.LBRACKET):
                depth += 1
            elif t.type in (TokenType.RPAREN, TokenType.RBRACKET):
                depth = max(0, depth - 1)
            elif depth == 0:
                if t.type in (TokenType.COLON, TokenType.LBRACE):
                    j = i + 1
                    while j < len(self.tokens) and self.tokens[j].type in (TokenType.NEWLINE, TokenType.INDENT):
                        j += 1
                    if j < len(self.tokens):
                        nt = self.tokens[j]
                        if nt.type == TokenType.CASE or (nt.type == TokenType.IDENTIFIER and nt.value == "case"):
                            return True
                    if t.type == TokenType.COLON and j > i + 1:
                        return True
                    return t.type == TokenType.LBRACE
                elif t.type in (TokenType.NEWLINE, TokenType.SEMICOLON, TokenType.EOF):
                    return False
            i += 1
        return False

    # ==========================================
    # Program & Statement Parsers
    # ==========================================
    def parse(self) -> Program:
        statements: list[Stmt] = []
        self.skip_newlines()

        while not self.check(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()

        first_tok = self.tokens[0] if self.tokens else Token(TokenType.EOF, "", 1, 1)
        return Program(line=first_tok.line, column=first_tok.column, statements=statements)

    def parse_statement(self) -> Stmt:
        self.skip_newlines()

        tok = self.current()

        if tok.type in (TokenType.LET, TokenType.CONST):
            return self.parse_var_decl()
        elif tok.type == TokenType.ENUM:
            return self.parse_enum_decl()
        elif tok.type == TokenType.STRUCT or (self._is_soft_keyword("struct") and self.peek(1).type == TokenType.IDENTIFIER):
            return self.parse_struct_def()
        elif tok.type == TokenType.FN:
            return self.parse_function_def()
        elif tok.type == TokenType.RETURN:
            return self.parse_return_stmt()
        elif tok.type == TokenType.IF:
            return self.parse_if_stmt()
        elif tok.type == TokenType.WHILE:
            return self.parse_while_stmt()
        elif tok.type == TokenType.FOR:
            return self.parse_for_stmt()
        elif tok.type == TokenType.PROMPT or (self._is_soft_keyword("prompt") and self.peek(1).type == TokenType.IDENTIFIER):
            return self.parse_prompt_def()
        elif tok.type == TokenType.AGENT or (self._is_soft_keyword("agent") and self.peek(1).type == TokenType.IDENTIFIER):
            return self.parse_agent_def()
        elif tok.type == TokenType.TOOL or (self._is_soft_keyword("tool") and self.peek(1).type == TokenType.IDENTIFIER):
            return self.parse_tool_def()
        elif self._is_match_stmt():
            return self._parse_match_stmt()
        elif tok.type in (TokenType.IMPORT, TokenType.FROM):
            return self.parse_import_stmt()
        elif tok.type == TokenType.PASS:
            self.advance()
            return PassStmt(line=tok.line, column=tok.column)
        elif tok.type == TokenType.BREAK:
            self.advance()
            return BreakStmt(line=tok.line, column=tok.column)
        elif tok.type == TokenType.CONTINUE:
            self.advance()
            return ContinueStmt(line=tok.line, column=tok.column)
        else:
            return self.parse_expr_or_assign_stmt()

    def parse_suite(self) -> list[Stmt]:
        """Bir süit (: sonrası blok) ayrıştırır."""
        self.expect(TokenType.COLON, "Expected ':' after block header")

        # Tek satırlı süit veya girintili blok
        if self.match(TokenType.NEWLINE):
            self.expect(TokenType.INDENT, "Expected indentation after ':'")
            statements: list[Stmt] = []
            while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.DEDENT) or self.check(TokenType.EOF):
                    break
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
                self.skip_newlines()
            self.expect(TokenType.DEDENT, "Expected unindent to close block")
            return statements
        else:
            # Tek satır
            stmt = self.parse_statement()
            return [stmt]

    def parse_var_decl(self) -> VarDeclStmt:
        tok = self.advance()  # let veya const
        is_const = (tok.type == TokenType.CONST)
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected variable name")
        name = id_tok.value

        type_annot = None
        if self.match(TokenType.COLON):
            type_annot = self.parse_type_annotation()

        self.expect(TokenType.EQUAL, "Expected '=' in variable declaration")
        val_expr = self.parse_expression()
        return VarDeclStmt(line=tok.line, column=tok.column, name=name, type_annot=type_annot, value=val_expr, is_const=is_const)

    def parse_expr_or_assign_stmt(self) -> Stmt:
        tok = self.current()
        expr = self.parse_expression()

        # Atama operatörü var mı? (=, +=, -=, *=, /=, @=)
        assign_ops = {
            TokenType.EQUAL: "=",
            TokenType.PLUS_EQUAL: "+=",
            TokenType.MINUS_EQUAL: "-=",
            TokenType.STAR_EQUAL: "*=",
            TokenType.SLASH_EQUAL: "/=",
            TokenType.AT_EQUAL: "@=",
        }

        if self.current().type in assign_ops:
            op_tok = self.advance()
            op_str = assign_ops[op_tok.type]
            val_expr = self.parse_expression()
            return AssignStmt(line=tok.line, column=tok.column, target=expr, op=op_str, value=val_expr)

        return ExprStmt(line=tok.line, column=tok.column, expr=expr)

    def parse_function_def(self) -> FunctionDef:
        tok = self.advance()  # fn
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected function name")
        params = self.parse_param_list()

        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type_annotation()

        body = self.parse_suite()
        return FunctionDef(line=tok.line, column=tok.column, name=id_tok.value, params=params, return_type=return_type, body=body)

    def parse_param_list(self) -> list[Param]:
        self.expect(TokenType.LPAREN, "Expected '(' for parameter list")
        params: list[Param] = []

        if not self.check(TokenType.RPAREN):
            while True:
                p_tok = self.expect(TokenType.IDENTIFIER, "Expected parameter name")
                type_annot = None
                if self.match(TokenType.COLON):
                    type_annot = self.parse_type_annotation()
                default_val = None
                if self.match(TokenType.EQUAL):
                    default_val = self.parse_expression()

                params.append(Param(line=p_tok.line, column=p_tok.column, name=p_tok.value, type_annot=type_annot, default_value=default_val))

                if not self.match(TokenType.COMMA):
                    break

        self.expect(TokenType.RPAREN, "Expected ')' to close parameter list")
        return params

    def parse_enum_decl(self) -> EnumDeclStmt:
        tok = self.advance()  # enum
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected enum name")
        name = id_tok.value
        self.expect(TokenType.COLON, "Expected ':' after enum name")

        variants: list[str] = []

        if self.match(TokenType.NEWLINE):
            self.expect(TokenType.INDENT, "Expected indentation after ':' in enum definition")
            while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.DEDENT) or self.check(TokenType.EOF):
                    break
                if self.check(TokenType.IDENTIFIER) or self.current().type in (TokenType.NONE, TokenType.BOOL):
                    v_tok = self.advance()
                    variants.append(str(v_tok.value))
                else:
                    v_tok = self.expect(TokenType.IDENTIFIER, "Expected enum variant name")
                    variants.append(v_tok.value)

                if self.match(TokenType.EQUAL):
                    self.parse_expression()

                while self.match(TokenType.COMMA):
                    self.skip_newlines()
                    if self.check(TokenType.IDENTIFIER) or self.current().type in (TokenType.NONE, TokenType.BOOL):
                        v2_tok = self.advance()
                        variants.append(str(v2_tok.value))
                        if self.match(TokenType.EQUAL):
                            self.parse_expression()

                self.skip_newlines()
            self.expect(TokenType.DEDENT, "Expected unindent to close enum definition")
        else:
            # Inline comma-separated: enum Color: Red, Green, Blue
            while True:
                if self.check(TokenType.IDENTIFIER) or self.current().type in (TokenType.NONE, TokenType.BOOL):
                    v_tok = self.advance()
                    variants.append(str(v_tok.value))
                else:
                    v_tok = self.expect(TokenType.IDENTIFIER, "Expected enum variant name")
                    variants.append(v_tok.value)

                if self.match(TokenType.EQUAL):
                    self.parse_expression()

                if not self.match(TokenType.COMMA):
                    break
                if self.check(TokenType.NEWLINE) or self.check(TokenType.EOF) or self.check(TokenType.SEMICOLON):
                    break

        return EnumDeclStmt(line=tok.line, column=tok.column, name=name, variants=variants)

    def parse_struct_def(self) -> StructDef:
        tok = self.advance()  # struct
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected struct name")
        name = id_tok.value
        self.expect(TokenType.COLON, "Expected ':' after struct name")

        fields: list[StructField] = []
        docstring: Optional[str] = None

        if self.match(TokenType.NEWLINE):
            self.expect(TokenType.INDENT, "Expected indentation after ':' in struct definition")
            while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.DEDENT) or self.check(TokenType.EOF):
                    break

                # Check for docstring literal at start of struct
                if self.check(TokenType.STRING) and docstring is None and not fields:
                    str_tok = self.advance()
                    docstring = str_tok.value
                    self.skip_newlines()
                    continue

                if self.check(TokenType.PASS):
                    self.advance()
                    self.skip_newlines()
                    continue

                f_tok = self.expect(TokenType.IDENTIFIER, "Expected struct field name")
                f_name = f_tok.value
                type_annot = None
                if self.match(TokenType.COLON):
                    type_annot = self.parse_type_annotation()
                default_val = None
                if self.match(TokenType.EQUAL):
                    default_val = self.parse_expression()

                fields.append(StructField(
                    line=f_tok.line,
                    column=f_tok.column,
                    name=f_name,
                    type_annot=type_annot,
                    default_value=default_val
                ))
                self.skip_newlines()
            self.expect(TokenType.DEDENT, "Expected unindent to close struct definition")
        else:
            # Inline struct: struct Point: x: float, y: float
            while True:
                if self.check(TokenType.NEWLINE) or self.check(TokenType.EOF) or self.check(TokenType.SEMICOLON):
                    break
                f_tok = self.expect(TokenType.IDENTIFIER, "Expected struct field name")
                f_name = f_tok.value
                type_annot = None
                if self.match(TokenType.COLON):
                    type_annot = self.parse_type_annotation()
                default_val = None
                if self.match(TokenType.EQUAL):
                    default_val = self.parse_expression()

                fields.append(StructField(
                    line=f_tok.line,
                    column=f_tok.column,
                    name=f_name,
                    type_annot=type_annot,
                    default_value=default_val
                ))
                if not self.match(TokenType.COMMA):
                    break
                if self.check(TokenType.NEWLINE) or self.check(TokenType.EOF) or self.check(TokenType.SEMICOLON):
                    break

        return StructDef(line=tok.line, column=tok.column, name=name, fields=fields, docstring=docstring)

    def parse_type_annotation(self) -> TypeAnnotation:
        start_tok = self.current()
        first_type = self._parse_primary_type()

        if self.match(TokenType.BAR):
            types = [first_type]
            while True:
                types.append(self._parse_primary_type())
                if not self.match(TokenType.BAR):
                    break
            return UnionType(types=types, line=start_tok.line, column=start_tok.column)

        return first_type

    def _parse_primary_type(self) -> TypeAnnotation:
        start_tok = self.current()
        if self.check(TokenType.IDENTIFIER):
            id_tok = self.advance()
            base_name = "Tensor" if id_tok.value.lower() == "tensor" else id_tok.value
        elif self.check(TokenType.INT):
            id_tok = self.advance()
            base_name = str(id_tok.value)
        elif self.check(TokenType.STRING):
            id_tok = self.advance()
            base_name = f"'{id_tok.value}'"
        elif self.check(TokenType.NONE):
            id_tok = self.advance()
            base_name = "None"
        elif self.check(TokenType.TENSOR):
            id_tok = self.advance()
            base_name = "Tensor"
        else:
            id_tok = self.expect(TokenType.IDENTIFIER, "Expected type name")
            base_name = "Tensor" if id_tok.value.lower() == "tensor" else id_tok.value

        if self.match(TokenType.LBRACKET):
            type_args: list[TypeAnnotation] = []
            while not self.check(TokenType.RBRACKET) and not self.check(TokenType.EOF):
                if self.check(TokenType.IDENTIFIER) and self.peek().type == TokenType.EQUAL:
                    kw_tok = self.advance()
                    self.advance()  # '='
                    val_tok = self.advance()
                    val_str = f"'{val_tok.value}'" if val_tok.type == TokenType.STRING else str(val_tok.value)
                    type_args.append(TypeAnnotation(raw=f"{kw_tok.value}={val_str}", line=kw_tok.line, column=kw_tok.column))
                else:
                    type_args.append(self.parse_type_annotation())
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RBRACKET, "Expected ']' in type annotation")
            return GenericType(base=base_name, type_args=type_args, line=start_tok.line, column=start_tok.column)


        return TypeAnnotation(raw=base_name, line=start_tok.line, column=start_tok.column)

    def parse_return_stmt(self) -> ReturnStmt:
        tok = self.advance()  # return
        if self.check(TokenType.NEWLINE) or self.check(TokenType.SEMICOLON) or self.check(TokenType.EOF):
            return ReturnStmt(line=tok.line, column=tok.column, value=None)
        val = self.parse_expression()
        return ReturnStmt(line=tok.line, column=tok.column, value=val)

    def parse_if_stmt(self) -> IfStmt:
        tok = self.advance()  # if
        cond = self.parse_expression()
        then_branch = self.parse_suite()

        elif_branches: list[tuple[Expr, list[Stmt]]] = []
        else_branch: Optional[list[Stmt]] = None

        self.skip_newlines()
        while self.match(TokenType.ELIF):
            elif_cond = self.parse_expression()
            elif_body = self.parse_suite()
            elif_branches.append((elif_cond, elif_body))
            self.skip_newlines()

        if self.match(TokenType.ELSE):
            else_branch = self.parse_suite()

        return IfStmt(line=tok.line, column=tok.column, condition=cond, then_branch=then_branch, elif_branches=elif_branches, else_branch=else_branch)

    def parse_while_stmt(self) -> WhileStmt:
        tok = self.advance()  # while
        cond = self.parse_expression()
        body = self.parse_suite()
        return WhileStmt(line=tok.line, column=tok.column, condition=cond, body=body)

    def parse_for_stmt(self) -> ForStmt:
        tok = self.advance()  # for
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected loop variable name")
        self.expect(TokenType.IN, "Expected 'in' in for loop")
        iterable = self.parse_expression()
        body = self.parse_suite()
        return ForStmt(line=tok.line, column=tok.column, target=id_tok.value, iterable=iterable, body=body)

    def parse_prompt_def(self) -> PromptDef:
        tok = self.advance()  # prompt
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected prompt name")
        params = self.parse_param_list() if self.check(TokenType.LPAREN) else []

        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type_annotation()

        fields: dict[str, Expr] = {}
        if self.match(TokenType.LBRACE):
            while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.RBRACE) or self.check(TokenType.EOF):
                    break
                field_tok = self.expect(TokenType.IDENTIFIER, "Expected prompt field name")
                self.expect(TokenType.COLON, "Expected ':' after prompt field name")
                expr = self.parse_expression()
                fields[field_tok.value] = expr
                self.match(TokenType.COMMA)
                self.skip_newlines()
            self.expect(TokenType.RBRACE, "Expected '}' to close prompt definition")
        else:
            self.expect(TokenType.COLON, "Expected ':' after prompt header")
            self.expect(TokenType.NEWLINE, "Expected newline after ':' in prompt definition")
            self.expect(TokenType.INDENT, "Expected indentation for prompt fields")

            while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.DEDENT) or self.check(TokenType.EOF):
                    break
                field_tok = self.expect(TokenType.IDENTIFIER, "Expected prompt field name")
                self.expect(TokenType.COLON, "Expected ':' after prompt field name")
                expr = self.parse_expression()
                fields[field_tok.value] = expr
                self.skip_newlines()

            self.expect(TokenType.DEDENT, "Expected unindent to close prompt definition")
        return PromptDef(line=tok.line, column=tok.column, name=id_tok.value, params=params, return_type=return_type, fields=fields)

    def parse_agent_def(self) -> AgentDef:
        tok = self.advance()  # agent
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected agent name")

        fields: dict[str, Expr] = {}
        if self.match(TokenType.LBRACE):
            while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.RBRACE) or self.check(TokenType.EOF):
                    break
                field_tok = self.expect(TokenType.IDENTIFIER, "Expected agent field (model, tools, instructions, memory)")
                self.expect(TokenType.COLON, "Expected ':' after agent field name")
                expr = self.parse_expression()
                fields[field_tok.value] = expr
                self.match(TokenType.COMMA)
                self.skip_newlines()
            self.expect(TokenType.RBRACE, "Expected '}' to close agent definition")
        else:
            self.expect(TokenType.COLON, "Expected ':' after agent name")
            self.expect(TokenType.NEWLINE, "Expected newline after ':' in agent definition")
            self.expect(TokenType.INDENT, "Expected indentation for agent fields")

            while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.DEDENT) or self.check(TokenType.EOF):
                    break
                field_tok = self.expect(TokenType.IDENTIFIER, "Expected agent field (model, tools, instructions, memory)")
                self.expect(TokenType.COLON, "Expected ':' after agent field name")
                expr = self.parse_expression()
                fields[field_tok.value] = expr
                self.skip_newlines()

            self.expect(TokenType.DEDENT, "Expected unindent to close agent definition")
        return AgentDef(line=tok.line, column=tok.column, name=id_tok.value, fields=fields)

    def parse_tool_def(self) -> ToolDef:
        tok = self.advance()  # tool
        id_tok = self.expect(TokenType.IDENTIFIER, "Expected tool name")
        params = self.parse_param_list() if self.check(TokenType.LPAREN) else []

        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.parse_type_annotation()

        if self.check(TokenType.LBRACE):
            self.advance()  # {
            body: list[Stmt] = []
            while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.RBRACE) or self.check(TokenType.EOF):
                    break
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)
                self.skip_newlines()
            self.expect(TokenType.RBRACE, "Expected '}' to close tool body")
        else:
            body = self.parse_suite()

        return ToolDef(line=tok.line, column=tok.column, name=id_tok.value, params=params, return_type=return_type, body=body)

    def _parse_match_stmt(self) -> MatchStmt:
        tok = self.advance()  # match
        subject = self.parse_expression()

        cases: list[MatchCase] = []
        if self.match(TokenType.LBRACE):
            while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.RBRACE) or self.check(TokenType.EOF):
                    break
                case_tok = self.current()
                if case_tok.type == TokenType.CASE or (case_tok.type == TokenType.IDENTIFIER and case_tok.value == "case"):
                    self.advance()
                else:
                    raise ParseError("Expected 'case' in match block", case_tok)
                pattern = self.parse_expression()
                guard = None
                if self.match(TokenType.IF):
                    guard = self.parse_expression()
                body = self.parse_suite()
                cases.append(MatchCase(line=case_tok.line, column=case_tok.column, pattern=pattern, body=body, guard=guard))
                self.skip_newlines()
            self.expect(TokenType.RBRACE, "Expected '}' to close match statement")
        else:
            self.expect(TokenType.COLON, "Expected ':' after match expression")
            self.expect(TokenType.NEWLINE, "Expected newline after ':' in match statement")
            self.expect(TokenType.INDENT, "Expected indentation for match cases")

            while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
                self.skip_newlines()
                if self.check(TokenType.DEDENT) or self.check(TokenType.EOF):
                    break
                case_tok = self.current()
                if case_tok.type == TokenType.CASE or (case_tok.type == TokenType.IDENTIFIER and case_tok.value == "case"):
                    self.advance()
                else:
                    raise ParseError("Expected 'case' in match statement", case_tok)
                pattern = self.parse_expression()
                guard = None
                if self.match(TokenType.IF):
                    guard = self.parse_expression()
                body = self.parse_suite()
                cases.append(MatchCase(line=case_tok.line, column=case_tok.column, pattern=pattern, body=body, guard=guard))
                self.skip_newlines()

            self.expect(TokenType.DEDENT, "Expected unindent to close match statement")

        return MatchStmt(line=tok.line, column=tok.column, subject=subject, cases=cases)

    def parse_import_stmt(self) -> ImportStmt:
        tok = self.advance()  # 'import' or 'from'
        is_python = False

        if tok.type == TokenType.FROM:
            p1 = self.expect(TokenType.IDENTIFIER, "Expected module name after 'from'")
            parts: list[str] = []
            if p1.value in ("py", "python"):
                is_python = True
                while self.match(TokenType.DOT):
                    p = self.expect(TokenType.IDENTIFIER, "Expected identifier after '.' in from import")
                    parts.append(p.value)
                self.expect(TokenType.IMPORT, "Expected 'import' after 'from ...'")
                imported_name = self.expect(TokenType.IDENTIFIER, "Expected imported module or name")
                parts.append(imported_name.value)
            else:
                parts.append(p1.value)
                while self.match(TokenType.DOT):
                    p = self.expect(TokenType.IDENTIFIER, "Expected identifier after '.' in from import")
                    parts.append(p.value)
                self.expect(TokenType.IMPORT, "Expected 'import' after 'from ...'")
                imported_name = self.expect(TokenType.IDENTIFIER, "Expected imported module or name")
                parts.append(imported_name.value)

            alias = None
            if self.match(TokenType.AS):
                alias_tok = self.expect(TokenType.IDENTIFIER, "Expected alias identifier after 'as'")
                alias = alias_tok.value

            return ImportStmt(line=tok.line, column=tok.column, module_path=parts, alias=alias, is_python=is_python)

        # tok.type == TokenType.IMPORT
        parts: list[str] = []
        p1 = self.expect(TokenType.IDENTIFIER, "Expected module name in import")
        if p1.value in ("py", "python") and self.match(TokenType.DOT):
            is_python = True
            p2 = self.expect(TokenType.IDENTIFIER, "Expected python module name after 'py.' or 'python.'")
            parts.append(p2.value)
        else:
            parts.append(p1.value)

        while self.match(TokenType.DOT):
            p = self.expect(TokenType.IDENTIFIER, "Expected identifier after '.' in import")
            parts.append(p.value)

        alias = None
        if self.match(TokenType.AS):
            alias_tok = self.expect(TokenType.IDENTIFIER, "Expected alias identifier after 'as'")
            alias = alias_tok.value

        return ImportStmt(line=tok.line, column=tok.column, module_path=parts, alias=alias, is_python=is_python)

    # ==========================================
    # Expressions & Precedence Climbing
    # ==========================================
    def parse_expression(self) -> Expr:
        return self.parse_pipe()

    def parse_pipe(self) -> Expr:
        expr = self.parse_logical_or()
        while self.match(TokenType.PIPE):
            pipe_tok = self.tokens[self.pos - 1]
            right = self.parse_logical_or()
            expr = PipeExpr(line=pipe_tok.line, column=pipe_tok.column, left=expr, right=right)
        return expr

    def parse_logical_or(self) -> Expr:
        expr = self.parse_bitwise_or()
        while self.match(TokenType.OR):
            op_tok = self.tokens[self.pos - 1]
            right = self.parse_bitwise_or()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op="or", right=right)
        return expr

    def parse_bitwise_or(self) -> Expr:
        expr = self.parse_logical_and()
        while self.match(TokenType.BAR):
            op_tok = self.tokens[self.pos - 1]
            right = self.parse_logical_and()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op="|", right=right)
        return expr

    def parse_logical_and(self) -> Expr:
        expr = self.parse_equality()
        while self.match(TokenType.AND):
            op_tok = self.tokens[self.pos - 1]
            right = self.parse_equality()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op="and", right=right)
        return expr

    def parse_equality(self) -> Expr:
        expr = self.parse_relational()
        while self.match(TokenType.EQUAL_EQUAL, TokenType.BANG_EQUAL):
            op_tok = self.tokens[self.pos - 1]
            op = "==" if op_tok.type == TokenType.EQUAL_EQUAL else "!="
            right = self.parse_relational()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op=op, right=right)
        return expr

    def parse_relational(self) -> Expr:
        expr = self.parse_additive()
        ops = {
            TokenType.LESS: "<",
            TokenType.LESS_EQUAL: "<=",
            TokenType.GREATER: ">",
            TokenType.GREATER_EQUAL: ">=",
        }
        while self.current().type in ops:
            op_tok = self.advance()
            right = self.parse_additive()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op=ops[op_tok.type], right=right)
        return expr

    def parse_additive(self) -> Expr:
        expr = self.parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_tok = self.tokens[self.pos - 1]
            op = "+" if op_tok.type == TokenType.PLUS else "-"
            right = self.parse_multiplicative()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op=op, right=right)
        return expr

    def parse_multiplicative(self) -> Expr:
        expr = self.parse_unary()
        ops = {
            TokenType.STAR: "*",
            TokenType.SLASH: "/",
            TokenType.DOUBLE_SLASH: "//",
            TokenType.PERCENT: "%",
            TokenType.AT: "@",
        }
        while self.current().type in ops:
            op_tok = self.advance()
            right = self.parse_unary()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op=ops[op_tok.type], right=right)
        return expr

    def parse_matmul(self) -> Expr:
        return self.parse_multiplicative()

    def parse_unary(self) -> Expr:
        if self.match(TokenType.MINUS, TokenType.PLUS, TokenType.NOT):
            op_tok = self.tokens[self.pos - 1]
            op = "not" if op_tok.type == TokenType.NOT else ("-" if op_tok.type == TokenType.MINUS else "+")
            operand = self.parse_unary()
            return UnaryExpr(line=op_tok.line, column=op_tok.column, op=op, operand=operand)
        return self.parse_power()

    def parse_power(self) -> Expr:
        expr = self.parse_postfix()
        while self.match(TokenType.DOUBLE_STAR):
            op_tok = self.tokens[self.pos - 1]
            right = self.parse_power()
            expr = BinaryExpr(line=op_tok.line, column=op_tok.column, left=expr, op="**", right=right)
        return expr

    def parse_postfix(self) -> Expr:
        expr = self.parse_atom()

        while True:
            # Fonksiyon Çağrısı f(...)
            if self.match(TokenType.LPAREN):
                l_tok = self.tokens[self.pos - 1]
                args, kwargs = self.parse_call_args()
                expr = CallExpr(line=l_tok.line, column=l_tok.column, callee=expr, args=args, kwargs=kwargs)
            # Dizi İndeksi a[...]
            elif self.match(TokenType.LBRACKET):
                l_tok = self.tokens[self.pos - 1]
                idx_expr = self.parse_expression()
                self.expect(TokenType.RBRACKET, "Expected ']' in index expression")
                expr = IndexExpr(line=l_tok.line, column=l_tok.column, target=expr, index=idx_expr)
            # Üye / Metod erişimi obj.prop
            elif self.match(TokenType.DOT):
                d_tok = self.tokens[self.pos - 1]
                tok = self.current()
                if (tok.type in (TokenType.IDENTIFIER, TokenType.NONE, TokenType.BOOL, TokenType.MATCH, TokenType.CASE)
                    or tok.type in (TokenType.GRAD, TokenType.TENSOR, TokenType.PROMPT, TokenType.AGENT, TokenType.TOOL)
                    or (isinstance(tok.value, str) and tok.value.isidentifier())):
                    member_name = "None" if tok.type == TokenType.NONE else str(tok.value)
                    self.advance()
                    expr = MemberExpr(line=d_tok.line, column=d_tok.column, target=expr, member=member_name)
                else:
                    raise ParseError("Expected identifier or property name after '.'", tok)
            # Try / Error propagation '?' (Rust tarzı postfix operatör)
            elif self.match(TokenType.QUESTION):
                q_tok = self.tokens[self.pos - 1]
                expr = TryExpr(line=q_tok.line, column=q_tok.column, expr=expr)
            else:
                break

        return expr

    def parse_call_args(self) -> tuple[list[Expr], dict[str, Expr]]:
        args: list[Expr] = []
        kwargs: dict[str, Expr] = {}

        if not self.check(TokenType.RPAREN):
            while True:
                # keyword arg mı? (name = expr)
                if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.EQUAL:
                    k_tok = self.advance()
                    self.advance()  # =
                    val_expr = self.parse_expression()
                    kwargs[k_tok.value] = val_expr
                else:
                    arg_expr = self.parse_expression()
                    args.append(arg_expr)

                if not self.match(TokenType.COMMA):
                    break

        self.expect(TokenType.RPAREN, "Expected ')' to close argument list")
        return args, kwargs

    def parse_atom(self) -> Expr:
        tok = self.current()

        # Sayılar
        if tok.type in (TokenType.INT, TokenType.FLOAT):
            self.advance()
            return LiteralExpr(line=tok.line, column=tok.column, value=tok.value)

        # String
        if tok.type == TokenType.STRING:
            self.advance()
            return LiteralExpr(line=tok.line, column=tok.column, value=tok.value)

        # Boolean
        if tok.type == TokenType.BOOL:
            self.advance()
            return LiteralExpr(line=tok.line, column=tok.column, value=tok.value)

        # None
        if tok.type == TokenType.NONE:
            self.advance()
            return LiteralExpr(line=tok.line, column=tok.column, value=None)

        # Tensor Literal or Identifier: tensor([[1, 2], [3, 4]], requires_grad=true)
        if tok.type == TokenType.TENSOR or (tok.type == TokenType.IDENTIFIER and tok.value == "tensor"):
            if self.peek().type == TokenType.LPAREN:
                self.advance()  # 'tensor'
                self.advance()  # '('
                args, kwargs = self.parse_call_args()
                data_expr = args[0] if args else ListLiteralExpr(line=tok.line, column=tok.column, elements=[])
                return TensorLiteralExpr(line=tok.line, column=tok.column, data=data_expr, kwargs=kwargs)
            elif tok.type == TokenType.TENSOR:
                self.advance()
                return IdentifierExpr(line=tok.line, column=tok.column, name="tensor")

        # Grad anahtar sözcüğü: grad(fn)
        if tok.type == TokenType.GRAD:
            self.advance()
            return IdentifierExpr(line=tok.line, column=tok.column, name="grad")

        # Tanımlayıcı (Identifier)
        if tok.type == TokenType.IDENTIFIER:
            self.advance()
            return IdentifierExpr(line=tok.line, column=tok.column, name=tok.value)

        # Liste Literali [...]
        if self.match(TokenType.LBRACKET):
            elements: list[Expr] = []
            if not self.check(TokenType.RBRACKET):
                while True:
                    elements.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
            self.expect(TokenType.RBRACKET, "Expected ']' to close list literal")
            return ListLiteralExpr(line=tok.line, column=tok.column, elements=elements)

        # Sözlük Literali {...}
        if self.match(TokenType.LBRACE):
            entries: list[tuple[Expr, Expr]] = []
            if not self.check(TokenType.RBRACE):
                while True:
                    key_expr = self.parse_expression()
                    self.expect(TokenType.COLON, "Expected ':' in dictionary entry")
                    val_expr = self.parse_expression()
                    entries.append((key_expr, val_expr))
                    if not self.match(TokenType.COMMA):
                        break
            self.expect(TokenType.RBRACE, "Expected '}' to close dictionary literal")
            return DictLiteralExpr(line=tok.line, column=tok.column, entries=entries)

        # Parantezli İfade (...)
        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN, "Expected ')' to close grouped expression")
            return expr

        raise ParseError(f"Unexpected token in expression: {tok.type.name}", tok)

    @classmethod
    def parse_source(
        cls,
        source: str,
        filename: str = "",
        use_cache: bool = True,
        cache: Optional[ASTCache] = None,
    ) -> Program:
        """Convenience method on Parser class to parse source code with AST caching."""
        return parse_source(source, filename=filename, use_cache=use_cache, cache=cache)


_default_cache: Optional[ASTCache] = None


def get_default_cache() -> Any:
    """Returns the process-wide default ASTCache instance."""
    global _default_cache
    if _default_cache is None:
        from synapse.compiler.cache import ASTCache
        _default_cache = ASTCache()
    return _default_cache


def parse_source(
    source: str,
    filename: str = "",
    use_cache: bool = True,
    cache: Optional[ASTCache] = None,
) -> Program:
    """
    Parses Synapse source code into an AST Program node.
    Integrates with ASTCache for fast compilation and disk/memory caching.

    Args:
        source: The Synapse source code string.
        filename: Optional filename or path for the source.
        use_cache: If True, attempts to retrieve or save the parsed AST in ASTCache.
        cache: Optional custom ASTCache instance (defaults to process-wide cache).

    Returns:
        Program: The root AST node for the parsed source.
    """
    if not source and filename and os.path.isfile(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            pass

    cache_instance = cache if cache is not None else get_default_cache()

    if use_cache:
        cached_ast = cache_instance.get(source, filename=filename)
        if cached_ast is not None:
            return cached_ast

    tokens = Lexer(source).tokenize()
    parser = Parser(tokens)
    ast = parser.parse()

    if use_cache:
        cache_instance.set(source, ast, filename=filename)

    return ast

