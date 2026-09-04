from dataclasses import dataclass, field
from typing import Any, Optional
from synapse.vm.opcodes import Opcode
from synapse.parser.ast_nodes import (
    Program, Stmt, Expr, LiteralExpr, IdentifierExpr, BinaryExpr, UnaryExpr,
    PipeExpr, CallExpr, MemberExpr, IndexExpr, ListLiteralExpr, DictLiteralExpr,
    TensorLiteralExpr, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, PassStmt,
    BreakStmt, ContinueStmt, IfStmt, WhileStmt, ForStmt, FunctionDef, ImportStmt,
    PromptDef, AgentDef, ToolDef, StructDef, EnumDeclStmt
)


@dataclass
class CodeObject:
    name: str
    filename: str = "<unknown>"
    instructions: list[tuple[Opcode, Any]] = field(default_factory=list)
    instruction_lines: list[int] = field(default_factory=list)
    constants: list[Any] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)

    def add_const(self, val: Any) -> int:
        for i, c in enumerate(self.constants):
            if c == val and type(c) == type(val):
                return i
        self.constants.append(val)
        return len(self.constants) - 1

    def add_name(self, name: str) -> int:
        if name in self.names:
            return self.names.index(name)
        self.names.append(name)
        return len(self.names) - 1

    def emit(self, opcode: Opcode, arg: Any = None, line: int = 1) -> int:
        self.instructions.append((opcode, arg))
        self.instruction_lines.append(line)
        return len(self.instructions) - 1


class Compiler:
    def __init__(
        self,
        name: str = "<module>",
        repl_mode: bool = False,
        filename: str = "<unknown>",
        options: Optional[Any] = None,
        profile: Optional[Any] = None,
        mode: Optional[Any] = None,
    ):
        self.code = CodeObject(name=name, filename=filename)
        self.name = name
        self.filename = filename
        self.repl_mode = repl_mode
        self.current_line: int = 1
        self.loop_starts: list[int] = []
        self.loop_breaks: list[list[int]] = []
        self.loop_continues: list[list[int]] = []

        if options is not None:
            self.options = options
        elif profile is not None:
            from synapse.compiler.options import CompilerOptions
            self.options = CompilerOptions(profile=profile, mode=mode or "strict")
        else:
            self.options = None

    def compile(self, node: Any) -> CodeObject:
        if self.options is not None and getattr(self.options, "profile", None) is not None:
            from synapse.compiler.options import check_profile_compliance
            check_profile_compliance(node, self.options.profile)

        if isinstance(node, Program):
            for stmt in node.statements:
                self.compile_stmt(stmt, is_top_level=True)
            # Modül sonu None döndürsün
            const_idx = self.code.add_const(None)
            self.code.emit(Opcode.LOAD_CONST, const_idx, line=self.current_line)
            self.code.emit(Opcode.RETURN_VALUE, line=self.current_line)
        return self.code

    def compile_stmt(self, stmt: Stmt, is_top_level: bool = False):
        stmt_line = getattr(stmt, "line", self.current_line) or self.current_line
        self.current_line = stmt_line

        if isinstance(stmt, VarDeclStmt):
            self.compile_expr(stmt.value)
            name_idx = self.code.add_name(stmt.name)
            self.code.emit(Opcode.STORE_NAME, name_idx, line=stmt.line)

        elif isinstance(stmt, AssignStmt):
            if isinstance(stmt.target, IdentifierExpr):
                if stmt.op == "=":
                    self.compile_expr(stmt.value)
                else:
                    # In-place operasyon: +=, -= vb.
                    bin_op = stmt.op.rstrip("=")
                    name_idx = self.code.add_name(stmt.target.name)
                    self.code.emit(Opcode.LOAD_NAME, name_idx, line=stmt.line)
                    self.compile_expr(stmt.value)
                    self._emit_binary_op(bin_op, line=stmt.line)
                name_idx = self.code.add_name(stmt.target.name)
                self.code.emit(Opcode.STORE_NAME, name_idx, line=stmt.line)
            elif isinstance(stmt.target, IndexExpr):
                # target[index] = value
                self.compile_expr(stmt.target.target)
                self.compile_expr(stmt.target.index)
                self.compile_expr(stmt.value)
                self.code.emit(Opcode.STORE_SUBSCR, line=stmt.line)
            elif isinstance(stmt.target, MemberExpr):
                self.compile_expr(stmt.target.target)
                self.compile_expr(stmt.value)
                attr_idx = self.code.add_name(stmt.target.member)
                self.code.emit(Opcode.STORE_ATTR, attr_idx, line=stmt.line)
            else:
                raise NotImplementedError(f"Assignment to {type(stmt.target)} not supported yet")

        elif isinstance(stmt, ExprStmt):
            if self.repl_mode and is_top_level:
                disp_idx = self.code.add_name("__repl_display__")
                self.code.emit(Opcode.LOAD_NAME, disp_idx, line=stmt.line)
                self.compile_expr(stmt.expr)
                self.code.emit(Opcode.CALL_FUNCTION, 1, line=stmt.line)
                self.code.emit(Opcode.POP_TOP, line=stmt.line)
            else:
                self.compile_expr(stmt.expr)
                self.code.emit(Opcode.POP_TOP, line=stmt.line)

        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self.compile_expr(stmt.value)
            else:
                const_idx = self.code.add_const(None)
                self.code.emit(Opcode.LOAD_CONST, const_idx, line=stmt.line)
            self.code.emit(Opcode.RETURN_VALUE, line=stmt.line)

        elif isinstance(stmt, IfStmt):
            self.compile_if(stmt)

        elif isinstance(stmt, WhileStmt):
            self.compile_while(stmt)

        elif isinstance(stmt, ForStmt):
            self.compile_for(stmt)

        elif isinstance(stmt, FunctionDef):
            self.compile_function(stmt)

        elif isinstance(stmt, PromptDef):
            self.compile_prompt(stmt)

        elif isinstance(stmt, AgentDef):
            self.compile_agent(stmt)

        elif isinstance(stmt, ToolDef):
            # Tool temelde bir fonksiyondur
            fn_stmt = FunctionDef(
                line=stmt.line,
                column=stmt.column,
                name=stmt.name,
                params=stmt.params,
                return_type=stmt.return_type,
                body=stmt.body
            )
            self.compile_function(fn_stmt)

        elif isinstance(stmt, ImportStmt):
            mod_str = ".".join(stmt.module_path)
            const_idx = self.code.add_const((mod_str, stmt.alias, stmt.is_python))
            self.code.emit(Opcode.IMPORT_PYTHON, const_idx, line=stmt.line)

        elif isinstance(stmt, BreakStmt):
            if not self.loop_breaks:
                raise SyntaxError("'break' outside loop")
            jump_idx = self.code.emit(Opcode.JUMP, 0, line=stmt.line)
            self.loop_breaks[-1].append(jump_idx)

        elif isinstance(stmt, ContinueStmt):
            if not self.loop_continues:
                raise SyntaxError("'continue' outside loop")
            jump_idx = self.code.emit(Opcode.JUMP, 0, line=stmt.line)
            self.loop_continues[-1].append(jump_idx)

        elif isinstance(stmt, (PassStmt, StructDef, EnumDeclStmt)):
            pass

    def compile_if(self, stmt: IfStmt):
        # condition
        self.compile_expr(stmt.condition)
        jump_to_next = self.code.emit(Opcode.JUMP_IF_FALSE, 0, line=stmt.line)

        # then_branch
        for s in stmt.then_branch:
            self.compile_stmt(s)

        end_jumps: list[int] = []
        end_jumps.append(self.code.emit(Opcode.JUMP, 0, line=stmt.line))

        # patch jump_to_next
        self.code.instructions[jump_to_next] = (Opcode.JUMP_IF_FALSE, len(self.code.instructions))

        # elif_branches
        for cond, body in stmt.elif_branches:
            cond_line = getattr(cond, "line", stmt.line) or stmt.line
            self.compile_expr(cond)
            jump_elif_next = self.code.emit(Opcode.JUMP_IF_FALSE, 0, line=cond_line)
            for s in body:
                self.compile_stmt(s)
            end_jumps.append(self.code.emit(Opcode.JUMP, 0, line=cond_line))
            self.code.instructions[jump_elif_next] = (Opcode.JUMP_IF_FALSE, len(self.code.instructions))

        # else_branch
        if stmt.else_branch:
            for s in stmt.else_branch:
                self.compile_stmt(s)

        # patch all end_jumps to here
        end_target = len(self.code.instructions)
        for j_idx in end_jumps:
            self.code.instructions[j_idx] = (Opcode.JUMP, end_target)

    def compile_while(self, stmt: WhileStmt):
        start_pos = len(self.code.instructions)
        self.loop_starts.append(start_pos)
        self.loop_breaks.append([])
        self.loop_continues.append([])

        self.compile_expr(stmt.condition)
        exit_jump = self.code.emit(Opcode.JUMP_IF_FALSE, 0, line=stmt.line)

        for s in stmt.body:
            self.compile_stmt(s)

        # Loop back
        self.code.emit(Opcode.JUMP, start_pos, line=stmt.line)

        end_pos = len(self.code.instructions)
        self.code.instructions[exit_jump] = (Opcode.JUMP_IF_FALSE, end_pos)

        for brk in self.loop_breaks.pop():
            self.code.instructions[brk] = (Opcode.JUMP, end_pos)
        for cont in self.loop_continues.pop():
            self.code.instructions[cont] = (Opcode.JUMP, start_pos)
        self.loop_starts.pop()

    def compile_for(self, stmt: ForStmt):
        # for i in iterable -> iterable'ı derle, index/length ile iterasyon
        # Basit for döngüsü: iterator/list üzerinden
        self.compile_expr(stmt.iterable)
        name_idx = self.code.add_name(f"__iter_{stmt.target}")
        self.code.emit(Opcode.STORE_NAME, name_idx, line=stmt.line)

        # __idx = 0
        zero_idx = self.code.add_const(0)
        self.code.emit(Opcode.LOAD_CONST, zero_idx, line=stmt.line)
        idx_var = self.code.add_name(f"__idx_{stmt.target}")
        self.code.emit(Opcode.STORE_NAME, idx_var, line=stmt.line)

        loop_start = len(self.code.instructions)
        self.loop_starts.append(loop_start)
        self.loop_breaks.append([])
        self.loop_continues.append([])

        # condition: __idx < len(__iter)
        self.code.emit(Opcode.LOAD_NAME, idx_var, line=stmt.line)
        len_name = self.code.add_name("len")
        self.code.emit(Opcode.LOAD_NAME, len_name, line=stmt.line)
        # call len(__iter)
        self.code.emit(Opcode.LOAD_NAME, name_idx, line=stmt.line)
        self.code.emit(Opcode.CALL_FUNCTION, 1, line=stmt.line)
        self.code.emit(Opcode.COMPARE_OP, "<", line=stmt.line)
        exit_jump = self.code.emit(Opcode.JUMP_IF_FALSE, 0, line=stmt.line)

        # stmt.target = __iter[__idx]
        self.code.emit(Opcode.LOAD_NAME, name_idx, line=stmt.line)
        self.code.emit(Opcode.LOAD_NAME, idx_var, line=stmt.line)
        self.code.emit(Opcode.BINARY_SUBSCR, line=stmt.line)
        target_name = self.code.add_name(stmt.target)
        self.code.emit(Opcode.STORE_NAME, target_name, line=stmt.line)

        # body
        for s in stmt.body:
            self.compile_stmt(s)

        # Step position: continue statement jumps here so __idx is incremented
        step_pos = len(self.code.instructions)
        for cont in self.loop_continues.pop():
            self.code.instructions[cont] = (Opcode.JUMP, step_pos)

        # __idx += 1
        self.code.emit(Opcode.LOAD_NAME, idx_var, line=stmt.line)
        one_const = self.code.add_const(1)
        self.code.emit(Opcode.LOAD_CONST, one_const, line=stmt.line)
        self.code.emit(Opcode.BINARY_ADD, line=stmt.line)
        self.code.emit(Opcode.STORE_NAME, idx_var, line=stmt.line)

        self.code.emit(Opcode.JUMP, loop_start, line=stmt.line)

        loop_end = len(self.code.instructions)
        self.code.instructions[exit_jump] = (Opcode.JUMP_IF_FALSE, loop_end)

        for brk in self.loop_breaks.pop():
            self.code.instructions[brk] = (Opcode.JUMP, loop_end)
        self.loop_starts.pop()

    def compile_function(self, stmt: FunctionDef):
        fn_compiler = Compiler(name=stmt.name, filename=self.filename, options=self.options)
        for p in stmt.params:
            fn_compiler.code.params.append(p.name)

        for s in stmt.body:
            fn_compiler.compile_stmt(s)

        # Fonksiyon sonuna default return None
        none_idx = fn_compiler.code.add_const(None)
        fn_compiler.code.emit(Opcode.LOAD_CONST, none_idx, line=stmt.line)
        fn_compiler.code.emit(Opcode.RETURN_VALUE, line=stmt.line)

        fn_code = fn_compiler.code
        const_idx = self.code.add_const(fn_code)
        self.code.emit(Opcode.LOAD_CONST, const_idx, line=stmt.line)
        self.code.emit(Opcode.MAKE_FUNCTION, len(stmt.params), line=stmt.line)

        fn_name_idx = self.code.add_name(stmt.name)
        self.code.emit(Opcode.STORE_NAME, fn_name_idx, line=stmt.line)

    def compile_prompt(self, stmt: PromptDef):
        # Prompt definition'ı derle
        # fields sözlüğünü ve parametrelerini paketle
        # Runtime'da bir PromptObject oluşturulur
        prompt_data = {
            "name": stmt.name,
            "params": [p.name for p in stmt.params],
            "return_type": stmt.return_type,
        }
        # Saha ifadelerini derlemek için her birini compile_expr ile çözüyoruz
        # Ancak static tanımlar için doğrudan field AST'lerini tutabiliriz
        const_idx = self.code.add_const((stmt.name, [p.name for p in stmt.params], stmt.fields))
        self.code.emit(Opcode.DEFINE_PROMPT, const_idx, line=stmt.line)
        name_idx = self.code.add_name(stmt.name)
        self.code.emit(Opcode.STORE_NAME, name_idx, line=stmt.line)

    def compile_agent(self, stmt: AgentDef):
        const_idx = self.code.add_const((stmt.name, stmt.fields))
        self.code.emit(Opcode.DEFINE_AGENT, const_idx, line=stmt.line)
        name_idx = self.code.add_name(stmt.name)
        self.code.emit(Opcode.STORE_NAME, name_idx, line=stmt.line)

    def compile_expr(self, expr: Expr):
        line = getattr(expr, "line", self.current_line) or self.current_line
        if isinstance(expr, LiteralExpr):
            const_idx = self.code.add_const(expr.value)
            self.code.emit(Opcode.LOAD_CONST, const_idx, line=line)

        elif isinstance(expr, IdentifierExpr):
            name_idx = self.code.add_name(expr.name)
            self.code.emit(Opcode.LOAD_NAME, name_idx, line=line)

        elif isinstance(expr, BinaryExpr):
            self.compile_expr(expr.left)
            self.compile_expr(expr.right)
            self._emit_binary_op(expr.op, line=line)

        elif isinstance(expr, UnaryExpr):
            self.compile_expr(expr.operand)
            if expr.op == "-":
                self.code.emit(Opcode.UNARY_NEGATIVE, line=line)
            elif expr.op == "not":
                self.code.emit(Opcode.UNARY_NOT, line=line)

        elif isinstance(expr, PipeExpr):
            # left |> right => right(left)
            self.compile_expr(expr.right)
            self.compile_expr(expr.left)
            self.code.emit(Opcode.CALL_FUNCTION, 1, line=line)

        elif isinstance(expr, CallExpr):
            self.compile_expr(expr.callee)
            for arg in expr.args:
                self.compile_expr(arg)
            # kwargs şimdilik tuple veya dict olarak
            if expr.kwargs:
                # kwargs için dict literal derle
                for k, v in expr.kwargs.items():
                    k_idx = self.code.add_const(k)
                    self.code.emit(Opcode.LOAD_CONST, k_idx, line=line)
                    self.compile_expr(v)
                self.code.emit(Opcode.BUILD_DICT, len(expr.kwargs), line=line)
                self.code.emit(Opcode.CALL_FUNCTION, len(expr.args) + 1000, line=line)  # +1000 bayrağı kwargs var demek
            else:
                self.code.emit(Opcode.CALL_FUNCTION, len(expr.args), line=line)

        elif isinstance(expr, MemberExpr):
            self.compile_expr(expr.target)
            attr_idx = self.code.add_name(expr.member)
            self.code.emit(Opcode.LOAD_ATTR, attr_idx, line=line)

        elif isinstance(expr, IndexExpr):
            self.compile_expr(expr.target)
            self.compile_expr(expr.index)
            self.code.emit(Opcode.BINARY_SUBSCR, line=line)

        elif isinstance(expr, ListLiteralExpr):
            for el in expr.elements:
                self.compile_expr(el)
            self.code.emit(Opcode.BUILD_LIST, len(expr.elements), line=line)

        elif isinstance(expr, DictLiteralExpr):
            for k, v in expr.entries:
                self.compile_expr(k)
                self.compile_expr(v)
            self.code.emit(Opcode.BUILD_DICT, len(expr.entries), line=line)

        elif isinstance(expr, TensorLiteralExpr):
            # Tensor literal: tensor(data, kwargs)
            self.compile_expr(expr.data)
            # kwargs dict oluştur
            for k, v in expr.kwargs.items():
                k_idx = self.code.add_const(k)
                self.code.emit(Opcode.LOAD_CONST, k_idx, line=line)
                self.compile_expr(v)
            self.code.emit(Opcode.BUILD_DICT, len(expr.kwargs), line=line)
            self.code.emit(Opcode.BUILD_TENSOR, line=line)

    def _emit_binary_op(self, op: str, line: int = 1):
        op_map = {
            "+": Opcode.BINARY_ADD,
            "-": Opcode.BINARY_SUB,
            "*": Opcode.BINARY_MUL,
            "/": Opcode.BINARY_DIV,
            "//": Opcode.BINARY_FLOOR_DIV,
            "%": Opcode.BINARY_MOD,
            "**": Opcode.BINARY_POW,
            "@": Opcode.BINARY_MATMUL,
        }
        if op in op_map:
            self.code.emit(op_map[op], line=line)
        elif op in ("==", "!=", "<", "<=", ">", ">="):
            self.code.emit(Opcode.COMPARE_OP, op, line=line)
        elif op == "and":
            self.code.emit(Opcode.COMPARE_OP, "and", line=line)
        elif op == "or":
            self.code.emit(Opcode.COMPARE_OP, "or", line=line)
        else:
            raise NotImplementedError(f"Binary operator {op} not implemented in compiler")
