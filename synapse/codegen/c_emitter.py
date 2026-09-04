import re
from typing import Any, Optional
from synapse.parser.ast_nodes import (
    Program, Stmt, Expr, LiteralExpr, IdentifierExpr, BinaryExpr, UnaryExpr,
    PipeExpr, CallExpr, MemberExpr, IndexExpr, ListLiteralExpr, DictLiteralExpr,
    TensorLiteralExpr, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, PassStmt,
    BreakStmt, ContinueStmt, IfStmt, WhileStmt, ForStmt, FunctionDef,
    MatchStmt, MatchCase, TryExpr, GenericType, TypeAnnotation
)


class CEmitter:
    def __init__(self, include_line_directives: bool = True):
        self.include_line_directives = include_line_directives
        self.current_filename = "<source>"
        self.last_emitted_line = -1
        self.indent_level = 1
        self.tensor_counter = 0
        self.loop_counter = 0
        self.match_counter = 0
        self.try_counter = 0
        self.current_loop_arena: Optional[str] = None
        self.current_loop_scope: Optional[str] = None
        self.current_fn_return_type: Optional[str] = None
        self.expected_type_stack: list[Optional[str]] = []
        self.scope_stack: list[dict[str, str]] = [{}]
        self.prototypes: list[str] = []
        self.functions_code: list[str] = []
        self.function_return_types: dict[str, str] = {}
        self.pre_stmts: list[str] = []
        # Monomorphic Tagged Union kayıtları:
        # suffix -> (c_type, suffix)
        self.registered_options: dict[str, tuple[str, str]] = {}
        # key -> (c_type_t, suffix_t, c_type_e, suffix_e)
        self.registered_results: dict[str, tuple[str, str, str, str]] = {}

    def _format_line_directive(self, node: Any, force: bool = False) -> Optional[str]:
        if not self.include_line_directives:
            return None
        line = getattr(node, "line", None)
        if line is not None and isinstance(line, int) and line > 0:
            if force or line != self.last_emitted_line:
                self.last_emitted_line = line
                escaped_filename = self.current_filename.replace('"', '\\"')
                return f'#line {line} "{escaped_filename}"'
        return None

    @property
    def declared_vars(self) -> dict[str, str]:
        return self.scope_stack[-1]

    def push_scope(self):
        self.scope_stack.append({})

    def pop_scope(self):
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()

    def lookup_var(self, name: str) -> Optional[str]:
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        return None

    def set_var(self, name: str, var_type: str):
        self.scope_stack[-1][name] = var_type

    def indent(self) -> str:
        return "    " * self.indent_level

    def get_type_info(self, type_annot: Any) -> tuple[str, str]:
        """Tip anotasyonunun (C_tipi, guvenli_suffix) ikilisini döner."""
        if not type_annot:
            return ("syn_tensor_t*", "tensor")

        if isinstance(type_annot, GenericType):
            if type_annot.base == "Option" and type_annot.type_args:
                opt_c = self.register_option_type(type_annot.type_args[0])
                suf = opt_c[len("SynOption_"):]
                return (opt_c, suf)
            elif type_annot.base == "Result":
                t_arg = type_annot.type_args[0] if len(type_annot.type_args) > 0 else "int"
                e_arg = type_annot.type_args[1] if len(type_annot.type_args) > 1 else "str"
                res_c = self.register_result_type(t_arg, e_arg)
                suf = res_c[len("SynResult_"):]
                return (res_c, suf)
            elif type_annot.base in ("Tensor", "syn_tensor"):
                return ("syn_tensor_t*", "tensor")

        raw_str = getattr(type_annot, "raw", type_annot)
        if not isinstance(raw_str, str):
            raw_str = str(raw_str)
        t = raw_str.strip()
        t_lower = t.lower()

        # Option[...] string
        m_opt = re.match(r"^Option\[\s*(.+?)\s*\]$", t, re.IGNORECASE)
        if m_opt:
            opt_c = self.register_option_type(m_opt.group(1))
            return (opt_c, opt_c[len("SynOption_"):])

        # Result[..., ...] string
        m_res = re.match(r"^Result\[\s*(.+?)\s*,\s*(.+?)\s*\]$", t, re.IGNORECASE)
        if m_res:
            res_c = self.register_result_type(m_res.group(1), m_res.group(2))
            return (res_c, res_c[len("SynResult_"):])

        m_res_single = re.match(r"^Result\[\s*(.+?)\s*\]$", t, re.IGNORECASE)
        if m_res_single:
            res_c = self.register_result_type(m_res_single.group(1), "str")
            return (res_c, res_c[len("SynResult_"):])

        if t_lower in ("int", "i32", "i64", "long"):
            return ("int", "int")
        elif t_lower in ("float", "double", "f32", "f64"):
            return ("double", "double")
        elif t_lower in ("bool", "boolean"):
            return ("int", "bool")
        elif t_lower in ("str", "string", "char*", "const char*"):
            return ("const char*", "str")
        elif t_lower == "void":
            return ("void", "void")
        elif "tensor" in t_lower or "syn_tensor" in t_lower:
            return ("syn_tensor_t*", "tensor")
        else:
            clean_suf = re.sub(r"[^a-zA-Z0-9_]", "_", t.replace("*", "_ptr"))
            return (t, clean_suf)

    def register_option_type(self, inner_type_annot: Any) -> str:
        """Verilen iç tip için monomorfik Option tipini kaydeder ve C adını döner."""
        c_type, suffix = self.get_type_info(inner_type_annot)
        if suffix not in self.registered_options:
            self.registered_options[suffix] = (c_type, suffix)
        return f"SynOption_{suffix}"

    def register_result_type(self, ok_type_annot: Any, err_type_annot: Any = "str") -> str:
        """Verilen Ok ve Err tipleri için monomorfik Result tipini kaydeder ve C adını döner."""
        c_t, s_t = self.get_type_info(ok_type_annot)
        c_e, s_e = self.get_type_info(err_type_annot)
        key = f"{s_t}_{s_e}"
        if key not in self.registered_results:
            self.registered_results[key] = (c_t, s_t, c_e, s_e)
        return f"SynResult_{key}"

    def emit_monomorphic_typedefs(self) -> list[str]:
        """Tüm kayıtlı Option[T] ve Result[T, E] türleri için C99 Tagged Union struct ve inline yapıcılarını üretir."""
        if not self.registered_options and not self.registered_results:
            return []

        lines: list[str] = [
            "/* ========================================================================= */",
            "/* C99 Monomorphic Tagged Unions (Option & Result)                           */",
            "/* ========================================================================= */",
        ]

        if self.registered_options:
            lines.extend([
                "#ifndef SYN_OPT_TAG_DEFINED",
                "#define SYN_OPT_TAG_DEFINED",
                "typedef enum { SYN_OPT_NONE = 0, SYN_OPT_SOME = 1 } SynOptTag;",
                "#endif",
                "",
            ])
            for suffix, (c_type, _) in sorted(self.registered_options.items()):
                lines.extend([
                    f"typedef struct {{ SynOptTag tag; union {{ {c_type} val; }} as; }} SynOption_{suffix};",
                    f"static inline SynOption_{suffix} syn_option_{suffix}_some({c_type} v) {{",
                    f"    SynOption_{suffix} o; o.tag = SYN_OPT_SOME; o.as.val = v; return o;",
                    f"}}",
                    f"static inline SynOption_{suffix} syn_option_{suffix}_none(void) {{",
                    f"    SynOption_{suffix} o; o.tag = SYN_OPT_NONE; memset(&o.as, 0, sizeof(o.as)); return o;",
                    f"}}",
                    "",
                ])

        if self.registered_results:
            lines.extend([
                "#ifndef SYN_RES_TAG_DEFINED",
                "#define SYN_RES_TAG_DEFINED",
                "typedef enum { SYN_RES_ERR = 0, SYN_RES_OK = 1 } SynResTag;",
                "#endif",
                "",
            ])
            for key, (c_type_t, suffix_t, c_type_e, suffix_e) in sorted(self.registered_results.items()):
                lines.extend([
                    f"typedef struct {{ SynResTag tag; union {{ {c_type_t} ok; {c_type_e} err; }} as; }} SynResult_{key};",
                    f"static inline SynResult_{key} syn_result_{key}_ok({c_type_t} v) {{",
                    f"    SynResult_{key} r; r.tag = SYN_RES_OK; r.as.ok = v; return r;",
                    f"}}",
                    f"static inline SynResult_{key} syn_result_{key}_err({c_type_e} e) {{",
                    f"    SynResult_{key} r; r.tag = SYN_RES_ERR; r.as.err = e; return r;",
                    f"}}",
                    "",
                ])

        return lines

    def map_type(self, type_annot: Any, default: str = "syn_tensor_t*") -> str:
        if not type_annot:
            return default

        # GenericType kontrolü
        if isinstance(type_annot, GenericType):
            if type_annot.base == "Option" and type_annot.type_args:
                return self.register_option_type(type_annot.type_args[0])
            elif type_annot.base == "Result":
                t_arg = type_annot.type_args[0] if len(type_annot.type_args) > 0 else "int"
                e_arg = type_annot.type_args[1] if len(type_annot.type_args) > 1 else "str"
                return self.register_result_type(t_arg, e_arg)
            elif type_annot.base in ("Tensor", "syn_tensor"):
                return "syn_tensor_t*"

        raw_str = getattr(type_annot, "raw", type_annot)
        if not isinstance(raw_str, str):
            raw_str = str(raw_str)
        t = raw_str.strip()
        t_lower = t.lower()

        # String Option[...] kontrolü
        m_opt = re.match(r"^Option\[\s*(.+?)\s*\]$", t, re.IGNORECASE)
        if m_opt:
            return self.register_option_type(m_opt.group(1))

        # String Result[..., ...] kontrolü
        m_res = re.match(r"^Result\[\s*(.+?)\s*,\s*(.+?)\s*\]$", t, re.IGNORECASE)
        if m_res:
            return self.register_result_type(m_res.group(1), m_res.group(2))

        m_res_single = re.match(r"^Result\[\s*(.+?)\s*\]$", t, re.IGNORECASE)
        if m_res_single:
            return self.register_result_type(m_res_single.group(1), "str")

        if t_lower in ("int", "i32", "i64", "long"):
            return "int"
        elif t_lower in ("float", "double", "f32", "f64"):
            return "double"
        elif t_lower in ("bool", "boolean"):
            return "int"
        elif t_lower in ("str", "string", "char*", "const char*"):
            return "const char*"
        elif t_lower in ("void",):
            return "void"
        elif "tensor" in t_lower or "syn_tensor" in t_lower:
            return "syn_tensor_t*"
        return default

    def emit(self, program: Program, filename: str = "<source>") -> str:
        body_lines: list[str] = []
        self.current_filename = (filename or "<source>").replace("\\", "/")
        self.last_emitted_line = -1
        self.prototypes.clear()
        self.functions_code.clear()
        self.function_return_types.clear()
        self.loop_counter = 0
        self.match_counter = 0
        self.try_counter = 0
        self.current_loop_arena = None
        self.current_loop_scope = None
        self.current_fn_return_type = None
        self.expected_type_stack.clear()
        self.pre_stmts.clear()
        self.scope_stack = [{}]

        # 1. Faz: Fonksiyon imzalarını ve dönüş tiplerini topla
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                ret_t = self._infer_fn_return_type(stmt)
                self.function_return_types[stmt.name] = ret_t
                for p in stmt.params:
                    if p.type_annot:
                        self.map_type(p.type_annot)

        # 2. Faz: Fonksiyonları ve ana gövdeyi emit et
        for stmt in program.statements:
            if isinstance(stmt, FunctionDef):
                self.functions_code.append(self.emit_function(stmt))
            else:
                self.current_fn_return_type = "int"
                body_lines.append(self.emit_stmt(stmt))
                self.current_fn_return_type = None

        c_code = [
            "/* Generated by Synapse Ahead-Of-Time (AOT) C Transpiler */",
            "/* High-Performance Hardware SIMD Optimization & Vectorization Hints */",
            "/* Recommended Compiler Flags: -O3 -mavx2 -mfma -fopenmp */",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "#include <string.h>",
            "#include <math.h>",
            '#include "synapse_runtime.h"',
            "",
            "/* High-Performance SIMD Vectorized Tensor Kernels (AVX2 / FMA / OpenMP SIMD) */",
            "#if defined(_OPENMP)",
            "#include <omp.h>",
            "#endif",
            "",
            "/* SIMD Kernel: Tensor Addition */",
            "static inline void syn_simd_tensor_add(const double* restrict a, const double* restrict b, double* restrict out, int size) {",
            "    #pragma omp simd",
            "    for (int i = 0; i < size; i++) {",
            "        out[i] = a[i] + b[i];",
            "    }",
            "}",
            "",
            "/* SIMD Kernel: Tensor Element-Wise Multiplication */",
            "static inline void syn_simd_tensor_mul(const double* restrict a, const double* restrict b, double* restrict out, int size) {",
            "    #pragma omp simd",
            "    for (int i = 0; i < size; i++) {",
            "        out[i] = a[i] * b[i];",
            "    }",
            "}",
            "",
            "/* SIMD Kernel: Matrix Multiplication (GEMM with FMA & Vectorization) */",
            "static inline void syn_simd_tensor_matmul(const double* restrict a, const double* restrict b, double* restrict out, int m, int k, int n) {",
            "    #pragma omp simd",
            "    for (int i = 0; i < m; i++) {",
            "        for (int j = 0; j < n; j++) {",
            "            double sum = 0.0;",
            "            #pragma omp simd reduction(+:sum)",
            "            for (int p = 0; p < k; p++) {",
            "                sum += a[i * k + p] * b[p * n + j];",
            "            }",
            "            out[i * n + j] = sum;",
            "        }",
            "    }",
            "}",
            "",
        ]

        monomorphic_defs = self.emit_monomorphic_typedefs()
        if monomorphic_defs:
            c_code.extend(monomorphic_defs)
            c_code.append("")

        if self.prototypes:
            c_code.append("/* Function Prototypes */")
            c_code.extend(self.prototypes)
            c_code.append("")

        if self.functions_code:
            c_code.append("/* Function Definitions */")
            c_code.extend(self.functions_code)
            c_code.append("")

        c_code.append("int main(int argc, char** argv) {")
        for line in body_lines:
            if line:
                c_code.append(line)
        c_code.append("    return 0;")
        c_code.append("}")
        c_code.append("")

        return "\n".join(c_code)

    def _infer_fn_return_type(self, fn: FunctionDef) -> str:
        if fn.return_type:
            return self.map_type(fn.return_type, "syn_tensor_t*")

        for s in fn.body:
            if isinstance(s, ReturnStmt):
                if s.value is None:
                    return "void"
                _, v_type = self.emit_expr(s.value)
                return v_type

        return "void"

    def emit_function(self, fn: FunctionDef) -> str:
        saved_last_line = self.last_emitted_line
        self.last_emitted_line = -1
        line_dir = self._format_line_directive(fn, force=True)
        ret_type = self.function_return_types.get(fn.name) or self._infer_fn_return_type(fn)
        self.current_fn_return_type = ret_type

        param_decls: list[str] = []
        fn_scope_params: dict[str, str] = {}

        for p in fn.params:
            p_type = self.map_type(p.type_annot, "syn_tensor_t*")
            param_decls.append(f"{p_type} {p.name}")
            fn_scope_params[p.name] = p_type

        params_str = ", ".join(param_decls) if param_decls else "void"

        proto = f"{ret_type} {fn.name}({params_str});"
        if proto not in self.prototypes:
            self.prototypes.append(proto)

        lines = []
        if line_dir:
            lines.append(line_dir)
        lines.append(f"{ret_type} {fn.name}({params_str}) {{")
        self.indent_level += 1
        self.push_scope()

        for name, p_type in fn_scope_params.items():
            self.set_var(name, p_type)

        for s in fn.body:
            stmt_code = self.emit_stmt(s)
            if stmt_code:
                lines.append(stmt_code)

        self.pop_scope()
        self.indent_level -= 1
        lines.append("}")
        self.last_emitted_line = saved_last_line
        self.current_fn_return_type = None
        return "\n".join(lines)

    def emit_stmt(self, stmt: Stmt) -> str:
        line_dir = self._format_line_directive(stmt)
        saved_pre = self.pre_stmts
        self.pre_stmts = []

        if isinstance(stmt, VarDeclStmt):
            c_type = self.map_type(stmt.type_annot, "syn_tensor_t*") if stmt.type_annot else None
            self.expected_type_stack.append(c_type)
            expr_code, expr_type = self.emit_expr(stmt.value)
            self.expected_type_stack.pop()
            if c_type is None:
                c_type = expr_type
            self.set_var(stmt.name, c_type)
            const_prefix = "const " if stmt.is_const else ""
            stmt_code = f"{self.indent()}{const_prefix}{c_type} {stmt.name} = {expr_code};"

        elif isinstance(stmt, AssignStmt):
            if isinstance(stmt.target, IdentifierExpr):
                target_name = stmt.target.name
                target_type = self.lookup_var(target_name)
                is_outer = (len(self.scope_stack) > 1 and target_name not in self.scope_stack[-1])
                self.expected_type_stack.append(target_type)
                expr_code, expr_type = self.emit_expr(stmt.value)
                self.expected_type_stack.pop()

                if target_type is None:
                    # İlk kez atanıyor -> tip tespiti ile deklare et
                    self.set_var(target_name, expr_type)
                    stmt_code = f"{self.indent()}{expr_type} {target_name} = {expr_code};"
                elif target_type == "syn_tensor_t*":
                    # Escape koruması: Dış kapsamdaki tensöre döngü içinden atama yapılıyorsa
                    if self.current_loop_arena and is_outer:
                        expr_code = f"syn_tensor_escape({expr_code}, {self.current_loop_arena})"
                    # Tensör bileşik atamaları
                    if stmt.op == "=":
                        stmt_code = f"{self.indent()}{target_name} = {expr_code};"
                    elif stmt.op == "+=":
                        val = f"syn_add({target_name}, {expr_code})"
                        if self.current_loop_arena and is_outer:
                            val = f"syn_tensor_escape({val}, {self.current_loop_arena})"
                        stmt_code = f"{self.indent()}{target_name} = {val};"
                    elif stmt.op == "-=":
                        val = f"syn_sub({target_name}, {expr_code})"
                        if self.current_loop_arena and is_outer:
                            val = f"syn_tensor_escape({val}, {self.current_loop_arena})"
                        stmt_code = f"{self.indent()}{target_name} = {val};"
                    elif stmt.op == "*=":
                        if expr_type in ("double", "int"):
                            val = f"syn_mul_scalar({target_name}, {expr_code})"
                        else:
                            val = f"syn_mul({target_name}, {expr_code})"
                        if self.current_loop_arena and is_outer:
                            val = f"syn_tensor_escape({val}, {self.current_loop_arena})"
                        stmt_code = f"{self.indent()}{target_name} = {val};"
                    elif stmt.op == "/=":
                        val = f"syn_div({target_name}, {expr_code})"
                        if self.current_loop_arena and is_outer:
                            val = f"syn_tensor_escape({val}, {self.current_loop_arena})"
                        stmt_code = f"{self.indent()}{target_name} = {val};"
                    elif stmt.op == "@=":
                        val = f"syn_matmul({target_name}, {expr_code})"
                        if self.current_loop_arena and is_outer:
                            val = f"syn_tensor_escape({val}, {self.current_loop_arena})"
                        stmt_code = f"{self.indent()}{target_name} = {val};"
                    else:
                        stmt_code = f"{self.indent()}{target_name} {stmt.op} {expr_code};"
                else:
                    # Skaler atama (int, double)
                    stmt_code = f"{self.indent()}{target_name} {stmt.op} {expr_code};"
            else:
                target_code, _ = self.emit_expr(stmt.target)
                val_code, val_type = self.emit_expr(stmt.value)
                if val_type == "syn_tensor_t*" and self.current_loop_arena:
                    val_code = f"syn_tensor_escape({val_code}, {self.current_loop_arena})"
                stmt_code = f"{self.indent()}{target_code} {stmt.op} {val_code};"

        elif isinstance(stmt, ExprStmt):
            expr_code, _ = self.emit_expr(stmt.expr)
            if expr_code.endswith(";"):
                stmt_code = f"{self.indent()}{expr_code}"
            else:
                stmt_code = f"{self.indent()}{expr_code};"

        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self.expected_type_stack.append(self.current_fn_return_type)
                val_code, val_type = self.emit_expr(stmt.value)
                self.expected_type_stack.pop()

                # Tensör döndüren fonksiyonda döngü arenası içindeysek güvenli referans aktarımı için escape
                if val_type == "syn_tensor_t*" and self.current_loop_arena:
                    stmt_code = f"{self.indent()}return syn_tensor_escape({val_code}, {self.current_loop_arena});"
                else:
                    stmt_code = f"{self.indent()}return {val_code};"
            else:
                stmt_code = f"{self.indent()}return;"

        elif isinstance(stmt, IfStmt):
            cond_code, _ = self.emit_expr(stmt.condition)
            lines = [f"{self.indent()}if ({cond_code}) {{"]

            self.indent_level += 1
            self.push_scope()
            for s in stmt.then_branch:
                code = self.emit_stmt(s)
                if code:
                    lines.append(code)
            self.pop_scope()
            self.indent_level -= 1

            for elif_cond, elif_body in stmt.elif_branches:
                elif_line_dir = self._format_line_directive(elif_cond)
                if elif_line_dir:
                    lines.append(elif_line_dir)
                elif_c, _ = self.emit_expr(elif_cond)
                lines.append(f"{self.indent()}}} else if ({elif_c}) {{")
                self.indent_level += 1
                self.push_scope()
                for s in elif_body:
                    code = self.emit_stmt(s)
                    if code:
                        lines.append(code)
                self.pop_scope()
                self.indent_level -= 1

            if stmt.else_branch:
                lines.append(f"{self.indent()}}} else {{")
                self.indent_level += 1
                self.push_scope()
                for s in stmt.else_branch:
                    code = self.emit_stmt(s)
                    if code:
                        lines.append(code)
                self.pop_scope()
                self.indent_level -= 1

            lines.append(f"{self.indent()}}}")
            stmt_code = "\n".join(lines)

        elif isinstance(stmt, WhileStmt):
            arena_name = "_loop_arena" if self.loop_counter == 0 else f"_loop_arena_{self.loop_counter}"
            scope_name = "_loop_scope" if self.loop_counter == 0 else f"_loop_scope_{self.loop_counter}"
            self.loop_counter += 1

            saved_arena = self.current_loop_arena
            saved_scope = self.current_loop_scope
            self.current_loop_arena = arena_name
            self.current_loop_scope = scope_name

            cond_code, _ = self.emit_expr(stmt.condition)
            lines = [
                f"{self.indent()}syn_arena_t* {arena_name} = syn_arena_create(1024 * 1024);",
                f"{self.indent()}while ({cond_code}) {{",
            ]
            self.indent_level += 1
            self.push_scope()
            lines.append(f"{self.indent()}syn_arena_scope_t {scope_name} = syn_arena_scope_enter({arena_name});")

            for s in stmt.body:
                code = self.emit_stmt(s)
                if code:
                    lines.append(code)

            lines.append(f"{self.indent()}syn_arena_scope_leave({scope_name});")
            self.pop_scope()
            self.indent_level -= 1
            lines.append(f"{self.indent()}}}")
            lines.append(f"{self.indent()}if ({arena_name}) syn_arena_free({arena_name});")

            self.current_loop_arena = saved_arena
            self.current_loop_scope = saved_scope
            stmt_code = "\n".join(lines)

        elif isinstance(stmt, ForStmt):
            arena_name = "_loop_arena" if self.loop_counter == 0 else f"_loop_arena_{self.loop_counter}"
            scope_name = "_loop_scope" if self.loop_counter == 0 else f"_loop_scope_{self.loop_counter}"
            self.loop_counter += 1

            saved_arena = self.current_loop_arena
            saved_scope = self.current_loop_scope
            self.current_loop_arena = arena_name
            self.current_loop_scope = scope_name

            self.push_scope()
            self.set_var(stmt.target, "int")
            loop_header = self._emit_for_header(stmt)
            lines = [
                f"{self.indent()}syn_arena_t* {arena_name} = syn_arena_create(1024 * 1024);",
            ]
            if getattr(stmt, "is_simd", False) or self._is_simd_eligible(stmt):
                lines.append(f"{self.indent()}#pragma omp simd")
            lines.append(f"{self.indent()}{loop_header} {{")
            self.indent_level += 1
            lines.append(f"{self.indent()}syn_arena_scope_t {scope_name} = syn_arena_scope_enter({arena_name});")

            for s in stmt.body:
                code = self.emit_stmt(s)
                if code:
                    lines.append(code)

            lines.append(f"{self.indent()}syn_arena_scope_leave({scope_name});")
            self.indent_level -= 1
            self.pop_scope()
            lines.append(f"{self.indent()}}}")
            lines.append(f"{self.indent()}if ({arena_name}) syn_arena_free({arena_name});")

            self.current_loop_arena = saved_arena
            self.current_loop_scope = saved_scope
            stmt_code = "\n".join(lines)

        elif isinstance(stmt, MatchStmt):
            stmt_code = self.emit_match_stmt(stmt)

        elif isinstance(stmt, BreakStmt):
            if self.current_loop_scope:
                stmt_code = f"{self.indent()}syn_arena_scope_leave({self.current_loop_scope});\n{self.indent()}break;"
            else:
                stmt_code = f"{self.indent()}break;"

        elif isinstance(stmt, ContinueStmt):
            if self.current_loop_scope:
                stmt_code = f"{self.indent()}syn_arena_scope_leave({self.current_loop_scope});\n{self.indent()}continue;"
            else:
                stmt_code = f"{self.indent()}continue;"

        elif isinstance(stmt, PassStmt):
            stmt_code = f"{self.indent()}/* pass */;"

        else:
            stmt_code = f"{self.indent()}/* unhandled statement */;"

        if line_dir:
            stmt_code = f"{line_dir}\n{stmt_code}"
        if self.pre_stmts:
            stmt_code = "\n".join(self.pre_stmts) + "\n" + stmt_code
        self.pre_stmts = saved_pre
        return stmt_code

    def emit_match_stmt(self, stmt: MatchStmt) -> str:
        """Pattern Matching (MatchStmt) ifadesini C switch/if-else bloklarına dönüştürür."""
        self.match_counter += 1
        subj_var = f"_match_subj_{self.match_counter}"
        subj_code, subj_type = self.emit_expr(stmt.subject)

        lines = [
            f"{self.indent()}{{",
            f"{self.indent()}    {subj_type} {subj_var} = {subj_code};",
        ]
        self.indent_level += 1

        if subj_type.startswith("SynOption_"):
            suf = subj_type[len("SynOption_"):]
            inner_type = self.registered_options.get(suf, ("int", "int"))[0]
            lines.append(f"{self.indent()}switch ({subj_var}.tag) {{")
            self.indent_level += 1

            for case in stmt.cases:
                p = case.pattern
                is_some = False
                is_none = False
                is_wildcard = False
                var_name = None

                if isinstance(p, LiteralExpr) and p.value is None:
                    is_none = True
                elif isinstance(p, CallExpr):
                    callee_name = getattr(p.callee, "name", "")
                    callee_member = getattr(p.callee, "member", "")
                    if callee_name == "Some" or callee_member == "Some":
                        is_some = True
                        if p.args and isinstance(p.args[0], IdentifierExpr):
                            var_name = p.args[0].name
                elif isinstance(p, IdentifierExpr):
                    if p.name == "None":
                        is_none = True
                    elif p.name == "_":
                        is_wildcard = True
                    else:
                        is_wildcard = True
                        var_name = p.name
                elif isinstance(p, MemberExpr):
                    if p.member == "None":
                        is_none = True
                elif type(p).__name__ in ("_NoneOption", "NoneOption"):
                    is_none = True

                if is_some:
                    lines.append(f"{self.indent()}case SYN_OPT_SOME: {{")
                    self.indent_level += 1
                    self.push_scope()
                    if var_name:
                        self.set_var(var_name, inner_type)
                        lines.append(f"{self.indent()}{inner_type} {var_name} = {subj_var}.as.val;")
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    lines.append(f"{self.indent()}break;")
                    self.indent_level -= 1
                    lines.append(f"{self.indent()}}}")
                elif is_none:
                    lines.append(f"{self.indent()}case SYN_OPT_NONE: {{")
                    self.indent_level += 1
                    self.push_scope()
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    lines.append(f"{self.indent()}break;")
                    self.indent_level -= 1
                    lines.append(f"{self.indent()}}}")
                elif is_wildcard:
                    lines.append(f"{self.indent()}default: {{")
                    self.indent_level += 1
                    self.push_scope()
                    if var_name:
                        self.set_var(var_name, subj_type)
                        lines.append(f"{self.indent()}{subj_type} {var_name} = {subj_var};")
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    lines.append(f"{self.indent()}break;")
                    self.indent_level -= 1
                    lines.append(f"{self.indent()}}}")

            self.indent_level -= 1
            lines.append(f"{self.indent()}}}")

        elif subj_type.startswith("SynResult_"):
            res_key = subj_type[len("SynResult_"):]
            info = self.registered_results.get(res_key)
            ok_type = info[0] if info else "int"
            err_type = info[2] if info else "const char*"

            lines.append(f"{self.indent()}switch ({subj_var}.tag) {{")
            self.indent_level += 1

            for case in stmt.cases:
                p = case.pattern
                is_ok = False
                is_err = False
                is_wildcard = False
                var_name = None

                if isinstance(p, CallExpr):
                    callee_name = getattr(p.callee, "name", "")
                    callee_member = getattr(p.callee, "member", "")
                    if callee_name == "Ok" or callee_member == "Ok":
                        is_ok = True
                        if p.args and isinstance(p.args[0], IdentifierExpr):
                            var_name = p.args[0].name
                    elif callee_name == "Err" or callee_member == "Err":
                        is_err = True
                        if p.args and isinstance(p.args[0], IdentifierExpr):
                            var_name = p.args[0].name
                elif isinstance(p, IdentifierExpr):
                    if p.name == "_":
                        is_wildcard = True
                    else:
                        is_wildcard = True
                        var_name = p.name

                if is_ok:
                    lines.append(f"{self.indent()}case SYN_RES_OK: {{")
                    self.indent_level += 1
                    self.push_scope()
                    if var_name:
                        self.set_var(var_name, ok_type)
                        lines.append(f"{self.indent()}{ok_type} {var_name} = {subj_var}.as.ok;")
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    lines.append(f"{self.indent()}break;")
                    self.indent_level -= 1
                    lines.append(f"{self.indent()}}}")
                elif is_err:
                    lines.append(f"{self.indent()}case SYN_RES_ERR: {{")
                    self.indent_level += 1
                    self.push_scope()
                    if var_name:
                        self.set_var(var_name, err_type)
                        lines.append(f"{self.indent()}{err_type} {var_name} = {subj_var}.as.err;")
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    lines.append(f"{self.indent()}break;")
                    self.indent_level -= 1
                    lines.append(f"{self.indent()}}}")
                elif is_wildcard:
                    lines.append(f"{self.indent()}default: {{")
                    self.indent_level += 1
                    self.push_scope()
                    if var_name:
                        self.set_var(var_name, subj_type)
                        lines.append(f"{self.indent()}{subj_type} {var_name} = {subj_var};")
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    lines.append(f"{self.indent()}break;")
                    self.indent_level -= 1
                    lines.append(f"{self.indent()}}}")

            self.indent_level -= 1
            lines.append(f"{self.indent()}}}")

        else:
            first = True
            for case in stmt.cases:
                p = case.pattern
                if isinstance(p, IdentifierExpr) and p.name == "_":
                    lines.append(f"{self.indent()}}} else {{")
                    self.indent_level += 1
                    self.push_scope()
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    self.indent_level -= 1
                else:
                    pat_code, _ = self.emit_expr(p)
                    cond = f"{subj_var} == {pat_code}"
                    if first:
                        lines.append(f"{self.indent()}if ({cond}) {{")
                        first = False
                    else:
                        lines.append(f"{self.indent()}}} else if ({cond}) {{")
                    self.indent_level += 1
                    self.push_scope()
                    for s in case.body:
                        code = self.emit_stmt(s)
                        if code:
                            lines.append(code)
                    self.pop_scope()
                    self.indent_level -= 1
            if not first:
                lines.append(f"{self.indent()}}}")

        self.indent_level -= 1
        lines.append(f"{self.indent()}}}")
        return "\n".join(lines)

    def _emit_for_header(self, stmt: ForStmt) -> str:
        """For döngü başlığını range veya sayaç durumuna göre C for sözdizimine dönüştürür."""
        target = stmt.target
        iterable = stmt.iterable

        # 1. range(...) çağrısı
        if isinstance(iterable, CallExpr) and isinstance(iterable.callee, IdentifierExpr) and iterable.callee.name == "range":
            args = iterable.args
            if len(args) == 1:
                stop_c, _ = self.emit_expr(args[0])
                return f"for (int {target} = 0; {target} < {stop_c}; {target}++)"
            elif len(args) == 2:
                start_c, _ = self.emit_expr(args[0])
                stop_c, _ = self.emit_expr(args[1])
                return f"for (int {target} = {start_c}; {target} < {stop_c}; {target}++)"
            elif len(args) >= 3:
                start_c, _ = self.emit_expr(args[0])
                stop_c, _ = self.emit_expr(args[1])
                step_c, _ = self.emit_expr(args[2])

                if step_c.startswith("(") and step_c.endswith(")"):
                    inner = step_c[1:-1].strip()
                    if not any(op in inner for op in (" ", "+", "*", "/", "%", "&", "|")):
                        step_c = inner

                is_neg = False
                if isinstance(args[2], LiteralExpr) and isinstance(args[2].value, (int, float)) and args[2].value < 0:
                    is_neg = True
                elif isinstance(args[2], UnaryExpr) and args[2].op == "-":
                    is_neg = True

                if is_neg:
                    return f"for (int {target} = {start_c}; {target} > {stop_c}; {target} += {step_c})"
                else:
                    return f"for (int {target} = {start_c}; {target} < {stop_c}; {target} += {step_c})"

        # 2. Doğrudan sayı literali: for i in 10
        if isinstance(iterable, LiteralExpr) and isinstance(iterable.value, int):
            return f"for (int {target} = 0; {target} < {iterable.value}; {target}++)"

        # 3. Değişken veya genel ifade (sayaç / tensör boyutu)
        iter_c, iter_type = self.emit_expr(iterable)
        if iter_type == "int":
            return f"for (int {target} = 0; {target} < {iter_c}; {target}++)"
        elif iter_type == "syn_tensor_t*":
            return f"for (int {target} = 0; {target} < (int)({iter_c}->size); {target}++)"

        return f"for (int {target} = 0; {target} < {iter_c}; {target}++)"

    def _is_simd_eligible(self, stmt: ForStmt) -> bool:
        """Determines if a loop is vectorizable with SIMD directives."""
        if hasattr(stmt, "is_simd") and stmt.is_simd:
            return True
        if isinstance(stmt.iterable, IdentifierExpr):
            t = self.lookup_var(stmt.iterable.name)
            if t == "syn_tensor_t*":
                return True
        return False

    def emit_simd_tensor_add_loop(self, a_var: str = "a", b_var: str = "b", out_var: str = "out", size_var: str = "size") -> str:
        """Emits a high-performance vector-friendly C loop for tensor addition with SIMD pragmas."""
        lines = [
            f"{self.indent()}#pragma omp simd",
            f"{self.indent()}for (int i = 0; i < {size_var}; i++) {{",
            f"{self.indent()}    {out_var}[i] = {a_var}[i] + {b_var}[i];",
            f"{self.indent()}}}",
        ]
        return "\n".join(lines)

    def emit_simd_tensor_mul_loop(self, a_var: str = "a", b_var: str = "b", out_var: str = "out", size_var: str = "size") -> str:
        """Emits a high-performance vector-friendly C loop for tensor multiplication with SIMD pragmas."""
        lines = [
            f"{self.indent()}#pragma omp simd",
            f"{self.indent()}for (int i = 0; i < {size_var}; i++) {{",
            f"{self.indent()}    {out_var}[i] = {a_var}[i] * {b_var}[i];",
            f"{self.indent()}}}",
        ]
        return "\n".join(lines)

    def emit_simd_tensor_matmul_loop(
        self, a_var: str = "a", b_var: str = "b", out_var: str = "out", m_var: str = "m", k_var: str = "k", n_var: str = "n"
    ) -> str:
        """Emits a high-performance vector-friendly nested C loop for GEMM matrix multiplication with SIMD directives."""
        lines = [
            f"{self.indent()}#pragma omp simd",
            f"{self.indent()}for (int i = 0; i < {m_var}; i++) {{",
            f"{self.indent()}    for (int j = 0; j < {n_var}; j++) {{",
            f"{self.indent()}        double sum = 0.0;",
            f"{self.indent()}        #pragma omp simd reduction(+:sum)",
            f"{self.indent()}        for (int p = 0; p < {k_var}; p++) {{",
            f"{self.indent()}            sum += {a_var}[i * {k_var} + p] * {b_var}[p * {n_var} + j];",
            f"{self.indent()}        }}",
            f"{self.indent()}        {out_var}[i * {n_var} + j] = sum;",
            f"{self.indent()}    }}",
            f"{self.indent()}}}",
        ]
        return "\n".join(lines)

    def emit_expr(self, expr: Expr) -> tuple[str, str]:
        """Dönüş: (c_code, c_type)"""
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, bool):
                return ("1" if expr.value else "0", "int")
            elif isinstance(expr.value, int):
                return (str(expr.value), "int")
            elif isinstance(expr.value, float):
                return (str(expr.value), "double")
            elif isinstance(expr.value, str):
                escaped = expr.value.replace('"', '\\"').replace("\n", "\\n")
                return (f'"{escaped}"', "const char*")
            elif expr.value is None:
                expected = self.expected_type_stack[-1] if self.expected_type_stack else None
                ret_t = expected or self.current_fn_return_type or "SynOption_int"
                if ret_t.startswith("SynOption_"):
                    suf = ret_t[len("SynOption_"):]
                else:
                    suf = "int"
                    self.register_option_type("int")
                return (f"syn_option_{suf}_none()", f"SynOption_{suf}")
            return ("0", "int")

        elif isinstance(expr, IdentifierExpr):
            if expr.name == "None":
                expected = self.expected_type_stack[-1] if self.expected_type_stack else None
                if expected and expected.startswith("SynOption_"):
                    suf = expected[len("SynOption_"):]
                else:
                    suf = "int"
                    self.register_option_type("int")
                return (f"syn_option_{suf}_none()", f"SynOption_{suf}")
            var_type = self.lookup_var(expr.name) or "double"
            return (expr.name, var_type)

        elif isinstance(expr, TensorLiteralExpr):
            return self.emit_tensor_literal(expr)

        elif isinstance(expr, TryExpr):
            self.try_counter += 1
            try_var = f"_try_res_{self.try_counter}"
            inner_code, inner_type = self.emit_expr(expr.expr)

            if inner_type.startswith("SynResult_"):
                res_key = inner_type[len("SynResult_"):]
                info = self.registered_results.get(res_key)
                ok_type = info[0] if info else "int"
                err_type = info[2] if info else "const char*"

                # Enclosing fonksiyon Result dönüyorsa ona uygun Result_err, aksi halde return _try_res
                if self.current_fn_return_type and self.current_fn_return_type.startswith("SynResult_"):
                    fn_res_key = self.current_fn_return_type[len("SynResult_"):]
                    if fn_res_key == res_key:
                        early_ret = f"return {try_var};"
                    else:
                        parts = fn_res_key.split("_")
                        fn_t = parts[0]
                        fn_e = parts[1] if len(parts) > 1 else "str"
                        early_ret = f"return syn_result_{fn_t}_{fn_e}_err({try_var}.as.err);"
                elif self.current_fn_return_type and self.current_fn_return_type.startswith("SynOption_"):
                    fn_suf = self.current_fn_return_type[len("SynOption_"):]
                    early_ret = f"return syn_option_{fn_suf}_none();"
                else:
                    early_ret = f"return {try_var};"

                self.pre_stmts.append(f"{self.indent()}{inner_type} {try_var} = {inner_code};")
                self.pre_stmts.append(f"{self.indent()}if ({try_var}.tag == SYN_RES_ERR) {{")
                self.pre_stmts.append(f"{self.indent()}    {early_ret}")
                self.pre_stmts.append(f"{self.indent()}}}")
                return (f"{try_var}.as.ok", ok_type)

            elif inner_type.startswith("SynOption_"):
                opt_suf = inner_type[len("SynOption_"):]
                info = self.registered_options.get(opt_suf)
                val_type = info[0] if info else "int"

                if self.current_fn_return_type and self.current_fn_return_type.startswith("SynOption_"):
                    fn_suf = self.current_fn_return_type[len("SynOption_"):]
                    early_ret = f"return syn_option_{fn_suf}_none();"
                else:
                    early_ret = f"return syn_option_{opt_suf}_none();"

                self.pre_stmts.append(f"{self.indent()}{inner_type} {try_var} = {inner_code};")
                self.pre_stmts.append(f"{self.indent()}if ({try_var}.tag == SYN_OPT_NONE) {{")
                self.pre_stmts.append(f"{self.indent()}    {early_ret}")
                self.pre_stmts.append(f"{self.indent()}}}")
                return (f"{try_var}.as.val", val_type)

            return (inner_code, inner_type)

        elif isinstance(expr, PipeExpr):
            # left |> right => right(left)
            left_code, _ = self.emit_expr(expr.left)
            if isinstance(expr.right, IdentifierExpr):
                # Özel fonksiyon mu? örn. relu, sigmoid, tanh, gelu, softmax, sum, mean
                if expr.right.name == "relu":
                    return (f"syn_relu({left_code})", "syn_tensor_t*")
                elif expr.right.name == "sigmoid":
                    return (f"syn_sigmoid({left_code})", "syn_tensor_t*")
                elif expr.right.name == "tanh":
                    return (f"syn_tanh({left_code})", "syn_tensor_t*")
                elif expr.right.name == "gelu":
                    return (f"syn_gelu({left_code})", "syn_tensor_t*")
                elif expr.right.name == "softmax":
                    return (f"syn_softmax({left_code})", "syn_tensor_t*")
                elif expr.right.name == "sum":
                    return (f"syn_sum({left_code})", "syn_tensor_t*")
                elif expr.right.name == "mean":
                    return (f"syn_mean({left_code})", "syn_tensor_t*")
                return (f"{expr.right.name}({left_code})", "syn_tensor_t*")
            elif isinstance(expr.right, CallExpr):
                right_callee = expr.right.callee
                callee_name = right_callee.name if hasattr(right_callee, "name") else "fn"
                args_code = [left_code] + [self.emit_expr(a)[0] for a in expr.right.args]
                return (f"{callee_name}({', '.join(args_code)})", "syn_tensor_t*")

        elif isinstance(expr, BinaryExpr):
            left_code, left_type = self.emit_expr(expr.left)
            right_code, right_type = self.emit_expr(expr.right)

            # Tensör işlemi mi?
            is_tensor_op = ("syn_tensor_t*" in (left_type, right_type))

            if is_tensor_op:
                if expr.op == "@":
                    return (f"syn_matmul({left_code}, {right_code})", "syn_tensor_t*")
                elif expr.op == "+":
                    if left_type in ("double", "int"):
                        return (f"syn_add_scalar({right_code}, {left_code})", "syn_tensor_t*")
                    elif right_type in ("double", "int"):
                        return (f"syn_add_scalar({left_code}, {right_code})", "syn_tensor_t*")
                    return (f"syn_add({left_code}, {right_code})", "syn_tensor_t*")
                elif expr.op == "-":
                    if right_type in ("double", "int"):
                        return (f"syn_sub_scalar({left_code}, {right_code})", "syn_tensor_t*")
                    return (f"syn_sub({left_code}, {right_code})", "syn_tensor_t*")
                elif expr.op == "*":
                    if left_type == "double" or left_type == "int":
                        return (f"syn_mul_scalar({right_code}, {left_code})", "syn_tensor_t*")
                    elif right_type == "double" or right_type == "int":
                        return (f"syn_mul_scalar({left_code}, {right_code})", "syn_tensor_t*")
                    return (f"syn_mul({left_code}, {right_code})", "syn_tensor_t*")
                elif expr.op == "/":
                    return (f"syn_div({left_code}, {right_code})", "syn_tensor_t*")
                elif expr.op == "**":
                    return (f"syn_pow_scalar({left_code}, {right_code})", "syn_tensor_t*")

            # Skaler aritmetik / karşılaştırma
            if expr.op == "%":
                return (f"((int){left_code} % (int){right_code})", "int")
            elif expr.op in ("==", "!=", "<", "<=", ">", ">="):
                return (f"({left_code} {expr.op} {right_code})", "int")
            elif expr.op == "and":
                return (f"({left_code} && {right_code})", "int")
            elif expr.op == "or":
                return (f"({left_code} || {right_code})", "int")
            else:
                ret_type = "int" if (left_type == "int" and right_type == "int" and expr.op in ("+", "-", "*")) else "double"
                return (f"({left_code} {expr.op} {right_code})", ret_type)

        elif isinstance(expr, UnaryExpr):
            op_code, op_type = self.emit_expr(expr.operand)
            if expr.op == "-":
                if op_type == "syn_tensor_t*":
                    return (f"syn_mul_scalar({op_code}, -1.0)", "syn_tensor_t*")
                return (f"(-{op_code})", op_type)
            elif expr.op == "not":
                return (f"(!{op_code})", "int")

        elif isinstance(expr, MemberExpr):
            if isinstance(expr.target, IdentifierExpr) and expr.target.name == "Option" and expr.member == "None":
                expected = self.expected_type_stack[-1] if self.expected_type_stack else None
                if expected and expected.startswith("SynOption_"):
                    suf = expected[len("SynOption_"):]
                else:
                    suf = "int"
                    self.register_option_type("int")
                return (f"syn_option_{suf}_none()", f"SynOption_{suf}")
            target_code, _ = self.emit_expr(expr.target)
            if expr.member == "T":
                return (f"syn_transpose({target_code})", "syn_tensor_t*")
            elif expr.member == "grad":
                return (f"({target_code}->grad)", "syn_tensor_t*")
            return (f"({target_code}->{expr.member})", "double")

        elif isinstance(expr, IndexExpr):
            target_code, target_type = self.emit_expr(expr.target)
            index_code, _ = self.emit_expr(expr.index)
            if target_type == "syn_tensor_t*":
                return (f"{target_code}->data[(int)({index_code})]", "double")
            return (f"{target_code}[(int)({index_code})]", "double")

        elif isinstance(expr, CallExpr):
            return self.emit_call(expr)

        return ("0", "int")

    def emit_call(self, call: CallExpr) -> tuple[str, str]:
        # Option / Result Yapıcıları: Some, None, Ok, Err (ve Option.Some, Result.Ok vb.)
        is_some = False
        if isinstance(call.callee, IdentifierExpr) and call.callee.name == "Some":
            is_some = True
        elif isinstance(call.callee, MemberExpr) and isinstance(call.callee.target, IdentifierExpr) and call.callee.target.name == "Option" and call.callee.member == "Some":
            is_some = True

        if is_some:
            val_arg = call.args[0] if call.args else LiteralExpr(value=0)
            val_code, val_type = self.emit_expr(val_arg)
            expected = self.expected_type_stack[-1] if self.expected_type_stack else None
            if expected and expected.startswith("SynOption_"):
                suf = expected[len("SynOption_"):]
            else:
                _, suf = self.get_type_info(val_type)
                self.register_option_type(val_type)
            return (f"syn_option_{suf}_some({val_code})", f"SynOption_{suf}")

        is_none_call = False
        if isinstance(call.callee, IdentifierExpr) and call.callee.name == "None":
            is_none_call = True
        elif isinstance(call.callee, MemberExpr) and isinstance(call.callee.target, IdentifierExpr) and call.callee.target.name == "Option" and call.callee.member == "None":
            is_none_call = True

        if is_none_call:
            expected = self.expected_type_stack[-1] if self.expected_type_stack else None
            if expected and expected.startswith("SynOption_"):
                suf = expected[len("SynOption_"):]
            else:
                suf = "int"
                self.register_option_type("int")
            return (f"syn_option_{suf}_none()", f"SynOption_{suf}")

        is_ok = False
        if isinstance(call.callee, IdentifierExpr) and call.callee.name == "Ok":
            is_ok = True
        elif isinstance(call.callee, MemberExpr) and isinstance(call.callee.target, IdentifierExpr) and call.callee.target.name == "Result" and call.callee.member == "Ok":
            is_ok = True

        if is_ok:
            val_arg = call.args[0] if call.args else LiteralExpr(value=0)
            val_code, val_type = self.emit_expr(val_arg)
            expected = self.expected_type_stack[-1] if self.expected_type_stack else None
            if expected and expected.startswith("SynResult_"):
                key = expected[len("SynResult_"):]
            else:
                _, s_t = self.get_type_info(val_type)
                s_e = "str"
                key = f"{s_t}_{s_e}"
                self.register_result_type(val_type, "str")
            return (f"syn_result_{key}_ok({val_code})", f"SynResult_{key}")

        is_err = False
        if isinstance(call.callee, IdentifierExpr) and call.callee.name == "Err":
            is_err = True
        elif isinstance(call.callee, MemberExpr) and isinstance(call.callee.target, IdentifierExpr) and call.callee.target.name == "Result" and call.callee.member == "Err":
            is_err = True

        if is_err:
            err_arg = call.args[0] if call.args else LiteralExpr(value="error")
            err_code, err_type = self.emit_expr(err_arg)
            expected = self.expected_type_stack[-1] if self.expected_type_stack else None
            if expected and expected.startswith("SynResult_"):
                key = expected[len("SynResult_"):]
            else:
                s_t = "int"
                _, s_e = self.get_type_info(err_type)
                key = f"{s_t}_{s_e}"
                self.register_result_type("int", err_type)
            return (f"syn_result_{key}_err({err_code})", f"SynResult_{key}")

        # Özel yerleşik fonksiyonlar: print, tensor, zeros, ones, math
        if isinstance(call.callee, IdentifierExpr):
            name = call.callee.name
            if name == "print":
                return self.emit_print(call.args)
            elif name == "tensor":
                data_arg = call.args[0] if call.args else LiteralExpr(value=0.0)
                req_grad = 1 if (call.kwargs.get("requires_grad") and getattr(call.kwargs["requires_grad"], "value", False)) else 0
                return self.emit_tensor_constructor(data_arg, req_grad)
            elif name in ("sqrt", "exp", "log", "sin", "cos", "tan", "fabs"):
                arg_c = self.emit_expr(call.args[0])[0] if call.args else "0.0"
                return (f"{name}((double)({arg_c}))", "double")
            elif name == "pow":
                base_c = self.emit_expr(call.args[0])[0] if len(call.args) > 0 else "0.0"
                exp_c = self.emit_expr(call.args[1])[0] if len(call.args) > 1 else "1.0"
                return (f"pow((double)({base_c}), (double)({exp_c}))", "double")

        # Üye metod çağrısı: obj.item(), obj.backward(), obj.sum(), obj.mean(), obj.relu(), obj.zero_grad(), obj.clone()
        if isinstance(call.callee, MemberExpr):
            target_code, _ = self.emit_expr(call.callee.target)
            method = call.callee.member
            if method == "item":
                return (f"syn_tensor_item({target_code})", "double")
            elif method == "backward":
                return (f"syn_backward({target_code})", "void")
            elif method == "zero_grad":
                return (f"syn_tensor_zero_grad({target_code})", "void")
            elif method == "clone":
                return (f"syn_tensor_clone({target_code})", "syn_tensor_t*")
            elif method == "sum":
                return (f"syn_sum({target_code})", "syn_tensor_t*")
            elif method == "mean":
                return (f"syn_mean({target_code})", "syn_tensor_t*")
            elif method == "relu":
                return (f"syn_relu({target_code})", "syn_tensor_t*")
            elif method == "sigmoid":
                return (f"syn_sigmoid({target_code})", "syn_tensor_t*")
            elif method == "tanh":
                return (f"syn_tanh({target_code})", "syn_tensor_t*")
            elif method == "gelu":
                return (f"syn_gelu({target_code})", "syn_tensor_t*")
            elif method == "softmax":
                return (f"syn_softmax({target_code})", "syn_tensor_t*")

        # Standart kullanıcı tanımlı fonksiyon çağrısı
        callee_code, _ = self.emit_expr(call.callee)
        args_code = [self.emit_expr(a)[0] for a in call.args]
        ret_type = "syn_tensor_t*"
        if isinstance(call.callee, IdentifierExpr):
            ret_type = self.function_return_types.get(call.callee.name, "syn_tensor_t*")

        return (f"{callee_code}({', '.join(args_code)})", ret_type)

    def emit_print(self, args: list[Expr]) -> tuple[str, str]:
        # C printf veya syn_tensor_print çağrısı
        if not args:
            return ('printf("\\n")', "void")

        if len(args) == 1:
            code, arg_type = self.emit_expr(args[0])
            if arg_type == "syn_tensor_t*":
                return (f'syn_tensor_print("", {code})', "void")
            elif arg_type == "const char*":
                return (f'printf("%s\\n", {code})', "void")
            elif arg_type == "int":
                return (f'printf("%d\\n", {code})', "void")
            else:
                return (f'printf("%.6f\\n", (double)({code}))', "void")

        # Çoklu argüman: örn. print("w:", w.item()) veya print("Tensor A:", A)
        has_tensor = False
        arg_results: list[tuple[str, str]] = []
        for a in args:
            c_code, c_type = self.emit_expr(a)
            arg_results.append((c_code, c_type))
            if c_type == "syn_tensor_t*":
                has_tensor = True

        if not has_tensor:
            fmt_parts: list[str] = []
            param_parts: list[str] = []
            for c_code, c_type in arg_results:
                if c_type == "const char*":
                    fmt_parts.append("%s")
                    param_parts.append(c_code)
                elif c_type == "int":
                    fmt_parts.append("%d")
                    param_parts.append(c_code)
                else:
                    fmt_parts.append("%.6f")
                    param_parts.append(f"(double)({c_code})")

            fmt_string = " ".join(fmt_parts) + "\\n"
            params_call = ", ".join(param_parts)
            return (f'printf("{fmt_string}", {params_call})', "void")

        print_calls: list[str] = []
        for i, (code, arg_type) in enumerate(arg_results):
            sep = " " if i < len(arg_results) - 1 else "\\n"
            if arg_type == "syn_tensor_t*":
                print_calls.append(f'syn_tensor_print("", {code})')
            elif arg_type == "const char*":
                print_calls.append(f'printf("%s{sep}", {code})')
            elif arg_type == "int":
                print_calls.append(f'printf("%d{sep}", {code})')
            else:
                print_calls.append(f'printf("%.6f{sep}", (double)({code}))')

        combined = ",\n    ".join(print_calls)
        return (f"({combined})", "void")

    def emit_tensor_literal(self, t_expr: TensorLiteralExpr) -> tuple[str, str]:
        req_grad = 1 if (t_expr.kwargs.get("requires_grad") and getattr(t_expr.kwargs["requires_grad"], "value", False)) else 0
        return self.emit_tensor_constructor(t_expr.data, req_grad)

    def emit_tensor_constructor(self, data_expr: Expr, req_grad: int) -> tuple[str, str]:
        # Skaler sayı mı?
        if isinstance(data_expr, LiteralExpr) and isinstance(data_expr.value, (int, float)):
            if self.current_loop_arena:
                return (
                    f"syn_tensor_create_arena({self.current_loop_arena}, "
                    f"(const double[]){{{float(data_expr.value):.6f}}}, (const int[]){{1}}, 1, {req_grad})",
                    "syn_tensor_t*"
                )
            return (f"syn_tensor_scalar({float(data_expr.value)}, {req_grad})", "syn_tensor_t*")
        elif isinstance(data_expr, IdentifierExpr):
            # Değişken değeri üzerinden skaler tensör
            if self.current_loop_arena:
                return (
                    f"syn_tensor_create_arena({self.current_loop_arena}, "
                    f"(const double[]){{(double)({data_expr.name})}}, (const int[]){{1}}, 1, {req_grad})",
                    "syn_tensor_t*"
                )
            return (f"syn_tensor_scalar({data_expr.name}, {req_grad})", "syn_tensor_t*")

        # Liste literali mi? (1D veya 2D)
        flat_vals, shape = self._flatten_and_get_shape(data_expr)
        self.tensor_counter += 1
        var_id = self.tensor_counter

        data_str = ", ".join(f"{v:.6f}" for v in flat_vals)
        shape_str = ", ".join(str(s) for s in shape)

        if self.current_loop_arena:
            init_code = (
                f"syn_tensor_create_arena({self.current_loop_arena}, "
                f"(const double[]){{{data_str}}}, "
                f"(const int[]){{{shape_str}}}, "
                f"{len(shape)}, {req_grad})"
            )
        else:
            init_code = (
                f"syn_tensor_create("
                f"(const double[]){{{data_str}}}, "
                f"(const int[]){{{shape_str}}}, "
                f"{len(shape)}, {req_grad})"
            )
        return (init_code, "syn_tensor_t*")

    def _flatten_and_get_shape(self, expr: Expr) -> tuple[list[float], list[int]]:
        flat: list[float] = []
        if isinstance(expr, ListLiteralExpr):
            if expr.elements and isinstance(expr.elements[0], ListLiteralExpr):
                # 2D matris
                rows = len(expr.elements)
                cols = len(expr.elements[0].elements)
                for row_el in expr.elements:
                    if isinstance(row_el, ListLiteralExpr):
                        for col_el in row_el.elements:
                            val = col_el.value if isinstance(col_el, LiteralExpr) else 0.0
                            flat.append(float(val))
                return flat, [rows, cols]
            else:
                # 1D tensör
                for el in expr.elements:
                    val = el.value if isinstance(el, LiteralExpr) else 0.0
                    flat.append(float(val))
                return flat, [len(flat)]
        return [0.0], [1]
