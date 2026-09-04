from __future__ import annotations
import re
import sys
import json
import difflib
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Union
from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.vm.compiler import Compiler
from synapse.core.tensor import TensorShapeMismatchError
from synapse.parser.ast_nodes import (
    Program, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, FunctionDef,
    BinaryExpr, UnaryExpr, CallExpr, MemberExpr, IdentifierExpr,
    ListLiteralExpr, TensorLiteralExpr, LiteralExpr,
    TensorType, ShapeAnnotation, TypeAnnotation
)


class TypeContractViolationError(Exception):
    """Raised when a static or runtime tensor type contract is violated."""
    def __init__(self, message: str, variable: str = "", expected_shape: Any = None, actual_shape: Any = None, line: int = 0, column: int = 0):
        super().__init__(message)
        self.message = message
        self.variable = variable
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.line = line
        self.column = column


@dataclass
class DiagnosticReport:
    status: str = "ok"                          # "ok" | "error"
    error_type: Optional[str] = None           # "LexerError" | "ParseError" | "MissingDeclaration" | etc.
    message: str = ""
    line: int = 0
    column: int = 0
    source_line: str = ""
    pointer: str = ""
    suggested_fix: Optional[str] = None
    ai_prompt_hint: Optional[str] = None
    diff: Optional[str] = None
    auto_fixed_code: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def generate_unified_diff(original: str, modified: str, filename: str = "source.syn") -> str:
    """İki kaynak metin arasında standart unified diff üretir."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    )
    return "".join(diff)


def fix_ai_drift(source_code: str, filename: str = "source.syn") -> tuple[str, str, list[str]]:
    """
    LLM'lerin Python alışkanlıklarıyla yazdığı kodları (def, missing let, numpy/torch imports,
    capitalized booleans, inverted matrix transpose vb.) analiz edip standart Synapse sözdizimine dönüştürür.
    Dönüş: (fixed_code, unified_diff, list_of_changes)
    """
    lines = source_code.splitlines()
    fixed_lines = []
    changes = []

    keywords_or_blocks = {
        "fn", "let", "const", "return", "if", "elif", "else", "while",
        "for", "pass", "break", "continue", "import", "from", "prompt",
        "agent", "tool", "print"
    }
    defined_vars = set()

    for idx, line in enumerate(lines, start=1):
        original_line = line

        # 0. Tab karakterlerini 4 boşluğa çevir
        if "\t" in original_line:
            original_line = original_line.replace("\t", "    ")
            changes.append(f"Line {idx}: Replaced tabs with spaces for indentation.")

        stripped = original_line.strip()

        if not stripped or stripped.startswith("#"):
            fixed_lines.append(original_line)
            continue

        indent = original_line[:len(original_line) - len(original_line.lstrip())]
        content = stripped

        # 1. 'import numpy', 'import torch', 'import tensorflow' tespiti
        if re.match(r"^(import\s+(numpy|torch|tensorflow|scipy|sklearn)(\s+as\s+\w+)?|from\s+(numpy|torch|tensorflow|scipy|sklearn)\s+import\s+.*)", content):
            fixed_lines.append(f"{indent}# Synapse notice: Tensors and neural layers are native language primitives; external import removed.")
            changes.append(f"Line {idx}: Removed unnecessary external deep learning import.")
            continue

        # 2. 'def func_name(' -> 'fn func_name('
        if re.match(r"^def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", content):
            content = re.sub(r"^def\s+", "fn ", content)
            changes.append(f"Line {idx}: Converted 'def' to 'fn'.")

        # 3. 'np.array' / 'torch.tensor' -> 'tensor'
        if "np.array(" in content or "torch.tensor(" in content:
            content = content.replace("np.array(", "tensor(").replace("torch.tensor(", "tensor(")
            changes.append(f"Line {idx}: Replaced np.array/torch.tensor with native 'tensor()'.")

        # 4. 'np.zeros' / 'torch.zeros' / 'np.empty' -> 'zeros'
        if "np.zeros(" in content or "torch.zeros(" in content or "np.empty(" in content:
            content = content.replace("np.zeros(", "zeros(").replace("torch.zeros(", "zeros(").replace("np.empty(", "zeros(")
            changes.append(f"Line {idx}: Replaced zeros call with native 'zeros()'.")

        # 5. 'np.ones' / 'torch.ones' -> 'ones'
        if "np.ones(" in content or "torch.ones(" in content:
            content = content.replace("np.ones(", "ones(").replace("torch.ones(", "ones(")
            changes.append(f"Line {idx}: Replaced ones call with native 'ones()'.")

        # 6. 'np.random.randn' / 'torch.randn' -> 'randn'
        if "np.random.randn(" in content or "torch.randn(" in content:
            content = content.replace("np.random.randn(", "randn(").replace("torch.randn(", "randn(")
            changes.append(f"Line {idx}: Replaced randn call with native 'randn()'.")

        # 7. 'torch.matmul(a, b)', 'np.matmul(a, b)', 'np.dot(a, b)', 'a.dot(b)' -> 'a @ b'
        mm_match = re.search(r"(?:torch|np)\.(?:matmul|dot)\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*,\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)", content)
        if mm_match:
            left_op, right_op = mm_match.group(1), mm_match.group(2)
            content = content[:mm_match.start()] + f"{left_op} @ {right_op}" + content[mm_match.end():]
            changes.append(f"Line {idx}: Replaced functional matmul/dot with native matrix multiplication '@'.")

        dot_method = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\.dot\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)", content)
        if dot_method:
            left_op, right_op = dot_method.group(1), dot_method.group(2)
            content = content[:dot_method.start()] + f"{left_op} @ {right_op}" + content[dot_method.end():]
            changes.append(f"Line {idx}: Replaced .dot() call with native matrix multiplication '@'.")

        # 8. 'A.t()' veya 'np.transpose(A)' -> 'A.T'
        if ".t()" in content:
            content = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\.t\(\)", r"\1.T", content)
            changes.append(f"Line {idx}: Replaced PyTorch .t() with Synapse property '.T'.")
        if "np.transpose(" in content or "torch.transpose(" in content:
            content = re.sub(r"(?:np\.transpose|torch\.transpose)\(\s*([A-Za-z_][A-Za-z0-9_]*)(?:,\s*[^)]*)?\)", r"\1.T", content)
            changes.append(f"Line {idx}: Replaced transpose call with native '.T'.")

        # 9. Python boolean ve None değişimi
        if re.search(r"\bTrue\b", content):
            content = re.sub(r"\bTrue\b", "true", content)
            changes.append(f"Line {idx}: Normalized 'True' to 'true'.")
        if re.search(r"\bFalse\b", content):
            content = re.sub(r"\bFalse\b", "false", content)
            changes.append(f"Line {idx}: Normalized 'False' to 'false'.")
        if re.search(r"\bNone\b", content):
            content = re.sub(r"\bNone\b", "none", content)
            changes.append(f"Line {idx}: Normalized 'None' to 'none'.")

        # 10. let/const eksikliği tespiti (x = 10 -> let x = 10, A: Tensor[32, 64] = ... -> let A: Tensor[32, 64] = ...)
        assign_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*:\s*[^=]+)?)\s*(=|\+=|-=|\*=|/=|@=)\s*(.*)", content)
        if assign_match:
            target_decl = assign_match.group(1)
            var_name = target_decl.split(":")[0].strip()
            op = assign_match.group(2)
            val = assign_match.group(3)

            first_word = content.split()[0].split(":")[0].strip()
            if first_word not in keywords_or_blocks:
                if op == "=" and var_name not in defined_vars:
                    content = f"let {target_decl} = {val}"
                    defined_vars.add(var_name)
                    changes.append(f"Line {idx}: Added 'let' declaration for variable '{var_name}'.")
                else:
                    defined_vars.add(var_name)
            else:
                if first_word in ("let", "const"):
                    parts = content.split()
                    if len(parts) > 1:
                        decl_name = parts[1].split(":")[0].strip()
                        defined_vars.add(decl_name)

        # 11. Blok başlıklarında iki nokta unutulması
        block_match = re.match(r"^(fn|if|elif|else|while|for|prompt|agent|tool)\b", content)
        if block_match and not content.endswith(":"):
            content = content + ":"
            changes.append(f"Line {idx}: Appended missing ':' to block header.")

        fixed_lines.append(f"{indent}{content}")

    # 12. İkinci Aşama: Statik Tensör Şekil Analizi ile Matris Çarpımında Transpoz Onarımı
    stage1_code = "\n".join(fixed_lines)
    if source_code.endswith("\n") and not stage1_code.endswith("\n"):
        stage1_code += "\n"

    try:
        tokens = Lexer(stage1_code).tokenize()
        ast = Parser(tokens).parse()
        env_shapes: dict[str, tuple[Union[int, str], ...]] = {}

        def shape_scan(node: Any):
            if node is None:
                return
            if isinstance(node, Program):
                for stmt in node.statements:
                    shape_scan(stmt)
            elif isinstance(node, FunctionDef):
                fn_env = dict(env_shapes)
                for p in node.params:
                    p_tt = getattr(p, "tensor_type", None)
                    if p_tt:
                        fn_env[p.name] = p_tt.to_shape_tuple()
                for stmt in node.body:
                    shape_scan_stmt(stmt, fn_env)
            else:
                shape_scan_stmt(node, env_shapes)

        def shape_scan_stmt(stmt: Any, current_env: dict[str, tuple[Union[int, str], ...]]):
            if isinstance(stmt, VarDeclStmt):
                tt = getattr(stmt, "tensor_type", None)
                if tt:
                    current_env[stmt.name] = tt.to_shape_tuple()
                else:
                    s = _infer_expr_shape(stmt.value, current_env)
                    if s:
                        current_env[stmt.name] = s
                expr_scan(stmt.value, current_env)
            elif isinstance(stmt, AssignStmt):
                s = _infer_expr_shape(stmt.value, current_env)
                if s and isinstance(stmt.target, IdentifierExpr):
                    current_env[stmt.target.name] = s
                expr_scan(stmt.value, current_env)
            elif isinstance(stmt, ExprStmt):
                expr_scan(stmt.expr, current_env)
            elif isinstance(stmt, ReturnStmt) and stmt.value:
                expr_scan(stmt.value, current_env)

        def expr_scan(expr: Any, current_env: dict[str, tuple[Union[int, str], ...]]):
            if expr is None:
                return
            if isinstance(expr, BinaryExpr):
                if expr.op == "@":
                    s1 = _infer_expr_shape(expr.left, current_env)
                    s2 = _infer_expr_shape(expr.right, current_env)
                    if s1 and s2 and len(s1) >= 2 and len(s2) >= 2 and s1[-1] != s2[-2]:
                        line_idx = getattr(expr, "line", 0)
                        if 1 <= line_idx <= len(fixed_lines):
                            line_text = fixed_lines[line_idx - 1]
                            if s1[-1] == s2[-1] and isinstance(expr.right, IdentifierExpr):
                                r_name = expr.right.name
                                pattern = rf"(@\s*{re.escape(r_name)})\b(?!\.T)"
                                if re.search(pattern, line_text):
                                    fixed_lines[line_idx - 1] = re.sub(pattern, r"\1.T", line_text)
                                    changes.append(
                                        f"Line {line_idx}: Detected inverted matrix dimension ({s1} @ {s2}); "
                                        f"transposed right operand '{r_name}' to '{r_name}.T' to align inner dimensions."
                                    )
                            elif s1[-2] == s2[-2] and isinstance(expr.left, IdentifierExpr):
                                l_name = expr.left.name
                                pattern = rf"\b({re.escape(l_name)})(\s*@)(?!\.T)"
                                if re.search(pattern, line_text):
                                    fixed_lines[line_idx - 1] = re.sub(pattern, r"\1.T\2", line_text)
                                    changes.append(
                                        f"Line {line_idx}: Detected inverted matrix dimension ({s1} @ {s2}); "
                                        f"transposed left operand '{l_name}' to '{l_name}.T' to align inner dimensions."
                                    )
                expr_scan(expr.left, current_env)
                expr_scan(expr.right, current_env)
            elif isinstance(expr, MemberExpr):
                expr_scan(expr.target, current_env)
            elif isinstance(expr, CallExpr):
                expr_scan(expr.callee, current_env)
                for a in expr.args:
                    expr_scan(a, current_env)
                for v in expr.kwargs.values():
                    expr_scan(v, current_env)

        shape_scan(ast)
    except Exception:
        pass

    fixed_code = "\n".join(fixed_lines)
    if source_code.endswith("\n") and not fixed_code.endswith("\n"):
        fixed_code += "\n"

    diff = generate_unified_diff(source_code, fixed_code, filename=filename)
    return fixed_code, diff, changes


def _extract_source_line(source: str, line_no: int) -> str:
    lines = source.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


def _generate_pointer(col_no: int) -> str:
    idx = max(0, col_no - 1)
    return (" " * idx) + "^"


def _suggest_fix_for_parse_error(message: str, token_val: Any, source_line: str) -> tuple[Optional[str], Optional[str]]:
    """Yapay zekanın tek hamlede düzeltebilmesi için akıllı öneri ve prompt ipucu üretir."""
    # 1. 'def' yerine 'fn' kullanımı
    if "def " in source_line or token_val == "def":
        suggested = re.sub(r"\bdef\b", "fn", source_line)
        hint = "In Synapse, functions are declared using 'fn', never 'def'."
        return suggested.strip(), hint

    # 2. Değişken tanımında let/const unutulması: örn: x = 10
    assign_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=|\+=|-=|\*=|/=|@=)", source_line)
    if assign_match and not any(kw in source_line for kw in ("let ", "const ", "if ", "while ", "for ")):
        var_name = assign_match.group(1)
        suggested = "let " + source_line.strip()
        hint = f"Variable '{var_name}' must be initialized with 'let' or 'const'."
        return suggested, hint

    # 3. İki nokta unutulması (if, elif, else, fn, while, for, prompt, agent)
    block_match = re.match(r"^\s*(fn|if|elif|else|while|for|prompt|agent)\b", source_line)
    if block_match and not source_line.rstrip().endswith(":"):
        suggested = source_line.rstrip() + ":"
        hint = "Block header must end with ':'."
        return suggested.strip(), hint

    # 4. Parantez veya köşeli ayraç kapanışı
    if "Expected ')'" in message or "Expected ']'" in message:
        return None, "Check for unclosed parentheses or brackets in expression."

    return None, f"Syntax mismatch near '{token_val}': {message}"


def _is_tensor_shape_mismatch(error: Optional[Union[Exception, str]], source: str = "") -> bool:
    """Hatanın veya kodun bir tensör matris çarpım boyut uyuşmazlığı olup olmadığını belirler."""
    if error is not None:
        err_str = str(error)
        err_cls = type(error).__name__ if isinstance(error, Exception) else str(error)
        if "TensorShapeMismatchError" in err_cls or "TensorShapeMismatchError" in err_str:
            return True
        if "Cannot multiply tensor of shape" in err_str:
            return True
        if "Inner dimensions must match" in err_str:
            return True
        if "shapes" in err_str and "not aligned" in err_str:
            return True
        if "mismatch in its core dimension" in err_str:
            return True
        if "matrix multiplication" in err_str.lower() and (
            "dimension" in err_str.lower()
            or "shape" in err_str.lower()
            or "mismatch" in err_str.lower()
            or "incompatible" in err_str.lower()
        ):
            return True

    if "TensorShapeMismatchError" in source:
        return True

    return False


def _resolve_error_location(
    error: Union[Exception, str],
    source: str,
    line: Optional[int] = None,
    column: Optional[int] = None,
    filepath: str = "<source>",
) -> tuple[int, int]:
    """
    VMRuntimeError veya çalışma zamanı istisnalarından gelen satır ve sütun bilgisini otomatik çözer.
    """
    line_no = line
    if line_no is None and isinstance(error, Exception):
        line_no = getattr(error, "line", None) or getattr(error, "lineno", None) or getattr(error, "line_no", None)

    if line_no is None:
        err_msg = str(error)
        m = re.search(r"(?:at\s+line|line\s*[:#]?|\[line\s*)\s*(\d+)", err_msg, re.IGNORECASE)
        if m:
            try:
                line_no = int(m.group(1))
            except ValueError:
                pass

    if line_no is None and isinstance(error, Exception) and getattr(error, "__traceback__", None) is not None:
        tb = error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename == filepath:
                line_no = tb.tb_lineno
            tb = tb.tb_next

    # Eğer tensör matris çarpım hatasıysa ve satır bulunamadıysa, kaynak koddaki '@' satırını bul
    if line_no is None and _is_tensor_shape_mismatch(error, source):
        lines = source.splitlines()
        for idx, l in enumerate(lines, start=1):
            if "@" in l:
                line_no = idx
                break

    if line_no is None or line_no <= 0:
        line_no = 1

    source_line = _extract_source_line(source, line_no)

    col_no = column
    if col_no is None and isinstance(error, Exception):
        col_no = getattr(error, "column", None) or getattr(error, "col", None) or getattr(error, "column_no", None)

    if col_no is None:
        err_msg = str(error)
        m_col = re.search(r"(?:col|column)\s*[:#]?\s*(\d+)", err_msg, re.IGNORECASE)
        if m_col:
            try:
                col_no = int(m_col.group(1))
            except ValueError:
                pass

    if col_no is None or col_no <= 0:
        if "@" in source_line:
            col_no = source_line.find("@") + 1
        else:
            stripped = source_line.lstrip()
            if stripped:
                col_no = (len(source_line) - len(stripped)) + 1
            else:
                col_no = 1

    return line_no, col_no


def diagnose_runtime_error(
    source: str,
    error: Union[Exception, str],
    filepath: str = "<source>",
    line: Optional[int] = None,
    column: Optional[int] = None,
) -> DiagnosticReport:
    """
    VMRuntimeError, TensorShapeMismatchError veya diğer çalışma zamanı istisnalarından
    satır bazlı ve LLM self-healing odaklı DiagnosticReport oluşturur.
    """
    line_no, col_no = _resolve_error_location(error, source, line=line, column=column, filepath=filepath)
    source_line = _extract_source_line(source, line_no)
    pointer = _generate_pointer(col_no)

    err_type = type(error).__name__ if isinstance(error, Exception) else "RuntimeError"
    message = str(error)
    details: dict[str, Any] = {"filepath": filepath}
    if isinstance(error, Exception):
        details["exception_class"] = type(error).__name__

    suggested_fix = None
    if _is_tensor_shape_mismatch(error, source):
        ai_prompt_hint = "Tensor shape mismatch in matrix multiplication (@). Verify tensor dimensions and use transpose (.T) or reshape so inner dimensions align."
        if "@" in source_line:
            suggested_fix = re.sub(r"(@\s*[A-Za-z_][A-Za-z0-9_]*)", r"\1.T", source_line).strip()
    else:
        ai_prompt_hint = f"Runtime execution failed: {message}"

    return DiagnosticReport(
        status="error",
        error_type=err_type,
        message=message,
        line=line_no,
        column=col_no,
        source_line=source_line,
        pointer=pointer,
        suggested_fix=suggested_fix,
        ai_prompt_hint=ai_prompt_hint,
        details=details,
    )


def _are_broadcast_compatible(s1: tuple[Union[int, str], ...], s2: tuple[Union[int, str], ...]) -> bool:
    """İki tensör şeklinin yayınlama (broadcasting) kurallarına göre uyumlu olup olmadığını kontrol eder."""
    for d1, d2 in zip(reversed(s1), reversed(s2)):
        if isinstance(d1, int) and isinstance(d2, int):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
    return True


def _broadcast_shapes(
    s1: tuple[Union[int, str], ...],
    s2: tuple[Union[int, str], ...]
) -> Optional[tuple[Union[int, str], ...]]:
    """İki şekli numpy yayınlama kurallarına göre birleştirir; uyumsuzsa None döner."""
    res = []
    for d1, d2 in zip(reversed(s1), reversed(s2)):
        if d1 == d2:
            res.append(d1)
        elif d1 == 1:
            res.append(d2)
        elif d2 == 1:
            res.append(d1)
        elif isinstance(d1, str):
            res.append(d2)
        elif isinstance(d2, str):
            res.append(d1)
        else:
            return None
    longer = s1 if len(s1) > len(s2) else s2
    prefix = longer[:len(longer) - min(len(s1), len(s2))]
    return tuple(prefix) + tuple(reversed(res))


def _infer_list_shape(elem: Any) -> Optional[tuple[int, ...]]:
    from synapse.parser.ast_nodes import ListLiteralExpr, LiteralExpr
    if isinstance(elem, ListLiteralExpr):
        if not elem.elements:
            return (0,)
        sub_shapes = [_infer_list_shape(sub) for sub in elem.elements]
        if any(s is None for s in sub_shapes):
            return None
        first = sub_shapes[0]
        for s in sub_shapes[1:]:
            if s != first:
                return None
        return (len(elem.elements),) + first
    elif isinstance(elem, LiteralExpr) and isinstance(elem.value, (int, float)):
        return ()
    return None


def _infer_expr_shape(
    expr: Any,
    env: dict[str, tuple[Union[int, str], ...]],
    fn_returns: Optional[dict[str, tuple[Union[int, str], ...]]] = None,
) -> Optional[tuple[Union[int, str], ...]]:
    from synapse.parser.ast_nodes import (
        LiteralExpr, IdentifierExpr, BinaryExpr, UnaryExpr, CallExpr,
        MemberExpr, ListLiteralExpr, TensorLiteralExpr
    )

    if expr is None:
        return None

    if fn_returns is None:
        fn_returns = {}

    if isinstance(expr, IdentifierExpr):
        return env.get(expr.name)

    if isinstance(expr, MemberExpr):
        if expr.member == "T":
            target_shape = _infer_expr_shape(expr.target, env, fn_returns)
            if target_shape is not None and len(target_shape) >= 2:
                return target_shape[:-2] + (target_shape[-1], target_shape[-2])
        return None

    if isinstance(expr, TensorLiteralExpr):
        return _infer_expr_shape(expr.data, env, fn_returns)

    if isinstance(expr, ListLiteralExpr):
        return _infer_list_shape(expr)

    if isinstance(expr, CallExpr):
        callee_name = getattr(expr.callee, "name", "")
        if callee_name == "tensor" and expr.args:
            return _infer_expr_shape(expr.args[0], env, fn_returns)
        elif callee_name in ("zeros", "ones", "randn", "empty") and expr.args:
            first_arg = expr.args[0]
            if isinstance(first_arg, ListLiteralExpr):
                dims: list[Union[int, str]] = []
                for el in first_arg.elements:
                    if isinstance(el, LiteralExpr) and isinstance(el.value, int):
                        dims.append(el.value)
                    elif isinstance(el, LiteralExpr) and isinstance(el.value, str):
                        dims.append(el.value)
                    elif isinstance(el, IdentifierExpr) and el.name in env:
                        val = env[el.name]
                        if isinstance(val, int):
                            dims.append(val)
                        else:
                            dims.append(el.name)
                    else:
                        return None
                return tuple(dims)
            elif all(isinstance(a, LiteralExpr) and isinstance(a.value, (int, str)) for a in expr.args):
                return tuple(a.value for a in expr.args)
        elif callee_name in fn_returns:
            return fn_returns[callee_name]
        elif isinstance(expr.callee, MemberExpr):
            member = expr.callee.member
            target = expr.callee.target
            if member == "reshape":
                if expr.args:
                    first = expr.args[0]
                    if isinstance(first, ListLiteralExpr):
                        return tuple(e.value for e in first.elements if isinstance(e, LiteralExpr) and isinstance(e.value, int))
                    elif all(isinstance(a, LiteralExpr) and isinstance(a.value, int) for a in expr.args):
                        return tuple(a.value for a in expr.args)
            elif member in ("transpose", "t"):
                t_shape = _infer_expr_shape(target, env, fn_returns)
                if t_shape is not None and len(t_shape) >= 2:
                    return t_shape[:-2] + (t_shape[-1], t_shape[-2])
        return None

    if isinstance(expr, BinaryExpr):
        if expr.op == "@":
            s1 = _infer_expr_shape(expr.left, env, fn_returns)
            s2 = _infer_expr_shape(expr.right, env, fn_returns)
            if s1 and s2:
                if len(s1) >= 2 and len(s2) >= 2:
                    if s1[-1] == s2[-2] or isinstance(s1[-1], str) or isinstance(s2[-2], str):
                        return s1[:-1] + (s2[-1],)
                elif len(s1) == 2 and len(s2) == 1:
                    if s1[-1] == s2[0] or isinstance(s1[-1], str) or isinstance(s2[0], str):
                        return (s1[0],)
                elif len(s1) == 1 and len(s2) == 2:
                    if s1[0] == s2[-2] or isinstance(s1[0], str) or isinstance(s2[-2], str):
                        return (s2[-1],)
        elif expr.op in ("+", "-", "*", "/", "**", "%"):
            s1 = _infer_expr_shape(expr.left, env, fn_returns)
            s2 = _infer_expr_shape(expr.right, env, fn_returns)
            if s1 == ():
                return s2
            if s2 == ():
                return s1
            if s1 and s2:
                b_shape = _broadcast_shapes(s1, s2)
                if b_shape is not None:
                    return b_shape
                return s1 if len(s1) >= len(s2) else s2
            return s1 or s2

    if isinstance(expr, UnaryExpr):
        if expr.op in ("-", "+"):
            return _infer_expr_shape(expr.operand, env, fn_returns)

    if isinstance(expr, LiteralExpr):
        if isinstance(expr.value, (int, float, bool)):
            return ()

    return None


def _check_static_tensor_shapes(ast_root: Any, source: str, filepath: str) -> Optional[DiagnosticReport]:
    """AST üzerinde tensör tipleri, şekil sözleşmeleri ve işlem uyumluluklarını statik olarak inceler."""
    from synapse.parser.ast_nodes import (
        Program, VarDeclStmt, AssignStmt, ExprStmt, ReturnStmt, FunctionDef,
        BinaryExpr, MemberExpr, IdentifierExpr, CallExpr, TensorType
    )

    env: dict[str, tuple[Union[int, str], ...]] = {}
    contracts: dict[str, TensorType] = {}
    fn_returns: dict[str, tuple[Union[int, str], ...]] = {}

    mismatch_node = None
    mismatch_info = None

    def visit_expr(expr: Any, current_env: dict[str, tuple[Union[int, str], ...]]):
        nonlocal mismatch_node, mismatch_info
        if expr is None or mismatch_node is not None:
            return

        if isinstance(expr, BinaryExpr):
            if expr.op == "@":
                s1 = _infer_expr_shape(expr.left, current_env, fn_returns)
                s2 = _infer_expr_shape(expr.right, current_env, fn_returns)
                if s1 is not None and s2 is not None:
                    if len(s1) >= 2 and len(s2) >= 2:
                        d1 = s1[-1]
                        d2 = s2[-2]
                        if isinstance(d1, int) and isinstance(d2, int) and d1 != d2:
                            mismatch_node = expr
                            mismatch_info = ("matmul", s1, s2, expr.left, expr.right)
                            return
                    elif len(s1) == 1 and len(s2) == 2:
                        if isinstance(s1[0], int) and isinstance(s2[-2], int) and s1[0] != s2[-2]:
                            mismatch_node = expr
                            mismatch_info = ("matmul", s1, s2, expr.left, expr.right)
                            return
                    elif len(s1) == 2 and len(s2) == 1:
                        if isinstance(s1[-1], int) and isinstance(s2[0], int) and s1[-1] != s2[0]:
                            mismatch_node = expr
                            mismatch_info = ("matmul", s1, s2, expr.left, expr.right)
                            return
            elif expr.op in ("+", "-"):
                s1 = _infer_expr_shape(expr.left, current_env, fn_returns)
                s2 = _infer_expr_shape(expr.right, current_env, fn_returns)
                if s1 is not None and s2 is not None and s1 != () and s2 != ():
                    if not _are_broadcast_compatible(s1, s2):
                        mismatch_node = expr
                        mismatch_info = ("elementwise", expr.op, s1, s2, expr.left, expr.right)
                        return

            visit_expr(expr.left, current_env)
            visit_expr(expr.right, current_env)

        elif isinstance(expr, MemberExpr):
            if expr.member == "T":
                s = _infer_expr_shape(expr.target, current_env, fn_returns)
                if s is not None and len(s) < 2:
                    mismatch_node = expr
                    mismatch_info = ("transpose", s)
                    return
            visit_expr(expr.target, current_env)

        elif isinstance(expr, CallExpr):
            visit_expr(expr.callee, current_env)
            for a in expr.args:
                visit_expr(a, current_env)
            for v in expr.kwargs.values():
                visit_expr(v, current_env)

    def visit(node: Any, current_env: dict[str, tuple[Union[int, str], ...]], current_contracts: dict[str, TensorType]):
        nonlocal mismatch_node, mismatch_info
        if node is None or mismatch_node is not None:
            return

        if isinstance(node, Program):
            for stmt in node.statements:
                visit(stmt, current_env, current_contracts)
                if mismatch_node is not None:
                    return

        elif isinstance(node, FunctionDef):
            if getattr(node, "return_tensor_type", None):
                fn_returns[node.name] = node.return_tensor_type.to_shape_tuple()
            elif node.return_type:
                r_tt = TensorType.from_string(node.return_type if isinstance(node.return_type, str) else str(node.return_type))
                if r_tt:
                    fn_returns[node.name] = r_tt.to_shape_tuple()

            fn_scope_env = dict(current_env)
            fn_scope_contracts = dict(current_contracts)

            for param in node.params:
                p_tt = getattr(param, "tensor_type", None)
                if p_tt is None and param.type_annot:
                    p_tt = TensorType.from_string(param.type_annot if isinstance(param.type_annot, str) else str(param.type_annot))
                if p_tt:
                    fn_scope_env[param.name] = p_tt.to_shape_tuple()
                    fn_scope_contracts[param.name] = p_tt

            for stmt in node.body:
                visit(stmt, fn_scope_env, fn_scope_contracts)
                if mismatch_node is not None:
                    return

        elif isinstance(node, VarDeclStmt):
            declared_tt = getattr(node, "tensor_type", None)
            if declared_tt is None and node.type_annot:
                declared_tt = TensorType.from_string(node.type_annot if isinstance(node.type_annot, str) else str(node.type_annot))

            if declared_tt is not None:
                current_contracts[node.name] = declared_tt
                declared_shape = declared_tt.to_shape_tuple()
                current_env[node.name] = declared_shape

                # Check subexpressions on RHS first
                visit_expr(node.value, current_env)
                if mismatch_node is not None:
                    return

                # Check if assigned value shape conforms to declared contract
                val_shape = _infer_expr_shape(node.value, current_env, fn_returns)
                if val_shape is not None and not declared_tt.matches_shape(val_shape):
                    mismatch_node = node
                    mismatch_info = ("contract", node.name, declared_shape, val_shape, declared_tt)
                    return
            else:
                visit_expr(node.value, current_env)
                if mismatch_node is not None:
                    return
                val_shape = _infer_expr_shape(node.value, current_env, fn_returns)
                if val_shape is not None:
                    current_env[node.name] = val_shape

        elif isinstance(node, AssignStmt):
            visit_expr(node.value, current_env)
            if mismatch_node is not None:
                return

            val_shape = _infer_expr_shape(node.value, current_env, fn_returns)
            if isinstance(node.target, IdentifierExpr):
                target_name = node.target.name
                if target_name in current_contracts:
                    declared_tt = current_contracts[target_name]
                    if val_shape is not None and not declared_tt.matches_shape(val_shape):
                        mismatch_node = node
                        mismatch_info = ("contract_reassign", target_name, declared_tt.to_shape_tuple(), val_shape, declared_tt)
                        return
                if val_shape is not None:
                    current_env[target_name] = val_shape

        elif isinstance(node, ReturnStmt):
            if node.value:
                visit_expr(node.value, current_env)

        elif isinstance(node, ExprStmt):
            visit_expr(node.expr, current_env)

        for attr in ("then_branch", "else_branch", "body"):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                for c in child:
                    visit(c, current_env, current_contracts)
                    if mismatch_node is not None:
                        return
            elif child is not None and isinstance(child, list):
                for c in child:
                    visit(c, current_env, current_contracts)
            elif child is not None and hasattr(child, "line"):
                visit(child, current_env, current_contracts)

    try:
        visit(ast_root, env, contracts)
    except Exception:
        return None

    if mismatch_node is not None and mismatch_info is not None:
        tag = mismatch_info[0]
        line_no = getattr(mismatch_node, "line", 1) or 1
        source_line = _extract_source_line(source, line_no)
        col_no = getattr(mismatch_node, "column", 1) or 1

        if tag == "contract":
            _, var_name, decl_shape, val_shape, tt = mismatch_info
            pointer = _generate_pointer(col_no)
            msg = (
                f"Type contract violation: Variable '{var_name}' declared with contract {tt} "
                f"(shape {decl_shape}) but assigned expression of shape {val_shape}."
            )
            return DiagnosticReport(
                status="error",
                error_type="TypeContractViolationError",
                message=msg,
                line=line_no,
                column=col_no,
                source_line=source_line,
                pointer=pointer,
                ai_prompt_hint=f"Adjust assigned expression dimensions or update '{var_name}' type annotation to match {val_shape}.",
                details={
                    "filepath": filepath,
                    "variable": var_name,
                    "contract": str(tt),
                    "expected_shape": list(decl_shape),
                    "actual_shape": list(val_shape),
                }
            )

        elif tag == "contract_reassign":
            _, var_name, decl_shape, val_shape, tt = mismatch_info
            pointer = _generate_pointer(col_no)
            msg = (
                f"Type contract violation: Reassignment to variable '{var_name}' with declared contract {tt} "
                f"(shape {decl_shape}) but assigned expression of shape {val_shape}."
            )
            return DiagnosticReport(
                status="error",
                error_type="TypeContractViolationError",
                message=msg,
                line=line_no,
                column=col_no,
                source_line=source_line,
                pointer=pointer,
                ai_prompt_hint=f"Ensure value assigned to '{var_name}' conforms to declared shape {decl_shape}.",
                details={
                    "filepath": filepath,
                    "variable": var_name,
                    "contract": str(tt),
                    "expected_shape": list(decl_shape),
                    "actual_shape": list(val_shape),
                }
            )

        elif tag == "matmul":
            _, s1, s2, left_expr, right_expr = mismatch_info
            if col_no <= 1 and "@" in source_line:
                col_no = source_line.find("@") + 1
            pointer = _generate_pointer(col_no)

            msg = f"Cannot multiply tensor of shape {s1} with tensor of shape {s2}. Inner dimensions must match: {s1[-1]} != {s2[-2]}."
            suggested_fix = None
            if len(s2) >= 2 and s1[-1] == s2[-1]:
                if "@" in source_line:
                    suggested_fix = re.sub(r"(@\s*[A-Za-z_][A-Za-z0-9_]*)", r"\1.T", source_line).strip()
            elif len(s1) >= 2 and s1[-2] == s2[-2]:
                if "@" in source_line:
                    suggested_fix = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)(\s*@)", r"\1.T\2", source_line).strip()
            elif "@" in source_line:
                suggested_fix = re.sub(r"(@\s*[A-Za-z_][A-Za-z0-9_]*)", r"\1.T", source_line).strip()

            return DiagnosticReport(
                status="error",
                error_type="TensorShapeMismatchError",
                message=msg,
                line=line_no,
                column=col_no,
                source_line=source_line,
                pointer=pointer,
                suggested_fix=suggested_fix,
                ai_prompt_hint="Tensor shape mismatch in matrix multiplication (@). Verify tensor dimensions and use transpose (.T) or reshape so inner dimensions align.",
                details={"filepath": filepath, "left_shape": list(s1), "right_shape": list(s2)}
            )

        elif tag == "elementwise":
            _, op, s1, s2, left_expr, right_expr = mismatch_info
            pointer = _generate_pointer(col_no)
            msg = f"Cannot perform elementwise '{op}' on tensor of shape {s1} and tensor of shape {s2}. Shapes are not broadcast-compatible."
            suggested_fix = None
            if len(s1) == 2 and len(s2) == 2 and s1 == s2[::-1]:
                suggested_fix = re.sub(rf"(\+\s*[A-Za-z_][A-Za-z0-9_]*)", r"\1.T", source_line).strip()
            return DiagnosticReport(
                status="error",
                error_type="TensorShapeMismatchError",
                message=msg,
                line=line_no,
                column=col_no,
                source_line=source_line,
                pointer=pointer,
                suggested_fix=suggested_fix,
                ai_prompt_hint=f"Elementwise '{op}' requires tensors to be broadcast-compatible.",
                details={"filepath": filepath, "left_shape": list(s1), "right_shape": list(s2), "op": op}
            )

        elif tag == "transpose":
            _, s = mismatch_info
            pointer = _generate_pointer(col_no)
            msg = f"Cannot transpose tensor of shape {s}. Transposition requires at least 2 dimensions."
            return DiagnosticReport(
                status="error",
                error_type="TensorShapeMismatchError",
                message=msg,
                line=line_no,
                column=col_no,
                source_line=source_line,
                pointer=pointer,
                ai_prompt_hint="Transpose (.T) is only valid on tensors with at least 2 dimensions.",
                details={"filepath": filepath, "shape": list(s)}
            )

    return None


def diagnose_code(
    source: str,
    filepath: str = "<source>",
    error: Optional[Union[Exception, str]] = None,
    line: Optional[int] = None,
    column: Optional[int] = None,
    include_diff: bool = True,
) -> DiagnosticReport:
    """
    Synapse kaynak kodunu veya çalışma zamanı hatasını analiz eder ve yapay zekanın (LLM)
    anında kendi kendini onarmasını sağlayacak yapılandırılmış bir DiagnosticReport üretir.
    """
    report = None

    # 0. Çalışma zamanı hatası doğrudan geçilmişse
    if error is not None:
        report = diagnose_runtime_error(source, error, filepath=filepath, line=line, column=column)

    # 1. Lexer Analizi
    if report is None:
        try:
            tokens = Lexer(source).tokenize()
        except LexerError as e:
            line_str = _extract_source_line(source, e.line)
            pointer = _generate_pointer(e.column)
            fix, hint = None, str(e)
            if "did you mean '!='" in str(e):
                fix = line_str.replace("!", "!=")
                hint = "Replace '!' with '!=' for inequality check."
            elif "Unterminated string" in str(e):
                hint = "Close open quotation marks before newline or end of file."

            report = DiagnosticReport(
                status="error",
                error_type="LexerError",
                message=str(e),
                line=e.line,
                column=e.column,
                source_line=line_str,
                pointer=pointer,
                suggested_fix=fix,
                ai_prompt_hint=hint,
                details={"filepath": filepath}
            )

    # 2. Parser Analizi
    if report is None:
        try:
            ast = Parser(tokens).parse()
        except ParseError as e:
            line_str = _extract_source_line(source, e.token.line)
            pointer = _generate_pointer(e.token.column)
            fix, hint = _suggest_fix_for_parse_error(str(e), e.token.value, line_str)

            report = DiagnosticReport(
                status="error",
                error_type="ParseError",
                message=str(e),
                line=e.token.line,
                column=e.token.column,
                source_line=line_str,
                pointer=pointer,
                suggested_fix=fix,
                ai_prompt_hint=hint,
                details={"filepath": filepath, "token_type": e.token.type.name, "token_val": str(e.token.value)}
            )

    # 3. Statik Tensör Şekil Kontrolü
    if report is None:
        static_report = _check_static_tensor_shapes(ast, source, filepath)
        if static_report is not None:
            report = static_report

    # 4. Derleyici Doğrulaması
    if report is None:
        try:
            Compiler(name=filepath).compile(ast)
        except Exception as e:
            err_msg = str(e)
            is_shape_mismatch = _is_tensor_shape_mismatch(e, source)
            ai_hint = (
                "Tensor shape mismatch in matrix multiplication (@). Verify tensor dimensions and use transpose (.T) or reshape so inner dimensions align."
                if is_shape_mismatch
                else f"Bytecode compilation failed: {e}"
            )
            line_no, col_no = _resolve_error_location(e, source, line=line, column=column, filepath=filepath)
            source_line = _extract_source_line(source, line_no)
            pointer = _generate_pointer(col_no)
            report = DiagnosticReport(
                status="error",
                error_type=type(e).__name__,
                message=err_msg,
                line=line_no,
                column=col_no,
                source_line=source_line,
                pointer=pointer,
                suggested_fix=None,
                ai_prompt_hint=ai_hint,
                details={"filepath": filepath}
            )

    # 5. Eğer bir exception bloğu içindeyken çağrıldıysa ve sys.exc_info() aktifse
    if report is None:
        exc_type, exc_val, _ = sys.exc_info()
        if exc_val is not None and isinstance(exc_val, Exception):
            if not isinstance(exc_val, (LexerError, ParseError)):
                report = diagnose_runtime_error(source, exc_val, filepath=filepath, line=line, column=column)

    # 6. Başarılı ise
    if report is None:
        report = DiagnosticReport(
            status="ok",
            message="Code syntax and AST verified successfully. Ready for execution.",
            auto_fixed_code=source,
            details={"filepath": filepath}
        )

    # Hata durumunda self-healing diff & auto_fixed_code ekle
    if include_diff and report.status == "error":
        try:
            fixed_code, diff, _ = fix_ai_drift(source, filename=filepath)
            if diff:
                report.diff = diff
                report.auto_fixed_code = fixed_code
            elif not report.auto_fixed_code:
                report.diff = diff
                report.auto_fixed_code = fixed_code
        except Exception:
            pass

    return report


def heal_and_execute(
    source_code: str,
    vm: Optional[Any] = None,
    filename: str = "source.syn",
) -> tuple[Any, DiagnosticReport]:
    """
    Synapse self-healing derleyici ve çalışma zamanı orkestratörü:
    1. Kaynak koddaki AI ve sözdizimi sapmalarını otomatik tespit eder ve onarır (fix_ai_drift):
       - def -> fn
       - Eksik let / const bildirimleri (tipli ve tipsiz)
       - Blok başlıklarındaki eksik iki nokta (':')
       - Gereksiz harici kütüphane importları (numpy, torch, tensorflow)
       - Statik şekil analizi ile tespit edilen ters çevrilmiş matris çarpımlarında transpoz (.T)
       - Bozuk girintileme ve Python anahtar kelimeleri
    2. Statik tensör şekil sözleşmelerini ve işlem uyumluluklarını doğrular.
    3. Kodda çözülemez bir hata varsa diff ve öneri içeren DiagnosticReport döner.
    4. Kod geçerliyse derler, verilen veya oluşturulan VM üzerinde çalıştırır ve
       (çalıştırma_sonucu, DiagnosticReport) çiftini döner.
    """
    if vm is None:
        from synapse.vm.virtual_machine import VirtualMachine
        vm = VirtualMachine()

    # Adım 1: AI drift analizi ve onarımı
    fixed_code, diff, changes = fix_ai_drift(source_code, filename=filename)

    # Adım 2: Kod teşhisi
    report = diagnose_code(fixed_code, filepath=filename, include_diff=False)

    if report.status == "error":
        report.diff = diff if diff else report.diff
        report.auto_fixed_code = fixed_code
        return None, report

    # Adım 3: Derleme ve Çalıştırma
    try:
        tokens = Lexer(fixed_code).tokenize()
        ast = Parser(tokens).parse()

        # Ek statik şekil kontrolü
        static_err = _check_static_tensor_shapes(ast, fixed_code, filename)
        if static_err is not None:
            static_err.diff = diff if diff else None
            static_err.auto_fixed_code = fixed_code
            return None, static_err

        code_obj = Compiler(name=filename).compile(ast)
        result = vm.execute(code_obj)

        success_report = DiagnosticReport(
            status="ok",
            message="Code healed and executed successfully." if (fixed_code != source_code) else "Code executed successfully.",
            diff=diff if (fixed_code != source_code) else None,
            auto_fixed_code=fixed_code,
            details={"changes": changes, "filepath": filename}
        )
        return result, success_report

    except Exception as e:
        runtime_report = diagnose_runtime_error(fixed_code, e, filepath=filename)
        runtime_report.diff = diff if diff else None
        runtime_report.auto_fixed_code = fixed_code
        return None, runtime_report
