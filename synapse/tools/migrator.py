"""
Synapse Python-to-Synapse Intelligent Code Migrator (Faz 4).

Transpiles standard Python source code into idiomatic Synapse syntax:
- Functions: `def foo(...) -> T:` -> `fn foo(...) -> T:`
- Variables:
    - Typed: `x: int = 10` -> `let x: int = 10`
    - Untyped: `x = 10` -> `let x = 10`
    - Uppercase constants: `MAX_VAL = 100` -> `const MAX_VAL = 100`
- Tensor operations:
    - `np.array(...)` / `torch.tensor(...)` -> `tensor(...)`
    - `torch.zeros(...)` / `np.zeros(...)` -> `zeros(...)`
    - `torch.ones(...)` / `np.ones(...)` -> `ones(...)`
    - `np.dot(a, b)` / `torch.matmul(a, b)` / `a.dot(b)` / `x.matmul(w)` -> `(a @ b)`
- Imports:
    - Strips redundant framework imports (`numpy`, `torch`, `typing`, `dataclasses`).
    - Maps Python standard libraries to Synapse interoperability: `import py.<lib> as <alias>`.
- Control flow:
    - `for`, `while`, `if`/`elif`/`else`, `return`, `pass`, `break`, `continue`.
    - Unpacks tuples in for loops cleanly: `for x, y in pairs:` -> `for _item in pairs: let x = _item[0] ...`
- Expressions & Desugaring:
    - Desugars list & dict comprehensions into imperative Synapse loops.
    - Desugars ternary if-expressions (`a if cond else b`) into conditional branches.
    - Desugars lambdas into functions.
    - Translates membership tests (`x in items`) into Synapse builtin `contains(items, x)`.
    - Preserves context managers (`with`) and classes as idiomatic Synapse structures.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Optional, Union


class PythonToSynapseMigrator:
    """
    Translates Python AST structures into idiomatic Synapse source code.
    """

    # Framework modules to strip completely because Synapse provides them natively.
    STRIP_MODULES = {
        "numpy",
        "torch",
        "typing",
        "dataclasses",
        "__future__",
        "tensorflow",
        "scipy",
        "sklearn",
    }

    # Module prefixes to strip
    STRIP_PREFIXES = (
        "numpy.",
        "torch.",
        "typing.",
        "torchvision.",
        "tensorflow.",
        "scipy.",
        "sklearn.",
    )

    # Tensor creation functions mapping
    TENSOR_FACTORIES = {
        ("np", "array"): "tensor",
        ("numpy", "array"): "tensor",
        ("np", "asarray"): "tensor",
        ("numpy", "asarray"): "tensor",
        ("torch", "tensor"): "tensor",
        ("torch", "Tensor"): "tensor",
        ("torch", "FloatTensor"): "tensor",
        ("torch", "as_tensor"): "tensor",
        ("th", "tensor"): "tensor",
        ("th", "Tensor"): "tensor",
    }

    # Math/helper functions mapping: (module, func) -> Synapse native function
    HELPER_FACTORIES = {
        ("np", "zeros"): "zeros",
        ("numpy", "zeros"): "zeros",
        ("torch", "zeros"): "zeros",
        ("th", "zeros"): "zeros",
        ("np", "zeros_like"): "zeros_like",
        ("numpy", "zeros_like"): "zeros_like",
        ("torch", "zeros_like"): "zeros_like",
        ("th", "zeros_like"): "zeros_like",
        ("np", "ones"): "ones",
        ("numpy", "ones"): "ones",
        ("torch", "ones"): "ones",
        ("th", "ones"): "ones",
        ("np", "ones_like"): "ones_like",
        ("numpy", "ones_like"): "ones_like",
        ("torch", "ones_like"): "ones_like",
        ("th", "ones_like"): "ones_like",
        ("torch", "randn"): "randn",
        ("th", "randn"): "randn",
        ("np", "random.randn"): "randn",
        ("torch", "eye"): "eye",
        ("np", "eye"): "eye",
        ("numpy", "eye"): "eye",
        ("torch", "arange"): "arange",
        ("th", "arange"): "arange",
        ("np", "arange"): "arange",
        ("numpy", "arange"): "arange",
        ("torch", "linspace"): "linspace",
        ("np", "linspace"): "linspace",
        ("numpy", "linspace"): "linspace",
        ("torch", "cat"): "concat",
        ("np", "concatenate"): "concat",
        ("numpy", "concatenate"): "concat",
        ("torch", "clamp"): "clamp",
        ("np", "clip"): "clamp",
        ("numpy", "clip"): "clamp",
        ("torch", "exp"): "exp",
        ("np", "exp"): "exp",
        ("numpy", "exp"): "exp",
        ("torch", "log"): "log",
        ("np", "log"): "log",
        ("numpy", "log"): "log",
        ("torch", "sqrt"): "sqrt",
        ("np", "sqrt"): "sqrt",
        ("numpy", "sqrt"): "sqrt",
        ("torch", "sin"): "sin",
        ("np", "sin"): "sin",
        ("numpy", "sin"): "sin",
        ("torch", "cos"): "cos",
        ("np", "cos"): "cos",
        ("numpy", "cos"): "cos",
        ("torch", "tan"): "tan",
        ("np", "tan"): "tan",
        ("numpy", "tan"): "tan",
        ("torch", "abs"): "abs",
        ("np", "abs"): "abs",
        ("numpy", "abs"): "abs",
        ("torch", "sum"): "sum",
        ("np", "sum"): "sum",
        ("numpy", "sum"): "sum",
        ("torch", "mean"): "mean",
        ("np", "mean"): "mean",
        ("numpy", "mean"): "mean",
        ("torch", "max"): "max",
        ("np", "max"): "max",
        ("numpy", "max"): "max",
        ("torch", "min"): "min",
        ("np", "min"): "min",
        ("numpy", "min"): "min",
    }

    # NN layer mappings
    NN_LAYERS = {
        "Linear",
        "Sequential",
        "ReLU",
        "GELU",
        "Sigmoid",
        "Tanh",
        "Dropout",
        "BatchNorm",
        "BatchNorm1d",
        "BatchNorm2d",
        "LayerNorm",
        "Embedding",
        "Conv2d",
        "MaxPool2d",
        "MSELoss",
        "CrossEntropyLoss",
        "Softmax",
        "LogSoftmax",
    }

    # Optimizers
    OPTIMIZERS = {
        "Adam",
        "AdamW",
        "SGD",
        "RMSprop",
    }

    # Functional activations and losses
    FUNCTIONAL_OPS = {
        "relu",
        "gelu",
        "sigmoid",
        "tanh",
        "softmax",
        "log_softmax",
        "mse_loss",
        "cross_entropy",
        "nll_loss",
    }

    # Operator precedence levels for parenthesizing
    PRECEDENCE = {
        "**": 9,
        "unary": 8,
        "@": 7,
        "*": 6,
        "/": 6,
        "//": 6,
        "%": 6,
        "+": 5,
        "-": 5,
        "<<": 4,
        ">>": 4,
        "&": 3,
        "^": 3,
        "|": 3,
        "<": 2,
        "<=": 2,
        ">": 2,
        ">=": 2,
        "==": 2,
        "!=": 2,
        "not": 1.5,
        "and": 1,
        "or": 0.5,
        "|>": 0,
    }

    def __init__(self, indent_spaces: int = 4) -> None:
        self.indent_spaces: int = indent_spaces
        self._tmp_counter: int = 0

    def _indent(self, level: int) -> str:
        return " " * (level * self.indent_spaces)

    def _new_tmp_var(self, prefix: str = "_tmp") -> str:
        self._tmp_counter += 1
        return f"{prefix}_{self._tmp_counter}"

    # =========================================================================
    # Public API
    # =========================================================================

    def migrate_code(self, python_code: str) -> str:
        """
        Parses a Python source code string and transforms it into Synapse syntax.

        Args:
            python_code: Source string written in Python.

        Returns:
            The converted Synapse source code string.
        """
        if not python_code or not python_code.strip():
            return ""

        try:
            tree = ast.parse(python_code)
        except SyntaxError as err:
            raise SyntaxError(f"Failed to parse Python code: {err}") from err

        lines: list[str] = []
        prev_is_block = False

        for stmt in tree.body:
            is_block = isinstance(
                stmt,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.If,
                    ast.While,
                    ast.For,
                ),
            )
            stmt_lines = self._visit_stmt(stmt, indent_level=0)
            if not stmt_lines:
                continue

            # Add extra newline between top-level functions/classes and statements for readability
            if lines and (is_block or prev_is_block):
                lines.append("")

            lines.extend(stmt_lines)
            prev_is_block = is_block

        result = "\n".join(lines)
        if result and not result.endswith("\n"):
            result += "\n"
        return result

    def migrate_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Reads a Python file, converts it into Synapse syntax, and optionally writes to output_path.

        Args:
            input_path: Path to the input Python source file.
            output_path: Optional path to write the converted Synapse file.

        Returns:
            The migrated Synapse source code string.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: '{input_path}'")

        with open(input_path, "r", encoding="utf-8") as f:
            python_code = f.read()

        synapse_code = self.migrate_code(python_code)

        if output_path is not None:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(synapse_code)

        return synapse_code

    # =========================================================================
    # Statement Visitor
    # =========================================================================

    def _visit_stmt(self, node: ast.stmt, indent_level: int) -> list[str]:
        """Translates a Python AST statement into lines of Synapse code."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._visit_function_def(node, indent_level)
        elif isinstance(node, ast.ClassDef):
            return self._visit_class_def(node, indent_level)
        elif isinstance(node, ast.AnnAssign):
            return self._visit_ann_assign(node, indent_level)
        elif isinstance(node, ast.Assign):
            return self._visit_assign(node, indent_level)
        elif isinstance(node, ast.AugAssign):
            return self._visit_aug_assign(node, indent_level)
        elif isinstance(node, ast.If):
            return self._visit_if(node, indent_level)
        elif isinstance(node, ast.While):
            return self._visit_while(node, indent_level)
        elif isinstance(node, ast.For):
            return self._visit_for(node, indent_level)
        elif isinstance(node, ast.Return):
            return self._visit_return(node, indent_level)
        elif isinstance(node, ast.Pass):
            return [f"{self._indent(indent_level)}pass"]
        elif isinstance(node, ast.Break):
            return [f"{self._indent(indent_level)}break"]
        elif isinstance(node, ast.Continue):
            return [f"{self._indent(indent_level)}continue"]
        elif isinstance(node, ast.Import):
            return self._visit_import(node, indent_level)
        elif isinstance(node, ast.ImportFrom):
            return self._visit_import_from(node, indent_level)
        elif isinstance(node, ast.Expr):
            return self._visit_expr_stmt(node, indent_level)
        elif isinstance(node, ast.Assert):
            return self._visit_assert(node, indent_level)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            return self._visit_with(node, indent_level)
        elif isinstance(node, ast.Try):
            return self._visit_try(node, indent_level)
        else:
            # Fallback for other statements
            expr_str = self.convert_expr(node)
            if expr_str:
                return [f"{self._indent(indent_level)}{expr_str}"]
            return []

    def _visit_function_def(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        indent_level: int,
        prefix_name: str = "",
    ) -> list[str]:
        """Converts `def foo(...) -> T:` into `fn foo(...) -> T:`."""
        fn_name = f"{prefix_name}{node.name}" if prefix_name else node.name

        # Parse parameters and default values
        params_list: list[str] = []
        pos_args = node.args.posonlyargs + node.args.args
        num_pos = len(pos_args)
        num_defaults = len(node.args.defaults)
        offset = num_pos - num_defaults

        for idx, arg in enumerate(pos_args):
            p_str = arg.arg
            if arg.annotation:
                p_str += f": {self.convert_type(arg.annotation)}"
            if idx >= offset:
                default_node = node.args.defaults[idx - offset]
                p_str += f" = {self.convert_expr(default_node)}"
            params_list.append(p_str)

        # Keyword-only arguments
        for idx, arg in enumerate(node.args.kwonlyargs):
            p_str = arg.arg
            if arg.annotation:
                p_str += f": {self.convert_type(arg.annotation)}"
            if idx < len(node.args.kw_defaults):
                kw_def = node.args.kw_defaults[idx]
                if kw_def is not None:
                    p_str += f" = {self.convert_expr(kw_def)}"
            params_list.append(p_str)

        # Vararg and kwarg (Synapse uses standard identifiers for them)
        if node.args.vararg:
            v_name = node.args.vararg.arg
            if node.args.vararg.annotation:
                v_name += f": {self.convert_type(node.args.vararg.annotation)}"
            params_list.append(v_name)
        if node.args.kwarg:
            k_name = node.args.kwarg.arg
            if node.args.kwarg.annotation:
                k_name += f": {self.convert_type(node.args.kwarg.annotation)}"
            params_list.append(k_name)

        # Return type annotation
        ret_annot = ""
        if node.returns:
            ret_annot = f" -> {self.convert_type(node.returns)}"

        header = f"{self._indent(indent_level)}fn {fn_name}({', '.join(params_list)}){ret_annot}:"

        body_lines: list[str] = []
        for body_stmt in node.body:
            body_lines.extend(self._visit_stmt(body_stmt, indent_level + 1))

        if not body_lines:
            body_lines.append(f"{self._indent(indent_level + 1)}pass")

        return [header] + body_lines

    def _visit_class_def(self, node: ast.ClassDef, indent_level: int) -> list[str]:
        """
        Translates Python `class Name:` by decomposing methods into idiomatic Synapse functions
        prefixed with `{ClassName}_` and desugaring class attributes.
        """
        lines: list[str] = [f"{self._indent(indent_level)}# class {node.name}"]
        class_name = node.name

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Convert methods into fn ClassName_method(self, ...)
                method_name = item.name
                if method_name.startswith("__") and method_name.endswith("__"):
                    clean_name = method_name.strip("_")
                    target_name = f"{class_name}_{clean_name}"
                else:
                    target_name = f"{class_name}_{method_name}"

                m_lines = self._visit_function_def(
                    item, indent_level, prefix_name=""
                )
                if m_lines:
                    # Replace the generated function header name with target_name
                    orig_header = m_lines[0]
                    # Format: "    fn orig_name(...):" -> "    fn target_name(...):"
                    fn_idx = orig_header.find("fn ")
                    if fn_idx != -1:
                        paren_idx = orig_header.find("(", fn_idx)
                        if paren_idx != -1:
                            new_header = (
                                orig_header[: fn_idx + 3]
                                + target_name
                                + orig_header[paren_idx:]
                            )
                            m_lines[0] = new_header
                    lines.append("")
                    lines.extend(m_lines)
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                lines.extend(self._visit_stmt(item, indent_level))

        return lines

    def _is_constant_name(self, name: str) -> bool:
        """Determines if an identifier name follows uppercase constant naming conventions."""
        return name.isupper() and any(c.isalpha() for c in name)

    def _visit_ann_assign(self, node: ast.AnnAssign, indent_level: int) -> list[str]:
        """Converts typed assignment `x: int = 10` -> `let x: int = 10`."""
        if isinstance(node.target, ast.Name):
            name = node.target.id
            type_str = self.convert_type(node.annotation)
            kw = "const" if self._is_constant_name(name) else "let"
            if node.value is not None:
                # Handle desugaring of ternary IfExp
                if isinstance(node.value, ast.IfExp):
                    return self._desugar_ifexp_assign(
                        name, kw, node.value, indent_level, type_str=type_str
                    )
                # Handle desugaring of ListComp
                if isinstance(node.value, ast.ListComp):
                    return self._desugar_listcomp_assign(
                        name, kw, node.value, indent_level
                    )
                # Handle desugaring of DictComp
                if isinstance(node.value, ast.DictComp):
                    return self._desugar_dictcomp_assign(
                        name, kw, node.value, indent_level
                    )

                val_str = self.convert_expr(node.value)
                return [f"{self._indent(indent_level)}{kw} {name}: {type_str} = {val_str}"]
            else:
                return [f"{self._indent(indent_level)}{kw} {name}: {type_str} = none"]
        else:
            # Attribute or Subscript target (e.g. self.x: int = 10)
            target_str = self.convert_expr(node.target)
            if node.value is not None:
                val_str = self.convert_expr(node.value)
                return [f"{self._indent(indent_level)}{target_str} = {val_str}"]
            return []

    def _visit_assign(self, node: ast.Assign, indent_level: int) -> list[str]:
        """Converts untyped assignment `x = 10` -> `let x = 10` or `const MAX = 10`."""
        # Check for lambda in single target assignment: f = lambda x: x * 2 -> fn f(x): return x * 2
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Lambda):
            target_name = node.targets[0].id
            return self._desugar_lambda_assign(target_name, node.value, indent_level)

        # Check for IfExp desugaring in single target assignment
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.IfExp):
            name = node.targets[0].id
            kw = "const" if self._is_constant_name(name) else "let"
            return self._desugar_ifexp_assign(name, kw, node.value, indent_level)

        # Check for ListComp desugaring in single target assignment
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.ListComp):
            name = node.targets[0].id
            kw = "const" if self._is_constant_name(name) else "let"
            return self._desugar_listcomp_assign(name, kw, node.value, indent_level)

        # Check for DictComp desugaring in single target assignment
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.DictComp):
            name = node.targets[0].id
            kw = "const" if self._is_constant_name(name) else "let"
            return self._desugar_dictcomp_assign(name, kw, node.value, indent_level)

        val_str = self.convert_expr(node.value)
        lines: list[str] = []

        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                kw = "const" if self._is_constant_name(name) else "let"
                lines.append(f"{self._indent(indent_level)}{kw} {name} = {val_str}")
            elif isinstance(target, (ast.Attribute, ast.Subscript)):
                target_str = self.convert_expr(target)
                lines.append(f"{self._indent(indent_level)}{target_str} = {val_str}")
            elif isinstance(target, (ast.Tuple, ast.List)):
                # Tuple unpacking: a, b = 1, 2
                if isinstance(node.value, (ast.Tuple, ast.List)) and len(node.value.elts) == len(target.elts):
                    for t, v in zip(target.elts, node.value.elts):
                        if isinstance(t, ast.Name):
                            kw = "const" if self._is_constant_name(t.id) else "let"
                            lines.append(f"{self._indent(indent_level)}{kw} {t.id} = {self.convert_expr(v)}")
                        else:
                            lines.append(f"{self._indent(indent_level)}{self.convert_expr(t)} = {self.convert_expr(v)}")
                else:
                    # If right hand side is a complex call or expression, evaluate once to avoid side effects
                    if isinstance(node.value, (ast.Call, ast.BinOp)):
                        tmp_var = self._new_tmp_var("_unpack")
                        lines.append(f"{self._indent(indent_level)}let {tmp_var} = {val_str}")
                        source_ref = tmp_var
                    else:
                        source_ref = val_str

                    for idx, t in enumerate(target.elts):
                        if isinstance(t, ast.Name):
                            kw = "const" if self._is_constant_name(t.id) else "let"
                            lines.append(f"{self._indent(indent_level)}{kw} {t.id} = {source_ref}[{idx}]")
                        else:
                            lines.append(f"{self._indent(indent_level)}{self.convert_expr(t)} = {source_ref}[{idx}]")
            else:
                target_str = self.convert_expr(target)
                lines.append(f"{self._indent(indent_level)}{target_str} = {val_str}")

        return lines

    def _desugar_lambda_assign(self, name: str, lam: ast.Lambda, indent_level: int) -> list[str]:
        """Desugars `f = lambda x, y: x + y` into `fn f(x, y): return x + y`."""
        args_strs = [a.arg for a in lam.args.args]
        body_str = self.convert_expr(lam.body)
        header = f"{self._indent(indent_level)}fn {name}({', '.join(args_strs)}):"
        return [header, f"{self._indent(indent_level + 1)}return {body_str}"]

    def _desugar_ifexp_assign(
        self,
        name: str,
        kw: str,
        if_exp: ast.IfExp,
        indent_level: int,
        type_str: Optional[str] = None,
    ) -> list[str]:
        """Desugars `x = a if cond else b` into clean `if cond: x = a else: x = b`."""
        test_str = self.convert_expr(if_exp.test)
        body_str = self.convert_expr(if_exp.body)
        orelse_str = self.convert_expr(if_exp.orelse)

        type_annot = f": {type_str}" if type_str else ""
        lines: list[str] = [
            f"{self._indent(indent_level)}{kw} {name}{type_annot} = none",
            f"{self._indent(indent_level)}if {test_str}:",
            f"{self._indent(indent_level + 1)}{name} = {body_str}",
            f"{self._indent(indent_level)}else:",
            f"{self._indent(indent_level + 1)}{name} = {orelse_str}",
        ]
        return lines

    def _desugar_listcomp_assign(
        self, name: str, kw: str, comp: ast.ListComp, indent_level: int
    ) -> list[str]:
        """Desugars `squares = [x * 2 for x in items if x > 0]` into a loop with `.append()`."""
        lines: list[str] = [f"{self._indent(indent_level)}{kw} {name} = []"]
        curr_indent = indent_level

        for gen in comp.generators:
            target_str = self.convert_expr(gen.target)
            iter_str = self.convert_expr(gen.iter)
            lines.append(f"{self._indent(curr_indent)}for {target_str} in {iter_str}:")
            curr_indent += 1

            for if_expr in gen.ifs:
                cond_str = self.convert_expr(if_expr)
                lines.append(f"{self._indent(curr_indent)}if {cond_str}:")
                curr_indent += 1

        elt_str = self.convert_expr(comp.elt)
        lines.append(f"{self._indent(curr_indent)}{name}.append({elt_str})")
        return lines

    def _desugar_dictcomp_assign(
        self, name: str, kw: str, comp: ast.DictComp, indent_level: int
    ) -> list[str]:
        """Desugars `{k: v for k, v in items}` into a loop."""
        lines: list[str] = [f"{self._indent(indent_level)}{kw} {name} = {{}}"]
        curr_indent = indent_level

        for gen in comp.generators:
            # Check for tuple unpacking in dict comprehension
            if isinstance(gen.target, (ast.Tuple, ast.List)):
                tmp_item = self._new_tmp_var("_item")
                iter_str = self.convert_expr(gen.iter)
                lines.append(f"{self._indent(curr_indent)}for {tmp_item} in {iter_str}:")
                curr_indent += 1
                for idx, t in enumerate(gen.target.elts):
                    if isinstance(t, ast.Name):
                        lines.append(f"{self._indent(curr_indent)}let {t.id} = {tmp_item}[{idx}]")
            else:
                target_str = self.convert_expr(gen.target)
                iter_str = self.convert_expr(gen.iter)
                lines.append(f"{self._indent(curr_indent)}for {target_str} in {iter_str}:")
                curr_indent += 1

            for if_expr in gen.ifs:
                cond_str = self.convert_expr(if_expr)
                lines.append(f"{self._indent(curr_indent)}if {cond_str}:")
                curr_indent += 1

        key_str = self.convert_expr(comp.key)
        val_str = self.convert_expr(comp.value)
        lines.append(f"{self._indent(curr_indent)}{name}[{key_str}] = {val_str}")
        return lines

    def _visit_aug_assign(self, node: ast.AugAssign, indent_level: int) -> list[str]:
        """Converts in-place assignments `x += 1`."""
        target_str = self.convert_expr(node.target)
        val_str = self.convert_expr(node.value)
        op_str = self._convert_aug_op(node.op)
        return [f"{self._indent(indent_level)}{target_str} {op_str}= {val_str}"]

    def _convert_aug_op(self, op: ast.operator) -> str:
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return "/"
        elif isinstance(op, ast.FloorDiv):
            return "//"
        elif isinstance(op, ast.Mod):
            return "%"
        elif isinstance(op, ast.MatMult):
            return "@"
        elif isinstance(op, ast.Pow):
            return "**"
        elif isinstance(op, ast.BitOr):
            return "|"
        return "+"

    def _visit_if(self, node: ast.If, indent_level: int) -> list[str]:
        """Converts `if / elif / else` structures."""
        test_str = self.convert_expr(node.test)
        lines = [f"{self._indent(indent_level)}if {test_str}:"]

        body_lines: list[str] = []
        for s in node.body:
            body_lines.extend(self._visit_stmt(s, indent_level + 1))
        if not body_lines:
            body_lines.append(f"{self._indent(indent_level + 1)}pass")
        lines.extend(body_lines)

        current_node = node
        while current_node.orelse:
            # Check for chained elif
            if len(current_node.orelse) == 1 and isinstance(current_node.orelse[0], ast.If):
                elif_node = current_node.orelse[0]
                elif_test_str = self.convert_expr(elif_node.test)
                lines.append(f"{self._indent(indent_level)}elif {elif_test_str}:")
                elif_body: list[str] = []
                for s in elif_node.body:
                    elif_body.extend(self._visit_stmt(s, indent_level + 1))
                if not elif_body:
                    elif_body.append(f"{self._indent(indent_level + 1)}pass")
                lines.extend(elif_body)
                current_node = elif_node
            else:
                # Final else block
                lines.append(f"{self._indent(indent_level)}else:")
                else_body: list[str] = []
                for s in current_node.orelse:
                    else_body.extend(self._visit_stmt(s, indent_level + 1))
                if not else_body:
                    else_body.append(f"{self._indent(indent_level + 1)}pass")
                lines.extend(else_body)
                break

        return lines

    def _visit_while(self, node: ast.While, indent_level: int) -> list[str]:
        """Converts `while cond:` blocks."""
        test_str = self.convert_expr(node.test)
        lines = [f"{self._indent(indent_level)}while {test_str}:"]

        body_lines: list[str] = []
        for s in node.body:
            body_lines.extend(self._visit_stmt(s, indent_level + 1))
        if not body_lines:
            body_lines.append(f"{self._indent(indent_level + 1)}pass")
        lines.extend(body_lines)
        return lines

    def _visit_for(self, node: ast.For, indent_level: int) -> list[str]:
        """Converts `for i in range(n):` blocks, unpacking tuples if necessary."""
        iter_str = self.convert_expr(node.iter)

        # Handle tuple unpacking: `for x, y in pairs:` -> Synapse requires a single identifier
        if isinstance(node.target, (ast.Tuple, ast.List)):
            tmp_item = self._new_tmp_var("_item")
            lines = [f"{self._indent(indent_level)}for {tmp_item} in {iter_str}:"]

            body_lines: list[str] = []
            for idx, t in enumerate(node.target.elts):
                if isinstance(t, ast.Name):
                    kw = "const" if self._is_constant_name(t.id) else "let"
                    body_lines.append(f"{self._indent(indent_level + 1)}{kw} {t.id} = {tmp_item}[{idx}]")
                else:
                    body_lines.append(f"{self._indent(indent_level + 1)}{self.convert_expr(t)} = {tmp_item}[{idx}]")

            for s in node.body:
                body_lines.extend(self._visit_stmt(s, indent_level + 1))
            if not body_lines:
                body_lines.append(f"{self._indent(indent_level + 1)}pass")
            lines.extend(body_lines)
            return lines

        target_str = self.convert_expr(node.target)
        lines = [f"{self._indent(indent_level)}for {target_str} in {iter_str}:"]

        body_lines = []
        for s in node.body:
            body_lines.extend(self._visit_stmt(s, indent_level + 1))
        if not body_lines:
            body_lines.append(f"{self._indent(indent_level + 1)}pass")
        lines.extend(body_lines)
        return lines

    def _visit_return(self, node: ast.Return, indent_level: int) -> list[str]:
        """Converts `return` or `return expr`, desugaring IfExp if present."""
        if node.value is None:
            return [f"{self._indent(indent_level)}return"]

        # Desugar `return a if cond else b` into conditional returns
        if isinstance(node.value, ast.IfExp):
            test_str = self.convert_expr(node.value.test)
            body_str = self.convert_expr(node.value.body)
            orelse_str = self.convert_expr(node.value.orelse)
            return [
                f"{self._indent(indent_level)}if {test_str}:",
                f"{self._indent(indent_level + 1)}return {body_str}",
                f"{self._indent(indent_level)}else:",
                f"{self._indent(indent_level + 1)}return {orelse_str}",
            ]

        # Desugar `return [x for x in items]`
        if isinstance(node.value, ast.ListComp):
            tmp_ret = self._new_tmp_var("_res")
            comp_lines = self._desugar_listcomp_assign(tmp_ret, "let", node.value, indent_level)
            comp_lines.append(f"{self._indent(indent_level)}return {tmp_ret}")
            return comp_lines

        return [f"{self._indent(indent_level)}return {self.convert_expr(node.value)}"]

    def _visit_import(self, node: ast.Import, indent_level: int) -> list[str]:
        """Filters or translates standard `import ...` statements."""
        lines: list[str] = []
        for alias in node.names:
            mod_name = alias.name
            as_name = alias.asname

            if mod_name in self.STRIP_MODULES or any(mod_name.startswith(p) for p in self.STRIP_PREFIXES):
                continue

            if mod_name.startswith("py."):
                if as_name:
                    lines.append(f"{self._indent(indent_level)}import {mod_name} as {as_name}")
                else:
                    lines.append(f"{self._indent(indent_level)}import {mod_name}")
            else:
                alias_target = as_name if as_name else mod_name.split(".")[-1]
                lines.append(f"{self._indent(indent_level)}import py.{mod_name} as {alias_target}")

        return lines

    def _visit_import_from(self, node: ast.ImportFrom, indent_level: int) -> list[str]:
        """Filters or translates `from ... import ...` statements."""
        mod = node.module or ""
        if mod in self.STRIP_MODULES or any(mod.startswith(p) for p in self.STRIP_PREFIXES):
            return []

        # Relative import e.g. from . import sibling
        if not mod:
            lines: list[str] = []
            for alias in node.names:
                if alias.asname:
                    lines.append(f"{self._indent(indent_level)}import {alias.name} as {alias.asname}")
                else:
                    lines.append(f"{self._indent(indent_level)}import {alias.name}")
            return lines

        # Standard library math functions that Synapse has built-in natively
        math_builtins = {"sqrt", "exp", "log", "sin", "cos", "tan", "fabs", "pi", "e"}
        if mod == "math" and all(a.name in math_builtins for a in node.names):
            return []

        # Synapse imports: `import py.<mod>.<symbol> as <alias>`
        lines = []
        for alias in node.names:
            target_alias = alias.asname or alias.name
            if mod.startswith("py."):
                lines.append(f"{self._indent(indent_level)}import {mod}.{alias.name} as {target_alias}")
            else:
                lines.append(f"{self._indent(indent_level)}import py.{mod}.{alias.name} as {target_alias}")

        return lines

    def _visit_expr_stmt(self, node: ast.Expr, indent_level: int) -> list[str]:
        """Converts expression statements and docstrings."""
        val_str = self.convert_expr(node.value)
        if not val_str:
            return []
        return [f"{self._indent(indent_level)}{val_str}"]

    def _visit_assert(self, node: ast.Assert, indent_level: int) -> list[str]:
        """Converts Python `assert cond` into Synapse builtin `assert(cond)`."""
        test_str = self.convert_expr(node.test)
        if node.msg:
            msg_str = self.convert_expr(node.msg)
            return [f"{self._indent(indent_level)}assert({test_str}, {msg_str})"]
        return [f"{self._indent(indent_level)}assert({test_str})"]

    def _visit_with(self, node: Union[ast.With, ast.AsyncWith], indent_level: int) -> list[str]:
        """
        Translates context managers `with torch.no_grad():` by preserving the inner execution
        logic and emitting a comment header.
        """
        item_descs: list[str] = []
        for item in node.items:
            ctx_str = self.convert_expr(item.context_expr)
            if item.optional_vars:
                var_str = self.convert_expr(item.optional_vars)
                item_descs.append(f"{ctx_str} as {var_str}")
            else:
                item_descs.append(ctx_str)

        lines: list[str] = [f"{self._indent(indent_level)}# with {', '.join(item_descs)}:"]
        for s in node.body:
            lines.extend(self._visit_stmt(s, indent_level))
        return lines

    def _visit_try(self, node: ast.Try, indent_level: int) -> list[str]:
        """Preserves try-except blocks by executing the protected statements."""
        lines: list[str] = [f"{self._indent(indent_level)}# try:"]
        for s in node.body:
            lines.extend(self._visit_stmt(s, indent_level))
        if node.finalbody:
            lines.append(f"{self._indent(indent_level)}# finally:")
            for s in node.finalbody:
                lines.extend(self._visit_stmt(s, indent_level))
        return lines

    # =========================================================================
    # Type Annotation Converter
    # =========================================================================

    def convert_type(self, node: Optional[ast.AST]) -> str:
        """Transforms a Python type annotation node into a Synapse type signature."""
        if node is None:
            return "Any"

        if isinstance(node, ast.Name):
            t_id = node.id
            if t_id in ("ndarray", "Tensor"):
                return "Tensor"
            elif t_id in ("int", "float", "str", "bool", "Any"):
                return t_id
            elif t_id in ("List", "list"):
                return "list"
            elif t_id in ("Dict", "dict"):
                return "dict"
            elif t_id in ("Tuple", "tuple"):
                return "list"
            elif t_id in ("None", "NoneType"):
                return "None"
            return t_id

        elif isinstance(node, ast.Constant):
            if node.value is None:
                return "None"
            return str(node.value)

        elif isinstance(node, ast.Attribute):
            full_attr = f"{self.convert_expr(node.value)}.{node.attr}"
            if full_attr in ("np.ndarray", "numpy.ndarray", "torch.Tensor", "torch.FloatTensor"):
                return "Tensor"
            return node.attr

        elif isinstance(node, ast.Subscript):
            base_type = self.convert_type(node.value)
            slice_node = node.slice

            if base_type == "Optional":
                inner_type = self.convert_type(slice_node)
                return f"{inner_type} | None"
            elif base_type == "Union":
                if isinstance(slice_node, ast.Tuple):
                    types = [self.convert_type(t) for t in slice_node.elts]
                    return " | ".join(types)
                return self.convert_type(slice_node)
            elif base_type == "Tensor":
                if isinstance(slice_node, ast.Tuple):
                    dims = [self.convert_type(t) for t in slice_node.elts]
                    return f"Tensor[{', '.join(dims)}]"
                return f"Tensor[{self.convert_type(slice_node)}]"
            elif base_type in ("Tuple", "tuple"):
                # Synapse represents composite sequence types as lists
                if isinstance(slice_node, ast.Tuple):
                    # Filter out Ellipsis (...)
                    args = [self.convert_type(t) for t in slice_node.elts if not (isinstance(t, ast.Constant) and t.value == Ellipsis)]
                    if args:
                        return f"list[{args[0]}]"
                    return "list"
                return "list"
            else:
                if isinstance(slice_node, ast.Tuple):
                    args = [self.convert_type(t) for t in slice_node.elts]
                    return f"{base_type}[{', '.join(args)}]"
                return f"{base_type}[{self.convert_type(slice_node)}]"

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self.convert_type(node.left)
            right = self.convert_type(node.right)
            return f"{left} | {right}"

        return "Any"

    # =========================================================================
    # Expression Converter
    # =========================================================================

    def convert_expr(self, node: ast.AST) -> str:
        """Transforms a Python AST expression node into Synapse syntax."""
        if isinstance(node, ast.Constant):
            return self._convert_constant(node)
        elif isinstance(node, ast.Name):
            return self._convert_name(node)
        elif isinstance(node, ast.UnaryOp):
            return self._convert_unary(node)
        elif isinstance(node, ast.BinOp):
            return self._convert_binary(node)
        elif isinstance(node, ast.BoolOp):
            return self._convert_bool_op(node)
        elif isinstance(node, ast.Compare):
            return self._convert_compare(node)
        elif isinstance(node, ast.Call):
            return self._convert_call(node)
        elif isinstance(node, ast.Attribute):
            return self._convert_attribute(node)
        elif isinstance(node, ast.Subscript):
            return self._convert_subscript(node)
        elif isinstance(node, ast.List):
            elts = [self.convert_expr(e) for e in node.elts]
            return f"[{', '.join(elts)}]"
        elif isinstance(node, ast.Tuple):
            # In Synapse syntax, bracketed lists are standard for sequences and shapes
            elts = [self.convert_expr(e) for e in node.elts]
            return f"[{', '.join(elts)}]"
        elif isinstance(node, ast.Dict):
            entries = [
                f"{self.convert_expr(k)}: {self.convert_expr(v)}"
                for k, v in zip(node.keys, node.values)
                if k is not None
            ]
            return f"{{{', '.join(entries)}}}"
        elif isinstance(node, ast.JoinedStr):
            return self._convert_joined_str(node)
        elif isinstance(node, ast.Slice):
            lower = self.convert_expr(node.lower) if node.lower else ""
            upper = self.convert_expr(node.upper) if node.upper else ""
            return f"{lower}:{upper}"
        else:
            return ""

    def _convert_constant(self, node: ast.Constant) -> str:
        val = node.value
        if val is True:
            return "true"
        elif val is False:
            return "false"
        elif val is None:
            return "none"
        elif isinstance(val, str):
            # Multiline string or docstring
            if "\n" in val:
                escaped = val.replace('"""', r'\"\"\"')
                return f'"""{escaped}"""'
            return repr(val)
        elif isinstance(val, (int, float)):
            return str(val)
        return repr(val)

    def _convert_name(self, node: ast.Name) -> str:
        name = node.id
        if name == "True":
            return "true"
        elif name == "False":
            return "false"
        elif name == "None":
            return "none"
        return name

    def _convert_unary(self, node: ast.UnaryOp) -> str:
        op = node.op
        operand_str = self.convert_expr(node.operand)

        # Parenthesize inner binary operations
        if isinstance(node.operand, (ast.BinOp, ast.BoolOp, ast.Compare)):
            operand_str = f"({operand_str})"

        if isinstance(op, ast.USub):
            return f"-{operand_str}"
        elif isinstance(op, ast.UAdd):
            return f"+{operand_str}"
        elif isinstance(op, ast.Not):
            return f"not {operand_str}"
        elif isinstance(op, ast.Invert):
            return f"~{operand_str}"
        return f"-{operand_str}"

    def _convert_binary(self, node: ast.BinOp) -> str:
        op_map: dict[type, str] = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.MatMult: "@",
            ast.BitOr: "|",
            ast.BitAnd: "&",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }
        op_str = op_map.get(type(node.op), "+")
        parent_prec = self.PRECEDENCE.get(op_str, 5)

        # Format left operand with precedence checking
        left_str = self.convert_expr(node.left)
        if isinstance(node.left, ast.BinOp):
            left_op_str = op_map.get(type(node.left.op), "+")
            left_prec = self.PRECEDENCE.get(left_op_str, 5)
            if left_prec < parent_prec:
                left_str = f"({left_str})"
        elif isinstance(node.left, (ast.BoolOp, ast.Compare)):
            left_str = f"({left_str})"

        # Format right operand with precedence checking
        right_str = self.convert_expr(node.right)
        if isinstance(node.right, ast.BinOp):
            right_op_str = op_map.get(type(node.right.op), "+")
            right_prec = self.PRECEDENCE.get(right_op_str, 5)
            if right_prec < parent_prec or (
                right_prec == parent_prec and op_str in ("-", "/", "//", "%", "**")
            ):
                right_str = f"({right_str})"
        elif isinstance(node.right, (ast.BoolOp, ast.Compare)):
            right_str = f"({right_str})"

        return f"{left_str} {op_str} {right_str}"

    def _convert_bool_op(self, node: ast.BoolOp) -> str:
        op_str = " and " if isinstance(node.op, ast.And) else " or "
        parts: list[str] = []
        for val in node.values:
            val_str = self.convert_expr(val)
            if isinstance(node.op, ast.And) and isinstance(val, ast.BoolOp) and isinstance(val.op, ast.Or):
                val_str = f"({val_str})"
            parts.append(val_str)
        return op_str.join(parts)

    def _convert_compare(self, node: ast.Compare) -> str:
        """
        Converts Python comparisons.
        Special handling:
        - `x in items` -> `contains(items, x)`
        - `x not in items` -> `not contains(items, x)`
        - Chained comparisons `a < b < c` -> `(a < b) and (b < c)`
        """
        # Handle `x in items` and `x not in items`
        if len(node.ops) == 1:
            op = node.ops[0]
            if isinstance(op, ast.In):
                target_str = self.convert_expr(node.left)
                container_str = self.convert_expr(node.comparators[0])
                return f"contains({container_str}, {target_str})"
            elif isinstance(op, ast.NotIn):
                target_str = self.convert_expr(node.left)
                container_str = self.convert_expr(node.comparators[0])
                return f"not contains({container_str}, {target_str})"

        cmp_map: dict[type, str] = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "==",
            ast.IsNot: "!=",
        }

        # If chained comparison (e.g. 1 < x < 10), desugar into (1 < x) and (x < 10)
        if len(node.ops) > 1:
            clauses: list[str] = []
            curr_left = node.left
            for op, comp in zip(node.ops, node.comparators):
                op_str = cmp_map.get(type(op), "==")
                c_left = self.convert_expr(curr_left)
                c_right = self.convert_expr(comp)
                clauses.append(f"({c_left} {op_str} {c_right})")
                curr_left = comp
            return " and ".join(clauses)

        parts = [self.convert_expr(node.left)]
        for op, comp in zip(node.ops, node.comparators):
            op_str = cmp_map.get(type(op), "==")
            comp_str = self.convert_expr(comp)
            parts.append(f"{op_str} {comp_str}")
        return " ".join(parts)

    def _convert_call(self, node: ast.Call) -> str:
        """Translates Python calls, handling tensor factories, math functions, methods, and layers."""
        if isinstance(node.func, ast.Attribute):
            caller_str = self.convert_expr(node.func.value)
            attr_name = node.func.attr

            # 1. np.array / torch.tensor -> tensor(...)
            if (caller_str, attr_name) in self.TENSOR_FACTORIES:
                return self._render_tensor_call(node)

            # 2. zeros, ones, zeros_like, ones_like, randn, eye, arange, linspace, clamp
            if (caller_str, attr_name) in self.HELPER_FACTORIES:
                factory_name = self.HELPER_FACTORIES[(caller_str, attr_name)]
                args = [self.convert_expr(a) for a in node.args]
                kwargs = [f"{kw.arg}={self.convert_expr(kw.value)}" for kw in node.keywords]
                return f"{factory_name}({', '.join(args + kwargs)})"

            # 3. Matrix multiplication via function: np.dot(a, b), torch.matmul(a, b), torch.mm(a, b)
            if (caller_str in ("np", "numpy", "torch", "th")) and attr_name in ("dot", "matmul", "mm"):
                if len(node.args) == 2 and not node.keywords:
                    a_str = self.convert_expr(node.args[0])
                    b_str = self.convert_expr(node.args[1])
                    return f"({a_str} @ {b_str})"

            # 4. Matrix multiplication via method: a.dot(b), x.matmul(w), x.mm(w)
            if attr_name in ("dot", "matmul", "mm") and len(node.args) == 1 and not node.keywords:
                a_str = caller_str
                b_str = self.convert_expr(node.args[0])
                return f"({a_str} @ {b_str})"

            # 5. Device calls: torch.device("cuda") -> "cuda" or Device("cuda")
            if caller_str in ("torch", "th") and attr_name == "device":
                if node.args:
                    return self.convert_expr(node.args[0])
                return '"cpu"'

            # 6. CUDA check: torch.cuda.is_available() -> is_cuda_available()
            if (caller_str in ("torch.cuda", "cuda") and attr_name == "is_available"):
                return "is_cuda_available()"

            # 7. Tensor sizing: x.size() -> x.shape, x.size(0) -> x.shape[0], x.dim() -> x.ndim
            if attr_name == "size":
                if len(node.args) == 1:
                    idx_str = self.convert_expr(node.args[0])
                    return f"{caller_str}.shape[{idx_str}]"
                elif len(node.args) == 0:
                    return f"{caller_str}.shape"
            if attr_name == "dim" and len(node.args) == 0:
                return f"{caller_str}.ndim"

            # 8. Transpose method: x.t() -> x.T
            if attr_name == "t" and len(node.args) == 0:
                return f"{caller_str}.T"

            # 9. Neural network layers: nn.Linear(...) -> Linear(...)
            if caller_str in ("nn", "torch.nn") and attr_name in self.NN_LAYERS:
                args = [self.convert_expr(a) for a in node.args]
                kwargs = [f"{kw.arg}={self.convert_expr(kw.value)}" for kw in node.keywords]
                return f"{attr_name}({', '.join(args + kwargs)})"

            # 10. Optimizers: optim.Adam(...) -> Adam(...)
            if caller_str in ("optim", "torch.optim") and attr_name in self.OPTIMIZERS:
                args = [self.convert_expr(a) for a in node.args]
                kwargs = [f"{kw.arg}={self.convert_expr(kw.value)}" for kw in node.keywords]
                return f"{attr_name}({', '.join(args + kwargs)})"

            # 11. Functional activations: F.relu(x) -> relu(x)
            if caller_str in ("F", "torch.nn.functional", "functional") and attr_name in self.FUNCTIONAL_OPS:
                args = [self.convert_expr(a) for a in node.args]
                kwargs = [f"{kw.arg}={self.convert_expr(kw.value)}" for kw in node.keywords]
                return f"{attr_name}({', '.join(args + kwargs)})"

        # General function call
        callee_str = self.convert_expr(node.func)
        args_strs = [self.convert_expr(a) for a in node.args]
        kwargs_strs = [f"{kw.arg}={self.convert_expr(kw.value)}" for kw in node.keywords]
        return f"{callee_str}({', '.join(args_strs + kwargs_strs)})"

    def _render_tensor_call(self, node: ast.Call) -> str:
        """Renders `tensor(...)` with translated kwargs (e.g. requires_grad=true)."""
        args_strs = [self.convert_expr(a) for a in node.args]
        kwargs_strs: list[str] = []

        for kw in node.keywords:
            kw_name = kw.arg
            if kw_name == "requires_grad":
                val = self.convert_expr(kw.value)
                kwargs_strs.append(f"requires_grad={val}")
            elif kw_name == "dtype":
                kwargs_strs.append(f"dtype={self._convert_dtype_expr(kw.value)}")
            elif kw_name == "device":
                val = self.convert_expr(kw.value)
                kwargs_strs.append(f"device={val}")
            else:
                if kw_name:
                    kwargs_strs.append(f"{kw_name}={self.convert_expr(kw.value)}")

        return f"tensor({', '.join(args_strs + kwargs_strs)})"

    def _convert_dtype_expr(self, node: ast.AST) -> str:
        """Converts framework dtypes like `torch.float32` or `np.float32` to `'float32'`."""
        if isinstance(node, ast.Attribute):
            attr = node.attr
            mapping = {
                "float32": '"float32"',
                "float": '"float32"',
                "float64": '"float64"',
                "double": '"float64"',
                "float16": '"float16"',
                "half": '"float16"',
                "int32": '"int32"',
                "int": '"int32"',
                "int64": '"int64"',
                "long": '"int64"',
                "bool": '"bool"',
            }
            return mapping.get(attr, f'"{attr}"')
        elif isinstance(node, ast.Constant):
            return repr(str(node.value))
        return self.convert_expr(node)

    def _convert_attribute(self, node: ast.Attribute) -> str:
        value_str = self.convert_expr(node.value)
        # Check for torch/numpy math attributes: e.g. np.pi -> pi
        if value_str in ("np", "numpy", "math") and node.attr in ("pi", "e"):
            return node.attr
        return f"{value_str}.{node.attr}"

    def _convert_subscript(self, node: ast.Subscript) -> str:
        value_str = self.convert_expr(node.value)
        slice_str = self.convert_expr(node.slice)
        return f"{value_str}[{slice_str}]"

    def _convert_joined_str(self, node: ast.JoinedStr) -> str:
        """Reconstructs Python f-strings `f"Hello {name}"`."""
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                expr_str = self.convert_expr(part.value)
                parts.append(f"{{{expr_str}}}")
        joined = "".join(parts)
        return f'f"{joined}"'


# =============================================================================
# Convenience Functions
# =============================================================================

def migrate_code(python_code: str) -> str:
    """Convenience helper to migrate a Python source code string into Synapse."""
    return PythonToSynapseMigrator().migrate_code(python_code)


def migrate_file(input_path: str, output_path: Optional[str] = None) -> str:
    """Convenience helper to migrate a Python source file into Synapse."""
    return PythonToSynapseMigrator().migrate_file(input_path, output_path)
