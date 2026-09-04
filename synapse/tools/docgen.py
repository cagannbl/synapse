"""
Synapse Documentation Generator (synapse-docgen)
Phase 5: Automated Documentation Engine & API Reference Builder.

Extracts AST representations, type annotations, static/symbolic tensor contracts,
structs, enums, AI-native prompts and agents from Synapse (.syn) source files.
Produces clean GitHub Flavored Markdown and modern, responsive HTML documentation,
with comprehensive docstring and type contract coverage auditing.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from synapse.lexer.lexer import Lexer
from synapse.parser.ast_nodes import (
    AgentDef,
    BinaryExpr,
    CallExpr,
    DictLiteralExpr,
    EnumDeclStmt,
    Expr,
    ExprStmt,
    FunctionDef,
    GenericType,
    IdentifierExpr,
    IndexExpr,
    ListLiteralExpr,
    LiteralExpr,
    MemberExpr,
    Param,
    PipeExpr,
    Program,
    PromptDef,
    Stmt,
    StructDef,
    StructField,
    TensorLiteralExpr,
    TensorType,
    ToolDef,
    TypeAnnotation,
    UnaryExpr,
    UnionType,
    VarDeclStmt,
)
from synapse.parser.parser import Parser


# =============================================================================
# AST Documentation Data Structures
# =============================================================================

@dataclass
class ParamDoc:
    """Documented function or prompt parameter."""
    name: str
    type_annot: Optional[str] = None
    default_value: Optional[str] = None
    tensor_contract: Optional[str] = None
    is_tensor: bool = False
    tensor_dims: tuple[Union[int, str], ...] = ()
    tensor_dtype: str = "float32"
    doc: str = ""

    def formatted_signature(self) -> str:
        s = self.name
        if self.type_annot:
            s += f": {self.type_annot}"
        if self.default_value is not None:
            s += f" = {self.default_value}"
        return s


@dataclass
class FunctionDoc:
    """Documented function or tool definition."""
    name: str
    signature: str
    params: list[ParamDoc] = field(default_factory=list)
    return_type: Optional[str] = None
    return_tensor_contract: Optional[str] = None
    is_tensor_return: bool = False
    docstring: str = ""
    line: int = 1
    is_tool: bool = False
    has_docstring: bool = False
    has_type_contract: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "params": [asdict(p) for p in self.params],
            "return_type": self.return_type,
            "return_tensor_contract": self.return_tensor_contract,
            "is_tensor_return": self.is_tensor_return,
            "docstring": self.docstring,
            "line": self.line,
            "is_tool": self.is_tool,
            "has_docstring": self.has_docstring,
            "has_type_contract": self.has_type_contract,
        }


@dataclass
class FieldDoc:
    """Documented field in a struct."""
    name: str
    type_annot: Optional[str] = None
    default_value: Optional[str] = None
    tensor_contract: Optional[str] = None
    docstring: str = ""
    line: int = 1


@dataclass
class StructDoc:
    """Documented struct definition."""
    name: str
    fields: list[FieldDoc] = field(default_factory=list)
    docstring: str = ""
    line: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fields": [asdict(f) for f in self.fields],
            "docstring": self.docstring,
            "line": self.line,
        }


@dataclass
class EnumDoc:
    """Documented enum declaration."""
    name: str
    variants: list[str] = field(default_factory=list)
    docstring: str = ""
    line: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variants": self.variants,
            "docstring": self.docstring,
            "line": self.line,
        }


@dataclass
class PromptDoc:
    """Documented AI-native prompt definition."""
    name: str
    params: list[ParamDoc] = field(default_factory=list)
    return_type: Optional[str] = None
    fields: dict[str, str] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[str] = None
    docstring: str = ""
    line: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": [asdict(p) for p in self.params],
            "return_type": self.return_type,
            "fields": self.fields,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "model": self.model,
            "temperature": self.temperature,
            "docstring": self.docstring,
            "line": self.line,
        }


@dataclass
class AgentDoc:
    """Documented AI-native autonomous agent definition."""
    name: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    docstring: str = ""
    line: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "instructions": self.instructions,
            "tools": self.tools,
            "fields": self.fields,
            "docstring": self.docstring,
            "line": self.line,
        }


@dataclass
class ConstantDoc:
    """Documented top-level variable or constant declaration."""
    name: str
    type_annot: Optional[str] = None
    value_preview: str = ""
    is_const: bool = False
    tensor_contract: Optional[str] = None
    docstring: str = ""
    line: int = 1


@dataclass
class ModuleDoc:
    """Documented Synapse source module."""
    filepath: str
    module_name: str
    module_docstring: str = ""
    functions: list[FunctionDoc] = field(default_factory=list)
    structs: list[StructDoc] = field(default_factory=list)
    enums: list[EnumDoc] = field(default_factory=list)
    prompts: list[PromptDoc] = field(default_factory=list)
    agents: list[AgentDoc] = field(default_factory=list)
    constants: list[ConstantDoc] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepath": self.filepath,
            "module_name": self.module_name,
            "module_docstring": self.module_docstring,
            "functions": [f.to_dict() for f in self.functions],
            "structs": [s.to_dict() for s in self.structs],
            "enums": [e.to_dict() for e in self.enums],
            "prompts": [p.to_dict() for p in self.prompts],
            "agents": [a.to_dict() for a in self.agents],
            "constants": [asdict(c) for c in self.constants],
        }


@dataclass
class DocCoverageReport:
    """Documentation and type contract coverage audit report."""
    total_functions: int = 0
    documented_functions: int = 0
    typed_functions: int = 0
    docstring_coverage: float = 0.0
    type_contract_coverage: float = 0.0
    missing_docstrings: list[str] = field(default_factory=list)
    missing_type_contracts: list[str] = field(default_factory=list)
    total_structs: int = 0
    total_enums: int = 0
    total_prompts: int = 0
    total_agents: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_functions": self.total_functions,
            "documented_functions": self.documented_functions,
            "typed_functions": self.typed_functions,
            "docstring_coverage": round(self.docstring_coverage, 2),
            "type_contract_coverage": round(self.type_contract_coverage, 2),
            "missing_docstrings": self.missing_docstrings,
            "missing_type_contracts": self.missing_type_contracts,
            "total_structs": self.total_structs,
            "total_enums": self.total_enums,
            "total_prompts": self.total_prompts,
            "total_agents": self.total_agents,
        }

    def format_summary(self) -> str:
        lines = [
            "============================================================",
            "           Synapse Documentation & Type Audit Report        ",
            "============================================================",
            f" Total Functions:         {self.total_functions}",
            f" Documented Functions:    {self.documented_functions} ({self.docstring_coverage:.1f}%)",
            f" Typed Functions:         {self.typed_functions} ({self.type_contract_coverage:.1f}%)",
            f" Total Structs:           {self.total_structs}",
            f" Total Enums:             {self.total_enums}",
            f" Total Prompts:           {self.total_prompts}",
            f" Total Agents:            {self.total_agents}",
            "------------------------------------------------------------",
        ]
        if self.missing_docstrings:
            lines.append(" Functions Missing Docstrings:")
            for fn in self.missing_docstrings:
                lines.append(f"   * {fn}")
        else:
            lines.append(" All functions have docstrings! (100% coverage)")

        if self.missing_type_contracts:
            lines.append(" Functions Missing Type Contracts:")
            for fn in self.missing_type_contracts:
                lines.append(f"   * {fn}")
        else:
            lines.append(" All functions have full type contracts! (100% typed)")

        lines.append("============================================================")
        return "\n".join(lines)


# =============================================================================
# Helper Utilities
# =============================================================================

def _clean_docstring(text: str) -> str:
    """Normalizes and dedents docstrings."""
    if not text:
        return ""
    lines = text.expandtabs(4).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    margin = sys.maxsize
    for line in lines[1:]:
        content = line.lstrip()
        if content:
            indent = len(line) - len(content)
            margin = min(margin, indent)
    if margin == sys.maxsize:
        margin = 0
    res = [lines[0].strip()]
    for line in lines[1:]:
        res.append(line[margin:].rstrip() if len(line) >= margin else line.strip())
    return "\n".join(res).strip()


def _format_expr(expr: Any) -> str:
    """Formats an AST expression into a concise string representation."""
    if expr is None:
        return ""
    if isinstance(expr, LiteralExpr):
        if isinstance(expr.value, str):
            val_escaped = expr.value.replace('"', '\\"')
            return f'"{val_escaped}"'
        return str(expr.value)
    if isinstance(expr, IdentifierExpr):
        return expr.name
    if isinstance(expr, ListLiteralExpr):
        elems = ", ".join(_format_expr(e) for e in expr.elements)
        return f"[{elems}]"
    if hasattr(expr, "name"):
        return getattr(expr, "name")
    return str(expr)


# =============================================================================
# SynapseDocGenerator Core Engine
# =============================================================================

class SynapseDocGenerator:
    """
    Automated Documentation Engine for Synapse programming language.
    Traverses AST trees, extracts signatures, tensor contracts, docstrings,
    and produces high-fidelity Markdown and modern HTML documentation.
    """

    EXCLUDED_DIRS = {
        ".git",
        ".syn_cache",
        "__pycache__",
        "venv",
        ".venv",
        "env",
        "node_modules",
        ".pytest_cache",
        ".vscode",
        "dist",
        "build",
        "target",
        ".idea",
    }

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        dir_path: Optional[Union[str, Path]] = None,
        source: Optional[str] = None,
        recursive: bool = True,
        title: Optional[str] = None,
    ):
        self.file_path = Path(file_path) if file_path else None
        self.dir_path = Path(dir_path) if dir_path else None
        self.raw_source = source
        self.recursive = recursive
        self.title = title
        self.modules: list[ModuleDoc] = []
        self._parsed = False

    def parse(self) -> list[ModuleDoc]:
        """Parses all configured inputs and builds internal documentation models."""
        if self._parsed:
            return self.modules

        self.modules = []
        if self.raw_source is not None:
            mod = self._parse_source_content(
                self.raw_source,
                filepath=str(self.file_path or "unnamed.syn"),
                module_name=self.file_path.stem if self.file_path else "main",
            )
            self.modules.append(mod)
        elif self.file_path and self.file_path.is_file():
            mod = self._parse_file(self.file_path)
            self.modules.append(mod)
        elif self.dir_path and self.dir_path.is_dir():
            syn_files = self._discover_syn_files(self.dir_path, self.recursive)
            for f in syn_files:
                try:
                    mod = self._parse_file(f, base_dir=self.dir_path)
                    self.modules.append(mod)
                except Exception:
                    pass
        elif self.file_path and not self.file_path.exists():
            raise FileNotFoundError(f"Synapse file not found: {self.file_path}")
        elif self.dir_path and not self.dir_path.exists():
            raise FileNotFoundError(f"Synapse directory not found: {self.dir_path}")

        self._parsed = True
        return self.modules

    def _discover_syn_files(self, root_dir: Path, recursive: bool) -> list[Path]:
        """Discovers all .syn files in the specified directory, respecting excludes."""
        found: list[Path] = []
        if recursive:
            for root, dirs, files in os.walk(root_dir):
                dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS and not d.startswith(".")]
                for file in files:
                    if file.endswith(".syn"):
                        found.append(Path(root) / file)
        else:
            for item in root_dir.iterdir():
                if item.is_file() and item.suffix == ".syn":
                    found.append(item)
        found.sort()
        return found

    def _parse_file(self, path: Path, base_dir: Optional[Path] = None) -> ModuleDoc:
        """Parses a single .syn file from disk."""
        content = path.read_text(encoding="utf-8")
        if base_dir:
            try:
                rel_path = path.relative_to(base_dir)
                mod_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
            except ValueError:
                mod_name = path.stem
        else:
            mod_name = path.stem

        return self._parse_source_content(content, filepath=str(path), module_name=mod_name)

    def _extract_comments_above(self, lines: list[str], target_line: int) -> str:
        """
        Extracts contiguous block of comment lines immediately preceding target_line (1-indexed).
        Handles '#' and '//' comment styles.
        """
        if target_line <= 1 or target_line > len(lines) + 1:
            return ""

        collected: list[str] = []
        curr_idx = target_line - 2  # 0-indexed line right above target_line

        while curr_idx >= 0:
            line_str = lines[curr_idx].strip()
            if not line_str:
                break
            if line_str.startswith("#"):
                comment_text = line_str.lstrip("#").strip()
                if not set(comment_text).issubset({"=", "-", "*", "_", " "}):
                    collected.append(comment_text)
            elif line_str.startswith("//"):
                comment_text = line_str.lstrip("/").strip()
                if not set(comment_text).issubset({"=", "-", "*", "_", " "}):
                    collected.append(comment_text)
            else:
                break
            curr_idx -= 1

        collected.reverse()
        return "\n".join(collected).strip()

    def _extract_module_docstring(self, lines: list[str], ast_prog: Program) -> str:
        """Extracts module-level docstring or initial comment header."""
        if ast_prog.statements:
            first = ast_prog.statements[0]
            if isinstance(first, ExprStmt) and isinstance(first.expr, LiteralExpr):
                if isinstance(first.expr.value, str):
                    return _clean_docstring(first.expr.value)

        header_comments: list[str] = []
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                if header_comments:
                    break
                continue
            if trimmed.startswith("#") or trimmed.startswith("//"):
                c = trimmed.lstrip("#/").strip()
                if c and not set(c).issubset({"=", "-", "*", "_", " "}):
                    header_comments.append(c)
            else:
                break
        return "\n".join(header_comments).strip()

    def _extract_function_docstring(self, fn: Union[FunctionDef, ToolDef], source_lines: list[str]) -> str:
        """Extracts function docstring from AST body or preceding comments."""
        if fn.body:
            first_stmt = fn.body[0]
            if isinstance(first_stmt, ExprStmt) and isinstance(first_stmt.expr, LiteralExpr):
                if isinstance(first_stmt.expr.value, str):
                    return _clean_docstring(first_stmt.expr.value)

        fn_line = getattr(fn, "line", 1)
        fallback = self._extract_comments_above(source_lines, fn_line)
        return _clean_docstring(fallback)

    def _parse_param(self, p: Param) -> ParamDoc:
        """Extracts parameter documentation and tensor contract information."""
        type_str = str(p.type_annot) if p.type_annot is not None else None
        default_str = _format_expr(p.default_value) if p.default_value is not None else None

        tensor_contract = None
        is_tensor = False
        tensor_dims: tuple[Union[int, str], ...] = ()
        tensor_dtype = "float32"

        t_type = getattr(p, "tensor_type", None)
        if t_type is None and isinstance(p.type_annot, TypeAnnotation):
            t_type = p.type_annot.tensor_type
        if t_type is None and isinstance(p.type_annot, GenericType) and p.type_annot.base == "Tensor":
            t_type = p.type_annot.tensor_type
        if t_type is None and type_str:
            t_type = TensorType.from_string(type_str)

        if t_type:
            is_tensor = True
            tensor_dims = t_type.dims
            tensor_dtype = getattr(t_type, "dtype", "float32")
            tensor_contract = str(t_type)
        elif type_str and "Tensor" in type_str:
            is_tensor = True
            tensor_contract = type_str

        return ParamDoc(
            name=p.name,
            type_annot=type_str,
            default_value=default_str,
            tensor_contract=tensor_contract,
            is_tensor=is_tensor,
            tensor_dims=tensor_dims,
            tensor_dtype=tensor_dtype,
        )

    def _parse_source_content(self, source: str, filepath: str, module_name: str) -> ModuleDoc:
        """Parses Synapse source code string and populates ModuleDoc."""
        source_lines = source.splitlines()
        try:
            tokens = Lexer(source).tokenize()
            ast_prog = Parser(tokens).parse()
        except Exception as ex:
            return ModuleDoc(
                filepath=filepath,
                module_name=module_name,
                module_docstring=f"Parse Notice: Could not build full AST ({ex})",
            )

        module_docstring = self._extract_module_docstring(source_lines, ast_prog)

        functions: list[FunctionDoc] = []
        structs: list[StructDoc] = []
        enums: list[EnumDoc] = []
        prompts: list[PromptDoc] = []
        agents: list[AgentDoc] = []
        constants: list[ConstantDoc] = []

        for stmt in ast_prog.statements:
            if isinstance(stmt, (FunctionDef, ToolDef)):
                is_tool = isinstance(stmt, ToolDef)
                params_doc = [self._parse_param(p) for p in stmt.params]
                ret_type_str = str(stmt.return_type) if stmt.return_type is not None else None

                ret_tensor_contract = None
                is_ret_tensor = False
                ret_t_type = getattr(stmt, "return_tensor_type", None)
                if ret_t_type is None and isinstance(stmt.return_type, TypeAnnotation):
                    ret_t_type = stmt.return_type.tensor_type
                if ret_t_type is None and isinstance(stmt.return_type, GenericType) and stmt.return_type.base == "Tensor":
                    ret_t_type = stmt.return_type.tensor_type
                if ret_t_type is None and ret_type_str:
                    ret_t_type = TensorType.from_string(ret_type_str)

                if ret_t_type:
                    is_ret_tensor = True
                    ret_tensor_contract = str(ret_t_type)
                elif ret_type_str and "Tensor" in ret_type_str:
                    is_ret_tensor = True
                    ret_tensor_contract = ret_type_str

                docstring = self._extract_function_docstring(stmt, source_lines)
                has_doc = bool(docstring.strip())
                has_contract = all(p.type_annot is not None for p in stmt.params) and (stmt.return_type is not None)

                kw = "tool" if is_tool else "fn"
                param_strs = [p.formatted_signature() for p in params_doc]
                sig = f"{kw} {stmt.name}({', '.join(param_strs)})"
                if ret_type_str:
                    sig += f" -> {ret_type_str}"

                functions.append(FunctionDoc(
                    name=stmt.name,
                    signature=sig,
                    params=params_doc,
                    return_type=ret_type_str,
                    return_tensor_contract=ret_tensor_contract,
                    is_tensor_return=is_ret_tensor,
                    docstring=docstring,
                    line=getattr(stmt, "line", 1),
                    is_tool=is_tool,
                    has_docstring=has_doc,
                    has_type_contract=has_contract,
                ))

            elif isinstance(stmt, StructDef):
                field_docs: list[FieldDoc] = []
                for f in stmt.fields:
                    f_type_str = str(f.type_annot) if f.type_annot is not None else None
                    f_default = _format_expr(f.default_value) if f.default_value is not None else None
                    f_contract = None
                    if getattr(f, "tensor_type", None):
                        f_contract = str(f.tensor_type)
                    elif f_type_str and "Tensor" in f_type_str:
                        f_contract = f_type_str

                    field_docs.append(FieldDoc(
                        name=f.name,
                        type_annot=f_type_str,
                        default_value=f_default,
                        tensor_contract=f_contract,
                        docstring=f.docstring or "",
                        line=getattr(f, "line", 1),
                    ))

                s_doc = stmt.docstring or self._extract_comments_above(source_lines, getattr(stmt, "line", 1))
                structs.append(StructDoc(
                    name=stmt.name,
                    fields=field_docs,
                    docstring=_clean_docstring(s_doc),
                    line=getattr(stmt, "line", 1),
                ))

            elif isinstance(stmt, EnumDeclStmt):
                e_doc = self._extract_comments_above(source_lines, getattr(stmt, "line", 1))
                enums.append(EnumDoc(
                    name=stmt.name,
                    variants=list(stmt.variants),
                    docstring=_clean_docstring(e_doc),
                    line=getattr(stmt, "line", 1),
                ))

            elif isinstance(stmt, PromptDef):
                params_doc = [self._parse_param(p) for p in stmt.params]
                field_dict: dict[str, str] = {}
                for k, v in stmt.fields.items():
                    field_dict[k] = _format_expr(v)

                system_p = field_dict.get("system")
                user_p = field_dict.get("user")
                model_p = field_dict.get("model")
                temp_p = field_dict.get("temperature")

                p_doc = self._extract_comments_above(source_lines, getattr(stmt, "line", 1))
                prompts.append(PromptDoc(
                    name=stmt.name,
                    params=params_doc,
                    return_type=stmt.return_type,
                    fields=field_dict,
                    system_prompt=system_p,
                    user_template=user_p,
                    model=model_p,
                    temperature=temp_p,
                    docstring=_clean_docstring(p_doc),
                    line=getattr(stmt, "line", 1),
                ))

            elif isinstance(stmt, AgentDef):
                field_dict: dict[str, str] = {}
                tools_list: list[str] = []
                for k, v in stmt.fields.items():
                    if k == "tools" and isinstance(v, ListLiteralExpr):
                        tools_list = [_format_expr(elem) for elem in v.elements]
                    field_dict[k] = _format_expr(v)

                model_val = field_dict.get("model")
                inst_val = field_dict.get("instructions")
                a_doc = self._extract_comments_above(source_lines, getattr(stmt, "line", 1))

                agents.append(AgentDoc(
                    name=stmt.name,
                    model=model_val,
                    instructions=inst_val,
                    tools=tools_list,
                    fields=field_dict,
                    docstring=_clean_docstring(a_doc),
                    line=getattr(stmt, "line", 1),
                ))

            elif isinstance(stmt, VarDeclStmt):
                val_preview = _format_expr(stmt.value)
                type_str = str(stmt.type_annot) if stmt.type_annot is not None else None
                tensor_contract = str(stmt.tensor_type) if stmt.tensor_type else (type_str if type_str and "Tensor" in type_str else None)
                v_doc = self._extract_comments_above(source_lines, getattr(stmt, "line", 1))

                constants.append(ConstantDoc(
                    name=stmt.name,
                    type_annot=type_str,
                    value_preview=val_preview,
                    is_const=stmt.is_const,
                    tensor_contract=tensor_contract,
                    docstring=_clean_docstring(v_doc),
                    line=getattr(stmt, "line", 1),
                ))

        return ModuleDoc(
            filepath=filepath,
            module_name=module_name,
            module_docstring=module_docstring,
            functions=functions,
            structs=structs,
            enums=enums,
            prompts=prompts,
            agents=agents,
            constants=constants,
        )

    # =========================================================================
    # Coverage Auditing
    # =========================================================================

    def get_coverage_report(self) -> DocCoverageReport:
        """
        Audits documentation and static type contract coverage across all parsed modules.
        Returns a structured DocCoverageReport.
        """
        self.parse()
        report = DocCoverageReport()

        for mod in self.modules:
            report.total_structs += len(mod.structs)
            report.total_enums += len(mod.enums)
            report.total_prompts += len(mod.prompts)
            report.total_agents += len(mod.agents)

            for fn in mod.functions:
                report.total_functions += 1
                fn_qualname = f"{mod.module_name}.{fn.name}" if len(self.modules) > 1 else fn.name

                if fn.has_docstring:
                    report.documented_functions += 1
                else:
                    report.missing_docstrings.append(fn_qualname)

                if fn.has_type_contract:
                    report.typed_functions += 1
                else:
                    report.missing_type_contracts.append(fn_qualname)

        if report.total_functions > 0:
            report.docstring_coverage = (report.documented_functions / report.total_functions) * 100.0
            report.type_contract_coverage = (report.typed_functions / report.total_functions) * 100.0
        else:
            report.docstring_coverage = 100.0
            report.type_contract_coverage = 100.0

        return report

    # =========================================================================
    # Markdown Generator (GitHub Flavored Markdown)
    # =========================================================================

    def generate_markdown(self) -> str:
        """
        Generates clean, modern GitHub Flavored Markdown (GFM) API reference.
        Includes table of contents, tensor shape contracts, structs, enums,
        and AI-native primitives.
        """
        self.parse()
        lines: list[str] = []

        main_title = self.title or (
            f"Synapse API Reference: `{self.modules[0].module_name}`"
            if len(self.modules) == 1
            else "Synapse Project API Reference"
        )
        lines.append(f"# {main_title}")
        lines.append("")
        lines.append("> Auto-generated by Synapse Documentation Engine (`synapse-docgen`).")
        lines.append("")

        cov = self.get_coverage_report()
        badges = [
            f"**Modules:** {len(self.modules)}",
            f"**Functions:** {cov.total_functions}",
            f"**Structs:** {cov.total_structs}",
            f"**Enums:** {cov.total_enums}",
            f"**AI Primitives:** {cov.total_prompts + cov.total_agents}",
            f"**Doc Coverage:** {cov.docstring_coverage:.1f}%",
            f"**Type Contract:** {cov.type_contract_coverage:.1f}%",
        ]
        lines.append(" | ".join(badges))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        lines.append("")
        for mod in self.modules:
            mod_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", mod.module_name.lower().replace(".", "-"))
            if len(self.modules) > 1:
                lines.append(f"- [Module: `{mod.module_name}`](#module-{mod_anchor})")
            if mod.structs:
                lines.append(f"  - [Structs](#structs-{mod_anchor})")
            if mod.enums:
                lines.append(f"  - [Enums](#enums-{mod_anchor})")
            if mod.functions:
                lines.append(f"  - [Functions](#functions-{mod_anchor})")
                for fn in mod.functions:
                    fn_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", f"{mod_anchor}-{fn.name}".lower())
                    lines.append(f"    - [`fn {fn.name}`](#fn-{fn_anchor})")
            if mod.prompts or mod.agents:
                lines.append(f"  - [AI Primitives](#ai-primitives-{mod_anchor})")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Render Modules
        for mod in self.modules:
            mod_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", mod.module_name.lower().replace(".", "-"))
            if len(self.modules) > 1:
                lines.append(f"## Module: `{mod.module_name}` <a id=\"module-{mod_anchor}\"></a>")
                lines.append(f"- **File:** `{mod.filepath}`")
                lines.append("")
            else:
                lines.append("## Module Overview")
                lines.append(f"- **Source File:** `{mod.filepath}`")
                lines.append("")

            if mod.module_docstring:
                lines.append(f"> {mod.module_docstring.replace(chr(10), chr(10) + '> ')}")
                lines.append("")

            # 1. Structs
            if mod.structs:
                lines.append(f"### Structs <a id=\"structs-{mod_anchor}\"></a>")
                lines.append("")
                for st in mod.structs:
                    st_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", f"{mod_anchor}-{st.name}".lower())
                    lines.append(f"#### `struct {st.name}` <a id=\"struct-{st_anchor}\"></a>")
                    lines.append(f"- **Defined at line:** `{st.line}`")
                    lines.append("")
                    if st.docstring:
                        lines.append(st.docstring)
                        lines.append("")

                    if st.fields:
                        lines.append("| Field | Type | Contract | Default | Description |")
                        lines.append("| :--- | :--- | :--- | :--- | :--- |")
                        for f in st.fields:
                            type_val = f"`{f.type_annot}`" if f.type_annot else "*None*"
                            contract_val = f"`{f.tensor_contract}`" if f.tensor_contract else "-"
                            def_val = f"`{f.default_value}`" if f.default_value is not None else "-"
                            doc_val = f.docstring or "-"
                            lines.append(f"| **`{f.name}`** | {type_val} | {contract_val} | {def_val} | {doc_val} |")
                        lines.append("")

            # 2. Enums
            if mod.enums:
                lines.append(f"### Enums <a id=\"enums-{mod_anchor}\"></a>")
                lines.append("")
                for en in mod.enums:
                    en_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", f"{mod_anchor}-{en.name}".lower())
                    lines.append(f"#### `enum {en.name}` <a id=\"enum-{en_anchor}\"></a>")
                    lines.append(f"- **Defined at line:** `{en.line}`")
                    lines.append("")
                    if en.docstring:
                        lines.append(en.docstring)
                        lines.append("")
                    lines.append("**Variants:**")
                    for v in en.variants:
                        lines.append(f"- `{v}`")
                    lines.append("")

            # 3. Functions
            if mod.functions:
                lines.append(f"### Functions <a id=\"functions-{mod_anchor}\"></a>")
                lines.append("")
                for fn in mod.functions:
                    fn_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", f"{mod_anchor}-{fn.name}".lower())
                    lines.append(f"#### `fn {fn.name}` <a id=\"fn-{fn_anchor}\"></a>")
                    lines.append("")
                    lines.append("```synapse")
                    lines.append(fn.signature)
                    lines.append("```")
                    lines.append("")

                    tensor_params = [p for p in fn.params if p.is_tensor]
                    if tensor_params or fn.is_tensor_return:
                        contracts = [f"`{p.name}: {p.tensor_contract}`" for p in tensor_params]
                        ret_contract = f"`{fn.return_tensor_contract}`" if fn.return_tensor_contract else "`void`"
                        lines.append(f"> **Tensor Shape Contract:** {', '.join(contracts)} ➔ {ret_contract}")
                        lines.append("")

                    if fn.docstring:
                        lines.append(fn.docstring)
                        lines.append("")

                    lines.append(f"- **Line:** `{fn.line}`")
                    if fn.params:
                        lines.append("")
                        lines.append("| Parameter | Type | Contract | Default |")
                        lines.append("| :--- | :--- | :--- | :--- |")
                        for p in fn.params:
                            t_val = f"`{p.type_annot}`" if p.type_annot else "*Any*"
                            c_val = f"`{p.tensor_contract}`" if p.tensor_contract else "-"
                            d_val = f"`{p.default_value}`" if p.default_value is not None else "-"
                            lines.append(f"| **`{p.name}`** | {t_val} | {c_val} | {d_val} |")

                    if fn.return_type:
                        lines.append("")
                        ret_str = f"`{fn.return_type}`"
                        if fn.return_tensor_contract and fn.return_tensor_contract != fn.return_type:
                            ret_str += f" (Contract: `{fn.return_tensor_contract}`)"
                        lines.append(f"- **Returns:** {ret_str}")

                    lines.append("")
                    lines.append("---")
                    lines.append("")

            # 4. AI Primitives
            if mod.prompts or mod.agents:
                lines.append(f"### AI Primitives <a id=\"ai-primitives-{mod_anchor}\"></a>")
                lines.append("")

                for pr in mod.prompts:
                    pr_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", f"{mod_anchor}-{pr.name}".lower())
                    lines.append(f"#### `prompt {pr.name}` <a id=\"prompt-{pr_anchor}\"></a>")
                    lines.append("")
                    lines.append("```synapse")
                    param_strs = [p.formatted_signature() for p in pr.params]
                    lines.append(f"prompt {pr.name}({', '.join(param_strs)}):")
                    for k, v in pr.fields.items():
                        lines.append(f"    {k}: {v}")
                    lines.append("```")
                    lines.append("")
                    if pr.docstring:
                        lines.append(pr.docstring)
                        lines.append("")
                    if pr.model:
                        lines.append(f"- **Model:** `{pr.model}`")
                    if pr.temperature:
                        lines.append(f"- **Temperature:** `{pr.temperature}`")
                    if pr.system_prompt:
                        lines.append(f"- **System Prompt:** *{pr.system_prompt}*")
                    lines.append("")

                for ag in mod.agents:
                    ag_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", f"{mod_anchor}-{ag.name}".lower())
                    lines.append(f"#### `agent {ag.name}` <a id=\"agent-{ag_anchor}\"></a>")
                    lines.append("")
                    lines.append("```synapse")
                    lines.append(f"agent {ag.name}:")
                    for k, v in ag.fields.items():
                        lines.append(f"    {k}: {v}")
                    lines.append("```")
                    lines.append("")
                    if ag.docstring:
                        lines.append(ag.docstring)
                        lines.append("")
                    if ag.model:
                        lines.append(f"- **Model:** `{ag.model}`")
                    if ag.instructions:
                        lines.append(f"- **Instructions:** *{ag.instructions}*")
                    if ag.tools:
                        lines.append(f"- **Tools:** {', '.join(f'`{t}`' for t in ag.tools)}")
                    lines.append("")

            # 5. Constants
            if mod.constants:
                lines.append(f"### Constants & Variables")
                lines.append("")
                lines.append("| Name | Kind | Type | Value | Contract |")
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for c in mod.constants:
                    k_val = "const" if c.is_const else "let"
                    t_val = f"`{c.type_annot}`" if c.type_annot else "*Inferred*"
                    v_val = f"`{c.value_preview}`" if c.value_preview else "-"
                    cnt_val = f"`{c.tensor_contract}`" if c.tensor_contract else "-"
                    lines.append(f"| **`{c.name}`** | `{k_val}` | {t_val} | {v_val} | {cnt_val} |")
                lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Modern HTML Generator
    # =========================================================================

    def generate_html(self) -> str:
        """
        Generates modern, responsive, dark-mode single-file HTML API documentation.
        Follows the Linear / Vercel design system: deep zinc backgrounds, 1px subtle borders,
        refined typography, pill badges, interactive client-side search, zero neon.
        """
        self.parse()
        cov = self.get_coverage_report()
        doc_title = self.title or (
            f"Synapse API Reference - {self.modules[0].module_name}"
            if len(self.modules) == 1
            else "Synapse Project API Reference"
        )

        sidebar_items: list[tuple[str, str, str, str]] = []
        for mod in self.modules:
            mod_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", mod.module_name.lower().replace(".", "-"))
            for st in mod.structs:
                sidebar_items.append(("struct", st.name, f"struct-{mod_anchor}-{st.name.lower()}", mod.module_name))
            for en in mod.enums:
                sidebar_items.append(("enum", en.name, f"enum-{mod_anchor}-{en.name.lower()}", mod.module_name))
            for fn in mod.functions:
                sidebar_items.append(("fn", fn.name, f"fn-{mod_anchor}-{fn.name.lower()}", mod.module_name))
            for pr in mod.prompts:
                sidebar_items.append(("prompt", pr.name, f"prompt-{mod_anchor}-{pr.name.lower()}", mod.module_name))
            for ag in mod.agents:
                sidebar_items.append(("agent", ag.name, f"agent-{mod_anchor}-{ag.name.lower()}", mod.module_name))

        html_out = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(doc_title)}</title>
  <style>
    :root {{
      --bg-base: #09090b;
      --bg-subtle: #121214;
      --bg-card: #18181b;
      --border: #27272a;
      --border-focus: #3f3f46;
      --text-main: #f4f4f5;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --accent: #e4e4e7;
      --emerald-bg: rgba(6, 78, 59, 0.35);
      --emerald-border: rgba(16, 185, 129, 0.3);
      --emerald-text: #6ee7b7;
      --purple-bg: rgba(88, 28, 135, 0.35);
      --purple-border: rgba(168, 85, 247, 0.3);
      --purple-text: #d8b4fe;
      --blue-bg: rgba(30, 58, 138, 0.35);
      --blue-border: rgba(59, 130, 246, 0.3);
      --blue-text: #93c5fd;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #sidebar {{
      width: 320px;
      background-color: var(--bg-subtle);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      height: 100%;
    }}
    .sidebar-header {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--border);
    }}
    .sidebar-brand {{
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .sidebar-brand span {{
      color: var(--text-dim);
      font-size: 0.85rem;
      font-weight: normal;
    }}
    .search-box {{
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
    }}
    .search-input {{
      width: 100%;
      background-color: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 12px;
      color: var(--text-main);
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.15s;
    }}
    .search-input:focus {{
      border-color: var(--border-focus);
    }}
    .nav-list {{
      list-style: none;
      overflow-y: auto;
      flex: 1;
      padding: 16px 12px;
    }}
    .nav-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.85rem;
      border-radius: 6px;
      transition: background-color 0.15s, color 0.15s;
    }}
    .nav-item:hover {{
      background-color: var(--bg-card);
      color: var(--text-main);
    }}
    .badge-tag {{
      display: inline-block;
      font-size: 0.68rem;
      font-family: monospace;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 500;
      line-height: 1.2;
    }}
    .tag-fn {{ background: var(--blue-bg); border: 1px solid var(--blue-border); color: var(--blue-text); }}
    .tag-struct {{ background: var(--emerald-bg); border: 1px solid var(--emerald-border); color: var(--emerald-text); }}
    .tag-enum {{ background: var(--bg-card); border: 1px solid var(--border); color: var(--text-muted); }}
    .tag-ai {{ background: var(--purple-bg); border: 1px solid var(--purple-border); color: var(--purple-text); }}

    #content {{
      flex: 1;
      overflow-y: auto;
      padding: 40px 64px;
      max-width: 1100px;
    }}
    .page-header {{
      margin-bottom: 32px;
    }}
    .page-title {{
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin-bottom: 8px;
    }}
    .page-subtitle {{
      color: var(--text-muted);
      font-size: 0.95rem;
    }}
    .metrics-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
      margin: 24px 0 40px 0;
    }}
    .metric-card {{
      background-color: var(--bg-subtle);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .metric-value {{
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text-main);
    }}
    .metric-label {{
      font-size: 0.75rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .section-title {{
      font-size: 1.35rem;
      font-weight: 600;
      margin: 40px 0 20px 0;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      color: var(--text-main);
    }}
    .symbol-card {{
      background-color: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      transition: border-color 0.15s;
    }}
    .symbol-card:hover {{
      border-color: var(--border-focus);
    }}
    .symbol-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .symbol-name {{
      font-size: 1.15rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .code-box {{
      background-color: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.88rem;
      overflow-x: auto;
      margin: 12px 0;
      color: #e4e4e7;
    }}
    .tensor-banner {{
      background: var(--emerald-bg);
      border: 1px solid var(--emerald-border);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.85rem;
      color: var(--emerald-text);
      margin: 12px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .doc-text {{
      color: var(--text-muted);
      font-size: 0.92rem;
      margin: 12px 0;
      white-space: pre-line;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-top: 14px;
    }}
    table.data-table th {{
      text-align: left;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      color: var(--text-dim);
      font-weight: 600;
    }}
    table.data-table td {{
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      color: var(--text-muted);
    }}
    .mono {{
      font-family: ui-monospace, "SF Mono", monospace;
      color: var(--text-main);
    }}
  </style>
</head>
<body>

  <aside id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-brand">
        Synapse API <span>v2.1</span>
      </div>
    </div>
    <div class="search-box">
      <input type="text" id="symbol-filter" class="search-input" placeholder="Search functions, structs, AI...">
    </div>
    <ul class="nav-list" id="nav-list">
"""

        for kind, name, anchor, mod_name in sidebar_items:
            tag_class = f"tag-{kind}" if kind in ("fn", "struct", "enum") else "tag-ai"
            html_out += f"""      <li>
        <a href="#{anchor}" class="nav-item" data-name="{name.lower()}">
          <span class="badge-tag {tag_class}">{kind}</span>
          <span>{html.escape(name)}</span>
        </a>
      </li>\n"""

        html_out += f"""    </ul>
  </aside>

  <main id="content">
    <header class="page-header">
      <h1 class="page-title">{html.escape(doc_title)}</h1>
      <p class="page-subtitle">Auto-generated API specification from Synapse Abstract Syntax Trees (AST).</p>

      <div class="metrics-bar">
        <div class="metric-card">
          <div class="metric-value">{len(self.modules)}</div>
          <div class="metric-label">Modules</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{cov.total_functions}</div>
          <div class="metric-label">Functions</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{cov.total_structs}</div>
          <div class="metric-label">Structs</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{cov.total_prompts + cov.total_agents}</div>
          <div class="metric-label">AI Primitives</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{cov.docstring_coverage:.1f}%</div>
          <div class="metric-label">Doc Coverage</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{cov.type_contract_coverage:.1f}%</div>
          <div class="metric-label">Type Contract</div>
        </div>
      </div>
    </header>
"""

        for mod in self.modules:
            mod_anchor = re.sub(r"[^a-zA-Z0-9_-]", "", mod.module_name.lower().replace(".", "-"))
            html_out += f"""    <section class="module-section">
      <h2 class="section-title">Module: {html.escape(mod.module_name)}</h2>
      <p class="mono" style="font-size:0.8rem; color:var(--text-dim); margin-bottom:16px;">Source: {html.escape(mod.filepath)}</p>
"""
            if mod.module_docstring:
                html_out += f"""      <p class="doc-text">{html.escape(mod.module_docstring)}</p>\n"""

            for st in mod.structs:
                st_id = f"struct-{mod_anchor}-{st.name.lower()}"
                html_out += f"""      <div class="symbol-card" id="{st_id}">
        <div class="symbol-header">
          <div class="symbol-name">
            <span class="badge-tag tag-struct">struct</span>
            {html.escape(st.name)}
          </div>
          <span class="mono" style="font-size:0.75rem; color:var(--text-dim);">Line {st.line}</span>
        </div>
"""
                if st.docstring:
                    html_out += f"""        <p class="doc-text">{html.escape(st.docstring)}</p>\n"""

                if st.fields:
                    html_out += """        <table class="data-table">
          <thead>
            <tr><th>Field</th><th>Type</th><th>Contract</th><th>Default</th></tr>
          </thead>
          <tbody>\n"""
                    for f in st.fields:
                        type_str = f.type_annot or "Any"
                        contract_str = f.tensor_contract or "-"
                        def_str = f.default_value if f.default_value is not None else "-"
                        html_out += f"""            <tr>
              <td class="mono">{html.escape(f.name)}</td>
              <td class="mono">{html.escape(type_str)}</td>
              <td class="mono">{html.escape(contract_str)}</td>
              <td class="mono">{html.escape(def_str)}</td>
            </tr>\n"""
                    html_out += """          </tbody>
        </table>\n"""
                html_out += "      </div>\n"

            for en in mod.enums:
                en_id = f"enum-{mod_anchor}-{en.name.lower()}"
                html_out += f"""      <div class="symbol-card" id="{en_id}">
        <div class="symbol-header">
          <div class="symbol-name">
            <span class="badge-tag tag-enum">enum</span>
            {html.escape(en.name)}
          </div>
          <span class="mono" style="font-size:0.75rem; color:var(--text-dim);">Line {en.line}</span>
        </div>
"""
                if en.docstring:
                    html_out += f"""        <p class="doc-text">{html.escape(en.docstring)}</p>\n"""

                variants_str = ", ".join(en.variants)
                html_out += f"""        <p style="font-size:0.85rem; color:var(--text-muted);"><strong style="color:var(--text-main)">Variants:</strong> <code class="mono">{html.escape(variants_str)}</code></p>
      </div>\n"""

            for fn in mod.functions:
                fn_id = f"fn-{mod_anchor}-{fn.name.lower()}"
                html_out += f"""      <div class="symbol-card" id="{fn_id}">
        <div class="symbol-header">
          <div class="symbol-name">
            <span class="badge-tag tag-fn">{"tool" if fn.is_tool else "fn"}</span>
            {html.escape(fn.name)}
          </div>
          <span class="mono" style="font-size:0.75rem; color:var(--text-dim);">Line {fn.line}</span>
        </div>
        <div class="code-box">{html.escape(fn.signature)}</div>
"""
                tensor_params = [p for p in fn.params if p.is_tensor]
                if tensor_params or fn.is_tensor_return:
                    c_strs = [f"{p.name}: {p.tensor_contract}" for p in tensor_params]
                    ret_c = fn.return_tensor_contract or "void"
                    html_out += f"""        <div class="tensor-banner">
          <span>⚡ <strong>Static Tensor Contract:</strong> {html.escape(', '.join(c_strs))} ➔ {html.escape(ret_c)}</span>
        </div>\n"""

                if fn.docstring:
                    html_out += f"""        <p class="doc-text">{html.escape(fn.docstring)}</p>\n"""

                if fn.params:
                    html_out += """        <table class="data-table">
          <thead>
            <tr><th>Parameter</th><th>Type</th><th>Contract</th><th>Default</th></tr>
          </thead>
          <tbody>\n"""
                    for p in fn.params:
                        t_str = p.type_annot or "Any"
                        c_str = p.tensor_contract or "-"
                        d_str = p.default_value if p.default_value is not None else "-"
                        html_out += f"""            <tr>
              <td class="mono">{html.escape(p.name)}</td>
              <td class="mono">{html.escape(t_str)}</td>
              <td class="mono">{html.escape(c_str)}</td>
              <td class="mono">{html.escape(d_str)}</td>
            </tr>\n"""
                    html_out += """          </tbody>
        </table>\n"""

                if fn.return_type:
                    html_out += f"""        <p style="font-size:0.85rem; margin-top:12px; color:var(--text-muted);">
          <strong style="color:var(--text-main)">Returns:</strong> <code class="mono">{html.escape(fn.return_type)}</code>
        </p>\n"""

                html_out += "      </div>\n"

            for pr in mod.prompts:
                pr_id = f"prompt-{mod_anchor}-{pr.name.lower()}"
                html_out += f"""      <div class="symbol-card" id="{pr_id}">
        <div class="symbol-header">
          <div class="symbol-name">
            <span class="badge-tag tag-ai">prompt</span>
            {html.escape(pr.name)}
          </div>
          <span class="mono" style="font-size:0.75rem; color:var(--text-dim);">Line {pr.line}</span>
        </div>
        <div class="code-box">prompt {html.escape(pr.name)}({html.escape(', '.join(p.formatted_signature() for p in pr.params))})</div>
"""
                if pr.docstring:
                    html_out += f"""        <p class="doc-text">{html.escape(pr.docstring)}</p>\n"""
                if pr.model:
                    html_out += f"""        <p style="font-size:0.85rem; color:var(--text-muted);">Model: <code class="mono">{html.escape(pr.model)}</code></p>\n"""
                if pr.system_prompt:
                    html_out += f"""        <p style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">System: <em>{html.escape(pr.system_prompt)}</em></p>\n"""
                html_out += "      </div>\n"

            for ag in mod.agents:
                ag_id = f"agent-{mod_anchor}-{ag.name.lower()}"
                html_out += f"""      <div class="symbol-card" id="{ag_id}">
        <div class="symbol-header">
          <div class="symbol-name">
            <span class="badge-tag tag-ai">agent</span>
            {html.escape(ag.name)}
          </div>
          <span class="mono" style="font-size:0.75rem; color:var(--text-dim);">Line {ag.line}</span>
        </div>
"""
                if ag.docstring:
                    html_out += f"""        <p class="doc-text">{html.escape(ag.docstring)}</p>\n"""
                if ag.model:
                    html_out += f"""        <p style="font-size:0.85rem; color:var(--text-muted);">Model: <code class="mono">{html.escape(ag.model)}</code></p>\n"""
                if ag.instructions:
                    html_out += f"""        <p style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">Instructions: <em>{html.escape(ag.instructions)}</em></p>\n"""
                if ag.tools:
                    html_out += f"""        <p style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">Tools: <code class="mono">{html.escape(', '.join(ag.tools))}</code></p>\n"""
                html_out += "      </div>\n"

            html_out += "    </section>\n"

        html_out += """  </main>

  <script>
    const filterInput = document.getElementById('symbol-filter');
    const navItems = document.querySelectorAll('#nav-list .nav-item');
    filterInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      navItems.forEach(item => {
        const name = item.getAttribute('data-name') || '';
        item.parentElement.style.display = name.includes(q) ? '' : 'none';
      });
    });
  </script>
</body>
</html>
"""
        return html_out

    # =========================================================================
    # Disk I/O Helpers
    # =========================================================================

    def save_markdown(self, output_path: Union[str, Path]) -> Path:
        """Generates markdown and saves it to specified file path."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_markdown()
        p.write_text(content, encoding="utf-8")
        return p

    def save_html(self, output_path: Union[str, Path]) -> Path:
        """Generates HTML and saves it to specified file path."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_html()
        p.write_text(content, encoding="utf-8")
        return p


# =============================================================================
# CLI Entry Point
# =============================================================================

def main(args_list: Optional[list[str]] = None) -> int:
    """CLI handler for synapse-docgen."""
    parser = argparse.ArgumentParser(
        prog="synapse-docgen",
        description="Synapse Automated Documentation Engine & API Reference Builder",
    )
    parser.add_argument("path", nargs="?", default=".", help="Path to a .syn file or directory")
    parser.add_argument("--dir", action="store_true", help="Treat path as directory and recursively discover .syn files")
    parser.add_argument("-o", "--output", help="Output file path (default: <name>.md or <name>.html)")
    parser.add_argument("--html", action="store_true", help="Generate HTML API reference instead of Markdown")
    parser.add_argument("--coverage", action="store_true", help="Display documentation and type contract coverage audit")
    parser.add_argument("--json", action="store_true", help="Output coverage or symbol catalog in JSON format")

    args = parser.parse_args(args_list)

    target_path = Path(args.path)
    is_dir = args.dir or target_path.is_dir()

    generator = SynapseDocGenerator(
        dir_path=target_path if is_dir else None,
        file_path=target_path if not is_dir else None,
        recursive=True,
    )

    if args.coverage:
        cov = generator.get_coverage_report()
        if args.json:
            print(json.dumps(cov.to_dict(), indent=2))
        else:
            print(cov.format_summary())
        return 0

    if args.html:
        out_content = generator.generate_html()
        default_ext = ".html"
    else:
        out_content = generator.generate_markdown()
        default_ext = ".md"

    if args.output:
        dest = Path(args.output)
    else:
        if is_dir:
            dest = target_path / f"API_REFERENCE{default_ext}"
        else:
            dest = target_path.with_suffix(default_ext)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out_content, encoding="utf-8")
    print(f"Documentation generated successfully: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
